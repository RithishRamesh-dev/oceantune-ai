"""
core/coordinator.py
-------------------
Coordinator — parallel dispatch loop for Stage 1 (vLLM config search).

The Coordinator is the central orchestrator.  It:
  1. Maintains a list of NodeClients (one per GPU Droplet).
  2. Polls MongoDB for pending configs (produced by the Planner).
  3. Finds a node with sufficient free GPUs.
  4. Submits the config to that node via HTTP and tracks the in-flight job.
  5. On node failure, re-queues the config in MongoDB (up to max_retries).
  6. Runs until all configs are processed or a stop event is set.

Usage
-----
    coordinator = Coordinator(db=db, nodes=[...], session_id="...")
    await coordinator.run(
        session_id="...",
        model_id="deepseek-ai/DeepSeek-V3.2",
        gpu_type="H100",
        context_configs=[(1024, 1024), (1024, 4096)],
    )
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from core.config import NodeConfig, CoordinatorConfig
from core.db import Database
from core.node_client import NodeClient, NodeClientError

log = logging.getLogger("core.coordinator")


@dataclass
class _InFlightJob:
    """Tracks a job dispatched to a node but not yet completed."""
    job_id: str
    config_id: str
    fingerprint: str
    node_client: NodeClient
    retry_count: int = 0


class Coordinator:
    """
    Async parallel dispatch coordinator.

    Parameters
    ----------
    db : Database
        Connected MongoDB client.
    node_configs : list of NodeConfig
        Node definitions from oceantune.yaml.
    coordinator_cfg : CoordinatorConfig
        Coordinator settings (max_parallel, port pool, retry, poll).
    """

    def __init__(
        self,
        db: Database,
        node_configs: List[NodeConfig],
        coordinator_cfg: Optional[CoordinatorConfig] = None,
    ) -> None:
        self._db = db
        self._cfg = coordinator_cfg or CoordinatorConfig()
        self._nodes: List[NodeClient] = [
            NodeClient(host=n.host, port=n.node_port)
            for n in node_configs
        ]
        self._in_flight: Dict[str, _InFlightJob] = {}  # job_id → job
        self._stop = asyncio.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        *,
        session_id: str,
        model_id: str,
        gpu_type: str,
        context_configs: List[tuple],
        total_configs: int,
    ) -> None:
        """
        Run the dispatch loop until all ``total_configs`` configs are processed.

        Parameters
        ----------
        session_id : str
        model_id : str
        gpu_type : str
        context_configs : list of (input_len, output_len) tuples
        total_configs : int
            How many configs were queued (used to detect completion).
        """
        log.info(
            "Coordinator starting: session=%s total_configs=%d nodes=%d",
            session_id, total_configs, len(self._nodes),
        )

        ctx_as_lists = [[c[0], c[1]] for c in context_configs]
        processed = 0

        while processed < total_configs and not self._stop.is_set():
            # 1. Poll completed in-flight jobs
            completed = await self._poll_in_flight()
            processed += completed

            # 2. Try to dispatch pending configs to available nodes
            dispatched = await self._dispatch_pending(
                session_id=session_id,
                model_id=model_id,
                gpu_type=gpu_type,
                context_configs=ctx_as_lists,
            )

            if not dispatched and not self._in_flight:
                # No pending configs and nothing in-flight — check if truly done
                remaining = await self._db.count_pending_configs(session_id)
                if remaining == 0:
                    log.info("Coordinator: no pending configs left — done")
                    break

            await asyncio.sleep(self._cfg.poll_interval_sec)

        # Wait for any remaining in-flight jobs
        await self._drain_in_flight()
        log.info(
            "Coordinator finished: session=%s processed=%d", session_id, processed
        )

    def stop(self) -> None:
        """Signal the coordinator to stop after current jobs complete."""
        self._stop.set()

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    async def _dispatch_pending(
        self,
        *,
        session_id: str,
        model_id: str,
        gpu_type: str,
        context_configs: List[List[int]],
    ) -> int:
        """
        Claim and dispatch as many pending configs as possible across nodes.
        Returns the number of configs dispatched this tick.
        """
        dispatched = 0
        for node in self._nodes:
            # Check capacity
            try:
                cap = await node.get_capacity()
            except NodeClientError as exc:
                log.warning("Node %s:%d unreachable: %s", node.host, node.port, exc)
                continue

            free_gpus = cap.get("free_gpus", 0)
            free_ports = cap.get("free_ports", 0)
            if free_gpus < 1 or free_ports < 1:
                continue

            # Claim a pending config from MongoDB
            config_doc = await self._db.claim_pending_config(session_id)
            if config_doc is None:
                break  # No more pending configs

            config_id = str(config_doc["_id"])
            fingerprint = config_doc["fingerprint"]
            flags = config_doc["flags"]

            try:
                job_id = await node.submit_job(
                    session_id=session_id,
                    config_id=config_id,
                    fingerprint=fingerprint,
                    flags=flags,
                    context_configs=context_configs,
                    model_id=model_id,
                    gpu_type=gpu_type,
                )
                self._in_flight[job_id] = _InFlightJob(
                    job_id=job_id,
                    config_id=config_id,
                    fingerprint=fingerprint,
                    node_client=node,
                )
                dispatched += 1
                log.info(
                    "Dispatched config %s to %s:%d (job_id=%s)",
                    fingerprint[:8], node.host, node.port, job_id,
                )

            except NodeClientError as exc:
                log.warning(
                    "Failed to dispatch config %s to %s:%d: %s — re-queuing",
                    fingerprint[:8], node.host, node.port, exc,
                )
                await self._db.requeue_config(config_id)

        return dispatched

    async def _poll_in_flight(self) -> int:
        """
        Poll all in-flight jobs and process completions.
        Returns the number of jobs that finished (done or failed).
        """
        if not self._in_flight:
            return 0

        finished_ids: List[str] = []
        completed_count = 0

        for job_id, job in list(self._in_flight.items()):
            try:
                status = await job.node_client.get_job_status(job_id)
            except NodeClientError as exc:
                log.warning("Poll failed for job %s: %s", job_id, exc)
                # Node may be down — re-queue config if under retry limit
                await self._handle_node_failure(job)
                finished_ids.append(job_id)
                continue

            if status["status"] in {"done", "failed"}:
                finished_ids.append(job_id)
                completed_count += 1
                log.info(
                    "Job %s %s (config=%s fitness=%s)",
                    job_id, status["status"],
                    job.fingerprint[:8], status.get("best_fitness"),
                )
                if status["status"] == "failed":
                    await self._handle_job_failure(job, status.get("error", ""))

        for job_id in finished_ids:
            self._in_flight.pop(job_id, None)

        return completed_count

    async def _handle_node_failure(self, job: _InFlightJob) -> None:
        """Re-queue a config whose node went offline (up to max_retries)."""
        retry_count = job.retry_count + 1
        if retry_count <= self._cfg.max_retries:
            log.warning(
                "Node failure — re-queuing config %s (attempt %d/%d)",
                job.fingerprint[:8], retry_count, self._cfg.max_retries,
            )
            await self._db.requeue_config(job.config_id)
        else:
            log.error(
                "Config %s exceeded max retries — marking failed",
                job.fingerprint[:8],
            )
            await self._db.mark_config_failed(
                job.config_id, "Exceeded max retries due to node failures"
            )

    async def _handle_job_failure(self, job: _InFlightJob, error: str) -> None:
        """Log a job failure (the Executor already updated MongoDB)."""
        log.warning("Job failure: config=%s error=%s", job.fingerprint[:8], error)

    async def _drain_in_flight(self) -> None:
        """Wait for all remaining in-flight jobs before returning."""
        if not self._in_flight:
            return
        log.info("Coordinator draining %d in-flight jobs...", len(self._in_flight))
        while self._in_flight:
            await self._poll_in_flight()
            await asyncio.sleep(self._cfg.poll_interval_sec)
