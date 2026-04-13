"""
node/node_worker.py
-------------------
Node Worker — executes a single vLLM + benchmark job on behalf of the
Coordinator.

The worker is called by the Node Server (node_server.py) for each job
dispatched by the Coordinator.  It:
  1. Acquires a GPU slot and port from the node-local allocators.
  2. Starts vLLM with the given flags + GPU slot env.
  3. Runs the benchmark ramp for each requested context length.
  4. Writes results to MongoDB.
  5. Releases GPU slot and port.
  6. Returns a JobResult to the Node Server.

This module is intentionally free of FastAPI dependencies so it can be
unit-tested without a running HTTP server.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agents.do_client import DOClient
from agents.executor import ExecutorAgent
from core.db import Database
from core.gpu_allocator import GPUSlotAllocator
from core.port_allocator import PortAllocator

log = logging.getLogger("node.worker")


@dataclass
class JobRequest:
    """Payload sent by the Coordinator to a Node Server."""
    session_id: str
    config_id: str
    fingerprint: str
    flags: Dict[str, Any]           # VLLMFlags fields as a plain dict
    context_configs: List[Tuple[int, int]]
    model_id: str
    gpu_type: str


@dataclass
class JobResult:
    """Response returned by the Node Worker to the Node Server."""
    config_id: str
    fingerprint: str
    status: str                     # "done" | "failed"
    best_fitness: float = 0.0
    error: Optional[str] = None
    run_ids: List[str] = field(default_factory=list)


class NodeWorker:
    """
    Executes benchmark jobs dispatched by the Coordinator.

    Parameters
    ----------
    db : Database
        Shared MongoDB client (already connected).
    do_client : DOClient
        DO Serverless Inference client (may have empty key — graceful fallback).
    gpu_alloc : GPUSlotAllocator
        Node-level GPU slot allocator.
    port_alloc : PortAllocator
        Node-level port allocator.
    node_host : str
        Hostname of this node (stored in benchmark_run documents).
    startup_timeout_sec : int
        vLLM startup timeout.
    primary_metric : str
        Fitness metric forwarded to MetricsCollector: throughput | p95_latency | ttft | tpot.
    """

    def __init__(
        self,
        *,
        db: Database,
        do_client: DOClient,
        gpu_alloc: GPUSlotAllocator,
        port_alloc: PortAllocator,
        node_host: str = "localhost",
        startup_timeout_sec: int = 1200,
        primary_metric: str = "throughput",
    ) -> None:
        self._db = db
        self._do_client = do_client
        self._gpu_alloc = gpu_alloc
        self._port_alloc = port_alloc
        self._node_host = node_host
        self._startup_timeout_sec = startup_timeout_sec
        self._primary_metric = primary_metric

        # Track running jobs (config_id → asyncio.Task)
        self._running: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Job submission
    # ------------------------------------------------------------------

    async def submit(self, job: JobRequest) -> JobResult:
        """
        Run a job synchronously (awaiting completion).

        For fire-and-forget parallel execution, wrap this in asyncio.create_task()
        at the Node Server layer.
        """
        log.info(
            "Worker: starting job config_id=%s fingerprint=%s",
            job.config_id, job.fingerprint[:8],
        )

        executor = ExecutorAgent(
            do_client=self._do_client,
            db=self._db,
            gpu_alloc=self._gpu_alloc,
            port_alloc=self._port_alloc,
            gpu_type=job.gpu_type,
            model_id=job.model_id,
            startup_timeout_sec=self._startup_timeout_sec,
            node_host=self._node_host,
            primary_metric=self._primary_metric,
        )

        config_doc = {
            "_id": job.config_id,
            "fingerprint": job.fingerprint,
            "flags": job.flags,
        }

        try:
            await executor.run(
                session_id=job.session_id,
                config_doc=config_doc,
                context_configs=job.context_configs,
            )

            # Fetch the best fitness from MongoDB for the response
            runs = await self._db.get_top_configs(job.session_id, n=1)
            best_fitness = runs[0]["fitness_score"] if runs else 0.0

            return JobResult(
                config_id=job.config_id,
                fingerprint=job.fingerprint,
                status="done",
                best_fitness=best_fitness,
            )

        except Exception as exc:
            log.error("Worker job failed: %s", exc)
            return JobResult(
                config_id=job.config_id,
                fingerprint=job.fingerprint,
                status="failed",
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Capacity query
    # ------------------------------------------------------------------

    @property
    def free_gpus(self) -> int:
        return self._gpu_alloc.available

    @property
    def free_ports(self) -> int:
        return self._port_alloc.available
