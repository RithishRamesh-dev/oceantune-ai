"""
agents/controller_agent.py
--------------------------
Controller Agent — v4 top-level orchestrator.

Wires together the full OceanTune AI v4 pipeline:
  Stage 1 — vLLM Config Search
    1. PlannerAgent   : validate + LLM-rank candidate configs
    2. MongoDB         : insert ranked configs as "pending" documents
    3. Coordinator    : parallel dispatch to GPU Droplet Node Servers
    4. AnalystAgent   : pick winner, explain why

  Stage 2 — Kernel-Level Search
    5. KernelOptimizerAgent : iterative LLM-guided kernel search
    6. ReportGenerator      : YAML recipe + shell script + Markdown report

Entry point:
    from agents.controller_agent import ControllerAgent
    agent = ControllerAgent()
    await agent.run()       # full async pipeline
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Optional, Tuple

from agents.analyst import AnalystAgent
from agents.do_client import DOClient
from agents.executor import ExecutorAgent
from agents.kernel_optimizer import KernelOptimizerAgent
from agents.planner import PlannerAgent
from core.config import OceanTuneConfig, load_config
from core.db import Database
from core.gpu_allocator import GPUSlotAllocator
from core.port_allocator import PortAllocator
from core.report_generator import ReportGenerator
from core.search_space import SearchSpace

log = logging.getLogger("agents.controller_agent")

REPO_ROOT = Path(__file__).resolve().parent.parent


class ControllerAgent:
    """
    Top-level pipeline orchestrator for OceanTune AI v4.

    Parameters
    ----------
    cfg : OceanTuneConfig, optional
        Full system configuration (loaded from YAML + env if not provided).
    session_id : str, optional
        UUID for this run (auto-generated if not provided).
    """

    def __init__(
        self,
        cfg: Optional[OceanTuneConfig] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.cfg = cfg or load_config()
        self.session_id = session_id or str(uuid.uuid4())

        # Shared clients
        self._db = Database(
            uri=self.cfg.database.uri,
            db_name=self.cfg.database.name,
        )
        self._do_client = DOClient.from_env(
            max_tokens=self.cfg.agent.max_tokens,
            temperature=self.cfg.agent.temperature,
            timeout_sec=float(self.cfg.agent.timeout_sec),
        )
        self._search_space = SearchSpace.load()

        log.info(
            "ControllerAgent v4 initialised: session=%s model=%s gpu=%s",
            self.session_id, self.cfg.model_id, self.cfg.gpu_type,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Synchronous wrapper — runs the full async pipeline."""
        asyncio.run(self._run_async())

    async def run_async(self) -> None:
        """Async entry point for the full pipeline."""
        await self._run_async()

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    async def _run_async(self) -> None:
        await self._db.connect()
        try:
            # Create MongoDB session document
            session_id = await self._db.create_session(
                model_id=self.cfg.model_id,
                gpu_type=self.cfg.gpu_type,
                strategy=self.cfg.optimiser.strategy,
                context_configs=[[c[0], c[1]] for c in self.cfg.context_configs],
            )
            self.session_id = session_id
            log.info("MongoDB session: %s", session_id)

            # ── Stage 1: vLLM Config Search ───────────────────────────────
            winner_flags, _ = await self._stage1(session_id)

            # ── Stage 2: Kernel Search ────────────────────────────────────
            best_kernel = {}
            if winner_flags:
                best_kernel = await self._stage2(session_id, winner_flags)
            else:
                log.warning("Stage 1 produced no winner — skipping Stage 2")

            # ── Report generation ─────────────────────────────────────────
            await self._generate_report(session_id, best_kernel)
            await self._db.update_session_status(session_id, "done")
            log.info("Pipeline complete: session=%s", session_id)

        except Exception as exc:
            log.error("Pipeline error: %s", exc, exc_info=True)
            try:
                await self._db.update_session_status(self.session_id, "error")
            except Exception:
                pass
            raise

        finally:
            await self._db.close()
            await self._do_client.close()

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------

    async def _stage1(
        self, session_id: str
    ) -> Tuple[dict, str]:
        """
        Run Stage 1: Plan → Queue → Coordinate → Analyse.

        Returns (winner_flags_dict, winner_fingerprint).
        """
        log.info("=== Stage 1: vLLM Config Search ===")

        # 1. Sample candidates from search space
        n_candidates = self.cfg.optimiser.population_size * self.cfg.optimiser.generations
        candidates = [
            self._search_space.sample_random()
            for _ in range(n_candidates)
        ]
        log.info("Sampled %d candidate configs", len(candidates))

        # 2. Planner: validate + LLM-rank
        planner = PlannerAgent(
            do_client=self._do_client,
            db=self._db,
            search_space=self._search_space,
        )
        ordered = await planner.plan(
            session_id=session_id,
            model_id=self.cfg.model_id,
            gpu_type=self.cfg.gpu_type,
            candidates=candidates,
            max_configs=n_candidates,
        )
        log.info("Planner returned %d ordered configs", len(ordered))

        if not ordered:
            log.error("Planner returned no valid configs — aborting Stage 1")
            return {}, ""

        # 3. Insert configs into MongoDB
        inserted = 0
        for priority, config_info in enumerate(reversed(ordered)):
            config_id = await self._db.insert_config(
                session_id=session_id,
                fingerprint=config_info["fingerprint"],
                flags=config_info["flags"],
                priority=priority,
            )
            if config_id:
                inserted += 1
        log.info("Inserted %d configs into MongoDB", inserted)

        # 4. Run configs in-process (no node server required)
        await self._run_local(
            session_id=session_id,
            total_configs=inserted,
            context_configs=list(self.cfg.context_configs),
        )
        log.info("Local execution finished all configs")

        # 5. Analyst: pick winner + explain
        analyst = AnalystAgent(do_client=self._do_client, db=self._db)
        analysis = await analyst.analyse(
            session_id=session_id,
            model_id=self.cfg.model_id,
            gpu_type=self.cfg.gpu_type,
        )

        log.info(
            "Stage 1 winner: fingerprint=%s fitness=%.4f",
            analysis.winner_fingerprint[:8], analysis.winner_fitness,
        )
        return analysis.winner_flags, analysis.winner_fingerprint

    # ------------------------------------------------------------------
    # Local in-process execution (single-droplet, no node server)
    # ------------------------------------------------------------------

    async def _run_local(
        self,
        session_id: str,
        total_configs: int,
        context_configs: list,
    ) -> None:
        """
        Run all pending configs directly in-process using ExecutorAgent.
        Replaces the Coordinator → Node Server HTTP path for single-droplet use.
        Concurrency is bounded by coordinator.max_parallel_per_node.
        """
        node_cfg = self.cfg.nodes[0]
        gpu_alloc = GPUSlotAllocator(
            gpu_indices=node_cfg.gpu_indices,
            gpu_type=node_cfg.gpu_type,
        )
        port_alloc = PortAllocator(
            start=self.cfg.coordinator.port_pool_start,
            end=self.cfg.coordinator.port_pool_end,
        )
        sem = asyncio.Semaphore(self.cfg.coordinator.max_parallel_per_node)

        async def _run_one(config_doc: dict) -> None:
            async with sem:
                executor = ExecutorAgent(
                    do_client=self._do_client,
                    db=self._db,
                    gpu_alloc=gpu_alloc,
                    port_alloc=port_alloc,
                    gpu_type=self.cfg.gpu_type,
                    model_id=self.cfg.model_id,
                    concurrency_levels=self.cfg.benchmark.concurrency_levels,
                    num_prompts=self.cfg.benchmark.num_prompts,
                    startup_timeout_sec=self.cfg.vllm.startup_timeout_sec,
                    primary_metric=self.cfg.optimiser.primary_metric,
                )
                await executor.run(
                    session_id=session_id,
                    config_doc=config_doc,
                    context_configs=context_configs,
                )

        tasks = []
        for _ in range(total_configs):
            config_doc = await self._db.claim_pending_config(session_id)
            if config_doc is None:
                break
            tasks.append(asyncio.create_task(_run_one(config_doc)))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------

    async def _stage2(
        self, session_id: str, winner_flags: dict
    ) -> dict:
        """
        Run Stage 2: LLM-guided kernel search.

        Returns the best kernel override dict to merge with winner_flags.
        """
        log.info("=== Stage 2: Kernel-Level Search ===")

        # Build node-local GPU/port allocators (use first node config)
        node_cfg = self.cfg.nodes[0]
        gpu_alloc = GPUSlotAllocator(
            gpu_indices=node_cfg.gpu_indices,
            gpu_type=node_cfg.gpu_type,
        )
        port_alloc = PortAllocator(
            start=self.cfg.coordinator.port_pool_start,
            end=self.cfg.coordinator.port_pool_end,
        )

        ko = KernelOptimizerAgent(
            do_client=self._do_client,
            db=self._db,
            gpu_alloc=gpu_alloc,
            port_alloc=port_alloc,
            model_id=self.cfg.model_id,
            gpu_type=self.cfg.gpu_type,
            concurrency_levels=self.cfg.benchmark.concurrency_levels,
            num_prompts=self.cfg.benchmark.num_prompts,
            startup_timeout_sec=self.cfg.vllm.startup_timeout_sec,
        )

        best_kernel = await ko.run(
            session_id=session_id,
            baseline_flags=winner_flags,
            context_configs=list(self.cfg.context_configs),
            max_iterations=10,
        )
        log.info("Stage 2 done: best_kernel=%s", best_kernel)
        return best_kernel

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    async def _generate_report(
        self, session_id: str, best_kernel: dict
    ) -> None:
        analyst = AnalystAgent(do_client=self._do_client, db=self._db)
        analysis = await analyst.analyse(
            session_id=session_id,
            model_id=self.cfg.model_id,
            gpu_type=self.cfg.gpu_type,
        )

        gen = ReportGenerator(
            output_dir=REPO_ROOT / "storage" / "results",
        )
        paths = gen.generate(
            analysis=analysis,
            best_kernel_config=best_kernel,
            model_id=self.cfg.model_id,
            gpu_type=self.cfg.gpu_type,
            session_id=session_id,
        )
        log.info("Reports written: %s", {k: str(v) for k, v in paths.items()})
