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
from core.logger import get_logger
from core.report_generator import ReportGenerator
from core.search_space import SearchSpace, VLLMFlags

log = get_logger("agents.controller_agent")

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
        Run Stage 1: Iterative agent-guided hyperparameter search.

        Iteration 0: bare minimum vLLM flags (establishes baseline).
        Iteration N: PlannerAgent.propose_next() observes all prior results
                     and proposes a single targeted change.

        Returns (winner_flags_dict, winner_fingerprint).
        """
        log.info("=== Stage 1: Agent-guided vLLM Config Search ===")

        n_gpus = len(self.cfg.nodes[0].gpu_indices)
        n_iterations = self.cfg.optimiser.generations
        context_configs = list(self.cfg.context_configs)

        planner = PlannerAgent(
            do_client=self._do_client,
            db=self._db,
            search_space=self._search_space,
        )

        # Iteration 0: bare minimum — let vLLM choose all defaults
        current_best = VLLMFlags(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            distributed_executor_backend="mp",
            cpu_offload_gb=0,
        )
        current_best.run_id = current_best.fingerprint()

        best_fitness = 0.0
        best_flags = current_best
        search_history: list = []

        for iteration in range(n_iterations):
            flags = current_best if iteration == 0 else None

            if iteration > 0:
                # Gather best metrics from DB for the agent to reason about
                top = await self._db.get_top_configs(session_id, n=1)
                best_metrics = top[0].get("metrics", {}) if top else {}

                flags, rationale = await planner.propose_next(
                    model_id=self.cfg.model_id,
                    gpu_type=self.cfg.gpu_type,
                    n_gpus=n_gpus,
                    current_best=best_flags,
                    current_best_metrics=best_metrics,
                    history=search_history,
                    iteration=iteration,
                )
                log.info("Iteration %d — agent proposal: %s", iteration, rationale[:120])
            else:
                log.info("Iteration 0 — baseline: bare minimum vLLM flags")
                rationale = "Baseline: vLLM defaults, no extra flags"

            from dataclasses import asdict
            config_id = await self._db.insert_config(
                session_id=session_id,
                fingerprint=flags.fingerprint(),
                flags={k: v for k, v in asdict(flags).items() if k != "run_id"},
                generation=iteration,
                priority=iteration,
            )
            if config_id is None:
                log.info("Iteration %d — config already seen, skipping", iteration)
                continue

            await self._run_single(
                session_id=session_id,
                config_id=config_id,
                context_configs=context_configs,
            )

            # Read result back from DB
            config_doc = await self._db.get_config_by_id(config_id)
            fitness = config_doc.get("fitness_score", 0.0) if config_doc else 0.0
            error_text = config_doc.get("error", "") if config_doc else ""
            log.info("Iteration %d — fitness=%.4f", iteration, fitness)
            if error_text:
                log.warning("Iteration %d — server error: %s", iteration, error_text[:200])

            # Record in history for agent context (include error so planner can avoid repeats)
            top_runs = await self._db.get_top_configs(session_id, n=1)
            history_entry: dict = {
                "iteration": iteration,
                "flags": {k: v for k, v in asdict(flags).items() if k != "run_id"},
                "fitness": fitness,
                "metrics": top_runs[0].get("metrics", {}) if top_runs else {},
                "rationale": rationale,
            }
            if error_text:
                history_entry["error"] = error_text
            search_history.append(history_entry)

            if fitness > best_fitness:
                best_fitness = fitness
                best_flags = flags

        if best_fitness == 0.0:
            log.warning("Stage 1: no successful benchmark runs")
            return {}, ""

        from dataclasses import asdict as _asdict
        log.info("Stage 1 complete: best_fitness=%.4f fingerprint=%s",
                 best_fitness, best_flags.fingerprint()[:8])
        return {k: v for k, v in _asdict(best_flags).items() if k != "run_id"}, best_flags.fingerprint()

    # ------------------------------------------------------------------
    # Single config execution
    # ------------------------------------------------------------------

    async def _run_single(
        self,
        session_id: str,
        config_id: str,
        context_configs: list,
    ) -> None:
        """Run one config doc in-process. Used by the iterative _stage1 loop."""
        config_doc = await self._db.get_config_by_id(config_id)
        if config_doc is None:
            log.error("Config %s not found in DB", config_id)
            return

        node_cfg = self.cfg.nodes[0]
        gpu_alloc = GPUSlotAllocator(
            gpu_indices=node_cfg.gpu_indices,
            gpu_type=node_cfg.gpu_type,
        )
        port_alloc = PortAllocator(
            start=self.cfg.coordinator.port_pool_start,
            end=self.cfg.coordinator.port_pool_end,
        )
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
            docker_image=self.cfg.vllm.docker_image,
        )
        await executor.run(
            session_id=session_id,
            config_doc=config_doc,
            context_configs=context_configs,
        )

    # ------------------------------------------------------------------
    # Legacy batch execution (kept for multi-node coordinator path)
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
        Configs are processed one at a time — on a single GPU there is no benefit
        to parallelism, and serial execution keeps GPU slot accounting simple.
        Configs whose tensor_parallel_size exceeds the available GPU count are
        skipped (marked failed) rather than silently re-queued forever.
        """
        node_cfg = self.cfg.nodes[0]
        n_gpus = len(node_cfg.gpu_indices)
        gpu_alloc = GPUSlotAllocator(
            gpu_indices=node_cfg.gpu_indices,
            gpu_type=node_cfg.gpu_type,
        )
        port_alloc = PortAllocator(
            start=self.cfg.coordinator.port_pool_start,
            end=self.cfg.coordinator.port_pool_end,
        )

        for _ in range(total_configs):
            config_doc = await self._db.claim_pending_config(session_id)
            if config_doc is None:
                break

            # Skip configs that need more GPUs than available
            tp = config_doc.get("flags", {}).get("tensor_parallel_size") or 1
            if tp > n_gpus:
                log.warning(
                    "Skipping config %s: tp=%d requires %d GPUs, only %d available — "
                    "reduce tensor_parallel_size in search space or add more GPUs",
                    config_doc.get("fingerprint", "?")[:8], tp, tp, n_gpus,
                )
                await self._db.mark_config_failed(
                    str(config_doc["_id"]),
                    f"tensor_parallel_size={tp} exceeds available GPUs ({n_gpus})",
                )
                continue

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
                docker_image=self.cfg.vllm.docker_image,
            )
            await executor.run(
                session_id=session_id,
                config_doc=config_doc,
                context_configs=context_configs,
            )

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
            docker_image=self.cfg.vllm.docker_image,
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
