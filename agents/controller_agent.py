"""
agents/controller_agent.py
--------------------------
Controller Agent — v5 top-level orchestrator.

Wires together the full OceanTune AI pipeline:
  Stage 1 — vLLM Config Search
    1. PlannerAgent         : LLM-guided iterative flag search
    2. ExecutorAgent        : vLLM Docker + benchmark + fitness scoring
    3. AnalystAgent         : per-iteration bottleneck diagnosis + session winner

  Stage 2 — Inference Strategy Search
    4. StrategyOptimizerAgent : KV cache, speculative decoding, attention backend,
                                prefill strategies, vendor-specific kernels

  Stage 3 — Deep Profiling + Bottleneck Reasoning
    5. ProfilerAgent        : Torch profiler trace at optimal concurrency
    6. NcuProfiler / RocprofProfiler : Hardware counters (Tensor Core util, DRAM BW,
                                        occupancy, warp stall reasons)
    7. BottleneckReasoningAgent : Multi-source bottleneck classification
    8. ResearchAgent        : ranked optimization recommendations

  Stage 4 — Autonomous Kernel Engineering
    9. KernelResearchAgent  : Research best kernel implementations for the bottleneck
   10. KernelGenerationAgent: Generate Triton/CUDA kernels
   11. CorrectnessFirewallAgent: Validate kernels against PyTorch reference
   12. KernelEvolutionAgent : keep/revert loop with experiment tree tracking
   13. ReportGenerator      : YAML recipe + shell script + Markdown report

Entry point:
    from agents.controller_agent import ControllerAgent
    agent = ControllerAgent()
    await agent.run()
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents.analyst import AnalystAgent
from agents.do_client import DOClient
from agents.executor import ExecutorAgent
from agents.strategy_optimizer import StrategyOptimizerAgent
from agents.profiler_agent import ProfilerAgent
from agents.research_agent import ResearchAgent
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
            winner_flags, _, stage1_fitness = await self._stage1(session_id)

            # ── Stage 2: Inference Strategy Search ───────────────────────
            best_strategy = {}
            winner_metrics: dict = {}
            stage2_fitness: float = 0.0
            if winner_flags:
                best_strategy, winner_metrics, stage2_fitness = await self._stage2(
                    session_id, winner_flags
                )
            else:
                log.warning("Stage 1 produced no winner — skipping Stage 2 and Stage 3")

            # ── Stage 3: Deep Profiling + Bottleneck Reasoning + Flag Trials ──
            from agents.research_agent import ResearchReport
            research_report: Optional[ResearchReport] = None
            bottleneck_analysis = None
            kernel_research = None
            evolution_result = None
            stage3_fitness: float = stage2_fitness
            applied_recs: list = []
            all_tried_recs: list = []
            stage3_flags = winner_flags  # may be updated by Stage 3 flag trials
            if winner_flags:
                (
                    research_report,
                    bottleneck_analysis,
                    kernel_research,
                    stage3_fitness,
                    applied_recs,
                    all_tried_recs,
                    stage3_flags,
                ) = await self._stage3(
                    session_id, winner_flags, winner_metrics,
                    stage2_fitness=stage2_fitness,
                    stage2_strategy=best_strategy,
                )

            # ── Stage 4: Autonomous Kernel Engineering ────────────────────
            stage4_enabled = getattr(self.cfg, "stage4_enabled", False)
            if (
                stage4_enabled
                and winner_flags
                and bottleneck_analysis is not None
                and kernel_research is not None
            ):
                evolution_result = await self._stage4(
                    session_id=session_id,
                    winner_flags=stage3_flags,   # use improved flags from Stage 3
                    bottleneck=bottleneck_analysis,
                    research=kernel_research,
                )

            # ── Report generation ─────────────────────────────────────────
            await self._generate_report(
                session_id, best_strategy, research_report,
                evolution_result=evolution_result,
                stage1_fitness=stage1_fitness,
                stage2_fitness=stage2_fitness,
                stage3_fitness=stage3_fitness,
                stage3_applied_recs=applied_recs,
                stage3_all_tried_recs=all_tried_recs,
            )
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
        analyst = AnalystAgent(do_client=self._do_client, db=self._db)

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
        last_analyst_eval: dict = {}

        for iteration in range(n_iterations):
            flags = current_best if iteration == 0 else None

            if iteration > 0:
                # Best run so far — pull enriched_metrics (correct key)
                top = await self._db.get_top_configs(session_id, n=1)
                best_run = top[0] if top else {}
                best_metrics = best_run.get("enriched_metrics") or best_run.get("raw_metrics") or {}

                flags, rationale = await planner.propose_next(
                    model_id=self.cfg.model_id,
                    gpu_type=self.cfg.gpu_type,
                    n_gpus=n_gpus,
                    current_best=best_flags,
                    current_best_metrics=best_metrics,
                    history=search_history,
                    iteration=iteration,
                    analyst_eval=last_analyst_eval,
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

            # Analyst evaluates this iteration — feeds into next proposal
            best_run_for_iter = await self._db.get_best_run_for_config(config_id)
            if best_run_for_iter and not error_text:
                last_analyst_eval = await analyst.evaluate_iteration(
                    iteration=iteration,
                    flags={k: v for k, v in asdict(flags).items() if k != "run_id"},
                    benchmark_run=best_run_for_iter,
                    history=search_history,
                    model_id=self.cfg.model_id,
                    gpu_type=self.cfg.gpu_type,
                )
                log.info(
                    "Iteration %d — analyst: bottleneck=%s rec=%s",
                    iteration,
                    last_analyst_eval.get("bottleneck", "?"),
                    last_analyst_eval.get("recommendation", "")[:80],
                )
            else:
                last_analyst_eval = {}

            # Record in history — use enriched_metrics with canonical field names
            em = (best_run_for_iter or {}).get("enriched_metrics") or {}
            history_entry: dict = {
                "iteration": iteration,
                "flags": {k: v for k, v in asdict(flags).items() if k != "run_id"},
                "fitness": fitness,
                "enriched_metrics": em,
                "rationale": rationale,
                "analyst_recommendation": last_analyst_eval.get("recommendation", ""),
            }
            if error_text:
                history_entry["error"] = error_text
            search_history.append(history_entry)

            if fitness > best_fitness:
                best_fitness = fitness
                best_flags = flags

        if best_fitness == 0.0:
            log.warning("Stage 1: no successful benchmark runs")
            return {}, "", 0.0

        from dataclasses import asdict as _asdict
        log.info("Stage 1 complete: best_fitness=%.4f fingerprint=%s",
                 best_fitness, best_flags.fingerprint()[:8])
        return (
            {k: v for k, v in _asdict(best_flags).items() if k != "run_id"},
            best_flags.fingerprint(),
            best_fitness,
        )

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
    # Stage 2 — Inference Strategy Search
    # ------------------------------------------------------------------

    async def _stage2(
        self, session_id: str, winner_flags: dict
    ) -> tuple:
        """
        Run Stage 2: LLM-guided inference strategy search.

        Explores KV cache strategies, speculative decoding, prefill strategies,
        attention backend selection, and vendor-specific kernel flags on top of
        the Stage 1 winner.

        Returns (best_strategy_config, winner_enriched_metrics).
        """
        log.info("=== Stage 2: Inference Strategy Search ===")

        # Pull Stage 1 winner metrics for the LLM context
        top = await self._db.get_top_configs(session_id, n=1)
        winner_metrics = {}
        if top:
            winner_metrics = top[0].get("enriched_metrics") or top[0].get("raw_metrics") or {}

        node_cfg = self.cfg.nodes[0]
        gpu_alloc = GPUSlotAllocator(
            gpu_indices=node_cfg.gpu_indices,
            gpu_type=node_cfg.gpu_type,
        )
        port_alloc = PortAllocator(
            start=self.cfg.coordinator.port_pool_start,
            end=self.cfg.coordinator.port_pool_end,
        )

        so = StrategyOptimizerAgent(
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
            primary_metric=self.cfg.optimiser.primary_metric,
        )

        best_strategy, stage2_fitness = await so.run(
            session_id=session_id,
            baseline_flags=winner_flags,
            baseline_metrics=winner_metrics,
            context_configs=list(self.cfg.context_configs),
            max_iterations=12,
        )
        log.info("Stage 2 done: best_strategy=%s best_fitness=%.4f", best_strategy, stage2_fitness)
        return best_strategy, winner_metrics, stage2_fitness

    # ------------------------------------------------------------------
    # Stage 3 — Deep Profiling + Bottleneck Reasoning
    # ------------------------------------------------------------------

    async def _stage3(
        self,
        session_id: str,
        winner_flags: dict,
        winner_metrics: dict,
        stage2_fitness: float = 0.0,
        stage2_strategy: Optional[Dict[str, Any]] = None,
    ):
        """
        Run Stage 3: profiling → bottleneck reasoning → try flag recommendations → kernel research.

        Pipeline:
          1. Torch profiler trace (category breakdown: attention/GEMM/MoE/comm)
          2. Hardware counters (Nsight Compute or rocprof — if tools available)
          3. BottleneckReasoningAgent (LLM synthesises all signals into a bottleneck class)
          4. ResearchAgent (LLM-ranked optimization recommendations with vllm_flags dicts)
          5. Try each stage3_flag recommendation — keep if fitness improves
          6. KernelResearchAgent (deep research on best kernel implementations for
             bottlenecks not solved by flag changes)

        Returns (research_report, bottleneck_analysis, kernel_research, stage3_fitness,
                 applied_recommendations, updated_winner_flags).
        """
        log.info("=== Stage 3: Deep Profiling + Bottleneck Reasoning ===")

        from agents.profiler_agent import ProfilerAgent
        from agents.research_agent import ResearchAgent
        from agents.bottleneck_reasoning_agent import BottleneckReasoningAgent
        from agents.kernel_research_agent import KernelResearchAgent

        optimal_concurrency = int(winner_metrics.get("best_concurrency", 64))
        context_configs = list(self.cfg.context_configs)
        input_len = context_configs[0][0] if context_configs else 1024
        output_len = context_configs[0][1] if context_configs else 1024

        node_cfg = self.cfg.nodes[0]
        gpu_alloc = GPUSlotAllocator(
            gpu_indices=node_cfg.gpu_indices,
            gpu_type=node_cfg.gpu_type,
        )
        port_alloc = PortAllocator(
            start=self.cfg.coordinator.port_pool_start,
            end=self.cfg.coordinator.port_pool_end,
        )

        # ── 3a. Torch profiler trace ──────────────────────────────────────
        profiler = ProfilerAgent(
            do_client=self._do_client,
            db=self._db,
            gpu_alloc=gpu_alloc,
            port_alloc=port_alloc,
            model_id=self.cfg.model_id,
            gpu_type=self.cfg.gpu_type,
            startup_timeout_sec=self.cfg.vllm.startup_timeout_sec,
            docker_image=self.cfg.vllm.docker_image,
        )

        trace = await profiler.run(
            session_id=session_id,
            winner_flags=winner_flags,
            optimal_concurrency=optimal_concurrency,
            input_len=input_len,
            output_len=output_len,
        )
        log.info(
            "Stage 3a profile: bottleneck=%s attention=%.1f%% gemm=%.1f%%",
            trace.bottleneck_type, trace.attention_pct, trace.gemm_pct,
        )

        # ── 3b. Hardware counters (ncu / rocprof) ─────────────────────────
        hw_counters = None
        top_kernel_name = trace.bottleneck_kernel or ""
        if top_kernel_name:
            hw_counters = await self._collect_hardware_counters(
                session_id=session_id,
                kernel_name=top_kernel_name,
                input_len=input_len,
                output_len=output_len,
                concurrency=optimal_concurrency,
            )
            if hw_counters:
                log.info("Stage 3b hardware counters: %s", hw_counters.summary())

        # ── 3c. Deep bottleneck reasoning ────────────────────────────────
        bottleneck_reasoner = BottleneckReasoningAgent(do_client=self._do_client)
        bottleneck_analysis = await bottleneck_reasoner.analyse(
            trace=trace,
            hw_counters=hw_counters,
            winner_flags=winner_flags,
            model_id=self.cfg.model_id,
            gpu_type=self.cfg.gpu_type,
            session_id=session_id,
        )
        log.info(
            "Stage 3c bottleneck: primary=%s component=%s action=%s",
            bottleneck_analysis.primary_bottleneck,
            bottleneck_analysis.primary_component,
            bottleneck_analysis.recommended_action,
        )

        # ── 3d. Research Agent (vLLM-level recommendations) ───────────────
        # Pass the full winner_flags (Stage1+2 merged) AND the Stage 2 delta
        # separately so the LLM knows exactly what's already been applied.
        researcher = ResearchAgent(do_client=self._do_client)
        research_report = await researcher.analyse(
            trace=trace,
            winner_flags=winner_flags,
            stage2_strategy=stage2_strategy or {},
            model_id=self.cfg.model_id,
            gpu_type=self.cfg.gpu_type,
        )
        log.info(
            "Stage 3d research: %d recommendations, custom_kernel_warranted=%s",
            len(research_report.recommendations),
            research_report.custom_kernel_warranted,
        )

        # ── 3e. Try flag recommendations immediately ───────────────────────
        # Validate each stage3_flag recommendation by benchmarking it now,
        # rather than deferring to a later stage.  Only recommendations that
        # actually improve fitness are kept; winner_flags is updated in-place
        # so subsequent steps (kernel research, Stage 4) see the best config.
        updated_flags, stage3_fitness, applied_recs, all_tried_recs = await self._try_flag_recommendations(
            session_id=session_id,
            winner_flags=winner_flags,
            research_report=research_report,
            current_fitness=stage2_fitness,
        )
        if applied_recs:
            log.info(
                "Stage 3e: %d flag change(s) accepted, fitness %.4f → %.4f",
                len(applied_recs), stage2_fitness, stage3_fitness,
            )
        else:
            log.info("Stage 3e: no flag recommendations improved fitness")

        # ── 3f. Kernel Research (deep kernel-level research) ──────────────
        # Only run if flag changes didn't already fully address the bottleneck,
        # or the research agent flagged that custom kernel work is warranted.
        kernel_research = None
        need_kernel_work = (
            research_report.custom_kernel_warranted
            or bottleneck_analysis.recommended_action.startswith("kernel_generation")
        )
        if need_kernel_work:
            log.info("Stage 3f: running deep kernel research...")
            kernel_researcher = KernelResearchAgent(do_client=self._do_client)
            kernel_research = await kernel_researcher.research(
                bottleneck=bottleneck_analysis,
                trace=trace,
                model_id=self.cfg.model_id,
                gpu_type=self.cfg.gpu_type,
                winner_flags=updated_flags,   # use the improved flags
            )
            log.info(
                "Stage 3f kernel research: %d approaches, proceed_to_generation=%s",
                len(kernel_research.approaches),
                kernel_research.proceed_to_generation,
            )

        return (
            research_report,
            bottleneck_analysis,
            kernel_research,
            stage3_fitness,
            applied_recs,
            all_tried_recs,
            updated_flags,
        )

    async def _collect_hardware_counters(
        self,
        *,
        session_id: str,
        kernel_name: str,
        input_len: int,
        output_len: int,
        concurrency: int,
    ):
        """
        Attempt to collect hardware counters using ncu (NVIDIA) or rocprof (AMD).
        Returns HardwareCounters or None if tools are unavailable.
        Never raises.
        """
        try:
            vendor = "amd" if self.cfg.gpu_type in {"MI300X", "MI325X", "MI350X"} else "nvidia"
            # Build a microbenchmark launch command for the top kernel
            launch_cmd = (
                f"python microbench/operator_bench.py "
                f"--op attention "
                f"--input-len {input_len} "
                f"--output-len {output_len} "
                f"--concurrency {concurrency}"
            )

            if vendor == "nvidia":
                from profiling.ncu_profiler import NcuProfiler
                ncu = NcuProfiler(gpu_type=self.cfg.gpu_type)
                if not ncu.available:
                    return None
                return await ncu.profile_kernel(
                    kernel_name=kernel_name,
                    launch_cmd=launch_cmd,
                    session_id=session_id,
                )
            else:
                from profiling.rocprof_profiler import RocprofProfiler
                rp = RocprofProfiler(gpu_type=self.cfg.gpu_type)
                if not rp.available:
                    return None
                return await rp.profile_kernel(
                    kernel_name=kernel_name,
                    launch_cmd=launch_cmd,
                    session_id=session_id,
                )
        except Exception as exc:
            log.warning("Hardware counter collection failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Stage 3 helper: try vLLM flag recommendations in-place
    # ------------------------------------------------------------------

    async def _try_flag_recommendations(
        self,
        session_id: str,
        winner_flags: Dict[str, Any],
        research_report,
        current_fitness: float,
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        For each stage3_flag recommendation with a non-empty vllm_flags dict,
        benchmark the flag change on its own and keep it if fitness improves.

        Returns
        -------
        (updated_flags, final_fitness, accepted_list, all_tried_list)

        accepted_list: recommendations that improved fitness
          keys: title, flags, fitness_before, fitness_after, delta

        all_tried_list: EVERY recommendation with its actual benchmark outcome
          keys: title, flags, estimated_improvement_pct, confidence,
                status ("accepted" | "rejected" | "skipped" | "not_tried"),
                fitness_before, fitness_after, actual_delta, actual_delta_pct
        """
        from dataclasses import asdict

        accepted: List[Dict[str, Any]] = []
        all_tried: List[Dict[str, Any]] = []
        current_flags: Dict[str, Any] = dict(winner_flags)

        known_fields = set(VLLMFlags.__dataclass_fields__)
        context_configs = list(self.cfg.context_configs)

        for rec in research_report.recommendations:
            base_record: Dict[str, Any] = {
                "rank": rec.rank,
                "title": rec.title,
                "category": rec.category,
                "stage": rec.stage,
                "flags": rec.vllm_flags,
                "estimated_improvement_pct": rec.expected_improvement_pct,
                "confidence": rec.confidence,
                "fitness_before": current_fitness,
                "fitness_after": None,
                "actual_delta": None,
                "actual_delta_pct": None,
                "status": "not_tried",
                "skip_reason": "",
            }

            # Only try flag-based recs that have machine-readable flags and differ from current
            if rec.stage not in ("stage3_flag", "stage2"):
                base_record["skip_reason"] = f"stage={rec.stage} (requires kernel/custom code work)"
                all_tried.append(base_record)
                continue

            if rec.requires_custom_code:
                base_record["skip_reason"] = "requires_custom_code=True"
                all_tried.append(base_record)
                continue

            if not rec.vllm_flags:
                base_record["skip_reason"] = "vllm_flags empty — LLM did not provide machine-readable flags"
                all_tried.append(base_record)
                continue

            already_applied = not any(
                winner_flags.get(k) != v for k, v in rec.vllm_flags.items()
            )
            if already_applied:
                base_record["status"] = "skipped"
                base_record["skip_reason"] = "flags already applied in Stage 1/2"
                all_tried.append(base_record)
                continue

            # --- Actually benchmark this recommendation ---
            trial_raw = {**current_flags, **rec.vllm_flags}
            trial_clean = {k: v for k, v in trial_raw.items() if k in known_fields}
            trial_flags = VLLMFlags(**trial_clean)

            config_id = await self._db.insert_config(
                session_id=session_id,
                fingerprint=trial_flags.fingerprint(),
                flags={k: v for k, v in asdict(trial_flags).items() if k != "run_id"},
                generation=-1,   # Stage 3 trials are generation -1
                priority=-1,
            )
            if config_id is None:
                # Already benchmarked in a prior run — look up cached result
                log.info("Stage 3: rec '%s' already benchmarked (cached)", rec.title)
                base_record["status"] = "skipped"
                base_record["skip_reason"] = "config fingerprint already benchmarked in this session"
                all_tried.append(base_record)
                continue

            await self._run_single(
                session_id=session_id,
                config_id=config_id,
                context_configs=context_configs,
            )

            config_doc = await self._db.get_config_by_id(config_id)
            fitness = config_doc.get("fitness_score", 0.0) if config_doc else 0.0
            delta = fitness - current_fitness
            delta_pct = (delta / current_fitness * 100) if current_fitness else 0.0

            base_record["fitness_after"] = fitness
            base_record["actual_delta"] = delta
            base_record["actual_delta_pct"] = delta_pct

            if fitness > current_fitness:
                base_record["status"] = "accepted"
                log.info(
                    "Stage 3 rec '%s' ACCEPTED: %.4f → %.4f (+%.4f, +%.1f%%)",
                    rec.title, current_fitness, fitness, delta, delta_pct,
                )
                accepted.append({
                    "title": rec.title,
                    "flags": rec.vllm_flags,
                    "fitness_before": current_fitness,
                    "fitness_after": fitness,
                    "delta": delta,
                })
                current_flags = {k: v for k, v in asdict(trial_flags).items() if k != "run_id"}
                current_fitness = fitness
            else:
                base_record["status"] = "rejected"
                log.info(
                    "Stage 3 rec '%s' REJECTED: %.4f → %.4f (%.4f, %.1f%%)",
                    rec.title, current_fitness, fitness, delta, delta_pct,
                )

            all_tried.append(base_record)

        n_tried = sum(1 for r in all_tried if r["status"] in ("accepted", "rejected"))
        n_accepted = len(accepted)
        log.info(
            "Stage 3 flag trials: %d benchmarked, %d accepted, %d rejected, %d skipped/not-tried",
            n_tried, n_accepted,
            sum(1 for r in all_tried if r["status"] == "rejected"),
            sum(1 for r in all_tried if r["status"] in ("skipped", "not_tried")),
        )
        return current_flags, current_fitness, accepted, all_tried

    # ------------------------------------------------------------------
    # Stage 4 — Autonomous Kernel Engineering
    # ------------------------------------------------------------------

    async def _stage4(
        self,
        *,
        session_id: str,
        winner_flags: dict,
        bottleneck,
        research,
    ):
        """
        Run Stage 4: autonomous kernel generation, validation, and evolution.

        Pipeline:
          1. KernelGenerationAgent   : Generate Triton kernel targeting bottleneck
          2. CorrectnessFirewallAgent : Validate against PyTorch reference
          3. KernelEvolutionAgent    : keep/revert loop for iterative improvement

        Returns EvolutionResult (or None on error).
        """
        log.info("=== Stage 4: Autonomous Kernel Engineering ===")

        try:
            from agents.kernel_evolution_agent import KernelEvolutionAgent

            node_cfg = self.cfg.nodes[0]
            device = f"cuda:{node_cfg.gpu_indices[0]}" if node_cfg.gpu_indices else "cuda:0"

            evolver = KernelEvolutionAgent(
                do_client=self._do_client,
                device=device,
                bench_timeout_sec=120,
            )

            evolution_result = await evolver.evolve(
                bottleneck=bottleneck,
                research=research,
                model_id=self.cfg.model_id,
                gpu_type=self.cfg.gpu_type,
                session_id=session_id,
                max_iterations=getattr(self.cfg, "stage4_iterations", 3),
            )

            log.info("Stage 4 complete: %s", evolution_result.summary())
            return evolution_result

        except Exception as exc:
            log.warning("Stage 4 error: %s", exc, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    async def _generate_report(
        self,
        session_id: str,
        best_strategy: dict,
        research_report=None,
        evolution_result=None,
        stage1_fitness: float = 0.0,
        stage2_fitness: float = 0.0,
        stage3_fitness: float = 0.0,
        stage3_applied_recs: Optional[List] = None,
        stage3_all_tried_recs: Optional[List] = None,
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
            best_kernel_config=best_strategy,
            model_id=self.cfg.model_id,
            gpu_type=self.cfg.gpu_type,
            session_id=session_id,
            research_report=research_report,
            evolution_result=evolution_result,
            stage1_fitness=stage1_fitness,
            stage2_fitness=stage2_fitness,
            stage3_fitness=stage3_fitness,
            stage3_applied_recs=stage3_applied_recs or [],
            stage3_all_tried_recs=stage3_all_tried_recs or [],
        )
        log.info("Reports written: %s", {k: str(v) for k, v in paths.items()})
