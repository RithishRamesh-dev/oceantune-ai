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
from typing import Optional, Tuple

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
            winner_flags, _ = await self._stage1(session_id)

            # ── Stage 2: Inference Strategy Search ───────────────────────
            best_strategy = {}
            winner_metrics: dict = {}
            if winner_flags:
                best_strategy, winner_metrics = await self._stage2(session_id, winner_flags)
            else:
                log.warning("Stage 1 produced no winner — skipping Stage 2 and Stage 3")

            # ── Stage 3: Deep Profiling + Bottleneck Reasoning ───────────
            from agents.research_agent import ResearchReport
            research_report: Optional[ResearchReport] = None
            bottleneck_analysis = None
            kernel_research = None
            evolution_result = None
            if winner_flags:
                research_report, bottleneck_analysis, kernel_research = await self._stage3(
                    session_id, winner_flags, winner_metrics
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
                    winner_flags=winner_flags,
                    bottleneck=bottleneck_analysis,
                    research=kernel_research,
                )

            # ── Report generation ─────────────────────────────────────────
            await self._generate_report(
                session_id, best_strategy, research_report,
                evolution_result=evolution_result,
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

        best_strategy = await so.run(
            session_id=session_id,
            baseline_flags=winner_flags,
            baseline_metrics=winner_metrics,
            context_configs=list(self.cfg.context_configs),
            max_iterations=12,
        )
        log.info("Stage 2 done: best_strategy=%s", best_strategy)
        return best_strategy, winner_metrics

    # ------------------------------------------------------------------
    # Stage 3 — Deep Profiling + Bottleneck Reasoning
    # ------------------------------------------------------------------

    async def _stage3(
        self,
        session_id: str,
        winner_flags: dict,
        winner_metrics: dict,
    ):
        """
        Run Stage 3: multi-source profiling + deep bottleneck reasoning.

        Pipeline:
          1. Torch profiler trace (category breakdown: attention/GEMM/MoE/comm)
          2. Hardware counters (Nsight Compute or rocprof — if tools available)
          3. BottleneckReasoningAgent (LLM synthesises all signals into a
             precise bottleneck class + evidence chain)
          4. ResearchAgent (LLM-ranked optimization recommendations)
          5. KernelResearchAgent (deep research on best kernel implementations)

        Returns (research_report, bottleneck_analysis, kernel_research).
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
        researcher = ResearchAgent(do_client=self._do_client)
        research_report = await researcher.analyse(
            trace=trace,
            winner_flags=winner_flags,
            model_id=self.cfg.model_id,
            gpu_type=self.cfg.gpu_type,
        )
        log.info(
            "Stage 3d research: %d recommendations, custom_kernel_warranted=%s",
            len(research_report.recommendations),
            research_report.custom_kernel_warranted,
        )

        # ── 3e. Kernel Research (deep kernel-level research) ──────────────
        kernel_research = None
        if research_report.custom_kernel_warranted or (
            bottleneck_analysis.recommended_action.startswith("kernel_generation")
        ):
            log.info("Stage 3e: running deep kernel research...")
            kernel_researcher = KernelResearchAgent(do_client=self._do_client)
            kernel_research = await kernel_researcher.research(
                bottleneck=bottleneck_analysis,
                trace=trace,
                model_id=self.cfg.model_id,
                gpu_type=self.cfg.gpu_type,
                winner_flags=winner_flags,
            )
            log.info(
                "Stage 3e kernel research: %d approaches, proceed_to_generation=%s",
                len(kernel_research.approaches),
                kernel_research.proceed_to_generation,
            )

        return research_report, bottleneck_analysis, kernel_research

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
        )
        log.info("Reports written: %s", {k: str(v) for k, v in paths.items()})
