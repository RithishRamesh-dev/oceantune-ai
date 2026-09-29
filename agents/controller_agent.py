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
from agents.critic_agent import CriticAgent
from core.flag_merge import infer_microbench_op, merge_flags, to_vllm_flags_dict
from core.measurement_gate import MeasurementGate
from core.search_space import SearchSpace, VLLMFlags
from core.session_breakdown import build_session_breakdown, write_session_breakdown
from core.experience_constraints import (
    collect_session_constraints,
    render_constraints_block,
)
from core.operating_point_sweep import SweepResult, run_operating_point_sweep
from core.kernel_integration import (
    plan_integration,
    run_e2e_rebench,
    e2e_rebench_placeholder,
)
from core.macro_cycle import (
    MacroCycleState,
    advance_macro_cycle,
    should_reloop_stage2_3,
)
from core.knowledge_pack import load_knowledge_block
from core.fusion import category_shares_from_trace, diagnose_fusion
from core.enablement import EnablementResult, apply_repair, repair_ladder
from core.policy_gate import PolicyGate
from core.framework_backend import get_framework_backend
from core.draft_registry import planner_speculative_hint
from core.kernel_campaign import run_fusion_campaign
from core.kernel_ledger import KernelLedger
from core.prelude import PreludeResult, accept_warm_replay, build_prelude_plan
from core.session_checkpoint import (
    RoundBudget,
    SessionCheckpoint,
    budget_exhausted,
    load_checkpoint,
    new_checkpoint,
    save_checkpoint,
    session_remaining_sec,
    should_skip_phase,
)
from core.attempt_ledger import AttemptLedger
from core.search_policy import SearchPolicyState, update_search_policy
from core.serving_patches import select_patches
from core.quantization_schemes import resolve_quantization, build_quantization_prompt
from core.recipe_kb import RecipeKnowledgeBase

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
        checkpoint: Optional[SessionCheckpoint] = None
        attempt_ledger: Optional[AttemptLedger] = None
        prelude_result: Optional[PreludeResult] = None
        search_state = SearchPolicyState()
        policy_denials: List[Dict[str, Any]] = []
        try:
            resume_id = (getattr(self.cfg, "resume_session_id", "") or "").strip()
            if resume_id:
                checkpoint = load_checkpoint(resume_id)
                if checkpoint:
                    session_id = resume_id
                    self.session_id = session_id
                    log.info("Resuming session from checkpoint: %s phase=%s", session_id, checkpoint.phase)
                else:
                    log.warning("Resume id %s has no checkpoint — starting fresh", resume_id)
                    resume_id = ""

            if not resume_id:
                # Create MongoDB session document
                session_id = await self._db.create_session(
                    model_id=self.cfg.model_id,
                    gpu_type=self.cfg.gpu_type,
                    strategy=self.cfg.optimiser.strategy,
                    context_configs=[[c[0], c[1]] for c in self.cfg.context_configs],
                )
                self.session_id = session_id
                max_min = float(getattr(self.cfg, "session_max_minutes", 0.0) or 0.0)
                checkpoint = new_checkpoint(
                    session_id,
                    budget=RoundBudget(session_max_sec=max_min * 60.0 if max_min > 0 else 0.0),
                )
                save_checkpoint(checkpoint)

            log.info("MongoDB session: %s", session_id)
            log.info(
                "Framework backend: %s",
                getattr(self.cfg, "framework", "vllm"),
            )
            # Validate framework is constructible
            get_framework_backend(getattr(self.cfg, "framework", "vllm"))

            attempt_ledger = AttemptLedger(session_id=session_id)

            # Serving patches → env for shadow / E2E
            if getattr(self.cfg, "serving_patches_enabled", True):
                patch_sel = select_patches(
                    framework=getattr(self.cfg, "framework", "vllm"),
                    framework_version=str(getattr(self.cfg, "framework_version", "0.6.0")),
                )
                for k, v in patch_sel.env.items():
                    import os
                    os.environ.setdefault(k, v)
                log.info(
                    "Serving patches selected=%d skipped=%d",
                    len(patch_sel.selected),
                    len(patch_sel.skipped),
                )

            # Quantization scheme → seed delta
            quant_seed: dict = {}
            qscheme = str(getattr(self.cfg, "quantization_scheme", "none") or "none")
            if qscheme and qscheme != "none":
                qdec = resolve_quantization(scheme=qscheme, gpu_type=self.cfg.gpu_type)
                if qdec.allowed:
                    quant_seed = dict(qdec.flags_delta)
                    log.info("Quantization scheme %s → %s", qscheme, quant_seed)
                else:
                    log.warning("Quantization scheme rejected: %s", qdec.reason)

            # ── Enablement (boot repair) ──────────────────────────────────
            enablement_result: Optional[EnablementResult] = None
            enablement_seed_flags: dict = dict(quant_seed)
            if not should_skip_phase(checkpoint, "enablement"):
                if getattr(self.cfg, "enablement_enabled", True):
                    d = PolicyGate().check_action("enablement", "baseline_probe")
                    if not d.allowed:
                        policy_denials.append(d.to_dict())
                    enablement_result = await self._run_enablement(session_id)
                    if enablement_result and enablement_result.winning_flags:
                        enablement_seed_flags = {
                            **enablement_seed_flags,
                            **dict(enablement_result.winning_flags),
                        }
                checkpoint.phase = "prelude"
                save_checkpoint(checkpoint)

            # ── Prelude (warm-recipe replay + reject seed) ────────────────
            prelude_seed_flags: dict = {}
            if (
                getattr(self.cfg, "prelude_enabled", True)
                and not should_skip_phase(checkpoint, "prelude")
            ):
                prelude_result = await self._run_prelude(
                    session_id,
                    attempt_ledger=attempt_ledger,
                )
                if prelude_result and prelude_result.replay_accepted:
                    prelude_seed_flags = dict(prelude_result.seed_flags)
                checkpoint.phase = "stage1"
                checkpoint.policy_denials = policy_denials
                save_checkpoint(checkpoint)

            stage1_seed = {**enablement_seed_flags, **prelude_seed_flags}

            # ── Stage 1: vLLM Config Search ───────────────────────────────
            if checkpoint and checkpoint.winner_flags and should_skip_phase(checkpoint, "stage1"):
                winner_flags = dict(checkpoint.winner_flags)
                stage1_fitness = float(checkpoint.stage1_fitness or 0.0)
                log.info("Skipping Stage 1 (resume) fitness=%.2f", stage1_fitness)
            else:
                winner_flags, _, stage1_fitness = await self._stage1(
                    session_id,
                    enablement_seed_flags=stage1_seed or None,
                    attempt_ledger=attempt_ledger,
                    search_state=search_state,
                )
                checkpoint.winner_flags = dict(winner_flags or {})
                checkpoint.stage1_fitness = float(stage1_fitness or 0.0)
                checkpoint.phase = "stage2"
                save_checkpoint(checkpoint)

            # ── Macro Stage 2↔3 cycles ────────────────────────────────────
            from agents.research_agent import ResearchReport
            best_strategy = {}
            winner_metrics: dict = {}
            stage2_fitness: float = 0.0
            merged_flags = dict(winner_flags) if winner_flags else {}
            attention_suite = None
            research_report: Optional[ResearchReport] = None
            bottleneck_analysis = None
            kernel_research = None
            evolution_result = None
            integration_plan = None
            campaign_result = None
            fusion_diagnosis = None
            stage3_fitness: float = stage1_fitness
            applied_recs: list = []
            all_tried_recs: list = []
            stage3_flags = merged_flags
            macro_state = MacroCycleState(
                macro_cycle=int(checkpoint.macro_cycle or 0) if checkpoint else 0,
                gain_at_cycle_start=stage1_fitness,
            )

            if not winner_flags:
                log.warning("Stage 1 produced no winner — skipping Stage 2 and Stage 3")
            else:
                cycle_flags = dict(winner_flags)
                while True:
                    if budget_exhausted(checkpoint):
                        log.warning("Session wall-clock budget exhausted — stopping macro")
                        break
                    log.info(
                        "=== Macro cycle %d: Stage 2 + Stage 3 ===",
                        macro_state.macro_cycle,
                    )
                    best_strategy, winner_metrics, stage2_fitness, attention_suite = (
                        await self._stage2(
                            session_id,
                            cycle_flags,
                            profiler_hints=(
                                {
                                    "primary_bottleneck": getattr(
                                        bottleneck_analysis, "primary_bottleneck", ""
                                    ),
                                    "fusion": (
                                        fusion_diagnosis.to_dict()
                                        if fusion_diagnosis else None
                                    ),
                                    "search_policy": update_search_policy(
                                        search_state,
                                        current_fitness=stage3_fitness,
                                        stall_limit=int(
                                            getattr(self.cfg, "search_stall_limit", 2)
                                        ),
                                    ).to_dict(),
                                }
                                if macro_state.macro_cycle > 0 and bottleneck_analysis
                                else None
                            ),
                        )
                    )
                    # PolicyGate: sanitize Stage 2 strategy keys
                    best_strategy = PolicyGate().sanitize_flags("stage2", best_strategy) or best_strategy
                    merged_flags = merge_flags(cycle_flags, best_strategy)

                    (
                        research_report,
                        bottleneck_analysis,
                        kernel_research,
                        stage3_fitness,
                        applied_recs,
                        all_tried_recs,
                        stage3_flags,
                        fusion_diagnosis,
                    ) = await self._stage3(
                        session_id, merged_flags, winner_metrics,
                        stage2_fitness=stage2_fitness,
                        stage2_strategy=best_strategy,
                        attention_suite=attention_suite,
                    )

                    rem = session_remaining_sec(checkpoint) if checkpoint else None
                    decision = should_reloop_stage2_3(
                        state=macro_state,
                        current_fitness=stage3_fitness,
                        enabled=bool(getattr(self.cfg, "macro_cycle_enabled", True)),
                        max_macro_cycles=int(getattr(self.cfg, "macro_cycle_max", 2)),
                        session_remaining_sec=rem,
                        min_remaining_sec=float(
                            getattr(self.cfg, "macro_cycle_min_remaining_sec", 1800.0)
                        ),
                    )
                    advance_macro_cycle(
                        macro_state,
                        decision=decision,
                        current_fitness=stage3_fitness,
                    )
                    checkpoint.macro_cycle = macro_state.macro_cycle
                    checkpoint.stage2_fitness = float(stage2_fitness or 0.0)
                    checkpoint.stage3_fitness = float(stage3_fitness or 0.0)
                    checkpoint.best_strategy = dict(best_strategy or {})
                    checkpoint.winner_flags = dict(stage3_flags or {})
                    save_checkpoint(checkpoint)
                    if not decision.reloop:
                        log.info(
                            "Macro-cycle stop: %s (cycle=%d gain=%.2f%%)",
                            decision.reason,
                            decision.macro_cycle,
                            decision.cycle_gain_pct,
                        )
                        break
                    cycle_flags = dict(stage3_flags)
                    log.info(
                        "Macro-cycle reloop → cycle %d with updated flags",
                        macro_state.macro_cycle,
                    )

            checkpoint.phase = "sweep"
            save_checkpoint(checkpoint)

            # ── Operating-point sweep (post-winner SWEEP) ─────────────────
            sweep_result: Optional[SweepResult] = None
            final_serving_flags = stage3_flags or merged_flags or winner_flags or {}
            if final_serving_flags and not budget_exhausted(checkpoint):
                d = PolicyGate().check_action("sweep", "operating_point_sweep")
                if not d.allowed:
                    policy_denials.append(d.to_dict())
                else:
                    sweep_result = await self._run_operating_point_sweep(
                        session_id=session_id,
                        flags=final_serving_flags,
                    )
            checkpoint.phase = "stage4"
            save_checkpoint(checkpoint)

            # ── Stage 4: Autonomous Kernel Engineering ────────────────────
            stage4_enabled = getattr(self.cfg, "stage4_enabled", False)
            if (
                stage4_enabled
                and winner_flags
                and bottleneck_analysis is not None
            ):
                evolution_result = await self._stage4(
                    session_id=session_id,
                    winner_flags=stage3_flags,
                    bottleneck=bottleneck_analysis,
                    research=kernel_research,
                )
                if evolution_result and evolution_result.best_kernel:
                    kpath = getattr(evolution_result.best_kernel, "file_path", "") or ""
                    env_flag = "OCEANTUNE_SHADOW_KERNEL"
                    if fusion_diagnosis and fusion_diagnosis.matched_patterns:
                        env_flag = fusion_diagnosis.matched_patterns[0].get(
                            "env_flag", env_flag
                        )
                    integration_plan = plan_integration(
                        op_type=getattr(evolution_result, "op_type", "") or "",
                        kernel_path=kpath,
                        micro_speedup_pct=float(
                            getattr(evolution_result, "best_speedup_pct", 0.0) or 0.0
                        ),
                        env_flag=env_flag,
                    )
                    if getattr(self.cfg, "stage4_e2e_enabled", True):
                        node_cfg = self.cfg.nodes[0]
                        gpu_alloc = GPUSlotAllocator(
                            gpu_indices=node_cfg.gpu_indices,
                            gpu_type=node_cfg.gpu_type,
                        )
                        port_alloc = PortAllocator(
                            start=self.cfg.coordinator.port_pool_start,
                            end=self.cfg.coordinator.port_pool_end,
                        )
                        ctx = list(self.cfg.context_configs) or [(1024, 1024)]
                        integration_plan = await run_e2e_rebench(
                            plan=integration_plan,
                            session_id=session_id,
                            model_id=self.cfg.model_id,
                            gpu_type=self.cfg.gpu_type,
                            flags=stage3_flags or final_serving_flags,
                            incumbent_fitness=max(
                                stage3_fitness, stage2_fitness, stage1_fitness
                            ),
                            gpu_alloc=gpu_alloc,
                            port_alloc=port_alloc,
                            docker_image=self.cfg.vllm.docker_image,
                            startup_timeout_sec=self.cfg.vllm.startup_timeout_sec,
                            concurrency_levels=list(
                                self.cfg.benchmark.concurrency_levels or [1, 4, 16, 64]
                            )[:4],
                            num_prompts=min(20, self.cfg.benchmark.num_prompts),
                            input_len=ctx[0][0],
                            output_len=ctx[0][1],
                            primary_metric=self.cfg.optimiser.primary_metric,
                            env_flag=env_flag,
                        )
                    else:
                        integration_plan = await e2e_rebench_placeholder(
                            plan=integration_plan,
                            session_id=session_id,
                        )

                # Stage 4b fusion campaign + kernel ledger
                if getattr(self.cfg, "stage4_campaign_enabled", True):
                    try:
                        node_cfg = self.cfg.nodes[0]
                        campaign_result = await run_fusion_campaign(
                            session_id=session_id,
                            model_id=self.cfg.model_id,
                            gpu_type=self.cfg.gpu_type,
                            matched_patterns=(
                                fusion_diagnosis.matched_patterns
                                if fusion_diagnosis else []
                            ),
                            evolution_result=evolution_result,
                            incumbent_fitness=max(
                                stage3_fitness, stage2_fitness, stage1_fitness
                            ),
                            stage3_flags=stage3_flags or final_serving_flags,
                            run_e2e=False,  # E2E already handled above when kernel exists
                            e2e_runner=None,
                            primary_metric=self.cfg.optimiser.primary_metric,
                        )
                        log.info(
                            "Stage 4b campaign: %d trials best=%s",
                            len(campaign_result.trials),
                            (
                                campaign_result.best_trial.pattern_id
                                if campaign_result.best_trial else None
                            ),
                        )
                    except Exception as exc:
                        log.warning("Stage 4b campaign failed: %s", exc)

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

            # ── Recipe KB sedimentation (CLOSE) ───────────────────────────
            final_flags = stage3_flags or winner_flags or {}
            final_fitness = max(stage3_fitness, stage2_fitness, stage1_fitness)
            recipe_id = ""
            if final_flags and final_fitness > 0:
                recipe = await self._sediment_recipe(
                    session_id=session_id,
                    final_flags=final_flags,
                    final_fitness=final_fitness,
                    stage1_fitness=stage1_fitness,
                    best_strategy=best_strategy,
                    research_report=research_report,
                    peak_throughput=(
                        sweep_result.peak_throughput if sweep_result else 0.0
                    ),
                )
                if recipe is not None:
                    recipe_id = getattr(recipe, "recipe_id", "") or ""

            # ── Session breakdown CLOSE artifact ──────────────────────────
            peak_thr = 0.0
            best_conc = 0
            if sweep_result and not sweep_result.skipped:
                peak_thr = sweep_result.peak_throughput
                best_conc = sweep_result.best_concurrency
            elif winner_metrics:
                peak_thr = float(
                    winner_metrics.get("peak_throughput_tokens_per_sec") or 0
                )
                best_conc = int(winner_metrics.get("best_concurrency") or 0)

            stage4_speedup = 0.0
            stage4_path = ""
            if evolution_result is not None:
                stage4_speedup = float(
                    getattr(evolution_result, "best_speedup_pct", 0.0) or 0.0
                )
                bk = getattr(evolution_result, "best_kernel", None)
                if bk is not None:
                    stage4_path = getattr(bk, "file_path", "") or ""

            breakdown = build_session_breakdown(
                session_id=session_id,
                model_id=self.cfg.model_id,
                gpu_type=self.cfg.gpu_type,
                stage1_fitness=stage1_fitness,
                stage2_fitness=stage2_fitness,
                stage3_fitness=stage3_fitness,
                winner_flags=final_flags,
                stage2_strategy=best_strategy,
                stage3_applied_recs=applied_recs,
                peak_throughput=peak_thr,
                best_concurrency=best_conc,
                bottleneck_primary=(
                    getattr(bottleneck_analysis, "primary_bottleneck", "")
                    if bottleneck_analysis is not None
                    else ""
                ),
                attention_backend=str(
                    (final_flags or {}).get("attention_backend") or ""
                ),
                recipe_id=recipe_id,
                stage4_speedup_pct=stage4_speedup,
                stage4_kernel_path=stage4_path,
                stop_reason="pipeline_complete",
                extras={
                    "operating_point_sweep": (
                        sweep_result.to_dict() if sweep_result else None
                    ),
                    "kernel_integration": (
                        integration_plan.to_dict() if integration_plan else None
                    ),
                    "macro_cycle": macro_state.to_dict(),
                    "fusion_diagnosis": (
                        fusion_diagnosis.to_dict() if fusion_diagnosis else None
                    ),
                    "enablement": (
                        enablement_result.to_dict() if enablement_result else None
                    ),
                    "kernel_campaign": (
                        campaign_result.to_dict() if campaign_result else None
                    ),
                    "framework": getattr(self.cfg, "framework", "vllm"),
                    "prelude": (
                        prelude_result.to_dict() if prelude_result else None
                    ),
                    "policy_denials": policy_denials,
                    "search_policy": search_state.to_dict(),
                    "attempt_ledger": (
                        attempt_ledger.to_list()[-40:] if attempt_ledger else []
                    ),
                    "quantization_scheme": getattr(
                        self.cfg, "quantization_scheme", "none"
                    ),
                },
            )
            if integration_plan is not None:
                breakdown["stage4"]["integrated_into_serving"] = bool(
                    integration_plan.e2e_verified
                )
                breakdown["stage4"]["integration_status"] = integration_plan.status
            write_session_breakdown(breakdown)

            checkpoint.phase = "done"
            save_checkpoint(checkpoint)
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
    # Prelude — warm-recipe replay + reject seed
    # ------------------------------------------------------------------

    async def _run_prelude(
        self,
        session_id: str,
        *,
        attempt_ledger: Optional[AttemptLedger] = None,
    ) -> PreludeResult:
        """Lookup recipe, build prelude plan; optionally accept claimed flags as seed."""
        log.info("=== Prelude: warm-recipe plan ===")
        PolicyGate().check_action("prelude", "warm_replay")
        recipe = None
        try:
            rkb = RecipeKnowledgeBase(self._db)
            recipe = await rkb.lookup(
                model_id=self.cfg.model_id,
                gpu_type=self.cfg.gpu_type,
                framework=getattr(self.cfg, "framework", "vllm"),
            )
        except Exception as exc:
            log.debug("Prelude recipe lookup failed: %s", exc)

        plan = build_prelude_plan(
            recipe=recipe,
            enable_profile_arm=False,
            min_confidence=float(getattr(self.cfg, "prelude_min_confidence", 0.7)),
            max_warm_trials=int(getattr(self.cfg, "warmstart_max_trials", 3)),
        )
        if attempt_ledger and plan.reject_pitfalls:
            for pit in plan.reject_pitfalls[:8]:
                attempt_ledger.record(
                    phase="prelude",
                    error=pit,
                    metadata={"source": "recipe_pitfall"},
                )

        # Confidence-gated seed without a separate GPU probe when fitness claim
        # is already measured in a prior session (retention check deferred to Stage 1).
        if plan.should_benchmark_replay and plan.replay_flags:
            # Accept as seed; Stage 1 will re-measure. Retention ratio applied
            # against claimed fitness only when a measured value is supplied later.
            result = accept_warm_replay(
                plan=plan,
                measured_fitness=float(plan.replay_fitness_claimed or 0.0),
                min_retention_ratio=0.85,
            )
            log.info(
                "Prelude warm-replay: accepted=%s reason=%s flags=%d rejects=%d",
                result.replay_accepted,
                result.stop_reason,
                len(result.seed_flags),
                len(plan.reject_fingerprints),
            )
            return result

        return PreludeResult(plan=plan, stop_reason="no_replay_armed")

    # ------------------------------------------------------------------
    # Enablement — boot repair ladder
    # ------------------------------------------------------------------

    async def _run_enablement(self, session_id: str) -> EnablementResult:
        """
        Probe a minimal vLLM boot; on failure walk the repair ladder.
        Returns EnablementResult with winning_flags if any config boots.
        """
        log.info("=== Enablement: baseline boot probe ===")
        PolicyGate().check_action("enablement", "baseline_probe")

        # Detect MoE / MLA from models.yaml when possible
        moe = False
        mla = False
        try:
            import yaml
            models = yaml.safe_load(
                (REPO_ROOT / "configs" / "models.yaml").read_text()
            ) or {}
            mid = self.cfg.model_id.lower()
            for _alias, meta in (models.get("models") or {}).items():
                hf = str((meta or {}).get("hf_id") or "").lower()
                if hf == mid or mid.endswith(hf.split("/")[-1]):
                    moe = bool((meta or {}).get("moe"))
                    mla = bool((meta or {}).get("mla"))
                    break
        except Exception:
            pass

        base = {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "distributed_executor_backend": "mp",
            "cpu_offload_gb": 0,
            "gpu_memory_utilization": 0.90,
            "block_size": 1 if mla else 16,
        }
        # DeepSeek-V4.1-Flash Quark-MXFP4 / MI355X known-good baseline recipe
        mid = (self.cfg.model_id or "").lower()
        if (
            "deepseek-v4" in mid
            or "deepseek_v4" in mid
            or "quark-mxfp4" in mid
            or getattr(self.cfg, "gpu_type", "") == "MI355X"
        ):
            base.update({
                "tensor_parallel_size": 4,
                "max_num_batched_tokens": 8192,
                "gpu_memory_utilization": 0.90,
                "enforce_eager": True,
                "trust_remote_code": True,
                "tokenizer_mode": "deepseek_v41",
            })
            log.info("Enablement: using DeepSeek-V4.1 / MI355X baseline recipe (TP=4)")
        known = set(VLLMFlags.__dataclass_fields__)
        result = EnablementResult(success=False)
        max_repairs = int(getattr(self.cfg, "enablement_max_repairs", 4))

        async def _probe(flags_dict: dict, label: str) -> tuple[bool, str]:
            clean = {k: v for k, v in flags_dict.items() if k in known}
            try:
                vf = VLLMFlags(**clean)
            except Exception as exc:
                return False, str(exc)
            from dataclasses import asdict as _asdict
            import hashlib
            # Unique per enablement step so repairs aren't skipped as duplicates
            fp = hashlib.sha1(
                f"{vf.fingerprint()}:enablement:{label}".encode()
            ).hexdigest()[:12]
            config_id = await self._db.insert_config(
                session_id=session_id,
                fingerprint=fp,
                flags={k: v for k, v in _asdict(vf).items() if k != "run_id"},
                generation=-100,
                priority=-100,
            )
            if config_id is None:
                return False, "duplicate_config"
            await self._run_single(
                session_id=session_id,
                config_id=config_id,
                context_configs=[(128, 32)],
            )
            doc = await self._db.get_config_by_id(config_id)
            err = (doc or {}).get("error") or ""
            fitness = float((doc or {}).get("fitness_score") or 0.0)
            ok = (not err) and fitness > 0
            return ok, err

        ok, err = await _probe(base, "baseline")
        result.steps_tried.append({
            "name": "baseline", "ok": ok, "error": err[:200] if err else "",
        })
        if ok:
            result.success = True
            result.winning_flags = base
            result.stop_reason = "baseline_ok"
            log.info("Enablement: baseline OK")
            return result

        steps = repair_ladder(base_flags=base, error=err, moe=moe, mla=mla)
        for step in steps[:max_repairs]:
            candidate = apply_repair(base, step)
            candidate = PolicyGate().sanitize_flags("enablement", candidate) or candidate
            ok, err2 = await _probe(candidate, step.name)
            result.steps_tried.append({
                "name": step.name,
                "ok": ok,
                "error": (err2 or "")[:200],
                "flags_delta": step.flags_delta,
            })
            if ok:
                result.success = True
                result.winning_flags = candidate
                result.stop_reason = f"repaired:{step.name}"
                log.info("Enablement: repaired via %s", step.name)
                return result
            err = err2 or err

        result.final_error = (err or "")[:300]
        result.stop_reason = "enablement_exhausted"
        log.warning("Enablement exhausted; continuing Stage 1 cold: %s", result.final_error)
        return result

    # ------------------------------------------------------------------
    # Recipe KB sedimentation (session CLOSE)
    # ------------------------------------------------------------------

    async def _sediment_recipe(
        self,
        *,
        session_id: str,
        final_flags: dict,
        final_fitness: float,
        stage1_fitness: float,
        best_strategy: dict,
        research_report=None,
        peak_throughput: float = 0.0,
    ):
        """Persist winning config + lessons/pitfalls into the Recipe KB.

        Returns the Recipe on success, or None.
        """
        try:
            from core.recipe_kb import (
                RecipeKnowledgeBase,
                lessons_from_analyst,
                pitfalls_from_failures,
            )

            peak = float(peak_throughput or 0.0)
            fingerprint = ""
            top = await self._db.get_top_configs(session_id, n=1)
            if top:
                em = top[0].get("enriched_metrics") or {}
                if peak <= 0:
                    peak = float(em.get("peak_throughput_tokens_per_sec") or 0.0)
                fingerprint = str(top[0].get("fingerprint") or "")

            lessons = []
            pitfalls = []
            if research_report is not None:
                lessons = lessons_from_analyst(
                    explanation=getattr(research_report, "bottleneck_explanation", "")
                    or "",
                    recommendation=(
                        research_report.recommendations[0].title
                        if research_report.recommendations
                        else ""
                    ),
                    session_id=session_id,
                )
            # OOM / failed configs → pitfalls
            failed = await self._db.list_failed_configs(session_id, limit=20)
            pitfalls = pitfalls_from_failures(
                [
                    {
                        "flags": c.get("flags"),
                        "error": c.get("error"),
                    }
                    for c in failed
                ],
                session_id=session_id,
            )

            what_worked = []
            if best_strategy:
                what_worked.append({
                    "name": "stage2_strategy",
                    "flags": best_strategy,
                    "gain_pct": (
                        ((final_fitness - stage1_fitness) / stage1_fitness * 100)
                        if stage1_fitness > 0
                        else 0.0
                    ),
                })
            try:
                for kl in KernelLedger().lessons_for_recipe(session_id):
                    what_worked.append(kl)
            except Exception:
                pass

            rkb = RecipeKnowledgeBase(self._db)
            recipe = await rkb.sediment(
                model_id=self.cfg.model_id,
                gpu_type=self.cfg.gpu_type,
                session_id=session_id,
                best_flags=final_flags,
                best_fitness=final_fitness,
                best_fingerprint=fingerprint,
                peak_throughput=peak,
                fitness_before=stage1_fitness,
                stage2_strategy=best_strategy or {},
                lessons=lessons,
                pitfalls=pitfalls,
                what_worked=what_worked,
            )
            return recipe
        except Exception as exc:
            log.warning("Recipe KB sedimentation failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------

    async def _stage1(
        self,
        session_id: str,
        enablement_seed_flags: Optional[dict] = None,
        attempt_ledger: Optional[AttemptLedger] = None,
        search_state: Optional[SearchPolicyState] = None,
    ) -> Tuple[dict, str, float]:
        """
        Run Stage 1: Iterative agent-guided hyperparameter search.

        Iteration 0: bare minimum vLLM flags (establishes baseline).
        Iteration N: PlannerAgent.propose_next() observes all prior results
                     and proposes a single targeted change.

        Returns (winner_flags_dict, winner_fingerprint, best_fitness).
        """
        log.info("=== Stage 1: Agent-guided vLLM Config Search ===")
        d = PolicyGate().check_action("stage1", "propose_flags")
        if not d.allowed:
            log.warning("PolicyGate denied stage1 propose_flags: %s", d.reason)

        _search = search_state or SearchPolicyState()
        _ = attempt_ledger  # used by callers for denylist; failures recorded in Mongo
        n_gpus = len(self.cfg.nodes[0].gpu_indices)
        n_iterations = self.cfg.optimiser.generations
        context_configs = list(self.cfg.context_configs)

        planner = PlannerAgent(
            do_client=self._do_client,
            db=self._db,
            search_space=self._search_space,
        )
        analyst = AnalystAgent(do_client=self._do_client, db=self._db)

        # ── Recipe KB warm-start (Hyperloom-inspired cascade) ─────────────
        recipe_context = ""
        recipe_seed_flags: list[VLLMFlags] = []
        recipe = None
        experience_block = ""

        # Enablement seeds first (boot-proven safer flags)
        if enablement_seed_flags:
            known = set(VLLMFlags.__dataclass_fields__)
            try:
                ef = VLLMFlags(**{
                    k: v for k, v in enablement_seed_flags.items() if k in known
                })
                ef.run_id = ef.fingerprint()
                recipe_seed_flags.append(ef)
                log.info("Enablement seed injected into Stage 1 queue")
            except Exception as e:
                log.debug("Enablement seed skipped: %s", e)

        try:
            from core.recipe_kb import RecipeKnowledgeBase
            rkb = RecipeKnowledgeBase(self._db)
            recipe = await rkb.lookup(
                model_id=self.cfg.model_id,
                gpu_type=self.cfg.gpu_type,
                framework=getattr(self.cfg, "framework", "vllm"),
            )
            if recipe:
                recipe_context = recipe.planner_context()
                log.info(
                    "Recipe KB hit: id=%s fitness=%.4f confidence=%.2f lessons=%d",
                    recipe.recipe_id,
                    recipe.best_fitness,
                    recipe.confidence,
                    len(recipe.lessons),
                )
                if recipe.best_flags:
                    known = set(VLLMFlags.__dataclass_fields__)
                    flags_clean = {
                        k: v for k, v in recipe.best_flags.items() if k in known
                    }
                    try:
                        rf = VLLMFlags(**flags_clean)
                        rf.run_id = rf.fingerprint()
                        recipe_seed_flags.append(rf)
                    except Exception as e:
                        log.debug("Recipe flag seed skipped: %s", e)
        except Exception as e:
            log.debug("Recipe KB lookup failed: %s", e)

        # Experience constraints (negative priors from failures + recipe pitfalls)
        try:
            constraints = await collect_session_constraints(
                self._db, session_id=session_id, recipe=recipe,
            )
            experience_block = render_constraints_block(constraints)
            if experience_block:
                log.info(
                    "Experience constraints ready (%d rules)",
                    len(constraints),
                )
        except Exception as e:
            log.debug("Experience constraint collect skipped: %s", e)

        # Vendor knowledge pack (NVIDIA / AMD levers)
        try:
            kb = load_knowledge_block(
                gpu_type=self.cfg.gpu_type,
                max_chars=3000,
                focus_keywords=["attention", "kv", "memory"],
            )
            if kb:
                experience_block = (
                    (experience_block + "\n\n" if experience_block else "") + kb
                )
        except Exception as e:
            log.debug("Knowledge pack skipped: %s", e)

        # Speculative draft registry hint
        try:
            vendor = (
                "amd"
                if self.cfg.gpu_type in {"MI300X", "MI325X", "MI350X", "MI355X"}
                else "nvidia"
            )
            spec_hint = planner_speculative_hint(self.cfg.model_id, vendor=vendor)
            experience_block = (
                (experience_block + "\n\n" if experience_block else "")
                + f"=== Speculative decoding policy ===\n{spec_hint}"
            )
        except Exception as e:
            log.debug("Draft registry hint skipped: %s", e)
        # ── Cross-session warm-start ─────────────────────────────────────
        # Query the best flags from prior sessions for this (model, GPU) pair.
        # These are inserted as the first configs so Stage 1 doesn't waste
        # iterations re-discovering configurations already known to be good.
        warm_start_flags: list[VLLMFlags] = list(recipe_seed_flags)
        try:
            prior_bests = await self._db.get_best_flags_for_model(
                model_id=self.cfg.model_id,
                gpu_type=self.cfg.gpu_type,
                exclude_session_id=session_id,
                top_n=3,
            )
            for pb in prior_bests:
                known = set(VLLMFlags.__dataclass_fields__)
                flags_clean = {k: v for k, v in pb["flags"].items() if k in known}
                try:
                    warm_flags = VLLMFlags(**flags_clean)
                    warm_flags.run_id = warm_flags.fingerprint()
                    warm_start_flags.append(warm_flags)
                    log.info(
                        "Warm-start seed: fingerprint=%s fitness=%.4f (session %s)",
                        pb["fingerprint"][:8], pb["fitness_score"], pb["session_id"][:8],
                    )
                except Exception as e:
                    log.debug("Warm-start seed skipped (invalid flags): %s", e)
        except Exception as e:
            log.debug("Cross-session warm-start query failed: %s", e)

        # Iteration 0: bare minimum — let vLLM choose all defaults
        # (except DeepSeek-V4.1 / MI355X which require tokenizer + trust flags)
        current_best = VLLMFlags(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            distributed_executor_backend="mp",
            cpu_offload_gb=0,
        )
        mid = (self.cfg.model_id or "").lower()
        if (
            "deepseek-v4" in mid
            or "deepseek_v4" in mid
            or "quark-mxfp4" in mid
            or self.cfg.gpu_type == "MI355X"
        ):
            current_best = VLLMFlags(
                tensor_parallel_size=4,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                distributed_executor_backend="mp",
                cpu_offload_gb=0,
                gpu_memory_utilization=0.90,
                max_num_batched_tokens=8192,
                enforce_eager=True,
                trust_remote_code=True,
                tokenizer_mode="deepseek_v41",
            )
        current_best.run_id = current_best.fingerprint()

        best_fitness = 0.0
        best_flags = current_best
        search_history: list = []
        last_analyst_eval: dict = {}

        # If MongoDB has no prior session data for this (model, GPU), fall back to
        # GPU-type known-good seeds from the planner's static table.
        if not warm_start_flags:
            gpu_seeds = PlannerAgent._GPU_SEEDS.get(self.cfg.gpu_type, [])
            for seed_dict in gpu_seeds:
                known = set(VLLMFlags.__dataclass_fields__)
                try:
                    seed_flags = VLLMFlags(**{
                        **{k: v for k, v in current_best.__dict__.items() if k in known and k != "run_id"},
                        **{k: v for k, v in seed_dict.items() if k in known},
                    })
                    seed_flags.run_id = seed_flags.fingerprint()
                    warm_start_flags.append(seed_flags)
                    log.info(
                        "GPU-type seed config for %s: %s",
                        self.cfg.gpu_type, seed_dict,
                    )
                except Exception as e:
                    log.debug("GPU-type seed skipped: %s", e)

        # Prepend warm-start seeds so they run in the first iterations
        iteration_queue: list = list(warm_start_flags)  # seeds first, then LLM proposals

        for iteration in range(n_iterations):
            flags = None
            rationale = ""

            if iteration == 0:
                # Always benchmark the vLLM-defaults baseline first
                flags = current_best
                rationale = "Baseline: vLLM defaults, no extra flags"
                log.info("Iteration 0 — baseline: bare minimum vLLM flags")

            elif iteration_queue:
                # Consume warm-start seeds from prior sessions before LLM proposals
                flags = iteration_queue.pop(0)
                rationale = f"Cross-session warm-start seed (fingerprint {flags.fingerprint()[:8]})"
                log.info("Iteration %d — warm-start seed: %s", iteration, rationale)

            else:
                # LLM-guided proposal
                top = await self._db.get_top_configs(session_id, n=1)
                best_run = top[0] if top else {}
                best_metrics = best_run.get("enriched_metrics") or best_run.get("raw_metrics") or {}

                # Refresh negative priors before each LLM proposal
                try:
                    constraints = await collect_session_constraints(
                        self._db, session_id=session_id, recipe=recipe,
                    )
                    experience_block = render_constraints_block(constraints)
                except Exception:
                    pass

                flags, rationale = await planner.propose_next(
                    model_id=self.cfg.model_id,
                    gpu_type=self.cfg.gpu_type,
                    n_gpus=n_gpus,
                    current_best=best_flags,
                    current_best_metrics=best_metrics,
                    history=search_history,
                    iteration=iteration,
                    analyst_eval=last_analyst_eval,
                    recipe_context=recipe_context or None,
                    experience_constraints=(
                        (experience_block or "")
                        + "\n"
                        + (
                            update_search_policy(
                                _search,
                                current_fitness=best_fitness,
                                stall_limit=int(
                                    getattr(self.cfg, "search_stall_limit", 2)
                                ),
                            ).prompt_hint
                            if best_fitness > 0
                            else ""
                        )
                        + (
                            ("\n" + attempt_ledger.planner_denylist_text())
                            if attempt_ledger
                            else ""
                        )
                    ) or None,
                )
                log.info("Iteration %d — agent proposal: %s", iteration, rationale[:120])

                # Inject any extra batch proposals into the queue so they get
                # benchmarked in subsequent iterations without additional LLM calls
                if planner._last_batch:
                    extras = planner._last_batch
                    planner._last_batch = []
                    iteration_queue[:0] = [f for f, _ in extras]  # prepend
                    log.info(
                        "Iteration %d — stashed %d extra batch proposals into queue",
                        iteration, len(extras),
                    )

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
            framework=getattr(self.cfg, "framework", "vllm"),
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
                framework=getattr(self.cfg, "framework", "vllm"),
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
        self,
        session_id: str,
        winner_flags: dict,
        profiler_hints: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, ...]:
        """
        Run Stage 2: LLM-guided inference strategy search.

        Explores KV cache strategies, speculative decoding, prefill strategies,
        attention backend selection, and vendor-specific kernel flags on top of
        the Stage 1 winner.

        Returns (best_strategy_config, winner_enriched_metrics, stage2_fitness).
        """
        log.info("=== Stage 2: Inference Strategy Search ===")

        # Stage 1 metrics for LLM context during search
        top = await self._db.get_top_configs(session_id, n=1)
        stage1_metrics = {}
        if top:
            stage1_metrics = (
                top[0].get("enriched_metrics") or top[0].get("raw_metrics") or {}
            )

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
            critic=CriticAgent(do_client=self._do_client),
        )

        # Pass experience constraints + vendor knowledge into Stage 2
        try:
            constraints = await collect_session_constraints(
                self._db, session_id=session_id, recipe=None,
            )
            so.set_experience_constraints(render_constraints_block(constraints))
        except Exception as exc:
            log.debug("Stage 2 experience constraints skipped: %s", exc)
        try:
            kb = load_knowledge_block(
                gpu_type=self.cfg.gpu_type,
                max_chars=3500,
                focus_keywords=["attention", "kv", "memory", "aiter"],
            )
            if kb:
                prev = getattr(so, "_experience_constraints", "") or ""
                so.set_experience_constraints(
                    prev + ("\n\n" if prev else "") + kb
                )
        except Exception as exc:
            log.debug("Stage 2 knowledge pack skipped: %s", exc)

        from core.attention_bench import AttentionBenchmarkSuite

        attention_suite = AttentionBenchmarkSuite(
            db=self._db,
            gpu_type=self.cfg.gpu_type,
            model_id=self.cfg.model_id,
        )
        seeded = await attention_suite.seed_kernel_metadata()
        log.info("Seeded %d kernel_metadata attention entries", seeded)
        micro_ids = await attention_suite.run_microbench_suite(session_id)
        log.info("Attention microbench suite: %d run(s) recorded", len(micro_ids))

        best_strategy, stage2_fitness, stage2_metrics = await so.run(
            session_id=session_id,
            baseline_flags=winner_flags,
            baseline_metrics=stage1_metrics,
            context_configs=list(self.cfg.context_configs),
            max_iterations=12,
            profiler_hints=profiler_hints,
        )

        if getattr(self.cfg, "attention_e2e_enabled", False):
            try:
                # Temporarily force matrix e2e.enabled for this call path
                attention_suite._matrix.setdefault("e2e", {})["enabled"] = True

                async def _bench(**kwargs):
                    return await so._benchmark_strategy(
                        context_configs=list(self.cfg.context_configs)[:1],
                        **kwargs,
                    )

                e2e_results = await attention_suite.run_e2e_backend_trials(
                    session_id=session_id,
                    baseline_flags=winner_flags,
                    benchmark_fn=_bench,
                )
                log.info("Attention E2E backend trials: %d", len(e2e_results or []))
            except Exception as exc:
                log.warning("Attention E2E trials skipped: %s", exc)

        # Prefer metrics from the best Stage 2 benchmark (includes merged config behavior)
        winner_metrics = stage2_metrics or await self._db.get_stage2_winner_metrics(
            session_id
        )
        if not winner_metrics:
            winner_metrics = stage1_metrics
        log.info(
            "Stage 2 done: best_strategy=%s best_fitness=%.4f best_concurrency=%s",
            best_strategy,
            stage2_fitness,
            winner_metrics.get("best_concurrency"),
        )
        return best_strategy, winner_metrics, stage2_fitness, attention_suite

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
        attention_suite=None,
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

        profile_flags = to_vllm_flags_dict(winner_flags)
        trace = await profiler.run(
            session_id=session_id,
            winner_flags=profile_flags,
            optimal_concurrency=optimal_concurrency,
            input_len=input_len,
            output_len=output_len,
        )
        log.info(
            "Stage 3a profile: bottleneck=%s attention=%.1f%% gemm=%.1f%%",
            trace.bottleneck_type, trace.attention_pct, trace.gemm_pct,
        )

        # ── 3a′. YAML kernel harness suite (phase-attributed microbenches) ─
        try:
            from core.kernel_harness import KernelHarness
            harness = KernelHarness(db=self._db, gpu_type=self.cfg.gpu_type)
            harness_results = await harness.run_suite(session_id)
            log.info(
                "Stage 3a harness: %d cases (%d ok)",
                len(harness_results),
                sum(1 for r in harness_results if r.success),
            )
        except Exception as exc:
            log.debug("Kernel harness skipped: %s", exc)

        if attention_suite is not None:
            await self._run_post_profile_attention_trials(
                session_id=session_id,
                winner_flags=winner_flags,
                trace=trace,
                attention_suite=attention_suite,
            )

        # ── 3b. Roofline microbench + hardware counters ───────────────────
        microbench_op = infer_microbench_op(
            bottleneck_kernel=trace.bottleneck_kernel or "",
            bottleneck_type=trace.bottleneck_type or "",
            attention_pct=trace.attention_pct,
            gemm_pct=trace.gemm_pct,
            moe_pct=trace.moe_pct,
        )
        roofline = await self._run_roofline_microbench(
            session_id=session_id,
            op_type=microbench_op,
            kernel_name=trace.bottleneck_kernel or microbench_op,
        )

        hw_counters = None
        top_kernel_name = trace.bottleneck_kernel or ""
        if top_kernel_name:
            hw_counters = await self._collect_hardware_counters(
                session_id=session_id,
                kernel_name=top_kernel_name,
                op_type=microbench_op,
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
            roofline=roofline,
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
        # Run kernel research whenever Stage 4 is enabled OR the research/
        # bottleneck agents flag that custom kernel work is warranted.
        # Previously this was gated on custom_kernel_warranted, so Stage 4
        # never triggered in practice because the LLM rarely sets that flag.
        kernel_research = None
        fusion_diagnosis = None
        try:
            other_pct = max(
                0.0,
                100.0
                - float(trace.attention_pct or 0)
                - float(trace.gemm_pct or 0)
                - float(trace.moe_pct or 0)
                - float(getattr(trace, "comm_pct", 0) or 0),
            )
            shares = category_shares_from_trace(
                attention_pct=float(trace.attention_pct or 0),
                gemm_pct=float(trace.gemm_pct or 0),
                moe_pct=float(trace.moe_pct or 0),
                communication_pct=float(getattr(trace, "comm_pct", 0) or 0),
                other_pct=other_pct,
                bottleneck_type=str(trace.bottleneck_type or ""),
                bottleneck_kernel=str(trace.bottleneck_kernel or ""),
            )
            vendor = (
                "amd"
                if self.cfg.gpu_type in {"MI300X", "MI325X", "MI350X", "MI355X"}
                else "nvidia"
            )
            fusion_diagnosis = diagnose_fusion(
                shares,
                framework=getattr(self.cfg, "framework", "vllm"),
                vendor=vendor,
            )
            log.info(
                "Stage 3 fusion diagnose: candidate=%s reason=%s patterns=%s",
                fusion_diagnosis.is_candidate,
                fusion_diagnosis.reason,
                [m["id"] for m in fusion_diagnosis.matched_patterns],
            )
        except Exception as exc:
            log.debug("Fusion diagnose skipped: %s", exc)

        stage4_enabled = getattr(self.cfg, "stage4_enabled", False)
        need_kernel_work = (
            stage4_enabled
            or research_report.custom_kernel_warranted
            or bottleneck_analysis.recommended_action.startswith("kernel_generation")
            or (fusion_diagnosis is not None and fusion_diagnosis.is_candidate)
        )
        if need_kernel_work:
            log.info("Stage 3f: running deep kernel research...")
            knowledge = ""
            try:
                knowledge = load_knowledge_block(
                    gpu_type=self.cfg.gpu_type,
                    max_chars=4000,
                    focus_keywords=["triton", "fusion", "attention", "hip", "cutlass"],
                )
            except Exception:
                pass
            fusion_ctx = fusion_diagnosis.prompt_block() if fusion_diagnosis else ""
            kernel_researcher = KernelResearchAgent(do_client=self._do_client)
            kernel_research = await kernel_researcher.research(
                bottleneck=bottleneck_analysis,
                trace=trace,
                model_id=self.cfg.model_id,
                gpu_type=self.cfg.gpu_type,
                winner_flags=updated_flags,
                fusion_context=fusion_ctx or None,
                knowledge_block=knowledge or None,
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
            fusion_diagnosis,
        )

    async def _run_post_profile_attention_trials(
        self,
        *,
        session_id: str,
        winner_flags: Dict[str, Any],
        trace,
        attention_suite,
    ) -> None:
        """
        When the profiler trace is compute-bound, benchmark untried attention backends.
        """
        proposals = attention_suite.attention_sweep_proposals(
            winner_flags,
            attention_pct=trace.attention_pct,
            gemm_pct=trace.gemm_pct,
        )
        if not proposals:
            return

        existing = await self._db.list_kernel_runs(session_id, limit=200)
        tried_backends = set()
        for run in existing:
            cfg = run.get("kernel_config") or {}
            if "attention_backend" in cfg:
                tried_backends.add(cfg["attention_backend"])

        context_configs = list(self.cfg.context_configs)
        known_fields = set(VLLMFlags.__dataclass_fields__)

        for prop in proposals[:3]:
            backend = prop.get("strategy_config", {}).get("attention_backend")
            if not backend or backend in tried_backends:
                continue
            trial_clean = to_vllm_flags_dict(
                merge_flags(winner_flags, prop["strategy_config"])
            )
            from dataclasses import asdict
            trial_flags = VLLMFlags(**trial_clean)
            config_id = await self._db.insert_config(
                session_id=session_id,
                fingerprint=trial_flags.fingerprint(),
                flags={k: v for k, v in asdict(trial_flags).items() if k != "run_id"},
                generation=-3,
                priority=-3,
            )
            if config_id is None:
                continue
            log.info(
                "Stage 3 attention trial: backend=%s (trace attention=%.0f%% gemm=%.0f%%)",
                backend, trace.attention_pct, trace.gemm_pct,
            )
            await self._run_single(
                session_id=session_id,
                config_id=config_id,
                context_configs=context_configs,
            )
            tried_backends.add(backend)

    async def _run_roofline_microbench(
        self,
        *,
        session_id: str,
        op_type: str,
        kernel_name: str,
    ):
        """
        Run isolated operator microbench and roofline analysis.
        Persists to kernel_benchmark_runs. Returns RooflineAnalysis or None.
        Never raises.
        """
        try:
            from microbench.operator_bench import OperatorBench
            from microbench.roofline import RooflineAnalyzer

            bench = OperatorBench(gpu_type=self.cfg.gpu_type)
            params = {
                "batch_size": 1,
                "seq_len": 2048,
                "num_heads": 32,
                "head_dim": 128,
                "num_kv_heads": 8,
            }
            if op_type == "gemm":
                params = {"M": 4096, "N": 4096, "K": 4096}
            elif op_type == "moe":
                params = {
                    "batch_size": 64,
                    "seq_len": 512,
                    "hidden_dim": 2048,
                    "num_experts": 64,
                    "top_k": 2,
                }

            result = await bench.run(op_type=op_type, params=params, num_warmup=5, num_iters=20)
            if not result.success:
                log.warning("Roofline microbench failed: %s", result.error)
                return None

            analyzer = RooflineAnalyzer(gpu_type=self.cfg.gpu_type)
            duration_s = max(result.latency_us_p50, 1.0) / 1e6
            op_flops = bench._estimate_flops(op_type, params)
            op_bytes = bench._estimate_memory_bytes(op_type, params)
            analysis = analyzer.analyze(
                kernel_name=kernel_name or op_type,
                op_flops=op_flops,
                op_bytes=op_bytes,
                duration_s=duration_s,
            )

            await self._db.insert_kernel_benchmark_run(
                session_id=session_id,
                op_type=op_type,
                backend="pytorch",
                gpu_type=self.cfg.gpu_type,
                kernel_name=kernel_name,
                params=params,
                metrics={
                    "latency_us_mean": result.latency_us_mean,
                    "latency_us_p50": result.latency_us_p50,
                    "latency_us_p99": result.latency_us_p99,
                    "throughput_tflops": result.tflops_achieved,
                    "memory_gbps": result.mem_bw_gbps,
                    "roofline_bound": (
                        analysis.points[0].bound if analysis.points else "unknown"
                    ),
                    "roofline_efficiency_pct": analysis.overall_efficiency_pct,
                },
                source="operator_bench",
            )
            log.info(
                "Stage 3b roofline: op=%s bound=%s efficiency=%.1f%%",
                op_type,
                analysis.points[0].bound if analysis.points else "unknown",
                analysis.overall_efficiency_pct,
            )
            return analysis
        except Exception as exc:
            log.warning("Roofline microbench skipped: %s", exc)
            return None

    async def _collect_hardware_counters(
        self,
        *,
        session_id: str,
        kernel_name: str,
        op_type: str,
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
            vendor = "amd" if self.cfg.gpu_type in {"MI300X", "MI325X", "MI350X", "MI355X"} else "nvidia"
            launch_cmd = (
                f"python microbench/operator_bench.py "
                f"--op {op_type} "
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

    @staticmethod
    def _scale_down_flags(
        rec_flags: Dict[str, Any],
        current_flags: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Produce a scaled-down variant of rec_flags for parameters that are
        numeric and higher than their current value.

        Rules:
        - Integer parameters (max_num_seqs, max_num_batched_tokens, block_size,
          max_model_len): scale to the midpoint between current and proposed,
          rounding to the nearest power of 2.
        - Float parameters (gpu_memory_utilization, scheduler_delay_factor):
          scale to the midpoint between current and proposed.

        Returns {} if no meaningful scaling can be produced.
        """
        _INT_PARAMS = {
            "max_num_seqs",
            "max_num_batched_tokens",
            "max_model_len",
        }
        _FLOAT_PARAMS = {
            "gpu_memory_utilization",
            "scheduler_delay_factor",
        }

        scaled: Dict[str, Any] = {}
        for k, proposed_val in rec_flags.items():
            current_val = current_flags.get(k)
            if current_val is None:
                continue
            try:
                if k in _INT_PARAMS and isinstance(proposed_val, (int, float)):
                    cur = int(current_val)
                    prop = int(proposed_val)
                    if prop <= cur:
                        continue
                    mid = (cur + prop) // 2
                    # Round to nearest power of 2 (min 1)
                    p2 = max(1, 1 << (mid - 1).bit_length() - 1)
                    # If rounding produced the same as current, try next power up
                    if p2 <= cur:
                        p2 = p2 * 2
                    if p2 < prop:
                        scaled[k] = p2
                elif k in _FLOAT_PARAMS and isinstance(proposed_val, float):
                    cur = float(current_val)
                    prop = float(proposed_val)
                    if abs(prop - cur) < 0.01:
                        continue
                    mid = round((cur + prop) / 2, 4)
                    if mid != cur and mid != prop:
                        scaled[k] = mid
            except (TypeError, ValueError):
                continue

        return scaled

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
        gate = MeasurementGate()
        critic = CriticAgent(do_client=self._do_client)
        constraints_block = ""
        try:
            constraints = await collect_session_constraints(
                self._db, session_id=session_id, recipe=None,
            )
            constraints_block = render_constraints_block(constraints)
        except Exception:
            pass

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
            trial_clean = to_vllm_flags_dict(merge_flags(current_flags, rec.vllm_flags))
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

            decision = gate.decide(
                incumbent_fitness=current_fitness,
                candidate_fitness=fitness,
                label=f"stage3:{rec.title[:40]}",
                metadata={"vllm_flags": rec.vllm_flags},
            )
            critic_verdict = await critic.review(
                gate=decision,
                primary_metric=self.cfg.optimiser.primary_metric,
                candidate_flags=rec.vllm_flags,
                incumbent_flags=current_flags,
                candidate_metrics=(config_doc or {}).get("enriched_metrics") or {},
                constraints_block=constraints_block,
                label=f"stage3:{rec.title[:40]}",
            )
            if decision.keep and critic_verdict.accept:
                base_record["status"] = "accepted"
                base_record["critic"] = critic_verdict.to_dict()
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
                base_record["critic"] = critic_verdict.to_dict()
                reject_reason = (
                    decision.reason
                    if not decision.keep
                    else f"critic:{critic_verdict.reason}"
                )
                log.info(
                    "Stage 3 rec '%s' REJECTED: %.4f → %.4f (%.4f, %.1f%%) reason=%s",
                    rec.title, current_fitness, fitness, delta, delta_pct, reject_reason,
                )
                # Smart retry: try a scaled-down version of the parameter values
                scaled_flags = self._scale_down_flags(rec.vllm_flags, current_flags)
                if scaled_flags:
                    scaled_raw = {**current_flags, **scaled_flags}
                    scaled_clean = {k: v for k, v in scaled_raw.items() if k in known_fields}
                    scaled_vllm = VLLMFlags(**scaled_clean)
                    scaled_id = await self._db.insert_config(
                        session_id=session_id,
                        fingerprint=scaled_vllm.fingerprint(),
                        flags={k: v for k, v in asdict(scaled_vllm).items() if k != "run_id"},
                        generation=-2,
                        priority=-2,
                    )
                    if scaled_id is not None:
                        log.info(
                            "Stage 3 rec '%s': trying scaled-down variant %s",
                            rec.title, scaled_flags,
                        )
                        await self._run_single(
                            session_id=session_id,
                            config_id=scaled_id,
                            context_configs=context_configs,
                        )
                        scaled_doc = await self._db.get_config_by_id(scaled_id)
                        scaled_fitness = scaled_doc.get("fitness_score", 0.0) if scaled_doc else 0.0
                        scaled_decision = gate.decide(
                            incumbent_fitness=current_fitness,
                            candidate_fitness=scaled_fitness,
                            label=f"stage3_scaled:{rec.title[:40]}",
                            metadata={"vllm_flags": scaled_flags},
                        )
                        scaled_critic = await critic.review(
                            gate=scaled_decision,
                            primary_metric=self.cfg.optimiser.primary_metric,
                            candidate_flags=scaled_flags,
                            incumbent_flags=current_flags,
                            constraints_block=constraints_block,
                            label=f"stage3_scaled:{rec.title[:40]}",
                        )
                        if scaled_decision.keep and scaled_critic.accept:
                            sd = scaled_fitness - current_fitness
                            sd_pct = (sd / current_fitness * 100) if current_fitness else 0.0
                            base_record["status"] = "accepted_scaled"
                            base_record["flags"] = scaled_flags
                            base_record["fitness_after"] = scaled_fitness
                            base_record["actual_delta"] = sd
                            base_record["actual_delta_pct"] = sd_pct
                            base_record["critic"] = scaled_critic.to_dict()
                            log.info(
                                "Stage 3 rec '%s' ACCEPTED (scaled): %.4f → %.4f (+%.4f)",
                                rec.title, current_fitness, scaled_fitness, sd,
                            )
                            accepted.append({
                                "title": f"{rec.title} (scaled)",
                                "flags": scaled_flags,
                                "fitness_before": current_fitness,
                                "fitness_after": scaled_fitness,
                                "delta": sd,
                            })
                            current_flags = {
                                k: v for k, v in asdict(scaled_vllm).items() if k != "run_id"
                            }
                            current_fitness = scaled_fitness
                        else:
                            log.info(
                                "Stage 3 rec '%s' scaled variant also rejected (%.4f)",
                                rec.title, scaled_fitness,
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
    # Operating-point sweep (post-winner)
    # ------------------------------------------------------------------

    async def _run_operating_point_sweep(
        self,
        *,
        session_id: str,
        flags: Dict[str, Any],
    ) -> SweepResult:
        """Start winner flags once, re-measure concurrency ladder, tear down."""
        from core.vllm_server import VLLMServer

        levels = list(self.cfg.benchmark.concurrency_levels or [1, 4, 16, 64, 128])
        # Extend ladder with a high point if missing (Hyperloom-style peak hunt)
        for extra in (256,):
            if extra not in levels and max(levels or [0]) < extra:
                levels.append(extra)

        node_cfg = self.cfg.nodes[0]
        gpu_alloc = GPUSlotAllocator(
            gpu_indices=node_cfg.gpu_indices,
            gpu_type=node_cfg.gpu_type,
        )
        port_alloc = PortAllocator(
            start=self.cfg.coordinator.port_pool_start,
            end=self.cfg.coordinator.port_pool_end,
        )

        known = set(VLLMFlags.__dataclass_fields__)
        clean = {k: v for k, v in (flags or {}).items() if k in known}
        try:
            vf = VLLMFlags(**clean)
        except Exception as exc:
            log.warning("Operating-point sweep skipped (bad flags): %s", exc)
            return SweepResult(skipped=True, skip_reason=f"bad_flags:{exc}")

        tp = vf.tensor_parallel_size or 1
        slot = await gpu_alloc.acquire(tp)
        if slot is None:
            return SweepResult(skipped=True, skip_reason="no_gpu_slot")
        port = await port_alloc.acquire()
        if port is None:
            await gpu_alloc.release(slot)
            return SweepResult(skipped=True, skip_reason="no_port")

        device_env = gpu_alloc.build_device_env(slot)
        server = VLLMServer(
            model_id=self.cfg.model_id,
            flags=vf,
            gpu_type=self.cfg.gpu_type,
            port=port,
            startup_timeout=self.cfg.vllm.startup_timeout_sec,
            extra_env=device_env,
            docker_image=self.cfg.vllm.docker_image,
        )
        try:
            await server.start()
            ctx = list(self.cfg.context_configs) or [(1024, 1024)]
            result = await run_operating_point_sweep(
                base_url=f"http://localhost:{port}",
                model_id=self.cfg.model_id,
                concurrency_levels=levels,
                context_configs=[ctx[0]],
                num_prompts=min(30, self.cfg.benchmark.num_prompts),
                flags=clean,
                gpu_type=self.cfg.gpu_type,
                primary_metric=self.cfg.optimiser.primary_metric,
            )
            log.info(
                "Operating-point sweep session=%s peak=%.1f @ c=%d",
                session_id[:8],
                result.peak_throughput,
                result.best_concurrency,
            )
            return result
        except Exception as exc:
            log.warning("Operating-point sweep failed: %s", exc)
            return SweepResult(skipped=True, skip_reason=str(exc)[:200])
        finally:
            try:
                await server.stop()
            except Exception:
                pass
            await gpu_alloc.release(slot)
            await port_alloc.release(port)

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

        attention_runs = await self._db.list_kernel_benchmark_runs(
            session_id, op_type="attention", limit=20
        )
        stage2_runs = await self._db.list_kernel_runs(session_id, limit=50)

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
            attention_benchmark_runs=attention_runs,
            stage2_kernel_runs=stage2_runs,
        )
        log.info("Reports written: %s", {k: str(v) for k, v in paths.items()})
