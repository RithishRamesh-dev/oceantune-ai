"""
core/kernel_campaign.py
-----------------------
Stage 4b fusion / kernel campaign runner.

Takes FusionDiagnosis matches → optional generated kernel → SNR / micro KEEP →
MeasurementGate → optional shadow E2E → KernelLedger.

OceanTune-native; does not vendor Hyperloom fusion loop source.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from core.fusion.patterns import FusionPattern, get_pattern
from core.kernel_ledger import KernelLedger
from core.measurement_gate import MeasurementGate
from core.policy_gate import PolicyGate
from core.snr_contract import evaluate_keep, speedup_from_latencies
from core.workspace_policy import open_workspace, stage_kernel_file

log = logging.getLogger("core.kernel_campaign")


@dataclass
class CampaignTrial:
    pattern_id: str
    status: str  # skipped | blocked | micro_kept | micro_reverted | e2e_integrated | e2e_rejected
    reason: str = ""
    kernel_path: str = ""
    micro_speedup_pct: float = 0.0
    e2e_verified: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CampaignResult:
    session_id: str
    trials: List[CampaignTrial] = field(default_factory=list)
    best_trial: Optional[CampaignTrial] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "trials": [t.to_dict() for t in self.trials],
            "best_trial": self.best_trial.to_dict() if self.best_trial else None,
        }


async def run_fusion_campaign(
    *,
    session_id: str,
    model_id: str,
    gpu_type: str,
    matched_patterns: List[Dict[str, Any]],
    evolution_result: Any = None,
    incumbent_fitness: float = 0.0,
    stage3_flags: Optional[Dict[str, Any]] = None,
    run_e2e: bool = False,
    e2e_runner: Any = None,
    gpu_alloc: Any = None,
    port_alloc: Any = None,
    docker_image: str = "",
    startup_timeout_sec: int = 1200,
    primary_metric: str = "throughput",
) -> CampaignResult:
    """
    Campaign over fusion pattern hits.

    If ``evolution_result.best_kernel`` exists, attach it to the top pattern
    and optionally run shadow E2E. Patterns without a kernel are recorded as
    skipped (research-only) so Recipe KB still sees the opportunity.
    """
    gate = PolicyGate()
    act = gate.check_action("stage4", "fusion_campaign")
    if not act.allowed:
        log.warning("Campaign blocked by PolicyGate: %s", act.reason)
        return CampaignResult(session_id=session_id)

    result = CampaignResult(session_id=session_id)
    ledger = KernelLedger()
    ws = open_workspace(session_id)

    kernel_path = ""
    micro_speedup = 0.0
    op_type = ""
    if evolution_result is not None and getattr(evolution_result, "best_kernel", None):
        bk = evolution_result.best_kernel
        kernel_path = getattr(bk, "file_path", "") or ""
        micro_speedup = float(getattr(evolution_result, "best_speedup_pct", 0.0) or 0.0)
        op_type = getattr(evolution_result, "op_type", "") or ""
        if kernel_path:
            try:
                staged = stage_kernel_file(ws, __import__("pathlib").Path(kernel_path))
                kernel_path = str(staged)
            except Exception as exc:
                log.warning("Workspace stage failed: %s", exc)

    patterns = matched_patterns or []
    if not patterns and kernel_path:
        patterns = [{"id": "ad_hoc", "trigger_share": 0.0, "env_flag": "OCEANTUNE_SHADOW_KERNEL"}]

    for i, m in enumerate(patterns[:3]):
        pid = str(m.get("id") or "unknown")
        pat: Optional[FusionPattern] = get_pattern(pid)
        env_flag = str(m.get("env_flag") or (pat.env_flag if pat else "OCEANTUNE_SHADOW_KERNEL"))
        trial = CampaignTrial(pattern_id=pid, status="skipped", reason="no_kernel_artifact")

        if not kernel_path or i > 0:
            # Only first pattern gets the evolved kernel in this campaign pass
            if not kernel_path:
                trial.reason = "research_only_no_generated_kernel"
                result.trials.append(trial)
                ledger.record(
                    session_id=session_id,
                    op_type=op_type or pid,
                    model_id=model_id,
                    gpu_type=gpu_type,
                    fusion_pattern=pid,
                    decision="skipped",
                    notes=trial.reason,
                )
                continue
            trial.status = "skipped"
            trial.reason = "kernel_already_bound_to_primary_pattern"
            result.trials.append(trial)
            continue

        trial.kernel_path = kernel_path
        trial.micro_speedup_pct = micro_speedup
        # Micro KEEP via SNR contract (ratio from speedup %)
        ratio = 1.0 + (micro_speedup / 100.0) if micro_speedup else 0.0
        keep = evaluate_keep([ratio], incumbent_mean_speedup=1.0, min_speedup=1.01)
        if not keep.keep:
            trial.status = "micro_reverted"
            trial.reason = keep.reason
            result.trials.append(trial)
            ledger.record(
                session_id=session_id,
                op_type=op_type or pid,
                model_id=model_id,
                gpu_type=gpu_type,
                kernel_path=kernel_path,
                fusion_pattern=pid,
                decision="reverted",
                micro_speedup_pct=micro_speedup,
                notes=keep.reason,
            )
            continue

        trial.status = "micro_kept"
        trial.reason = keep.reason
        mg = MeasurementGate()
        # Fitness-proxy: treat micro keep as candidate bump for gate bookkeeping
        decision = mg.decide(
            incumbent_fitness=incumbent_fitness,
            candidate_fitness=incumbent_fitness * ratio if incumbent_fitness > 0 else ratio,
            label=f"campaign:{pid}",
            metadata={"pattern": pid, "micro_only": not run_e2e},
        )
        trial.extras["gate"] = decision.to_dict()

        if run_e2e and e2e_runner is not None and decision.keep:
            from core.kernel_integration import plan_integration

            plan = plan_integration(
                op_type=op_type or pid,
                kernel_path=kernel_path,
                micro_speedup_pct=micro_speedup,
                env_flag=env_flag,
            )
            plan = await e2e_runner(
                plan=plan,
                session_id=session_id,
                model_id=model_id,
                gpu_type=gpu_type,
                flags=stage3_flags or {},
                incumbent_fitness=incumbent_fitness,
                gpu_alloc=gpu_alloc,
                port_alloc=port_alloc,
                docker_image=docker_image,
                startup_timeout_sec=startup_timeout_sec,
                primary_metric=primary_metric,
                env_flag=env_flag,
            )
            trial.e2e_verified = bool(plan.e2e_verified)
            trial.status = "e2e_integrated" if plan.e2e_verified else "e2e_rejected"
            trial.reason = plan.status
            trial.extras["integration"] = plan.to_dict()
            ledger.record(
                session_id=session_id,
                op_type=op_type or pid,
                model_id=model_id,
                gpu_type=gpu_type,
                kernel_path=kernel_path,
                fusion_pattern=pid,
                decision=trial.status,
                micro_speedup_pct=micro_speedup,
                e2e_fitness_delta=float(plan.e2e_fitness or 0) - incumbent_fitness,
                notes=trial.reason,
                extras={"env_flag": env_flag},
            )
        else:
            ledger.record(
                session_id=session_id,
                op_type=op_type or pid,
                model_id=model_id,
                gpu_type=gpu_type,
                kernel_path=kernel_path,
                fusion_pattern=pid,
                decision="kept",
                micro_speedup_pct=micro_speedup,
                notes="micro_kept_pending_e2e" if not run_e2e else keep.reason,
            )

        result.trials.append(trial)
        if result.best_trial is None or trial.micro_speedup_pct > result.best_trial.micro_speedup_pct:
            if trial.status in ("micro_kept", "e2e_integrated"):
                result.best_trial = trial

    return result
