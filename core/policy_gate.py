"""
core/policy_gate.py
-------------------
Stage-allowed action policy (Hyperloom PolicyGate analog).

Prevents agents from proposing actions outside the current stage's contract
(e.g. Stage 1 must not invent custom Triton; Stage 4 must not change TP).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set

log = logging.getLogger("core.policy_gate")


@dataclass(frozen=True)
class StagePolicy:
    stage: str
    allowed_actions: FrozenSet[str]
    allowed_flag_keys: FrozenSet[str]
    forbid_custom_kernels: bool = True
    notes: str = ""


# Flag families by stage
_STAGE1_FLAGS = frozenset({
    "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
    "gpu_memory_utilization", "max_num_seqs", "max_num_batched_tokens",
    "block_size", "kv_cache_dtype", "dtype", "quantization",
    "enforce_eager", "enable_prefix_caching", "attention_backend",
    "distributed_executor_backend", "cpu_offload_gb", "trust_remote_code",
    "enable_chunked_prefill", "max_num_chunked_tokens",
})

_STAGE2_FLAGS = frozenset({
    "kv_cache_dtype", "enable_prefix_caching", "attention_backend",
    "enable_chunked_prefill", "max_num_chunked_tokens", "num_scheduler_steps",
    "scheduler_delay_factor", "speculative_model", "num_speculative_tokens",
    "max_num_batched_tokens", "max_num_seqs",
})

_STAGE3_FLAGS = _STAGE2_FLAGS | frozenset({
    "gpu_memory_utilization", "block_size", "enforce_eager",
})

STAGE_POLICIES: Dict[str, StagePolicy] = {
    "enablement": StagePolicy(
        stage="enablement",
        allowed_actions=frozenset({"repair_oom", "repair_startup", "baseline_probe"}),
        allowed_flag_keys=_STAGE1_FLAGS,
        forbid_custom_kernels=True,
        notes="Boot repair only — no strategy search",
    ),
    "prelude": StagePolicy(
        stage="prelude",
        allowed_actions=frozenset({
            "warm_replay", "reject_seed", "profile_arm", "baseline_probe",
        }),
        allowed_flag_keys=_STAGE1_FLAGS,
        forbid_custom_kernels=True,
        notes="Warm-recipe replay + reject ledger before Stage 1",
    ),
    "stage1": StagePolicy(
        stage="stage1",
        allowed_actions=frozenset({"propose_flags", "benchmark", "analyse"}),
        allowed_flag_keys=_STAGE1_FLAGS,
        forbid_custom_kernels=True,
    ),
    "stage2": StagePolicy(
        stage="stage2",
        allowed_actions=frozenset({"propose_strategy", "benchmark", "attention_sweep"}),
        allowed_flag_keys=_STAGE2_FLAGS,
        forbid_custom_kernels=True,
    ),
    "stage3": StagePolicy(
        stage="stage3",
        allowed_actions=frozenset({
            "profile", "flag_trial", "kernel_research", "fusion_diagnose", "harness",
        }),
        allowed_flag_keys=_STAGE3_FLAGS,
        forbid_custom_kernels=True,
    ),
    "stage4": StagePolicy(
        stage="stage4",
        allowed_actions=frozenset({
            "generate_kernel", "validate_snr", "evolve", "fusion_campaign",
            "shadow_e2e", "ledger_write",
        }),
        allowed_flag_keys=frozenset(),  # serving flags frozen
        forbid_custom_kernels=False,
        notes="Serving flags frozen; kernels only",
    ),
    "sweep": StagePolicy(
        stage="sweep",
        allowed_actions=frozenset({"operating_point_sweep"}),
        allowed_flag_keys=frozenset(),
        forbid_custom_kernels=True,
    ),
    "close": StagePolicy(
        stage="close",
        allowed_actions=frozenset({"sediment_recipe", "session_breakdown", "report"}),
        allowed_flag_keys=frozenset(),
        forbid_custom_kernels=True,
    ),
}


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str
    stage: str
    action: str = ""
    rejected_keys: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PolicyGate:
    """Enforce stage contracts on actions and flag proposals."""

    def __init__(self, policies: Optional[Dict[str, StagePolicy]] = None) -> None:
        self._policies = policies or STAGE_POLICIES

    def check_action(self, stage: str, action: str) -> PolicyDecision:
        pol = self._policies.get(stage)
        if pol is None:
            return PolicyDecision(
                allowed=False, reason=f"unknown_stage:{stage}", stage=stage, action=action,
            )
        if action not in pol.allowed_actions:
            return PolicyDecision(
                allowed=False,
                reason=f"action_not_allowed_in_{stage}",
                stage=stage,
                action=action,
            )
        return PolicyDecision(allowed=True, reason="ok", stage=stage, action=action)

    def filter_flags(
        self,
        stage: str,
        flags: Dict[str, Any],
        *,
        require_custom_kernel: bool = False,
    ) -> PolicyDecision:
        pol = self._policies.get(stage)
        if pol is None:
            return PolicyDecision(
                allowed=False, reason=f"unknown_stage:{stage}", stage=stage,
            )
        if require_custom_kernel and pol.forbid_custom_kernels:
            return PolicyDecision(
                allowed=False,
                reason="custom_kernels_forbidden_in_stage",
                stage=stage,
            )
        if not pol.allowed_flag_keys:
            # Frozen serving flags — reject any mutation
            if flags:
                return PolicyDecision(
                    allowed=False,
                    reason="serving_flags_frozen",
                    stage=stage,
                    rejected_keys=sorted(flags.keys()),
                )
            return PolicyDecision(allowed=True, reason="ok_no_flags", stage=stage)

        rejected = [k for k in flags if k not in pol.allowed_flag_keys]
        if rejected:
            log.info("PolicyGate[%s]: stripping keys %s", stage, rejected)
            return PolicyDecision(
                allowed=True,
                reason="stripped_disallowed_keys",
                stage=stage,
                rejected_keys=rejected,
            )
        return PolicyDecision(allowed=True, reason="ok", stage=stage)

    def sanitize_flags(self, stage: str, flags: Dict[str, Any]) -> Dict[str, Any]:
        pol = self._policies.get(stage)
        if pol is None or not pol.allowed_flag_keys:
            return {}
        return {k: v for k, v in (flags or {}).items() if k in pol.allowed_flag_keys}
