"""
core/fusion/diagnose.py
-----------------------
Decide whether a profiler trace is a fusion campaign candidate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from core.fusion.patterns import (
    FusionPattern,
    match_patterns,
    patterns_prompt_block,
)


_LAUNCH_BOUND = frozenset({
    "elementwise", "copy", "reduce", "cast", "rmsnorm", "layernorm",
    "rope", "add", "mul", "activation",
})


@dataclass
class FusionDiagnosis:
    is_candidate: bool
    reason: str
    launch_bound_share: float = 0.0
    category_shares: Dict[str, float] = field(default_factory=dict)
    busy_fraction_of_wall: float = 0.0
    predicted_e2e_gain: float = 0.0
    matched_patterns: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def prompt_block(self) -> str:
        if not self.matched_patterns:
            return ""
        # Rebuild lightweight match list for prompt
        from core.fusion.patterns import get_pattern
        pairs = []
        for m in self.matched_patterns:
            p = get_pattern(m["id"])
            if p:
                pairs.append((p, float(m.get("trigger_share", 0))))
        return patterns_prompt_block(pairs)


def category_shares_from_trace(
    *,
    attention_pct: float = 0.0,
    gemm_pct: float = 0.0,
    moe_pct: float = 0.0,
    communication_pct: float = 0.0,
    other_pct: float = 0.0,
    bottleneck_type: str = "",
    bottleneck_kernel: str = "",
) -> Dict[str, float]:
    """
    Map OceanTune profiler percents into fusion taxonomy shares (0–1).

    Coarse but enough to trigger residual/SwiGLU/QK patterns when kernels
    look elementwise / norm / rope dominated.
    """
    shares: Dict[str, float] = {
        "attention": max(0.0, attention_pct) / 100.0,
        "gemm": max(0.0, gemm_pct) / 100.0,
        "moe": max(0.0, moe_pct) / 100.0,
        "communication": max(0.0, communication_pct) / 100.0,
    }
    # Attribute "other" to elementwise bucket as launch-bound proxy
    other = max(0.0, other_pct) / 100.0
    shares["elementwise"] = other * 0.5
    shares["add"] = other * 0.2
    shares["mul"] = other * 0.15
    shares["rmsnorm"] = other * 0.1
    shares["activation"] = other * 0.05

    kn = (bottleneck_kernel or "").lower()
    bt = (bottleneck_type or "").lower()
    if "rms" in kn or "layernorm" in kn or "norm" in kn:
        shares["rmsnorm"] = max(shares.get("rmsnorm", 0), 0.12)
        shares["elementwise"] = max(shares.get("elementwise", 0), 0.08)
    if "silu" in kn or "gelu" in kn or "swiglu" in kn:
        shares["activation"] = max(shares.get("activation", 0), 0.08)
        shares["mul"] = max(shares.get("mul", 0), 0.05)
    if "rope" in kn or "rotary" in kn:
        shares["rope"] = max(shares.get("rope", 0), 0.10)
        shares["rmsnorm"] = max(shares.get("rmsnorm", 0), 0.05)
    if "element" in bt or "memory" in bt:
        shares["elementwise"] = max(shares.get("elementwise", 0), 0.12)
        shares["add"] = max(shares.get("add", 0), 0.08)
    return shares


def diagnose_fusion(
    category_shares: Dict[str, float],
    *,
    framework: str = "vllm",
    vendor: str = "nvidia",
    busy_fraction_of_wall: float = 0.0,
    min_launch_bound_share: float = 0.10,
) -> FusionDiagnosis:
    """Return fusion candidacy + matched patterns."""
    launch_share = sum(
        float(category_shares.get(c, 0.0) or 0.0) for c in _LAUNCH_BOUND
    )
    if launch_share < min_launch_bound_share:
        return FusionDiagnosis(
            is_candidate=False,
            reason=f"launch_bound_share_{launch_share:.3f}_below_{min_launch_bound_share}",
            launch_bound_share=launch_share,
            category_shares=dict(category_shares),
            busy_fraction_of_wall=busy_fraction_of_wall,
        )

    matches = match_patterns(
        category_shares,
        framework=framework,
        vendor=vendor,
        min_launch_bound_share=min_launch_bound_share,
    )
    # Soft note when GPU already very busy — still candidate
    reason = "fusion_candidate"
    if busy_fraction_of_wall > 0.45:
        reason = "fusion_candidate_high_busy_validate_carefully"

    predicted = min(0.15, launch_share * 0.5) if matches else 0.0
    return FusionDiagnosis(
        is_candidate=bool(matches),
        reason=reason if matches else "no_pattern_above_trigger_share",
        launch_bound_share=launch_share,
        category_shares=dict(category_shares),
        busy_fraction_of_wall=busy_fraction_of_wall,
        predicted_e2e_gain=predicted,
        matched_patterns=[
            {"id": p.id, "trigger_share": s, "env_flag": p.env_flag}
            for p, s in matches
        ],
    )
