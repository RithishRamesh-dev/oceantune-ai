"""
core/fusion/patterns.py
-----------------------
Fusion pattern library for OceanTune Stage 4 campaigns.

Inspired by Hyperloom fusion pattern IDs / triggers — reimplemented without
vendoring Hyperloom source. Patterns guide KernelResearch / Generation agents
toward fuse sites that are launch-bound in profiler traces.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class FusionPattern:
    id: str
    trigger_categories: Tuple[str, ...]
    min_trigger_share: float
    description: str
    source_hints: Tuple[str, ...]
    fusion_math: str
    eager_reference_hint: str
    env_flag: str
    frameworks: Tuple[str, ...] = ("vllm",)
    fused_markers: Tuple[str, ...] = ()
    vendors: Tuple[str, ...] = ("nvidia", "amd")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


FUSION_PATTERNS: Tuple[FusionPattern, ...] = (
    FusionPattern(
        id="residual_add_rmsnorm",
        trigger_categories=("add", "rmsnorm", "elementwise"),
        min_trigger_share=0.10,
        description="Fuse residual add + RMSNorm into one kernel (common LLM block).",
        source_hints=("residual", "rms_norm", "RMSNorm", "fused_add_rms"),
        fusion_math="y = rmsnorm(x + residual); write both residual stream and normed output",
        eager_reference_hint="torch: out = rms_norm(x + residual, weight)",
        env_flag="OCEANTUNE_FUSED_RESIDUAL",
        fused_markers=("fused_add_rms", "FusedAddRMSNorm"),
    ),
    FusionPattern(
        id="swiglu_silu_mul",
        trigger_categories=("activation", "mul", "elementwise"),
        min_trigger_share=0.03,
        description="Fuse SiLU(gate) * up projection (SwiGLU MLP).",
        source_hints=("silu", "SwiGLU", "gate_up", "silu_and_mul"),
        fusion_math="out = silu(gate) * up",
        eager_reference_hint="torch.nn.functional.silu(gate) * up",
        env_flag="OCEANTUNE_FUSED_SILU",
        fused_markers=("SiluAndMul", "fused_silu_mul"),
    ),
    FusionPattern(
        id="scaled_residual_add_rmsnorm",
        trigger_categories=("add", "mul", "rmsnorm"),
        min_trigger_share=0.08,
        description="Scaled residual + RMSNorm (μP / Granite-style).",
        source_hints=("residual_scale", "granite", "mup"),
        fusion_math="y = rmsnorm(x + scale * residual)",
        eager_reference_hint="rms_norm(x + scale * residual, weight)",
        env_flag="OCEANTUNE_FUSED_SCALED_RESIDUAL",
        fused_markers=("scaled_residual_rms",),
    ),
    FusionPattern(
        id="qk_norm_rope",
        trigger_categories=("rmsnorm", "rope", "attention"),
        min_trigger_share=0.12,
        description="Fuse Q/K RMSNorm with RoPE application.",
        source_hints=("qk_norm", "apply_rotary", "rope", "q_norm"),
        fusion_math="q',k' = rope(rmsnorm(q), rmsnorm(k))",
        eager_reference_hint="apply_rope(rms_norm(q), rms_norm(k), cos, sin)",
        env_flag="OCEANTUNE_FUSED_QK",
        fused_markers=("fused_qk_rope", "QKNormRoPE"),
    ),
    FusionPattern(
        id="hybrid_scale_combine",
        trigger_categories=("mul", "add", "elementwise"),
        min_trigger_share=0.06,
        description="Fuse elementwise scale + combine (hybrid attention paths).",
        source_hints=("scale_combine", "prescale", "hybrid"),
        fusion_math="out = a * scale_a + b * scale_b",
        eager_reference_hint="a * sa + b * sb",
        env_flag="OCEANTUNE_FUSED_SCALES",
        fused_markers=("scale_combine",),
    ),
    FusionPattern(
        id="dual_affine_scaling",
        trigger_categories=("add", "mul", "elementwise"),
        min_trigger_share=0.06,
        description="Dual affine scale of residual streams before norm.",
        source_hints=("affine", "residual_scale", "layer_scale"),
        fusion_math="out = scale1 * (x + residual) + shift",
        eager_reference_hint="scale * (x + residual) + shift",
        env_flag="OCEANTUNE_FUSED_RESIDUAL_SCALE",
        fused_markers=("dual_affine",),
    ),
)


_LAUNCH_BOUND = frozenset({
    "elementwise", "copy", "reduce", "cast", "rmsnorm", "layernorm",
    "rope", "add", "mul", "activation",
})


def get_pattern(pattern_id: str) -> Optional[FusionPattern]:
    for p in FUSION_PATTERNS:
        if p.id == pattern_id:
            return p
    return None


def match_patterns(
    category_shares: Dict[str, float],
    *,
    framework: str = "vllm",
    vendor: str = "nvidia",
    min_launch_bound_share: float = 0.10,
) -> List[Tuple[FusionPattern, float]]:
    """
    Rank fusion patterns by trigger share when launch-bound work is significant.

    Returns list of (pattern, trigger_share) sorted by share descending.
    """
    launch_share = sum(
        float(category_shares.get(c, 0.0) or 0.0) for c in _LAUNCH_BOUND
    )
    if launch_share < min_launch_bound_share:
        return []

    hits: List[Tuple[FusionPattern, float]] = []
    for pat in FUSION_PATTERNS:
        if framework not in pat.frameworks:
            continue
        if vendor not in pat.vendors and "all" not in pat.vendors:
            continue
        share = sum(
            float(category_shares.get(c, 0.0) or 0.0)
            for c in pat.trigger_categories
        )
        if share >= pat.min_trigger_share:
            hits.append((pat, share))
    hits.sort(key=lambda x: x[1], reverse=True)
    return hits


def patterns_prompt_block(
    matches: List[Tuple[FusionPattern, float]],
    *,
    max_patterns: int = 3,
) -> str:
    """Render matched patterns for KernelResearch / Generation prompts."""
    if not matches:
        return ""
    lines = ["=== Fusion campaign candidates (launch-bound) ==="]
    for pat, share in matches[:max_patterns]:
        lines.append(
            f"- {pat.id} (trigger_share={share:.2f}, env={pat.env_flag}): "
            f"{pat.description}"
        )
        lines.append(f"  math: {pat.fusion_math}")
        lines.append(f"  locate: {', '.join(pat.source_hints[:4])}")
    return "\n".join(lines)
