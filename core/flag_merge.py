"""
core/flag_merge.py
------------------
Canonical merge of vLLM flag dicts across pipeline stages.

Stage 1 produces base flags; Stage 2/3 apply deltas. All agents that
benchmark or profile should use merge_flags() so Stage 3 sees the same
config as production serving.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Mapping, Optional

from core.search_space import VLLMFlags

# Keys that are metadata / env-only — never passed to VLLMFlags dataclass
_SKIP_KEYS = frozenset({
    "_category",
    "run_id",
})


def merge_flags(
    base: Mapping[str, Any],
    *deltas: Mapping[str, Any],
    vllm_fields_only: bool = False,
) -> Dict[str, Any]:
    """
    Merge flag dictionaries left-to-right; later values override earlier.

    Parameters
    ----------
    base : mapping
        Stage 1 winner (or current best).
    *deltas : mapping
        Stage 2 strategy, Stage 3 trial overrides, etc.
    vllm_fields_only : bool
        If True, keep only keys that exist on VLLMFlags (for server startup).

    Returns
    -------
    dict
        Merged configuration.
    """
    merged: Dict[str, Any] = dict(base)
    known = set(VLLMFlags.__dataclass_fields__.keys())
    for delta in deltas:
        if not delta:
            continue
        for key, value in delta.items():
            if key in _SKIP_KEYS:
                continue
            if vllm_fields_only and key not in known:
                continue
            merged[key] = value
    return merged


def merged_fingerprint(
    base: Mapping[str, Any],
    *deltas: Mapping[str, Any],
) -> str:
    """Deterministic fingerprint for merged flags (same scheme as VLLMFlags)."""
    flags = merge_flags(base, *deltas, vllm_fields_only=True)
    try:
        vf = VLLMFlags(**{k: v for k, v in flags.items() if k in VLLMFlags.__dataclass_fields__})
        return vf.fingerprint()
    except (TypeError, ValueError):
        payload = json.dumps(sorted(flags.items()), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def to_vllm_flags_dict(merged: Mapping[str, Any]) -> Dict[str, Any]:
    """Filter merged dict to fields valid for VLLMFlags construction."""
    known = set(VLLMFlags.__dataclass_fields__.keys())
    return {k: v for k, v in merged.items() if k in known}


def infer_microbench_op(
    *,
    bottleneck_kernel: str = "",
    bottleneck_type: str = "",
    attention_pct: float = 0.0,
    gemm_pct: float = 0.0,
    moe_pct: float = 0.0,
) -> str:
    """
    Map profiler output to operator_bench --op value.

    Returns one of: attention | gemm | rmsnorm | rope | moe
    """
    name = (bottleneck_kernel or "").lower()
    if any(k in name for k in (
        "flash_attn", "flash_fwd", "flash_bwd", "sdpa", "paged_attn",
        "self_attn", "attention",
    )):
        return "attention"
    if any(k in name for k in (
        "gemm", "cutlass", "cublas", "sgemm", "hgemm", "matmul", "linear",
    )):
        return "gemm"
    if any(k in name for k in ("rmsnorm", "layernorm", "norm")):
        return "rmsnorm"
    if any(k in name for k in ("rope", "rotary")):
        return "rope"
    if any(k in name for k in ("moe", "expert", "dispatch", "grouped_gemm")):
        return "moe"
    if any(k in name for k in ("nccl", "all_reduce", "rccl")):
        return "gemm"  # comm-heavy; gemm microbench as proxy for tensor ops

    btype = (bottleneck_type or "").lower()
    if "attention" in btype:
        return "attention"
    if "moe" in btype:
        return "moe"
    if "gemm" in btype or "compute" in btype:
        return "gemm"

    # Fall back to dominant trace category
    pcts = {
        "attention": attention_pct,
        "gemm": gemm_pct,
        "moe": moe_pct,
    }
    if max(pcts.values()) > 0:
        return max(pcts, key=pcts.get)  # type: ignore[arg-type]
    return "attention"
