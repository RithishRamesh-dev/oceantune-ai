"""
core/enablement.py
------------------
Enablement / boot-repair ladder (Hyperloom ENABLEMENT analog).

When the Stage 1 baseline (or first warm-start) fails with OOM / startup
timeout / CUDA errors, try a deterministic sequence of safer configs before
abandoning the session.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger("core.enablement")


@dataclass
class RepairStep:
    name: str
    flags_delta: Dict[str, Any]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnablementResult:
    success: bool
    steps_tried: List[Dict[str, Any]] = field(default_factory=list)
    winning_flags: Dict[str, Any] = field(default_factory=dict)
    final_error: str = ""
    stop_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def classify_boot_failure(error: str) -> str:
    e = (error or "").lower()
    if any(x in e for x in ("out of memory", "oom", "cuda oom", "hip out of memory")):
        return "oom"
    if any(x in e for x in ("startup", "health", "timeout", "failed to start")):
        return "startup_timeout"
    if any(x in e for x in ("nccl", "rccl", "distributed")):
        return "distributed"
    if any(x in e for x in ("flashinfer", "attention", "backend")):
        return "attention_backend"
    if any(x in e for x in ("mla", "block_size")):
        return "mla"
    return "unknown"


def repair_ladder(
    *,
    base_flags: Dict[str, Any],
    error: str,
    moe: bool = False,
    mla: bool = False,
) -> List[RepairStep]:
    """
    Ordered repair proposals. Caller benchmarks each until one boots.
    """
    kind = classify_boot_failure(error)
    steps: List[RepairStep] = []
    util = float(base_flags.get("gpu_memory_utilization") or 0.90)

    if kind == "oom" or kind == "unknown":
        steps.append(RepairStep(
            name="lower_util_0_85",
            flags_delta={"gpu_memory_utilization": min(util, 0.85)},
            reason="Reduce VRAM reservation after OOM",
        ))
        steps.append(RepairStep(
            name="fp8_kv",
            flags_delta={"kv_cache_dtype": "fp8", "gpu_memory_utilization": min(util, 0.85)},
            reason="FP8 KV frees capacity for weights + sequences",
        ))
        steps.append(RepairStep(
            name="lower_util_0_75_max_seqs",
            flags_delta={
                "gpu_memory_utilization": 0.75,
                "max_num_seqs": min(int(base_flags.get("max_num_seqs") or 256), 64),
                "kv_cache_dtype": "fp8",
            },
            reason="Aggressive memory retreat for large MoE",
        ))
        if moe:
            steps.append(RepairStep(
                name="moe_eager_safe",
                flags_delta={
                    "enforce_eager": True,
                    "gpu_memory_utilization": 0.80,
                    "kv_cache_dtype": "fp8",
                },
                reason="Disable CUDA graphs during MoE bring-up",
            ))

    if kind in ("startup_timeout", "unknown"):
        steps.append(RepairStep(
            name="eager_mode",
            flags_delta={"enforce_eager": True},
            reason="CUDA graph capture can hang on first boot",
        ))
        steps.append(RepairStep(
            name="drop_attention_to_flash",
            flags_delta={"attention_backend": "FLASH_ATTN"},
            reason="Revert experimental attention backend",
        ))

    if kind == "attention_backend":
        steps.append(RepairStep(
            name="flash_attn_only",
            flags_delta={"attention_backend": "FLASH_ATTN"},
            reason="FlashInfer/other backend unavailable",
        ))
        steps.append(RepairStep(
            name="rocm_flash",
            flags_delta={"attention_backend": "ROCM_FLASH"},
            reason="AMD default attention path",
        ))

    if kind == "distributed" or int(base_flags.get("tensor_parallel_size") or 1) > 1:
        steps.append(RepairStep(
            name="tp_to_1",
            flags_delta={"tensor_parallel_size": 1, "pipeline_parallel_size": 1},
            reason="Isolate single-GPU boot before TP",
        ))

    if mla or kind == "mla":
        steps.append(RepairStep(
            name="mla_block_size_1",
            flags_delta={"block_size": 1},
            reason="MLA requires block_size=1",
        ))

    # Deduplicate by name
    seen = set()
    out: List[RepairStep] = []
    for s in steps:
        if s.name in seen:
            continue
        seen.add(s.name)
        out.append(s)
    return out


def apply_repair(base: Dict[str, Any], step: RepairStep) -> Dict[str, Any]:
    merged = dict(base or {})
    merged.update(step.flags_delta)
    return merged
