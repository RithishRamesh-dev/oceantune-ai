"""
core/quantization_schemes.py
----------------------------
Quantization scheme registry + GPU gating (OceanTune-native).

Inspired by Hyperloom's quantization_schemes phase but focused on vLLM /
SGLang flags OceanTune already understands — no Quark runtime dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set


SUPPORTED_SCHEMES = (
    "none",
    "fp8",
    "fp8_kv",
    "awq",
    "gptq",
    "nvfp4",
    "bitsandbytes",
)


# GPU families that can run a scheme (empty set = all)
_SCHEME_GPU_ALLOW: Dict[str, Set[str]] = {
    "none": set(),
    "fp8": {"H100", "H200", "B300", "MI300X", "MI325X", "MI350X", "MI355X"},
    "fp8_kv": {"H100", "H200", "B300", "MI300X", "MI325X", "MI350X", "MI355X"},
    "awq": set(),
    "gptq": set(),
    "nvfp4": {"B300"},
    "bitsandbytes": set(),
}


@dataclass
class QuantizationConfig:
    global_scheme: str = "none"
    kv_cache_dtype: Optional[str] = None
    exclude_layers: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SchemeDecision:
    allowed: bool
    reason: str
    flags_delta: Dict[str, Any] = field(default_factory=dict)
    config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_scheme(scheme: str) -> str:
    key = (scheme or "none").strip().lower()
    if key not in SUPPORTED_SCHEMES:
        raise ValueError(
            f"unknown quantization scheme {scheme!r}; choose one of {list(SUPPORTED_SCHEMES)}"
        )
    return key


def scheme_allowed_on_gpu(scheme: str, gpu_type: str) -> bool:
    key = validate_scheme(scheme)
    allow = _SCHEME_GPU_ALLOW.get(key, set())
    if not allow:
        return True
    return (gpu_type or "").upper() in allow


def resolve_quantization(
    *,
    scheme: str,
    gpu_type: str,
    exclude_layers: Optional[List[str]] = None,
) -> SchemeDecision:
    """
    Map a named scheme to OceanTune flag deltas, gated by GPU.
    """
    try:
        key = validate_scheme(scheme)
    except ValueError as exc:
        return SchemeDecision(allowed=False, reason=str(exc))

    if not scheme_allowed_on_gpu(key, gpu_type):
        return SchemeDecision(
            allowed=False,
            reason=f"scheme_{key}_not_supported_on_{gpu_type}",
        )

    cfg = QuantizationConfig(
        global_scheme=key,
        exclude_layers=list(exclude_layers or []),
    )
    delta: Dict[str, Any] = {}

    if key == "none":
        cfg.notes = "No weight quantization"
    elif key == "fp8":
        delta["quantization"] = "fp8"
        cfg.notes = "FP8 weights via vLLM quantization=fp8"
    elif key == "fp8_kv":
        delta["kv_cache_dtype"] = "fp8"
        cfg.kv_cache_dtype = "fp8"
        cfg.notes = "FP8 KV cache only"
    elif key == "awq":
        delta["quantization"] = "awq"
        cfg.notes = "AWQ checkpoint expected"
    elif key == "gptq":
        delta["quantization"] = "gptq"
        cfg.notes = "GPTQ checkpoint expected"
    elif key == "nvfp4":
        delta["quantization"] = "modelopt_fp4"
        cfg.notes = "Blackwell NVFP4 / modelopt path"
    elif key == "bitsandbytes":
        delta["quantization"] = "bitsandbytes"
        cfg.notes = "bitsandbytes load path"

    return SchemeDecision(
        allowed=True,
        reason="ok",
        flags_delta=delta,
        config=cfg.to_dict(),
    )


def build_quantization_prompt(cfg: QuantizationConfig) -> str:
    """Natural-language paragraph for Planner / Strategy prompts."""
    sentences = [f"Apply {cfg.global_scheme} as the global quantization scheme."]
    if cfg.kv_cache_dtype:
        sentences.append(f"Set kv_cache_dtype={cfg.kv_cache_dtype}.")
    if cfg.exclude_layers:
        sentences.append(
            "Exclude layers from quantization: " + ", ".join(cfg.exclude_layers) + "."
        )
    if cfg.notes:
        sentences.append(cfg.notes)
    return " ".join(sentences)


def schemes_for_gpu(gpu_type: str) -> List[str]:
    return [s for s in SUPPORTED_SCHEMES if scheme_allowed_on_gpu(s, gpu_type)]
