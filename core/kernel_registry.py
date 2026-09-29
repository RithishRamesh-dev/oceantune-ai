"""
core/kernel_registry.py
-----------------------
Load attention (and future) kernel metadata from YAML and filter by GPU/model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ATTENTION_REGISTRY = _REPO_ROOT / "configs" / "kernel_registry" / "attention.yaml"

_AMD_GPU_TYPES = frozenset({"MI300X", "MI325X", "MI350X", "MI355X"})
_NVIDIA_GPU_TYPES = frozenset({"H100", "H200", "B300", "A100", "A100_80G", "A6000"})


def _vendor_for_gpu(gpu_type: str) -> str:
    if gpu_type in _AMD_GPU_TYPES:
        return "amd"
    return "nvidia"


def model_has_gqa(model_meta: Optional[Dict[str, Any]], model_id: str = "") -> bool:
    """Heuristic GQA detection from models.yaml metadata or model id."""
    if not model_meta:
        model_meta = {}
    arch = str(model_meta.get("architecture", "")).upper()
    if "GQA" in arch or "MLA" in arch:
        return True
    if model_meta.get("moe") or model_meta.get("mla"):
        return True
    mid = model_id.lower()
    # Common open models with GQA (7B+ recent families)
    for token in ("qwen", "mistral", "llama", "gemma", "deepseek", "minimax", "kimi"):
        if token in mid:
            return True
    return False


class KernelRegistry:
    """In-memory registry backed by configs/kernel_registry/*.yaml."""

    def __init__(self, attention_path: Optional[Path] = None) -> None:
        path = attention_path or _ATTENTION_REGISTRY
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        self._attention: Dict[str, Dict[str, Any]] = raw.get("implementations") or {}

    def list_attention_backends(self) -> List[str]:
        return list(self._attention.keys())

    def get(self, backend: str) -> Optional[Dict[str, Any]]:
        return self._attention.get(backend)

    def filter_attention_backends(
        self,
        *,
        gpu_type: str,
        model_meta: Optional[Dict[str, Any]] = None,
        model_id: str = "",
        require_gqa: bool = False,
    ) -> List[str]:
        """
        Return vLLM attention_backend values valid for this GPU and model.
        """
        vendor = _vendor_for_gpu(gpu_type)
        gqa = model_has_gqa(model_meta, model_id)
        out: List[str] = []
        for name, spec in self._attention.items():
            vendors = spec.get("vendors") or ["all"]
            if vendor not in vendors and "all" not in vendors:
                continue
            if spec.get("requires_fa3_stack") and gpu_type not in ("H100", "H200", "B300"):
                continue
            if require_gqa and gqa and not spec.get("supports_gqa", True):
                continue
            # Skip debug-only unless explicitly requested
            if name == "TORCH_SDPA":
                continue
            if name == "XFORMERS" and gpu_type in ("H200", "B300", "H100"):
                continue
            out.append(spec.get("vllm_flag_value", name))
        # Deduplicate while preserving order
        seen: set = set()
        deduped: List[str] = []
        for b in out:
            if b not in seen:
                seen.add(b)
                deduped.append(b)
        return deduped

    def metadata_documents(self) -> List[Dict[str, Any]]:
        """MongoDB-ready kernel_metadata documents."""
        docs: List[Dict[str, Any]] = []
        for backend, spec in self._attention.items():
            docs.append({
                "op_type": "attention",
                "backend": backend,
                "implementation_id": spec.get("implementation_id", backend),
                "display_name": spec.get("display_name", backend),
                "vllm_flag_value": spec.get("vllm_flag_value", backend),
                "vendors": spec.get("vendors", []),
                "gpu_architectures": spec.get("gpu_architectures", []),
                "supports_fp8_kv": spec.get("supports_fp8_kv", False),
                "supports_gqa": spec.get("supports_gqa", False),
                "supports_mla": spec.get("supports_mla", False),
                "supports_paged_kv": spec.get("supports_paged_kv", False),
                "min_vllm_version": spec.get("min_vllm_version", ""),
                "notes": spec.get("notes", ""),
            })
        return docs


class CapabilityDetector:
    """Filter strategy proposals and benchmark matrices by hardware/model."""

    def __init__(self, registry: Optional[KernelRegistry] = None) -> None:
        self._registry = registry or KernelRegistry()

    def supported_attention_backends(
        self,
        gpu_type: str,
        model_meta: Optional[Dict[str, Any]] = None,
        model_id: str = "",
    ) -> List[str]:
        return self._registry.filter_attention_backends(
            gpu_type=gpu_type,
            model_meta=model_meta,
            model_id=model_id,
        )

    def flashinfer_recommended(
        self,
        *,
        gpu_type: str,
        model_meta: Optional[Dict[str, Any]] = None,
        model_id: str = "",
        gpu_memory_utilization: float = 0.0,
    ) -> bool:
        """True when FLASHINFER is supported and GQA + high VRAM pressure."""
        backends = self.supported_attention_backends(
            gpu_type, model_meta, model_id
        )
        if "FLASHINFER" not in backends:
            return False
        return model_has_gqa(model_meta, model_id) and gpu_memory_utilization >= 0.85

    def filter_strategy_config(
        self,
        strategy_config: Dict[str, Any],
        *,
        gpu_type: str,
        model_meta: Optional[Dict[str, Any]] = None,
        model_id: str = "",
    ) -> Dict[str, Any]:
        """Drop unsupported keys (e.g. FLASHINFER on AMD)."""
        if not strategy_config:
            return strategy_config
        allowed = set(
            self.supported_attention_backends(gpu_type, model_meta, model_id)
        )
        filtered = dict(strategy_config)
        backend = filtered.get("attention_backend")
        if backend is not None and backend not in allowed:
            del filtered["attention_backend"]
        return filtered
