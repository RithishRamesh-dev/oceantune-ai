"""
core/framework_backend.py
-------------------------
Multi-framework serving backend abstraction (vLLM | SGLang).

Hyperloom optimizes across frameworks; OceanTune historically was vLLM-only.
This module gives Stage 1–2 a pluggable launch/benchmark surface without
vendoring Hyperloom.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

log = logging.getLogger("core.framework_backend")

SUPPORTED_FRAMEWORKS = ("vllm", "sglang")


@dataclass
class BackendLaunchSpec:
    """How to start a serving process for one trial."""

    framework: str
    model_id: str
    port: int
    docker_image: str
    cli_args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    extra_docker_args: List[str] = field(default_factory=list)
    health_path: str = "/health"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "framework": self.framework,
            "model_id": self.model_id,
            "port": self.port,
            "docker_image": self.docker_image,
            "cli_args": list(self.cli_args),
            "env": dict(self.env),
            "extra_docker_args": list(self.extra_docker_args),
            "health_path": self.health_path,
            "notes": self.notes,
        }


class FrameworkBackend(ABC):
    """Abstract serving backend used by Executor / StrategyOptimizer."""

    name: str = "base"

    @abstractmethod
    def build_launch_spec(
        self,
        *,
        model_id: str,
        flags: Dict[str, Any],
        port: int,
        gpu_type: str,
        docker_image: str = "",
        extra_env: Optional[Dict[str, str]] = None,
    ) -> BackendLaunchSpec:
        ...

    @abstractmethod
    def normalize_flags(self, flags: Dict[str, Any]) -> Dict[str, Any]:
        """Map OceanTune flag dict into backend-native keys."""
        ...

    def supports_flag(self, name: str) -> bool:
        return True

    def openai_compatible(self) -> bool:
        return True


class VLLMBackend(FrameworkBackend):
    name = "vllm"

    def normalize_flags(self, flags: Dict[str, Any]) -> Dict[str, Any]:
        return dict(flags or {})

    def build_launch_spec(
        self,
        *,
        model_id: str,
        flags: Dict[str, Any],
        port: int,
        gpu_type: str,
        docker_image: str = "",
        extra_env: Optional[Dict[str, str]] = None,
    ) -> BackendLaunchSpec:
        from core.search_space import VLLMFlags

        known = set(VLLMFlags.__dataclass_fields__)
        clean = {k: v for k, v in (flags or {}).items() if k in known}
        vf = VLLMFlags(**clean)
        args = vf.to_vllm_args(model_id=model_id, gpu_type=gpu_type)
        image = docker_image or ""
        if not image:
            try:
                from core.vllm_server import _load_gpu_profile
                image = str((_load_gpu_profile(gpu_type) or {}).get("docker_image") or "")
            except Exception:
                image = "vllm/vllm-openai:latest"
        return BackendLaunchSpec(
            framework="vllm",
            model_id=model_id,
            port=port,
            docker_image=image,
            cli_args=list(args),
            env=dict(extra_env or {}),
            health_path="/health",
            notes="OceanTune primary backend",
        )


# SGLang flag aliases (OceanTune flag → sglang CLI / env)
# Expanded remap for deeper SGLang search-space parity with OceanTune Stage 1–2.
_SGLANG_FLAG_MAP = {
    "tensor_parallel_size": "--tp-size",
    "pipeline_parallel_size": "--pp-size",
    "data_parallel_size": "--dp-size",
    "gpu_memory_utilization": "--mem-fraction-static",
    "max_num_seqs": "--max-running-requests",
    "max_num_batched_tokens": "--chunked-prefill-size",
    "dtype": "--dtype",
    "kv_cache_dtype": "--kv-cache-dtype",
    "enable_prefix_caching": "--enable-prefix-caching",
    "attention_backend": "--attention-backend",
    "enable_chunked_prefill": "--chunked-prefill-size",  # bool → set size if present
    "block_size": "--page-size",
    "enforce_eager": "--disable-cuda-graph",
    "schedule_conservativeness": "--schedule-conservativeness",
    "speculative_model": "--speculative-draft-model-path",
    "num_speculative_tokens": "--speculative-num-draft-tokens",
}


def remap_flags_for_sglang(flags: Dict[str, Any]) -> Dict[str, Any]:
    """Public helper: OceanTune flags → SGLang-supported subset."""
    return SGLangBackend().normalize_flags(flags)


class SGLangBackend(FrameworkBackend):
    """
    SGLang adapter — OpenAI-compatible HTTP surface for BenchmarkEngine.

    Launch uses `python -m sglang.launch_server` inside a Docker image.
    Not all vLLM flags map 1:1; unsupported keys are dropped with a log.
    """

    name = "sglang"

    DEFAULT_IMAGE = "lmsysorg/sglang:latest"

    def normalize_flags(self, flags: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in (flags or {}).items():
            if k in _SGLANG_FLAG_MAP or k in ("trust_remote_code", "quantization"):
                out[k] = v
            else:
                log.debug("SGLangBackend: dropping unsupported flag %s", k)
        # Chunked prefill: OceanTune bool → sglang size from max_num_batched_tokens
        if out.get("enable_chunked_prefill") is True and "max_num_batched_tokens" in (flags or {}):
            out["max_num_batched_tokens"] = flags["max_num_batched_tokens"]
        elif out.get("enable_chunked_prefill") is False:
            out.pop("max_num_batched_tokens", None)
        return out

    def supports_flag(self, name: str) -> bool:
        return name in _SGLANG_FLAG_MAP or name in ("trust_remote_code", "quantization")

    def build_launch_spec(
        self,
        *,
        model_id: str,
        flags: Dict[str, Any],
        port: int,
        gpu_type: str,
        docker_image: str = "",
        extra_env: Optional[Dict[str, str]] = None,
    ) -> BackendLaunchSpec:
        norm = self.normalize_flags(flags)
        args: List[str] = [
            "-m", "sglang.launch_server",
            "--model-path", model_id,
            "--host", "0.0.0.0",
            "--port", str(port),
        ]
        seen_cli: set = set()
        for key, cli in _SGLANG_FLAG_MAP.items():
            if key not in norm or norm[key] is None:
                continue
            if key == "enable_chunked_prefill":
                # Handled via max_num_batched_tokens / chunked-prefill-size
                continue
            if cli in seen_cli:
                continue
            val = norm[key]
            if isinstance(val, bool):
                if val:
                    args.append(cli)
                    seen_cli.add(cli)
                # enforce_eager True → --disable-cuda-graph (already bool True path)
            else:
                args.extend([cli, str(val)])
                seen_cli.add(cli)
        if norm.get("trust_remote_code"):
            args.append("--trust-remote-code")
        if norm.get("quantization"):
            args.extend(["--quantization", str(norm["quantization"])])

        image = docker_image or self.DEFAULT_IMAGE
        env = dict(extra_env or {})
        env.setdefault("OCEANTUNE_FRAMEWORK", "sglang")
        return BackendLaunchSpec(
            framework="sglang",
            model_id=model_id,
            port=port,
            docker_image=image,
            cli_args=args,
            env=env,
            health_path="/health",
            notes="SGLang OpenAI-compatible adapter; remapped OceanTune Stage 1–2 flags",
        )


_REGISTRY: Dict[str, Type[FrameworkBackend]] = {
    "vllm": VLLMBackend,
    "sglang": SGLangBackend,
}


def get_framework_backend(name: str) -> FrameworkBackend:
    key = (name or "vllm").strip().lower()
    cls = _REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown framework backend '{name}'. Choose one of: {sorted(_REGISTRY)}"
        )
    return cls()


def list_frameworks() -> List[str]:
    return sorted(_REGISTRY.keys())
