"""
core/kernel_benchmark.py
------------------------
Data models for isolated kernel / operator benchmarks (Phase 0 schema stub).

Persisted in MongoDB collection ``kernel_benchmark_runs``. Full benchmark
implementations are added in later research phases.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class KernelBenchmarkMetrics:
    """Per-run metrics from an isolated operator benchmark."""
    latency_us_mean: float = 0.0
    latency_us_p50: float = 0.0
    latency_us_p99: float = 0.0
    throughput_tflops: float = 0.0
    memory_gbps: float = 0.0
    roofline_bound: str = "unknown"  # compute | memory | unknown
    roofline_efficiency_pct: float = 0.0
    compute_efficiency_pct: float = 0.0
    mem_bw_efficiency_pct: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KernelBenchmarkRun:
    """
    One isolated kernel benchmark document (maps to kernel_benchmark_runs).
    """
    session_id: str
    op_type: str  # attention | kv | decode | moe | gemm | rmsnorm | rope
    backend: str = "pytorch"  # pytorch | triton | flash_attn | flashinfer | custom
    gpu_type: str = ""
    kernel_name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: KernelBenchmarkMetrics = field(default_factory=KernelBenchmarkMetrics)
    source: str = "operator_bench"  # operator_bench | ncu | manual
    created_at: datetime = field(default_factory=_utc_now)

    def to_mongo_doc(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "op_type": self.op_type,
            "backend": self.backend,
            "gpu_type": self.gpu_type,
            "kernel_name": self.kernel_name,
            "params": self.params,
            "metrics": self.metrics.to_dict(),
            "source": self.source,
            "created_at": self.created_at,
        }

    @classmethod
    def from_mongo_doc(cls, doc: Dict[str, Any]) -> "KernelBenchmarkRun":
        metrics_raw = doc.get("metrics") or {}
        metrics = KernelBenchmarkMetrics(
            latency_us_mean=float(metrics_raw.get("latency_us_mean", 0)),
            latency_us_p50=float(metrics_raw.get("latency_us_p50", 0)),
            latency_us_p99=float(metrics_raw.get("latency_us_p99", 0)),
            throughput_tflops=float(metrics_raw.get("throughput_tflops", 0)),
            memory_gbps=float(metrics_raw.get("memory_gbps", 0)),
            roofline_bound=str(metrics_raw.get("roofline_bound", "unknown")),
            roofline_efficiency_pct=float(
                metrics_raw.get("roofline_efficiency_pct", 0)
            ),
            error=metrics_raw.get("error"),
        )
        return cls(
            session_id=str(doc.get("session_id", "")),
            op_type=str(doc.get("op_type", "")),
            backend=str(doc.get("backend", "pytorch")),
            gpu_type=str(doc.get("gpu_type", "")),
            kernel_name=str(doc.get("kernel_name", "")),
            params=dict(doc.get("params") or {}),
            metrics=metrics,
            source=str(doc.get("source", "operator_bench")),
            created_at=doc.get("created_at") or _utc_now(),
        )
