"""
core/kernel_benchmark_engine.py
-------------------------------
Backend-agnostic operator microbenchmark runner with MongoDB persistence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from core.db import Database

log = logging.getLogger("core.kernel_benchmark_engine")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MATRIX = _REPO_ROOT / "configs" / "attention_benchmark_matrix.yaml"


class KernelBenchmarkEngine:
    """
    Run isolated operator benchmarks and persist to kernel_benchmark_runs.
    """

    def __init__(
        self,
        *,
        db: Database,
        gpu_type: str,
        matrix_path: Optional[Path] = None,
    ) -> None:
        self._db = db
        self._gpu_type = gpu_type
        self._matrix_path = matrix_path or _DEFAULT_MATRIX
        self._matrix = self._load_matrix()

    def _load_matrix(self) -> Dict[str, Any]:
        with open(self._matrix_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    async def run_microbench(
        self,
        *,
        session_id: str,
        op_type: Optional[str] = None,
        backend: str = "pytorch_reference",
        kernel_name: str = "",
    ) -> Optional[str]:
        """
        Run one microbench from matrix config and insert a kernel_benchmark_run.

        Returns run_id or None on failure.
        """
        mb = self._matrix.get("microbench") or {}
        op = op_type or mb.get("op_type", "attention")
        params = dict(mb.get("params") or {})
        warmup = int(mb.get("warmup", 10))
        iters = int(mb.get("iters", 50))

        try:
            from microbench.operator_bench import OperatorBench
            from microbench.roofline import RooflineAnalyzer

            bench = OperatorBench(gpu_type=self._gpu_type)
            result = await bench.run(
                op_type=op,
                params=params,
                num_warmup=warmup,
                num_iters=iters,
            )
            if not result.success:
                log.warning("KernelBenchmarkEngine microbench failed: %s", result.error)
                return await self._db.insert_kernel_benchmark_run(
                    session_id=session_id,
                    op_type=op,
                    backend=backend,
                    gpu_type=self._gpu_type,
                    kernel_name=kernel_name or op,
                    params=params,
                    metrics={"error": result.error},
                    source="operator_bench",
                )

            duration_s = max(result.latency_us_p50, 1.0) / 1e6
            op_flops = bench._estimate_flops(op, params)
            op_bytes = bench._estimate_memory_bytes(op, params)
            roofline = RooflineAnalyzer(gpu_type=self._gpu_type).analyze(
                kernel_name=kernel_name or op,
                op_flops=op_flops,
                op_bytes=op_bytes,
                duration_s=duration_s,
            )
            bound = roofline.points[0].bound if roofline.points else result.roofline_bound

            return await self._db.insert_kernel_benchmark_run(
                session_id=session_id,
                op_type=op,
                backend=backend,
                gpu_type=self._gpu_type,
                kernel_name=kernel_name or op,
                params=params,
                metrics={
                    "latency_us_mean": result.latency_us_mean,
                    "latency_us_p50": result.latency_us_p50,
                    "latency_us_p99": result.latency_us_p99,
                    "throughput_tflops": result.tflops_achieved,
                    "memory_gbps": result.mem_bw_gbps,
                    "roofline_bound": bound,
                    "roofline_efficiency_pct": roofline.overall_efficiency_pct,
                    "compute_efficiency_pct": result.compute_efficiency_pct,
                    "mem_bw_efficiency_pct": result.mem_bw_efficiency_pct,
                },
                source="operator_bench",
            )
        except Exception as exc:
            log.warning("KernelBenchmarkEngine error: %s", exc)
            return None

    async def run_attention_microbench_suite(
        self,
        session_id: str,
        *,
        backends: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Run microbench once per logical backend label (same isolated op;
        backend label records intended vLLM comparison target).
        """
        labels = backends or ["pytorch_reference"]
        run_ids: List[str] = []
        for label in labels:
            rid = await self.run_microbench(
                session_id=session_id,
                op_type="attention",
                backend=label,
                kernel_name=f"isolated_attention_{label}",
            )
            if rid:
                run_ids.append(rid)
        return run_ids
