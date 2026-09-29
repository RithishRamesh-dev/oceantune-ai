"""
core/kernel_harness.py
----------------------
YAML-driven isolated kernel harness cases (prefill / decode / GQA / MLA proxy).

Closes the Hyperloom-inspired gap: OperatorBench previously used one ad-hoc
shape. Harness cases feed ``kernel_benchmark_runs`` so Stage 3/4 and Recipe KB
can attribute bottlenecks by phase.

No Hyperloom source is vendored.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

log = logging.getLogger("core.kernel_harness")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CASES = _REPO_ROOT / "configs" / "kernel_harness_cases.yaml"


@dataclass
class HarnessCase:
    id: str
    op_type: str
    phase: str = "both"  # prefill | decode | both
    tags: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    warmup: int = 5
    iters: int = 30

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HarnessCaseResult:
    case_id: str
    op_type: str
    phase: str
    success: bool
    run_id: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_harness_cases(
    path: Optional[Path] = None,
    *,
    case_ids: Optional[Sequence[str]] = None,
    tags: Optional[Sequence[str]] = None,
    phase: Optional[str] = None,
) -> List[HarnessCase]:
    """Load and optionally filter harness cases from YAML."""
    p = path or _DEFAULT_CASES
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    defaults = raw.get("defaults") or {}
    default_warmup = int(defaults.get("warmup", 5))
    default_iters = int(defaults.get("iters", 30))

    cases: List[HarnessCase] = []
    for row in raw.get("cases") or []:
        c = HarnessCase(
            id=str(row["id"]),
            op_type=str(row["op_type"]),
            phase=str(row.get("phase") or "both"),
            tags=list(row.get("tags") or []),
            params=dict(row.get("params") or {}),
            notes=str(row.get("notes") or ""),
            warmup=int(row.get("warmup", default_warmup)),
            iters=int(row.get("iters", default_iters)),
        )
        cases.append(c)

    if case_ids is None and path is None:
        # Prefer stage3_default when no explicit filter
        default_ids = raw.get("stage3_default")
        if default_ids:
            case_ids = list(default_ids)

    if case_ids is not None:
        allow = set(case_ids)
        cases = [c for c in cases if c.id in allow]
    if tags:
        tagset = set(tags)
        cases = [c for c in cases if tagset.intersection(c.tags)]
    if phase:
        cases = [c for c in cases if c.phase in (phase, "both")]
    return cases


class KernelHarness:
    """Run YAML harness cases and persist to MongoDB kernel_benchmark_runs."""

    def __init__(
        self,
        *,
        db,
        gpu_type: str,
        cases_path: Optional[Path] = None,
    ) -> None:
        self._db = db
        self._gpu_type = gpu_type
        self._cases_path = cases_path or _DEFAULT_CASES

    async def run_suite(
        self,
        session_id: str,
        *,
        case_ids: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[str]] = None,
        phase: Optional[str] = None,
    ) -> List[HarnessCaseResult]:
        cases = load_harness_cases(
            self._cases_path,
            case_ids=case_ids,
            tags=tags,
            phase=phase,
        )
        if not cases:
            log.warning("KernelHarness: no cases selected")
            return []

        from microbench.operator_bench import OperatorBench
        from microbench.roofline import RooflineAnalyzer

        bench = OperatorBench(gpu_type=self._gpu_type)
        analyzer = RooflineAnalyzer(gpu_type=self._gpu_type)
        results: List[HarnessCaseResult] = []

        for case in cases:
            log.info(
                "Harness case %s op=%s phase=%s",
                case.id, case.op_type, case.phase,
            )
            try:
                result = await bench.run(
                    op_type=case.op_type,
                    params=case.params,
                    num_warmup=case.warmup,
                    num_iters=case.iters,
                )
                if not result.success:
                    rid = await self._db.insert_kernel_benchmark_run(
                        session_id=session_id,
                        op_type=case.op_type,
                        backend=f"harness:{case.id}",
                        gpu_type=self._gpu_type,
                        kernel_name=case.id,
                        params={**case.params, "phase": case.phase, "tags": case.tags},
                        metrics={"error": result.error},
                        source="kernel_harness",
                    )
                    results.append(HarnessCaseResult(
                        case_id=case.id,
                        op_type=case.op_type,
                        phase=case.phase,
                        success=False,
                        run_id=rid or "",
                        error=result.error,
                    ))
                    continue

                duration_s = max(result.latency_us_p50, 1.0) / 1e6
                op_flops = bench._estimate_flops(case.op_type, case.params)
                op_bytes = bench._estimate_memory_bytes(case.op_type, case.params)
                roof = analyzer.analyze(
                    kernel_name=case.id,
                    op_flops=op_flops,
                    op_bytes=op_bytes,
                    duration_s=duration_s,
                )
                bound = roof.points[0].bound if roof.points else result.roofline_bound
                metrics = {
                    "latency_us_mean": result.latency_us_mean,
                    "latency_us_p50": result.latency_us_p50,
                    "latency_us_p99": result.latency_us_p99,
                    "throughput_tflops": result.tflops_achieved,
                    "memory_gbps": result.mem_bw_gbps,
                    "roofline_bound": bound,
                    "roofline_efficiency_pct": roof.overall_efficiency_pct,
                    "compute_efficiency_pct": result.compute_efficiency_pct,
                    "mem_bw_efficiency_pct": result.mem_bw_efficiency_pct,
                    "phase": case.phase,
                    "harness_case_id": case.id,
                }
                rid = await self._db.insert_kernel_benchmark_run(
                    session_id=session_id,
                    op_type=case.op_type,
                    backend=f"harness:{case.id}",
                    gpu_type=self._gpu_type,
                    kernel_name=case.id,
                    params={**case.params, "phase": case.phase, "tags": case.tags},
                    metrics=metrics,
                    source="kernel_harness",
                )
                results.append(HarnessCaseResult(
                    case_id=case.id,
                    op_type=case.op_type,
                    phase=case.phase,
                    success=True,
                    run_id=rid or "",
                    metrics=metrics,
                ))
            except Exception as exc:
                log.warning("Harness case %s failed: %s", case.id, exc)
                results.append(HarnessCaseResult(
                    case_id=case.id,
                    op_type=case.op_type,
                    phase=case.phase,
                    success=False,
                    error=str(exc),
                ))

        ok = sum(1 for r in results if r.success)
        log.info("KernelHarness done: %d/%d cases succeeded", ok, len(results))
        return results
