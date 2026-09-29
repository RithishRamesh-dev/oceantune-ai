"""
core/operating_point_sweep.py
-----------------------------
Post-winner operating-point sweep (Hyperloom SWEEP analog).

After Stage 1–2 find a good config, re-measure across a concurrency ladder
and optional extra context lengths so the recipe records the true peak
operating point — without re-running the full search.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("core.operating_point_sweep")


@dataclass
class SweepPoint:
    concurrency: int
    input_len: int
    output_len: int
    output_tokens_per_sec: float = 0.0
    p95_latency_ms: float = 0.0
    mean_ttft_ms: float = 0.0
    fitness_score: float = 0.0
    failed: bool = False
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SweepResult:
    points: List[SweepPoint] = field(default_factory=list)
    best_concurrency: int = 0
    peak_throughput: float = 0.0
    best_point: Optional[SweepPoint] = None
    skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "points": [p.to_dict() for p in self.points],
            "best_concurrency": self.best_concurrency,
            "peak_throughput": self.peak_throughput,
            "best_point": self.best_point.to_dict() if self.best_point else None,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


DEFAULT_SWEEP_CONCURRENCY = [1, 4, 16, 64, 128, 256]


async def run_operating_point_sweep(
    *,
    base_url: str,
    model_id: str,
    concurrency_levels: Optional[List[int]] = None,
    context_configs: Optional[List[Tuple[int, int]]] = None,
    num_prompts: int = 30,
    flags: Optional[Dict[str, Any]] = None,
    gpu_type: str = "",
    primary_metric: str = "throughput",
) -> SweepResult:
    """
    Run a short concurrency ladder against an already-running vLLM server.

    Does not start/stop the server — caller owns lifecycle (like Stage 2 bench).
    """
    from core.benchmark_runner import BenchmarkEngine
    from core.log_analyzer import LogAnalyzer
    from core.metrics_collector import MetricsCollector
    from core.search_space import VLLMFlags

    levels = concurrency_levels or DEFAULT_SWEEP_CONCURRENCY
    contexts = context_configs or [(1024, 1024)]
    # Use first context for sweep speed; callers can pass decode-heavy too.
    input_len, output_len = contexts[0]

    result = SweepResult()
    known = set(VLLMFlags.__dataclass_fields__)
    vf = None
    if flags:
        try:
            vf = VLLMFlags(**{k: v for k, v in flags.items() if k in known})
        except Exception:
            vf = None

    gpu_profile: Dict[str, Any] = {}
    try:
        import yaml
        from pathlib import Path
        gp_path = Path(__file__).resolve().parent.parent / "configs" / "gpu_profiles.yaml"
        raw = yaml.safe_load(gp_path.read_text()) or {}
        gpu_profile = (raw.get("gpu_profiles") or {}).get(gpu_type, {})
    except Exception:
        pass

    for conc in levels:
        engine = BenchmarkEngine(
            base_url=base_url,
            model_id=model_id,
            concurrency_levels=[conc],
            num_prompts=num_prompts,
            input_len=input_len,
            output_len=output_len,
        )
        try:
            ramp = await engine.run()
            analysis = LogAnalyzer.analyze("")  # no server logs in this path
            if vf is not None:
                enriched = MetricsCollector.collect(
                    ramp=ramp,
                    analysis=analysis,
                    flags=vf,
                    gpu_profile=gpu_profile,
                    primary_metric=primary_metric,
                )
                fit = enriched.fitness_score
                summary = ramp.summary if hasattr(ramp, "summary") else {}
                if not summary and ramp.levels:
                    lvl = ramp.levels[0]
                    thr = getattr(lvl, "output_tokens_per_sec", 0.0)
                    p95 = getattr(lvl, "p95_latency_ms", 0.0)
                    ttft = getattr(lvl, "mean_ttft_ms", 0.0)
                else:
                    thr = summary.get("peak_throughput_tokens_per_sec", 0.0)
                    p95 = summary.get("p95_latency_at_peak_ms", 0.0)
                    ttft = summary.get("mean_ttft_ms", 0.0)
                point = SweepPoint(
                    concurrency=conc,
                    input_len=input_len,
                    output_len=output_len,
                    output_tokens_per_sec=float(thr or 0),
                    p95_latency_ms=float(p95 or 0),
                    mean_ttft_ms=float(ttft or 0),
                    fitness_score=float(fit or 0),
                    failed=False,
                )
            else:
                # Minimal parse without full MetricsCollector
                thr = 0.0
                p95 = 0.0
                ttft = 0.0
                if ramp.levels:
                    lvl = ramp.levels[0]
                    thr = float(getattr(lvl, "output_tokens_per_sec", 0) or 0)
                    p95 = float(getattr(lvl, "p95_latency_ms", 0) or 0)
                    ttft = float(getattr(lvl, "mean_ttft_ms", 0) or 0)
                point = SweepPoint(
                    concurrency=conc,
                    input_len=input_len,
                    output_len=output_len,
                    output_tokens_per_sec=thr,
                    p95_latency_ms=p95,
                    mean_ttft_ms=ttft,
                    fitness_score=thr,
                    failed=getattr(ramp.levels[0], "failed", False) if ramp.levels else True,
                )
        except Exception as exc:
            log.warning("Sweep concurrency=%d failed: %s", conc, exc)
            point = SweepPoint(
                concurrency=conc,
                input_len=input_len,
                output_len=output_len,
                failed=True,
                error=str(exc)[:200],
            )

        result.points.append(point)
        if not point.failed and point.output_tokens_per_sec > result.peak_throughput:
            result.peak_throughput = point.output_tokens_per_sec
            result.best_concurrency = point.concurrency
            result.best_point = point

    log.info(
        "Operating-point sweep done: peak=%.1f tok/s @ concurrency=%d (%d points)",
        result.peak_throughput,
        result.best_concurrency,
        len(result.points),
    )
    return result
