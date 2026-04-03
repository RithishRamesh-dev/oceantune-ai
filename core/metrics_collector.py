"""
core/metrics_collector.py
--------------------------
Derives enriched metrics and a normalised fitness score from raw
benchmark results and log analysis.

Inputs
------
- RampResult       from core.benchmark_runner (Step 4)
- LogAnalysis      from core.log_analyzer     (Step 5)
- VLLMFlags        from core.search_space     (Step 2)
- GPU profile dict from configs/gpu_profiles.yaml

Outputs
-------
EnrichedMetrics dataclass containing:
  - All RampResult summary fields (throughput, latency, TTFT, TPOT)
  - GPU efficiency: output tokens per second per GB of VRAM used
  - Memory headroom: fraction of VRAM still free after model load
  - Normalised fitness score: single float in [0, 1] for the optimiser
  - Failure penalty: 0.0 for clean runs, 1.0 for complete failures

The fitness score formula is configurable via the primary_metric field
in OceanTuneConfig. Default: weighted combination of throughput and p95
latency, penalised by error rate.

Design notes
------------
- All division is guarded against zero denominators.
- Missing GPU telemetry (kv_cache_gb not in logs) degrades gracefully.
- The normalisation baselines are stored as class constants so they can
  be calibrated per model in future steps without code changes.

Usage
-----
    from core.metrics_collector import MetricsCollector
    enriched = MetricsCollector.collect(
        ramp=ramp_result,
        analysis=log_analysis,
        flags=vllm_flags,
        gpu_profile=profile_dict,
        primary_metric="throughput",
    )
    print(enriched.fitness_score)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional

from core.benchmark_runner import RampResult
from core.log_analyzer import LogAnalysis
from core.search_space import VLLMFlags
from core.logger import get_logger, log_dict

log = get_logger("core.metrics_collector")


# ===========================================================================
# 1.  Normalisation baselines
# ===========================================================================
# These are conservative lower-bounds used to normalise raw metrics into
# [0, 1] fitness scores. They represent "barely acceptable" performance.
# Values are intentionally conservative so a wide range of hardware can
# produce meaningful scores.

# Throughput baseline: 100 output tokens/sec is achievable on a single GPU
_THROUGHPUT_BASELINE = 100.0       # tok/s

# Latency baseline: 2000ms p95 E2E is the "barely acceptable" ceiling
_LATENCY_BASELINE_MS = 2000.0      # ms

# TTFT baseline: 500ms mean TTFT
_TTFT_BASELINE_MS = 500.0          # ms

# TPOT baseline: 50ms per output token
_TPOT_BASELINE_MS = 50.0           # ms

# Penalty multiplier for each failed concurrency level
_FAILED_LEVEL_PENALTY = 0.1


# ===========================================================================
# 2.  EnrichedMetrics dataclass
# ===========================================================================

@dataclass
class EnrichedMetrics:
    """
    Enriched metrics for one experiment run.

    Combines raw benchmark numbers with GPU-efficiency derived metrics
    and a single normalised fitness score.
    """

    # ── Raw summary passthrough ───────────────────────────────────────────
    peak_throughput_tokens_per_sec: float = 0.0
    peak_requests_per_sec: float = 0.0
    p95_latency_at_peak_ms: float = 0.0
    p99_latency_at_peak_ms: float = 0.0
    mean_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    mean_tpot_ms: float = 0.0
    p95_tpot_ms: float = 0.0
    mean_itl_ms: float = 0.0
    error_rate_max: float = 0.0
    valid_levels: int = 0
    best_concurrency: int = 0

    # ── Derived GPU-efficiency metrics ────────────────────────────────────
    # Tokens/sec per GB of VRAM used — allows fair comparison across GPU sizes
    throughput_per_gb_vram: Optional[float] = None
    # Fraction of total GPU VRAM NOT used by the model (0.0 = fully loaded)
    memory_headroom_fraction: Optional[float] = None
    # Tokens/sec per watt (requires power telemetry — future step)
    throughput_per_watt: Optional[float] = None

    # ── Startup overhead ──────────────────────────────────────────────────
    model_load_time_sec: Optional[float] = None
    total_startup_sec: Optional[float] = None
    kv_cache_blocks: Optional[int] = None
    kv_cache_gb: Optional[float] = None

    # ── Quality flags ─────────────────────────────────────────────────────
    has_errors: bool = False
    failed_levels: int = 0
    all_levels_failed: bool = False

    # ── Fitness score ─────────────────────────────────────────────────────
    # Single float in [0, 1]. Higher = better. Used by the optimiser.
    fitness_score: float = 0.0
    primary_metric_used: str = "throughput"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_usable(self) -> bool:
        """True if there is at least one valid benchmark level."""
        return self.valid_levels > 0 and not self.all_levels_failed


# ===========================================================================
# 3.  MetricsCollector
# ===========================================================================

class MetricsCollector:
    """
    Stateless enriched-metrics producer.

    Call MetricsCollector.collect(...) directly — no instance needed.
    """

    @classmethod
    def collect(
        cls,
        ramp: RampResult,
        analysis: LogAnalysis,
        flags: VLLMFlags,
        gpu_profile: dict,
        primary_metric: str = "throughput",
    ) -> EnrichedMetrics:
        """
        Compute enriched metrics from benchmark + log analysis.

        Parameters
        ----------
        ramp : RampResult
            Full concurrency-ramp results from BenchmarkEngine.
        analysis : LogAnalysis
            Structured log analysis from LogAnalyzer.
        flags : VLLMFlags
            The config that produced this run.
        gpu_profile : dict
            Entry from gpu_profiles.yaml for the target GPU
            (used for VRAM capacity).
        primary_metric : str
            Which metric drives the fitness score.
            One of: throughput | p95_latency | ttft | tpot

        Returns
        -------
        EnrichedMetrics — guaranteed not to raise.
        """
        summary = ramp.summary

        em = EnrichedMetrics(
            # ── Pass through raw summary ──────────────────────────────────
            peak_throughput_tokens_per_sec=summary.get(
                "peak_throughput_tokens_per_sec", 0.0),
            peak_requests_per_sec=summary.get("peak_requests_per_sec", 0.0),
            p95_latency_at_peak_ms=summary.get("p95_latency_at_peak_ms", 0.0),
            p99_latency_at_peak_ms=summary.get("p99_latency_at_peak_ms", 0.0),
            mean_ttft_ms=summary.get("mean_ttft_ms", 0.0),
            p95_ttft_ms=summary.get("p95_ttft_ms", 0.0),
            mean_tpot_ms=summary.get("mean_tpot_ms", 0.0),
            p95_tpot_ms=summary.get("p95_tpot_ms", 0.0),
            mean_itl_ms=summary.get("mean_itl_ms", 0.0),
            error_rate_max=summary.get("error_rate_max", 0.0),
            valid_levels=int(summary.get("valid_levels", 0)),
            best_concurrency=int(summary.get("best_concurrency", 0)),
            # ── Status ────────────────────────────────────────────────────
            has_errors=analysis.has_errors,
            failed_levels=ramp.failed_levels,
            all_levels_failed=ramp.all_failed(),
            # ── Log-derived startup metrics ───────────────────────────────
            model_load_time_sec=analysis.model_load_time_sec,
            total_startup_sec=analysis.total_startup_sec,
            kv_cache_blocks=analysis.kv_cache_blocks,
            kv_cache_gb=analysis.kv_cache_gb,
            primary_metric_used=primary_metric,
        )

        # ── Derived: GPU efficiency ───────────────────────────────────────
        em.throughput_per_gb_vram = cls._throughput_per_gb(
            em.peak_throughput_tokens_per_sec,
            analysis.gpu_memory_used_gb,
        )
        em.memory_headroom_fraction = cls._memory_headroom(
            analysis.gpu_memory_used_gb,
            gpu_profile.get("vram_gb"),
            flags.tensor_parallel_size,
        )

        # ── Fitness score ─────────────────────────────────────────────────
        em.fitness_score = cls._compute_fitness(em, primary_metric)

        log_dict(
            log, "info", "Metrics collected",
            throughput=round(em.peak_throughput_tokens_per_sec, 1),
            p95_ms=round(em.p95_latency_at_peak_ms, 1),
            ttft_ms=round(em.mean_ttft_ms, 1),
            fitness=round(em.fitness_score, 4),
            valid_levels=em.valid_levels,
            primary_metric=primary_metric,
        )

        return em

    # ── GPU efficiency helpers ────────────────────────────────────────────

    @staticmethod
    def _throughput_per_gb(
        throughput: float,
        gpu_memory_used_gb: Optional[float],
    ) -> Optional[float]:
        """Output tokens/sec per GB of GPU memory used."""
        if gpu_memory_used_gb is None or gpu_memory_used_gb <= 0:
            return None
        return round(throughput / gpu_memory_used_gb, 3)

    @staticmethod
    def _memory_headroom(
        gpu_memory_used_gb: Optional[float],
        vram_gb: Optional[float],
        tensor_parallel_size: int = 1,
    ) -> Optional[float]:
        """
        Fraction of total VRAM not used by the model.

        For multi-GPU setups, vram_gb is the per-GPU capacity.
        Total capacity = vram_gb * tensor_parallel_size.
        Returns None if VRAM capacity is unknown.
        """
        if vram_gb is None or vram_gb <= 0:
            return None
        if gpu_memory_used_gb is None or gpu_memory_used_gb <= 0:
            return None
        total_vram = vram_gb * max(1, tensor_parallel_size)
        headroom = 1.0 - (gpu_memory_used_gb / total_vram)
        return round(max(0.0, min(1.0, headroom)), 4)

    # ── Fitness score ─────────────────────────────────────────────────────

    @classmethod
    def _compute_fitness(
        cls,
        em: EnrichedMetrics,
        primary_metric: str,
    ) -> float:
        """
        Compute a normalised fitness score in [0.0, 1.0].

        Higher is always better. The optimiser maximises this value.

        Score formula depends on primary_metric:
          throughput  — weighted: 70% normalised throughput + 30% latency score
          p95_latency — weighted: 30% throughput + 70% latency score
          ttft        — weighted: 20% throughput + 80% TTFT score
          tpot        — weighted: 20% throughput + 80% TPOT score

        All scores are penalised by:
          - error rate (linear, up to -50% penalty)
          - failed levels (each adds -10% penalty)
          - OOM / crash (0.0 immediately)
        """
        # Hard fail: no usable results → 0.0
        if em.all_levels_failed or em.valid_levels == 0:
            return 0.0

        # ── Component scores (each in [0, 1]) ─────────────────────────────

        # Throughput score: log-scaled so doublings have equal value
        throughput_score = cls._log_score(
            em.peak_throughput_tokens_per_sec,
            baseline=_THROUGHPUT_BASELINE,
            ceiling=50000.0,   # ~theoretical max for largest GPUs
        )

        # Latency score: inverted — lower latency = higher score
        latency_score = cls._inv_score(
            em.p95_latency_at_peak_ms,
            baseline_bad=_LATENCY_BASELINE_MS,
            baseline_good=10.0,   # 10ms is near-perfect
        )

        # TTFT score
        ttft_score = cls._inv_score(
            em.mean_ttft_ms,
            baseline_bad=_TTFT_BASELINE_MS,
            baseline_good=5.0,
        )

        # TPOT score
        tpot_score = cls._inv_score(
            em.mean_tpot_ms,
            baseline_bad=_TPOT_BASELINE_MS,
            baseline_good=0.5,
        )

        # ── Weighted combination ──────────────────────────────────────────
        weights: Dict[str, tuple] = {
            #               throughput  latency  ttft  tpot
            "throughput":  (0.70,       0.30,    0.00, 0.00),
            "p95_latency": (0.30,       0.70,    0.00, 0.00),
            "ttft":        (0.20,       0.00,    0.80, 0.00),
            "tpot":        (0.20,       0.00,    0.00, 0.80),
        }
        wt, wl, wttft, wtpot = weights.get(
            primary_metric, weights["throughput"]
        )
        raw_score = (
            wt    * throughput_score +
            wl    * latency_score +
            wttft * ttft_score +
            wtpot * tpot_score
        )

        # ── Penalties ─────────────────────────────────────────────────────
        # Error rate: linear penalty, max 50%
        error_penalty = min(0.5, em.error_rate_max * 0.5)

        # Failed levels: each level that failed costs 10%
        level_penalty = min(0.5, em.failed_levels * _FAILED_LEVEL_PENALTY)

        # OOM-specific additional penalty
        oom_penalty = 0.3 if em.has_errors else 0.0

        total_penalty = min(0.99, error_penalty + level_penalty + oom_penalty)
        final_score = raw_score * (1.0 - total_penalty)

        return round(max(0.0, min(1.0, final_score)), 6)

    @staticmethod
    def _log_score(
        value: float,
        baseline: float,
        ceiling: float,
    ) -> float:
        """
        Map value to [0, 1] using log scale.

        value <= 0      → 0.0
        value = baseline → ~0.0 (just above)
        value = ceiling  → 1.0
        """
        if value <= 0 or baseline <= 0 or ceiling <= baseline:
            return 0.0
        log_val = math.log(max(value, baseline)) - math.log(baseline)
        log_max = math.log(ceiling) - math.log(baseline)
        return min(1.0, log_val / log_max)

    @staticmethod
    def _inv_score(
        value: float,
        baseline_bad: float,
        baseline_good: float,
    ) -> float:
        """
        Map a "lower is better" value to [0, 1].

        value >= baseline_bad  → 0.0 (terrible)
        value <= baseline_good → 1.0 (perfect)
        """
        if value <= 0:
            return 1.0   # zero latency = perfect (shouldn't happen in practice)
        if value >= baseline_bad:
            return 0.0
        if value <= baseline_good:
            return 1.0
        # Linear interpolation between good and bad
        return (baseline_bad - value) / (baseline_bad - baseline_good)