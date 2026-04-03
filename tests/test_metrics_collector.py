"""
tests/test_metrics_collector.py
--------------------------------
Unit tests for core/metrics_collector.py.

Run with:
    pytest tests/test_metrics_collector.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from core.metrics_collector import MetricsCollector, EnrichedMetrics
from core.benchmark_runner import RampResult, BenchmarkResult
from core.log_analyzer import LogAnalysis
from core.search_space import VLLMFlags


# ===========================================================================
# Helpers
# ===========================================================================

H100_PROFILE = {
    "vram_gb": 80,
    "vendor": "nvidia",
    "fp8": True,
    "bf16": True,
}

MI300X_PROFILE = {
    "vram_gb": 192,
    "vendor": "amd",
    "fp8": True,
    "bf16": True,
}


def make_ramp(
    throughputs=(500.0, 1500.0, 3000.0, 2800.0),
    p95s=(200.0, 400.0, 600.0, 900.0),
    ttft=80.0,
    error_rate=0.0,
    failed_levels=0,
) -> RampResult:
    levels = []
    for i, (tp, p95) in enumerate(zip(throughputs, p95s)):
        r = BenchmarkResult(
            concurrency=2 ** i,
            output_tokens_per_sec=tp,
            requests_per_sec=tp / 150,
            p95_latency_ms=p95,
            p99_latency_ms=p95 * 1.2,
            mean_ttft_ms=ttft,
            p95_ttft_ms=ttft * 1.5,
            mean_tpot_ms=5.0,
            p95_tpot_ms=7.0,
            mean_itl_ms=5.0,
            error_rate=error_rate,
            failed=(tp == 0),
        )
        levels.append(r)

    from core.benchmark_runner import _compute_summary
    summary = _compute_summary(levels)

    return RampResult(
        endpoint="http://localhost:8000/v1",
        input_len=1024,
        output_len=1024,
        levels=levels,
        summary=summary,
        total_duration_sec=120.0,
        failed_levels=failed_levels,
    )


def make_analysis(
    model_load_sec=40.0,
    kv_blocks=8192,
    kv_gb=12.0,
    gpu_mem_gb=45.0,
    ready=True,
    errors=None,
) -> LogAnalysis:
    a = LogAnalysis(
        model_load_time_sec=model_load_sec,
        kv_cache_blocks=kv_blocks,
        kv_cache_gb=kv_gb,
        gpu_memory_used_gb=gpu_mem_gb,
        server_became_ready=ready,
        total_log_lines=100,
        error_classes=set(errors or []),
    )
    return a


def make_flags(**kwargs) -> VLLMFlags:
    defaults = dict(tensor_parallel_size=1, gpu_memory_utilization=0.90)
    defaults.update(kwargs)
    f = VLLMFlags(**defaults)
    f.run_id = f.fingerprint()
    return f


# ===========================================================================
# Tests: EnrichedMetrics dataclass
# ===========================================================================

class TestEnrichedMetrics:
    def test_is_usable_true_when_valid_levels(self):
        em = EnrichedMetrics(valid_levels=3, all_levels_failed=False)
        assert em.is_usable is True

    def test_is_usable_false_when_all_failed(self):
        em = EnrichedMetrics(valid_levels=0, all_levels_failed=True)
        assert em.is_usable is False

    def test_is_usable_false_when_zero_valid(self):
        em = EnrichedMetrics(valid_levels=0, all_levels_failed=False)
        assert em.is_usable is False

    def test_to_dict_returns_dict(self):
        em = EnrichedMetrics(peak_throughput_tokens_per_sec=2000.0)
        d = em.to_dict()
        assert isinstance(d, dict)
        assert d["peak_throughput_tokens_per_sec"] == 2000.0


# ===========================================================================
# Tests: MetricsCollector.collect
# ===========================================================================

class TestCollectPassthrough:
    def test_throughput_passed_through(self):
        ramp = make_ramp()
        analysis = make_analysis()
        flags = make_flags()
        em = MetricsCollector.collect(ramp, analysis, flags, H100_PROFILE)
        assert em.peak_throughput_tokens_per_sec == ramp.summary[
            "peak_throughput_tokens_per_sec"]

    def test_p95_latency_passed_through(self):
        ramp = make_ramp()
        analysis = make_analysis()
        flags = make_flags()
        em = MetricsCollector.collect(ramp, analysis, flags, H100_PROFILE)
        assert em.p95_latency_at_peak_ms == ramp.summary["p95_latency_at_peak_ms"]

    def test_ttft_passed_through(self):
        ramp = make_ramp(ttft=75.0)
        analysis = make_analysis()
        flags = make_flags()
        em = MetricsCollector.collect(ramp, analysis, flags, H100_PROFILE)
        assert em.mean_ttft_ms == ramp.summary["mean_ttft_ms"]

    def test_valid_levels_passed_through(self):
        ramp = make_ramp()
        analysis = make_analysis()
        flags = make_flags()
        em = MetricsCollector.collect(ramp, analysis, flags, H100_PROFILE)
        assert em.valid_levels == int(ramp.summary["valid_levels"])

    def test_log_fields_passed_through(self):
        ramp = make_ramp()
        analysis = make_analysis(model_load_sec=55.0, kv_blocks=4096)
        flags = make_flags()
        em = MetricsCollector.collect(ramp, analysis, flags, H100_PROFILE)
        assert em.model_load_time_sec == 55.0
        assert em.kv_cache_blocks == 4096


# ===========================================================================
# Tests: GPU-efficiency derived metrics
# ===========================================================================

class TestGPUEfficiency:
    def test_throughput_per_gb_computed(self):
        ramp = make_ramp()
        analysis = make_analysis(gpu_mem_gb=40.0)
        flags = make_flags()
        em = MetricsCollector.collect(ramp, analysis, flags, H100_PROFILE)
        assert em.throughput_per_gb_vram is not None
        expected = em.peak_throughput_tokens_per_sec / 40.0
        assert abs(em.throughput_per_gb_vram - expected) < 0.01

    def test_throughput_per_gb_none_when_no_gpu_mem(self):
        ramp = make_ramp()
        analysis = make_analysis(gpu_mem_gb=None)
        flags = make_flags()
        em = MetricsCollector.collect(ramp, analysis, flags, H100_PROFILE)
        assert em.throughput_per_gb_vram is None

    def test_memory_headroom_fraction_computed(self):
        # H100 = 80 GB, used 40 GB, TP=1 → headroom = 0.5
        ramp = make_ramp()
        analysis = make_analysis(gpu_mem_gb=40.0)
        flags = make_flags(tensor_parallel_size=1)
        em = MetricsCollector.collect(ramp, analysis, flags, H100_PROFILE)
        assert em.memory_headroom_fraction is not None
        assert abs(em.memory_headroom_fraction - 0.5) < 0.01

    def test_memory_headroom_accounts_for_tp(self):
        # H100 = 80 GB × TP=2 = 160 GB total, used 40 GB → headroom = 0.75
        ramp = make_ramp()
        analysis = make_analysis(gpu_mem_gb=40.0)
        flags = make_flags(tensor_parallel_size=2)
        em = MetricsCollector.collect(ramp, analysis, flags, H100_PROFILE)
        assert em.memory_headroom_fraction is not None
        assert abs(em.memory_headroom_fraction - 0.75) < 0.01

    def test_memory_headroom_none_when_no_profile_vram(self):
        ramp = make_ramp()
        analysis = make_analysis(gpu_mem_gb=40.0)
        flags = make_flags()
        em = MetricsCollector.collect(ramp, analysis, flags, {})  # no vram_gb
        assert em.memory_headroom_fraction is None

    def test_memory_headroom_clamped_to_zero_not_negative(self):
        # Used more than total (shouldn't happen but guard anyway)
        ramp = make_ramp()
        analysis = make_analysis(gpu_mem_gb=200.0)
        flags = make_flags(tensor_parallel_size=1)
        em = MetricsCollector.collect(ramp, analysis, flags, H100_PROFILE)
        assert em.memory_headroom_fraction == 0.0


# ===========================================================================
# Tests: Fitness score
# ===========================================================================

class TestFitnessScore:
    def test_fitness_in_zero_one_range(self):
        ramp = make_ramp()
        analysis = make_analysis()
        flags = make_flags()
        em = MetricsCollector.collect(ramp, analysis, flags, H100_PROFILE)
        assert 0.0 <= em.fitness_score <= 1.0

    def test_all_failed_gives_zero_fitness(self):
        # All levels failed
        levels = [
            BenchmarkResult(concurrency=c, output_tokens_per_sec=0.0,
                            failed=True) for c in [1, 2, 4]
        ]
        ramp = RampResult(
            endpoint="http://localhost:8000/v1",
            input_len=512, output_len=512,
            levels=levels,
            summary={"valid_levels": 0.0, "peak_throughput_tokens_per_sec": 0.0},
            failed_levels=3,
        )
        analysis = make_analysis()
        flags = make_flags()
        em = MetricsCollector.collect(ramp, analysis, flags, H100_PROFILE)
        assert em.fitness_score == 0.0

    def test_higher_throughput_gives_higher_fitness(self):
        low_ramp = make_ramp(throughputs=(100.0, 200.0, 300.0, 250.0))
        high_ramp = make_ramp(throughputs=(500.0, 1500.0, 3000.0, 2800.0))
        analysis = make_analysis()
        flags = make_flags()
        em_low = MetricsCollector.collect(low_ramp, analysis, flags,
                                          H100_PROFILE, "throughput")
        em_high = MetricsCollector.collect(high_ramp, analysis, flags,
                                           H100_PROFILE, "throughput")
        assert em_high.fitness_score > em_low.fitness_score

    def test_error_rate_reduces_fitness(self):
        clean_ramp = make_ramp(error_rate=0.0)
        errored_ramp = make_ramp(error_rate=0.5)
        analysis = make_analysis()
        flags = make_flags()
        em_clean = MetricsCollector.collect(clean_ramp, analysis, flags, H100_PROFILE)
        em_err = MetricsCollector.collect(errored_ramp, analysis, flags, H100_PROFILE)
        assert em_clean.fitness_score > em_err.fitness_score

    def test_oom_error_reduces_fitness(self):
        ramp = make_ramp()
        clean_analysis = make_analysis(errors=[])
        oom_analysis = make_analysis(errors=["oom"])
        flags = make_flags()
        em_clean = MetricsCollector.collect(ramp, clean_analysis, flags, H100_PROFILE)
        em_oom = MetricsCollector.collect(ramp, oom_analysis, flags, H100_PROFILE)
        assert em_clean.fitness_score > em_oom.fitness_score

    def test_primary_metric_throughput(self):
        ramp = make_ramp()
        analysis = make_analysis()
        flags = make_flags()
        em = MetricsCollector.collect(ramp, analysis, flags, H100_PROFILE,
                                      primary_metric="throughput")
        assert em.primary_metric_used == "throughput"

    def test_primary_metric_p95_latency(self):
        ramp = make_ramp()
        analysis = make_analysis()
        flags = make_flags()
        em = MetricsCollector.collect(ramp, analysis, flags, H100_PROFILE,
                                      primary_metric="p95_latency")
        assert em.primary_metric_used == "p95_latency"
        assert 0.0 <= em.fitness_score <= 1.0

    def test_primary_metric_ttft(self):
        ramp = make_ramp()
        analysis = make_analysis()
        flags = make_flags()
        em = MetricsCollector.collect(ramp, analysis, flags, H100_PROFILE,
                                      primary_metric="ttft")
        assert 0.0 <= em.fitness_score <= 1.0

    def test_primary_metric_tpot(self):
        ramp = make_ramp()
        analysis = make_analysis()
        flags = make_flags()
        em = MetricsCollector.collect(ramp, analysis, flags, H100_PROFILE,
                                      primary_metric="tpot")
        assert 0.0 <= em.fitness_score <= 1.0

    def test_unknown_primary_metric_falls_back_to_throughput(self):
        ramp = make_ramp()
        analysis = make_analysis()
        flags = make_flags()
        # Should not raise even with unknown metric
        em = MetricsCollector.collect(ramp, analysis, flags, H100_PROFILE,
                                      primary_metric="unknown_metric")
        assert 0.0 <= em.fitness_score <= 1.0

    def test_fitness_score_is_rounded(self):
        ramp = make_ramp()
        analysis = make_analysis()
        flags = make_flags()
        em = MetricsCollector.collect(ramp, analysis, flags, H100_PROFILE)
        # Should have at most 6 decimal places
        assert em.fitness_score == round(em.fitness_score, 6)


# ===========================================================================
# Tests: log score / inv score helpers
# ===========================================================================

class TestScoreHelpers:
    def test_log_score_zero_for_zero_input(self):
        score = MetricsCollector._log_score(0.0, baseline=100.0, ceiling=50000.0)
        assert score == 0.0

    def test_log_score_one_at_ceiling(self):
        score = MetricsCollector._log_score(50000.0, baseline=100.0, ceiling=50000.0)
        assert abs(score - 1.0) < 0.01

    def test_log_score_above_ceiling_clamped(self):
        score = MetricsCollector._log_score(100000.0, baseline=100.0, ceiling=50000.0)
        assert score <= 1.0

    def test_inv_score_one_at_best(self):
        score = MetricsCollector._inv_score(5.0, baseline_bad=2000.0, baseline_good=5.0)
        assert score == 1.0

    def test_inv_score_zero_at_worst(self):
        score = MetricsCollector._inv_score(3000.0, baseline_bad=2000.0, baseline_good=5.0)
        assert score == 0.0

    def test_inv_score_intermediate(self):
        score = MetricsCollector._inv_score(1000.0, baseline_bad=2000.0, baseline_good=0.0)
        assert 0.0 < score < 1.0

    def test_inv_score_zero_latency_is_perfect(self):
        # Zero latency is impossible but guarded
        score = MetricsCollector._inv_score(0.0, baseline_bad=2000.0, baseline_good=5.0)
        assert score == 1.0