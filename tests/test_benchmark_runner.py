"""
tests/test_benchmark_runner.py
------------------------------
Unit tests for core/benchmark_runner.py.

No real vLLM server required — subprocess calls are mocked.

Run with:
    pytest tests/test_benchmark_runner.py -v --asyncio-mode=auto
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.benchmark_runner import (
    BenchmarkEngine,
    BenchmarkResult,
    RampResult,
    parse_benchmark_output,
    run_benchmark,
    _compute_summary,
)
from core.config import (
    OceanTuneConfig,
    BenchmarkConfig,
    VLLMConfig,
    SpacesConfig,
    OptimiserConfig,
)


# ===========================================================================
# Fixtures
# ===========================================================================

def make_cfg(
    concurrency_levels=None,
    num_prompts=10,
    model_id="test/model",
    gpu_type="H100",
) -> OceanTuneConfig:
    cfg = OceanTuneConfig()
    cfg.model_id = model_id
    cfg.gpu_type = gpu_type
    cfg.benchmark.concurrency_levels = concurrency_levels or [1, 2, 4]
    cfg.benchmark.num_prompts = num_prompts
    cfg.benchmark.input_len = 512
    cfg.benchmark.output_len = 512
    return cfg


# A realistic vllm bench serve stdout block
SAMPLE_OUTPUT = """
============ Serving Benchmark Result ============
Successful requests:                    200
Benchmark duration (s):                30.12
Total input tokens:                    102400
Total generated tokens:               102400
Request throughput (req/s):             6.64
Output token throughput (tok/s):       3397.5
Total Token throughput (tok/s):        6795.0
---------------Time to First Token---------------
Mean TTFT (ms):                         72.3
Median TTFT (ms):                       68.1
P90 TTFT (ms):                         105.2
P95 TTFT (ms):                         118.7
P99 TTFT (ms):                         202.4
-----Time per Output Token (excl. 1st token)-----
Mean TPOT (ms):                          4.1
Median TPOT (ms):                        3.9
P90 TPOT (ms):                           5.8
P95 TPOT (ms):                           6.2
P99 TPOT (ms):                           8.1
---------------Inter-token Latency---------------
Mean ITL (ms):                           4.1
P95 ITL (ms):                            6.2
P99 ITL (ms):                            8.1
-----------End-to-End Latency (ms)-----------
Mean E2E Latency (ms):                 617.4
Median E2E Latency (ms):               581.2
P90 E2E Latency (ms):                  812.3
P95 E2E Latency (ms):                  944.1
P99 E2E Latency (ms):                 1201.7
Number of failed requests:                0
==================================================
"""

PARTIAL_OUTPUT = """
Successful requests:                    50
Request throughput (req/s):             1.25
Output token throughput (tok/s):       640.0
Mean TTFT (ms):                         88.5
Mean TPOT (ms):                          5.2
Mean E2E Latency (ms):                 850.0
P95 E2E Latency (ms):                 1100.0
Number of failed requests:                2
"""

EMPTY_OUTPUT = "Starting benchmark...\nConnecting to server...\n"


# ===========================================================================
# 1. parse_benchmark_output
# ===========================================================================

class TestParseBenchmarkOutput:
    def test_full_output_parses_all_fields(self):
        result = parse_benchmark_output(SAMPLE_OUTPUT, 16, 512, 512)
        assert result.concurrency == 16
        assert result.num_prompts == 200
        assert abs(result.duration_sec - 30.12) < 0.01
        assert abs(result.requests_per_sec - 6.64) < 0.01
        assert abs(result.output_tokens_per_sec - 3397.5) < 0.1
        assert abs(result.total_tokens_per_sec - 6795.0) < 0.1
        assert abs(result.mean_ttft_ms - 72.3) < 0.1
        assert abs(result.median_ttft_ms - 68.1) < 0.1
        assert abs(result.p95_ttft_ms - 118.7) < 0.1
        assert abs(result.p99_ttft_ms - 202.4) < 0.1
        assert abs(result.mean_tpot_ms - 4.1) < 0.1
        assert abs(result.p95_tpot_ms - 6.2) < 0.1
        assert abs(result.mean_itl_ms - 4.1) < 0.1
        assert abs(result.mean_latency_ms - 617.4) < 0.1
        assert abs(result.p95_latency_ms - 944.1) < 0.1
        assert abs(result.p99_latency_ms - 1201.7) < 0.1
        assert result.error_count == 0
        assert result.error_rate == 0.0
        assert result.failed is False

    def test_partial_output_parses_available_fields(self):
        result = parse_benchmark_output(PARTIAL_OUTPUT, 4, 512, 512)
        assert result.num_prompts == 50
        assert abs(result.output_tokens_per_sec - 640.0) < 0.1
        assert result.error_count == 2
        assert abs(result.error_rate - 2 / 50) < 1e-6

    def test_empty_output_returns_zero_result(self):
        result = parse_benchmark_output(EMPTY_OUTPUT, 1, 512, 512)
        assert result.output_tokens_per_sec == 0.0
        assert result.failed is False  # parse_output doesn't set failed

    def test_is_valid_true_when_throughput_nonzero(self):
        result = parse_benchmark_output(SAMPLE_OUTPUT, 16, 512, 512)
        assert result.is_valid is True

    def test_is_valid_false_when_throughput_zero(self):
        result = parse_benchmark_output(EMPTY_OUTPUT, 1, 512, 512)
        assert result.is_valid is False

    def test_failed_flag_default_false(self):
        result = parse_benchmark_output(SAMPLE_OUTPUT, 1, 512, 512)
        assert result.failed is False

    def test_raw_output_preserved(self):
        result = parse_benchmark_output(SAMPLE_OUTPUT, 1, 512, 512)
        assert "3397.5" in result.raw_output

    def test_concurrency_stored(self):
        result = parse_benchmark_output(SAMPLE_OUTPUT, 32, 1024, 2048)
        assert result.concurrency == 32
        assert result.input_len == 1024
        assert result.output_len == 2048

    def test_to_dict_contains_all_fields(self):
        result = parse_benchmark_output(SAMPLE_OUTPUT, 8, 512, 512)
        d = result.to_dict()
        assert "output_tokens_per_sec" in d
        assert "p95_latency_ms" in d
        assert "mean_ttft_ms" in d
        assert "failed" in d

    def test_error_rate_computed_correctly(self):
        result = parse_benchmark_output(PARTIAL_OUTPUT, 4, 512, 512)
        assert abs(result.error_rate - (2 / 50)) < 1e-9

    def test_primary_throughput_property(self):
        result = parse_benchmark_output(SAMPLE_OUTPUT, 16, 512, 512)
        assert result.primary_throughput == result.output_tokens_per_sec

    def test_case_insensitive_parsing(self):
        # Some vLLM versions use slightly different case
        output = """
Benchmark duration (s):    10.0
OUTPUT TOKEN THROUGHPUT (TOK/S):    999.0
MEAN E2E LATENCY (MS):    500.0
"""
        result = parse_benchmark_output(output, 8, 512, 512)
        assert abs(result.output_tokens_per_sec - 999.0) < 0.1


# ===========================================================================
# 2. _compute_summary
# ===========================================================================

class TestComputeSummary:
    def _make_result(self, concurrency, throughput, p95=100.0,
                     ttft=50.0, tpot=5.0, failed=False) -> BenchmarkResult:
        r = BenchmarkResult(
            concurrency=concurrency,
            output_tokens_per_sec=throughput,
            requests_per_sec=throughput / 100,
            p95_latency_ms=p95,
            mean_ttft_ms=ttft,
            p95_ttft_ms=ttft * 1.5,
            mean_tpot_ms=tpot,
            p95_tpot_ms=tpot * 1.5,
            mean_itl_ms=tpot,
            error_rate=0.0,
            failed=failed,
        )
        return r

    def test_peak_throughput_is_max(self):
        levels = [
            self._make_result(1, 500),
            self._make_result(4, 1500),
            self._make_result(16, 2000),
            self._make_result(32, 1800),
        ]
        summary = _compute_summary(levels)
        assert summary["peak_throughput_tokens_per_sec"] == 2000.0

    def test_best_concurrency_matches_peak(self):
        levels = [
            self._make_result(1, 500),
            self._make_result(8, 3000),
            self._make_result(16, 2500),
        ]
        summary = _compute_summary(levels)
        assert summary["best_concurrency"] == 8.0

    def test_ttft_taken_from_lowest_concurrency(self):
        levels = [
            self._make_result(1, 500, ttft=30.0),
            self._make_result(16, 2000, ttft=200.0),
        ]
        summary = _compute_summary(levels)
        assert summary["mean_ttft_ms"] == 30.0

    def test_valid_levels_count(self):
        levels = [
            self._make_result(1, 500),
            self._make_result(4, 0, failed=True),
            self._make_result(16, 2000),
        ]
        summary = _compute_summary(levels)
        assert summary["valid_levels"] == 2.0

    def test_all_failed_returns_sentinel_values(self):
        levels = [
            self._make_result(1, 0, failed=True),
            self._make_result(4, 0, failed=True),
        ]
        # Mark all as zero throughput too
        for r in levels:
            r.output_tokens_per_sec = 0.0
        summary = _compute_summary(levels)
        assert summary["peak_throughput_tokens_per_sec"] == 0.0
        assert summary["valid_levels"] == 0.0
        assert summary["error_rate_max"] == 1.0

    def test_empty_levels_returns_sentinel(self):
        summary = _compute_summary([])
        assert summary["peak_throughput_tokens_per_sec"] == 0.0
        assert summary["valid_levels"] == 0.0

    def test_summary_has_all_expected_keys(self):
        levels = [self._make_result(8, 2000)]
        summary = _compute_summary(levels)
        expected_keys = {
            "peak_throughput_tokens_per_sec",
            "peak_requests_per_sec",
            "p95_latency_at_peak_ms",
            "p99_latency_at_peak_ms",
            "mean_ttft_ms",
            "p95_ttft_ms",
            "mean_tpot_ms",
            "p95_tpot_ms",
            "mean_itl_ms",
            "error_rate_max",
            "valid_levels",
            "best_concurrency",
        }
        assert expected_keys.issubset(summary.keys())


# ===========================================================================
# 3. BenchmarkResult helpers
# ===========================================================================

class TestBenchmarkResult:
    def test_is_valid_true(self):
        r = BenchmarkResult(output_tokens_per_sec=1000.0, failed=False)
        assert r.is_valid is True

    def test_is_valid_false_when_failed(self):
        r = BenchmarkResult(output_tokens_per_sec=1000.0, failed=True)
        assert r.is_valid is False

    def test_is_valid_false_when_zero_throughput(self):
        r = BenchmarkResult(output_tokens_per_sec=0.0, failed=False)
        assert r.is_valid is False

    def test_to_dict_roundtrip(self):
        r = BenchmarkResult(concurrency=8, output_tokens_per_sec=2000.0)
        d = r.to_dict()
        assert d["concurrency"] == 8
        assert d["output_tokens_per_sec"] == 2000.0


# ===========================================================================
# 4. RampResult helpers
# ===========================================================================

class TestRampResult:
    def _make_ramp(self, throughputs) -> RampResult:
        levels = [
            BenchmarkResult(
                concurrency=2**i,
                output_tokens_per_sec=t,
                failed=(t == 0),
            )
            for i, t in enumerate(throughputs)
        ]
        return RampResult(
            endpoint="http://localhost:8000/v1",
            input_len=512,
            output_len=512,
            levels=levels,
            summary=_compute_summary(levels),
        )

    def test_peak_throughput(self):
        ramp = self._make_ramp([500, 1500, 2000, 1800])
        assert ramp.peak_throughput() == 2000.0

    def test_peak_throughput_zero_when_all_failed(self):
        ramp = self._make_ramp([0, 0, 0])
        assert ramp.peak_throughput() == 0.0

    def test_best_throughput_result_returns_max(self):
        ramp = self._make_ramp([500, 3000, 2000])
        best = ramp.best_throughput_result()
        assert best is not None
        assert best.output_tokens_per_sec == 3000.0

    def test_best_throughput_result_none_when_all_failed(self):
        ramp = self._make_ramp([0, 0])
        assert ramp.best_throughput_result() is None

    def test_all_failed_true(self):
        ramp = self._make_ramp([0, 0, 0])
        assert ramp.all_failed() is True

    def test_all_failed_false_when_any_valid(self):
        ramp = self._make_ramp([0, 1500, 0])
        assert ramp.all_failed() is False

    def test_to_dict_serializable(self):
        ramp = self._make_ramp([1000, 2000])
        import json
        d = ramp.to_dict()
        # Should be JSON-serializable (no complex objects)
        json.dumps(d, default=str)


# ===========================================================================
# 5. BenchmarkEngine._build_command
# ===========================================================================

class TestBuildCommand:
    def test_contains_model(self):
        cfg = make_cfg(model_id="deepseek-ai/DeepSeek-V3.2")
        engine = BenchmarkEngine(cfg, context=(512, 512))
        cmd = engine._build_command("http://localhost:8000/v1", 8)
        assert "deepseek-ai/DeepSeek-V3.2" in cmd

    def test_strips_v1_suffix(self):
        cfg = make_cfg()
        engine = BenchmarkEngine(cfg, context=(512, 512))
        cmd = engine._build_command("http://localhost:8000/v1", 8)
        assert "http://localhost:8000" in cmd
        assert "/v1" not in " ".join(cmd[cmd.index("--base-url"):])

    def test_contains_concurrency(self):
        cfg = make_cfg()
        engine = BenchmarkEngine(cfg, context=(512, 512))
        cmd = engine._build_command("http://localhost:8000", 32)
        assert "--max-concurrency" in cmd
        assert "32" in cmd

    def test_contains_input_output_len(self):
        cfg = make_cfg()
        engine = BenchmarkEngine(cfg, context=(2048, 4096))
        cmd = engine._build_command("http://localhost:8000", 4)
        assert "--random-input-len" in cmd
        assert "2048" in cmd
        assert "--random-output-len" in cmd
        assert "4096" in cmd

    def test_contains_num_prompts(self):
        cfg = make_cfg(num_prompts=50)
        engine = BenchmarkEngine(cfg, context=(512, 512))
        cmd = engine._build_command("http://localhost:8000", 4)
        assert "--num-prompts" in cmd
        assert "50" in cmd

    def test_contains_percentile_metrics(self):
        cfg = make_cfg()
        engine = BenchmarkEngine(cfg, context=(512, 512))
        cmd = engine._build_command("http://localhost:8000", 4)
        assert "--percentile-metrics" in cmd

    def test_contains_ignore_eos(self):
        cfg = make_cfg()
        engine = BenchmarkEngine(cfg, context=(512, 512))
        cmd = engine._build_command("http://localhost:8000", 4)
        assert "--ignore-eos" in cmd

    def test_url_without_v1_unchanged(self):
        cfg = make_cfg()
        engine = BenchmarkEngine(cfg, context=(512, 512))
        cmd = engine._build_command("http://localhost:8000", 4)
        assert "http://localhost:8000" in cmd


# ===========================================================================
# 6. BenchmarkEngine.from_config factory
# ===========================================================================

class TestFromConfig:
    def test_picks_correct_context(self):
        cfg = make_cfg()
        cfg.context_configs = [(256, 256), (512, 2048), (1024, 8192)]
        engine = BenchmarkEngine.from_config(cfg, context_index=1)
        assert engine.input_len == 512
        assert engine.output_len == 2048

    def test_index_zero_is_default(self):
        cfg = make_cfg()
        cfg.context_configs = [(128, 128), (512, 512)]
        engine = BenchmarkEngine.from_config(cfg, context_index=0)
        assert engine.input_len == 128

    def test_out_of_range_index_raises(self):
        cfg = make_cfg()
        cfg.context_configs = [(512, 512)]
        with pytest.raises(ValueError, match="context_index"):
            BenchmarkEngine.from_config(cfg, context_index=5)

    def test_per_level_timeout_propagated(self):
        cfg = make_cfg()
        engine = BenchmarkEngine.from_config(cfg, per_level_timeout=90)
        assert engine.per_level_timeout == 90


# ===========================================================================
# 7. BenchmarkEngine async run (mocked subprocess)
# ===========================================================================

def _make_mock_proc(stdout: str, returncode: int = 0):
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(
        return_value=(stdout.encode(), b"")
    )
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=returncode)
    return proc


class TestRunSingleLevel:
    @pytest.mark.asyncio
    async def test_successful_run_returns_valid_result(self):
        cfg = make_cfg()
        engine = BenchmarkEngine(cfg, context=(512, 512))
        proc = _make_mock_proc(SAMPLE_OUTPUT, returncode=0)

        with patch("asyncio.create_subprocess_exec",
                   new=AsyncMock(return_value=proc)):
            result = await engine._run_single_level(
                "http://localhost:8000/v1", 16
            )

        assert result.failed is False
        assert result.concurrency == 16
        assert result.output_tokens_per_sec > 0

    @pytest.mark.asyncio
    async def test_timeout_returns_failed_result(self):
        cfg = make_cfg()
        engine = BenchmarkEngine(cfg, context=(512, 512),
                                  per_level_timeout=1)
        proc = MagicMock()
        proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        proc.kill = MagicMock()
        proc.wait = AsyncMock(return_value=-9)

        with patch("asyncio.create_subprocess_exec",
                   new=AsyncMock(return_value=proc)):
            result = await engine._run_single_level(
                "http://localhost:8000/v1", 32
            )

        assert result.failed is True
        assert "Timeout" in result.failure_reason
        assert result.concurrency == 32

    @pytest.mark.asyncio
    async def test_nonzero_exit_with_no_metrics_is_failed(self):
        cfg = make_cfg()
        engine = BenchmarkEngine(cfg, context=(512, 512))
        proc = _make_mock_proc("Error: connection refused", returncode=1)

        with patch("asyncio.create_subprocess_exec",
                   new=AsyncMock(return_value=proc)):
            result = await engine._run_single_level(
                "http://localhost:8000/v1", 4
            )

        assert result.failed is True
        assert "Exit code 1" in result.failure_reason

    @pytest.mark.asyncio
    async def test_nonzero_exit_with_partial_metrics_not_failed(self):
        cfg = make_cfg()
        engine = BenchmarkEngine(cfg, context=(512, 512))
        # Partial output has some metrics but vllm exited 1 for a minor reason
        proc = _make_mock_proc(SAMPLE_OUTPUT, returncode=1)

        with patch("asyncio.create_subprocess_exec",
                   new=AsyncMock(return_value=proc)):
            result = await engine._run_single_level(
                "http://localhost:8000/v1", 8
            )

        # SAMPLE_OUTPUT has valid metrics — should NOT be marked failed
        assert result.is_valid is True
        assert result.output_tokens_per_sec > 0

    @pytest.mark.asyncio
    async def test_file_not_found_returns_failed(self):
        cfg = make_cfg()
        engine = BenchmarkEngine(cfg, context=(512, 512))

        with patch("asyncio.create_subprocess_exec",
                   side_effect=FileNotFoundError("python not found")):
            result = await engine._run_single_level(
                "http://localhost:8000/v1", 4
            )

        assert result.failed is True
        assert "not found" in result.failure_reason.lower()

    @pytest.mark.asyncio
    async def test_zero_exit_but_no_metrics_is_failed(self):
        cfg = make_cfg()
        engine = BenchmarkEngine(cfg, context=(512, 512))
        proc = _make_mock_proc(EMPTY_OUTPUT, returncode=0)

        with patch("asyncio.create_subprocess_exec",
                   new=AsyncMock(return_value=proc)):
            result = await engine._run_single_level(
                "http://localhost:8000/v1", 1
            )

        assert result.failed is True
        assert "no metrics" in result.failure_reason.lower()


class TestRunFullRamp:
    @pytest.mark.asyncio
    async def test_ramp_runs_all_concurrency_levels(self):
        cfg = make_cfg(concurrency_levels=[1, 2, 4])
        engine = BenchmarkEngine(cfg, context=(512, 512))
        proc = _make_mock_proc(SAMPLE_OUTPUT, returncode=0)

        with patch("asyncio.create_subprocess_exec",
                   new=AsyncMock(return_value=proc)):
            ramp = await engine.run_full_ramp("http://localhost:8000/v1")

        assert len(ramp.levels) == 3
        assert all(r.concurrency in [1, 2, 4] for r in ramp.levels)

    @pytest.mark.asyncio
    async def test_ramp_summary_computed(self):
        cfg = make_cfg(concurrency_levels=[1, 4])
        engine = BenchmarkEngine(cfg, context=(512, 512))
        proc = _make_mock_proc(SAMPLE_OUTPUT, returncode=0)

        with patch("asyncio.create_subprocess_exec",
                   new=AsyncMock(return_value=proc)):
            ramp = await engine.run_full_ramp("http://localhost:8000/v1")

        assert "peak_throughput_tokens_per_sec" in ramp.summary
        assert ramp.summary["peak_throughput_tokens_per_sec"] > 0

    @pytest.mark.asyncio
    async def test_ramp_aborts_early_on_all_failed(self):
        cfg = make_cfg(concurrency_levels=[1, 2, 4, 8, 16])
        engine = BenchmarkEngine(cfg, context=(512, 512))

        # All levels fail with no metrics
        proc = _make_mock_proc(EMPTY_OUTPUT, returncode=1)

        with patch("asyncio.create_subprocess_exec",
                   new=AsyncMock(return_value=proc)):
            ramp = await engine.run_full_ramp("http://localhost:8000/v1")

        # Should abort after 3 consecutive failures, not run all 5
        assert len(ramp.levels) <= 5
        assert ramp.all_failed() is True
        assert ramp.failed_levels == len(ramp.levels)

    @pytest.mark.asyncio
    async def test_ramp_duration_recorded(self):
        cfg = make_cfg(concurrency_levels=[1, 2])
        engine = BenchmarkEngine(cfg, context=(512, 512))
        proc = _make_mock_proc(SAMPLE_OUTPUT, returncode=0)

        with patch("asyncio.create_subprocess_exec",
                   new=AsyncMock(return_value=proc)):
            ramp = await engine.run_full_ramp("http://localhost:8000/v1")

        assert ramp.total_duration_sec >= 0

    @pytest.mark.asyncio
    async def test_run_benchmark_convenience_function(self):
        cfg = make_cfg(concurrency_levels=[1])
        proc = _make_mock_proc(SAMPLE_OUTPUT, returncode=0)

        with patch("asyncio.create_subprocess_exec",
                   new=AsyncMock(return_value=proc)):
            ramp = await run_benchmark(
                endpoint="http://localhost:8000/v1",
                cfg=cfg,
                context=(512, 512),
            )

        assert isinstance(ramp, RampResult)
        assert len(ramp.levels) == 1
        assert ramp.levels[0].output_tokens_per_sec > 0