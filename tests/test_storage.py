"""
tests/test_storage.py
---------------------
Unit tests for core/storage.py.

Run with:
    pytest tests/test_storage.py -v
"""

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from core.storage import (
    RunRecord,
    ResultStorage,
    ResultLoader,
    make_run_record,
    _get_git_hash,
    _extract_failure_reason,
)
from core.benchmark_runner import RampResult, BenchmarkResult
from core.log_analyzer import LogAnalysis
from core.metrics_collector import EnrichedMetrics
from core.search_space import VLLMFlags
from core.config import OceanTuneConfig


# ===========================================================================
# Helpers
# ===========================================================================

def make_cfg(gpu_type="H100") -> OceanTuneConfig:
    cfg = OceanTuneConfig()
    cfg.model_id = "test/model"
    cfg.gpu_type = gpu_type
    return cfg


def make_flags(run_id="abc123def456") -> VLLMFlags:
    f = VLLMFlags(tensor_parallel_size=2, gpu_memory_utilization=0.90)
    f.run_id = run_id
    return f


def make_ramp(valid=True) -> RampResult:
    tp = 2000.0 if valid else 0.0
    r = BenchmarkResult(
        concurrency=8,
        output_tokens_per_sec=tp,
        requests_per_sec=tp / 150,
        p95_latency_ms=500.0,
        mean_ttft_ms=80.0,
        failed=not valid,
    )
    summary = {
        "peak_throughput_tokens_per_sec": tp,
        "peak_requests_per_sec": tp / 150,
        "p95_latency_at_peak_ms": 500.0,
        "p99_latency_at_peak_ms": 600.0,
        "mean_ttft_ms": 80.0,
        "p95_ttft_ms": 120.0,
        "mean_tpot_ms": 5.0,
        "p95_tpot_ms": 7.0,
        "mean_itl_ms": 5.0,
        "error_rate_max": 0.0,
        "valid_levels": 1.0 if valid else 0.0,
        "best_concurrency": 8.0 if valid else 0.0,
    }
    return RampResult(
        endpoint="http://localhost:8000/v1",
        input_len=1024,
        output_len=1024,
        levels=[r],
        summary=summary,
        total_duration_sec=60.0,
        failed_levels=0 if valid else 1,
    )


def make_analysis(has_oom=False) -> LogAnalysis:
    return LogAnalysis(
        model_load_time_sec=40.0,
        kv_cache_blocks=8192,
        kv_cache_gb=12.0,
        gpu_memory_used_gb=45.0,
        server_became_ready=True,
        total_log_lines=100,
        error_classes={"oom"} if has_oom else set(),
    )


def make_enriched(score=0.75) -> EnrichedMetrics:
    return EnrichedMetrics(
        peak_throughput_tokens_per_sec=2000.0,
        p95_latency_at_peak_ms=500.0,
        mean_ttft_ms=80.0,
        valid_levels=4,
        fitness_score=score,
        all_levels_failed=False,
    )


# ===========================================================================
# Tests: RunRecord
# ===========================================================================

class TestRunRecord:
    def test_to_dict_contains_all_top_level_fields(self):
        record = RunRecord(
            run_id="abc123",
            model_id="test/model",
            gpu_type="H100",
        )
        d = record.to_dict()
        assert d["run_id"] == "abc123"
        assert d["model_id"] == "test/model"
        assert d["gpu_type"] == "H100"

    def test_to_csv_row_flattens_flags(self):
        record = RunRecord(
            run_id="abc123",
            flags={"tensor_parallel_size": 4, "dtype": "bfloat16"},
        )
        row = record.to_csv_row()
        assert row["flag__tensor_parallel_size"] == 4
        assert row["flag__dtype"] == "bfloat16"

    def test_to_csv_row_flattens_benchmark_summary(self):
        record = RunRecord(
            run_id="abc123",
            benchmark_summary={"peak_throughput_tokens_per_sec": 2000.0},
        )
        row = record.to_csv_row()
        assert row["bench__peak_throughput_tokens_per_sec"] == 2000.0

    def test_to_csv_row_flattens_log_fields(self):
        record = RunRecord(
            run_id="abc123",
            log_analysis={
                "model_load_time_sec": 40.0,
                "kv_cache_blocks": 8192,
                "has_oom": False,
            },
        )
        row = record.to_csv_row()
        assert row["log__model_load_time_sec"] == 40.0
        assert row["log__kv_cache_blocks"] == 8192
        assert row["log__has_oom"] is False

    def test_to_csv_row_has_required_identity_fields(self):
        record = RunRecord(run_id="xyz789", model_id="m", gpu_type="H100")
        row = record.to_csv_row()
        for key in ("run_id", "model_id", "gpu_type", "fitness_score",
                    "run_successful", "input_len", "output_len"):
            assert key in row


# ===========================================================================
# Tests: make_run_record factory
# ===========================================================================

class TestMakeRunRecord:
    def test_run_id_from_flags(self):
        flags = make_flags(run_id="deadbeef1234")
        record = make_run_record(
            flags=flags, ramp=make_ramp(), analysis=make_analysis(),
            enriched=make_enriched(), cfg=make_cfg(),
        )
        assert record.run_id == "deadbeef1234"

    def test_model_id_from_cfg(self):
        cfg = make_cfg()
        record = make_run_record(
            flags=make_flags(), ramp=make_ramp(), analysis=make_analysis(),
            enriched=make_enriched(), cfg=cfg,
        )
        assert record.model_id == "test/model"

    def test_gpu_type_from_cfg(self):
        record = make_run_record(
            flags=make_flags(), ramp=make_ramp(), analysis=make_analysis(),
            enriched=make_enriched(), cfg=make_cfg(gpu_type="MI300X"),
        )
        assert record.gpu_type == "MI300X"

    def test_context_label_formatted(self):
        ramp = make_ramp()
        ramp.input_len = 1024
        ramp.output_len = 4096
        record = make_run_record(
            flags=make_flags(), ramp=ramp, analysis=make_analysis(),
            enriched=make_enriched(), cfg=make_cfg(),
        )
        assert record.context_label == "1K/4K"

    def test_fitness_score_from_enriched(self):
        record = make_run_record(
            flags=make_flags(), ramp=make_ramp(), analysis=make_analysis(),
            enriched=make_enriched(score=0.88), cfg=make_cfg(),
        )
        assert abs(record.fitness_score - 0.88) < 1e-6

    def test_run_successful_true_for_valid_run(self):
        record = make_run_record(
            flags=make_flags(), ramp=make_ramp(valid=True),
            analysis=make_analysis(), enriched=make_enriched(score=0.7),
            cfg=make_cfg(),
        )
        assert record.run_successful is True

    def test_failure_reason_for_oom(self):
        record = make_run_record(
            flags=make_flags(), ramp=make_ramp(valid=False),
            analysis=make_analysis(has_oom=True), enriched=make_enriched(score=0.0),
            cfg=make_cfg(),
        )
        assert "OOM" in record.failure_reason or "oom" in record.failure_reason.lower()

    def test_benchmark_levels_stored(self):
        record = make_run_record(
            flags=make_flags(), ramp=make_ramp(), analysis=make_analysis(),
            enriched=make_enriched(), cfg=make_cfg(),
        )
        assert len(record.benchmark_levels) == 1
        assert "output_tokens_per_sec" in record.benchmark_levels[0]

    def test_generation_stored(self):
        record = make_run_record(
            flags=make_flags(), ramp=make_ramp(), analysis=make_analysis(),
            enriched=make_enriched(), cfg=make_cfg(), generation=3,
        )
        assert record.generation == 3

    def test_session_id_stored(self):
        record = make_run_record(
            flags=make_flags(), ramp=make_ramp(), analysis=make_analysis(),
            enriched=make_enriched(), cfg=make_cfg(), session_id="sess-001",
        )
        assert record.session_id == "sess-001"

    def test_git_commit_is_string(self):
        record = make_run_record(
            flags=make_flags(), ramp=make_ramp(), analysis=make_analysis(),
            enriched=make_enriched(), cfg=make_cfg(),
        )
        assert isinstance(record.git_commit, str)
        assert len(record.git_commit) > 0


# ===========================================================================
# Tests: ResultStorage
# ===========================================================================

class TestResultStorage:
    def test_save_creates_json_file(self, tmp_path):
        cfg = make_cfg()
        storage = ResultStorage(cfg, results_dir=tmp_path / "results")
        record = make_run_record(
            flags=make_flags(run_id="testrun001"), ramp=make_ramp(),
            analysis=make_analysis(), enriched=make_enriched(), cfg=cfg,
        )
        storage.save(record)
        json_path = storage.runs_dir / "testrun001.json"
        assert json_path.exists()

    def test_json_file_is_valid(self, tmp_path):
        cfg = make_cfg()
        storage = ResultStorage(cfg, results_dir=tmp_path / "results")
        record = make_run_record(
            flags=make_flags(run_id="testrun002"), ramp=make_ramp(),
            analysis=make_analysis(), enriched=make_enriched(), cfg=cfg,
        )
        storage.save(record)
        json_path = storage.runs_dir / "testrun002.json"
        with open(json_path) as f:
            data = json.load(f)
        assert data["run_id"] == "testrun002"

    def test_save_creates_csv_file(self, tmp_path):
        cfg = make_cfg()
        storage = ResultStorage(cfg, results_dir=tmp_path / "results")
        record = make_run_record(
            flags=make_flags(run_id="csvtest001"), ramp=make_ramp(),
            analysis=make_analysis(), enriched=make_enriched(), cfg=cfg,
        )
        storage.save(record)
        assert storage.csv_path.exists()

    def test_csv_has_header_and_data_row(self, tmp_path):
        cfg = make_cfg()
        storage = ResultStorage(cfg, results_dir=tmp_path / "results")
        record = make_run_record(
            flags=make_flags(run_id="csvtest002"), ramp=make_ramp(),
            analysis=make_analysis(), enriched=make_enriched(), cfg=cfg,
        )
        storage.save(record)
        with open(storage.csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["run_id"] == "csvtest002"

    def test_multiple_saves_append_rows(self, tmp_path):
        cfg = make_cfg()
        storage = ResultStorage(cfg, results_dir=tmp_path / "results")
        for i in range(3):
            record = make_run_record(
                flags=make_flags(run_id=f"run{i:04d}"), ramp=make_ramp(),
                analysis=make_analysis(), enriched=make_enriched(), cfg=cfg,
            )
            storage.save(record)
        with open(storage.csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3

    def test_spaces_skipped_without_credentials(self, tmp_path):
        cfg = make_cfg()
        cfg.spaces.access_key = ""
        cfg.spaces.secret_key = ""
        storage = ResultStorage(cfg, results_dir=tmp_path / "results")
        record = make_run_record(
            flags=make_flags(run_id="noupload"), ramp=make_ramp(),
            analysis=make_analysis(), enriched=make_enriched(), cfg=cfg,
        )
        # Should not raise even without credentials
        storage.save(record)
        assert (storage.runs_dir / "noupload.json").exists()


# ===========================================================================
# Tests: ResultLoader
# ===========================================================================

class TestResultLoader:
    def test_load_csv_empty_when_no_file(self, tmp_path):
        rows = ResultLoader.load_csv(tmp_path / "nonexistent.csv")
        assert rows == []

    def test_load_csv_returns_all_rows(self, tmp_path):
        cfg = make_cfg()
        storage = ResultStorage(cfg, results_dir=tmp_path / "results")
        for i in range(4):
            record = make_run_record(
                flags=make_flags(run_id=f"hist{i:03d}"), ramp=make_ramp(),
                analysis=make_analysis(), enriched=make_enriched(score=i * 0.2),
                cfg=cfg,
            )
            storage.save(record)
        rows = ResultLoader.load_csv(storage.csv_path)
        assert len(rows) == 4

    def test_get_best_runs_returns_top_n(self, tmp_path):
        cfg = make_cfg()
        storage = ResultStorage(cfg, results_dir=tmp_path / "results")
        scores = [0.9, 0.3, 0.7, 0.1, 0.6]
        for i, score in enumerate(scores):
            record = make_run_record(
                flags=make_flags(run_id=f"score{i:03d}"),
                ramp=make_ramp(),
                analysis=make_analysis(),
                enriched=make_enriched(score=score),
                cfg=cfg,
            )
            storage.save(record)
        best = ResultLoader.get_best_runs(storage.csv_path, n=3)
        assert len(best) == 3
        fitness_scores = [float(r["fitness_score"]) for r in best]
        assert fitness_scores[0] >= fitness_scores[1] >= fitness_scores[2]
        assert abs(fitness_scores[0] - 0.9) < 1e-6

    def test_get_best_runs_empty_when_no_csv(self, tmp_path):
        best = ResultLoader.get_best_runs(tmp_path / "none.csv", n=5)
        assert best == []

    def test_get_failed_fingerprints(self, tmp_path):
        cfg = make_cfg()
        storage = ResultStorage(cfg, results_dir=tmp_path / "results")
        # One successful, one failed
        for run_id, valid in [("good001", True), ("bad001", False)]:
            record = make_run_record(
                flags=make_flags(run_id=run_id),
                ramp=make_ramp(valid=valid),
                analysis=make_analysis(),
                enriched=make_enriched(score=0.8 if valid else 0.0),
                cfg=cfg,
            )
            record.run_successful = valid
            storage._save_json(record)
            storage._append_csv(record)
        failed = ResultLoader.get_failed_fingerprints(storage.csv_path)
        assert "bad001" in failed
        assert "good001" not in failed

    def test_load_json_returns_none_for_missing(self, tmp_path):
        result = ResultLoader.load_json("nonexistent_id", runs_dir=tmp_path)
        assert result is None

    def test_load_json_returns_dict_for_existing(self, tmp_path):
        cfg = make_cfg()
        storage = ResultStorage(cfg, results_dir=tmp_path / "results")
        record = make_run_record(
            flags=make_flags(run_id="jsonload001"), ramp=make_ramp(),
            analysis=make_analysis(), enriched=make_enriched(), cfg=cfg,
        )
        storage.save(record)
        data = ResultLoader.load_json("jsonload001", runs_dir=storage.runs_dir)
        assert data is not None
        assert data["run_id"] == "jsonload001"


# ===========================================================================
# Tests: Helpers
# ===========================================================================

class TestHelpers:
    def test_get_git_hash_returns_string(self):
        result = _get_git_hash()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_extract_failure_reason_oom(self):
        analysis = make_analysis(has_oom=True)
        ramp = make_ramp(valid=False)
        reason = _extract_failure_reason(ramp, analysis)
        assert "OOM" in reason or "oom" in reason.lower()

    def test_extract_failure_reason_empty_on_success(self):
        analysis = make_analysis(has_oom=False)
        ramp = make_ramp(valid=True)
        reason = _extract_failure_reason(ramp, analysis)
        assert reason == ""