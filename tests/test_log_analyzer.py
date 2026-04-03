"""
tests/test_log_analyzer.py
--------------------------
Unit tests for core/log_analyzer.py.

Run with:
    pytest tests/test_log_analyzer.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from core.log_analyzer import LogAnalyzer, LogAnalysis


# ===========================================================================
# Sample log fixtures
# ===========================================================================

HEALTHY_STARTUP_LOG = """
INFO 2024-01-15 12:00:00.000 vllm.engine Starting vLLM engine...
INFO 2024-01-15 12:00:01.000 vllm.model Loading model weights...
INFO 2024-01-15 12:00:45.500 vllm.model Loading model weights took 44.5s
INFO 2024-01-15 12:00:46.000 vllm.cache # GPU blocks: 8192
INFO 2024-01-15 12:00:46.100 vllm.cache KV cache 12.3 GB allocated
INFO 2024-01-15 12:00:46.200 vllm.cache GPU memory used: 45.2 GB
INFO 2024-01-15 12:00:46.300 vllm.engine max_model_len = 32768
INFO 2024-01-15 12:00:46.400 vllm.server Application startup complete.
INFO 2024-01-15 12:00:46.500 vllm.server Uvicorn running on http://0.0.0.0:8000
""".strip().splitlines()

OOM_LOG = """
INFO 2024-01-15 12:00:00.000 vllm.engine Starting...
INFO 2024-01-15 12:00:30.000 vllm.model Loading model weights took 30.0s
ERROR 2024-01-15 12:00:31.000 vllm.engine CUDA out of memory. Tried to allocate 20 GiB
ERROR 2024-01-15 12:00:31.100 vllm.engine OutOfMemoryError: CUDA out of memory.
""".strip().splitlines()

NCCL_LOG = """
INFO 2024-01-15 12:00:00.000 vllm.engine Starting...
ERROR 2024-01-15 12:00:15.000 vllm.distributed NCCL error: unhandled system error
ERROR 2024-01-15 12:00:15.100 vllm.distributed ncclInternalError: Process group fail
""".strip().splitlines()

PARTIAL_LOG = """
INFO 2024-01-15 12:00:00.000 vllm.engine Starting...
INFO 2024-01-15 12:00:20.000 vllm.model Loading model weights took 20.0s
INFO 2024-01-15 12:00:21.000 vllm.cache # GPU blocks: 4096
""".strip().splitlines()

MULTI_ERROR_LOG = """
INFO 2024-01-15 12:00:00.000 vllm.engine Starting...
ERROR 2024-01-15 12:00:10.000 vllm.quantizer GPTQ error: cannot load weights
ERROR 2024-01-15 12:00:11.000 vllm.engine CUDA error: device-side assert triggered
""".strip().splitlines()

TP_LOG = """
INFO 2024-01-15 12:00:00.000 vllm.engine Starting TP init...
INFO 2024-01-15 12:00:05.000 vllm.distributed Tensor parallel init took 5.0s
INFO 2024-01-15 12:00:06.000 vllm.engine CUDA graph captured in 3.2s
INFO 2024-01-15 12:00:07.000 vllm.server Application startup complete.
""".strip().splitlines()


# ===========================================================================
# Tests: LogAnalyzer.analyze
# ===========================================================================

class TestAnalyzeEmpty:
    def test_empty_list_returns_default(self):
        result = LogAnalyzer.analyze([])
        assert isinstance(result, LogAnalysis)
        assert result.total_log_lines == 0
        assert result.server_became_ready is False
        assert result.error_classes == set()

    def test_empty_returns_none_for_optional_fields(self):
        result = LogAnalyzer.analyze([])
        assert result.model_load_time_sec is None
        assert result.kv_cache_blocks is None


class TestSingleValueExtraction:
    def test_model_load_time_parsed(self):
        result = LogAnalyzer.analyze(HEALTHY_STARTUP_LOG)
        assert result.model_load_time_sec is not None
        assert abs(result.model_load_time_sec - 44.5) < 0.1

    def test_kv_cache_blocks_parsed(self):
        result = LogAnalyzer.analyze(HEALTHY_STARTUP_LOG)
        assert result.kv_cache_blocks == 8192

    def test_kv_cache_gb_parsed(self):
        result = LogAnalyzer.analyze(HEALTHY_STARTUP_LOG)
        assert result.kv_cache_gb is not None
        assert abs(result.kv_cache_gb - 12.3) < 0.1

    def test_gpu_memory_used_parsed(self):
        result = LogAnalyzer.analyze(HEALTHY_STARTUP_LOG)
        assert result.gpu_memory_used_gb is not None
        assert abs(result.gpu_memory_used_gb - 45.2) < 0.1

    def test_max_seq_len_parsed(self):
        result = LogAnalyzer.analyze(HEALTHY_STARTUP_LOG)
        assert result.max_seq_len_from_log == 32768

    def test_tp_init_time_parsed(self):
        result = LogAnalyzer.analyze(TP_LOG)
        assert result.tp_init_time_sec is not None
        assert abs(result.tp_init_time_sec - 5.0) < 0.1

    def test_cuda_graph_time_parsed(self):
        result = LogAnalyzer.analyze(TP_LOG)
        assert result.cuda_graph_capture_sec is not None
        assert abs(result.cuda_graph_capture_sec - 3.2) < 0.1

    def test_partial_log_missing_fields_are_none(self):
        result = LogAnalyzer.analyze(PARTIAL_LOG)
        assert result.model_load_time_sec is not None  # present
        assert result.tp_init_time_sec is None          # not present
        assert result.cuda_graph_capture_sec is None    # not present

    def test_kv_blocks_is_int(self):
        result = LogAnalyzer.analyze(HEALTHY_STARTUP_LOG)
        assert isinstance(result.kv_cache_blocks, int)

    def test_load_time_is_float(self):
        result = LogAnalyzer.analyze(HEALTHY_STARTUP_LOG)
        assert isinstance(result.model_load_time_sec, float)


class TestErrorClassDetection:
    def test_oom_detected(self):
        result = LogAnalyzer.analyze(OOM_LOG)
        assert "oom" in result.error_classes

    def test_oom_property(self):
        result = LogAnalyzer.analyze(OOM_LOG)
        assert result.has_oom is True

    def test_nccl_detected(self):
        result = LogAnalyzer.analyze(NCCL_LOG)
        assert "nccl" in result.error_classes

    def test_nccl_property(self):
        result = LogAnalyzer.analyze(NCCL_LOG)
        assert result.has_nccl_error is True

    def test_quantization_error_detected(self):
        result = LogAnalyzer.analyze(MULTI_ERROR_LOG)
        assert "quantization" in result.error_classes

    def test_cuda_kernel_error_detected(self):
        result = LogAnalyzer.analyze(MULTI_ERROR_LOG)
        assert "cuda_kernel" in result.error_classes

    def test_multiple_errors_all_captured(self):
        result = LogAnalyzer.analyze(MULTI_ERROR_LOG)
        assert len(result.error_classes) >= 2

    def test_healthy_log_no_errors(self):
        result = LogAnalyzer.analyze(HEALTHY_STARTUP_LOG)
        assert result.error_classes == set()
        assert result.has_errors is False

    def test_error_lines_populated_on_error(self):
        result = LogAnalyzer.analyze(OOM_LOG)
        assert len(result.error_lines) > 0
        assert any("out of memory" in line.lower() for line in result.error_lines)

    def test_error_lines_empty_on_clean_log(self):
        result = LogAnalyzer.analyze(HEALTHY_STARTUP_LOG)
        assert result.error_lines == []


class TestReadyDetection:
    def test_ready_detected_application_startup(self):
        result = LogAnalyzer.analyze(HEALTHY_STARTUP_LOG)
        assert result.server_became_ready is True

    def test_ready_detected_uvicorn(self):
        lines = ["INFO Uvicorn running on http://0.0.0.0:8000"]
        result = LogAnalyzer.analyze(lines)
        assert result.server_became_ready is True

    def test_not_ready_when_crashed(self):
        result = LogAnalyzer.analyze(OOM_LOG)
        assert result.server_became_ready is False

    def test_not_ready_partial_log(self):
        result = LogAnalyzer.analyze(PARTIAL_LOG)
        assert result.server_became_ready is False


class TestStartupTimeEstimate:
    def test_startup_time_computed_from_timestamps(self):
        result = LogAnalyzer.analyze(HEALTHY_STARTUP_LOG)
        # From 12:00:00 to 12:00:46.5 = 46.5s
        assert result.total_startup_sec is not None
        assert result.total_startup_sec > 0
        assert result.total_startup_sec < 120

    def test_startup_time_none_without_timestamps(self):
        lines = ["Starting...", "Loaded.", "Ready."]
        result = LogAnalyzer.analyze(lines)
        assert result.total_startup_sec is None

    def test_startup_time_none_with_single_timestamp(self):
        lines = ["INFO 2024-01-15 12:00:00.000 only one timestamp"]
        result = LogAnalyzer.analyze(lines)
        assert result.total_startup_sec is None


class TestOutputShape:
    def test_to_dict_keys(self):
        result = LogAnalyzer.analyze(HEALTHY_STARTUP_LOG)
        d = result.to_dict()
        assert "model_load_time_sec" in d
        assert "kv_cache_blocks" in d
        assert "server_became_ready" in d
        assert "error_classes" in d
        assert isinstance(d["error_classes"], list)

    def test_total_log_lines_correct(self):
        result = LogAnalyzer.analyze(HEALTHY_STARTUP_LOG)
        assert result.total_log_lines == len(HEALTHY_STARTUP_LOG)

    def test_log_tail_snippet_max_20(self):
        long_log = [f"INFO line {i}" for i in range(100)]
        result = LogAnalyzer.analyze(long_log)
        assert len(result.log_tail_snippet) == 20

    def test_log_tail_snippet_is_last_lines(self):
        lines = [f"line {i}" for i in range(30)]
        result = LogAnalyzer.analyze(lines)
        assert result.log_tail_snippet[0] == "line 10"
        assert result.log_tail_snippet[-1] == "line 29"

    def test_analyze_never_raises_on_garbage(self):
        garbage = ["\x00\x01\x02", "���", "None", "", "   "]
        result = LogAnalyzer.analyze(garbage)
        assert isinstance(result, LogAnalysis)


class TestAnalyzeFile:
    def test_nonexistent_file_returns_empty(self, tmp_path):
        result = LogAnalyzer.analyze_file(str(tmp_path / "nonexistent.log"))
        assert isinstance(result, LogAnalysis)
        assert result.total_log_lines == 0

    def test_file_analysis_matches_line_analysis(self, tmp_path):
        log_file = tmp_path / "test.log"
        log_file.write_text("\n".join(HEALTHY_STARTUP_LOG))
        from_file = LogAnalyzer.analyze_file(str(log_file))
        from_lines = LogAnalyzer.analyze(HEALTHY_STARTUP_LOG)
        assert from_file.server_became_ready == from_lines.server_became_ready
        assert from_file.error_classes == from_lines.error_classes