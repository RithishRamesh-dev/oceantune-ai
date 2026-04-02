"""
tests/test_vllm_server.py
--------------------------
Unit tests for core/vllm_server.py.

No real GPU or vLLM installation required — all subprocess and HTTP
calls are mocked.

Run with:
    pytest tests/test_vllm_server.py -v
"""

from __future__ import annotations

import asyncio
import sys
from collections import deque
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.vllm_server import (
    VLLMServer,
    OOMError,
    StartupTimeout,
    PortConflict,
    CUDAError,
    ProcessCrash,
    ServerFailure,
    ServerState,
    _classify_log_failure,
    make_server,
)
from core.search_space import VLLMFlags


# ===========================================================================
# Helpers
# ===========================================================================

def make_flags(**kwargs) -> VLLMFlags:
    f = VLLMFlags(**kwargs)
    f.run_id = f.fingerprint()
    return f


def make_server_instance(**kwargs) -> VLLMServer:
    defaults = dict(
        model_id="test/model",
        flags=make_flags(),
        port=9999,
        startup_timeout=5,
        gpu_type="H100",
        hf_token="test-token",
    )
    defaults.update(kwargs)
    return VLLMServer(**defaults)


def _make_mock_process(returncode=None, stdout_lines=None):
    """Build an asyncio.Process mock that streams given lines then EOF."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.pid = 12345

    if stdout_lines is not None:
        encoded = [line.encode() + b"\n" for line in stdout_lines]

        async def _aiter():
            for line in encoded:
                yield line

        mock_stdout = MagicMock()
        mock_stdout.__aiter__ = lambda self: _aiter()
        proc.stdout = mock_stdout
    else:
        proc.stdout = None

    proc.wait = AsyncMock(return_value=0)
    return proc


# ===========================================================================
# 1. Failure classification
# ===========================================================================

class TestClassifyLogFailure:
    def test_oom_detected(self):
        lines = ["some startup", "CUDA out of memory, tried to allocate 20 GiB"]
        result = _classify_log_failure(lines)
        assert isinstance(result, OOMError)

    def test_oom_detected_capitalised(self):
        lines = ["RuntimeError: CUDA out of memory"]
        assert isinstance(_classify_log_failure(lines), OOMError)

    def test_oom_detected_oom_keyword(self):
        lines = ["torch.cuda.OutOfMemoryError: OOM when allocating"]
        assert isinstance(_classify_log_failure(lines), OOMError)

    def test_port_conflict_detected(self):
        lines = ["OSError: [Errno 98] Address already in use"]
        assert isinstance(_classify_log_failure(lines), PortConflict)

    def test_cuda_error_detected(self):
        lines = ["CUDA error: device-side assert triggered"]
        assert isinstance(_classify_log_failure(lines), CUDAError)

    def test_no_failure_returns_none(self):
        lines = ["INFO: Loading model weights", "INFO: Model loaded successfully"]
        assert _classify_log_failure(lines) is None

    def test_empty_log_returns_none(self):
        assert _classify_log_failure([]) is None

    def test_oom_takes_priority_over_cuda(self):
        lines = ["CUDA error: ...", "CUDA out of memory"]
        result = _classify_log_failure(lines)
        assert isinstance(result, OOMError)


# ===========================================================================
# 2. VLLMServer properties
# ===========================================================================

class TestVLLMServerProperties:
    def test_endpoint(self):
        s = make_server_instance(port=8001)
        assert s.endpoint == "http://localhost:8001/v1"

    def test_health_url(self):
        s = make_server_instance(port=8001)
        assert s.health_url == "http://localhost:8001/health"

    def test_initial_state_stopped(self):
        s = make_server_instance()
        assert s.state == ServerState.STOPPED

    def test_is_alive_false_when_no_process(self):
        s = make_server_instance()
        assert not s.is_alive()

    def test_log_tail_empty_initially(self):
        s = make_server_instance()
        assert s.log_tail == []

    def test_diagnostic_report_keys(self):
        s = make_server_instance()
        report = s.diagnostic_report()
        assert "run_id" in report
        assert "state" in report
        assert "port" in report
        assert "log_tail" in report


# ===========================================================================
# 3. _build_command
# ===========================================================================

class TestBuildCommand:
    def test_contains_model(self):
        s = make_server_instance(model_id="mistralai/Mistral-7B")
        cmd = s._build_command()
        assert "mistralai/Mistral-7B" in cmd

    def test_contains_port(self):
        s = make_server_instance(port=7777)
        cmd = s._build_command()
        assert "--port" in cmd
        assert "7777" in cmd

    def test_contains_host(self):
        s = make_server_instance(host="127.0.0.1")
        cmd = s._build_command()
        assert "--host" in cmd
        assert "127.0.0.1" in cmd

    def test_starts_with_python_m(self):
        s = make_server_instance()
        cmd = s._build_command()
        assert cmd[0] == "python3"
        assert cmd[1] == "-m"

    def test_tensor_parallel_in_args(self):
        flags = make_flags(tensor_parallel_size=4)
        s = make_server_instance(flags=flags)
        cmd = s._build_command()
        assert "--tensor-parallel-size" in cmd
        assert "4" in cmd


# ===========================================================================
# 4. _build_env
# ===========================================================================

class TestBuildEnv:
    def test_hf_token_set(self):
        s = make_server_instance(hf_token="mytoken")
        env = s._build_env()
        assert env["HUGGING_FACE_HUB_TOKEN"] == "mytoken"
        assert env["HF_TOKEN"] == "mytoken"

    def test_tokenizers_parallelism_disabled(self):
        s = make_server_instance()
        env = s._build_env()
        assert env["TOKENIZERS_PARALLELISM"] == "false"

    def test_nccl_debug_set(self):
        s = make_server_instance()
        env = s._build_env()
        assert "NCCL_DEBUG" in env

    def test_empty_hf_token_not_set(self):
        s = make_server_instance(hf_token="")
        env = s._build_env()
        assert env.get("HUGGING_FACE_HUB_TOKEN", "") == ""

    def test_amd_env_for_mi300x(self):
        s = make_server_instance(gpu_type="MI300X")
        env = s._build_env()
        # Profile-driven ROCm vars should be present
        assert env.get("VLLM_ROCM_USE_AITER") == "1"
        assert env.get("TOKENIZERS_PARALLELISM") == "false"


# ===========================================================================
# 5. Health check (mocked HTTP)
# ===========================================================================

class TestIsHealthy:
    @pytest.mark.asyncio
    async def test_healthy_on_200(self):
        s = make_server_instance()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("core.vllm_server.httpx.AsyncClient", return_value=mock_client):
            result = await s.is_healthy()
        assert result is True

    @pytest.mark.asyncio
    async def test_unhealthy_on_503(self):
        s = make_server_instance()
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("core.vllm_server.httpx.AsyncClient", return_value=mock_client):
            result = await s.is_healthy()
        assert result is False

    @pytest.mark.asyncio
    async def test_unhealthy_on_connection_error(self):
        s = make_server_instance()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("core.vllm_server.httpx.AsyncClient", return_value=mock_client):
            result = await s.is_healthy()
        assert result is False


# ===========================================================================
# 6. Log capture
# ===========================================================================

class TestLogCapture:
    @pytest.mark.asyncio
    async def test_lines_written_to_buffer(self):
        s = make_server_instance()
        lines = ["Starting model load...", "Model loaded.", "Server ready."]
        s._process = _make_mock_process(stdout_lines=lines)
        await s._capture_logs()
        assert list(s._log_buffer) == lines

    @pytest.mark.asyncio
    async def test_buffer_respects_maxlen(self):
        s = make_server_instance(log_buffer_size=3)
        s._log_buffer = deque(maxlen=3)
        lines = ["line1", "line2", "line3", "line4", "line5"]
        s._process = _make_mock_process(stdout_lines=lines)
        await s._capture_logs()
        assert list(s._log_buffer) == ["line3", "line4", "line5"]

    @pytest.mark.asyncio
    async def test_no_process_returns_immediately(self):
        s = make_server_instance()
        s._process = None
        await s._capture_logs()
        assert s.log_tail == []


# ===========================================================================
# 7. Start / stop integration (fully mocked)
# ===========================================================================

class TestStartStop:
    @pytest.mark.asyncio
    async def test_start_reaches_healthy_state(self):
        s = make_server_instance(startup_timeout=10)
        proc = _make_mock_process(returncode=None, stdout_lines=[])

        async def fake_subprocess(*a, **kw):
            return proc

        call_count = 0
        async def fake_healthy():
            nonlocal call_count
            call_count += 1
            return call_count >= 2

        with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess), \
             patch.object(s, "is_healthy", side_effect=fake_healthy), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            await s.start()

        assert s.state == ServerState.HEALTHY

    @pytest.mark.asyncio
    async def test_stop_changes_state_to_stopped(self):
        s = make_server_instance(startup_timeout=10)
        proc = _make_mock_process(returncode=None, stdout_lines=[])

        async def fake_subprocess(*a, **kw):
            return proc

        async def fake_healthy():
            return True

        with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess), \
             patch.object(s, "is_healthy", side_effect=fake_healthy), \
             patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("os.getpgid", return_value=12345), \
             patch("os.killpg"):
            await s.start()
            assert s.state == ServerState.HEALTHY
            await s.stop()

        assert s.state == ServerState.STOPPED

    @pytest.mark.asyncio
    async def test_double_stop_is_safe(self):
        s = make_server_instance()
        s._state = ServerState.STOPPED
        await s.stop()
        assert s.state == ServerState.STOPPED

    @pytest.mark.asyncio
    async def test_oom_in_logs_raises_oom_error(self):
        s = make_server_instance(startup_timeout=2)
        oom_lines = ["Loading weights...", "CUDA out of memory, tried to allocate 40 GiB"]
        proc = _make_mock_process(returncode=None, stdout_lines=oom_lines)

        async def fake_subprocess(*a, **kw):
            s._log_buffer.extend(oom_lines)
            return proc

        async def fake_healthy():
            return False

        with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess), \
             patch.object(s, "is_healthy", side_effect=fake_healthy), \
             patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("os.getpgid", return_value=12345), \
             patch("os.killpg"):
            with pytest.raises(OOMError):
                await s.start()

    @pytest.mark.asyncio
    async def test_port_conflict_in_logs_raises_port_conflict(self):
        s = make_server_instance(startup_timeout=2)
        conflict_lines = ["OSError: [Errno 98] Address already in use"]
        proc = _make_mock_process(returncode=None, stdout_lines=[])

        async def fake_subprocess(*a, **kw):
            s._log_buffer.extend(conflict_lines)
            return proc

        async def fake_healthy():
            return False

        with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess), \
             patch.object(s, "is_healthy", side_effect=fake_healthy), \
             patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("os.getpgid", return_value=12345), \
             patch("os.killpg"):
            with pytest.raises(PortConflict):
                await s.start()

    @pytest.mark.asyncio
    async def test_process_crash_raises(self):
        s = make_server_instance(startup_timeout=2)
        proc = _make_mock_process(returncode=1, stdout_lines=[])

        async def fake_subprocess(*a, **kw):
            return proc

        async def fake_healthy():
            return False

        with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess), \
             patch.object(s, "is_healthy", side_effect=fake_healthy), \
             patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("os.getpgid", return_value=12345), \
             patch("os.killpg"):
            with pytest.raises((ProcessCrash, OOMError, PortConflict, CUDAError, StartupTimeout)):
                await s.start()

    @pytest.mark.asyncio
    async def test_timeout_raises_startup_timeout(self):
        s = make_server_instance(startup_timeout=0)
        proc = _make_mock_process(returncode=None, stdout_lines=[])

        async def fake_subprocess(*a, **kw):
            return proc

        async def fake_healthy():
            return False

        with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess), \
             patch.object(s, "is_healthy", side_effect=fake_healthy), \
             patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("os.getpgid", return_value=12345), \
             patch("os.killpg"):
            with pytest.raises((StartupTimeout, ProcessCrash)):
                await s.start()

    @pytest.mark.asyncio
    async def test_vllm_not_installed_raises_server_failure(self):
        s = make_server_instance()

        async def fake_subprocess(*a, **kw):
            raise FileNotFoundError("python not found")

        with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess):
            with pytest.raises(ServerFailure):
                await s.start()

    @pytest.mark.asyncio
    async def test_context_manager_stops_on_exit(self):
        s = make_server_instance(startup_timeout=10)
        proc = _make_mock_process(returncode=None, stdout_lines=[])

        async def fake_subprocess(*a, **kw):
            return proc

        async def fake_healthy():
            return True

        with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess), \
             patch.object(s, "is_healthy", side_effect=fake_healthy), \
             patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("os.getpgid", return_value=12345), \
             patch("os.killpg"):
            async with s:
                assert s.state == ServerState.HEALTHY
            assert s.state == ServerState.STOPPED

    @pytest.mark.asyncio
    async def test_context_manager_stops_on_exception(self):
        s = make_server_instance(startup_timeout=10)
        proc = _make_mock_process(returncode=None, stdout_lines=[])

        async def fake_subprocess(*a, **kw):
            return proc

        async def fake_healthy():
            return True

        with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess), \
             patch.object(s, "is_healthy", side_effect=fake_healthy), \
             patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("os.getpgid", return_value=12345), \
             patch("os.killpg"):
            with pytest.raises(ValueError):
                async with s:
                    raise ValueError("something went wrong inside the block")

        assert s.state == ServerState.STOPPED


# ===========================================================================
# 8. make_server factory
# ===========================================================================

class TestMakeServer:
    def test_factory_uses_cfg_values(self):
        cfg = MagicMock()
        cfg.model_id = "some/model"
        cfg.gpu_type = "H100"
        cfg.hf_token = "tok"
        cfg.vllm.host = "0.0.0.0"
        cfg.vllm.port = 8080
        cfg.vllm.startup_timeout_sec = 120

        flags = make_flags()
        server = make_server(cfg, flags)

        assert server.model_id == "some/model"
        assert server.gpu_type == "H100"
        assert server.port == 8080
        assert server.startup_timeout == 120

    def test_factory_port_override(self):
        cfg = MagicMock()
        cfg.model_id = "m"
        cfg.gpu_type = "H100"
        cfg.hf_token = ""
        cfg.vllm.host = "0.0.0.0"
        cfg.vllm.port = 8000
        cfg.vllm.startup_timeout_sec = 60

        flags = make_flags()
        server = make_server(cfg, flags, port=9001)
        assert server.port == 9001


# ===========================================================================
# 9. ServerFailure exception hierarchy
# ===========================================================================

class TestExceptionHierarchy:
    def test_oom_is_server_failure(self):
        assert issubclass(OOMError, ServerFailure)

    def test_startup_timeout_is_server_failure(self):
        assert issubclass(StartupTimeout, ServerFailure)

    def test_port_conflict_is_server_failure(self):
        assert issubclass(PortConflict, ServerFailure)

    def test_cuda_error_is_server_failure(self):
        assert issubclass(CUDAError, ServerFailure)

    def test_process_crash_is_server_failure(self):
        assert issubclass(ProcessCrash, ServerFailure)

    def test_formatted_tail_empty(self):
        exc = OOMError("oom", log_tail=[])
        assert exc.formatted_tail() == "(no log captured)"

    def test_formatted_tail_truncates(self):
        lines = [f"line {i}" for i in range(100)]
        exc = OOMError("oom", log_tail=lines)
        tail = exc.formatted_tail(n=5)
        assert tail.count("\n") == 4
        assert "line 99" in tail
