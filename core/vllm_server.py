"""
core/vllm_server.py
-------------------
Manages the full lifecycle of a vLLM server process for OceanTune AI.

Responsibilities
----------------
- Launch vLLM as a subprocess with flags from VLLMFlags.to_vllm_args()
- Stream stdout/stderr into a rotating in-memory log buffer (for OOM detection)
- Poll GET /health until the server is ready or a timeout is exceeded
- Classify failures: OOM, startup timeout, port conflict, CUDA error
- Tear down cleanly: SIGTERM → grace period → SIGKILL

Usage (async context manager — preferred)
------------------------------------------
    async with VLLMServer(cfg, flags) as server:
        endpoint = server.endpoint   # "http://localhost:8000/v1"
        ...
    # server is stopped and cleaned up here

Usage (manual)
--------------
    server = VLLMServer(cfg, flags)
    try:
        await server.start()
        ...
    finally:
        await server.stop()

Design notes
------------
- All I/O is async (asyncio) so the benchmark engine can read from the server
  while the log-capture task runs concurrently.
- Log lines are stored in a deque(maxlen=500) — enough to diagnose failures
  without unbounded memory growth across thousands of experiments.
- The health poller uses exponential backoff (capped at 10s) to avoid
  hammering the server during slow model load.
- OOM and CUDA errors are detected by scanning the log buffer with compiled
  regex patterns — this is faster than string.find() at high log volume.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List, Optional

import httpx
import yaml

from core.logger import get_logger, log_dict
from core.search_space import VLLMFlags, _AMD_GPU_TYPES

log = get_logger("core.vllm_server")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_VLLM_SH = REPO_ROOT / "scripts" / "run_vllm.sh"
GPU_PROFILES_YAML = REPO_ROOT / "configs" / "gpu_profiles.yaml"


def _load_gpu_profile(gpu_type: str) -> dict:
    """Load a single GPU profile from gpu_profiles.yaml."""
    if not GPU_PROFILES_YAML.exists():
        return {}
    with open(GPU_PROFILES_YAML, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw.get("gpu_profiles", {}).get(gpu_type, {})

# ---------------------------------------------------------------------------
# Failure hierarchy
# ---------------------------------------------------------------------------

class ServerFailure(Exception):
    """Base class for all vLLM server failures."""
    def __init__(self, message: str, log_tail: List[str] | None = None):
        super().__init__(message)
        self.log_tail = log_tail or []

    def formatted_tail(self, n: int = 20) -> str:
        lines = self.log_tail[-n:]
        return "\n".join(lines) if lines else "(no log captured)"


class OOMError(ServerFailure):
    """CUDA out-of-memory — reduce gpu_memory_utilization or model size."""


class StartupTimeout(ServerFailure):
    """Server process started but /health never returned 200."""


class PortConflict(ServerFailure):
    """The requested port is already bound by another process."""


class CUDAError(ServerFailure):
    """Generic CUDA runtime error (not OOM)."""


class ProcessCrash(ServerFailure):
    """The server process exited with a non-zero return code."""


# ---------------------------------------------------------------------------
# Log-scanning patterns
# ---------------------------------------------------------------------------
_OOM_PATTERNS = [
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"OutOfMemoryError", re.IGNORECASE),
    re.compile(r"out of memory", re.IGNORECASE),
    re.compile(r"\bOOM\b", re.IGNORECASE),
    re.compile(r"RuntimeError.*memory", re.IGNORECASE),
]

_CUDA_ERROR_PATTERNS = [
    re.compile(r"CUDA error", re.IGNORECASE),
    re.compile(r"CUDAError", re.IGNORECASE),
    re.compile(r"cudaCheckError", re.IGNORECASE),
    re.compile(r"device-side assert", re.IGNORECASE),
]

_PORT_CONFLICT_PATTERNS = [
    re.compile(r"address already in use", re.IGNORECASE),
    re.compile(r"OSError.*98", re.IGNORECASE),
    re.compile(r"bind.*failed", re.IGNORECASE),
]

_READY_PATTERNS = [
    re.compile(r"Application startup complete", re.IGNORECASE),
    re.compile(r"Uvicorn running on", re.IGNORECASE),
    re.compile(r"Started server process", re.IGNORECASE),
]


def _classify_log_failure(lines: List[str]) -> ServerFailure | None:
    """
    Scan log lines for known failure signatures.
    Returns the most specific ServerFailure subclass or None if no match.
    """
    text = "\n".join(lines)
    if any(p.search(text) for p in _OOM_PATTERNS):
        return OOMError("CUDA out of memory during vLLM startup", log_tail=lines)
    if any(p.search(text) for p in _PORT_CONFLICT_PATTERNS):
        return PortConflict("Port already in use", log_tail=lines)
    if any(p.search(text) for p in _CUDA_ERROR_PATTERNS):
        return CUDAError("CUDA runtime error during vLLM startup", log_tail=lines)
    return None


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------

class ServerState:
    STOPPED = "stopped"
    STARTING = "starting"
    HEALTHY = "healthy"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# VLLMServer
# ---------------------------------------------------------------------------

@dataclass
class VLLMServer:
    """
    Manages one vLLM server process.

    Parameters
    ----------
    model_id : str
        Hugging Face model ID passed to vLLM.
    flags : VLLMFlags
        Config produced by SearchSpace / optimiser.
    host : str
        Address the server binds to.
    port : int
        Port the server listens on.
    startup_timeout : int
        Seconds to wait for /health to return 200.
    gpu_type : str
        Used to select vendor-specific flags (AMD = ROCm).
    hf_token : str
        Hugging Face token; passed via HUGGING_FACE_HUB_TOKEN env var.
    log_buffer_size : int
        Max log lines kept in memory per server instance.
    """

    model_id: str
    flags: VLLMFlags
    host: str = "0.0.0.0"
    port: int = 8000
    startup_timeout: int = 300
    gpu_type: str = "H100"
    hf_token: str = ""
    log_buffer_size: int = 500
    extra_env: dict = field(default_factory=dict)
    # Docker image override — empty means use gpu_profiles.yaml default.
    # Can also be set via VLLM_IMAGE env var.
    docker_image: str = ""

    # Internal state — not part of __init__ signature
    _process: Optional[asyncio.subprocess.Process] = field(
        default=None, init=False, repr=False
    )
    _log_buffer: Deque[str] = field(
        default_factory=lambda: deque(maxlen=500), init=False, repr=False
    )
    _log_task: Optional[asyncio.Task] = field(
        default=None, init=False, repr=False
    )
    _state: str = field(default=ServerState.STOPPED, init=False, repr=False)
    _started_at: float = field(default=0.0, init=False, repr=False)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def endpoint(self) -> str:
        return f"http://localhost:{self.port}/v1"

    @property
    def health_url(self) -> str:
        return f"http://localhost:{self.port}/health"

    @property
    def state(self) -> str:
        return self._state

    @property
    def log_tail(self) -> List[str]:
        return list(self._log_buffer)

    def is_alive(self) -> bool:
        return (
            self._process is not None
            and self._process.returncode is None
        )

    # ── Context manager ───────────────────────────────────────────────────

    async def __aenter__(self) -> "VLLMServer":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()

    # ── Start ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Launch vLLM and wait until healthy.

        Raises
        ------
        OOMError, StartupTimeout, PortConflict, CUDAError, ProcessCrash
        """
        if self._state == ServerState.HEALTHY:
            log.warning("VLLMServer.start() called on already-healthy server")
            return

        self._state = ServerState.STARTING
        self._started_at = time.monotonic()
        self._log_buffer = deque(maxlen=self.log_buffer_size)

        cmd = self._build_command()
        env = self._build_env()

        log_dict(
            log, "info", "Launching vLLM server",
            run_id=self.flags.run_id,
            model=self.model_id,
            port=self.port,
            cmd=" ".join(cmd[:6]) + " ...",
        )

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            self._state = ServerState.FAILED
            raise ServerFailure(
                f"vLLM executable not found. "
                f"Is vLLM installed? (original error: {exc})"
            ) from exc

        self._log_task = asyncio.create_task(
            self._capture_logs(), name=f"log-capture-{self.port}"
        )

        try:
            await self._wait_healthy()
        except (OOMError, StartupTimeout, PortConflict, CUDAError, ProcessCrash):
            await self.stop()
            raise

        elapsed = time.monotonic() - self._started_at
        log_dict(
            log, "info", "vLLM server healthy",
            run_id=self.flags.run_id,
            port=self.port,
            elapsed_sec=round(elapsed, 1),
        )
        self._state = ServerState.HEALTHY

    # ── Stop ──────────────────────────────────────────────────────────────

    async def stop(self, grace_sec: int = 10) -> None:
        """
        Stop the vLLM server process.

        Sends SIGTERM first, waits *grace_sec* seconds, then SIGKILL.
        """
        if self._state == ServerState.STOPPED:
            return

        log_dict(
            log, "info", "Stopping vLLM server",
            run_id=self.flags.run_id, port=self.port,
        )

        if self._log_task and not self._log_task.done():
            self._log_task.cancel()
            try:
                await self._log_task
            except asyncio.CancelledError:
                pass

        if self._process and self._process.returncode is None:
            try:
                pgid = os.getpgid(self._process.pid)
                os.killpg(pgid, signal.SIGTERM)
                log.debug(f"Sent SIGTERM to process group {pgid}")
            except (ProcessLookupError, PermissionError):
                pass

            try:
                await asyncio.wait_for(self._process.wait(), timeout=grace_sec)
                log.debug("vLLM server exited after SIGTERM")
            except asyncio.TimeoutError:
                log.warning("vLLM did not exit within grace period — sending SIGKILL")
                try:
                    pgid = os.getpgid(self._process.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                await self._process.wait()

        self._process = None
        self._state = ServerState.STOPPED

        # Belt-and-suspenders: ensure the Docker container is gone
        # (handles cases where SIGTERM doesn't propagate into the container)
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "stop", "-t", "5", f"oceantune-vllm-{self.port}",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=10)
        except Exception:
            pass

        log_dict(
            log, "info", "vLLM server stopped",
            run_id=self.flags.run_id, port=self.port,
        )

    # ── Health check ──────────────────────────────────────────────────────

    async def is_healthy(self) -> bool:
        """Return True if /health responds with 200."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(self.health_url)
                return resp.status_code == 200
        except Exception:
            return False

    # ── Internal: wait for healthy ────────────────────────────────────────

    async def _wait_healthy(self) -> None:
        """
        Poll /health with exponential backoff until 200 or timeout.

        Also monitors the log buffer for OOM / CUDA errors so we can
        fail fast instead of waiting the full timeout.
        """
        deadline = self._started_at + self.startup_timeout
        delay = 2.0
        max_delay = 10.0
        attempt = 0

        while time.monotonic() < deadline:
            if not self.is_alive():
                rc = self._process.returncode if self._process else -1
                failure = _classify_log_failure(self.log_tail)
                if failure:
                    raise failure
                raise ProcessCrash(
                    f"vLLM process exited with code {rc} before becoming healthy",
                    log_tail=self.log_tail,
                )

            failure = _classify_log_failure(self.log_tail)
            if failure:
                raise failure

            if await self.is_healthy():
                return

            attempt += 1
            remaining = deadline - time.monotonic()
            sleep_time = min(delay, remaining, max_delay)
            if sleep_time <= 0:
                break

            log.debug(
                f"Waiting for vLLM /health (attempt {attempt}, "
                f"{remaining:.0f}s remaining) ..."
            )
            await asyncio.sleep(sleep_time)
            delay = min(delay * 1.5, max_delay)

        failure = _classify_log_failure(self.log_tail)
        if failure:
            raise failure

        raise StartupTimeout(
            f"vLLM server did not become healthy within {self.startup_timeout}s",
            log_tail=self.log_tail,
        )

    # ── Internal: log capture ─────────────────────────────────────────────

    async def _capture_logs(self) -> None:
        """
        Read stdout/stderr from the vLLM process line by line.

        Stores lines in the deque and writes them to the OceanTune logger.
        """
        if not self._process or not self._process.stdout:
            return

        try:
            async for raw_line in self._process.stdout:
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                self._log_buffer.append(line)
                log.debug(f"[vllm:{self.port}] {line}")
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.warning(f"Log capture error: {exc}")

    # ── Internal: resolve docker image ───────────────────────────────────

    def _resolve_docker_image(self) -> str:
        """
        Resolve the Docker image to use, in priority order:
          1. VLLM_IMAGE env var
          2. self.docker_image (from YAML vllm.docker_image)
          3. gpu_profiles.yaml docker_image for this gpu_type
        """
        if img := os.environ.get("VLLM_IMAGE"):
            return img
        if self.docker_image:
            return self.docker_image
        profile = _load_gpu_profile(self.gpu_type)
        if img := profile.get("docker_image"):
            return img
        raise ServerFailure(
            f"No Docker image configured for gpu_type={self.gpu_type}. "
            f"Set VLLM_IMAGE env var or vllm.docker_image in oceantune.yaml."
        )

    # ── Internal: build command ───────────────────────────────────────────

    def _build_command(self) -> List[str]:
        """Build the docker run command that launches vLLM inside a container."""
        profile = _load_gpu_profile(self.gpu_type)
        docker_image = self._resolve_docker_image()

        hf_token = (
            self.hf_token
            or os.environ.get("HF_TOKEN", "")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
        )
        hf_cache = os.path.expanduser("~/.cache/huggingface")

        cmd = [
            "docker", "run", "--rm",
            "--name", f"oceantune-vllm-{self.port}",
            "-p", f"{self.port}:{self.port}",
            "--shm-size", "2g",
            "-v", f"{hf_cache}:/root/.cache/huggingface",
        ]

        # IPC host mode — required for tensor-parallel shared memory
        if profile.get("ipc_host", False):
            cmd.append("--ipc=host")

        # GPU assignment
        if self.gpu_type in _AMD_GPU_TYPES:
            # AMD ROCm: expose KFD/DRI devices; use ROCR_VISIBLE_DEVICES for slot isolation
            cmd += ["--device", "/dev/kfd", "--device", "/dev/dri", "--group-add", "video"]
            rocr_devs = self.extra_env.get("ROCR_VISIBLE_DEVICES", "")
            if rocr_devs:
                cmd += ["-e", f"ROCR_VISIBLE_DEVICES={rocr_devs}"]
        else:
            # NVIDIA: --gpus device=X,Y for slot isolation
            gpu_devices = self.extra_env.get("CUDA_VISIBLE_DEVICES", "all")
            cmd += ["--gpus", f"device={gpu_devices}"]

        # HuggingFace auth
        if hf_token:
            cmd += ["-e", f"HF_TOKEN={hf_token}", "-e", f"HUGGING_FACE_HUB_TOKEN={hf_token}"]

        # Profile env vars (NCCL, ROCm perf knobs, etc.)
        for key, val in profile.get("env_vars", {}).items():
            cmd += ["-e", f"{key}={val}"]

        # Any extra env vars (excluding GPU visibility — handled above)
        skip = {"CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"}
        for key, val in self.extra_env.items():
            if key not in skip:
                cmd += ["-e", f"{key}={val}"]

        # Docker image
        cmd.append(docker_image)

        # vLLM server args (passed as container CMD)
        vllm_args = self.flags.to_vllm_args(model_id=self.model_id, gpu_type=self.gpu_type)
        extra_vllm_args: List[str] = profile.get("vllm_extra_args", [])
        cmd += [
            "--host", "0.0.0.0",
            "--port", str(self.port),
            *vllm_args,
            *extra_vllm_args,
        ]

        return cmd

    # ── Internal: build environment ───────────────────────────────────────

    def _build_env(self) -> dict:
        """Return host OS environment for the docker CLI subprocess."""
        return os.environ.copy()

    # ── Diagnostics ───────────────────────────────────────────────────────

    def diagnostic_report(self) -> dict:
        """Return a dict summarising the server's current state."""
        return {
            "run_id": self.flags.run_id,
            "model_id": self.model_id,
            "port": self.port,
            "state": self._state,
            "uptime_sec": round(time.monotonic() - self._started_at, 1)
                          if self._started_at else 0,
            "log_lines_captured": len(self._log_buffer),
            "log_tail": self.log_tail[-10:],
        }


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def make_server(cfg, flags: VLLMFlags, port: int | None = None) -> VLLMServer:
    """
    Convenience factory: build a VLLMServer from an OceanTuneConfig.

    Parameters
    ----------
    cfg : OceanTuneConfig
        Loaded by core.config.load_config().
    flags : VLLMFlags
        Config point produced by the search space.
    port : int, optional
        Override port (useful when running parallel experiments).
    """
    return VLLMServer(
        model_id=cfg.model_id,
        flags=flags,
        host=cfg.vllm.host,
        port=port or cfg.vllm.port,
        startup_timeout=cfg.vllm.startup_timeout_sec,
        gpu_type=cfg.gpu_type,
        hf_token=cfg.hf_token,
    )
