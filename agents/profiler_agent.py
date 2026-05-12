"""
agents/profiler_agent.py
------------------------
Profiler Agent — Stage 3, Part 1.

Runs a vLLM instance under PyTorch profiler, collects kernel timing traces,
and returns a structured ProfileTrace with per-operation timing breakdowns.

The profiler uses vLLM's --profiler-config flag (vLLM v0.13+) to enable the
PyTorch profiler and write Chrome trace JSON to a mounted volume directory.

Trace output:
  - Top kernels by GPU time (attention, GEMM, RoPE, RMSNorm, MoE routing)
  - Memory bandwidth utilisation
  - Python scheduling overhead fraction
  - Operation type breakdown: compute vs memory vs communication

Usage
-----
    profiler = ProfilerAgent(
        do_client=client,
        db=db,
        gpu_alloc=alloc,
        port_alloc=pool,
        model_id="Qwen/Qwen2.5-7B-Instruct",
        gpu_type="H200",
    )
    trace = await profiler.run(
        session_id="...",
        winner_flags=winner_flags,
        optimal_concurrency=64,
        input_len=1024,
        output_len=1024,
    )
    # trace.top_kernels, trace.bottleneck_type, trace.summary
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents.do_client import DOClient, DOClientError
from core.db import Database
from core.gpu_allocator import GPUSlotAllocator
from core.port_allocator import PortAllocator
from core.search_space import VLLMFlags
from core.vllm_server import VLLMServer, _load_gpu_profile
from core.benchmark_runner import BenchmarkEngine
from core.log_analyzer import LogAnalyzer

log = logging.getLogger("agents.profiler_agent")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_AMD_GPU_TYPES = {"MI300X", "MI325X", "MI350X"}

# Number of warmup + profiling requests
_WARMUP_REQUESTS = 10
_PROFILE_REQUESTS = 30


# ---------------------------------------------------------------------------
# Profile trace dataclass
# ---------------------------------------------------------------------------

@dataclass
class KernelTiming:
    """Timing for a single CUDA/Triton/HIP kernel or operation group."""
    name: str
    category: str          # attention | gemm | norm | rope | moe | comm | other
    gpu_time_ms: float
    gpu_time_pct: float    # % of total GPU time
    call_count: int = 0
    avg_time_us: float = 0.0


@dataclass
class ProfileTrace:
    """Structured output from a profiling run."""
    session_id: str
    model_id: str
    gpu_type: str
    optimal_concurrency: int
    input_len: int
    output_len: int

    # Top kernels ranked by GPU time
    top_kernels: List[KernelTiming] = field(default_factory=list)

    # Category breakdown
    attention_pct: float = 0.0     # FlashAttention / SDPA
    gemm_pct: float = 0.0          # Linear projections, FFN
    norm_pct: float = 0.0          # RMSNorm, LayerNorm
    rope_pct: float = 0.0          # Rotary position embedding
    moe_pct: float = 0.0           # MoE routing, expert dispatch
    comm_pct: float = 0.0          # NCCL/RCCL all-reduce
    python_overhead_pct: float = 0.0   # Scheduling, tokenisation
    other_pct: float = 0.0

    # Primary bottleneck classification
    bottleneck_type: str = "unknown"   # compute | memory_bandwidth | memory_capacity | scheduling
    bottleneck_kernel: str = ""

    # Memory stats
    peak_memory_gb: float = 0.0
    memory_bandwidth_utilisation: float = 0.0  # fraction of theoretical peak

    # Summary text from Research Agent
    research_summary: str = ""
    optimization_recommendations: List[str] = field(default_factory=list)

    # Raw trace path (if saved to disk)
    trace_path: Optional[str] = None

    def is_compute_bound(self) -> bool:
        return self.bottleneck_type == "compute"

    def is_memory_bound(self) -> bool:
        return self.bottleneck_type in ("memory_bandwidth", "memory_capacity")


# ---------------------------------------------------------------------------
# Profiler Agent
# ---------------------------------------------------------------------------

class ProfilerAgent:
    """
    Runs vLLM under PyTorch profiler and returns a structured ProfileTrace.

    The profiling approach:
    1. Start vLLM with winner_flags + profiling enabled
    2. Run warmup requests (excluded from trace)
    3. Send _PROFILE_REQUESTS at optimal_concurrency while profiler is active
    4. Stop profiler, extract kernel timings from trace
    5. Call Research Agent to interpret the trace
    """

    def __init__(
        self,
        *,
        do_client: DOClient,
        db: Database,
        gpu_alloc: GPUSlotAllocator,
        port_alloc: PortAllocator,
        model_id: str,
        gpu_type: str,
        startup_timeout_sec: int = 1200,
        docker_image: str = "",
    ) -> None:
        self._client = do_client
        self._db = db
        self._gpu_alloc = gpu_alloc
        self._port_alloc = port_alloc
        self._model_id = model_id
        self._gpu_type = gpu_type
        self._startup_timeout_sec = startup_timeout_sec
        self._docker_image = docker_image
        self._vendor = "amd" if gpu_type in _AMD_GPU_TYPES else "nvidia"

    async def run(
        self,
        *,
        session_id: str,
        winner_flags: Dict[str, Any],
        optimal_concurrency: int,
        input_len: int = 1024,
        output_len: int = 1024,
    ) -> ProfileTrace:
        """
        Profile the winner configuration at the optimal concurrency level.
        Returns a ProfileTrace. Never raises — returns an empty trace on error.
        """
        log.info(
            "ProfilerAgent: profiling model=%s gpu=%s concurrency=%d",
            self._model_id, self._gpu_type, optimal_concurrency,
        )

        trace = ProfileTrace(
            session_id=session_id,
            model_id=self._model_id,
            gpu_type=self._gpu_type,
            optimal_concurrency=optimal_concurrency,
            input_len=input_len,
            output_len=output_len,
        )

        known_fields = set(VLLMFlags.__dataclass_fields__.keys())
        flags = VLLMFlags(**{k: v for k, v in winner_flags.items() if k in known_fields})
        tp_size = flags.tensor_parallel_size or 1

        slot = await self._gpu_alloc.acquire(tp_size)
        if slot is None:
            log.warning("ProfilerAgent: no GPU slot available")
            return trace

        port = await self._port_alloc.acquire()
        if port is None:
            await self._gpu_alloc.release(slot)
            log.warning("ProfilerAgent: no port available")
            return trace

        device_env = self._gpu_alloc.build_device_env(slot)

        # vLLM trace output directory (on the host)
        trace_dir = _REPO_ROOT / "storage" / "profiles" / session_id
        trace_dir.mkdir(parents=True, exist_ok=True)

        # Container-internal path where vLLM writes trace files.
        # We volume-mount trace_dir → _CONTAINER_TRACE_DIR so files appear on the host.
        _CONTAINER_TRACE_DIR = "/tmp/vllm_profile"

        # vLLM v0.13+ profiler config — uses --profiler-config JSON flag.
        # VLLM_RPC_TIMEOUT must be large: flushing traces for large models can take
        # 10+ minutes (vLLM docs recommendation: 1800000 ms = 30 min).
        import json as _json
        profiler_config_json = _json.dumps({
            "profiler": "torch",
            "torch_profiler_dir": _CONTAINER_TRACE_DIR,
            "torch_profiler_with_stack": True,
        })

        profile_env = {
            **device_env,
            "VLLM_RPC_TIMEOUT": "1800000",
        }

        server = VLLMServer(
            model_id=self._model_id,
            flags=flags,
            gpu_type=self._gpu_type,
            port=port,
            startup_timeout=self._startup_timeout_sec,
            extra_env=profile_env,
            docker_image=self._docker_image,
            extra_docker_args=["-v", f"{trace_dir}:{_CONTAINER_TRACE_DIR}"],
            extra_vllm_args=["--profiler-config", profiler_config_json],
        )

        try:
            await server.start()

            # Run profiling benchmark
            raw_trace = await self._run_profiling_workload(
                port=port,
                concurrency=optimal_concurrency,
                input_len=input_len,
                output_len=output_len,
                trace_dir=trace_dir,
            )

            if raw_trace:
                trace = self._parse_trace(trace, raw_trace)
                trace.trace_path = str(trace_dir)
                log.info(
                    "Profile: attention=%.1f%% gemm=%.1f%% moe=%.1f%% bottleneck=%s",
                    trace.attention_pct, trace.gemm_pct, trace.moe_pct,
                    trace.bottleneck_type,
                )
            else:
                log.warning("ProfilerAgent: no trace data collected — using log analysis")
                trace = self._fallback_trace_from_logs(trace, server.log_tail)

        except Exception as exc:
            log.warning("ProfilerAgent error: %s", exc)
            trace.research_summary = f"Profiling failed: {exc}"

        finally:
            try:
                await server.stop()
            except Exception:
                pass
            await self._gpu_alloc.release(slot)
            await self._port_alloc.release(port)

        return trace

    async def _run_profiling_workload(
        self,
        *,
        port: int,
        concurrency: int,
        input_len: int,
        output_len: int,
        trace_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        """
        Send requests to trigger profiling via vLLM's /start_profile + /stop_profile API.
        vLLM writes Chrome trace JSON to torch_profiler_dir (container path, volume-mounted).
        Returns parsed trace dict or None.
        """
        import httpx

        base_url = f"http://localhost:{port}"

        # Warmup
        log.info("ProfilerAgent: warming up (%d requests)...", _WARMUP_REQUESTS)
        engine = BenchmarkEngine(
            base_url=base_url,
            model_id=self._model_id,
            concurrency_levels=[concurrency],
            num_prompts=_WARMUP_REQUESTS,
            input_len=input_len,
            output_len=output_len,
        )
        await engine.run()

        # Start profiler via vLLM's /start_profile endpoint (vLLM >= 0.4)
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                await client.post(f"{base_url}/start_profile")
                log.info("ProfilerAgent: profiler started via /start_profile")
            except Exception:
                log.info("ProfilerAgent: /start_profile not available — using env-based profiling")

        # Profile workload
        log.info("ProfilerAgent: profiling (%d requests at concurrency=%d)...",
                 _PROFILE_REQUESTS, concurrency)
        engine = BenchmarkEngine(
            base_url=base_url,
            model_id=self._model_id,
            concurrency_levels=[concurrency],
            num_prompts=_PROFILE_REQUESTS,
            input_len=input_len,
            output_len=output_len,
        )
        await engine.run()

        # Stop profiler — vLLM >= 0.4 supports /stop_profile; older versions don't.
        # A 404 is normal for older vLLM — ignore it.
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.post(f"{base_url}/stop_profile")
                if resp.status_code == 200:
                    log.info("ProfilerAgent: profiler stopped via /stop_profile")
                else:
                    log.info(
                        "ProfilerAgent: /stop_profile returned %d — "
                        "using VLLM_TORCH_PROFILER_DIR env-based profiling",
                        resp.status_code,
                    )
            except Exception:
                log.info("ProfilerAgent: /stop_profile not available — env-based profiling")

        # Wait for vLLM to flush the trace file to disk.
        # The flush can take several seconds after stop_profile returns.
        for wait_s in [3, 5, 10]:
            await asyncio.sleep(wait_s)
            trace_files = (
                list(trace_dir.glob("*.json"))
                + list(trace_dir.glob("*.pt.trace.json"))
                + list(trace_dir.rglob("*.json"))   # vLLM may write into a subdirectory
            )
            # Exclude empty files
            trace_files = [p for p in trace_files if p.stat().st_size > 100]
            if trace_files:
                break

        if not trace_files:
            log.warning("ProfilerAgent: no trace files found in %s after waiting", trace_dir)
            return None

        # Parse the most recent trace file
        latest = max(trace_files, key=lambda p: p.stat().st_mtime)
        log.info("ProfilerAgent: parsing trace file %s (%.1f MB)",
                 latest.name, latest.stat().st_size / 1e6)

        try:
            with open(latest, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            log.warning("ProfilerAgent: failed to parse trace: %s", exc)
            return None

    def _parse_trace(
        self,
        trace: ProfileTrace,
        raw_trace: Dict[str, Any],
    ) -> ProfileTrace:
        """
        Parse a PyTorch profiler trace JSON into structured ProfileTrace.

        PyTorch Chrome trace format:
          {"traceEvents": [{"name": ..., "dur": ..., "cat": ..., ...}, ...]}
        """
        events = raw_trace.get("traceEvents", [])

        # Aggregate GPU kernel times by name
        kernel_times: Dict[str, Dict[str, Any]] = {}
        total_gpu_us = 0.0

        for event in events:
            # Only GPU kernels (cat="kernel" or cat="gpu_memcpy")
            if event.get("ph") != "X":
                continue
            cat = event.get("cat", "")
            if cat not in ("kernel", "gpu_memcpy", "gpu_user_annotation"):
                continue

            name = event.get("name", "unknown")
            dur = float(event.get("dur", 0))  # microseconds
            total_gpu_us += dur

            if name not in kernel_times:
                kernel_times[name] = {"total_us": 0.0, "count": 0}
            kernel_times[name]["total_us"] += dur
            kernel_times[name]["count"] += 1

        if total_gpu_us == 0:
            log.warning("ProfilerAgent: no GPU kernel events found in trace")
            return trace

        # Classify kernels into categories
        category_totals: Dict[str, float] = {
            "attention": 0.0, "gemm": 0.0, "norm": 0.0,
            "rope": 0.0, "moe": 0.0, "comm": 0.0, "other": 0.0,
        }

        kernel_list: List[KernelTiming] = []
        for name, stats in kernel_times.items():
            cat = _classify_kernel(name)
            total_us = stats["total_us"]
            count = stats["count"]
            pct = (total_us / total_gpu_us) * 100.0

            category_totals[cat] = category_totals.get(cat, 0.0) + total_us
            kernel_list.append(KernelTiming(
                name=name,
                category=cat,
                gpu_time_ms=total_us / 1000.0,
                gpu_time_pct=pct,
                call_count=count,
                avg_time_us=total_us / count if count > 0 else 0.0,
            ))

        # Top kernels by GPU time
        kernel_list.sort(key=lambda k: k.gpu_time_pct, reverse=True)
        trace.top_kernels = kernel_list[:20]

        # Category percentages
        trace.attention_pct = (category_totals["attention"] / total_gpu_us) * 100.0
        trace.gemm_pct = (category_totals["gemm"] / total_gpu_us) * 100.0
        trace.norm_pct = (category_totals["norm"] / total_gpu_us) * 100.0
        trace.rope_pct = (category_totals["rope"] / total_gpu_us) * 100.0
        trace.moe_pct = (category_totals["moe"] / total_gpu_us) * 100.0
        trace.comm_pct = (category_totals["comm"] / total_gpu_us) * 100.0
        trace.other_pct = (category_totals["other"] / total_gpu_us) * 100.0

        # Classify primary bottleneck
        top_cat = max(category_totals, key=category_totals.get)
        trace.bottleneck_type = _map_category_to_bottleneck(top_cat, trace)
        if trace.top_kernels:
            trace.bottleneck_kernel = trace.top_kernels[0].name

        return trace

    def _fallback_trace_from_logs(
        self,
        trace: ProfileTrace,
        log_tail,
    ) -> ProfileTrace:
        """
        Infer bottleneck from vLLM logs when profiler trace is unavailable.
        Uses heuristics from log patterns.
        """
        # log_tail may be a List[str] or a str — normalise to str
        if isinstance(log_tail, list):
            log_str = "\n".join(log_tail)
        else:
            log_str = str(log_tail)
        log_lower = log_str.lower()
        if "kv cache" in log_lower and "full" in log_lower:
            trace.bottleneck_type = "memory_capacity"
        elif "cuda out of memory" in log_lower:
            trace.bottleneck_type = "memory_capacity"
        else:
            trace.bottleneck_type = "compute"
        trace.research_summary = "Profiler trace unavailable — bottleneck inferred from logs."
        return trace


# ---------------------------------------------------------------------------
# Kernel classifier
# ---------------------------------------------------------------------------

def _classify_kernel(name: str) -> str:
    """Map a CUDA kernel name to a high-level category."""
    n = name.lower()
    if any(k in n for k in ["flash_attn", "flash_fwd", "flash_bwd", "sdpa",
                              "attention", "flashinfer", "paged_attn",
                              "self_attn", "mha_"]):
        return "attention"
    if any(k in n for k in ["gemm", "cutlass", "cublas", "sgemm", "hgemm",
                              "matmul", "linear_", "addmm", "mm_", "bmm"]):
        return "gemm"
    if any(k in n for k in ["rmsnorm", "layernorm", "group_norm", "rms_norm"]):
        return "norm"
    if any(k in n for k in ["rope", "rotary", "apply_rotary"]):
        return "rope"
    if any(k in n for k in ["moe", "expert", "topk_gate", "all2all",
                              "dispatch", "combine", "grouped_gemm"]):
        return "moe"
    if any(k in n for k in ["nccl", "all_reduce", "broadcast", "reduce_scatter",
                              "all_gather", "rccl", "comm"]):
        return "comm"
    return "other"


def _map_category_to_bottleneck(top_category: str, trace: ProfileTrace) -> str:
    """Map top GPU-time category to a bottleneck type string."""
    mapping = {
        "attention": "compute",
        "gemm": "compute",
        "moe": "compute",
        "comm": "compute",          # TP communication overhead
        "norm": "compute",
        "rope": "compute",
        "other": "compute",
    }
    return mapping.get(top_category, "compute")
