"""
microbench/operator_bench.py
-----------------------------
Isolated operator microbenchmarking framework.

Benchmarks individual GPU operations (attention, GEMM, RMSNorm, RoPE, etc.)
in complete isolation from the vLLM serving stack. This gives clean signal
about the operator's hardware efficiency without scheduling noise.

For each operator, reports:
  - Latency (mean, median, p99) in microseconds
  - Effective throughput (TFLOP/s or GB/s depending on operation type)
  - Achieved fraction of peak hardware throughput
  - Memory bandwidth utilisation (for memory-bound ops)
  - Roofline bound classification

Usage
-----
    bench = OperatorBench(gpu_type="H100")
    result = await bench.run(
        op_type="attention",
        params={"batch_size": 1, "seq_len": 2048, "num_heads": 32, "head_dim": 128, "num_kv_heads": 8},
        num_warmup=20,
        num_iters=100,
    )
    print(result.latency_us_p50, result.tflops_achieved, result.roofline_efficiency_pct)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("microbench.operator_bench")

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Peak hardware specs for roofline model
_GPU_SPECS: Dict[str, Dict[str, float]] = {
    "H100":   {"fp16_tflops": 989.0,  "hbm_gbps": 3350.0, "l2_gbps": 12000.0},
    "H200":   {"fp16_tflops": 989.0,  "hbm_gbps": 4800.0, "l2_gbps": 12000.0},
    "A100":   {"fp16_tflops": 312.0,  "hbm_gbps": 2000.0, "l2_gbps": 6000.0},
    "A6000":  {"fp16_tflops": 154.0,  "hbm_gbps": 768.0,  "l2_gbps": 3000.0},
    "MI300X": {"fp16_tflops": 1307.0, "hbm_gbps": 5300.0, "l2_gbps": 16000.0},
    "MI325X": {"fp16_tflops": 1307.0, "hbm_gbps": 6000.0, "l2_gbps": 16000.0},
    "MI350X": {"fp16_tflops": 2600.0, "hbm_gbps": 8000.0, "l2_gbps": 24000.0},
}

# FLOP counts per operation (as functions of shape parameters)
# These are used for roofline analysis
def _attn_flops(bs: int, nheads: int, seqlen: int, head_dim: int) -> float:
    """FLOPs for one self-attention forward pass (FP16)."""
    # QK^T + softmax + AV: ~4 * bs * nheads * seqlen^2 * head_dim
    return 4.0 * bs * nheads * seqlen * seqlen * head_dim

def _gemm_flops(M: int, N: int, K: int) -> float:
    """FLOPs for one GEMM."""
    return 2.0 * M * N * K

def _attn_memory_bytes(bs: int, nheads: int, seqlen: int, head_dim: int, nkv_heads: int, dtype_bytes: int = 2) -> float:
    """Bytes read/written for attention (Q, K, V, output)."""
    q_bytes = bs * nheads * seqlen * head_dim * dtype_bytes
    kv_bytes = bs * nkv_heads * seqlen * head_dim * dtype_bytes * 2  # K + V
    out_bytes = q_bytes
    return q_bytes + kv_bytes + out_bytes


@dataclass
class BenchResult:
    """Result of an operator microbenchmark run."""
    op_type: str                    # attention | gemm | rmsnorm | rope | moe
    params: Dict[str, Any]          # Shape/config parameters
    gpu_type: str

    # Latency (microseconds)
    latency_us_mean: float = 0.0
    latency_us_p50: float = 0.0
    latency_us_p95: float = 0.0
    latency_us_p99: float = 0.0

    # Compute throughput
    tflops_achieved: float = 0.0    # Achieved TFLOP/s
    tflops_peak: float = 0.0        # Hardware peak TFLOP/s
    compute_efficiency_pct: float = 0.0  # tflops_achieved / tflops_peak * 100

    # Memory bandwidth
    mem_bw_gbps: float = 0.0        # Achieved memory bandwidth GB/s
    mem_bw_peak_gbps: float = 0.0   # Hardware peak GB/s
    mem_bw_efficiency_pct: float = 0.0

    # Roofline classification
    arithmetic_intensity: float = 0.0   # FLOP/byte
    ridge_point: float = 0.0        # Hardware ridge point FLOP/byte
    roofline_bound: str = "unknown"  # compute | memory

    # Whether this ran successfully
    success: bool = False
    error: str = ""

    def summary(self) -> str:
        if not self.success:
            return f"[{self.op_type}] FAILED: {self.error}"
        return (
            f"[{self.op_type}] latency={self.latency_us_p50:.1f}us "
            f"tflops={self.tflops_achieved:.1f}/{self.tflops_peak:.0f} "
            f"({self.compute_efficiency_pct:.1f}%) "
            f"bound={self.roofline_bound}"
        )


class OperatorBench:
    """
    Microbenchmarking harness for GPU operators.

    Generates and runs a small Python benchmark script in a subprocess so it
    can be run against any installed PyTorch / flash-attn / triton stack without
    polluting the OceanTune process.
    """

    def __init__(
        self,
        gpu_type: str = "H100",
        device: str = "cuda:0",
    ) -> None:
        self._gpu_type = gpu_type
        self._device = device
        self._specs = _GPU_SPECS.get(gpu_type, _GPU_SPECS["H100"])

    async def run(
        self,
        *,
        op_type: str,
        params: Dict[str, Any],
        num_warmup: int = 20,
        num_iters: int = 100,
        timeout_sec: int = 120,
    ) -> BenchResult:
        """
        Benchmark a single operator.

        Parameters
        ----------
        op_type : str
            One of: attention | gemm | rmsnorm | rope | moe_dispatch
        params : dict
            Shape parameters (batch_size, seq_len, etc.)
        num_warmup : int
            Warmup iterations (excluded from timing).
        num_iters : int
            Measured iterations.

        Returns
        -------
        BenchResult with latency and efficiency statistics.
        """
        result = BenchResult(op_type=op_type, params=params, gpu_type=self._gpu_type)
        result.tflops_peak = self._specs["fp16_tflops"]
        result.mem_bw_peak_gbps = self._specs["hbm_gbps"]

        # Generate benchmark script for the requested operation
        script = self._generate_script(op_type, params, num_warmup, num_iters)
        if script is None:
            result.error = f"Unknown op_type: {op_type}"
            return result

        # Write to temp file and run
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="oceantune_bench_"
        ) as f:
            f.write(script)
            script_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": self._device.replace("cuda:", "")},
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)

            if proc.returncode != 0:
                result.error = stderr.decode()[:300]
                return result

            # Parse JSON output from the script
            output = stdout.decode().strip()
            metrics = json.loads(output)

            result.latency_us_mean = metrics.get("mean_us", 0.0)
            result.latency_us_p50 = metrics.get("p50_us", 0.0)
            result.latency_us_p95 = metrics.get("p95_us", 0.0)
            result.latency_us_p99 = metrics.get("p99_us", 0.0)

            # Compute roofline metrics
            flops = self._estimate_flops(op_type, params)
            mem_bytes = self._estimate_memory_bytes(op_type, params)
            latency_s = result.latency_us_p50 * 1e-6

            if latency_s > 0:
                result.tflops_achieved = (flops / 1e12) / latency_s
                result.mem_bw_gbps = (mem_bytes / 1e9) / latency_s
                result.compute_efficiency_pct = (result.tflops_achieved / result.tflops_peak) * 100
                result.mem_bw_efficiency_pct = (result.mem_bw_gbps / result.mem_bw_peak_gbps) * 100

            # Roofline classification
            if mem_bytes > 0:
                result.arithmetic_intensity = flops / mem_bytes
                result.ridge_point = (self._specs["fp16_tflops"] * 1e12) / (self._specs["hbm_gbps"] * 1e9)
                result.roofline_bound = (
                    "compute" if result.arithmetic_intensity > result.ridge_point else "memory"
                )

            result.success = True

        except asyncio.TimeoutError:
            result.error = f"Benchmark timed out after {timeout_sec}s"
        except json.JSONDecodeError as exc:
            result.error = f"JSON parse error: {exc} — output: {stdout.decode()[:200]}"
        except Exception as exc:
            result.error = str(exc)
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

        return result

    def _estimate_flops(self, op_type: str, params: Dict[str, Any]) -> float:
        if op_type == "attention":
            return _attn_flops(
                bs=params.get("batch_size", 1),
                nheads=params.get("num_heads", 32),
                seqlen=params.get("seq_len", 2048),
                head_dim=params.get("head_dim", 128),
            )
        if op_type == "gemm":
            return _gemm_flops(
                M=params.get("M", 1024),
                N=params.get("N", 4096),
                K=params.get("K", 4096),
            )
        return 0.0

    def _estimate_memory_bytes(self, op_type: str, params: Dict[str, Any]) -> float:
        if op_type == "attention":
            return _attn_memory_bytes(
                bs=params.get("batch_size", 1),
                nheads=params.get("num_heads", 32),
                seqlen=params.get("seq_len", 2048),
                head_dim=params.get("head_dim", 128),
                nkv_heads=params.get("num_kv_heads", params.get("num_heads", 32)),
            )
        if op_type == "gemm":
            M, N, K = params.get("M", 1024), params.get("N", 4096), params.get("K", 4096)
            return (M * K + K * N + M * N) * 2  # fp16 = 2 bytes
        return 0.0

    def _generate_script(
        self,
        op_type: str,
        params: Dict[str, Any],
        num_warmup: int,
        num_iters: int,
    ) -> Optional[str]:
        """Generate a self-contained Python benchmark script."""
        if op_type == "attention":
            return self._gen_attention_script(params, num_warmup, num_iters)
        if op_type == "gemm":
            return self._gen_gemm_script(params, num_warmup, num_iters)
        if op_type == "rmsnorm":
            return self._gen_rmsnorm_script(params, num_warmup, num_iters)
        if op_type == "rope":
            return self._gen_rope_script(params, num_warmup, num_iters)
        if op_type == "moe_dispatch":
            return self._gen_moe_script(params, num_warmup, num_iters)
        return None

    def _gen_attention_script(
        self, params: Dict[str, Any], num_warmup: int, num_iters: int
    ) -> str:
        bs = params.get("batch_size", 1)
        nheads = params.get("num_heads", 32)
        nkv = params.get("num_kv_heads", nheads)
        seqlen = params.get("seq_len", 2048)
        hdim = params.get("head_dim", 128)
        return f"""\
import torch, json, statistics
device = "cuda"
bs, nheads, nkv, seqlen, hdim = {bs}, {nheads}, {nkv}, {seqlen}, {hdim}
q = torch.randn(bs, nheads, seqlen, hdim, dtype=torch.float16, device=device)
k = torch.randn(bs, nkv, seqlen, hdim, dtype=torch.float16, device=device)
v = torch.randn(bs, nkv, seqlen, hdim, dtype=torch.float16, device=device)

# Try flash-attn, fall back to torch SDPA
try:
    from flash_attn import flash_attn_func
    def _run():
        q2 = q.transpose(1, 2)  # (bs, seqlen, nheads, hdim)
        k2 = k.transpose(1, 2)
        v2 = v.transpose(1, 2)
        return flash_attn_func(q2, k2, v2, causal=True)
except ImportError:
    import torch.nn.functional as F
    def _run():
        kv_rpt = nheads // nkv
        k2 = k.repeat_interleave(kv_rpt, dim=1) if kv_rpt > 1 else k
        v2 = v.repeat_interleave(kv_rpt, dim=1) if kv_rpt > 1 else v
        return F.scaled_dot_product_attention(q, k2, v2, is_causal=True)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Warmup
for _ in range({num_warmup}):
    _ = _run()
torch.cuda.synchronize()

# Measure
times = []
for _ in range({num_iters}):
    start.record()
    _ = _run()
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end) * 1000)  # ms -> us

times.sort()
import json
print(json.dumps({{
    "mean_us": statistics.mean(times),
    "p50_us": times[len(times)//2],
    "p95_us": times[int(len(times)*0.95)],
    "p99_us": times[int(len(times)*0.99)],
}}))
"""

    def _gen_gemm_script(
        self, params: Dict[str, Any], num_warmup: int, num_iters: int
    ) -> str:
        M = params.get("M", 1024)
        N = params.get("N", 4096)
        K = params.get("K", 4096)
        return f"""\
import torch, json, statistics
device = "cuda"
a = torch.randn({M}, {K}, dtype=torch.float16, device=device)
b = torch.randn({K}, {N}, dtype=torch.float16, device=device)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for _ in range({num_warmup}):
    _ = torch.mm(a, b)
torch.cuda.synchronize()

times = []
for _ in range({num_iters}):
    start.record()
    _ = torch.mm(a, b)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end) * 1000)

times.sort()
print(json.dumps({{
    "mean_us": statistics.mean(times),
    "p50_us": times[len(times)//2],
    "p95_us": times[int(len(times)*0.95)],
    "p99_us": times[int(len(times)*0.99)],
}}))
"""

    def _gen_rmsnorm_script(
        self, params: Dict[str, Any], num_warmup: int, num_iters: int
    ) -> str:
        bs = params.get("batch_size", 64)
        seqlen = params.get("seq_len", 2048)
        hdim = params.get("hidden_dim", 4096)
        return f"""\
import torch, json, statistics
device = "cuda"
x = torch.randn({bs}, {seqlen}, {hdim}, dtype=torch.float16, device=device)
w = torch.ones({hdim}, dtype=torch.float16, device=device)

def rmsnorm(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for _ in range({num_warmup}):
    _ = rmsnorm(x, w)
torch.cuda.synchronize()

times = []
for _ in range({num_iters}):
    start.record()
    _ = rmsnorm(x, w)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end) * 1000)

times.sort()
print(json.dumps({{
    "mean_us": statistics.mean(times),
    "p50_us": times[len(times)//2],
    "p95_us": times[int(len(times)*0.95)],
    "p99_us": times[int(len(times)*0.99)],
}}))
"""

    def _gen_rope_script(
        self, params: Dict[str, Any], num_warmup: int, num_iters: int
    ) -> str:
        bs = params.get("batch_size", 1)
        seqlen = params.get("seq_len", 2048)
        nheads = params.get("num_heads", 32)
        hdim = params.get("head_dim", 128)
        return f"""\
import torch, json, statistics
device = "cuda"
x = torch.randn({bs}, {seqlen}, {nheads}, {hdim}, dtype=torch.float16, device=device)

def apply_rope(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for _ in range({num_warmup}):
    _ = apply_rope(x)
torch.cuda.synchronize()

times = []
for _ in range({num_iters}):
    start.record()
    _ = apply_rope(x)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end) * 1000)

times.sort()
print(json.dumps({{
    "mean_us": statistics.mean(times),
    "p50_us": times[len(times)//2],
    "p95_us": times[int(len(times)*0.95)],
    "p99_us": times[int(len(times)*0.99)],
}}))
"""

    def _gen_moe_script(
        self, params: Dict[str, Any], num_warmup: int, num_iters: int
    ) -> str:
        bs = params.get("batch_size", 64)
        seqlen = params.get("seq_len", 512)
        hidden = params.get("hidden_dim", 2048)
        n_experts = params.get("num_experts", 64)
        top_k = params.get("top_k", 2)
        return f"""\
import torch, json, statistics
device = "cuda"
x = torch.randn({bs} * {seqlen}, {hidden}, dtype=torch.float16, device=device)
gate = torch.nn.Linear({hidden}, {n_experts}, bias=False, dtype=torch.float16, device=device)

def moe_dispatch(x, gate, top_k):
    logits = gate(x)
    _, indices = torch.topk(logits, top_k, dim=-1)
    return indices

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for _ in range({num_warmup}):
    _ = moe_dispatch(x, gate, {top_k})
torch.cuda.synchronize()

times = []
for _ in range({num_iters}):
    start.record()
    _ = moe_dispatch(x, gate, {top_k})
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end) * 1000)

times.sort()
print(json.dumps({{
    "mean_us": statistics.mean(times),
    "p50_us": times[len(times)//2],
    "p95_us": times[int(len(times)*0.95)],
    "p99_us": times[int(len(times)*0.99)],
}}))
"""
