"""
microbench/roofline.py
-----------------------
Roofline model analysis for GPU kernel performance.

The roofline model classifies whether a kernel is limited by:
  - Compute (Tensor Cores / CUDA cores): achieved FLOP/s < peak FLOP/s
  - Memory bandwidth: achieved GB/s ≈ peak GB/s, FLOP/s constrained by data movement

The ridge point (FLOP/byte) is the arithmetic intensity at which a kernel
transitions from memory-bound to compute-bound.

Usage
-----
    analyzer = RooflineAnalyzer(gpu_type="H100")
    analysis = analyzer.analyze(
        op_flops=4e12,          # total FLOPs
        op_bytes=500e6,         # total bytes transferred
        duration_s=0.001,       # measured time in seconds
    )
    print(analysis.bound)           # "memory" | "compute"
    print(analysis.efficiency_pct)  # How close to roofline ceiling?
    print(analysis.recommendations) # What to do to improve
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# Hardware specs: peak FP16 TFLOP/s and memory BW GB/s
_GPU_SPECS = {
    "H100":   {"fp16_tflops": 989.0,  "bf16_tflops": 989.0,  "hbm_gbps": 3350.0, "l2_gbps": 12000.0},
    "H200":   {"fp16_tflops": 989.0,  "bf16_tflops": 989.0,  "hbm_gbps": 4800.0, "l2_gbps": 12000.0},
    "A100":   {"fp16_tflops": 312.0,  "bf16_tflops": 312.0,  "hbm_gbps": 2000.0, "l2_gbps": 6000.0},
    "A100_80G": {"fp16_tflops": 312.0, "bf16_tflops": 312.0, "hbm_gbps": 2000.0, "l2_gbps": 6000.0},
    "A6000":  {"fp16_tflops": 154.0,  "bf16_tflops": 154.0,  "hbm_gbps": 768.0,  "l2_gbps": 3000.0},
    "RTX4090":{"fp16_tflops": 165.0,  "bf16_tflops": 165.0,  "hbm_gbps": 1008.0, "l2_gbps": 6000.0},
    "MI300X": {"fp16_tflops": 1307.0, "bf16_tflops": 1307.0, "hbm_gbps": 5300.0, "l2_gbps": 16000.0},
    "MI325X": {"fp16_tflops": 1307.0, "bf16_tflops": 1307.0, "hbm_gbps": 6000.0, "l2_gbps": 16000.0},
    "MI350X": {"fp16_tflops": 2600.0, "bf16_tflops": 2600.0, "hbm_gbps": 8000.0, "l2_gbps": 24000.0},
}


@dataclass
class RooflinePoint:
    """A single kernel's position on the roofline plot."""
    # Input
    flops: float                    # Total FLOPs for the kernel invocation
    bytes_transferred: float        # Total bytes read+written (HBM)
    duration_s: float               # Wall time for the kernel

    # Derived
    arithmetic_intensity: float = 0.0   # FLOP/byte (x-axis of roofline plot)
    achieved_tflops: float = 0.0        # Achieved compute throughput
    achieved_gbps: float = 0.0          # Achieved memory bandwidth

    # Roofline classification
    ridge_point_flop_per_byte: float = 0.0  # Hardware ridge point
    bound: str = "unknown"          # "compute" | "memory"

    # Distance from roofline ceiling (0=at ceiling, positive=below ceiling)
    gap_to_ceiling_pct: float = 0.0     # How much performance is left on the table
    efficiency_pct: float = 0.0         # Fraction of theoretical max achieved

    # Recommendations
    recommendations: List[str] = field(default_factory=list)


@dataclass
class RooflineAnalysis:
    """Full roofline analysis for a profiling session."""
    gpu_type: str
    kernel_name: str

    # Measurements
    points: List[RooflinePoint] = field(default_factory=list)

    # Aggregate
    is_memory_bound: bool = False
    is_compute_bound: bool = False
    overall_efficiency_pct: float = 0.0

    # Summary
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)


class RooflineAnalyzer:
    """Compute roofline analysis for a kernel or set of kernels."""

    def __init__(self, gpu_type: str = "H100") -> None:
        self._gpu_type = gpu_type
        self._specs = _GPU_SPECS.get(gpu_type, _GPU_SPECS["H100"])
        self._peak_tflops = self._specs["fp16_tflops"]
        self._peak_gbps = self._specs["hbm_gbps"]
        self._ridge_point = (self._peak_tflops * 1e12) / (self._peak_gbps * 1e9)

    def analyze(
        self,
        *,
        kernel_name: str,
        op_flops: float,
        op_bytes: float,
        duration_s: float,
    ) -> RooflineAnalysis:
        """
        Analyse a single kernel's performance against the roofline model.

        Parameters
        ----------
        kernel_name : str
            Name of the kernel.
        op_flops : float
            Total floating point operations performed.
        op_bytes : float
            Total bytes transferred to/from HBM.
        duration_s : float
            Kernel duration in seconds.

        Returns
        -------
        RooflineAnalysis
        """
        analysis = RooflineAnalysis(gpu_type=self._gpu_type, kernel_name=kernel_name)

        if duration_s <= 0 or (op_flops <= 0 and op_bytes <= 0):
            analysis.summary = "Insufficient data for roofline analysis"
            return analysis

        point = RooflinePoint(
            flops=op_flops,
            bytes_transferred=op_bytes,
            duration_s=duration_s,
        )
        point.achieved_tflops = (op_flops / 1e12) / duration_s if op_flops > 0 else 0.0
        point.achieved_gbps = (op_bytes / 1e9) / duration_s if op_bytes > 0 else 0.0
        point.arithmetic_intensity = op_flops / op_bytes if op_bytes > 0 else float("inf")
        point.ridge_point_flop_per_byte = self._ridge_point

        # Classify bound
        if point.arithmetic_intensity > self._ridge_point:
            # Compute bound: throughput limited by compute
            point.bound = "compute"
            ceiling_tflops = self._peak_tflops
            point.efficiency_pct = (point.achieved_tflops / ceiling_tflops) * 100
            point.gap_to_ceiling_pct = 100 - point.efficiency_pct
        else:
            # Memory bound: throughput limited by HBM bandwidth
            point.bound = "memory"
            ceiling_gbps = self._peak_gbps
            point.efficiency_pct = (point.achieved_gbps / ceiling_gbps) * 100
            point.gap_to_ceiling_pct = 100 - point.efficiency_pct

        # Recommendations based on bound and efficiency
        point.recommendations = self._generate_recommendations(point)
        analysis.points.append(point)

        # Aggregate
        analysis.is_memory_bound = point.bound == "memory"
        analysis.is_compute_bound = point.bound == "compute"
        analysis.overall_efficiency_pct = point.efficiency_pct

        analysis.recommendations = point.recommendations
        analysis.summary = self._build_summary(kernel_name, point)

        return analysis

    def _generate_recommendations(self, point: RooflinePoint) -> List[str]:
        recs: List[str] = []

        if point.bound == "memory":
            if point.efficiency_pct < 30:
                recs.append(
                    "Memory bandwidth utilisation is very low (<30%). "
                    "This suggests the kernel has poor memory access patterns or "
                    "is limited by memory latency rather than bandwidth. "
                    "Consider: tiling to improve L1/L2 cache reuse, coalescing global "
                    "memory accesses, or using shared memory to stage data."
                )
            elif point.efficiency_pct < 60:
                recs.append(
                    "Memory bandwidth utilisation is moderate (30-60%). "
                    "Improving L2 cache hit rate or increasing arithmetic intensity "
                    "via kernel fusion could help. Consider fusing RMSNorm with "
                    "the preceding GEMM or fusing RoPE with Q/K projections."
                )
            else:
                recs.append(
                    "Kernel is near the memory bandwidth roofline — "
                    "further gains require either compressing data (FP8 KV cache) "
                    "or reducing bytes transferred (recompute vs cache trade-off)."
                )
            recs.append(
                f"Arithmetic intensity = {point.arithmetic_intensity:.1f} FLOP/byte "
                f"(ridge point = {point.ridge_point_flop_per_byte:.1f}). "
                "To shift to compute-bound: increase batch size, use GQA/MQA to "
                "reduce KV size, or implement paged attention with larger page sizes."
            )

        else:  # compute bound
            if point.efficiency_pct < 40:
                recs.append(
                    "Compute-bound but only {:.0f}% of peak TFLOP/s achieved. "
                    "Low Tensor Core utilisation suggests: warp divergence, "
                    "poor tile sizes for Tensor Core (need multiples of 16 for fp16), "
                    "or excessive register spilling. "
                    "Try: BLOCK_M/BLOCK_N = 128, BLOCK_K = 64 for flash-attention, "
                    "or use `triton.autotune` to sweep tile configurations.".format(
                        point.efficiency_pct
                    )
                )
            elif point.efficiency_pct < 70:
                recs.append(
                    "Compute-bound at {:.0f}% efficiency. "
                    "Remaining gap likely due to pipeline bubbles between matmul and softmax, "
                    "or non-optimal warp specialization. "
                    "Consider Flash-Attention v3 persistent kernel / warp specialization.".format(
                        point.efficiency_pct
                    )
                )
            else:
                recs.append(
                    f"Compute-bound at {point.efficiency_pct:.0f}% efficiency — "
                    "near theoretical maximum. "
                    "Further gains require algorithmic changes (e.g. sparsity, quantization) "
                    "rather than kernel tuning."
                )

        return recs

    def _build_summary(self, kernel_name: str, point: RooflinePoint) -> str:
        return (
            f"Kernel: {kernel_name[:50]}\n"
            f"  Bound:       {point.bound}\n"
            f"  Arith intens: {point.arithmetic_intensity:.1f} FLOP/byte "
            f"(ridge = {point.ridge_point_flop_per_byte:.1f})\n"
            f"  Achieved:    {point.achieved_tflops:.1f} TFLOP/s "
            f"({point.efficiency_pct:.1f}% of peak)\n"
            f"  Mem BW:      {point.achieved_gbps:.0f} GB/s "
            f"({(point.achieved_gbps / (self._peak_gbps or 1)) * 100:.1f}% of peak)\n"
            f"  Gap to ceil: {point.gap_to_ceiling_pct:.1f}%"
        )
