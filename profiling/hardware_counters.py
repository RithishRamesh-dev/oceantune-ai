"""
profiling/hardware_counters.py
-------------------------------
Hardware counter data structures for NVIDIA (Nsight Compute) and AMD (rocprof/Omniperf).

These counters give a detailed picture of GPU utilisation that the PyTorch profiler
cannot provide: SM occupancy, memory bandwidth saturation, Tensor Core utilisation,
warp stall reasons, L1/L2 cache hit rates, register pressure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class NvidiaCounters:
    """
    Key metrics collected by Nsight Compute (ncu).

    All utilisation values are 0.0–100.0 (percent of peak).
    All bandwidth values are in GB/s.
    """
    # Compute utilisation
    sm_active_pct: float = 0.0          # sm__active_cycles.avg.pct_of_peak
    sm_throughput_pct: float = 0.0      # sm__throughput.avg.pct_of_peak_sustained_elapsed
    tensor_active_pct: float = 0.0      # sm__pipe_tensor_cycles_active.avg.pct_of_peak
    fp16_active_pct: float = 0.0        # sm__pipe_fp16_cycles_active.avg.pct_of_peak
    fp32_active_pct: float = 0.0        # sm__pipe_fp32_cycles_active.avg.pct_of_peak

    # Memory bandwidth
    dram_read_gbps: float = 0.0         # dram__bytes_read.sum.per_second / 1e9
    dram_write_gbps: float = 0.0        # dram__bytes_write.sum.per_second / 1e9
    dram_bw_utilisation_pct: float = 0.0  # dram__throughput.avg.pct_of_peak
    l2_read_hit_rate_pct: float = 0.0   # lts__t_sector_hit_rate.pct
    l1_hit_rate_pct: float = 0.0        # l1tex__t_sector_hit_rate.pct
    shared_mem_bw_gbps: float = 0.0     # l1tex__data_bank_reads.sum.per_second

    # Occupancy
    achieved_occupancy_pct: float = 0.0   # sm__warps_active.avg.pct_of_peak_sustained_active
    theoretical_occupancy_pct: float = 0.0
    active_warps_per_sm: float = 0.0

    # Warp stall reasons (% of all stall cycles)
    stall_memory_dependency_pct: float = 0.0  # Most common: waiting for global load
    stall_long_scoreboard_pct: float = 0.0    # Waiting for L2/DRAM
    stall_math_pipe_throttle_pct: float = 0.0 # Math pipe busy
    stall_not_selected_pct: float = 0.0       # Ready but not selected by scheduler
    stall_no_instructions_pct: float = 0.0    # Instruction cache miss

    # Bottleneck classification derived from counters
    bottleneck: str = "unknown"  # compute | memory_bandwidth | occupancy | launch_overhead

    # Raw metrics dict for inspection
    raw: Dict[str, float] = field(default_factory=dict)

    # Kernel name this was measured on
    kernel_name: str = ""
    duration_us: float = 0.0


@dataclass
class AmdCounters:
    """
    Key metrics collected by rocprof / Omniperf on AMD GPUs.

    CU = Compute Unit, VALU = Vector ALU, MFMA = Matrix Fused Multiply-Add.
    """
    # Compute utilisation
    valu_utilisation_pct: float = 0.0   # VALU (FP16/32/64) utilisation
    mfma_utilisation_pct: float = 0.0   # Matrix Fused Multiply-Add (MFMA) utilisation
    salu_utilisation_pct: float = 0.0   # Scalar ALU
    lds_utilisation_pct: float = 0.0    # Local Data Share (shared memory on AMD)
    branch_utilisation_pct: float = 0.0

    # Memory bandwidth
    hbm_read_gbps: float = 0.0
    hbm_write_gbps: float = 0.0
    hbm_bw_utilisation_pct: float = 0.0
    l2_cache_hit_rate_pct: float = 0.0
    l1_cache_hit_rate_pct: float = 0.0

    # Occupancy
    wavefront_occupancy: float = 0.0    # Avg active wavefronts per CU
    vector_wavefronts: int = 0          # Total vector wavefronts dispatched

    # LDS (Local Data Share) metrics
    lds_bank_conflicts: float = 0.0     # Bank conflicts per access
    lds_utilised: bool = False

    # Bottleneck classification
    bottleneck: str = "unknown"  # compute | memory_bandwidth | lds | wavefront_occupancy

    raw: Dict[str, float] = field(default_factory=dict)
    kernel_name: str = ""
    duration_us: float = 0.0


@dataclass
class HardwareCounters:
    """
    Unified hardware counter report for a profiling session.
    Contains either NvidiaCounters or AmdCounters depending on GPU vendor.
    """
    vendor: str                  # "nvidia" | "amd"
    gpu_type: str
    kernel_name: str
    session_id: str

    nvidia: Optional[NvidiaCounters] = None
    amd: Optional[AmdCounters] = None

    # Derived analysis
    is_memory_bound: bool = False
    is_compute_bound: bool = False
    is_occupancy_limited: bool = False
    is_launch_overhead: bool = False

    # Recommended action based on counters
    primary_bottleneck: str = "unknown"
    bottleneck_detail: str = ""

    # Roofline analysis
    arithmetic_intensity: float = 0.0   # FLOP / byte (how compute-intensive the kernel is)
    flop_per_byte_ridge_point: float = 0.0  # Hardware ridge point: peak_flops / peak_bw
    roofline_bound: str = "unknown"     # "compute" | "memory"

    @classmethod
    def from_nvidia(
        cls,
        counters: NvidiaCounters,
        gpu_type: str,
        session_id: str,
    ) -> "HardwareCounters":
        hc = cls(
            vendor="nvidia",
            gpu_type=gpu_type,
            kernel_name=counters.kernel_name,
            session_id=session_id,
            nvidia=counters,
        )
        hc._classify_nvidia()
        return hc

    @classmethod
    def from_amd(
        cls,
        counters: AmdCounters,
        gpu_type: str,
        session_id: str,
    ) -> "HardwareCounters":
        hc = cls(
            vendor="amd",
            gpu_type=gpu_type,
            kernel_name=counters.kernel_name,
            session_id=session_id,
            amd=counters,
        )
        hc._classify_amd()
        return hc

    def _classify_nvidia(self) -> None:
        c = self.nvidia
        if c is None:
            return
        # Memory bound: DRAM utilisation high, Tensor Core utilisation low
        if c.dram_bw_utilisation_pct > 60 and c.tensor_active_pct < 40:
            self.is_memory_bound = True
            self.primary_bottleneck = "memory_bandwidth"
            self.bottleneck_detail = (
                f"DRAM BW at {c.dram_bw_utilisation_pct:.1f}% of peak, "
                f"Tensor Core only at {c.tensor_active_pct:.1f}%"
            )
        elif c.tensor_active_pct > 60:
            self.is_compute_bound = True
            self.primary_bottleneck = "compute"
            self.bottleneck_detail = (
                f"Tensor Core at {c.tensor_active_pct:.1f}% utilisation"
            )
        elif c.achieved_occupancy_pct < 50:
            self.is_occupancy_limited = True
            self.primary_bottleneck = "occupancy"
            self.bottleneck_detail = (
                f"Occupancy only {c.achieved_occupancy_pct:.1f}% — "
                f"likely register pressure or shared memory over-subscription"
            )
        else:
            self.primary_bottleneck = "mixed"
            self.bottleneck_detail = "No single dominant bottleneck"

        # Warp stall analysis
        if c.stall_memory_dependency_pct > 30 or c.stall_long_scoreboard_pct > 20:
            self.bottleneck_detail += (
                f"; warp stalls: mem_dep={c.stall_memory_dependency_pct:.1f}% "
                f"long_scoreboard={c.stall_long_scoreboard_pct:.1f}%"
            )

    def _classify_amd(self) -> None:
        c = self.amd
        if c is None:
            return
        if c.hbm_bw_utilisation_pct > 60 and c.mfma_utilisation_pct < 40:
            self.is_memory_bound = True
            self.primary_bottleneck = "memory_bandwidth"
            self.bottleneck_detail = (
                f"HBM BW at {c.hbm_bw_utilisation_pct:.1f}% of peak, "
                f"MFMA only at {c.mfma_utilisation_pct:.1f}%"
            )
        elif c.mfma_utilisation_pct > 60:
            self.is_compute_bound = True
            self.primary_bottleneck = "compute"
            self.bottleneck_detail = (
                f"MFMA at {c.mfma_utilisation_pct:.1f}% utilisation"
            )
        elif c.wavefront_occupancy < 20:
            self.is_occupancy_limited = True
            self.primary_bottleneck = "wavefront_occupancy"
            self.bottleneck_detail = (
                f"Wavefront occupancy {c.wavefront_occupancy:.1f} — low scheduling pressure"
            )
        elif c.lds_bank_conflicts > 1.5:
            self.primary_bottleneck = "lds_bank_conflicts"
            self.bottleneck_detail = (
                f"LDS bank conflicts: {c.lds_bank_conflicts:.2f}x per access"
            )
        else:
            self.primary_bottleneck = "mixed"
            self.bottleneck_detail = "No single dominant bottleneck"

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"[{self.gpu_type}] kernel={self.kernel_name[:40]} "
            f"bottleneck={self.primary_bottleneck} "
            f"detail={self.bottleneck_detail[:80]}"
        )
