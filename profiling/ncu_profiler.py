"""
profiling/ncu_profiler.py
--------------------------
Nsight Compute (ncu) integration for NVIDIA GPU hardware counter collection.

Runs `ncu` against a standalone benchmark script and parses the CSV/NVReport
output into a structured NvidiaCounters dataclass.

The profiler targets the top GPU-time kernel from the torch profiler trace
(e.g., the flash-attention kernel or the dominant GEMM) and collects:

  - SM throughput and Tensor Core utilisation
  - DRAM bandwidth utilisation
  - Achieved occupancy
  - Top warp stall reasons (L2 latency, register scoreboard, math throttle)
  - L1/L2 cache hit rates

Usage
-----
    profiler = NcuProfiler(gpu_type="H100")
    counters = await profiler.profile_kernel(
        kernel_name="flash_fwd_kernel",
        launch_cmd="python microbench/attention_bench.py --mode decode --bs 64",
        session_id="...",
    )
    print(counters.tensor_active_pct)  # Tensor Core utilisation %
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from profiling.hardware_counters import NvidiaCounters, HardwareCounters

log = logging.getLogger("profiling.ncu_profiler")

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Nsight Compute metric groups to collect
# These cover the most common bottleneck categories:
# compute, memory, occupancy, warp stall
_NCU_METRIC_SETS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__active_cycles_pm.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_fp16_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_fp32_cycles_active.avg.pct_of_peak_sustained_active",
    "dram__bytes_read.sum.per_second",
    "dram__bytes_write.sum.per_second",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sector_hit_rate.pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__maximum_warps_avg_pct",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_math_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
    "smsp__warp_issue_stalled_no_instructions_per_warp_active.pct",
]


class NcuProfiler:
    """
    Wraps the `ncu` CLI to collect hardware counters for a specific kernel.

    Requires:
      - `ncu` in PATH (part of CUDA Toolkit >= 11.0 or separate download)
      - Sufficient permissions (ncu needs CAP_SYS_ADMIN on Linux or
        NV_PERFWORKS_ENABLED=1 via admin setup)

    If `ncu` is not available or profiling fails, returns zeroed-out counters
    with `bottleneck="unavailable"` — never raises.
    """

    def __init__(
        self,
        gpu_type: str = "H100",
        ncu_path: Optional[str] = None,
    ) -> None:
        self._gpu_type = gpu_type
        self._ncu_path = ncu_path or shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
        self._available = os.path.isfile(self._ncu_path)
        if not self._available:
            log.warning("ncu not found at %s — hardware counter profiling disabled", self._ncu_path)

    @property
    def available(self) -> bool:
        return self._available

    async def profile_kernel(
        self,
        *,
        kernel_name: str,
        launch_cmd: str,
        session_id: str,
        timeout_sec: int = 600,
    ) -> HardwareCounters:
        """
        Profile a specific kernel using Nsight Compute.

        Parameters
        ----------
        kernel_name : str
            Kernel name to target (partial match, case-insensitive).
            e.g. "flash_fwd_kernel" or "gemm_kernel"
        launch_cmd : str
            Shell command that triggers the kernel (Python script or binary).
            e.g. "python microbench/attention_bench.py --mode decode --seq 2048"
        session_id : str
            Used for output file naming.
        timeout_sec : int
            Max time to wait for ncu to finish.

        Returns
        -------
        HardwareCounters (nvidia vendor)
        """
        empty = HardwareCounters(
            vendor="nvidia",
            gpu_type=self._gpu_type,
            kernel_name=kernel_name,
            session_id=session_id,
            nvidia=NvidiaCounters(kernel_name=kernel_name),
        )

        if not self._available:
            empty.primary_bottleneck = "unavailable"
            empty.bottleneck_detail = "ncu not found — hardware counter profiling disabled"
            return empty

        out_dir = _REPO_ROOT / "storage" / "profiles" / session_id / "ncu"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"ncu_{kernel_name[:40].replace('/', '_')}.csv"

        metrics_arg = ",".join(_NCU_METRIC_SETS)
        cmd = [
            self._ncu_path,
            "--csv",
            "--metrics", metrics_arg,
            "--kernel-name", f"regex:{re.escape(kernel_name[:40])}",
            "--kernel-name-base", "demangled",
            "--target-processes", "all",
            "--output", str(out_csv.with_suffix("")),
            "--force-overwrite",
        ] + launch_cmd.split()

        log.info("NcuProfiler: running ncu for kernel=%s", kernel_name[:40])
        log.debug("NcuProfiler command: %s", " ".join(cmd))

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(_REPO_ROOT),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)

            if proc.returncode != 0:
                log.warning(
                    "NcuProfiler: ncu exited %d — stderr: %s",
                    proc.returncode, stderr.decode()[:400],
                )
                empty.primary_bottleneck = "ncu_error"
                empty.bottleneck_detail = stderr.decode()[:200]
                return empty

            # ncu writes the .csv file; parse it
            csv_file = out_csv if out_csv.exists() else (out_csv.with_suffix(".csv"))
            if not csv_file.exists():
                log.warning("NcuProfiler: output CSV not found at %s", csv_file)
                empty.primary_bottleneck = "no_output"
                return empty

            counters = self._parse_ncu_csv(csv_file, kernel_name)
            log.info(
                "NcuProfiler: tensor_active=%.1f%% dram_bw=%.1f%% occupancy=%.1f%%",
                counters.tensor_active_pct, counters.dram_bw_utilisation_pct,
                counters.achieved_occupancy_pct,
            )
            return HardwareCounters.from_nvidia(counters, self._gpu_type, session_id)

        except asyncio.TimeoutError:
            log.warning("NcuProfiler: timed out after %ds", timeout_sec)
            empty.primary_bottleneck = "timeout"
            return empty
        except Exception as exc:
            log.warning("NcuProfiler error: %s", exc)
            empty.primary_bottleneck = "error"
            empty.bottleneck_detail = str(exc)
            return empty

    def _parse_ncu_csv(self, csv_path: Path, kernel_name: str) -> NvidiaCounters:
        """
        Parse Nsight Compute CSV output into NvidiaCounters.

        NCU CSV format:
          Row 0: "ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time",...
          Row 1: same headers as above (unit row)
          Row 2+: data rows, one per kernel invocation
        """
        counters = NvidiaCounters(kernel_name=kernel_name)
        raw: Dict[str, float] = {}

        try:
            with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                return counters

            # Find row matching our kernel name (first match)
            target_row: Optional[Dict[str, str]] = None
            for row in rows:
                kname = row.get("Kernel Name", "") or row.get("Metric Name", "")
                if kernel_name.lower()[:20] in kname.lower():
                    target_row = row
                    break
            if target_row is None and rows:
                # Fall back to first row
                target_row = rows[0]

            # Parse each metric column
            def _get(key: str) -> float:
                val_str = (target_row or {}).get(key, "0").strip().replace(",", "")
                try:
                    return float(val_str)
                except (ValueError, TypeError):
                    return 0.0

            # Map metric names to fields
            counters.sm_throughput_pct = _get("sm__throughput.avg.pct_of_peak_sustained_elapsed")
            counters.sm_active_pct = _get("sm__active_cycles_pm.avg.pct_of_peak_sustained_elapsed")
            counters.tensor_active_pct = _get("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active")
            counters.fp16_active_pct = _get("sm__pipe_fp16_cycles_active.avg.pct_of_peak_sustained_active")
            counters.fp32_active_pct = _get("sm__pipe_fp32_cycles_active.avg.pct_of_peak_sustained_active")

            dram_read_bytes_ps = _get("dram__bytes_read.sum.per_second")
            dram_write_bytes_ps = _get("dram__bytes_write.sum.per_second")
            counters.dram_read_gbps = dram_read_bytes_ps / 1e9
            counters.dram_write_gbps = dram_write_bytes_ps / 1e9
            counters.dram_bw_utilisation_pct = _get("dram__throughput.avg.pct_of_peak_sustained_elapsed")

            counters.l2_read_hit_rate_pct = _get("lts__t_sector_hit_rate.pct")
            counters.l1_hit_rate_pct = _get("l1tex__t_sector_hit_rate.pct")
            counters.achieved_occupancy_pct = _get("sm__warps_active.avg.pct_of_peak_sustained_active")
            counters.theoretical_occupancy_pct = _get("smsp__maximum_warps_avg_pct")

            counters.stall_memory_dependency_pct = _get(
                "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct"
            )
            counters.stall_long_scoreboard_pct = _get(
                "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"
            )
            counters.stall_math_pipe_throttle_pct = _get(
                "smsp__warp_issue_stalled_math_throttle_per_warp_active.pct"
            )
            counters.stall_not_selected_pct = _get(
                "smsp__warp_issue_stalled_not_selected_per_warp_active.pct"
            )
            counters.stall_no_instructions_pct = _get(
                "smsp__warp_issue_stalled_no_instructions_per_warp_active.pct"
            )

            # Derived bottleneck
            if counters.dram_bw_utilisation_pct > 60 and counters.tensor_active_pct < 40:
                counters.bottleneck = "memory_bandwidth"
            elif counters.tensor_active_pct > 70:
                counters.bottleneck = "compute"
            elif counters.achieved_occupancy_pct < 50:
                counters.bottleneck = "occupancy"
            elif counters.stall_long_scoreboard_pct > 20:
                counters.bottleneck = "memory_latency"
            else:
                counters.bottleneck = "mixed"

            counters.raw = raw

        except Exception as exc:
            log.warning("NcuProfiler CSV parse error: %s", exc)

        return counters
