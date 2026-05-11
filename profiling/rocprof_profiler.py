"""
profiling/rocprof_profiler.py
------------------------------
rocprof / Omniperf integration for AMD GPU hardware counter collection.

Collects MFMA utilisation, HBM bandwidth saturation, wavefront occupancy,
LDS bank conflicts, and other AMD-specific metrics from MI300X/MI325X/MI350X.

Uses `rocprof` (ROCm < 5.7) or `rocprofv2` (ROCm >= 5.7) or `omniperf`
depending on what's available.

Usage
-----
    profiler = RocprofProfiler(gpu_type="MI300X")
    counters = await profiler.profile_kernel(
        kernel_name="flash_attention_fwd",
        launch_cmd="python microbench/attention_bench.py --mode decode --bs 64",
        session_id="...",
    )
    print(counters.mfma_utilisation_pct)
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from profiling.hardware_counters import AmdCounters, HardwareCounters

log = logging.getLogger("profiling.rocprof_profiler")

_REPO_ROOT = Path(__file__).resolve().parent.parent

# rocprof counter groups targeting the most important bottleneck dimensions
# These map to hardware counter names on CDNA3 (MI300X)
_ROCPROF_COUNTERS = [
    # MFMA (matrix) utilisation
    "TCC_EA_RDREQ_sum",           # L2 read requests
    "TCC_EA_WRREQ_sum",           # L2 write requests
    "FETCH_SIZE",                  # Total fetched from L2 (bytes)
    "WRITE_SIZE",                  # Total written from L2 (bytes)
    "GPUBusy",                     # GPU busy cycles
    "Wavefronts",                  # Total wavefronts dispatched
    "VALUInsts",                   # Vector ALU instructions
    "SALUInsts",                   # Scalar ALU instructions
    "SFetchInsts",                 # Scalar fetch instructions
    "VFetchInsts",                 # Vector fetch instructions
    "VWriteInsts",                 # Vector write instructions
    "FlatVMemInsts",               # Flat memory instructions
    "LDSInsts",                    # LDS instructions
    "GDSInsts",                    # GDS instructions
    "VALUUtilization",             # VALU utilisation
    "VALUBusy",                    # VALU busy cycles
    "SALUBusy",                    # SALU busy cycles
    "FetchSize",                   # HBM fetch bandwidth
    "WriteSize",                   # HBM write bandwidth
    "MemUnitBusy",                 # Memory unit busy cycles
    "MemUnitStalled",              # Memory unit stalled
    "WriteUnitStalled",            # Write unit stalled
    "LDSBankConflict",             # LDS bank conflicts
]


class RocprofProfiler:
    """
    Wraps rocprof / omniperf to collect AMD GPU hardware counters.

    Automatically detects which tool is available:
    1. omniperf (preferred for MI300X — gives more detailed metrics)
    2. rocprofv2
    3. rocprof

    Falls back gracefully if none are available.
    """

    def __init__(
        self,
        gpu_type: str = "MI300X",
        rocprof_path: Optional[str] = None,
    ) -> None:
        self._gpu_type = gpu_type
        self._tool, self._tool_path = self._detect_tool(rocprof_path)
        if self._tool == "none":
            log.warning("No ROCm profiling tool found — hardware counter profiling disabled")

    @property
    def available(self) -> bool:
        return self._tool != "none"

    def _detect_tool(self, explicit_path: Optional[str]) -> Tuple[str, str]:
        if explicit_path and os.path.isfile(explicit_path):
            return "rocprof", explicit_path
        # Detection order: omniperf > rocprofv2 > rocprof
        for name in ("omniperf", "rocprofv2", "rocprof"):
            path = shutil.which(name)
            if path:
                log.info("RocprofProfiler: using %s at %s", name, path)
                return name, path
        return "none", ""

    async def profile_kernel(
        self,
        *,
        kernel_name: str,
        launch_cmd: str,
        session_id: str,
        timeout_sec: int = 600,
    ) -> HardwareCounters:
        """
        Profile a kernel using the available ROCm profiling tool.
        Returns HardwareCounters with AmdCounters populated.
        Never raises.
        """
        empty = HardwareCounters(
            vendor="amd",
            gpu_type=self._gpu_type,
            kernel_name=kernel_name,
            session_id=session_id,
            amd=AmdCounters(kernel_name=kernel_name),
        )

        if not self.available:
            empty.primary_bottleneck = "unavailable"
            empty.bottleneck_detail = "No ROCm profiling tool found"
            return empty

        if self._tool == "omniperf":
            return await self._profile_omniperf(
                kernel_name=kernel_name,
                launch_cmd=launch_cmd,
                session_id=session_id,
                timeout_sec=timeout_sec,
            )
        else:
            return await self._profile_rocprof(
                kernel_name=kernel_name,
                launch_cmd=launch_cmd,
                session_id=session_id,
                timeout_sec=timeout_sec,
            )

    async def _profile_rocprof(
        self,
        *,
        kernel_name: str,
        launch_cmd: str,
        session_id: str,
        timeout_sec: int,
    ) -> HardwareCounters:
        """Run rocprof / rocprofv2 with hardware counter collection."""
        out_dir = _REPO_ROOT / "storage" / "profiles" / session_id / "rocprof"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write counter input file for rocprof
        counter_file = out_dir / "counters.txt"
        counter_file.write_text("\n".join(_ROCPROF_COUNTERS) + "\n")

        out_csv = out_dir / "rocprof_output.csv"
        cmd = [
            self._tool_path,
            "--stats",
            "-i", str(counter_file),
            "-o", str(out_csv),
            "--kernel-name", kernel_name[:60],
        ] + launch_cmd.split()

        log.info("RocprofProfiler: running %s for kernel=%s", self._tool, kernel_name[:40])

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
                    "RocprofProfiler: %s exited %d — stderr: %s",
                    self._tool, proc.returncode, stderr.decode()[:400],
                )

            # Parse output CSV
            if out_csv.exists():
                counters = self._parse_rocprof_csv(out_csv, kernel_name)
                return HardwareCounters.from_amd(counters, self._gpu_type, session_id)

            # rocprof may output <out_csv_stem>.stats.csv
            stats_csv = out_csv.parent / (out_csv.stem + ".stats.csv")
            if stats_csv.exists():
                counters = self._parse_rocprof_csv(stats_csv, kernel_name)
                return HardwareCounters.from_amd(counters, self._gpu_type, session_id)

            log.warning("RocprofProfiler: no output CSV found")
            empty = HardwareCounters(
                vendor="amd", gpu_type=self._gpu_type,
                kernel_name=kernel_name, session_id=session_id,
                amd=AmdCounters(kernel_name=kernel_name),
            )
            empty.primary_bottleneck = "no_output"
            return empty

        except asyncio.TimeoutError:
            log.warning("RocprofProfiler: timed out after %ds", timeout_sec)
            empty = HardwareCounters(
                vendor="amd", gpu_type=self._gpu_type,
                kernel_name=kernel_name, session_id=session_id,
                amd=AmdCounters(kernel_name=kernel_name),
            )
            empty.primary_bottleneck = "timeout"
            return empty
        except Exception as exc:
            log.warning("RocprofProfiler error: %s", exc)
            empty = HardwareCounters(
                vendor="amd", gpu_type=self._gpu_type,
                kernel_name=kernel_name, session_id=session_id,
                amd=AmdCounters(kernel_name=kernel_name),
            )
            empty.primary_bottleneck = "error"
            empty.bottleneck_detail = str(exc)
            return empty

    async def _profile_omniperf(
        self,
        *,
        kernel_name: str,
        launch_cmd: str,
        session_id: str,
        timeout_sec: int,
    ) -> HardwareCounters:
        """Run Omniperf for MI300X-specific deep analysis."""
        out_dir = _REPO_ROOT / "storage" / "profiles" / session_id / "omniperf"
        out_dir.mkdir(parents=True, exist_ok=True)

        # omniperf profile mode: collect metrics
        cmd = [
            self._tool_path,
            "profile",
            "--name", f"oceantune_{session_id[:8]}",
            "--path", str(out_dir),
            "--roof-only",
            "--", *launch_cmd.split(),
        ]

        log.info("RocprofProfiler(omniperf): profiling kernel=%s", kernel_name[:40])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(_REPO_ROOT),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)

            # Omniperf writes JSON results — find and parse them
            json_files = list(out_dir.glob("**/*.json"))
            if json_files:
                jf = json_files[0]
                with open(jf) as f:
                    raw = json.load(f)
                counters = self._parse_omniperf_json(raw, kernel_name)
                return HardwareCounters.from_amd(counters, self._gpu_type, session_id)

            log.warning("RocprofProfiler: omniperf produced no JSON output")
            empty = HardwareCounters(
                vendor="amd", gpu_type=self._gpu_type,
                kernel_name=kernel_name, session_id=session_id,
                amd=AmdCounters(kernel_name=kernel_name),
            )
            empty.primary_bottleneck = "no_output"
            return empty

        except Exception as exc:
            log.warning("Omniperf error: %s", exc)
            empty = HardwareCounters(
                vendor="amd", gpu_type=self._gpu_type,
                kernel_name=kernel_name, session_id=session_id,
                amd=AmdCounters(kernel_name=kernel_name),
            )
            empty.primary_bottleneck = "error"
            empty.bottleneck_detail = str(exc)
            return empty

    def _parse_rocprof_csv(self, csv_path: Path, kernel_name: str) -> AmdCounters:
        """Parse rocprof CSV output into AmdCounters."""
        counters = AmdCounters(kernel_name=kernel_name)
        try:
            with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                rows = [r for r in reader if kernel_name.lower()[:20] in (r.get("KernelName", "")).lower()]
                if not rows:
                    reader = csv.DictReader(open(csv_path, encoding="utf-8", errors="replace"))
                    rows = list(reader)
                if not rows:
                    return counters
                row = rows[0]

            def _get(key: str) -> float:
                val = row.get(key, "0").strip().replace(",", "")
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return 0.0

            counters.valu_utilisation_pct = _get("VALUUtilization")
            counters.lds_bank_conflicts = _get("LDSBankConflict")

            # Estimate HBM BW from fetch/write sizes and duration
            duration_ns = _get("DurationNs") or 1.0
            fetch_bytes = _get("FetchSize") * 1024  # rocprof reports in KB
            write_bytes = _get("WriteSize") * 1024
            counters.hbm_read_gbps = (fetch_bytes / duration_ns)  # bytes/ns = GB/s
            counters.hbm_write_gbps = (write_bytes / duration_ns)

            # Wavefront occupancy
            counters.wavefront_occupancy = _get("Wavefronts") / max(_get("GPUBusy"), 1.0) * 64

            counters.raw = {k: _get(k) for k in _ROCPROF_COUNTERS if k in row}

        except Exception as exc:
            log.warning("RocprofProfiler CSV parse error: %s", exc)

        return counters

    def _parse_omniperf_json(self, raw: Dict[str, Any], kernel_name: str) -> AmdCounters:
        """Parse Omniperf JSON output into AmdCounters."""
        counters = AmdCounters(kernel_name=kernel_name)
        try:
            # Omniperf stores metrics per-kernel in a nested structure
            # Attempt to find relevant kernel data
            kernels = raw.get("kernels", raw.get("results", {}))
            if isinstance(kernels, dict):
                kernel_data = kernels.get(kernel_name, next(iter(kernels.values()), {}))
            elif isinstance(kernels, list) and kernels:
                kernel_data = next(
                    (k for k in kernels if kernel_name.lower()[:20] in str(k.get("name", "")).lower()),
                    kernels[0]
                )
            else:
                kernel_data = raw

            metrics = kernel_data if isinstance(kernel_data, dict) else {}

            def _get(key: str, default: float = 0.0) -> float:
                val = metrics.get(key, default)
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return default

            counters.mfma_utilisation_pct = _get("mfma_utilization_pct", 0.0)
            counters.valu_utilisation_pct = _get("valu_utilization_pct", 0.0)
            counters.hbm_bw_utilisation_pct = _get("hbm_bw_utilization_pct", 0.0)
            counters.hbm_read_gbps = _get("hbm_read_gbps", 0.0)
            counters.hbm_write_gbps = _get("hbm_write_gbps", 0.0)
            counters.l2_cache_hit_rate_pct = _get("l2_cache_hit_rate", 0.0)
            counters.wavefront_occupancy = _get("wavefront_occupancy", 0.0)
            counters.lds_bank_conflicts = _get("lds_bank_conflicts", 0.0)

        except Exception as exc:
            log.warning("Omniperf JSON parse error: %s", exc)

        return counters
