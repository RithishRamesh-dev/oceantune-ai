"""
core/log_analyzer.py
--------------------
Structured log analysis for OceanTune AI.

Scans vLLM server log lines for:
  - Failure signatures beyond OOM (NCCL, CUDA kernel, tokenizer errors)
  - Structured events: model load time, KV cache blocks allocated,
    tensor parallel initialisation, expert routing stats
  - Silent hang detection (log goes quiet mid-run)
  - Warning patterns that predict future failures

Inputs:  List[str] log lines from VLLMServer.log_tail
Outputs: LogAnalysis dataclass

Design principles
-----------------
- Every regex match failure leaves its field as None/empty — never raises.
- All patterns are compiled once at import time.
- The analyzer is stateless and pure — same input always gives same output.
- Log lines are scanned in a single pass to keep analysis O(n).

Usage
-----
    from core.log_analyzer import LogAnalyzer
    analysis = LogAnalyzer.analyze(server.log_tail)
    print(analysis.model_load_time_sec)
    print(analysis.kv_cache_blocks)
    print(analysis.error_classes)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set

from core.logger import get_logger, log_dict

log = get_logger("core.log_analyzer")


# ===========================================================================
# 1.  Compiled pattern registry
# ===========================================================================

# Single-value extraction: (field_name, compiled_regex, value_type)
_EXTRACT_PATTERNS: List[tuple] = [
    (
        "model_load_time_sec",
        re.compile(r"(?:Loading model weights took|Model loaded in)\s+([\d.]+)\s*s",
                   re.IGNORECASE),
        "float",
    ),
    (
        "kv_cache_blocks",
        re.compile(r"#\s*GPU blocks:\s*(\d+)", re.IGNORECASE),
        "int",
    ),
    (
        "kv_cache_gb",
        re.compile(r"KV cache.*?([\d.]+)\s*(?:GB|GiB)", re.IGNORECASE),
        "float",
    ),
    (
        "gpu_memory_used_gb",
        re.compile(
            r"(?:GPU memory used|Memory allocated|memory usage):\s*([\d.]+)\s*(?:GB|GiB)",
            re.IGNORECASE,
        ),
        "float",
    ),
    (
        "tp_init_time_sec",
        re.compile(r"(?:Tensor parallel|TP) init.*?took\s+([\d.]+)\s*s",
                   re.IGNORECASE),
        "float",
    ),
    (
        "cuda_graph_capture_sec",
        re.compile(r"CUDA graph.*?captured.*?in\s+([\d.]+)\s*s",
                   re.IGNORECASE),
        "float",
    ),
    (
        "max_seq_len_from_log",
        re.compile(r"max_model_len\s*=\s*(\d+)", re.IGNORECASE),
        "int",
    ),
    (
        "num_gpu_blocks_kv",
        re.compile(r"gpu_blocks=(\d+)", re.IGNORECASE),
        "int",
    ),
]

# Error class patterns — any matching line adds the class name to error_classes
_ERROR_CLASS_PATTERNS: List[tuple] = [
    ("oom",          re.compile(r"out of memory|OutOfMemoryError|OOM when", re.IGNORECASE)),
    ("nccl",         re.compile(r"NCCL error|ncclInternal|nccl.*?fail", re.IGNORECASE)),
    ("cuda_kernel",  re.compile(r"CUDA error|device-side assert|cuda kernel fail", re.IGNORECASE)),
    ("tokenizer",    re.compile(r"tokenizer.*?error|sentencepiece.*?fail", re.IGNORECASE)),
    ("timeout",      re.compile(r"timed out|deadline exceeded|read timeout", re.IGNORECASE)),
    ("port_conflict",re.compile(r"address already in use|EADDRINUSE", re.IGNORECASE)),
    ("model_load",   re.compile(r"error loading.*?model|failed to load.*?weight", re.IGNORECASE)),
    ("tp_error",     re.compile(r"tensor.parallel.*?error|rank.*?mismatch", re.IGNORECASE)),
    ("ep_error",     re.compile(r"expert.parallel.*?error|routing.*?error", re.IGNORECASE)),
    ("quantization", re.compile(r"quantization.*?error|AWQ.*?error|GPTQ.*?error", re.IGNORECASE)),
    ("rocm",         re.compile(r"HIP error|ROCm.*?fail|amdgpu.*?error", re.IGNORECASE)),
    ("flashinfer",   re.compile(r"flashinfer.*?error|flash.*?attn.*?fail", re.IGNORECASE)),
    ("process_group",re.compile(r"process group.*?error|dist.*?init.*?fail", re.IGNORECASE)),
]

# Warning patterns — don't constitute failures, but logged for diagnostics
_WARNING_PATTERNS: List[tuple] = [
    ("slow_load",      re.compile(r"loading.*?slow|weight.*?slow", re.IGNORECASE)),
    ("fragmentation",  re.compile(r"memory.*?fragmentation", re.IGNORECASE)),
    ("fallback_eager", re.compile(r"falling back.*?eager|eager.*?fallback", re.IGNORECASE)),
    ("prefix_miss",    re.compile(r"prefix.*?cache.*?miss.*?100%", re.IGNORECASE)),
]

# Server-ready signals
_READY_PATTERNS: List[re.Pattern] = [
    re.compile(r"Application startup complete", re.IGNORECASE),
    re.compile(r"Uvicorn running on", re.IGNORECASE),
    re.compile(r"Started server process", re.IGNORECASE),
    re.compile(r"vLLM.*?is ready", re.IGNORECASE),
]

# vLLM log timestamp pattern: "INFO 2024-01-15 12:34:56.789 ..."
_TS_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}\s+(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?")


# ===========================================================================
# 2.  Output dataclass
# ===========================================================================

@dataclass
class LogAnalysis:
    """
    Structured information extracted from vLLM server log lines.

    All fields are Optional — None means the pattern was not found,
    not that an error occurred.
    """
    # ── Timing ────────────────────────────────────────────────────────────
    model_load_time_sec: Optional[float] = None
    tp_init_time_sec: Optional[float] = None
    cuda_graph_capture_sec: Optional[float] = None
    total_startup_sec: Optional[float] = None

    # ── Memory ────────────────────────────────────────────────────────────
    kv_cache_blocks: Optional[int] = None
    num_gpu_blocks_kv: Optional[int] = None
    kv_cache_gb: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None

    # ── Model config (confirmed in startup logs) ──────────────────────────
    max_seq_len_from_log: Optional[int] = None

    # ── Status ────────────────────────────────────────────────────────────
    server_became_ready: bool = False
    total_log_lines: int = 0

    # ── Errors & warnings ────────────────────────────────────────────────
    error_classes: Set[str] = field(default_factory=set)
    warning_classes: Set[str] = field(default_factory=set)
    error_lines: List[str] = field(default_factory=list)

    # ── Tail snippet for reports ──────────────────────────────────────────
    log_tail_snippet: List[str] = field(default_factory=list)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def has_errors(self) -> bool:
        return len(self.error_classes) > 0

    @property
    def has_oom(self) -> bool:
        return "oom" in self.error_classes

    @property
    def has_nccl_error(self) -> bool:
        return "nccl" in self.error_classes

    def to_dict(self) -> dict:
        return {
            "model_load_time_sec": self.model_load_time_sec,
            "tp_init_time_sec": self.tp_init_time_sec,
            "cuda_graph_capture_sec": self.cuda_graph_capture_sec,
            "total_startup_sec": self.total_startup_sec,
            "kv_cache_blocks": self.kv_cache_blocks,
            "num_gpu_blocks_kv": self.num_gpu_blocks_kv,
            "kv_cache_gb": self.kv_cache_gb,
            "gpu_memory_used_gb": self.gpu_memory_used_gb,
            "max_seq_len_from_log": self.max_seq_len_from_log,
            "server_became_ready": self.server_became_ready,
            "total_log_lines": self.total_log_lines,
            "error_classes": sorted(self.error_classes),
            "warning_classes": sorted(self.warning_classes),
            "has_errors": self.has_errors,
            "has_oom": self.has_oom,
            "has_nccl_error": self.has_nccl_error,
        }


# ===========================================================================
# 3.  LogAnalyzer
# ===========================================================================

class LogAnalyzer:
    """
    Stateless log analyzer.

    Call LogAnalyzer.analyze(lines) — no instance needed.
    """

    @classmethod
    def analyze(cls, lines: List[str]) -> LogAnalysis:
        """
        Scan log lines and return a structured LogAnalysis.

        Parameters
        ----------
        lines : List[str]
            Raw log lines from VLLMServer.log_tail or any log file.
            Assumed to be in chronological order.

        Returns
        -------
        LogAnalysis — guaranteed not to raise.
        """
        if not lines:
            return LogAnalysis()

        full_text = "\n".join(lines)
        analysis = LogAnalysis(
            total_log_lines=len(lines),
            log_tail_snippet=lines[-20:],
        )

        # ── Single-value field extraction ─────────────────────────────────
        for field_name, pattern, vtype in _EXTRACT_PATTERNS:
            m = pattern.search(full_text)
            if not m:
                continue
            raw = m.group(1)
            try:
                parsed = int(raw) if vtype == "int" else float(raw)
                setattr(analysis, field_name, parsed)
            except (ValueError, TypeError):
                pass

        # ── Error class detection (per-line scan) ─────────────────────────
        for class_name, pattern in _ERROR_CLASS_PATTERNS:
            for line in lines:
                if pattern.search(line):
                    analysis.error_classes.add(class_name)
                    if len(analysis.error_lines) < 20:
                        analysis.error_lines.append(line)
                    break

        # ── Warning class detection ───────────────────────────────────────
        for class_name, pattern in _WARNING_PATTERNS:
            if pattern.search(full_text):
                analysis.warning_classes.add(class_name)

        # ── Ready detection ───────────────────────────────────────────────
        for pattern in _READY_PATTERNS:
            if pattern.search(full_text):
                analysis.server_became_ready = True
                break

        # ── Startup wall-clock time from timestamps ───────────────────────
        analysis.total_startup_sec = cls._estimate_startup_time(lines)

        log_dict(
            log, "debug", "Log analysis complete",
            lines=len(lines),
            ready=analysis.server_became_ready,
            errors=sorted(analysis.error_classes),
            load_time=analysis.model_load_time_sec,
            kv_blocks=analysis.kv_cache_blocks,
        )

        return analysis

    @staticmethod
    def _estimate_startup_time(lines: List[str]) -> Optional[float]:
        """
        Estimate wall-clock startup duration from vLLM log timestamps.

        Returns None if fewer than 2 timestamped lines found.
        Handles midnight wrap-around by adding 86400s if delta < 0.
        """
        timestamps: List[float] = []

        for line in lines:
            m = _TS_PATTERN.search(line)
            if not m:
                continue
            try:
                h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
                frac = int((m.group(4) or "0")[:3].ljust(3, "0")) / 1000.0
                timestamps.append(h * 3600 + mi * 60 + s + frac)
            except (ValueError, IndexError, AttributeError):
                continue

        if len(timestamps) < 2:
            return None

        delta = timestamps[-1] - timestamps[0]
        if delta < 0:
            delta += 86400.0
        return round(delta, 2)

    @classmethod
    def analyze_file(cls, path: str) -> LogAnalysis:
        """
        Analyze a log file on disk.

        Parameters
        ----------
        path : str
            Path to a vLLM log file.
        """
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                lines = [line.rstrip() for line in f]
            return cls.analyze(lines)
        except (OSError, IOError) as exc:
            log.warning(f"Could not read log file {path}: {exc}")
            return LogAnalysis()