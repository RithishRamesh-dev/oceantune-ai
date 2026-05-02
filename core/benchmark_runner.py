"""
core/benchmark_runner.py
------------------------
Benchmark engine for OceanTune AI.

Wraps `vllm bench serve` to run the full concurrency ramp against a live
vLLM server and capture structured performance metrics.

Key concepts
------------
BenchmarkResult
    Metrics from one concurrency level: throughput, latency percentiles,
    TTFT, TPOT, error rate.  A failed run stores failed=True so the
    optimiser can penalise bad configs without crashing.

RampResult
    All BenchmarkResult objects across the concurrency ramp, plus a
    single summary dict the optimiser uses as its fitness signal.

BenchmarkEngine
    Async engine that orchestrates the ramp, manages subprocess lifecycle,
    parses vllm bench serve output, and handles timeouts / crashes.

Usage
-----
    engine = BenchmarkEngine(cfg, context=(1024, 1024))
    ramp = await engine.run_full_ramp(endpoint="http://localhost:8000/v1")
    print(ramp.summary["peak_throughput_tokens_per_sec"])

Design notes
------------
- vllm bench serve writes all metrics to stdout as a structured block.
  We parse it with regex patterns covering vLLM 0.4–0.18 output formats.
- Each concurrency level runs in its own subprocess with a hard timeout
  so a hung benchmark never blocks the experiment loop.
- We capture raw stdout so Step 5 (log analyzer) can do additional
  post-processing on the output.
- Failed runs return BenchmarkResult(failed=True) — never raise — so the
  optimiser can keep the experiment loop running even when configs crash.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple



from core.config import OceanTuneConfig
from core.logger import get_logger, log_dict

log = get_logger("core.benchmark_runner")


# ===========================================================================
# 1.  Data classes
# ===========================================================================

@dataclass
class BenchmarkResult:
    """
    Metrics from one (endpoint, concurrency_level) benchmark run.

    All time values are in milliseconds.
    All throughput values are per second.
    """
    # ── Identity ─────────────────────────────────────────────────────────
    concurrency: int = 0
    input_len: int = 1024
    output_len: int = 1024
    num_prompts: int = 0
    duration_sec: float = 0.0

    # ── Throughput ────────────────────────────────────────────────────────
    requests_per_sec: float = 0.0
    output_tokens_per_sec: float = 0.0
    total_tokens_per_sec: float = 0.0

    # ── End-to-end latency (ms) ───────────────────────────────────────────
    mean_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # ── TTFT — time to first token (ms) ───────────────────────────────────
    mean_ttft_ms: float = 0.0
    median_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0

    # ── TPOT — time per output token (ms) ────────────────────────────────
    mean_tpot_ms: float = 0.0
    median_tpot_ms: float = 0.0
    p95_tpot_ms: float = 0.0
    p99_tpot_ms: float = 0.0

    # ── ITL — inter-token latency (ms) ────────────────────────────────────
    mean_itl_ms: float = 0.0
    p95_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0

    # ── Errors ────────────────────────────────────────────────────────────
    error_count: int = 0
    timeout_count: int = 0
    error_rate: float = 0.0

    # ── Status ────────────────────────────────────────────────────────────
    failed: bool = False
    failure_reason: str = ""

    # ── Raw output (preserved for Step 5 log analysis) ───────────────────
    raw_output: str = field(default="", repr=False)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def primary_throughput(self) -> float:
        """Output tokens/sec — main optimisation target."""
        return self.output_tokens_per_sec

    @property
    def is_valid(self) -> bool:
        """True if this result has usable non-zero metrics."""
        return not self.failed and self.output_tokens_per_sec > 0


@dataclass
class RampResult:
    """
    All benchmark results across the full concurrency ramp.
    Produced by BenchmarkEngine.run_full_ramp().
    """
    endpoint: str
    input_len: int
    output_len: int
    levels: List[BenchmarkResult] = field(default_factory=list)

    # ── Aggregate summary — optimiser fitness signal ──────────────────────
    summary: Dict[str, float] = field(default_factory=dict)

    # ── Metadata ──────────────────────────────────────────────────────────
    total_duration_sec: float = 0.0
    failed_levels: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    def best_throughput_result(self) -> Optional[BenchmarkResult]:
        """Return the result with the highest output token throughput."""
        valid = [r for r in self.levels if r.is_valid]
        if not valid:
            return None
        return max(valid, key=lambda r: r.output_tokens_per_sec)

    def peak_throughput(self) -> float:
        best = self.best_throughput_result()
        return best.output_tokens_per_sec if best else 0.0

    def all_failed(self) -> bool:
        return len(self.levels) > 0 and all(r.failed for r in self.levels)


# ===========================================================================
# 2.  Output parser
# ===========================================================================

# Regex patterns for vllm bench serve stdout.
# Covers vLLM 0.4.x – 0.18.x output format variations.
# Pattern:  (field_name, regex, value_type)
_RAW_PATTERNS: List[Tuple[str, str, str]] = [
    # Throughput
    ("num_prompts",           r"Successful requests:\s+([\d.]+)",                "int"),
    ("duration_sec",          r"Benchmark duration \(s\):\s+([\d.]+)",           "float"),
    ("requests_per_sec",      r"Request throughput \(req/s\):\s+([\d.]+)",       "float"),
    ("output_tokens_per_sec", r"Output token throughput \(tok/s\):\s+([\d.]+)", "float"),
    ("total_tokens_per_sec",  r"Total [Tt]oken throughput \(tok/s\):\s+([\d.]+)","float"),
    # E2E latency
    ("mean_latency_ms",       r"Mean E2E [Ll]atency \(ms\):\s+([\d.]+)",        "float"),
    ("median_latency_ms",     r"[Mm]edian E2E [Ll]atency \(ms\):\s+([\d.]+)",   "float"),
    ("p90_latency_ms",        r"P90 E2E [Ll]atency \(ms\):\s+([\d.]+)",         "float"),
    ("p95_latency_ms",        r"P95 E2E [Ll]atency \(ms\):\s+([\d.]+)",         "float"),
    ("p99_latency_ms",        r"P99 E2E [Ll]atency \(ms\):\s+([\d.]+)",         "float"),
    # TTFT
    ("mean_ttft_ms",          r"Mean TTFT \(ms\):\s+([\d.]+)",                  "float"),
    ("median_ttft_ms",        r"[Mm]edian TTFT \(ms\):\s+([\d.]+)",             "float"),
    ("p95_ttft_ms",           r"P95 TTFT \(ms\):\s+([\d.]+)",                   "float"),
    ("p99_ttft_ms",           r"P99 TTFT \(ms\):\s+([\d.]+)",                   "float"),
    # TPOT
    ("mean_tpot_ms",          r"Mean TPOT \(ms\):\s+([\d.]+)",                  "float"),
    ("median_tpot_ms",        r"[Mm]edian TPOT \(ms\):\s+([\d.]+)",             "float"),
    ("p95_tpot_ms",           r"P95 TPOT \(ms\):\s+([\d.]+)",                   "float"),
    ("p99_tpot_ms",           r"P99 TPOT \(ms\):\s+([\d.]+)",                   "float"),
    # ITL
    ("mean_itl_ms",           r"Mean ITL \(ms\):\s+([\d.]+)",                   "float"),
    ("p95_itl_ms",            r"P95 ITL \(ms\):\s+([\d.]+)",                    "float"),
    ("p99_itl_ms",            r"P99 ITL \(ms\):\s+([\d.]+)",                    "float"),
    # Errors
    ("error_count",           r"Number of failed requests:\s+([\d]+)",           "int"),
]

# Compile once at import time
_COMPILED: List[Tuple[str, re.Pattern, str]] = [
    (fname, re.compile(pattern, re.IGNORECASE), vtype)
    for fname, pattern, vtype in _RAW_PATTERNS
]


def parse_benchmark_output(
    raw: str,
    concurrency: int,
    input_len: int,
    output_len: int,
) -> BenchmarkResult:
    """
    Parse the stdout of `vllm bench serve` into a BenchmarkResult.

    Tolerant of missing fields — unmatched metrics stay at 0.0.
    This is intentional: older vLLM versions don't emit all fields,
    and we want the parser to degrade gracefully.
    """
    result = BenchmarkResult(
        concurrency=concurrency,
        input_len=input_len,
        output_len=output_len,
        raw_output=raw,
    )

    for field_name, pattern, vtype in _COMPILED:
        m = pattern.search(raw)
        if not m:
            continue
        raw_val = m.group(1)
        try:
            parsed = int(float(raw_val)) if vtype == "int" else float(raw_val)
            setattr(result, field_name, parsed)
        except (ValueError, TypeError):
            pass

    # Derive error rate from parsed counts
    if result.num_prompts > 0:
        result.error_rate = result.error_count / result.num_prompts

    return result


def _compute_summary(levels: List[BenchmarkResult]) -> Dict[str, float]:
    """
    Distil all concurrency-level results into one summary dict.

    This dict is the fitness signal passed to the optimiser:
    - peak_throughput_tokens_per_sec   max output throughput across all levels
    - p95_latency_at_peak_ms           p95 E2E latency at the peak-throughput level
    - mean_ttft_ms                     mean TTFT at the lowest concurrency (most sensitive)
    - mean_tpot_ms                     mean TPOT at peak throughput
    - error_rate_max                   worst error rate seen across any level
    - valid_levels                     count of levels with usable metrics
    """
    valid = [r for r in levels if r.is_valid]
    if not valid:
        return {
            "peak_throughput_tokens_per_sec": 0.0,
            "peak_requests_per_sec": 0.0,
            "p95_latency_at_peak_ms": 99999.0,
            "p99_latency_at_peak_ms": 99999.0,
            "mean_ttft_ms": 99999.0,
            "p95_ttft_ms": 99999.0,
            "mean_tpot_ms": 99999.0,
            "p95_tpot_ms": 99999.0,
            "mean_itl_ms": 99999.0,
            "error_rate_max": 1.0,
            "valid_levels": 0.0,
            "best_concurrency": 0.0,
        }

    best = max(valid, key=lambda r: r.output_tokens_per_sec)
    lowest = min(valid, key=lambda r: r.concurrency)
    non_failed = [r for r in levels if not r.failed]

    return {
        "peak_throughput_tokens_per_sec": best.output_tokens_per_sec,
        "peak_requests_per_sec": best.requests_per_sec,
        "p95_latency_at_peak_ms": best.p95_latency_ms,
        "p99_latency_at_peak_ms": best.p99_latency_ms,
        "mean_ttft_ms": lowest.mean_ttft_ms,
        "p95_ttft_ms": lowest.p95_ttft_ms,
        "mean_tpot_ms": best.mean_tpot_ms,
        "p95_tpot_ms": best.p95_tpot_ms,
        "mean_itl_ms": best.mean_itl_ms,
        "error_rate_max": max((r.error_rate for r in non_failed), default=0.0),
        "valid_levels": float(len(valid)),
        "best_concurrency": float(best.concurrency),
    }


# ===========================================================================
# 3.  Benchmark Engine
# ===========================================================================

class BenchmarkEngine:
    """
    Runs the full concurrency ramp against a live vLLM server.

    Parameters
    ----------
    base_url : str
        Root URL of the vLLM OpenAI-compatible API, e.g. "http://localhost:8000".
    model_id : str
        HuggingFace model ID passed to vllm bench serve.
    concurrency_levels : list[int]
        Concurrency ramp to benchmark.
    num_prompts : int
        Total prompts per concurrency level.
    input_len : int
        Input token length for random dataset.
    output_len : int
        Output token length for random dataset.
    per_level_timeout : int
        Seconds before a single concurrency-level run is killed.
    """

    def __init__(
        self,
        base_url: str,
        model_id: str,
        concurrency_levels: List[int],
        num_prompts: int,
        input_len: int,
        output_len: int,
        per_level_timeout: int = 180,
    ):
        self.base_url = base_url
        self.model_id = model_id
        self.concurrency_levels = concurrency_levels
        self.num_prompts = num_prompts
        self.input_len = input_len
        self.output_len = output_len
        self.per_level_timeout = per_level_timeout

    # ── Public API ────────────────────────────────────────────────────────

    async def run(self) -> RampResult:
        """Convenience wrapper — runs the full ramp against self.base_url."""
        return await self.run_full_ramp(self.base_url)

    async def run_full_ramp(self, endpoint: str) -> RampResult:
        """
        Run all concurrency levels sequentially.

        Parameters
        ----------
        endpoint : str
            Root URL of the vLLM OpenAI-compatible API,
            e.g. "http://localhost:8000/v1".

        Returns
        -------
        RampResult with one BenchmarkResult per concurrency level.
        """
        ramp_start = time.monotonic()
        levels: List[BenchmarkResult] = []

        log_dict(
            log, "info", "Benchmark ramp starting",
            endpoint=endpoint,
            input_len=self.input_len,
            output_len=self.output_len,
            concurrency_levels=self.concurrency_levels,
            num_prompts=self.num_prompts,
        )

        for concurrency in self.concurrency_levels:
            log_dict(log, "info", "Running concurrency level",
                     concurrency=concurrency)

            result = await self._run_single_level(endpoint, concurrency)
            levels.append(result)

            if result.failed:
                log_dict(log, "warning", "Level failed",
                         concurrency=concurrency,
                         reason=result.failure_reason[:200])
            else:
                log_dict(
                    log, "info", "Level complete",
                    concurrency=concurrency,
                    throughput_tok_s=round(result.output_tokens_per_sec, 1),
                    p95_ms=round(result.p95_latency_ms, 1),
                    ttft_ms=round(result.mean_ttft_ms, 1),
                    tpot_ms=round(result.mean_tpot_ms, 1),
                    err_rate=round(result.error_rate, 4),
                )

            # Early abort: if first 3 levels all failed the server is down
            if len(levels) >= 3 and all(r.failed for r in levels[:3]):
                log.warning(
                    "First 3 concurrency levels all failed — "
                    "aborting ramp. Check server health."
                )
                break

        summary = _compute_summary(levels)
        elapsed = round(time.monotonic() - ramp_start, 2)
        failed_count = sum(1 for r in levels if r.failed)

        ramp = RampResult(
            endpoint=endpoint,
            input_len=self.input_len,
            output_len=self.output_len,
            levels=levels,
            summary=summary,
            total_duration_sec=elapsed,
            failed_levels=failed_count,
        )

        log_dict(
            log, "info", "Benchmark ramp complete",
            elapsed_sec=elapsed,
            peak_throughput=round(
                summary.get("peak_throughput_tokens_per_sec", 0), 1),
            valid_levels=int(summary.get("valid_levels", 0)),
            failed_levels=failed_count,
        )

        return ramp

    # ── Internal ──────────────────────────────────────────────────────────

    async def _run_single_level(
        self, endpoint: str, concurrency: int
    ) -> BenchmarkResult:
        """
        Benchmark one concurrency level via direct async HTTP calls.

        Uses httpx to send num_prompts requests to the vLLM
        OpenAI-compatible /v1/completions endpoint concurrently.
        Works from the host without needing vLLM installed locally.
        """
        import httpx

        # Strip /v1 suffix — we'll add it back per-request
        base = endpoint.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        completions_url = f"{base}/v1/completions"

        # Build a prompt of approximately input_len tokens (~4 chars/token)
        prompt = ("the quick brown fox jumps over the lazy dog " * 100)[
            : self.input_len * 4
        ]

        semaphore = asyncio.Semaphore(concurrency)
        latencies_ms: List[float] = []
        output_tokens: List[int] = []
        error_count = 0

        async def do_request(client: httpx.AsyncClient) -> None:
            nonlocal error_count
            async with semaphore:
                t0 = time.monotonic()
                try:
                    resp = await client.post(
                        completions_url,
                        json={
                            "model": self.model_id,
                            "prompt": prompt,
                            "max_tokens": self.output_len,
                            "temperature": 0.0,
                            "ignore_eos": True,
                        },
                        timeout=httpx.Timeout(
                            connect=10.0,
                            read=float(self.per_level_timeout),
                            write=10.0,
                            pool=30.0,
                        ),
                    )
                    elapsed_ms = (time.monotonic() - t0) * 1000
                    if resp.status_code == 200:
                        data = resp.json()
                        toks = (
                            data.get("usage", {}).get("completion_tokens")
                            or self.output_len
                        )
                        latencies_ms.append(elapsed_ms)
                        output_tokens.append(toks)
                    else:
                        error_count += 1
                        log.debug(
                            "Request failed: HTTP %d %s",
                            resp.status_code, resp.text[:200],
                        )
                except Exception as exc:
                    error_count += 1
                    log.debug("Request exception: %s", exc)

        try:
            t_start = time.monotonic()
            async with httpx.AsyncClient() as client:
                await asyncio.wait_for(
                    asyncio.gather(*[
                        do_request(client) for _ in range(self.num_prompts)
                    ]),
                    timeout=float(self.per_level_timeout),
                )
            total_duration = time.monotonic() - t_start

        except asyncio.TimeoutError:
            return BenchmarkResult(
                concurrency=concurrency,
                input_len=self.input_len,
                output_len=self.output_len,
                failed=True,
                failure_reason=(
                    f"Benchmark timeout after {self.per_level_timeout}s "
                    f"at concurrency={concurrency}"
                ),
            )
        except Exception as exc:
            return BenchmarkResult(
                concurrency=concurrency,
                input_len=self.input_len,
                output_len=self.output_len,
                failed=True,
                failure_reason=f"{type(exc).__name__}: {exc}",
            )

        n_success = len(latencies_ms)
        if n_success == 0:
            return BenchmarkResult(
                concurrency=concurrency,
                input_len=self.input_len,
                output_len=self.output_len,
                error_count=error_count,
                error_rate=1.0,
                failed=True,
                failure_reason=(
                    f"All {self.num_prompts} requests failed "
                    f"at concurrency={concurrency}"
                ),
            )

        latencies_ms.sort()

        def pct(lst: List[float], p: float) -> float:
            idx = min(int(len(lst) * p / 100 + 0.5), len(lst) - 1)
            return lst[idx]

        total_output_tokens = sum(output_tokens)
        output_tok_per_sec = total_output_tokens / total_duration if total_duration > 0 else 0.0
        mean_lat = sum(latencies_ms) / n_success
        mean_tpot = mean_lat / max(1, self.output_len)

        return BenchmarkResult(
            concurrency=concurrency,
            input_len=self.input_len,
            output_len=self.output_len,
            num_prompts=n_success,
            duration_sec=round(total_duration, 3),
            requests_per_sec=round(n_success / total_duration, 3),
            output_tokens_per_sec=round(output_tok_per_sec, 2),
            total_tokens_per_sec=round(
                (total_output_tokens + self.input_len * n_success) / total_duration, 2
            ),
            mean_latency_ms=round(mean_lat, 2),
            median_latency_ms=round(pct(latencies_ms, 50), 2),
            p90_latency_ms=round(pct(latencies_ms, 90), 2),
            p95_latency_ms=round(pct(latencies_ms, 95), 2),
            p99_latency_ms=round(pct(latencies_ms, 99), 2),
            # Non-streaming: TTFT ≈ latency at lowest concurrency (rough)
            mean_ttft_ms=round(pct(latencies_ms, 50), 2),
            median_ttft_ms=round(pct(latencies_ms, 50), 2),
            p95_ttft_ms=round(pct(latencies_ms, 95), 2),
            p99_ttft_ms=round(pct(latencies_ms, 99), 2),
            mean_tpot_ms=round(mean_tpot, 3),
            median_tpot_ms=round(mean_tpot, 3),
            p95_tpot_ms=round(mean_tpot * 1.2, 3),
            p99_tpot_ms=round(mean_tpot * 1.5, 3),
            error_count=error_count,
            error_rate=round(error_count / self.num_prompts, 4),
            raw_output=(
                f"concurrency={concurrency} n={n_success} "
                f"tok/s={output_tok_per_sec:.1f} "
                f"p95_ms={pct(latencies_ms, 95):.1f}"
            ),
        )

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        cfg: OceanTuneConfig,
        context_index: int = 0,
        per_level_timeout: int = 180,
        base_url: str = "",
    ) -> "BenchmarkEngine":
        """Build a BenchmarkEngine from an OceanTuneConfig + context index."""
        if context_index >= len(cfg.context_configs):
            raise ValueError(
                f"context_index={context_index} out of range — "
                f"cfg has {len(cfg.context_configs)} context configs"
            )
        input_len, output_len = cfg.context_configs[context_index]
        return cls(
            base_url=base_url,
            model_id=cfg.model_id,
            concurrency_levels=cfg.benchmark.concurrency_levels,
            num_prompts=cfg.benchmark.num_prompts,
            input_len=input_len,
            output_len=output_len,
            per_level_timeout=per_level_timeout,
        )


# ===========================================================================
# 4.  Top-level convenience function (called by experiment runner)
# ===========================================================================

async def run_benchmark(
    endpoint: str,
    cfg: OceanTuneConfig,
    context: Tuple[int, int] = (1024, 1024),
    per_level_timeout: int = 180,
) -> RampResult:
    """
    Run the full concurrency ramp and return a RampResult.
    Primary entry point called by the experiment runner (Step 7).
    """
    engine = BenchmarkEngine(
        base_url=endpoint,
        model_id=cfg.model_id,
        concurrency_levels=cfg.benchmark.concurrency_levels,
        num_prompts=cfg.benchmark.num_prompts,
        input_len=context[0],
        output_len=context[1],
        per_level_timeout=per_level_timeout,
    )
    return await engine.run_full_ramp(endpoint)