"""
core/snr_contract.py
--------------------
Numerical SNR pre-filter + statistical KEEP contract.

Hyperloom separates correctness (SNR ≥ ~30 dB) from performance KEEP
(noise-adjusted speedup vs incumbent). OceanTune Stage 4 historically used a
fixed 1% microbench threshold; this module makes the dual-gate explicit.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger("core.snr_contract")

DEFAULT_SNR_THRESHOLD_DB = 30.0
DEFAULT_KEEP_MIN_MARGIN = 0.001  # 0.1% relative margin over incumbent
DEFAULT_MIN_SPEEDUP = 1.01      # 1% absolute speedup floor


@dataclass
class SNRResult:
    passed: bool
    snr_db: float
    threshold_db: float = DEFAULT_SNR_THRESHOLD_DB
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KeepEval:
    keep: bool
    required_speedup: float
    observed_speedup: float
    incumbent_speedup: float = 1.0
    sigma: Optional[float] = None
    reason: str = ""
    snr: Optional[SNRResult] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.snr is not None:
            d["snr"] = self.snr.to_dict()
        return d


def snr_db_from_tensors(
    reference: Sequence[float],
    candidate: Sequence[float],
    *,
    eps: float = 1e-12,
) -> float:
    """
    Compute SNR in dB: 10 * log10(signal_power / noise_power).

    ``reference`` is the trusted output; ``candidate`` is the kernel under test.
    Both must be same length flat sequences.
    """
    if len(reference) == 0 or len(reference) != len(candidate):
        return 0.0
    sig = 0.0
    noise = 0.0
    for r, c in zip(reference, candidate):
        rf = float(r)
        cf = float(c)
        sig += rf * rf
        d = rf - cf
        noise += d * d
    if noise < eps:
        return 200.0  # effectively exact
    if sig < eps:
        return 0.0
    return 10.0 * math.log10(sig / noise)


def validate_snr(
    reference: Sequence[float],
    candidate: Sequence[float],
    *,
    threshold_db: float = DEFAULT_SNR_THRESHOLD_DB,
) -> SNRResult:
    """SNR pre-filter — necessary but not sufficient for KEEP."""
    snr = snr_db_from_tensors(reference, candidate)
    passed = snr >= threshold_db
    return SNRResult(
        passed=passed,
        snr_db=snr,
        threshold_db=threshold_db,
        reason="snr_ok" if passed else f"snr_below_{threshold_db}_db",
    )


def required_keep_speedup(
    incumbent_mean: float = 1.0,
    *,
    sigma: Optional[float] = None,
    min_margin: float = DEFAULT_KEEP_MIN_MARGIN,
    min_speedup: float = DEFAULT_MIN_SPEEDUP,
    n_samples: int = 3,
) -> float:
    """
    Minimum mean speedup vs pristine needed to KEEP.

    Uses a simple noise margin: incumbent + t*sigma/sqrt(n) style floor,
    never below ``min_speedup``.
    """
    base = max(float(incumbent_mean), 1.0)
    margin = base * min_margin
    if sigma is not None and n_samples > 0:
        # Approximate one-sided 95% t≈2 for small n
        margin = max(margin, 2.0 * float(sigma) / math.sqrt(n_samples))
    return max(min_speedup, base + margin)


def evaluate_keep(
    measurement_speedups: Sequence[float],
    *,
    incumbent_mean_speedup: float = 1.0,
    sigma: Optional[float] = None,
    min_margin: float = DEFAULT_KEEP_MIN_MARGIN,
    min_speedup: float = DEFAULT_MIN_SPEEDUP,
    snr: Optional[SNRResult] = None,
) -> KeepEval:
    """
    Performance KEEP after SNR (if provided) has passed.

    ``measurement_speedups`` are pristine-relative ratios (e.g. 1.05 = +5%).
    """
    if snr is not None and not snr.passed:
        return KeepEval(
            keep=False,
            required_speedup=0.0,
            observed_speedup=0.0,
            incumbent_speedup=incumbent_mean_speedup,
            sigma=sigma,
            reason=f"snr_prefilter_failed:{snr.reason}",
            snr=snr,
        )

    vals = [float(x) for x in measurement_speedups if x is not None]
    if not vals:
        return KeepEval(
            keep=False,
            required_speedup=min_speedup,
            observed_speedup=0.0,
            incumbent_speedup=incumbent_mean_speedup,
            sigma=sigma,
            reason="no_measurements",
            snr=snr,
        )

    observed = sum(vals) / len(vals)
    if sigma is None and len(vals) > 1:
        mean = observed
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        sigma = math.sqrt(var)

    required = required_keep_speedup(
        incumbent_mean_speedup,
        sigma=sigma,
        min_margin=min_margin,
        min_speedup=min_speedup,
        n_samples=len(vals),
    )
    keep = observed >= required
    return KeepEval(
        keep=keep,
        required_speedup=required,
        observed_speedup=observed,
        incumbent_speedup=incumbent_mean_speedup,
        sigma=sigma,
        reason="keep_speedup_ok" if keep else "speedup_below_noise_margin",
        snr=snr,
    )


def speedup_from_latencies(baseline_us: float, candidate_us: float) -> float:
    """Convert latencies to pristine-relative speedup (higher is better)."""
    if candidate_us <= 0 or baseline_us <= 0:
        return 0.0
    return float(baseline_us) / float(candidate_us)
