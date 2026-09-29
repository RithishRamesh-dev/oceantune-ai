"""
core/measurement_convergence.py
-------------------------------
Reject noisy / non-stationary benchmark ramps before KEEP.

Hyperloom discards warmup, checks spread, and rejects monotonic climb that
looks like cache warm-up rather than a true steady state. OceanTune applies
the same ideas to concurrency-ramp samples without vendoring Hyperloom.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class ConvergenceAssessment:
    converged: bool
    reason: str
    n_samples: int = 0
    mean: float = 0.0
    stdev: float = 0.0
    spread_pct: float = 0.0
    discarded_warmup: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def assess_convergence(
    values: Sequence[float],
    *,
    warmup_discard: int = 1,
    max_spread_pct: float = 15.0,
    min_samples: int = 2,
    reject_strict_monotonic_climb: bool = True,
) -> ConvergenceAssessment:
    """
    Assess whether a series of fitness / throughput samples has settled.

    Parameters
    ----------
    values :
        Ordered measurements (e.g. per concurrency level or repeated trials).
    warmup_discard :
        Drop this many leading samples (cold start / graph capture).
    max_spread_pct :
        (max-min)/mean * 100 on the retained window; above → not converged.
    min_samples :
        Require at least this many post-warmup samples.
    reject_strict_monotonic_climb :
        If every retained sample is strictly greater than the previous,
        treat as non-stationary warm-up artifact.
    """
    raw = [float(v) for v in values if v is not None and float(v) > 0]
    discarded = min(max(0, int(warmup_discard)), max(0, len(raw) - 1))
    window = raw[discarded:]

    if len(window) < min_samples:
        return ConvergenceAssessment(
            converged=False,
            reason="insufficient_samples",
            n_samples=len(window),
            discarded_warmup=discarded,
            metadata={"raw_n": len(raw)},
        )

    mean = statistics.fmean(window)
    stdev = statistics.pstdev(window) if len(window) > 1 else 0.0
    spread_pct = ((max(window) - min(window)) / mean * 100.0) if mean > 0 else 0.0

    if reject_strict_monotonic_climb and len(window) >= 3:
        if all(window[i] < window[i + 1] for i in range(len(window) - 1)):
            return ConvergenceAssessment(
                converged=False,
                reason="monotonic_climb",
                n_samples=len(window),
                mean=mean,
                stdev=stdev,
                spread_pct=spread_pct,
                discarded_warmup=discarded,
            )

    if spread_pct > max_spread_pct:
        return ConvergenceAssessment(
            converged=False,
            reason="spread_too_high",
            n_samples=len(window),
            mean=mean,
            stdev=stdev,
            spread_pct=spread_pct,
            discarded_warmup=discarded,
        )

    return ConvergenceAssessment(
        converged=True,
        reason="ok",
        n_samples=len(window),
        mean=mean,
        stdev=stdev,
        spread_pct=spread_pct,
        discarded_warmup=discarded,
    )


def stable_fitness(
    values: Sequence[float],
    *,
    warmup_discard: int = 1,
) -> Optional[float]:
    """Return mean of post-warmup positive samples, or None if empty."""
    raw = [float(v) for v in values if v is not None and float(v) > 0]
    discarded = min(max(0, int(warmup_discard)), max(0, len(raw) - 1))
    window = raw[discarded:]
    if not window:
        return None
    return float(statistics.fmean(window))
