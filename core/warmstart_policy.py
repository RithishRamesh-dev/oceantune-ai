"""
core/warmstart_policy.py
------------------------
Bounds on recipe warm-start / warm-replay probes.

Prevents unbounded boot cost when transferring low-confidence recipes across
GPU or framework boundaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class WarmstartDecision:
    accept: bool
    reason: str
    flags: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    claimed_fitness: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def decide_warmstart(
    *,
    flags: Optional[Dict[str, Any]],
    claimed_fitness: float = 0.0,
    confidence: float = 0.0,
    min_confidence: float = 0.7,
    min_fitness: float = 0.0,
    max_warm_trials: int = 3,
    trials_already: int = 0,
) -> WarmstartDecision:
    """Gate whether a recipe's flags may be used as a warm probe / Stage 1 seed."""
    flags = dict(flags or {})
    if not flags:
        return WarmstartDecision(accept=False, reason="empty_flags", confidence=confidence)
    if trials_already >= max_warm_trials:
        return WarmstartDecision(
            accept=False,
            reason="warm_trial_budget_exhausted",
            confidence=confidence,
            claimed_fitness=claimed_fitness,
        )
    if confidence < min_confidence:
        return WarmstartDecision(
            accept=False,
            reason="confidence_too_low",
            confidence=confidence,
            claimed_fitness=claimed_fitness,
            flags=flags,
        )
    if claimed_fitness < min_fitness:
        return WarmstartDecision(
            accept=False,
            reason="fitness_floor",
            confidence=confidence,
            claimed_fitness=claimed_fitness,
            flags=flags,
        )
    return WarmstartDecision(
        accept=True,
        reason="ok",
        flags=flags,
        confidence=confidence,
        claimed_fitness=claimed_fitness,
    )
