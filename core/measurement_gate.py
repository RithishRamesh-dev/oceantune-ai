"""
core/measurement_gate.py
------------------------
Coordinator-owned keep/revert decisions.

Inspired by Hyperloom's invariant: agents propose; measurements decide.
OceanTune fitness scores already drive Stage 1–3; this module makes the
rule explicit and reusable so Stage 2/3/4 share one decision API.

No Hyperloom source is vendored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger("core.measurement_gate")


@dataclass
class GateDecision:
    """Outcome of comparing a candidate measurement to the incumbent."""

    keep: bool
    reason: str
    incumbent_fitness: float
    candidate_fitness: float
    delta: float = 0.0
    delta_pct: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "keep": self.keep,
            "reason": self.reason,
            "incumbent_fitness": self.incumbent_fitness,
            "candidate_fitness": self.candidate_fitness,
            "delta": self.delta,
            "delta_pct": self.delta_pct,
            "metadata": self.metadata,
        }


class MeasurementGate:
    """
    Decide KEEP vs REVERT from measured fitness only.

    Parameters
    ----------
    min_relative_improvement : float
        Require candidate > incumbent * (1 + this). Default 0.0 = any
        strict improvement. Use 0.01 for 1% gates (e.g. Stage 4 kernels).
    min_absolute_improvement : float
        Also require absolute fitness delta ≥ this (default 0).
    allow_equal : bool
        If True, fitness == incumbent is KEEP (rare; default False).
    """

    def __init__(
        self,
        *,
        min_relative_improvement: float = 0.0,
        min_absolute_improvement: float = 0.0,
        allow_equal: bool = False,
    ) -> None:
        self._min_rel = min_relative_improvement
        self._min_abs = min_absolute_improvement
        self._allow_equal = allow_equal
        self._history: List[GateDecision] = []

    @property
    def history(self) -> List[GateDecision]:
        return list(self._history)

    def decide(
        self,
        *,
        incumbent_fitness: float,
        candidate_fitness: float,
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GateDecision:
        """
        Compare candidate to incumbent.

        Predicted / LLM-claimed gains MUST NOT be passed here — only
        BenchmarkEngine / MetricsCollector fitness.
        """
        inc = float(incumbent_fitness or 0.0)
        cand = float(candidate_fitness or 0.0)
        delta = cand - inc
        delta_pct = (delta / inc * 100.0) if inc > 0 else (100.0 if cand > 0 else 0.0)

        if cand <= 0:
            decision = GateDecision(
                keep=False,
                reason="candidate_fitness_zero_or_failed",
                incumbent_fitness=inc,
                candidate_fitness=cand,
                delta=delta,
                delta_pct=delta_pct,
                metadata=dict(metadata or {}),
            )
        elif self._allow_equal and abs(delta) < 1e-12:
            decision = GateDecision(
                keep=True,
                reason="equal_fitness_allowed",
                incumbent_fitness=inc,
                candidate_fitness=cand,
                delta=delta,
                delta_pct=delta_pct,
                metadata=dict(metadata or {}),
            )
        else:
            need_abs = delta >= self._min_abs
            if inc <= 0:
                need_rel = cand > 0
            else:
                need_rel = cand > inc * (1.0 + self._min_rel)
            keep = bool(need_abs and need_rel and delta > 0)
            if keep:
                reason = "measured_improvement"
            elif delta <= 0:
                reason = "no_improvement"
            elif not need_rel:
                reason = f"below_relative_threshold_{self._min_rel:.4f}"
            else:
                reason = f"below_absolute_threshold_{self._min_abs:.6f}"
            decision = GateDecision(
                keep=keep,
                reason=reason,
                incumbent_fitness=inc,
                candidate_fitness=cand,
                delta=delta,
                delta_pct=delta_pct,
                metadata=dict(metadata or {}),
            )

        self._history.append(decision)
        log.info(
            "MeasurementGate%s: keep=%s reason=%s fitness %.4f → %.4f (Δ=%+.4f / %+.1f%%)",
            f"[{label}]" if label else "",
            decision.keep,
            decision.reason,
            inc,
            cand,
            delta,
            delta_pct,
        )
        return decision

    def summary(self) -> Dict[str, Any]:
        kept = sum(1 for d in self._history if d.keep)
        return {
            "decisions": len(self._history),
            "kept": kept,
            "reverted": len(self._history) - kept,
            "history": [d.to_dict() for d in self._history],
        }
