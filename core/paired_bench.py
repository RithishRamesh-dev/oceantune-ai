"""
core/paired_bench.py
--------------------
Paired A/B keep helper for short interleaved measurements.

Agents propose; MeasurementGate decides. This module adds a fidelity check:
if two short probes disagree on the sign of improvement (or sit inside noise),
refuse KEEP even when a single full-ramp looks better.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from core.measurement_gate import GateDecision, MeasurementGate


@dataclass
class PairedProbe:
    """One short measurement of incumbent vs candidate."""

    incumbent: float
    candidate: float
    label: str = ""

    @property
    def delta(self) -> float:
        return float(self.candidate) - float(self.incumbent)


@dataclass
class PairedBenchResult:
    keep: bool
    reason: str
    probes: List[Dict[str, Any]] = field(default_factory=list)
    mean_delta: float = 0.0
    sign_agreement: bool = False
    gate: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_paired_probes(
    probes: Sequence[PairedProbe],
    *,
    noise_band: float = 0.005,
    min_agreeing_pairs: int = 2,
    gate: Optional[MeasurementGate] = None,
) -> PairedBenchResult:
    """
    Require consistent positive delta across short pairs before KEEP.

    ``noise_band`` is relative to incumbent (default 0.5%).
    """
    cleaned = [p for p in probes if p.incumbent > 0 or p.candidate > 0]
    if len(cleaned) < min_agreeing_pairs:
        return PairedBenchResult(
            keep=False,
            reason="insufficient_pairs",
            probes=[{"incumbent": p.incumbent, "candidate": p.candidate, "label": p.label} for p in cleaned],
        )

    signs: List[int] = []
    deltas: List[float] = []
    for p in cleaned:
        inc = max(float(p.incumbent), 1e-9)
        d = float(p.candidate) - float(p.incumbent)
        rel = d / inc
        deltas.append(d)
        if abs(rel) <= noise_band:
            signs.append(0)
        elif d > 0:
            signs.append(1)
        else:
            signs.append(-1)

    mean_delta = sum(deltas) / len(deltas)
    positive = sum(1 for s in signs if s > 0)
    negative = sum(1 for s in signs if s < 0)
    agree = positive >= min_agreeing_pairs and negative == 0

    probe_dicts = [
        {
            "incumbent": p.incumbent,
            "candidate": p.candidate,
            "label": p.label,
            "delta": p.delta,
        }
        for p in cleaned
    ]

    if not agree:
        return PairedBenchResult(
            keep=False,
            reason="sign_disagreement_or_noise",
            probes=probe_dicts,
            mean_delta=mean_delta,
            sign_agreement=False,
        )

    # Final gate on mean fitness of last probe (or mean candidate vs mean incumbent)
    mean_inc = sum(p.incumbent for p in cleaned) / len(cleaned)
    mean_cand = sum(p.candidate for p in cleaned) / len(cleaned)
    g = gate or MeasurementGate(min_relative_improvement=noise_band)
    decision: GateDecision = g.decide(
        incumbent_fitness=mean_inc,
        candidate_fitness=mean_cand,
        label="paired_bench",
    )
    return PairedBenchResult(
        keep=decision.keep,
        reason=decision.reason if decision.keep else f"gate_{decision.reason}",
        probes=probe_dicts,
        mean_delta=mean_delta,
        sign_agreement=True,
        gate=decision.to_dict(),
    )
