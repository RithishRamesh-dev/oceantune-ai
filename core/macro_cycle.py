"""
core/macro_cycle.py
-------------------
Budgeted Stage 2↔3 reloop (Hyperloom macro-cycle analog).

OceanTune stages are linear by default. When a cycle still produces validated
fitness gain and session budget remains, the controller may re-enter Stage 2
with Stage 3 winner flags + bottleneck hints — without restarting Stage 1.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger("core.macro_cycle")

DEFAULT_MAX_MACRO_CYCLES = 2
DEFAULT_NO_GAIN_STREAK = 2
DEFAULT_MIN_REMAINING_SEC = 1800.0  # 30 min floor for a useful reloop


def decaying_min_gain_pct(macro_cycle: int) -> float:
    """
    Minimum validated gain (%) required to justify another Stage2↔3 cycle.

    Cycle 0: ~1.0%, cycle 1: ~0.55%, asymptote toward 0.1%.
    """
    c = max(0, int(macro_cycle))
    return 0.1 + 0.9 / (c + 1)


@dataclass
class ReloopDecision:
    reloop: bool
    reason: str
    macro_cycle: int
    cycle_gain_pct: float = 0.0
    min_gain_pct: float = 0.0
    no_gain_streak: int = 0
    next_cycle: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MacroCycleState:
    """Mutable bookkeeping for the Stage2↔3 loop."""

    macro_cycle: int = 0
    gain_at_cycle_start: float = 0.0
    no_gain_streak: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "macro_cycle": self.macro_cycle,
            "gain_at_cycle_start": self.gain_at_cycle_start,
            "no_gain_streak": self.no_gain_streak,
            "history": list(self.history),
        }


def should_reloop_stage2_3(
    *,
    state: MacroCycleState,
    current_fitness: float,
    enabled: bool = True,
    max_macro_cycles: int = DEFAULT_MAX_MACRO_CYCLES,
    no_gain_limit: int = DEFAULT_NO_GAIN_STREAK,
    session_remaining_sec: Optional[float] = None,
    min_remaining_sec: float = DEFAULT_MIN_REMAINING_SEC,
    saturated: bool = False,
    target_reached: bool = False,
) -> ReloopDecision:
    """
    Decide whether to start another Stage 2↔3 cycle.

    Reloop requires: enabled, under cycle cap, not saturated/target, enough
    time remaining, cycle gain above decaying threshold, and no_gain streak
    below limit.
    """
    cycle = int(state.macro_cycle)
    min_gain = decaying_min_gain_pct(cycle)
    start = float(state.gain_at_cycle_start or 0.0)
    cur = float(current_fitness or 0.0)
    if start > 0:
        cycle_gain_pct = (cur - start) / start * 100.0
    else:
        cycle_gain_pct = 0.0 if cur <= 0 else 100.0

    cycle_gained = cycle_gain_pct > min_gain
    streak = int(state.no_gain_streak)
    if not cycle_gained:
        streak = streak + 1
    else:
        streak = 0

    def _block(reason: str) -> ReloopDecision:
        return ReloopDecision(
            reloop=False,
            reason=reason,
            macro_cycle=cycle,
            cycle_gain_pct=cycle_gain_pct,
            min_gain_pct=min_gain,
            no_gain_streak=streak,
            next_cycle=None,
        )

    if not enabled:
        return _block("macro_cycle_disabled")
    if target_reached:
        return _block("target_reached")
    if saturated:
        return _block("directions_saturated")
    if cycle + 1 >= max_macro_cycles:
        return _block("max_macro_cycles")
    if session_remaining_sec is not None and session_remaining_sec < min_remaining_sec:
        return _block("insufficient_remaining_time")
    if streak >= no_gain_limit:
        return _block("global_converged_no_gain")
    if not cycle_gained:
        return _block("cycle_gain_below_threshold")

    log.info(
        "Macro-cycle reloop approved: cycle=%d→%d gain=%.2f%% (min=%.2f%%)",
        cycle, cycle + 1, cycle_gain_pct, min_gain,
    )
    return ReloopDecision(
        reloop=True,
        reason="cycle_gain_and_budget_ok",
        macro_cycle=cycle,
        cycle_gain_pct=cycle_gain_pct,
        min_gain_pct=min_gain,
        no_gain_streak=streak,
        next_cycle=cycle + 1,
    )


def advance_macro_cycle(
    state: MacroCycleState,
    *,
    decision: ReloopDecision,
    current_fitness: float,
) -> MacroCycleState:
    """Update state after a cycle (whether or not we reloop)."""
    state.history.append({
        "cycle": state.macro_cycle,
        "fitness": current_fitness,
        "decision": decision.to_dict(),
    })
    state.no_gain_streak = decision.no_gain_streak
    if decision.reloop and decision.next_cycle is not None:
        state.macro_cycle = decision.next_cycle
        state.gain_at_cycle_start = float(current_fitness or 0.0)
    return state
