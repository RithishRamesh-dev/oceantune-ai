"""Tests for macro-cycle Stage2↔3 reloop decisions."""

from core.macro_cycle import (
    MacroCycleState,
    advance_macro_cycle,
    decaying_min_gain_pct,
    should_reloop_stage2_3,
)


def test_decaying_threshold():
    assert decaying_min_gain_pct(0) == 1.0
    assert abs(decaying_min_gain_pct(1) - 0.55) < 1e-9


def test_reloop_on_gain():
    state = MacroCycleState(macro_cycle=0, gain_at_cycle_start=1.0)
    d = should_reloop_stage2_3(
        state=state,
        current_fitness=1.05,  # +5% > 1% threshold
        max_macro_cycles=2,
    )
    assert d.reloop is True
    assert d.next_cycle == 1


def test_block_max_cycles():
    state = MacroCycleState(macro_cycle=1, gain_at_cycle_start=1.0)
    d = should_reloop_stage2_3(
        state=state,
        current_fitness=1.10,
        max_macro_cycles=2,
    )
    assert d.reloop is False
    assert d.reason == "max_macro_cycles"


def test_block_no_gain():
    state = MacroCycleState(macro_cycle=0, gain_at_cycle_start=1.0)
    d = should_reloop_stage2_3(
        state=state,
        current_fitness=1.001,  # below 1% threshold
        max_macro_cycles=3,
    )
    assert d.reloop is False
    assert "below_threshold" in d.reason or "converged" in d.reason


def test_advance_updates_state():
    state = MacroCycleState(macro_cycle=0, gain_at_cycle_start=1.0)
    d = should_reloop_stage2_3(
        state=state, current_fitness=1.05, max_macro_cycles=3,
    )
    advance_macro_cycle(state, decision=d, current_fitness=1.05)
    assert state.macro_cycle == 1
    assert state.gain_at_cycle_start == 1.05
    assert len(state.history) == 1
