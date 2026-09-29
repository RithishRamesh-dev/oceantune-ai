"""Tests for MeasurementGate."""

from __future__ import annotations

from core.measurement_gate import MeasurementGate


def test_keep_on_improvement():
    g = MeasurementGate()
    d = g.decide(incumbent_fitness=0.5, candidate_fitness=0.55)
    assert d.keep is True
    assert d.reason == "measured_improvement"


def test_revert_on_regression():
    g = MeasurementGate()
    d = g.decide(incumbent_fitness=0.5, candidate_fitness=0.49)
    assert d.keep is False
    assert d.reason == "no_improvement"


def test_relative_threshold():
    g = MeasurementGate(min_relative_improvement=0.01)  # 1%
    d = g.decide(incumbent_fitness=1.0, candidate_fitness=1.005)
    assert d.keep is False
    d2 = g.decide(incumbent_fitness=1.0, candidate_fitness=1.02)
    assert d2.keep is True


def test_zero_candidate_rejected():
    g = MeasurementGate()
    d = g.decide(incumbent_fitness=0.5, candidate_fitness=0.0)
    assert d.keep is False
    assert d.reason == "candidate_fitness_zero_or_failed"


def test_summary_counts():
    g = MeasurementGate()
    g.decide(incumbent_fitness=0.5, candidate_fitness=0.6)
    g.decide(incumbent_fitness=0.6, candidate_fitness=0.55)
    s = g.summary()
    assert s["kept"] == 1
    assert s["reverted"] == 1
