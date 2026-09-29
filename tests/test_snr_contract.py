"""Tests for SNR + statistical KEEP contracts."""

from core.snr_contract import (
    evaluate_keep,
    snr_db_from_tensors,
    speedup_from_latencies,
    validate_snr,
)


def test_snr_exact_match():
    ref = [1.0, 2.0, 3.0]
    r = validate_snr(ref, ref, threshold_db=30.0)
    assert r.passed
    assert r.snr_db >= 30.0


def test_snr_fails_noisy():
    ref = [1.0, 1.0, 1.0]
    cand = [0.0, 0.0, 0.0]
    r = validate_snr(ref, cand, threshold_db=30.0)
    assert r.passed is False


def test_evaluate_keep_requires_snr():
    from core.snr_contract import SNRResult

    snr = SNRResult(passed=False, snr_db=10.0, threshold_db=30.0, reason="low")
    k = evaluate_keep([1.2], snr=snr)
    assert k.keep is False
    assert "snr_prefilter" in k.reason


def test_evaluate_keep_speedup():
    k = evaluate_keep([1.05, 1.06, 1.04], incumbent_mean_speedup=1.0, min_speedup=1.01)
    assert k.keep is True
    assert k.observed_speedup > 1.04


def test_speedup_from_latencies():
    assert abs(speedup_from_latencies(100.0, 80.0) - 1.25) < 1e-9


def test_snr_db_helper():
    assert snr_db_from_tensors([1.0], [1.0]) >= 100.0
