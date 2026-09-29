"""Tests for fusion pattern matching + diagnose."""

from core.fusion import (
    category_shares_from_trace,
    diagnose_fusion,
    match_patterns,
)


def test_match_residual_pattern():
    shares = {
        "add": 0.08,
        "rmsnorm": 0.08,
        "elementwise": 0.05,
    }
    hits = match_patterns(shares, framework="vllm", vendor="nvidia")
    ids = [p.id for p, _ in hits]
    assert "residual_add_rmsnorm" in ids


def test_diagnose_not_candidate_when_gemm_bound():
    shares = {"gemm": 0.7, "attention": 0.2}
    d = diagnose_fusion(shares)
    assert d.is_candidate is False


def test_diagnose_from_trace_norm_kernel():
    shares = category_shares_from_trace(
        attention_pct=10,
        gemm_pct=20,
        other_pct=40,
        bottleneck_kernel="fused_rms_norm_kernel",
        bottleneck_type="memory_bandwidth",
    )
    d = diagnose_fusion(shares)
    assert d.launch_bound_share >= 0.10
    # May or may not match depending on shares; ensure no crash and prompt works
    _ = d.prompt_block()
