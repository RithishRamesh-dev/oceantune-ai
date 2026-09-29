"""Tests for experience constraints distillation."""

from core.experience_constraints import (
    ExperienceConstraint,
    constraints_from_error,
    constraints_from_failed_configs,
    constraints_from_recipe_pitfalls,
    render_constraints_block,
)


def test_oom_signature_produces_high_severity():
    cs = constraints_from_error("CUDA out of memory on device 0")
    assert cs
    assert any("gpu_memory_utilization" in c.rule for c in cs)
    assert any(c.severity == "high" for c in cs)


def test_mla_block_size_rule():
    cs = constraints_from_error("ValueError: MLA requires block_size=1")
    assert any("block_size=1" in c.rule for c in cs)


def test_failed_config_util_floor():
    failed = [
        {
            "flags": {"gpu_memory_utilization": 0.95},
            "error": "HIP out of memory",
        }
    ]
    cs = constraints_from_failed_configs(failed)
    assert any("0.95" in c.rule for c in cs)


def test_recipe_pitfalls():
    class Pit:
        description = "FLASHINFER failed to load"
        severity = "high"

    cs = constraints_from_recipe_pitfalls([Pit()])
    assert cs
    assert any("FLASHINFER" in c.rule.upper() or "flashinfer" in c.rule.lower() for c in cs)


def test_render_constraints_block_empty():
    assert render_constraints_block([]) == ""


def test_render_constraints_block_dedupes():
    cs = [
        ExperienceConstraint(rule="same", source="a", severity="medium"),
        ExperienceConstraint(rule="same", source="b", severity="high"),
    ]
    block = render_constraints_block(cs)
    assert block.count("same") == 1
    assert "[high]" in block
