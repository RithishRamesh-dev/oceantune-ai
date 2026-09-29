"""Tests for kernel harness YAML cases and ShadowHookRegistry."""

from __future__ import annotations

from pathlib import Path

from core.kernel_harness import load_harness_cases
from core.kernel_integration import (
    ShadowHookRegistry,
    materialize_shadow_package,
)


def test_load_default_stage3_cases():
    cases = load_harness_cases()
    ids = {c.id for c in cases}
    assert "attention_prefill_gqa" in ids
    assert "attention_decode_gqa" in ids
    assert len(cases) >= 3


def test_filter_by_phase_decode():
    cases = load_harness_cases(case_ids=None, phase="decode")
    # stage3_default includes decode cases; when also filtering phase,
    # load_harness_cases applies stage3_default first then phase filter
    assert all(c.phase in ("decode", "both") for c in cases)


def test_filter_by_tags_moe():
    # Load all cases then tag-filter — bypass stage3_default by passing all ids
    all_cases = load_harness_cases(
        case_ids=[
            "attention_prefill_gqa",
            "moe_dispatch_proxy",
            "gemm_ffn_wide",
        ]
    )
    moe = [c for c in all_cases if "moe" in c.tags]
    assert len(moe) == 1
    assert moe[0].op_type == "moe"


def test_shadow_hook_registry_by_fusion():
    reg = ShadowHookRegistry()
    h = reg.by_fusion_pattern("residual_add_rmsnorm")
    assert h is not None
    assert h.env_flag == "OCEANTUNE_FUSED_RESIDUAL"
    assert "layernorm" in h.import_targets[0]


def test_shadow_hook_resolve_op_attention():
    reg = ShadowHookRegistry()
    h = reg.resolve_for_plan(op_type="attention")
    assert h is not None
    assert h.hook_id == "attention_custom"


def test_materialize_writes_hook_json(tmp_path, monkeypatch):
    import core.kernel_integration as ki

    monkeypatch.setattr(ki, "_SHADOW_ROOT", tmp_path)
    kernel = tmp_path / "k.py"
    kernel.write_text(
        "import torch\n@triton.jit\ndef f():\n    pass\n",
        encoding="utf-8",
    )
    spec = materialize_shadow_package(
        kernel_path=str(kernel),
        op_type="rmsnorm",
        session_id="abc123sessionid",
        env_flag="OCEANTUNE_FUSED_RESIDUAL",
        fusion_pattern_id="residual_add_rmsnorm",
    )
    dest = spec.host_shadow_dir("abc123sessionid")
    hook_json = dest / "oceantune_shadow" / "hook.json"
    assert hook_json.is_file()
    text = hook_json.read_text(encoding="utf-8")
    assert "OCEANTUNE_FUSED_RESIDUAL" in text
    assert "layernorm" in text
    site = dest / "sitecustomize.py"
    assert "documented_targets" in site.read_text(encoding="utf-8")
