"""Tests for FrameworkBackend, PolicyGate, enablement, draft registry, campaign, ledger."""

import asyncio
from pathlib import Path

from core.framework_backend import (
    SGLangBackend,
    VLLMBackend,
    get_framework_backend,
    list_frameworks,
)
from core.policy_gate import PolicyGate
from core.enablement import classify_boot_failure, repair_ladder, apply_repair
from core.draft_registry import (
    find_draft_for_target,
    speculative_strategy_allowed,
    planner_speculative_hint,
)
from core.kernel_ledger import KernelLedger, kernel_identity
from core.workspace_policy import open_workspace, assert_writable, WorkspacePolicyError
from core.numerical_pipeline import run_numerical_pipeline
from core.kernel_campaign import run_fusion_campaign


def test_frameworks_registered():
    assert "vllm" in list_frameworks()
    assert "sglang" in list_frameworks()
    assert get_framework_backend("vllm").name == "vllm"
    assert get_framework_backend("sglang").name == "sglang"


def test_sglang_drops_unknown_flags():
    b = SGLangBackend()
    norm = b.normalize_flags({
        "tensor_parallel_size": 2,
        "totally_fake_flag": 1,
    })
    assert "tensor_parallel_size" in norm
    assert "totally_fake_flag" not in norm


def test_vllm_launch_spec():
    spec = VLLMBackend().build_launch_spec(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        flags={"tensor_parallel_size": 1, "gpu_memory_utilization": 0.9},
        port=8000,
        gpu_type="H200",
        docker_image="vllm/vllm-openai:test",
    )
    assert spec.framework == "vllm"
    assert spec.port == 8000


def test_policy_gate_stage1_blocks_custom_kernel():
    g = PolicyGate()
    d = g.filter_flags("stage1", {"gpu_memory_utilization": 0.9}, require_custom_kernel=True)
    assert d.allowed is False


def test_policy_gate_stage4_freezes_flags():
    g = PolicyGate()
    d = g.filter_flags("stage4", {"gpu_memory_utilization": 0.95})
    assert d.allowed is False
    assert "frozen" in d.reason


def test_policy_sanitize_stage2():
    g = PolicyGate()
    out = g.sanitize_flags("stage2", {
        "attention_backend": "FLASH_ATTN",
        "tensor_parallel_size": 8,  # not a stage2 strategy key
    })
    assert "attention_backend" in out
    assert "tensor_parallel_size" not in out


def test_enablement_oom_ladder():
    steps = repair_ladder(
        base_flags={"gpu_memory_utilization": 0.95},
        error="CUDA out of memory",
        moe=True,
    )
    assert steps
    assert any("util" in s.name or "fp8" in s.name for s in steps)
    merged = apply_repair({"gpu_memory_utilization": 0.95}, steps[0])
    assert merged["gpu_memory_utilization"] <= 0.95


def test_classify_boot_failure():
    assert classify_boot_failure("HIP out of memory") == "oom"
    assert classify_boot_failure("startup timeout") == "startup_timeout"


def test_draft_registry_qwen():
    pair = find_draft_for_target("Qwen/Qwen2.5-7B-Instruct")
    assert pair is not None
    ok, reason = speculative_strategy_allowed(
        "Qwen/Qwen2.5-7B-Instruct",
        {"speculative_model": pair.draft_model},
    )
    assert ok
    bad, _ = speculative_strategy_allowed(
        "Qwen/Qwen2.5-7B-Instruct",
        {"speculative_model": "totally/wrong-draft"},
    )
    assert bad is False
    hint = planner_speculative_hint("deepseek-ai/DeepSeek-V3.2")
    assert "draft" in hint.lower() and "DeepSeek" in hint
    hint_unknown = planner_speculative_hint("totally-unknown/Model-XYZ")
    assert "NOT" in hint_unknown or "Do NOT" in hint_unknown or "no draft" in hint_unknown.lower()


def test_kernel_ledger_roundtrip(tmp_path: Path):
    led = KernelLedger(root=tmp_path)
    e = led.record(
        session_id="sess1234567890",
        op_type="rmsnorm",
        model_id="m",
        gpu_type="H200",
        decision="kept",
        micro_speedup_pct=5.0,
    )
    assert e.identity
    listed = led.list_session("sess1234567890")
    assert len(listed) == 1
    lessons = led.lessons_for_recipe("sess1234567890")
    assert lessons


def test_workspace_policy(tmp_path: Path):
    ws = open_workspace("abc123sessionid", root=tmp_path)
    p = assert_writable(ws, "kernels/k.py")
    p.write_text("x=1", encoding="utf-8")
    try:
        assert_writable(ws, "agents/hack.py")
        raise AssertionError("should deny")
    except WorkspacePolicyError:
        pass


def test_numerical_pipeline_snr_fail():
    r = run_numerical_pipeline([1.0, 1.0], [0.0, 0.0])
    assert r.passed is False
    assert r.stages[0].stage == "snr"


def test_numerical_pipeline_keep():
    r = run_numerical_pipeline(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        speedups=[1.05, 1.06, 1.04],
    )
    assert r.passed is True
    assert r.keep is not None and r.keep.keep


def test_fusion_campaign_research_only():
    result = asyncio.run(run_fusion_campaign(
        session_id="camp1234567890abcd",
        model_id="m",
        gpu_type="H200",
        matched_patterns=[{"id": "residual_add_rmsnorm", "env_flag": "OCEANTUNE_FUSED_RESIDUAL"}],
        evolution_result=None,
    ))
    assert result.trials
    assert result.trials[0].status == "skipped"
