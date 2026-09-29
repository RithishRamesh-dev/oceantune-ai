"""Tests for KernelIntegrationLayer (shadow path)."""

import asyncio
from pathlib import Path

from core.kernel_integration import (
    e2e_rebench_placeholder,
    plan_integration,
    validate_kernel_artifact,
)


def test_validate_missing_file():
    blockers = validate_kernel_artifact("/tmp/does_not_exist_oceantune_kernel.py")
    assert any("missing_file" in b for b in blockers)


def test_plan_blocked_skeleton(tmp_path: Path):
    p = tmp_path / "k.py"
    p.write_text("# TODO: Implement kernel\n", encoding="utf-8")
    plan = plan_integration(op_type="attention", kernel_path=str(p), micro_speedup_pct=5.0)
    assert plan.status == "blocked"
    assert any("skeleton" in b for b in plan.blockers)


def test_plan_e2e_pending_valid(tmp_path: Path):
    p = tmp_path / "k.py"
    p.write_text(
        "import triton\n@triton.jit\ndef foo():\n    pass\n",
        encoding="utf-8",
    )
    plan = plan_integration(op_type="rmsnorm", kernel_path=str(p), micro_speedup_pct=5.0)
    assert plan.status == "e2e_pending"
    assert plan.e2e_verified is False

    plan2 = asyncio.run(e2e_rebench_placeholder(plan=plan, session_id="s1shadow"))
    assert plan2.e2e_verified is False
    assert plan2.shadow_env.get("OCEANTUNE_SHADOW_KERNEL") == "1"
    assert any("awaiting_docker_e2e_rebench" in b for b in plan2.blockers)


def test_plan_low_speedup_blocked(tmp_path: Path):
    p = tmp_path / "k.py"
    p.write_text("import torch\nx = torch.randn(1)\n", encoding="utf-8")
    plan = plan_integration(op_type="gemm", kernel_path=str(p), micro_speedup_pct=0.1)
    assert plan.status == "blocked"
