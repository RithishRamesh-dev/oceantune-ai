"""Tests for knowledge packs + shadow kernel integration."""

import asyncio
from pathlib import Path

from core.knowledge_pack import load_knowledge_block, vendor_for_gpu
from core.kernel_integration import (
    materialize_shadow_package,
    plan_integration,
    run_e2e_rebench,
)


def test_vendor_for_gpu():
    assert vendor_for_gpu("H200") == "nvidia"
    assert vendor_for_gpu("MI300X") == "amd"


def test_load_nvidia_pack():
    block = load_knowledge_block(gpu_type="H200", max_chars=2000)
    assert "Vendor knowledge pack" in block
    assert "nvidia" in block.lower()


def test_load_amd_pack():
    block = load_knowledge_block(gpu_type="MI300X", max_chars=2000)
    assert "amd" in block.lower()


def test_shadow_materialize_and_dry_run(tmp_path: Path):
    k = tmp_path / "kernel.py"
    k.write_text("import triton\n@triton.jit\ndef f():\n    pass\n", encoding="utf-8")
    plan = plan_integration(op_type="rmsnorm", kernel_path=str(k), micro_speedup_pct=5.0)
    assert plan.status == "e2e_pending"

    spec = materialize_shadow_package(
        kernel_path=str(k), op_type="rmsnorm", session_id="abcd1234sess",
    )
    assert (spec.host_shadow_dir("abcd1234sess") / "sitecustomize.py").is_file()
    assert (spec.host_shadow_dir("abcd1234sess") / "oceantune_shadow" / "kernel.py").is_file()

    plan2 = asyncio.run(
        run_e2e_rebench(
            plan=plan,
            session_id="abcd1234sess",
            model_id="m",
            gpu_type="H200",
            flags={},
            incumbent_fitness=1.0,
            gpu_alloc=None,
            port_alloc=None,
            dry_run=True,
        )
    )
    assert plan2.status == "e2e_pending"
    assert "dry_run_no_docker" in plan2.blockers
    assert plan2.shadow_env.get("OCEANTUNE_SHADOW_KERNEL") == "1"
