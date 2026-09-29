"""Tests for core.flag_merge."""

from __future__ import annotations

from core.flag_merge import (
    infer_microbench_op,
    merge_flags,
    merged_fingerprint,
    to_vllm_flags_dict,
)
from core.search_space import VLLMFlags


def test_merge_flags_stage2_delta_overrides_stage1():
    base = {"gpu_memory_utilization": 0.9, "attention_backend": "FLASH_ATTN"}
    delta = {"attention_backend": "FLASHINFER", "kv_cache_dtype": "fp8"}
    merged = merge_flags(base, delta)
    assert merged["gpu_memory_utilization"] == 0.9
    assert merged["attention_backend"] == "FLASHINFER"
    assert merged["kv_cache_dtype"] == "fp8"


def test_merge_flags_vllm_fields_only_strips_env_keys():
    base = {"tensor_parallel_size": 1}
    delta = {"nccl_min_nchannels": 112, "attention_backend": "FLASHINFER"}
    merged = merge_flags(base, delta, vllm_fields_only=True)
    assert "nccl_min_nchannels" not in merged
    assert merged["attention_backend"] == "FLASHINFER"


def test_merged_fingerprint_stable():
    base = {"tensor_parallel_size": 1, "gpu_memory_utilization": 0.9}
    delta = {"max_num_batched_tokens": 8192}
    fp1 = merged_fingerprint(base, delta)
    fp2 = merged_fingerprint(base, delta)
    assert fp1 == fp2
    assert len(fp1) == 12


def test_to_vllm_flags_dict():
    merged = merge_flags(
        {"tensor_parallel_size": 1, "_category": "kernel"},
        {"attention_backend": "FLASHINFER"},
    )
    vf_dict = to_vllm_flags_dict(merged)
    assert "_category" not in vf_dict
    vf = VLLMFlags(**vf_dict)
    assert vf.attention_backend == "FLASHINFER"


def test_merge_matches_report_generator_pattern():
    """Stage 1 + Stage 2 merge should match report_generator shell script logic."""
    stage1 = {
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "attention_backend": "FLASH_ATTN",
    }
    stage2 = {"attention_backend": "FLASHINFER", "kv_cache_dtype": "fp8"}

    from core.flag_merge import merge_flags as mf
    merged_util = mf(stage1, stage2, vllm_fields_only=True)

    known = set(VLLMFlags.__dataclass_fields__)
    merged_report = {
        **stage1,
        **{k: v for k, v in stage2.items() if k in known},
    }
    assert merged_util == merged_report


def test_infer_microbench_op_from_kernel_name():
    assert infer_microbench_op(bottleneck_kernel="flash_attn_fwd") == "attention"
    assert infer_microbench_op(bottleneck_kernel="cutlass_gemm") == "gemm"
    assert infer_microbench_op(bottleneck_kernel="moe_dispatch") == "moe"


def test_infer_microbench_op_from_trace_pcts():
    assert infer_microbench_op(
        bottleneck_kernel="unknown_kernel",
        gemm_pct=50.0,
        attention_pct=30.0,
    ) == "gemm"
