"""Tests for kernel registry and capability detection."""

from __future__ import annotations

from core.kernel_registry import CapabilityDetector, KernelRegistry, model_has_gqa


def test_registry_lists_flash_attn():
    reg = KernelRegistry()
    assert "FLASH_ATTN" in reg.list_attention_backends()


def test_nvidia_filter_excludes_rocm_flash():
    reg = KernelRegistry()
    backends = reg.filter_attention_backends(gpu_type="H200", model_id="Qwen/Qwen2.5-7B-Instruct")
    assert "FLASH_ATTN" in backends
    assert "ROCM_FLASH" not in backends


def test_amd_filter_includes_rocm_flash():
    reg = KernelRegistry()
    backends = reg.filter_attention_backends(gpu_type="MI300X", model_id="Qwen/Qwen2.5-7B-Instruct")
    assert "ROCM_FLASH" in backends


def test_capability_detector_filters_flashinfer_on_amd():
    det = CapabilityDetector()
    filtered = det.filter_strategy_config(
        {"attention_backend": "FLASHINFER"},
        gpu_type="MI300X",
        model_id="Qwen/Qwen2.5-7B-Instruct",
    )
    assert "attention_backend" not in filtered


def test_model_has_gqa_qwen():
    assert model_has_gqa({}, "Qwen/Qwen2.5-7B-Instruct") is True


def test_flashinfer_recommended_high_vram():
    det = CapabilityDetector()
    assert det.flashinfer_recommended(
        gpu_type="H200",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        gpu_memory_utilization=0.9,
    )
