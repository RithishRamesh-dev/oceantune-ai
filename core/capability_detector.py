"""
core/capability_detector.py
---------------------------
Re-exports CapabilityDetector and helpers from kernel_registry.
"""

from core.kernel_registry import (
    CapabilityDetector,
    KernelRegistry,
    model_has_gqa,
)

__all__ = ["CapabilityDetector", "KernelRegistry", "model_has_gqa"]
