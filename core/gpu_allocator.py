"""
core/gpu_allocator.py
---------------------
GPU slot allocator for parallel vLLM instances on a single node.

Each vLLM instance gets an exclusive contiguous subset of GPU indices
(a "slot") of size ``tensor_parallel_size``.  The allocator tracks free
slots and sets ``CUDA_VISIBLE_DEVICES`` / ``ROCR_VISIBLE_DEVICES`` so
each instance only sees its own GPUs.

Usage
-----
    alloc = GPUSlotAllocator(gpu_indices=[0,1,2,3,4,5,6,7])
    slot = alloc.acquire(tensor_parallel_size=4)   # e.g. [0,1,2,3]
    # ... launch vLLM with CUDA_VISIBLE_DEVICES="0,1,2,3" ...
    alloc.release(slot)
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional, Set

log = logging.getLogger("core.gpu_allocator")

# GPU vendors whose device-visibility env var is ROCR_VISIBLE_DEVICES
_AMD_GPU_TYPES: Set[str] = {"MI300X", "MI325X", "MI350X"}


class GPUSlotAllocator:
    """
    Thread-safe (asyncio-safe) GPU slot manager.

    Parameters
    ----------
    gpu_indices : list of int
        Physical GPU indices available on this node (e.g. [0,1,2,3,4,5,6,7]).
    gpu_type : str
        GPU SKU name — used to pick the correct env-var name for device
        visibility (CUDA_VISIBLE_DEVICES vs ROCR_VISIBLE_DEVICES).
    """

    def __init__(
        self,
        gpu_indices: Optional[List[int]] = None,
        gpu_type: str = "H100",
    ) -> None:
        self._all_indices: List[int] = list(gpu_indices or [0])
        self._gpu_type = gpu_type
        self._free: List[int] = list(self._all_indices)
        self._lock = asyncio.Lock()
        log.info(
            "GPUSlotAllocator ready: %d GPUs available (%s)",
            len(self._all_indices), self._all_indices,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total(self) -> int:
        return len(self._all_indices)

    @property
    def available(self) -> int:
        return len(self._free)

    @property
    def device_env_var(self) -> str:
        """Return the correct visibility env var for this GPU type."""
        if self._gpu_type in _AMD_GPU_TYPES:
            return "ROCR_VISIBLE_DEVICES"
        return "CUDA_VISIBLE_DEVICES"

    # ------------------------------------------------------------------
    # Acquire / release
    # ------------------------------------------------------------------

    async def acquire(self, tensor_parallel_size: int = 1) -> Optional[List[int]]:
        """
        Acquire a contiguous slot of ``tensor_parallel_size`` GPU indices.

        Returns the list of acquired indices, or ``None`` if there are not
        enough free GPUs.  The caller is responsible for calling
        ``release()`` when the vLLM instance terminates.
        """
        async with self._lock:
            if len(self._free) < tensor_parallel_size:
                log.debug(
                    "GPU slot acquire failed: need %d, have %d free",
                    tensor_parallel_size, len(self._free),
                )
                return None

            # Take the first N free indices (preserve allocation order)
            slot = self._free[:tensor_parallel_size]
            self._free = self._free[tensor_parallel_size:]
            log.info(
                "GPU slot acquired: %s (remaining free: %s)", slot, self._free
            )
            return slot

    async def release(self, slot: List[int]) -> None:
        """Return a previously acquired slot to the free pool."""
        async with self._lock:
            # Re-insert in sorted order to keep allocations deterministic
            self._free = sorted(self._free + slot)
            log.info("GPU slot released: %s (free now: %s)", slot, self._free)

    # ------------------------------------------------------------------
    # Helpers for building vLLM env
    # ------------------------------------------------------------------

    def slot_to_env(self, slot: List[int]) -> str:
        """
        Convert a slot (list of GPU indices) to the comma-separated string
        expected by CUDA_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES.
        """
        return ",".join(str(i) for i in slot)

    def build_device_env(self, slot: List[int]) -> dict:
        """
        Return a dict with the correct visibility env var set for ``slot``.
        Safe to pass directly to subprocess.Popen(env=...).
        """
        return {self.device_env_var: self.slot_to_env(slot)}
