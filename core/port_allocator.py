"""
core/port_allocator.py
----------------------
Port pool manager for parallel vLLM instances.

Maintains a pool of TCP ports (default 8000–8099) and hands them out
exclusively to vLLM instances.  Ports are reclaimed when an instance stops.

Usage
-----
    pool = PortAllocator(start=8000, end=8099)
    port = await pool.acquire()   # e.g. 8000
    # ... launch vLLM on port ...
    await pool.release(port)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Set

log = logging.getLogger("core.port_allocator")


class PortAllocator:
    """
    Asyncio-safe port pool.

    Parameters
    ----------
    start : int
        First port in the pool (inclusive).
    end : int
        Last port in the pool (inclusive).
    """

    def __init__(self, start: int = 8000, end: int = 8099) -> None:
        if start > end:
            raise ValueError(f"start ({start}) must be <= end ({end})")
        self._pool: list[int] = list(range(start, end + 1))
        self._in_use: Set[int] = set()
        self._lock = asyncio.Lock()
        log.info(
            "PortAllocator ready: %d ports (%d–%d)",
            len(self._pool), start, end,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available(self) -> int:
        return len(self._pool)

    @property
    def in_use(self) -> Set[int]:
        return set(self._in_use)

    # ------------------------------------------------------------------
    # Acquire / release
    # ------------------------------------------------------------------

    async def acquire(self) -> Optional[int]:
        """
        Acquire the next available port.

        Returns the port number, or ``None`` if the pool is exhausted.
        """
        async with self._lock:
            if not self._pool:
                log.warning("Port pool exhausted — no free ports")
                return None
            port = self._pool.pop(0)
            self._in_use.add(port)
            log.debug("Port acquired: %d (remaining: %d)", port, len(self._pool))
            return port

    async def release(self, port: int) -> None:
        """Return a port to the pool."""
        async with self._lock:
            if port not in self._in_use:
                log.warning("Attempted to release port %d which is not in use", port)
                return
            self._in_use.discard(port)
            # Insert in sorted order
            self._pool = sorted(self._pool + [port])
            log.debug("Port released: %d (available: %d)", port, len(self._pool))
