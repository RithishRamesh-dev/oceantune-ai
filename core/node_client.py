"""
core/node_client.py
-------------------
HTTP client for communicating with a remote Node Server (node/node_server.py).

The Coordinator uses one NodeClient per GPU Droplet to:
  - Check the node's health and capacity.
  - Submit benchmark jobs.
  - Poll job status until completion.

Usage
-----
    client = NodeClient(host="10.0.0.2", port=9000)
    capacity = await client.get_capacity()
    job_id = await client.submit_job(payload)
    result = await client.wait_for_job(job_id)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx

log = logging.getLogger("core.node_client")

_DEFAULT_TIMEOUT = 30.0         # seconds for normal requests
_JOB_SUBMIT_TIMEOUT = 10.0     # submit is fire-and-forget on the server side
_POLL_INTERVAL = 5.0            # seconds between job status polls
_MAX_POLL_SECONDS = 7200        # 2 hours max wait per job


class NodeClientError(Exception):
    """Raised when a Node Server request fails."""


class NodeClient:
    """
    Async HTTP client for a single GPU Droplet Node Server.

    Parameters
    ----------
    host : str
        Hostname or IP of the node.
    port : int
        Port the Node Server is listening on.
    timeout : float
        Default request timeout in seconds.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9000,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.host = host
        self.port = port
        self._base_url = f"http://{host}:{port}"
        self._timeout = timeout
        self._http: Optional[httpx.AsyncClient] = None

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._http

    async def close(self) -> None:
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    # ------------------------------------------------------------------
    # Health & capacity
    # ------------------------------------------------------------------

    async def health(self) -> Dict[str, Any]:
        """Return the /health response dict, or raise NodeClientError."""
        http = await self._get_http()
        try:
            r = await http.get("/health")
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            raise NodeClientError(f"Health check failed [{self._base_url}]: {exc}") from exc

    async def is_alive(self) -> bool:
        """Non-throwing health check — returns True if the node is reachable."""
        try:
            await self.health()
            return True
        except NodeClientError:
            return False

    async def get_capacity(self) -> Dict[str, Any]:
        """Return the /capacity response dict."""
        http = await self._get_http()
        try:
            r = await http.get("/capacity")
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            raise NodeClientError(f"Capacity query failed [{self._base_url}]: {exc}") from exc

    # ------------------------------------------------------------------
    # Job submission
    # ------------------------------------------------------------------

    async def submit_job(
        self,
        *,
        session_id: str,
        config_id: str,
        fingerprint: str,
        flags: Dict[str, Any],
        context_configs: List[List[int]],
        model_id: str,
        gpu_type: str,
    ) -> str:
        """
        Submit a benchmark job to the Node Server.

        Returns the job_id string.  The job runs asynchronously on the node;
        use ``wait_for_job()`` to poll until completion.
        """
        payload = {
            "session_id": session_id,
            "config_id": config_id,
            "fingerprint": fingerprint,
            "flags": flags,
            "context_configs": context_configs,
            "model_id": model_id,
            "gpu_type": gpu_type,
        }
        http = await self._get_http()
        try:
            r = await http.post("/jobs", json=payload, timeout=_JOB_SUBMIT_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            job_id: str = data["job_id"]
            log.info(
                "Job submitted to %s: job_id=%s config_id=%s",
                self._base_url, job_id, config_id,
            )
            return job_id
        except Exception as exc:
            raise NodeClientError(f"Job submission failed [{self._base_url}]: {exc}") from exc

    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Poll /jobs/{job_id} and return the status dict."""
        http = await self._get_http()
        try:
            r = await http.get(f"/jobs/{job_id}")
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            raise NodeClientError(f"Job status failed [{self._base_url}/{job_id}]: {exc}") from exc

    async def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = _POLL_INTERVAL,
        max_wait: float = _MAX_POLL_SECONDS,
    ) -> Dict[str, Any]:
        """
        Poll until the job reaches "done" or "failed", then return the status dict.

        Raises NodeClientError if ``max_wait`` is exceeded.
        """
        elapsed = 0.0
        while elapsed < max_wait:
            status = await self.get_job_status(job_id)
            if status["status"] in {"done", "failed"}:
                log.info(
                    "Job %s finished on %s: status=%s fitness=%s",
                    job_id, self._base_url, status["status"],
                    status.get("best_fitness"),
                )
                return status
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise NodeClientError(
            f"Job {job_id} timed out after {max_wait}s on {self._base_url}"
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "NodeClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
