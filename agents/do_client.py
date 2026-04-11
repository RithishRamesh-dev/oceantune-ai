"""
agents/do_client.py
-------------------
DigitalOcean Serverless Inference client used by all OceanTune agents.

All agents (Planner, Executor, Analyst, KernelOptimizer) call this module to
interact with the DO-hosted LLM API.  The client is a thin wrapper around the
OpenAI-compatible REST interface that DO Serverless Inference exposes.

Credentials
-----------
Set these environment variables (or put them in .env):
    DO_INFERENCE_KEY       — your DigitalOcean API key
    DO_INFERENCE_ENDPOINT  — base URL (default: https://inference.do-ai.run/v1)
    DO_INFERENCE_MODEL     — model override (optional; "auto" picks from registry)

Usage
-----
    from agents.do_client import DOClient
    client = DOClient.from_env()
    reply = await client.chat([{"role": "user", "content": "Hello!"}])
    print(reply)
"""

from __future__ import annotations

import json
import logging
import os
import asyncio
from typing import Any, Dict, List, Optional

import httpx

log = logging.getLogger("agents.do_client")

# Default DO Serverless Inference base URL
_DEFAULT_ENDPOINT = "https://inference.do-ai.run/v1"

# Retry settings
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0  # seconds; doubled each retry


class DOClientError(Exception):
    """Raised when the DO Inference API returns an error after all retries."""


class DOClient:
    """
    Async HTTP client for DigitalOcean Serverless Inference (OpenAI-compatible).

    Parameters
    ----------
    api_key : str
        DO API key. If empty, ``chat()`` raises DOClientError immediately.
    endpoint : str
        Base URL for the inference API.
    model : str
        Model ID to send in each request.
    max_tokens : int
        Maximum completion tokens per call.
    temperature : float
        Sampling temperature.
    timeout_sec : float
        Per-request HTTP timeout in seconds.
    """

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str = _DEFAULT_ENDPOINT,
        model: str = "deepseek-ai/DeepSeek-R1",
        max_tokens: int = 4096,
        temperature: float = 0.3,
        timeout_sec: float = 120.0,
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_sec = timeout_sec

        self._http: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        *,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        timeout_sec: float = 120.0,
    ) -> "DOClient":
        """Build a DOClient from environment variables."""
        api_key = os.getenv("DO_INFERENCE_KEY", "")
        endpoint = os.getenv("DO_INFERENCE_ENDPOINT", _DEFAULT_ENDPOINT)
        resolved_model = (
            model
            or os.getenv("DO_INFERENCE_MODEL", "")
            or "deepseek-ai/DeepSeek-R1"
        )
        return cls(
            api_key=api_key,
            endpoint=endpoint,
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_sec=timeout_sec,
        )

    # ------------------------------------------------------------------
    # HTTP lifecycle
    # ------------------------------------------------------------------

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout_sec,
            )
        return self._http

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    # ------------------------------------------------------------------
    # Core chat method
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        system: Optional[str] = None,
        json_mode: bool = False,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Send a chat completion request and return the assistant's reply text.

        Parameters
        ----------
        messages : list of {"role": ..., "content": ...} dicts
        system : optional system prompt prepended to messages
        json_mode : if True, adds ``response_format={"type": "json_object"}``
        extra_kwargs : any additional body fields passed through verbatim

        Returns
        -------
        str — the assistant message content

        Raises
        ------
        DOClientError — on auth failure, timeout, or non-200 after retries
        """
        if not self.api_key:
            raise DOClientError(
                "DO_INFERENCE_KEY is not set. "
                "Export it in your environment or .env file."
            )

        full_messages: List[Dict[str, str]] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": full_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if extra_kwargs:
            payload.update(extra_kwargs)

        url = f"{self.endpoint}/chat/completions"
        http = await self._get_http()

        last_exc: Optional[Exception] = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                log.debug(
                    "DO Inference request attempt=%d model=%s messages=%d",
                    attempt, self.model, len(full_messages),
                )
                response = await http.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                content: str = data["choices"][0]["message"]["content"]
                log.debug("DO Inference response tokens=%s", data.get("usage"))
                return content

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                body = exc.response.text[:500]
                log.warning(
                    "DO Inference HTTP %d attempt=%d: %s", status, attempt, body
                )
                # 4xx are not retryable (except 429 rate-limit)
                if status != 429 and 400 <= status < 500:
                    raise DOClientError(f"HTTP {status}: {body}") from exc
                last_exc = exc

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                log.warning("DO Inference network error attempt=%d: %s", attempt, exc)
                last_exc = exc

            if attempt < _MAX_RETRIES:
                wait = _RETRY_BACKOFF * (2 ** (attempt - 1))
                log.info("Retrying in %.1fs...", wait)
                await asyncio.sleep(wait)

        raise DOClientError(
            f"DO Inference API failed after {_MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Convenience: parse JSON response
    # ------------------------------------------------------------------

    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        system: Optional[str] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Like ``chat()`` but sets json_mode=True and parses the response.
        Returns a Python object (dict or list).
        """
        text = await self.chat(
            messages,
            system=system,
            json_mode=True,
            extra_kwargs=extra_kwargs,
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise DOClientError(f"Response was not valid JSON: {text[:500]}") from exc

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "DOClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
