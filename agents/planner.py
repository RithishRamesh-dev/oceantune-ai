"""
agents/planner.py
-----------------
Planner Agent — Stage 1, Round 0.

Responsibilities
----------------
1. Validate that the requested model is compatible with the target GPU(s).
2. Call the DO Serverless Inference LLM to order / prune the candidate
   VLLMFlags configs before they are queued in MongoDB.
3. Return an ordered list of config dicts with a brief LLM rationale for
   each choice.

The Planner does NOT launch any vLLM processes.  It is a pure reasoning step
that shapes the search before the Coordinator dispatches work.

Usage
-----
    from agents.planner import PlannerAgent
    planner = PlannerAgent(do_client=client, db=db)
    ordered = await planner.plan(
        session_id="...",
        model_id="deepseek-ai/DeepSeek-V3.2",
        gpu_type="H100",
        candidates=[flags1, flags2, ...],
    )
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import yaml
from pathlib import Path

from agents.do_client import DOClient, DOClientError
from core.db import Database
from core.search_space import VLLMFlags, ConfigValidator, SearchSpace

log = logging.getLogger("agents.planner")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODELS_YAML = _REPO_ROOT / "configs" / "models.yaml"
_GPU_PROFILES_YAML = _REPO_ROOT / "configs" / "gpu_profiles.yaml"


def _load_yaml(path: Path) -> Dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_SYSTEM_PROMPT = """\
You are an expert vLLM performance engineer.
Given a target model, GPU type, and a list of candidate vLLM flag configurations,
your task is to:
1. Filter out any configurations that are hardware-incompatible.
2. Rank the remaining configurations from most to least promising for maximising
   throughput (tokens/second).
3. Return ONLY a JSON array of the config indices in your ranked order, plus a
   one-sentence rationale for each.

Output format (strict JSON):
[
  {"index": <original_index>, "rationale": "<one sentence>"},
  ...
]

Do not include any text outside the JSON array.
"""


class PlannerAgent:
    """
    Hardware-aware config planner backed by DO Serverless Inference.

    Falls back to ConfigValidator-only ordering when the API key is absent.
    """

    def __init__(
        self,
        do_client: DOClient,
        db: Database,
        search_space: Optional[SearchSpace] = None,
    ) -> None:
        self._client = do_client
        self._db = db
        self._search_space = search_space
        self._models = _load_yaml(_MODELS_YAML)
        self._gpu_profiles = _load_yaml(_GPU_PROFILES_YAML)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def plan(
        self,
        *,
        session_id: str,
        model_id: str,
        gpu_type: str,
        candidates: List[VLLMFlags],
        max_configs: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Validate + order candidates for a session.

        Parameters
        ----------
        session_id : str
            MongoDB session ID (used for deduplication).
        model_id : str
            Hugging Face model ID.
        gpu_type : str
            Target GPU profile name (e.g. "H100").
        candidates : list of VLLMFlags
            Raw candidate configs from the search space.
        max_configs : int
            Cap on how many configs to queue.

        Returns
        -------
        list of dicts — ordered configs with keys:
            flags      : dict of VLLMFlags field → value
            fingerprint: str SHA1 fingerprint
            rationale  : str
        """
        # 1. Hardware-validate every candidate
        validator = ConfigValidator(gpu_type=gpu_type, model_id=model_id)
        valid: List[VLLMFlags] = []
        for c in candidates:
            errs = validator.validate(c)
            if not errs:
                valid.append(c)
            else:
                log.debug("Pruned invalid config %s: %s", c.fingerprint(), errs)

        if not valid:
            log.warning("All %d candidates failed hardware validation", len(candidates))
            return []

        log.info(
            "Planner: %d/%d configs passed hardware validation",
            len(valid), len(candidates),
        )

        # 2. Skip fingerprints already seen cross-session
        seen = set(
            await self._db.cross_session_seen_fingerprints(model_id, gpu_type)
        )
        novel = [c for c in valid if c.fingerprint() not in seen]
        log.info("Planner: %d novel configs (not seen in prior sessions)", len(novel))

        if not novel:
            return []

        # 3. LLM ranking (best-effort — falls back to validator order on error)
        ordered = await self._llm_rank(novel, model_id=model_id, gpu_type=gpu_type)

        # 4. Cap and return
        result = ordered[:max_configs]
        log.info("Planner: returning %d ordered configs", len(result))
        return result

    # ------------------------------------------------------------------
    # LLM ranking
    # ------------------------------------------------------------------

    async def _llm_rank(
        self,
        candidates: List[VLLMFlags],
        *,
        model_id: str,
        gpu_type: str,
    ) -> List[Dict[str, Any]]:
        """Call the LLM to rank candidates. Falls back to original order on failure."""
        # Build a compact summary of each candidate for the LLM
        summaries = []
        for i, c in enumerate(candidates):
            flags_dict = {
                k: v for k, v in c.__dict__.items() if v is not None
            }
            summaries.append({"index": i, "flags": flags_dict})

        gpu_profile = self._gpu_profiles.get(gpu_type, {})
        user_message = (
            f"Model: {model_id}\n"
            f"GPU: {gpu_type} ({gpu_profile.get('vram_gb', '?')} GB VRAM)\n"
            f"Candidate configs ({len(summaries)} total):\n"
            f"{json.dumps(summaries, indent=2)}\n\n"
            "Return your ranked JSON array now."
        )

        try:
            ranked: List[Dict] = await self._client.chat_json(
                messages=[{"role": "user", "content": user_message}],
                system=_SYSTEM_PROMPT,
            )

            # Build ordered result list
            result: List[Dict[str, Any]] = []
            seen_indices = set()
            for item in ranked:
                idx = item.get("index")
                if idx is None or idx in seen_indices or idx >= len(candidates):
                    continue
                seen_indices.add(idx)
                c = candidates[idx]
                result.append({
                    "flags": c.__dict__.copy(),
                    "fingerprint": c.fingerprint(),
                    "rationale": item.get("rationale", ""),
                })

            # Append any candidates the LLM omitted (preserve them at the end)
            for i, c in enumerate(candidates):
                if i not in seen_indices:
                    result.append({
                        "flags": c.__dict__.copy(),
                        "fingerprint": c.fingerprint(),
                        "rationale": "LLM did not rank this config; appended in order.",
                    })

            log.info("LLM ranking complete: %d items", len(result))
            return result

        except DOClientError as exc:
            log.warning("LLM ranking failed (%s); using validator order", exc)
            return [
                {
                    "flags": c.__dict__.copy(),
                    "fingerprint": c.fingerprint(),
                    "rationale": "LLM unavailable — ordered by ConfigValidator.",
                }
                for c in candidates
            ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_model_meta(self, model_id: str) -> Dict:
        """Look up model metadata from models.yaml by hf_id or alias."""
        for alias, meta in self._models.items():
            if meta.get("hf_id") == model_id or alias == model_id:
                return meta
        return {}
