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

import copy
import json
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

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

_PROPOSE_SYSTEM_PROMPT = """\
You are an expert vLLM performance engineer doing iterative hyperparameter optimization.
Your goal: maximize throughput (tokens/second) on a single-GPU vLLM server.

You will receive:
- The model and GPU details
- The current best configuration and its benchmark metrics (including the full concurrency curve)
- History of all configurations tried so far and their results (including any server errors)
- The analyst's evaluation of the most recent run: bottleneck diagnosis + recommendation

Your task:
1. Read the analyst's evaluation first — it contains a bottleneck diagnosis and a specific
   recommendation. Use it as your primary signal for what to change.
2. Cross-check with the concurrency curve: if throughput is still scaling at the highest
   concurrency tested, the model is NOT saturated and increasing max_num_seqs or
   max_num_batched_tokens may help. If it plateaus, the GPU is compute-bound.
3. If any prior iteration has an "error" field, read it carefully — you MUST NOT propose
   a configuration that would trigger the same error.
4. Propose the single most impactful configuration change to try next.
5. Return a JSON object with the full new configuration and a rationale.

Output format (strict JSON):
{
  "flags": {<complete VLLMFlags — all fields must be present>},
  "rationale": "<2-3 sentences: what you changed, why, what bottleneck you're targeting>"
}

Hard constraints (never violate these):
- tensor_parallel_size: always 1 (single GPU)
- pipeline_parallel_size: always 1
- data_parallel_size: always 1
- distributed_executor_backend: always "mp" (Ray is not installed)
- cpu_offload_gb: always 0 (GPU has enough VRAM)
- speculative_model: always null (not applicable here)
- num_speculative_tokens: always null

If history shows "No module named 'ray'" → distributed_executor_backend must be "mp", never "ray".
If history shows "World size.*larger than.*GPUs" → tensor_parallel_size must be 1, always.
If history shows "CUDA out of memory" → reduce gpu_memory_utilization by at least 0.05.
If history shows "GatedRepoError" or "401" → this is an auth issue, not a config issue; keep flags.

Tunable single-GPU parameters (choose ONE to change from current best):
- gpu_memory_utilization: [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
- block_size: [8, 16, 32] — use 16 for most models
- kv_cache_dtype: ["auto", "fp8"] — fp8 reduces KV memory on H100/H200
- enable_prefix_caching: [true, false]
- max_num_seqs: [32, 64, 128, 256, 512]
- max_num_batched_tokens: [2048, 4096, 8192, 16384, 32768, 65536]
- dtype: ["auto", "bfloat16", "float16"]
- max_model_len: [4096, 8192, 16384, 32768]
- enforce_eager: [true, false]
- enable_chunked_prefill: [true, false]

Do not include any text outside the JSON object.
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
        validator = ConfigValidator(gpu_type=gpu_type)
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
    # Iterative proposal (agent-driven search)
    # ------------------------------------------------------------------

    # Fallback variations tried in order when the LLM is unavailable.
    # Each entry is a dict of field overrides applied to the current best.
    _FALLBACK_VARIATIONS: List[Dict[str, Any]] = [
        {"gpu_memory_utilization": 0.95},
        {"enable_prefix_caching": True},
        {"max_num_batched_tokens": 16384},
        {"kv_cache_dtype": "fp8"},
        {"enable_chunked_prefill": True},
        {"max_num_seqs": 512},
        {"block_size": 32},
        {"dtype": "bfloat16"},
        {"max_num_batched_tokens": 32768},
        {"gpu_memory_utilization": 0.85},
    ]

    async def propose_next(
        self,
        *,
        model_id: str,
        gpu_type: str,
        n_gpus: int,
        current_best: VLLMFlags,
        current_best_metrics: Dict[str, Any],
        history: List[Dict[str, Any]],
        iteration: int = 0,
        analyst_eval: Optional[Dict[str, Any]] = None,
    ) -> Tuple[VLLMFlags, str]:
        """
        Ask the LLM to propose the next configuration to benchmark.

        Parameters
        ----------
        analyst_eval : optional dict from AnalystAgent.evaluate_iteration()
            Contains: bottleneck, diagnosis, flag_insights, recommendation.
            When provided, the LLM uses this as the primary signal for what
            to change next.

        Returns (VLLMFlags, rationale_string).
        Falls back to a curated list of single-parameter variations when
        the LLM is unavailable.
        """
        gpu_profile = self._gpu_profiles.get("gpu_profiles", {}).get(gpu_type, {})
        current_dict = asdict(current_best)

        # Build history summary (limit to last 5 to stay within token budget)
        recent_history = history[-5:] if len(history) > 5 else history
        history_summary = []
        for i, h in enumerate(recent_history):
            entry = {
                "iteration": h.get("iteration", i),
                "flags": h.get("flags", {}),
                "fitness": h.get("fitness"),
                # use canonical enriched_metrics field names
                "peak_tok_s": h.get("enriched_metrics", {}).get(
                    "peak_throughput_tokens_per_sec"
                ),
                "p95_latency_ms": h.get("enriched_metrics", {}).get(
                    "p95_latency_at_peak_ms"
                ),
                "best_concurrency": h.get("enriched_metrics", {}).get(
                    "best_concurrency"
                ),
                "analyst_recommendation": h.get("analyst_recommendation", ""),
            }
            if h.get("error"):
                entry["error"] = h["error"][:500]
            history_summary.append(entry)

        # Analyst evaluation of the most recent run
        eval_section = ""
        if analyst_eval:
            eval_section = (
                f"\nAnalyst evaluation of most recent run:\n"
                f"  Bottleneck: {analyst_eval.get('bottleneck', '?')}\n"
                f"  Diagnosis:  {analyst_eval.get('diagnosis', '')}\n"
                f"  Flag insights: {analyst_eval.get('flag_insights', '')}\n"
                f"  Recommendation: {analyst_eval.get('recommendation', '')}\n"
            )

        user_msg = (
            f"Model: {model_id}\n"
            f"GPU: {gpu_type} ({gpu_profile.get('vram_gb', '?')}GB VRAM)\n"
            f"Available GPUs: {n_gpus}\n\n"
            f"Current best configuration:\n{json.dumps(current_dict, indent=2)}\n\n"
            f"Current best metrics:\n{json.dumps(current_best_metrics, indent=2)}\n"
            f"{eval_section}\n"
            f"History (last {len(recent_history)} iterations):\n"
            f"{json.dumps(history_summary, indent=2)}\n\n"
            f"Propose the next configuration to try."
        )

        try:
            result = await self._client.chat_json(
                messages=[{"role": "user", "content": user_msg}],
                system=_PROPOSE_SYSTEM_PROMPT,
            )
            if not isinstance(result, dict) or "flags" not in result:
                raise DOClientError("Invalid response format")

            # Build VLLMFlags from LLM response, enforce single-GPU constraints
            flags_dict = result["flags"]
            flags_dict.update({
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 1,
                "distributed_executor_backend": "mp",
                "cpu_offload_gb": 0,
                "speculative_model": None,
                "num_speculative_tokens": None,
            })
            # Build VLLMFlags, ignoring any unknown keys
            known_fields = set(VLLMFlags.__dataclass_fields__.keys())
            safe_dict = {k: v for k, v in flags_dict.items() if k in known_fields}
            proposed = copy.deepcopy(current_best)
            for k, v in safe_dict.items():
                setattr(proposed, k, v)
            proposed.run_id = proposed.fingerprint()

            rationale = result.get("rationale", "LLM proposal")
            log.info("LLM proposed config: %s", rationale[:100])
            return proposed, rationale

        except DOClientError as exc:
            log.warning("LLM proposal failed (%s); using fallback variation", exc)
            # Collect all errors from history to skip unsafe variations
            all_errors = " ".join(h.get("error", "") for h in history if h.get("error"))

            # Cycle through fallback variations, skipping unsafe ones
            tried = 0
            idx = iteration % len(self._FALLBACK_VARIATIONS)
            while tried < len(self._FALLBACK_VARIATIONS):
                overrides = self._FALLBACK_VARIATIONS[idx]
                # Skip variations that would trigger known errors
                skip = False
                if "ray" in all_errors.lower() and overrides.get("distributed_executor_backend") == "ray":
                    skip = True
                if "out of memory" in all_errors.lower() and overrides.get("gpu_memory_utilization", 0) > 0.90:
                    skip = True
                if not skip:
                    break
                idx = (idx + 1) % len(self._FALLBACK_VARIATIONS)
                tried += 1

            proposed = copy.deepcopy(current_best)
            for k, v in overrides.items():
                setattr(proposed, k, v)
            proposed.run_id = proposed.fingerprint()
            rationale = f"Fallback variation {idx+1}: {overrides}"
            return proposed, rationale

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_model_meta(self, model_id: str) -> Dict:
        """Look up model metadata from models.yaml by hf_id or alias."""
        for alias, meta in self._models.items():
            if meta.get("hf_id") == model_id or alias == model_id:
                return meta
        return {}
