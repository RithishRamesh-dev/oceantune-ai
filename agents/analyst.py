"""
agents/analyst.py
-----------------
Analyst Agent — Stage 1 finale.

After the Coordinator finishes all benchmark runs for a session, the Analyst:
  1. Reads the benchmark_runs collection from MongoDB.
  2. Runs aggregation pipelines (top configs, OOM patterns, perf over time).
  3. Calls the DO Serverless Inference LLM to:
     a. Identify the winner configuration.
     b. Explain *why* it won (which flags mattered most).
     c. Produce a human-readable analysis summary.
  4. Returns an AnalysisResult with the winner flags + LLM commentary.

Usage
-----
    analyst = AnalystAgent(do_client=client, db=db)
    result = await analyst.analyse(
        session_id="...",
        model_id="deepseek-ai/DeepSeek-V3.2",
        gpu_type="H100",
    )
    print(result.winner_flags)
    print(result.summary)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.do_client import DOClient, DOClientError
from core.db import Database

log = logging.getLogger("agents.analyst")


_ANALYSIS_SYSTEM_PROMPT = """\
You are an expert vLLM performance analyst.
You will be given:
  - The top benchmark results for a session (model, GPU, flags, metrics).
  - OOM pattern data.
  - A time-series of fitness scores showing optimiser progress.

Your task:
1. Identify the single best configuration and explain WHY it outperforms others
   (e.g., which flags drove the improvement, and the likely mechanism).
2. Call out any risky or surprising flag combinations found in the top configs.
3. Summarise the OOM patterns — which flags are associated with OOM failures.
4. Note whether the search converged or if more generations are recommended.

Respond with a JSON object:
{
  "winner_index": <index in top_configs list>,
  "key_flags": [<flag_name>, ...],
  "explanation": "<2-4 sentence explanation>",
  "oom_insight": "<1-2 sentences about OOM patterns, or null>",
  "convergence_note": "<1 sentence about convergence>",
  "recommendation": "<one actionable next step>"
}
"""


@dataclass
class AnalysisResult:
    """Output of the Analyst agent."""
    session_id: str
    winner_flags: Dict[str, Any]
    winner_fingerprint: str
    winner_fitness: float
    key_flags: List[str]
    explanation: str
    oom_insight: Optional[str]
    convergence_note: str
    recommendation: str
    top_configs: List[Dict[str, Any]] = field(default_factory=list)
    raw_llm_response: Optional[str] = None


class AnalystAgent:
    """
    Post-search analysis agent.

    Parameters
    ----------
    do_client : DOClient
        Shared DO Serverless Inference client.
    db : Database
        Connected MongoDB client.
    """

    def __init__(self, do_client: DOClient, db: Database) -> None:
        self._client = do_client
        self._db = db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyse(
        self,
        *,
        session_id: str,
        model_id: str,
        gpu_type: str,
        top_n: int = 10,
    ) -> AnalysisResult:
        """
        Run the full analysis pipeline for a completed session.

        Returns an AnalysisResult with the winner config and LLM commentary.
        """
        log.info("Analyst: analysing session %s", session_id)

        # 1. Pull analytics data from MongoDB
        top_runs = await self._db.get_top_configs(session_id, n=top_n)
        oom_data = await self._db.oom_patterns(session_id)
        perf_series = await self._db.performance_over_time(session_id)
        throughput_agg = await self._db.top_configs_by_throughput(session_id, n=top_n)

        if not top_runs:
            log.warning("Analyst: no benchmark runs found for session %s", session_id)
            return self._empty_result(session_id)

        best_run = top_runs[0]

        # 2. Fetch the config document to get the flags
        from bson import ObjectId
        config_doc = await self._db.db["configs"].find_one(
            {"_id": ObjectId(best_run["config_id"])}
        )
        winner_flags = config_doc["flags"] if config_doc else {}
        winner_fingerprint = best_run.get("fingerprint", "")
        winner_fitness = best_run.get("fitness_score", 0.0)

        # 3. Call LLM for analysis (best-effort)
        llm_result = await self._llm_analyse(
            top_runs=top_runs,
            oom_data=oom_data,
            perf_series=perf_series,
            model_id=model_id,
            gpu_type=gpu_type,
        )

        log.info(
            "Analyst: winner fingerprint=%s fitness=%.4f",
            winner_fingerprint[:8], winner_fitness,
        )

        return AnalysisResult(
            session_id=session_id,
            winner_flags=winner_flags,
            winner_fingerprint=winner_fingerprint,
            winner_fitness=winner_fitness,
            key_flags=llm_result.get("key_flags", []),
            explanation=llm_result.get("explanation", ""),
            oom_insight=llm_result.get("oom_insight"),
            convergence_note=llm_result.get("convergence_note", ""),
            recommendation=llm_result.get("recommendation", ""),
            top_configs=top_runs,
            raw_llm_response=llm_result.get("_raw"),
        )

    # ------------------------------------------------------------------
    # LLM analysis
    # ------------------------------------------------------------------

    async def _llm_analyse(
        self,
        *,
        top_runs: List[Dict],
        oom_data: List[Dict],
        perf_series: List[Dict],
        model_id: str,
        gpu_type: str,
    ) -> Dict[str, Any]:
        """Call the LLM to analyse results. Returns parsed JSON or empty dict."""
        # Build compact payload (avoid sending huge raw_metrics blobs)
        top_summary = [
            {
                "index": i,
                "fingerprint": r.get("fingerprint", "")[:12],
                "fitness_score": r.get("fitness_score"),
                "raw_metrics": {
                    k: r.get("raw_metrics", {}).get(k)
                    for k in ["throughput_tok_s", "p95_latency_ms", "ttft_ms", "error_rate"]
                },
            }
            for i, r in enumerate(top_runs)
        ]
        oom_summary = oom_data[:10]  # cap to avoid token bloat

        # Fitness progression: sample every 5th point
        fitness_series = [
            {"t": i, "fitness": r.get("fitness_score")}
            for i, r in enumerate(perf_series)
            if i % 5 == 0
        ][-50:]  # last 50 samples

        user_msg = (
            f"Model: {model_id}\nGPU: {gpu_type}\n\n"
            f"Top {len(top_summary)} configs:\n{json.dumps(top_summary, indent=2)}\n\n"
            f"OOM patterns ({len(oom_summary)} samples):\n{json.dumps(oom_summary, indent=2)}\n\n"
            f"Fitness time-series ({len(fitness_series)} points):\n"
            f"{json.dumps(fitness_series, indent=2)}\n\n"
            "Provide your analysis JSON now."
        )

        try:
            raw_text = await self._client.chat(
                messages=[{"role": "user", "content": user_msg}],
                system=_ANALYSIS_SYSTEM_PROMPT,
                json_mode=True,
            )
            import json as _json
            parsed = _json.loads(raw_text)
            parsed["_raw"] = raw_text
            return parsed
        except (DOClientError, Exception) as exc:
            log.warning("Analyst LLM call failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Fallback for empty sessions
    # ------------------------------------------------------------------

    def _empty_result(self, session_id: str) -> AnalysisResult:
        return AnalysisResult(
            session_id=session_id,
            winner_flags={},
            winner_fingerprint="",
            winner_fitness=0.0,
            key_flags=[],
            explanation="No benchmark runs found for this session.",
            oom_insight=None,
            convergence_note="",
            recommendation="Run the optimiser with more candidates.",
        )
