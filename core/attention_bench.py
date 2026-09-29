"""
core/attention_bench.py
-----------------------
Phase 1 attention benchmark suite: isolated microbench + optional short E2E
per attention_backend.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from core.kernel_registry import CapabilityDetector
from core.db import Database
from core.flag_merge import merge_flags, to_vllm_flags_dict
from core.kernel_benchmark_engine import KernelBenchmarkEngine
from core.kernel_registry import KernelRegistry, model_has_gqa
from core.search_space import VLLMFlags

log = logging.getLogger("core.attention_bench")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MATRIX_PATH = _REPO_ROOT / "configs" / "attention_benchmark_matrix.yaml"


def load_model_meta(model_id: str) -> Dict[str, Any]:
    """Look up models.yaml entry by hf_id."""
    path = _REPO_ROOT / "configs" / "models.yaml"
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        for _key, meta in (raw.get("models") or {}).items():
            if meta.get("hf_id") == model_id:
                return meta
    except OSError:
        pass
    return {}


class AttentionBenchmarkSuite:
    """
    Phase 1 attention benchmarks tied to kernel_benchmark_runs + kernel_runs.
    """

    def __init__(
        self,
        *,
        db: Database,
        gpu_type: str,
        model_id: str,
        matrix_path: Optional[Path] = None,
    ) -> None:
        self._db = db
        self._gpu_type = gpu_type
        self._model_id = model_id
        self._matrix_path = matrix_path or _MATRIX_PATH
        with open(self._matrix_path, encoding="utf-8") as f:
            self._matrix = yaml.safe_load(f) or {}
        self._registry = KernelRegistry()
        self._capabilities = CapabilityDetector(self._registry)
        self._model_meta = load_model_meta(model_id)
        self._engine = KernelBenchmarkEngine(
            db=db, gpu_type=gpu_type, matrix_path=self._matrix_path
        )

    async def seed_kernel_metadata(self) -> int:
        """Upsert attention registry into MongoDB kernel_metadata."""
        n = 0
        for doc in self._registry.metadata_documents():
            if await self._db.upsert_kernel_metadata(doc):
                n += 1
        return n

    async def run_microbench_suite(self, session_id: str) -> List[str]:
        """
        Isolated attention microbench for each supported backend label.
        """
        backends = self._capabilities.supported_attention_backends(
            self._gpu_type, self._model_meta, self._model_id
        )
        if not backends:
            backends = ["FLASH_ATTN"]
        log.info(
            "Attention microbench suite: backends=%s session=%s",
            backends, session_id,
        )
        return await self._engine.run_attention_microbench_suite(
            session_id, backends=backends
        )

    async def run_e2e_backend_trials(
        self,
        *,
        session_id: str,
        baseline_flags: Dict[str, Any],
        benchmark_fn,
    ) -> List[Dict[str, Any]]:
        """
        Short E2E ramp per attention_backend (optional, matrix e2e.enabled).

        benchmark_fn: async (session_id, iteration, baseline_flags, override, ...)
            -> (fitness, enriched_metrics) — StrategyOptimizer._benchmark_strategy.
        """
        e2e = self._matrix.get("e2e") or {}
        if not e2e.get("enabled", False):
            return []

        backends = e2e.get("backends") or ["FLASH_ATTN", "FLASHINFER"]
        allowed = set(
            self._capabilities.supported_attention_backends(
                self._gpu_type, self._model_meta, self._model_id
            )
        )
        results: List[Dict[str, Any]] = []
        iteration = 9000
        for backend in backends:
            if backend not in allowed:
                continue
            override = {"attention_backend": backend}
            fitness, em = await benchmark_fn(
                session_id=session_id,
                iteration=iteration,
                baseline_flags=baseline_flags,
                strategy_override=override,
                llm_reasoning=f"Attention E2E trial: {backend}",
                category="attention_e2e",
            )
            results.append({
                "backend": backend,
                "fitness_score": fitness,
                "enriched_metrics": em,
            })
            iteration += 1
        return results

    def attention_sweep_proposals(
        self,
        baseline_flags: Dict[str, Any],
        *,
        attention_pct: float = 0.0,
        gemm_pct: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Deterministic Stage 2 queue entries for attention backend comparison.
        """
        if attention_pct + gemm_pct <= 50.0:
            return []

        current = baseline_flags.get("attention_backend", "FLASH_ATTN")
        util = float(baseline_flags.get("gpu_memory_utilization", 0.9) or 0.9)
        backends = self._capabilities.supported_attention_backends(
            self._gpu_type, self._model_meta, self._model_id
        )
        proposals: List[Dict[str, Any]] = []

        if self._capabilities.flashinfer_recommended(
            gpu_type=self._gpu_type,
            model_meta=self._model_meta,
            model_id=self._model_id,
            gpu_memory_utilization=util,
        ) and "FLASHINFER" in backends and current != "FLASHINFER":
            proposals.append({
                "strategy_config": {"attention_backend": "FLASHINFER"},
                "category": "kernel",
                "rationale": (
                    "Phase 1 policy: GQA model with gpu_memory_utilization>=0.85; "
                    "FLASHINFER often improves paged GQA decode."
                ),
            })

        for backend in backends:
            if backend == current:
                continue
            if any(p["strategy_config"].get("attention_backend") == backend for p in proposals):
                continue
            proposals.append({
                "strategy_config": {"attention_backend": backend},
                "category": "kernel",
                "rationale": (
                    f"Phase 1 attention sweep: compare {backend} vs {current} "
                    f"(trace attention={attention_pct:.0f}% gemm={gemm_pct:.0f}%)."
                ),
            })
        return proposals
