"""
core/session_breakdown.py
-------------------------
Machine-readable session CLOSE artifact (Hyperloom session_breakdown analog).

Written to storage/results/session_breakdown_<session_id>.json after every
successful pipeline run so dashboards, Recipe KB, and CI can consume a stable
schema without parsing Markdown reports.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("core.session_breakdown")

_SCHEMA_VERSION = "1.0.0"
_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "storage" / "results"


def build_session_breakdown(
    *,
    session_id: str,
    model_id: str,
    gpu_type: str,
    stage1_fitness: float = 0.0,
    stage2_fitness: float = 0.0,
    stage3_fitness: float = 0.0,
    winner_flags: Optional[Dict[str, Any]] = None,
    stage2_strategy: Optional[Dict[str, Any]] = None,
    stage3_applied_recs: Optional[List[Dict[str, Any]]] = None,
    peak_throughput: float = 0.0,
    best_concurrency: int = 0,
    bottleneck_primary: str = "",
    attention_backend: str = "",
    recipe_id: str = "",
    stage4_speedup_pct: float = 0.0,
    stage4_kernel_path: str = "",
    stop_reason: str = "pipeline_complete",
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the session_breakdown document."""
    final_fitness = max(stage3_fitness, stage2_fitness, stage1_fitness)
    return {
        "schema_version": _SCHEMA_VERSION,
        "session_id": session_id,
        "model_id": model_id,
        "gpu_type": gpu_type,
        "framework": "vllm",
        "closed_at": datetime.now(timezone.utc).isoformat(),
        "stop_reason": stop_reason,
        "fitness": {
            "stage1": stage1_fitness,
            "stage2": stage2_fitness,
            "stage3": stage3_fitness,
            "final": final_fitness,
            "gain_vs_stage1_pct": (
                ((final_fitness - stage1_fitness) / stage1_fitness * 100.0)
                if stage1_fitness > 0
                else 0.0
            ),
        },
        "performance": {
            "peak_throughput_tokens_per_sec": peak_throughput,
            "best_concurrency": best_concurrency,
        },
        "config": {
            "winner_flags": dict(winner_flags or {}),
            "stage2_strategy": dict(stage2_strategy or {}),
            "attention_backend": attention_backend
            or (winner_flags or {}).get("attention_backend", ""),
        },
        "stage3": {
            "bottleneck_primary": bottleneck_primary,
            "applied_recommendations": list(stage3_applied_recs or []),
        },
        "stage4": {
            "speedup_pct": stage4_speedup_pct,
            "kernel_path": stage4_kernel_path,
            "integrated_into_serving": False,  # until KernelIntegrationLayer lands
        },
        "recipe_id": recipe_id,
        "extras": dict(extras or {}),
    }


def write_session_breakdown(
    breakdown: Dict[str, Any],
    *,
    output_dir: Optional[Path] = None,
) -> Path:
    """Atomically write session_breakdown JSON; returns path."""
    out = output_dir or _DEFAULT_DIR
    out.mkdir(parents=True, exist_ok=True)
    sid = breakdown.get("session_id", "unknown")[:24]
    path = out / f"session_breakdown_{sid}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(breakdown, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    log.info("Wrote session breakdown: %s", path)
    return path
