"""
core/session_checkpoint.py
--------------------------
Session resume + wall-clock / round budget tracking.

Persists a lightweight JSON cursor under storage/sessions/<id>/checkpoint.json
so interrupted runs can resume after Stage 1 / macro / Stage 4 boundaries.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("core.session_checkpoint")

REPO_ROOT = Path(__file__).resolve().parent.parent
SESSIONS_DIR = REPO_ROOT / "storage" / "sessions"


@dataclass
class RoundBudget:
    """Per-stage and session wall-clock budgets (seconds)."""

    session_max_sec: float = 0.0  # 0 = unlimited
    enablement_max_sec: float = 600.0
    stage1_max_sec: float = 0.0
    stage2_max_sec: float = 0.0
    stage3_max_sec: float = 0.0
    stage4_max_sec: float = 0.0
    sweep_max_sec: float = 900.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RoundBudget":
        return cls(**{
            k: float(d.get(k, getattr(cls, k)))
            for k in cls.__dataclass_fields__
        })


@dataclass
class SessionCheckpoint:
    session_id: str
    phase: str = "start"  # start|enablement|prelude|stage1|stage2|stage3|sweep|stage4|close|done
    winner_flags: Dict[str, Any] = field(default_factory=dict)
    best_strategy: Dict[str, Any] = field(default_factory=dict)
    stage1_fitness: float = 0.0
    stage2_fitness: float = 0.0
    stage3_fitness: float = 0.0
    macro_cycle: int = 0
    started_at: float = 0.0
    updated_at: float = 0.0
    budget: Dict[str, Any] = field(default_factory=dict)
    policy_denials: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionCheckpoint":
        return cls(
            session_id=str(d.get("session_id") or ""),
            phase=str(d.get("phase") or "start"),
            winner_flags=dict(d.get("winner_flags") or {}),
            best_strategy=dict(d.get("best_strategy") or {}),
            stage1_fitness=float(d.get("stage1_fitness") or 0.0),
            stage2_fitness=float(d.get("stage2_fitness") or 0.0),
            stage3_fitness=float(d.get("stage3_fitness") or 0.0),
            macro_cycle=int(d.get("macro_cycle") or 0),
            started_at=float(d.get("started_at") or 0.0),
            updated_at=float(d.get("updated_at") or 0.0),
            budget=dict(d.get("budget") or {}),
            policy_denials=list(d.get("policy_denials") or []),
            notes=list(d.get("notes") or []),
        )


def checkpoint_path(session_id: str) -> Path:
    return SESSIONS_DIR / session_id / "checkpoint.json"


def new_checkpoint(
    session_id: str,
    *,
    budget: Optional[RoundBudget] = None,
) -> SessionCheckpoint:
    now = time.time()
    b = budget or RoundBudget()
    return SessionCheckpoint(
        session_id=session_id,
        phase="start",
        started_at=now,
        updated_at=now,
        budget=b.to_dict(),
    )


def save_checkpoint(cp: SessionCheckpoint) -> Path:
    path = checkpoint_path(cp.session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    cp.updated_at = time.time()
    path.write_text(json.dumps(cp.to_dict(), indent=2), encoding="utf-8")
    return path


def load_checkpoint(session_id: str) -> Optional[SessionCheckpoint]:
    path = checkpoint_path(session_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return SessionCheckpoint.from_dict(data)
    except Exception as exc:
        log.warning("Failed to load checkpoint %s: %s", path, exc)
        return None


def session_remaining_sec(cp: SessionCheckpoint) -> Optional[float]:
    """Seconds remaining under session_max_sec, or None if unlimited."""
    max_sec = float((cp.budget or {}).get("session_max_sec") or 0.0)
    if max_sec <= 0:
        return None
    elapsed = time.time() - float(cp.started_at or time.time())
    return max(0.0, max_sec - elapsed)


def budget_exhausted(cp: SessionCheckpoint, stage: str = "") -> bool:
    rem = session_remaining_sec(cp)
    if rem is not None and rem <= 0:
        return True
    if stage:
        key = f"{stage}_max_sec"
        cap = float((cp.budget or {}).get(key) or 0.0)
        # Stage caps are advisory here; controller can track stage start separately.
        _ = cap
    return False


PHASE_ORDER = (
    "start",
    "enablement",
    "prelude",
    "stage1",
    "stage2",
    "stage3",
    "sweep",
    "stage4",
    "close",
    "done",
)


def should_skip_phase(cp: SessionCheckpoint, phase: str) -> bool:
    """When resuming, skip phases already completed."""
    try:
        done_idx = PHASE_ORDER.index(cp.phase)
        target_idx = PHASE_ORDER.index(phase)
    except ValueError:
        return False
    # cp.phase is the *next* phase to run, or last completed — treat as resume cursor:
    # skip if target is strictly before current cursor.
    return target_idx < done_idx
