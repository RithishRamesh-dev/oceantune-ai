"""
core/attempt_ledger.py
----------------------
Structured attempt / failure evidence for Planner + enablement.

Failed configs become typed rows (phase, error class, fingerprint) rather than
only free-text Recipe pitfalls.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.enablement import classify_boot_failure

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AttemptRow:
    phase: str
    fingerprint: str = ""
    flags: Dict[str, Any] = field(default_factory=dict)
    error_class: str = "unknown"
    error: str = ""
    fitness: float = 0.0
    kept: bool = False
    ts: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AttemptLedger:
    """In-memory + optional JSONL persistence for one session."""

    def __init__(self, session_id: str = "", persist: bool = True) -> None:
        self.session_id = session_id
        self._rows: List[AttemptRow] = []
        self._persist = persist
        self._path: Optional[Path] = None
        if persist and session_id:
            self._path = (
                REPO_ROOT / "storage" / "sessions" / session_id / "attempts.jsonl"
            )
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        phase: str,
        flags: Optional[Dict[str, Any]] = None,
        fingerprint: str = "",
        error: str = "",
        fitness: float = 0.0,
        kept: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AttemptRow:
        err = error or ""
        row = AttemptRow(
            phase=phase,
            fingerprint=fingerprint,
            flags=dict(flags or {}),
            error_class=classify_boot_failure(err) if err else ("ok" if kept else "unknown"),
            error=err[:500],
            fitness=float(fitness or 0.0),
            kept=bool(kept),
            ts=time.time(),
            metadata=dict(metadata or {}),
        )
        self._rows.append(row)
        if self._path is not None:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row.to_dict()) + "\n")
        return row

    @property
    def rows(self) -> List[AttemptRow]:
        return list(self._rows)

    def failures(self) -> List[AttemptRow]:
        return [r for r in self._rows if not r.kept and r.error]

    def reject_fingerprints(self) -> List[str]:
        return list(dict.fromkeys(
            r.fingerprint for r in self.failures() if r.fingerprint
        ))

    def planner_denylist_text(self, limit: int = 12) -> str:
        fails = self.failures()[-limit:]
        if not fails:
            return ""
        lines = ["Previous failed attempts (do not repeat):"]
        for r in fails:
            lines.append(
                f"- [{r.phase}/{r.error_class}] fp={r.fingerprint[:12] or 'n/a'} "
                f"{r.error[:120]}"
            )
        return "\n".join(lines)

    def to_list(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._rows]
