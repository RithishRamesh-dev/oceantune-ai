"""
core/kernel_ledger.py
---------------------
Kernel provenance / experience ledger (Hyperloom experience_ledger analog).

Tracks Stage 4 kernel attempts with identity, SNR, speedup, E2E verdict, and
session linkage — beyond flag-only Recipe KB.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("core.kernel_ledger")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DIR = _REPO_ROOT / "storage" / "kernel_ledger"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def kernel_identity(
    *,
    op_type: str,
    model_id: str,
    gpu_type: str,
    kernel_path: str = "",
    fusion_pattern: str = "",
) -> str:
    raw = "|".join([
        (op_type or "").lower(),
        (model_id or "").lower(),
        (gpu_type or "").upper(),
        (fusion_pattern or "").lower(),
        Path(kernel_path).name if kernel_path else "",
    ])
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


@dataclass
class KernelLedgerEntry:
    entry_id: str
    session_id: str
    identity: str
    op_type: str
    model_id: str
    gpu_type: str
    kernel_path: str = ""
    fusion_pattern: str = ""
    decision: str = ""  # kept | reverted | e2e_integrated | e2e_rejected | blocked
    snr_db: Optional[float] = None
    micro_speedup_pct: float = 0.0
    e2e_fitness_delta: float = 0.0
    notes: str = ""
    created_at: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class KernelLedger:
    """Append-only JSONL ledger + optional Mongo mirror."""

    def __init__(
        self,
        *,
        root: Optional[Path] = None,
        db: Any = None,
    ) -> None:
        self.root = root or _DEFAULT_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self._db = db

    def _path(self, session_id: str) -> Path:
        return self.root / f"ledger_{session_id[:24]}.jsonl"

    def append(self, entry: KernelLedgerEntry) -> Path:
        if not entry.created_at:
            entry.created_at = _utc()
        path = self._path(entry.session_id)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), default=str) + "\n")
        log.info(
            "KernelLedger: %s decision=%s op=%s",
            entry.entry_id[:8], entry.decision, entry.op_type,
        )
        return path

    def record(
        self,
        *,
        session_id: str,
        op_type: str,
        model_id: str,
        gpu_type: str,
        kernel_path: str = "",
        fusion_pattern: str = "",
        decision: str = "",
        snr_db: Optional[float] = None,
        micro_speedup_pct: float = 0.0,
        e2e_fitness_delta: float = 0.0,
        notes: str = "",
        extras: Optional[Dict[str, Any]] = None,
    ) -> KernelLedgerEntry:
        ident = kernel_identity(
            op_type=op_type,
            model_id=model_id,
            gpu_type=gpu_type,
            kernel_path=kernel_path,
            fusion_pattern=fusion_pattern,
        )
        entry = KernelLedgerEntry(
            entry_id=hashlib.sha256(
                f"{session_id}:{ident}:{decision}:{_utc()}".encode()
            ).hexdigest()[:16],
            session_id=session_id,
            identity=ident,
            op_type=op_type,
            model_id=model_id,
            gpu_type=gpu_type,
            kernel_path=kernel_path,
            fusion_pattern=fusion_pattern,
            decision=decision,
            snr_db=snr_db,
            micro_speedup_pct=micro_speedup_pct,
            e2e_fitness_delta=e2e_fitness_delta,
            notes=notes,
            extras=dict(extras or {}),
        )
        self.append(entry)
        return entry

    def list_session(self, session_id: str) -> List[KernelLedgerEntry]:
        path = self._path(session_id)
        if not path.is_file():
            return []
        out: List[KernelLedgerEntry] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            out.append(KernelLedgerEntry(**{
                k: d.get(k) for k in KernelLedgerEntry.__dataclass_fields__
            }))
        return out

    def lessons_for_recipe(self, session_id: str) -> List[Dict[str, Any]]:
        """Convert kept/integrated entries into Recipe KB lesson-shaped dicts."""
        lessons = []
        for e in self.list_session(session_id):
            if e.decision not in ("kept", "e2e_integrated"):
                continue
            lessons.append({
                "name": f"kernel_{e.op_type}_{e.decision}",
                "description": (
                    f"{e.op_type} {e.decision}: micro +{e.micro_speedup_pct:.1f}% "
                    f"e2e_delta={e.e2e_fitness_delta:+.4f} pattern={e.fusion_pattern or '-'}"
                ),
                "source": "kernel_ledger",
                "identity": e.identity,
            })
        return lessons
