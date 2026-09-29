"""
core/draft_registry.py
----------------------
Speculative-decoding draft model registry.

Stage 2 must not propose speculative_model without a known compatible draft.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

log = logging.getLogger("core.draft_registry")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PATH = _REPO_ROOT / "configs" / "draft_models.yaml"


@dataclass
class DraftPair:
    target_model: str
    draft_model: str
    num_speculative_tokens: int = 5
    notes: str = ""
    gpu_vendors: List[str] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.gpu_vendors is None:
            self.gpu_vendors = ["nvidia", "amd"]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_draft_pairs(path: Optional[Path] = None) -> List[DraftPair]:
    p = path or _DEFAULT_PATH
    if not p.is_file():
        return []
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out: List[DraftPair] = []
    for row in raw.get("pairs") or []:
        out.append(DraftPair(
            target_model=str(row["target_model"]),
            draft_model=str(row["draft_model"]),
            num_speculative_tokens=int(row.get("num_speculative_tokens") or 5),
            notes=str(row.get("notes") or ""),
            gpu_vendors=list(row.get("gpu_vendors") or ["nvidia", "amd"]),
        ))
    return out


def find_draft_for_target(
    target_model: str,
    *,
    vendor: str = "nvidia",
    path: Optional[Path] = None,
) -> Optional[DraftPair]:
    tid = (target_model or "").strip().lower()
    for pair in load_draft_pairs(path):
        if pair.target_model.lower() != tid and not tid.endswith(
            pair.target_model.lower().split("/")[-1]
        ):
            # also allow substring match on HF id
            if pair.target_model.lower() not in tid and tid not in pair.target_model.lower():
                continue
        if vendor not in (pair.gpu_vendors or ["nvidia", "amd"]):
            continue
        return pair
    return None


def speculative_strategy_allowed(
    target_model: str,
    proposed: Dict[str, Any],
    *,
    vendor: str = "nvidia",
) -> tuple[bool, str]:
    """Return (ok, reason) for a Stage 2 speculative proposal."""
    draft = proposed.get("speculative_model")
    if not draft:
        return True, "no_speculative"
    pair = find_draft_for_target(target_model, vendor=vendor)
    if pair is None:
        return False, f"no_registered_draft_for:{target_model}"
    if str(draft) != pair.draft_model:
        return False, f"draft_mismatch:expected={pair.draft_model}"
    return True, "ok"


def planner_speculative_hint(target_model: str, *, vendor: str = "nvidia") -> str:
    pair = find_draft_for_target(target_model, vendor=vendor)
    if pair is None:
        return (
            "Do NOT enable speculative decoding — no draft model is registered "
            f"for {target_model}."
        )
    return (
        f"Speculative decoding allowed with draft={pair.draft_model}, "
        f"num_speculative_tokens≈{pair.num_speculative_tokens}. {pair.notes}"
    )
