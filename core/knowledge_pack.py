"""
core/knowledge_pack.py
----------------------
Load NVIDIA/AMD vendor knowledge packs into planner / strategy / kernel prompts.

Packs live under ``knowledge/`` (OceanTune-owned markdown). No Hyperloom
``local_knowledge`` files are vendored.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("core.knowledge_pack")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_KNOWLEDGE_ROOT = _REPO_ROOT / "knowledge"

_AMD_GPUS = frozenset({"MI300X", "MI325X", "MI350X", "MI355X"})


def vendor_for_gpu(gpu_type: str) -> str:
    return "amd" if gpu_type in _AMD_GPUS else "nvidia"


def knowledge_root() -> Path:
    return _KNOWLEDGE_ROOT


def list_pack_files(vendor: str, *, max_files: int = 12) -> List[Path]:
    """Return markdown files for vendor + common, INDEX first."""
    roots = [
        _KNOWLEDGE_ROOT / "common",
        _KNOWLEDGE_ROOT / vendor,
    ]
    files: List[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        index = root / "INDEX.md"
        if index.is_file():
            files.append(index)
        for p in sorted(root.rglob("*.md")):
            if p.name == "INDEX.md":
                continue
            files.append(p)
    return files[:max_files]


def load_knowledge_block(
    *,
    gpu_type: str,
    max_chars: int = 6000,
    focus_keywords: Optional[List[str]] = None,
) -> str:
    """
    Render a prompt section from knowledge packs.

    Prefer files whose path/name match ``focus_keywords`` when provided.
    """
    vendor = vendor_for_gpu(gpu_type)
    files = list_pack_files(vendor)
    if not files:
        return ""

    focus = [k.lower() for k in (focus_keywords or [])]
    if focus:
        scored = []
        for p in files:
            key = str(p).lower()
            score = sum(1 for k in focus if k in key)
            scored.append((score, p))
        scored.sort(key=lambda x: (-x[0], str(x[1])))
        files = [p for _, p in scored]

    chunks: List[str] = [
        f"=== Vendor knowledge pack ({vendor} / {gpu_type}) ===",
        "Use these as methodology levers — verify with MeasurementGate.",
    ]
    used = 0
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        rel = path.relative_to(_KNOWLEDGE_ROOT)
        header = f"\n--- {rel} ---\n"
        piece = header + text.strip()
        if used + len(piece) > max_chars:
            remain = max_chars - used - len(header) - 20
            if remain < 200:
                break
            piece = header + text.strip()[:remain] + "\n...(truncated)\n"
            chunks.append(piece)
            break
        chunks.append(piece)
        used += len(piece)

    block = "\n".join(chunks)
    log.debug("Knowledge pack loaded: vendor=%s chars=%d", vendor, len(block))
    return block
