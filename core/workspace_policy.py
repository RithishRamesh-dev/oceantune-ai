"""
core/workspace_policy.py
------------------------
Lightweight Stage 4 workspace isolation (Hyperloom workspace_policy analog).

Keeps generated kernels and drivers under a session-scoped directory with
allow/deny path checks — without requiring a full git worktree forge.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Set

log = logging.getLogger("core.workspace_policy")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WORK_ROOT = _REPO_ROOT / "storage" / "workspaces"

# Paths relative to workspace that Stage 4 may write
_ALLOWED_WRITE_PREFIXES = (
    "kernels/",
    "drivers/",
    "artifacts/",
    "ledger/",
    "tmp/",
)

# Never write outside workspace into these repo paths
_DENIED_REPO_PREFIXES = (
    "agents/",
    "core/",
    "configs/",
    "Hyperloom-main/",
    ".git/",
)


@dataclass
class Workspace:
    session_id: str
    root: Path
    allowed_writes: List[str] = field(default_factory=lambda: list(_ALLOWED_WRITE_PREFIXES))

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def ensure(self) -> None:
        for sub in ("kernels", "drivers", "artifacts", "ledger", "tmp"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)


class WorkspacePolicyError(PermissionError):
    pass


def open_workspace(session_id: str, *, root: Optional[Path] = None) -> Workspace:
    base = root or _WORK_ROOT
    ws = Workspace(session_id=session_id[:24], root=base / session_id[:24])
    ws.ensure()
    log.info("Workspace ready: %s", ws.root)
    return ws


def assert_writable(ws: Workspace, rel_path: str) -> Path:
    """Raise if rel_path escapes workspace or hits a denied prefix."""
    rel = rel_path.lstrip("/").replace("\\", "/")
    # Absolute escape check
    target = (ws.root / rel).resolve()
    if not str(target).startswith(str(ws.root.resolve())):
        raise WorkspacePolicyError(f"path_escapes_workspace:{rel}")
    if not any(rel.startswith(p) for p in ws.allowed_writes):
        raise WorkspacePolicyError(f"write_prefix_denied:{rel}")
    # Also refuse writing into protected repo trees if somehow mapped
    for denied in _DENIED_REPO_PREFIXES:
        try:
            if denied in str(target.relative_to(_REPO_ROOT.resolve())):
                raise WorkspacePolicyError(f"denied_repo_path:{denied}")
        except ValueError:
            pass
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def stage_kernel_file(ws: Workspace, src: Path, *, name: str = "kernel.py") -> Path:
    dest = assert_writable(ws, f"kernels/{name}")
    shutil.copy2(src, dest)
    return dest


def cleanup_workspace(ws: Workspace, *, keep: bool = True) -> None:
    if keep:
        return
    shutil.rmtree(ws.root, ignore_errors=True)
