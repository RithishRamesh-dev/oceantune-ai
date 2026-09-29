"""
core/serving_patches.py
-----------------------
Versioned serving-patch registry with env-gated apply.

Hyperloom ships version-dir patches under data/serving_patches/. OceanTune
keeps a native registry of manifests (JSON/YAML) that describe env gates and
shadow hook ids — actual diffs are optional files under data/serving_patches/.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("core.serving_patches")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATCH_ROOT = REPO_ROOT / "data" / "serving_patches"


@dataclass
class PatchManifest:
    patch_id: str
    framework: str
    min_version: str = "0.0.0"
    max_version: str = "999.0.0"
    env_flag: str = ""
    hook_ids: List[str] = field(default_factory=list)
    relative_path: str = ""
    description: str = ""
    enabled_by_default: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PatchSelection:
    selected: List[PatchManifest] = field(default_factory=list)
    skipped: List[Dict[str, str]] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected": [p.to_dict() for p in self.selected],
            "skipped": list(self.skipped),
            "env": dict(self.env),
        }


def _parse_version(v: str) -> tuple:
    parts = re.findall(r"\d+", v or "0")
    nums = tuple(int(x) for x in parts[:3])
    while len(nums) < 3:
        nums = nums + (0,)
    return nums


def version_in_range(version: str, min_v: str, max_v: str) -> bool:
    v = _parse_version(version)
    return _parse_version(min_v) <= v <= _parse_version(max_v)


def load_manifests(root: Optional[Path] = None) -> List[PatchManifest]:
    root = root or DEFAULT_PATCH_ROOT
    manifests: List[PatchManifest] = []
    if not root.exists():
        return manifests
    for path in sorted(root.rglob("manifest.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Bad patch manifest %s: %s", path, exc)
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            manifests.append(PatchManifest(
                patch_id=str(item.get("patch_id") or path.parent.name),
                framework=str(item.get("framework") or "vllm").lower(),
                min_version=str(item.get("min_version") or "0.0.0"),
                max_version=str(item.get("max_version") or "999.0.0"),
                env_flag=str(item.get("env_flag") or ""),
                hook_ids=list(item.get("hook_ids") or []),
                relative_path=str(path.parent.relative_to(root)),
                description=str(item.get("description") or ""),
                enabled_by_default=bool(item.get("enabled_by_default", False)),
            ))
    return manifests


def select_patches(
    *,
    framework: str = "vllm",
    framework_version: str = "0.6.0",
    root: Optional[Path] = None,
    enable_env_prefix: str = "OCEANTUNE_PATCH_",
) -> PatchSelection:
    """
    Select manifests matching framework + version. Env gate:
      OCEANTUNE_PATCH_<PATCH_ID>=1  or  enabled_by_default
    """
    sel = PatchSelection()
    fw = (framework or "vllm").lower()
    for m in load_manifests(root):
        if m.framework != fw:
            sel.skipped.append({"patch_id": m.patch_id, "reason": "framework_mismatch"})
            continue
        if not version_in_range(framework_version, m.min_version, m.max_version):
            sel.skipped.append({"patch_id": m.patch_id, "reason": "version_out_of_range"})
            continue
        env_key = m.env_flag or f"{enable_env_prefix}{m.patch_id.upper().replace('-', '_')}"
        enabled = m.enabled_by_default or os.environ.get(env_key, "") in ("1", "true", "True")
        if not enabled:
            sel.skipped.append({"patch_id": m.patch_id, "reason": "env_gate_off"})
            continue
        sel.selected.append(m)
        if env_key:
            sel.env[env_key] = "1"
        for hid in m.hook_ids:
            # Align with ShadowHookRegistry env flags when possible
            sel.env[f"OCEANTUNE_HOOK_{hid.upper()}"] = "1"
    return sel


def apply_verification_marker(hook_id: str) -> str:
    """String the shadow sitecustomize should log so E2E can prove the hook fired."""
    return f"OCEANTUNE_SHADOW_HOOK_ACTIVE:{hook_id}"
