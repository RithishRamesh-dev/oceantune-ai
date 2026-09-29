"""
core/experience_constraints.py
------------------------------
Distill failed configs and recipe pitfalls into planner negative priors.

Inspired by Hyperloom experience distillation (error signature → constraint),
implemented OceanTune-natively for Stage 1 Planner and Stage 2 Strategy prompts.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

log = logging.getLogger("core.experience_constraints")

# Error signature → reusable planning constraint (Hyperloom-style rules, NVIDIA+AMD).
_CONSTRAINT_RULES: List[tuple] = [
    (
        re.compile(r"out of memory|oom|cuda.?oom|hip.?oom", re.I),
        "Avoid raising gpu_memory_utilization above the failing value; prefer "
        "kv_cache_dtype=fp8 or lower max_num_seqs before pushing VRAM further.",
    ),
    (
        re.compile(r"startup.?timeout|health.?check|failed to start", re.I),
        "Do not combine aggressive TP/EP changes with untested quantization in one "
        "step; change one parallelism or load flag at a time.",
    ),
    (
        re.compile(r"flashinfer|FLASHINFER", re.I),
        "FLASHINFER may be unavailable or slower on this stack — prefer FLASH_ATTN "
        "or ROCM_FLASH unless CapabilityDetector lists FLASHINFER as supported.",
    ),
    (
        re.compile(r"nccl|rccl|all.?reduce|distributed", re.I),
        "Communication failures: keep tensor_parallel_size within NCCL/RCCL-stable "
        "values; do not increase TP while also enabling experimental all2all backends.",
    ),
    (
        re.compile(r"block.?size|mla|latent", re.I),
        "MLA models require block_size=1 — never propose block_size>1 for MLA architectures.",
    ),
    (
        re.compile(r"speculative|draft.?model", re.I),
        "Do not enable speculative decoding without a known compatible draft model.",
    ),
]


@dataclass
class ExperienceConstraint:
    rule: str
    source: str  # oom | recipe_pitfall | fingerprint | signature
    severity: str = "medium"

    def to_prompt_line(self) -> str:
        return f"- [{self.severity}] {self.rule}"


def constraints_from_error(error: str, *, source: str = "signature") -> List[ExperienceConstraint]:
    """Map one error string to zero or more constraints."""
    if not error:
        return []
    out: List[ExperienceConstraint] = []
    for pattern, rule in _CONSTRAINT_RULES:
        if pattern.search(error):
            sev = "high" if "oom" in rule.lower() or "memory" in rule.lower() else "medium"
            out.append(ExperienceConstraint(rule=rule, source=source, severity=sev))
    return out


def constraints_from_failed_configs(
    failed: List[Dict[str, Any]],
) -> List[ExperienceConstraint]:
    """Build constraints from failed config documents."""
    seen: Set[str] = set()
    out: List[ExperienceConstraint] = []
    for doc in failed:
        err = str(doc.get("error") or "")
        for c in constraints_from_error(err, source="failed_config"):
            if c.rule not in seen:
                seen.add(c.rule)
                out.append(c)
        flags = doc.get("flags") or {}
        util = flags.get("gpu_memory_utilization")
        if util is not None and ("oom" in err.lower() or "memory" in err.lower()):
            rule = (
                f"Do not set gpu_memory_utilization >= {util} — prior run OOMed at that value."
            )
            if rule not in seen:
                seen.add(rule)
                out.append(
                    ExperienceConstraint(rule=rule, source="oom", severity="high")
                )
    return out


def constraints_from_recipe_pitfalls(
    pitfalls: List[Any],
) -> List[ExperienceConstraint]:
    """Convert Recipe.pitfalls into planner constraints."""
    out: List[ExperienceConstraint] = []
    for pit in pitfalls or []:
        desc = pit.description if hasattr(pit, "description") else str(
            (pit or {}).get("description") if isinstance(pit, dict) else pit
        )
        sev = (
            pit.severity
            if hasattr(pit, "severity")
            else (pit.get("severity") if isinstance(pit, dict) else "medium")
        )
        if not desc:
            continue
        # Prefer signature rules when they match; else use raw pitfall text.
        matched = constraints_from_error(desc, source="recipe_pitfall")
        if matched:
            out.extend(matched)
        else:
            out.append(
                ExperienceConstraint(
                    rule=f"Prior session pitfall: {desc[:300]}",
                    source="recipe_pitfall",
                    severity=str(sev or "medium"),
                )
            )
    return out


def render_constraints_block(
    constraints: List[ExperienceConstraint],
    *,
    max_items: int = 12,
) -> str:
    """Prompt section for Planner / StrategyOptimizer."""
    if not constraints:
        return ""
    # Deduplicate by rule text, prefer high severity
    by_rule: Dict[str, ExperienceConstraint] = {}
    for c in constraints:
        prev = by_rule.get(c.rule)
        if prev is None or (c.severity == "high" and prev.severity != "high"):
            by_rule[c.rule] = c
    ordered = sorted(
        by_rule.values(),
        key=lambda x: (0 if x.severity == "high" else 1, x.rule),
    )[:max_items]
    lines = ["=== Experience constraints (do NOT repeat these failures) ==="]
    lines.extend(c.to_prompt_line() for c in ordered)
    return "\n".join(lines)


async def collect_session_constraints(
    db: Any,
    *,
    session_id: str,
    recipe: Any = None,
) -> List[ExperienceConstraint]:
    """Gather constraints from current-session failures + recipe pitfalls."""
    constraints: List[ExperienceConstraint] = []
    try:
        failed = await db.list_failed_configs(session_id, limit=30)
        constraints.extend(constraints_from_failed_configs(failed))
    except Exception as exc:
        log.debug("Failed-config constraint collect skipped: %s", exc)
    if recipe is not None and getattr(recipe, "pitfalls", None):
        constraints.extend(constraints_from_recipe_pitfalls(recipe.pitfalls))
    return constraints
