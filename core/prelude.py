"""
core/prelude.py
---------------
PRELUDE planning: warm-recipe replay gate + reject-ledger seed + profile arm.

Runs *before* Stage 1 search (after optional enablement). Does not vendor
Hyperloom; OceanTune-native recipe confidence + measurement rules decide whether
a prior winner is replayed as the Stage 1 seed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from core.warmstart_policy import WarmstartDecision, decide_warmstart


@dataclass
class PreludePlan:
    """What the controller should do in PRELUDE."""

    replay_flags: Dict[str, Any] = field(default_factory=dict)
    replay_fitness_claimed: float = 0.0
    replay_confidence: float = 0.0
    should_benchmark_replay: bool = False
    reject_fingerprints: List[str] = field(default_factory=list)
    reject_pitfalls: List[str] = field(default_factory=list)
    profile_arm: bool = False
    warmstart: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreludeResult:
    """Outcome after optional warm-replay measurement."""

    plan: PreludePlan
    replay_measured_fitness: float = 0.0
    replay_accepted: bool = False
    seed_flags: Dict[str, Any] = field(default_factory=dict)
    stop_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["plan"] = self.plan.to_dict()
        return d


def build_prelude_plan(
    *,
    recipe: Any = None,
    enable_profile_arm: bool = False,
    min_confidence: float = 0.7,
    min_fitness: float = 0.0,
    max_warm_trials: int = 3,
) -> PreludePlan:
    """
    Build a prelude plan from a Recipe (or None).

    ``recipe`` may be a Recipe dataclass or a dict-like with flags/confidence.
    """
    plan = PreludePlan(profile_arm=bool(enable_profile_arm))
    if recipe is None:
        plan.notes.append("no_recipe")
        return plan

    if hasattr(recipe, "best_flags"):
        flags = dict(getattr(recipe, "best_flags", None) or {})
        fitness = float(getattr(recipe, "best_fitness", 0.0) or 0.0)
        confidence = float(getattr(recipe, "confidence", 0.0) or 0.0)
        what_failed = list(getattr(recipe, "what_failed", None) or [])
        pitfalls = list(getattr(recipe, "pitfalls", None) or [])
        fingerprint = str(getattr(recipe, "best_fingerprint", "") or "")
    else:
        flags = dict(recipe.get("best_flags") or recipe.get("flags") or {})
        fitness = float(recipe.get("best_fitness") or 0.0)
        confidence = float(recipe.get("confidence") or 0.0)
        what_failed = list(recipe.get("what_failed") or [])
        pitfalls = list(recipe.get("pitfalls") or [])
        fingerprint = str(recipe.get("best_fingerprint") or "")

    decision: WarmstartDecision = decide_warmstart(
        flags=flags,
        claimed_fitness=fitness,
        confidence=confidence,
        min_confidence=min_confidence,
        min_fitness=min_fitness,
        max_warm_trials=max_warm_trials,
        trials_already=0,
    )
    plan.warmstart = decision.to_dict()
    plan.replay_flags = dict(decision.flags) if decision.accept else {}
    plan.replay_fitness_claimed = fitness
    plan.replay_confidence = confidence
    plan.should_benchmark_replay = bool(decision.accept and plan.replay_flags)
    if decision.accept:
        plan.notes.append("warm_replay_armed")
    else:
        plan.notes.append(f"warm_replay_skipped:{decision.reason}")

    # Seed reject ledger from prior failures
    for item in what_failed:
        if isinstance(item, dict):
            fp = str(item.get("fingerprint") or item.get("flags_fingerprint") or "")
            if fp:
                plan.reject_fingerprints.append(fp)
            desc = str(item.get("error") or item.get("reason") or item.get("description") or "")
            if desc:
                plan.reject_pitfalls.append(desc[:300])
        elif isinstance(item, str) and item:
            plan.reject_pitfalls.append(item[:300])

    for p in pitfalls:
        if hasattr(p, "description"):
            plan.reject_pitfalls.append(str(p.description)[:300])
        elif isinstance(p, dict):
            plan.reject_pitfalls.append(str(p.get("description") or "")[:300])
        elif isinstance(p, str):
            plan.reject_pitfalls.append(p[:300])

    if fingerprint and fingerprint not in plan.reject_fingerprints:
        # Do not reject the winning fingerprint itself
        pass

    # Dedupe
    plan.reject_fingerprints = list(dict.fromkeys(plan.reject_fingerprints))
    plan.reject_pitfalls = list(dict.fromkeys(plan.reject_pitfalls))[:20]
    return plan


def accept_warm_replay(
    *,
    plan: PreludePlan,
    measured_fitness: float,
    min_retention_ratio: float = 0.85,
) -> PreludeResult:
    """
    Accept replay seed only if measured fitness retains enough of the claim.
    """
    result = PreludeResult(plan=plan, replay_measured_fitness=float(measured_fitness or 0.0))
    if not plan.should_benchmark_replay or not plan.replay_flags:
        result.stop_reason = "no_replay_armed"
        return result
    claimed = float(plan.replay_fitness_claimed or 0.0)
    measured = float(measured_fitness or 0.0)
    if measured <= 0:
        result.stop_reason = "replay_failed"
        return result
    if claimed > 0 and measured < claimed * min_retention_ratio:
        result.stop_reason = "replay_below_retention"
        return result
    result.replay_accepted = True
    result.seed_flags = dict(plan.replay_flags)
    result.stop_reason = "replay_accepted"
    return result
