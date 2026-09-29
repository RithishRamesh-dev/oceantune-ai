"""
core/search_policy.py
---------------------
EXPLOIT vs DIVERSIFY search mode for Stage 1–2 proposal loops.

When fitness stalls, widen mutation radius / force new flag families instead of
repeating the same LLM proposals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence


MODE_EXPLOIT = "exploit"
MODE_DIVERSIFY = "diversify"


@dataclass
class SearchPolicyState:
    mode: str = MODE_EXPLOIT
    stall_count: int = 0
    last_fitness: float = 0.0
    history: List[float] = field(default_factory=list)
    forced_families: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchPolicyDecision:
    mode: str
    reason: str
    mutation_radius: float = 0.2
    force_new_families: bool = False
    suggested_families: List[str] = field(default_factory=list)
    prompt_hint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Flag families used when diversifying
_DIVERSIFY_FAMILIES = (
    "attention_backend",
    "kv_cache_dtype",
    "enable_prefix_caching",
    "enable_chunked_prefill",
    "speculative_model",
    "gpu_memory_utilization",
    "max_num_batched_tokens",
    "block_size",
    "scheduler_delay_factor",
    "num_scheduler_steps",
)


def update_search_policy(
    state: SearchPolicyState,
    *,
    current_fitness: float,
    stall_epsilon_pct: float = 0.5,
    stall_limit: int = 2,
    exploit_radius: float = 0.15,
    diversify_radius: float = 0.45,
) -> SearchPolicyDecision:
    """
    Update stall counter and return the mode for the next proposal round.
    """
    cur = float(current_fitness or 0.0)
    prev = float(state.last_fitness or 0.0)
    state.history.append(cur)

    if prev > 0:
        gain_pct = (cur - prev) / prev * 100.0
    else:
        gain_pct = 100.0 if cur > 0 else 0.0

    if gain_pct < stall_epsilon_pct:
        state.stall_count += 1
    else:
        state.stall_count = 0

    state.last_fitness = cur

    if state.stall_count >= stall_limit:
        state.mode = MODE_DIVERSIFY
        # Rotate families based on stall count
        start = (state.stall_count - stall_limit) % len(_DIVERSIFY_FAMILIES)
        families = list(_DIVERSIFY_FAMILIES[start:] + _DIVERSIFY_FAMILIES[:start])[:4]
        state.forced_families = families
        return SearchPolicyDecision(
            mode=MODE_DIVERSIFY,
            reason=f"stalled_{state.stall_count}_rounds",
            mutation_radius=diversify_radius,
            force_new_families=True,
            suggested_families=families,
            prompt_hint=(
                "Search has stalled. DIVERSIFY: try unused flag families "
                f"{families}. Avoid small perturbations of the incumbent."
            ),
        )

    state.mode = MODE_EXPLOIT
    return SearchPolicyDecision(
        mode=MODE_EXPLOIT,
        reason="improving_or_early",
        mutation_radius=exploit_radius,
        force_new_families=False,
        prompt_hint="EXPLOIT: refine around the current winner with small changes.",
    )


def prompt_addon(decision: SearchPolicyDecision) -> str:
    return decision.prompt_hint or ""
