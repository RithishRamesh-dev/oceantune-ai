"""
agents/critic_agent.py
----------------------
Critic Agent — mission-aware KEEP/REVERT review (Hyperloom Critic analog).

Does NOT replace MeasurementGate. The gate decides from fitness numbers;
the Critic reviews whether a *kept* change still serves the optimization
mission (e.g. latency-sensitive workload accepting a throughput-only win
that destroys TTFT).

Falls back to accept-measured-keep when LLM unavailable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from agents.do_client import DOClient, DOClientError
from core.measurement_gate import GateDecision

log = logging.getLogger("agents.critic_agent")

_CRITIC_SYSTEM = """\
You are the OceanTune Critic. A MeasurementGate has already decided KEEP or REVERT
based on measured fitness. Your job is to review KEEP candidates only:

- Reject a KEEP if it violates hard constraints (OOM risk, MLA block_size, unsupported backend).
- Reject a KEEP if primary_metric is latency/ttft/tpot and the change clearly hurts that axis
  while only improving throughput slightly.
- Accept otherwise. Prefer accepting measured improvements.

Respond with JSON only:
{
  "verdict": "accept" | "reject",
  "reason": "<one sentence>",
  "mission_aligned": true | false
}
"""


@dataclass
class CriticVerdict:
    accept: bool
    reason: str
    mission_aligned: bool = True
    used_llm: bool = False
    gate: Optional[GateDecision] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accept": self.accept,
            "reason": self.reason,
            "mission_aligned": self.mission_aligned,
            "used_llm": self.used_llm,
            "gate": self.gate.to_dict() if self.gate else None,
        }


class CriticAgent:
    """Reviews MeasurementGate KEEP decisions against mission context."""

    def __init__(self, do_client: Optional[DOClient] = None) -> None:
        self._client = do_client

    async def review(
        self,
        *,
        gate: GateDecision,
        primary_metric: str = "throughput",
        candidate_flags: Optional[Dict[str, Any]] = None,
        incumbent_flags: Optional[Dict[str, Any]] = None,
        candidate_metrics: Optional[Dict[str, Any]] = None,
        constraints_block: str = "",
        label: str = "",
    ) -> CriticVerdict:
        """
        If gate.keep is False → reject immediately (measurement already said no).
        If gate.keep is True → optional LLM mission review; default accept.
        """
        if not gate.keep:
            return CriticVerdict(
                accept=False,
                reason=f"measurement_gate:{gate.reason}",
                mission_aligned=False,
                used_llm=False,
                gate=gate,
            )

        # Hard constraint heuristics (no LLM)
        hard = self._hard_reject(
            candidate_flags or {},
            incumbent_flags or {},
            constraints_block,
        )
        if hard:
            return CriticVerdict(
                accept=False,
                reason=hard,
                mission_aligned=False,
                used_llm=False,
                gate=gate,
            )

        if self._client is None:
            return CriticVerdict(
                accept=True,
                reason="accepted_measured_keep_no_llm",
                mission_aligned=True,
                used_llm=False,
                gate=gate,
            )

        user_msg = (
            f"Label: {label}\n"
            f"Primary metric: {primary_metric}\n"
            f"Gate: keep={gate.keep} reason={gate.reason} "
            f"fitness {gate.incumbent_fitness:.4f} → {gate.candidate_fitness:.4f} "
            f"(Δ={gate.delta:+.4f})\n"
            f"Incumbent flags: {json.dumps(incumbent_flags or {}, default=str)[:1500]}\n"
            f"Candidate flags: {json.dumps(candidate_flags or {}, default=str)[:1500]}\n"
            f"Candidate metrics: {json.dumps(candidate_metrics or {}, default=str)[:1000]}\n"
            f"{constraints_block}\n"
            "Review this KEEP."
        )
        try:
            raw = await self._client.chat(
                messages=[{"role": "user", "content": user_msg}],
                system=_CRITIC_SYSTEM,
                json_mode=True,
            )
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            verdict = str((parsed or {}).get("verdict", "accept")).lower()
            reason = str((parsed or {}).get("reason") or "critic_review")
            aligned = bool((parsed or {}).get("mission_aligned", True))
            accept = verdict == "accept"
            log.info(
                "Critic[%s]: accept=%s reason=%s",
                label, accept, reason[:120],
            )
            return CriticVerdict(
                accept=accept,
                reason=reason,
                mission_aligned=aligned,
                used_llm=True,
                gate=gate,
            )
        except (DOClientError, Exception) as exc:
            log.warning("Critic LLM unavailable (%s); accepting measured KEEP", exc)
            return CriticVerdict(
                accept=True,
                reason="accepted_measured_keep_llm_fallback",
                mission_aligned=True,
                used_llm=False,
                gate=gate,
            )

    @staticmethod
    def _hard_reject(
        candidate: Dict[str, Any],
        incumbent: Dict[str, Any],
        constraints_block: str,
    ) -> str:
        """Return reject reason or empty string."""
        # MLA block_size guard
        if candidate.get("block_size") not in (None, 1) and "mla" in constraints_block.lower():
            if int(candidate.get("block_size") or 0) > 1:
                return "hard_reject:block_size>1_with_mla_constraint"
        # Constraint text explicitly mentions avoiding this util
        util = candidate.get("gpu_memory_utilization")
        if util is not None and constraints_block:
            marker = f"gpu_memory_utilization >= {util}"
            if marker in constraints_block:
                return f"hard_reject:oom_constraint_at_util_{util}"
        return ""
