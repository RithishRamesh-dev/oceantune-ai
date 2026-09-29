"""Tests for CriticAgent (no LLM)."""

import asyncio

from agents.critic_agent import CriticAgent
from core.measurement_gate import GateDecision


def _run(coro):
    return asyncio.run(coro)


def test_critic_rejects_when_gate_reverts():
    gate = GateDecision(
        keep=False,
        reason="no_improvement",
        incumbent_fitness=1.0,
        candidate_fitness=0.9,
        delta=-0.1,
        delta_pct=-10.0,
    )
    verdict = _run(CriticAgent().review(gate=gate, label="t"))
    assert verdict.accept is False
    assert "measurement_gate" in verdict.reason


def test_critic_accepts_keep_without_llm():
    gate = GateDecision(
        keep=True,
        reason="improved",
        incumbent_fitness=1.0,
        candidate_fitness=1.1,
        delta=0.1,
        delta_pct=10.0,
    )
    verdict = _run(CriticAgent().review(gate=gate, label="t"))
    assert verdict.accept is True
    assert verdict.used_llm is False


def test_critic_hard_reject_mla_block_size():
    gate = GateDecision(
        keep=True,
        reason="improved",
        incumbent_fitness=1.0,
        candidate_fitness=1.2,
        delta=0.2,
        delta_pct=20.0,
    )
    verdict = _run(
        CriticAgent().review(
            gate=gate,
            candidate_flags={"block_size": 16},
            constraints_block="MLA models require block_size=1",
            label="t",
        )
    )
    assert verdict.accept is False
    assert "block_size" in verdict.reason
