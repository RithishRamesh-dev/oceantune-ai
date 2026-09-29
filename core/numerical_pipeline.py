"""
core/numerical_pipeline.py
--------------------------
Multi-stage numerical validation (SNR → repeatability → KEEP).

Mirrors Hyperloom's staged correctness pipeline without vendoring it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from core.snr_contract import (
    DEFAULT_SNR_THRESHOLD_DB,
    KeepEval,
    SNRResult,
    evaluate_keep,
    validate_snr,
)


@dataclass
class PipelineStageResult:
    stage: str
    passed: bool
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NumericalPipelineReport:
    passed: bool
    stages: List[PipelineStageResult] = field(default_factory=list)
    keep: Optional[KeepEval] = None
    snr: Optional[SNRResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "stages": [s.to_dict() for s in self.stages],
            "keep": self.keep.to_dict() if self.keep else None,
            "snr": self.snr.to_dict() if self.snr else None,
        }


def run_numerical_pipeline(
    reference: Sequence[float],
    candidate: Sequence[float],
    *,
    speedups: Optional[Sequence[float]] = None,
    snr_threshold_db: float = DEFAULT_SNR_THRESHOLD_DB,
    repeat_refs: Optional[List[Sequence[float]]] = None,
    max_repeat_rel_err: float = 1e-3,
) -> NumericalPipelineReport:
    """
    Stage 1: SNR vs reference
    Stage 2: optional repeatability (candidate vs repeated refs)
    Stage 3: statistical KEEP on speedups (if provided)
    """
    stages: List[PipelineStageResult] = []
    snr = validate_snr(reference, candidate, threshold_db=snr_threshold_db)
    stages.append(PipelineStageResult(
        stage="snr",
        passed=snr.passed,
        detail=snr.to_dict(),
    ))
    if not snr.passed:
        return NumericalPipelineReport(passed=False, stages=stages, snr=snr)

    if repeat_refs:
        ok = True
        max_err = 0.0
        for rep in repeat_refs:
            if len(rep) != len(candidate):
                ok = False
                break
            for a, b in zip(candidate, rep):
                denom = max(abs(float(a)), 1e-12)
                err = abs(float(a) - float(b)) / denom
                max_err = max(max_err, err)
                if err > max_repeat_rel_err:
                    ok = False
        stages.append(PipelineStageResult(
            stage="repeatability",
            passed=ok,
            detail={"max_rel_err": max_err, "threshold": max_repeat_rel_err},
        ))
        if not ok:
            return NumericalPipelineReport(passed=False, stages=stages, snr=snr)

    keep = None
    if speedups is not None:
        keep = evaluate_keep(list(speedups), snr=snr)
        stages.append(PipelineStageResult(
            stage="keep",
            passed=keep.keep,
            detail=keep.to_dict(),
        ))
        return NumericalPipelineReport(
            passed=keep.keep, stages=stages, snr=snr, keep=keep,
        )

    return NumericalPipelineReport(passed=True, stages=stages, snr=snr)
