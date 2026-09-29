"""OceanTune fusion campaign helpers."""

from core.fusion.diagnose import (
    FusionDiagnosis,
    category_shares_from_trace,
    diagnose_fusion,
)
from core.fusion.patterns import (
    FUSION_PATTERNS,
    FusionPattern,
    get_pattern,
    match_patterns,
    patterns_prompt_block,
)

__all__ = [
    "FUSION_PATTERNS",
    "FusionDiagnosis",
    "FusionPattern",
    "category_shares_from_trace",
    "diagnose_fusion",
    "get_pattern",
    "match_patterns",
    "patterns_prompt_block",
]
