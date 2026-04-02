"""
core/experiment_runner.py
--------------------------
Stub — implemented fully in Step 3.
Responsible for starting / stopping the vLLM server process and
managing experiment execution.
"""

from core.logger import get_logger

log = get_logger(__name__)


def run_experiment(config, vllm_flags: dict) -> dict:
    """
    Start vLLM with the given flags, run benchmarks, return metrics.
    Fully implemented in Step 3.
    """
    log.warning("experiment_runner: stub — not yet implemented")
    return {}
