"""
core/report_generator.py
------------------------
Report Generator — produces human-readable artefacts from a completed session.

Given the Analyst's AnalysisResult and the winning kernel config, this module
emits three artefacts:
  1. A YAML recipe file (ready to paste into oceantune.yaml or a CI pipeline).
  2. A shell script (docker run / vllm serve command with all flags).
  3. A Markdown summary report (human-readable with tables and LLM commentary).

Usage
-----
    gen = ReportGenerator(output_dir=Path("storage/results"))
    paths = gen.generate(
        analysis=analyst_result,
        best_kernel_config=kernel_cfg,
        model_id="deepseek-ai/DeepSeek-V3.2",
        gpu_type="H100",
        session_id="...",
    )
    print(paths)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from agents.analyst import AnalysisResult
from core.search_space import VLLMFlags

_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "storage" / "results"


class ReportGenerator:
    """
    Generates YAML recipe, shell script, and Markdown report.

    Parameters
    ----------
    output_dir : Path
        Directory where artefacts are written.  Created if it does not exist.
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self._output_dir = output_dir or _DEFAULT_OUTPUT_DIR
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        *,
        analysis: AnalysisResult,
        best_kernel_config: Dict[str, Any],
        model_id: str,
        gpu_type: str,
        session_id: str,
        docker_image: str = "vllm/vllm-openai:latest",
    ) -> Dict[str, Path]:
        """
        Write all report artefacts and return a dict of {type: path}.

        Returns
        -------
        dict with keys: "yaml", "shell", "markdown"
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        slug = f"{gpu_type}_{model_id.replace('/', '_').replace('-', '_')}_{ts}"

        yaml_path = self._write_yaml_recipe(
            slug=slug,
            analysis=analysis,
            best_kernel_config=best_kernel_config,
            model_id=model_id,
            gpu_type=gpu_type,
            session_id=session_id,
        )
        shell_path = self._write_shell_script(
            slug=slug,
            analysis=analysis,
            best_kernel_config=best_kernel_config,
            model_id=model_id,
            gpu_type=gpu_type,
            docker_image=docker_image,
        )
        md_path = self._write_markdown_report(
            slug=slug,
            analysis=analysis,
            best_kernel_config=best_kernel_config,
            model_id=model_id,
            gpu_type=gpu_type,
            session_id=session_id,
        )

        return {"yaml": yaml_path, "shell": shell_path, "markdown": md_path}

    # ------------------------------------------------------------------
    # YAML recipe
    # ------------------------------------------------------------------

    def _write_yaml_recipe(
        self,
        *,
        slug: str,
        analysis: AnalysisResult,
        best_kernel_config: Dict[str, Any],
        model_id: str,
        gpu_type: str,
        session_id: str,
    ) -> Path:
        merged_flags = {**analysis.winner_flags, **best_kernel_config}
        recipe = {
            "# OceanTune AI — Optimised Recipe": None,
            "session_id": session_id,
            "model_id": model_id,
            "gpu_type": gpu_type,
            "fitness_score": analysis.winner_fitness,
            "fingerprint": analysis.winner_fingerprint,
            "key_flags": analysis.key_flags,
            "vllm_flags": merged_flags,
            "kernel_config": best_kernel_config,
            "analyst_explanation": analysis.explanation,
            "recommendation": analysis.recommendation,
        }
        path = self._output_dir / f"recipe_{slug}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(recipe, f, default_flow_style=False, allow_unicode=True)
        return path

    # ------------------------------------------------------------------
    # Shell script
    # ------------------------------------------------------------------

    def _write_shell_script(
        self,
        *,
        slug: str,
        analysis: AnalysisResult,
        best_kernel_config: Dict[str, Any],
        model_id: str,
        gpu_type: str,
        docker_image: str,
    ) -> Path:
        flags = analysis.winner_flags

        # Rebuild a VLLMFlags object to get the canonical CLI args
        try:
            vf = VLLMFlags(**{
                k: v for k, v in flags.items()
                if k in VLLMFlags.__dataclass_fields__
            })
            cli_args = vf.to_vllm_args(model_id=model_id, gpu_type=gpu_type)
        except Exception:
            cli_args = []

        # Collect env vars from kernel_config (env_var entries)
        env_lines: List[str] = []
        for name, val in best_kernel_config.items():
            # Try to look up the env_var name from kernel search space
            env_var = _kernel_env_var(name)
            if env_var:
                v = str(val).lower() if isinstance(val, bool) else str(val)
                env_lines.append(f"  -e {env_var}={v} \\")

        args_str = " \\\n  ".join(cli_args)
        env_str = "\n".join(env_lines)

        script = (
            "#!/usr/bin/env bash\n"
            "# OceanTune AI — Optimised vLLM launch script\n"
            f"# Model  : {model_id}\n"
            f"# GPU    : {gpu_type}\n"
            f"# Fitness: {analysis.winner_fitness:.4f}\n"
            "#\n"
            "# Usage: bash this_script.sh\n"
            "\n"
            f"MODEL={model_id}\n"
            f"IMAGE={docker_image}\n"
            "\n"
            "docker run --gpus all --ipc=host \\\n"
            f"{env_str}\n"
            "  -p 8000:8000 \\\n"
            '  "$IMAGE" \\\n'
            f"  {args_str}\n"
        )

        path = self._output_dir / f"launch_{slug}.sh"
        with open(path, "w", encoding="utf-8") as f:
            f.write(script)
        os.chmod(path, 0o755)
        return path

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------

    def _write_markdown_report(
        self,
        *,
        slug: str,
        analysis: AnalysisResult,
        best_kernel_config: Dict[str, Any],
        model_id: str,
        gpu_type: str,
        session_id: str,
    ) -> Path:
        top = analysis.top_configs[:5]

        top_table = "| Rank | Fingerprint | Fitness | Throughput (tok/s) | P95 Latency (ms) |\n"
        top_table += "|------|-------------|---------|---------------------|------------------|\n"
        for i, r in enumerate(top, 1):
            fp = r.get("fingerprint", "")[:12]
            fit = r.get("fitness_score", 0)
            raw = r.get("raw_metrics", {})
            thr = raw.get("throughput_tok_s", "—")
            p95 = raw.get("p95_latency_ms", "—")
            top_table += f"| {i} | `{fp}` | {fit:.4f} | {thr} | {p95} |\n"

        kernel_section = ""
        if best_kernel_config:
            kernel_section = (
                "\n## Stage 2 — Kernel Optimisation\n\n"
                "| Parameter | Value |\n"
                "|-----------|-------|\n"
                + "".join(
                    f"| `{k}` | `{v}` |\n"
                    for k, v in best_kernel_config.items()
                )
            )

        md = (
            f"# OceanTune AI — Optimisation Report\n\n"
            f"**Session:** `{session_id}`  \n"
            f"**Model:** `{model_id}`  \n"
            f"**GPU:** `{gpu_type}`  \n"
            f"**Generated:** {datetime.now(timezone.utc).isoformat()}\n\n"
            "---\n\n"
            "## Winner Configuration\n\n"
            f"**Fingerprint:** `{analysis.winner_fingerprint[:16]}`  \n"
            f"**Fitness Score:** `{analysis.winner_fitness:.4f}`\n\n"
            "### Key Flags\n\n"
            + (
                "| Flag | Value |\n|------|-------|\n"
                + "".join(
                    f"| `{k}` | `{v}` |\n"
                    for k, v in analysis.winner_flags.items()
                    if k in analysis.key_flags or not analysis.key_flags
                )
            )
            + "\n\n"
            "---\n\n"
            "## LLM Analysis\n\n"
            f"**Explanation:**\n{analysis.explanation}\n\n"
            + (
                f"**OOM Insight:**\n{analysis.oom_insight}\n\n"
                if analysis.oom_insight else ""
            )
            + f"**Convergence:**\n{analysis.convergence_note}\n\n"
            f"**Recommendation:**\n{analysis.recommendation}\n\n"
            "---\n\n"
            "## Top 5 Configurations\n\n"
            + top_table
            + kernel_section
            + "\n\n---\n\n"
            "*Generated by [OceanTune AI](https://github.com/RithishRamesh-dev/oceantune-ai)*\n"
        )

        path = self._output_dir / f"report_{slug}.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
        return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Inline mapping of kernel param name → env_var (mirrors kernel_search_space.yaml)
_KERNEL_ENV_MAP: Dict[str, str] = {
    "vllm_rocm_use_aiter": "VLLM_ROCM_USE_AITER",
    "vllm_rocm_use_aiter_mla": "VLLM_ROCM_USE_AITER_MLA",
    "vllm_rocm_use_aiter_rmsnorm": "VLLM_ROCM_USE_AITER_RMSNORM",
    "vllm_rocm_use_aiter_moe": "VLLM_ROCM_USE_AITER_MOE",
    "nccl_min_nchannels": "NCCL_MIN_NCHANNELS",
    "nccl_socket_nthreads": "NCCL_SOCKET_NTHREADS",
    "rccl_enable_intranode": "RCCL_ENABLE_INTRANODE_COMM",
    "hsa_no_scratch_reclaim": "HSA_NO_SCRATCH_RECLAIM",
}


def _kernel_env_var(param_name: str) -> Optional[str]:
    return _KERNEL_ENV_MAP.get(param_name)
