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
        research_report=None,
        evolution_result=None,
        stage1_fitness: float = 0.0,
        stage2_fitness: float = 0.0,
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
            research_report=research_report,
            evolution_result=evolution_result,
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
            research_report=research_report,
            evolution_result=evolution_result,
            stage1_fitness=stage1_fitness,
            stage2_fitness=stage2_fitness,
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
        research_report=None,
        evolution_result=None,
    ) -> Path:
        merged_flags = {**analysis.winner_flags, **best_kernel_config}
        recipe: Dict[str, Any] = {
            "# OceanTune AI — Optimised Recipe": None,
            "session_id": session_id,
            "model_id": model_id,
            "gpu_type": gpu_type,
            "fitness_score": analysis.winner_fitness,
            "fingerprint": analysis.winner_fingerprint,
            "key_flags": analysis.key_flags,
            "vllm_flags": merged_flags,
            "stage2_strategy": best_kernel_config,
            "analyst_explanation": analysis.explanation,
            "recommendation": analysis.recommendation,
        }
        if research_report is not None:
            recipe["stage3_research"] = {
                "bottleneck_type": research_report.bottleneck_type,
                "bottleneck_explanation": research_report.bottleneck_explanation,
                "custom_kernel_warranted": research_report.custom_kernel_warranted,
                "custom_kernel_rationale": research_report.custom_kernel_rationale,
                "recommendations": [
                    {
                        "rank": r.rank,
                        "title": r.title,
                        "category": r.category,
                        "expected_improvement_pct": r.expected_improvement_pct,
                        "confidence": r.confidence,
                        "implementation": r.implementation,
                        "stage": r.stage,
                    }
                    for r in research_report.recommendations
                ],
            }
        if evolution_result is not None:
            recipe["stage4_kernel_engineering"] = {
                "op_type": evolution_result.op_type,
                "iterations_run": evolution_result.iterations_run,
                "best_speedup_pct": evolution_result.best_speedup_pct,
                "kernels_kept": evolution_result.total_kept,
                "kernels_reverted": evolution_result.total_reverted,
                "best_kernel_path": (
                    evolution_result.best_kernel.file_path
                    if evolution_result.best_kernel else None
                ),
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
        research_report=None,
        evolution_result=None,
        stage1_fitness: float = 0.0,
        stage2_fitness: float = 0.0,
    ) -> Path:
        top = analysis.top_configs[:5]

        # Per-stage improvement summary
        winner_em = {}
        if top:
            winner_em = top[0].get("enriched_metrics") or top[0].get("raw_metrics") or {}
        baseline_fitness = analysis.top_configs[-1].get("fitness_score", 0.0) if top else 0.0
        # Use the actual stage1 fitness passed in; fall back to winner fitness if not provided
        s1 = stage1_fitness or analysis.winner_fitness
        s2 = stage2_fitness or 0.0
        s2_delta = s2 - s1 if s2 > 0 else 0.0
        s2_delta_str = f"+{s2_delta:.4f} (+{s2_delta/s1*100:.1f}%)" if s1 > 0 and s2_delta > 0 else (
            f"{s2_delta:.4f} ({s2_delta/s1*100:.1f}%)" if s1 > 0 and s2_delta < 0 else "—"
        )
        s4_speedup = evolution_result.best_speedup_pct if evolution_result else 0.0

        stage_summary = (
            "## Pipeline Performance Summary\n\n"
            "| Stage | Fitness | Throughput (tok/s) | vs Previous |\n"
            "|-------|---------|---------------------|-------------|\n"
            f"| Stage 1 — vLLM Config Search | `{s1:.4f}` | "
            f"{winner_em.get('peak_throughput_tokens_per_sec', '—') if top else '—'} | baseline |\n"
        )
        if s2 > 0:
            stage_summary += (
                f"| Stage 2 — Inference Strategy | `{s2:.4f}` | "
                f"{'—'} | `{s2_delta_str}` |\n"
            )
        else:
            stage_summary += "| Stage 2 — Inference Strategy | — | — | no improvement |\n"
        if s4_speedup > 0:
            stage_summary += (
                f"| Stage 4 — Kernel Engineering | — | — | `+{s4_speedup:.1f}%` kernel speedup |\n"
            )
        stage_summary += "\n"

        top_table = "| Rank | Fingerprint | Fitness | Throughput (tok/s) | P95 Latency (ms) |\n"
        top_table += "|------|-------------|---------|---------------------|------------------|\n"
        for i, r in enumerate(top, 1):
            fp = r.get("fingerprint", "")[:12]
            fit = r.get("fitness_score", 0)
            em = r.get("enriched_metrics") or r.get("raw_metrics") or {}
            thr_raw = em.get("peak_throughput_tokens_per_sec", em.get("throughput_tok_s"))
            p95_raw = em.get("p95_latency_at_peak_ms", em.get("p95_latency_ms"))
            thr = f"{thr_raw:.1f}" if isinstance(thr_raw, (int, float)) else "—"
            p95 = f"{p95_raw:.1f}" if isinstance(p95_raw, (int, float)) else "—"
            top_table += f"| {i} | `{fp}` | {fit:.4f} | {thr} | {p95} |\n"

        # Stage 2 section
        stage2_section = ""
        if best_kernel_config:
            rows = "".join(
                f"| `{k}` | `{v}` |\n"
                for k, v in best_kernel_config.items()
                if not k.startswith("_")
            )
            s2_header = ""
            if s2 > 0 and s1 > 0:
                s2_header = (
                    f"**Fitness:** `{s1:.4f}` → `{s2:.4f}` "
                    f"(`{s2_delta_str}` improvement over Stage 1)\n\n"
                )
            stage2_section = (
                "\n---\n\n## Stage 2 — Inference Strategy\n\n"
                + s2_header
                + "Best strategy found on top of the Stage 1 winner:\n\n"
                "| Parameter | Value |\n"
                "|-----------|-------|\n"
                + (rows if rows else "| (no improvement over baseline) | — |\n")
            )
        else:
            stage2_section = (
                "\n---\n\n## Stage 2 — Inference Strategy\n\n"
                "No strategy override improved on the Stage 1 baseline. "
                "The winner configuration already uses near-optimal serving settings.\n"
            )

        # Stage 3 section
        stage3_section = ""
        if research_report is not None:
            recs_table = (
                "| Rank | Title | Category | Est. Gain | Confidence | Stage |\n"
                "|------|-------|----------|-----------|------------|-------|\n"
            )
            for rec in research_report.recommendations:
                recs_table += (
                    f"| {rec.rank} | {rec.title} | `{rec.category}` | "
                    f"+{rec.expected_improvement_pct:.0f}% | {rec.confidence} | `{rec.stage}` |\n"
                )

            impl_details = ""
            for rec in research_report.recommendations[:5]:
                impl_details += (
                    f"\n### {rec.rank}. {rec.title}\n\n"
                    f"**Category:** `{rec.category}` | "
                    f"**Expected gain:** +{rec.expected_improvement_pct:.0f}% | "
                    f"**Confidence:** {rec.confidence}\n\n"
                    f"{rec.description}\n\n"
                    f"**Implementation:** `{rec.implementation}`\n\n"
                    f"**Evidence:** {rec.evidence}\n"
                )

            custom_kernel_note = ""
            if research_report.custom_kernel_warranted:
                custom_kernel_note = (
                    f"\n> **Stage 4 Custom Kernel warranted:** "
                    f"{research_report.custom_kernel_rationale}\n"
                )

            stage3_section = (
                "\n---\n\n## Stage 3 — Profiling & Research\n\n"
                f"**Bottleneck type:** `{research_report.bottleneck_type}`\n\n"
                f"{research_report.bottleneck_explanation}\n\n"
                + (f"**Architecture notes:** {research_report.architecture_notes}\n\n"
                   if research_report.architecture_notes else "")
                + "### Optimization Recommendations\n\n"
                + recs_table
                + impl_details
                + custom_kernel_note
            )
        else:
            stage3_section = (
                "\n---\n\n## Stage 3 — Profiling & Research\n\n"
                "Profiling not run in this session.\n"
            )

        # Stage 4 section
        stage4_section = ""
        if evolution_result is not None:
            best_kernel = evolution_result.best_kernel
            iter_rows = ""
            for it in evolution_result.iterations:
                icon = "✅" if it.decision == "kept" else "❌" if it.decision in ("reverted", "failed_correctness") else "⏭"
                speedup = ""
                if it.benchmark:
                    speedup = f"+{it.benchmark.speedup_pct:.1f}%"
                iter_rows += f"| {it.iteration} | {icon} {it.decision} | {speedup} | {it.reason[:80]} |\n"

            stage4_section = (
                "\n---\n\n## Stage 4 — Autonomous Kernel Engineering\n\n"
                f"**Operation:** `{evolution_result.op_type}`  \n"
                f"**Iterations:** {evolution_result.iterations_run}  \n"
                f"**Best speedup:** `+{evolution_result.best_speedup_pct:.1f}%`  \n"
                f"**Kernels kept:** {evolution_result.total_kept} / "
                f"**reverted:** {evolution_result.total_reverted}\n\n"
            )
            if best_kernel:
                stage4_section += (
                    f"**Best kernel:** `{best_kernel.kernel_name}`  \n"
                    f"**Algorithm:** {best_kernel.algorithm or '—'}  \n"
                    f"**File:** `{best_kernel.file_path}`\n\n"
                )
            if iter_rows:
                stage4_section += (
                    "### Evolution History\n\n"
                    "| Iter | Decision | Speedup | Reason |\n"
                    "|------|----------|---------|--------|\n"
                    + iter_rows
                )
        elif research_report is not None and research_report.custom_kernel_warranted:
            stage4_section = (
                "\n---\n\n## Stage 4 — Autonomous Kernel Engineering\n\n"
                "Custom kernel generation was identified as warranted but "
                "Stage 4 was not enabled in this session. "
                "Set `stage4_enabled: true` in `oceantune.yaml` to activate.\n"
            )

        md = (
            f"# OceanTune AI — Optimisation Report\n\n"
            f"**Session:** `{session_id}`  \n"
            f"**Model:** `{model_id}`  \n"
            f"**GPU:** `{gpu_type}`  \n"
            f"**Generated:** {datetime.now(timezone.utc).isoformat()}\n\n"
            "---\n\n"
            + stage_summary
            + "---\n\n"
            "## Stage 1 Winner Configuration\n\n"
            f"**Fingerprint:** `{analysis.winner_fingerprint[:16]}`  \n"
            f"**Fitness Score:** `{analysis.winner_fitness:.4f}`\n\n"
            "### Key Flags\n\n"
            "| Flag | Value |\n|------|-------|\n"
            + "".join(
                f"| `{k}` | `{v}` |\n"
                for k, v in analysis.winner_flags.items()
                if k in analysis.key_flags or not analysis.key_flags
            )
            + "\n\n"
            "---\n\n"
            "## LLM Analysis\n\n"
            f"**Explanation:**\n{analysis.explanation}\n\n"
            + (f"**OOM Insight:**\n{analysis.oom_insight}\n\n" if analysis.oom_insight else "")
            + f"**Convergence:**\n{analysis.convergence_note}\n\n"
            f"**Recommendation:**\n{analysis.recommendation}\n\n"
            "---\n\n"
            "## Top 5 Configurations\n\n"
            + top_table
            + stage2_section
            + stage3_section
            + stage4_section
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
