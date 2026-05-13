"""
agents/research_agent.py
------------------------
Research Agent — Stage 3, Part 2.

Given a ProfileTrace (kernel timing breakdown for the winning configuration),
the Research Agent:

  1. Identifies the dominant kernels and their categories
  2. Searches for model-specific and kernel-specific optimization literature:
       - vLLM GitHub changelog and issues
       - FlashInfer / FlashAttention papers and benchmarks
       - Triton kernel repositories
       - arXiv papers on the specific model architecture
  3. Cross-references findings against the GPU vendor and model architecture
  4. Returns a ResearchReport with:
       - Ranked optimization recommendations (each with expected impact + evidence)
       - Specific vLLM flags or Triton kernel modifications to try
       - Assessment of whether custom kernel development is warranted (Stage 4)

The agent uses the DO Serverless Inference LLM to synthesize findings from
multiple sources into actionable recommendations.

Usage
-----
    researcher = ResearchAgent(do_client=client)
    report = await researcher.analyse(
        trace=profile_trace,
        winner_flags=winner_flags,
        model_id="Qwen/Qwen2.5-7B-Instruct",
        gpu_type="H200",
    )
    for rec in report.recommendations:
        print(rec.title, rec.expected_improvement_pct)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.do_client import DOClient, DOClientError, _strip_json_fences
from agents.profiler_agent import ProfileTrace

log = logging.getLogger("agents.research_agent")


# ---------------------------------------------------------------------------
# Research report dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OptimizationRecommendation:
    """A single ranked optimization recommendation."""
    rank: int
    title: str
    category: str             # kernel_flag | triton_kernel | vllm_flag | architecture
    description: str
    expected_improvement_pct: float   # estimated throughput gain %
    confidence: str           # high | medium | low
    evidence: str             # brief citation / rationale
    implementation: str       # concrete what-to-do: flag to set, code to change
    stage: str                # stage2 (flag change) | stage3_flag | stage4_custom_kernel
    requires_custom_code: bool = False
    # Machine-readable VLLMFlags field changes to apply when stage != "stage4_custom_kernel".
    # Keys must match VLLMFlags dataclass field names (snake_case).
    # Empty dict means the change cannot be expressed as a simple flag (requires code).
    vllm_flags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchReport:
    """Output of the Research Agent."""
    model_id: str
    gpu_type: str
    bottleneck_type: str

    # Ranked list of optimizations, highest expected impact first
    recommendations: List[OptimizationRecommendation] = field(default_factory=list)

    # Whether custom kernel development is warranted
    custom_kernel_warranted: bool = False
    custom_kernel_rationale: str = ""

    # Dominant bottleneck explanation
    bottleneck_explanation: str = ""

    # Model architecture notes relevant to optimization
    architecture_notes: str = ""

    # Raw LLM response for debugging
    raw_response: Optional[str] = None


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_RESEARCH_SYSTEM_PROMPT = """\
You are a world-class GPU inference optimization researcher with deep expertise in:
  - vLLM internals, attention kernels, and scheduling
  - FlashAttention, FlashInfer, and Triton kernel optimization
  - CUDA/HIP/ROCm performance profiling and kernel tuning
  - LLM model architectures (Transformer, MoE, MLA, GQA, MQA)
  - NVIDIA Hopper/Blackwell and AMD CDNA3/CDNA4 hardware characteristics

You will receive:
  - A kernel timing profile of a running vLLM inference server
  - The model architecture details
  - The GPU hardware profile
  - The current vLLM configuration

Your task: provide a deep, evidence-based analysis of optimization opportunities.

For each recommendation:
1. Explain EXACTLY which kernel or component to optimize and why the profiler shows it
2. Give a specific, concrete implementation (flag name, environment variable, or code change)
3. Cite the mechanism: why this change helps for this specific model+GPU combination
4. Estimate realistic throughput improvement based on published benchmarks or first principles
5. Distinguish between: (a) vLLM flag changes (low effort, try immediately),
   (b) existing Triton/CUTLASS kernel variants to enable, (c) custom kernel writing required

Be brutally honest about confidence level. "High" confidence means there is published
benchmark evidence for this specific combination. "Medium" means sound theoretical
reasoning. "Low" means speculative.

Respond with a JSON object:
{
  "bottleneck_explanation": "<2-3 sentences explaining what the profile shows>",
  "architecture_notes": "<1-2 sentences about this model's architecture that matter for optimization>",
  "recommendations": [
    {
      "rank": 1,
      "title": "<short title>",
      "category": "kernel_flag|triton_kernel|vllm_flag|architecture",
      "description": "<detailed description>",
      "expected_improvement_pct": <float>,
      "confidence": "high|medium|low",
      "evidence": "<citation or reasoning>",
      "implementation": "<exact flag, env var, or code change>",
      "stage": "stage2|stage3_flag|stage4_custom_kernel",
      "requires_custom_code": false,
      "vllm_flags": {"attention_backend": "FLASHINFER"}
    }
  ],
  "custom_kernel_warranted": false,
  "custom_kernel_rationale": "<if true, explain what custom kernel to write and expected gain>"
}

CRITICAL RULES FOR "vllm_flags":
- For stage2 and stage3_flag recommendations you MUST populate vllm_flags with the
  exact machine-readable dict needed to apply the change. An EMPTY vllm_flags ({})
  means the recommendation will be SILENTLY SKIPPED and never benchmarked.
- Use VLLMFlags Python field names (snake_case): "attention_backend", "kv_cache_dtype",
  "enable_prefix_caching", "enable_chunked_prefill", "max_num_batched_tokens",
  "max_num_seqs", "gpu_memory_utilization", "block_size", "scheduler_delay_factor",
  "num_scheduler_steps", "quantization", "dtype", "enforce_eager", etc.
- Examples:
    FlashInfer backend:      {"attention_backend": "FLASHINFER"}
    FP8 KV cache:            {"kv_cache_dtype": "fp8"}
    Prefix caching:          {"enable_prefix_caching": true}
    More sequences:          {"max_num_seqs": 512}
    More batched tokens:     {"max_num_batched_tokens": 32768}
    Smaller block size:      {"block_size": 16}
    Multi-step scheduling:   {"num_scheduler_steps": 8, "scheduler_delay_factor": 0.1}
- For stage4_custom_kernel set vllm_flags to {} (no flag change possible).

Do not include any text outside the JSON object.
"""

_CONTEXT_PROMPT_TEMPLATE = """\
Model: {model_id}
GPU: {gpu_type} ({vendor})

=== PROFILER RESULTS ===

NOTE: The profiling run used enforce_eager=True to disable CUDA graph capture.
This makes individual kernels (attention, GEMM, norm, rope) visible in the trace.
The production run uses CUDA graphs (enforce_eager=False) which are faster — the
kernel timings below reflect eager-mode proportions; relative ratios still indicate
the real bottleneck.

Bottleneck type: {bottleneck_type}
Top bottleneck kernel: {bottleneck_kernel}

Kernel timing breakdown:
  Attention:          {attention_pct:.1f}% of GPU time
  GEMM (projections): {gemm_pct:.1f}%
  MoE routing:        {moe_pct:.1f}%
  RoPE:               {rope_pct:.1f}%
  Normalization:      {norm_pct:.1f}%
  Communication:      {comm_pct:.1f}%
  Other:              {other_pct:.1f}%
  Python overhead:    {python_overhead_pct:.1f}% (est. CPU scheduling idle time)

Top 10 kernels by GPU time:
{top_kernels_table}

=== CURRENT VLLM CONFIGURATION (already applied — Stage 1 + Stage 2) ===
{winner_flags_json}

=== STAGE 2 STRATEGY CHANGES (these were already benchmarked and applied) ===
{stage2_strategy_json}
Do NOT re-recommend anything already listed above. Focus only on optimizations
not yet tried. Assign stage="stage3_flag" to any flag-based recommendation so
that Stage 3 can immediately benchmark it.

=== MODEL ARCHITECTURE ===
{architecture_json}

=== HARDWARE PROFILE ===
{gpu_profile_json}

Based on this profile, provide optimization recommendations for flags NOT yet applied.
REMEMBER: vllm_flags MUST be populated for every stage2/stage3_flag recommendation.
"""


class ResearchAgent:
    """
    Synthesizes kernel profiling data into ranked optimization recommendations.
    Uses the DO Serverless Inference LLM for deep technical analysis.
    """

    def __init__(self, do_client: DOClient) -> None:
        self._client = do_client

    async def analyse(
        self,
        *,
        trace: ProfileTrace,
        winner_flags: Dict[str, Any],
        model_id: str,
        gpu_type: str,
        stage2_strategy: Optional[Dict[str, Any]] = None,
        model_meta: Optional[Dict[str, Any]] = None,
        gpu_profile: Optional[Dict[str, Any]] = None,
    ) -> ResearchReport:
        """
        Analyse a ProfileTrace and return optimization recommendations.

        Parameters
        ----------
        trace : ProfileTrace
            Output from ProfilerAgent.run()
        winner_flags : dict
            The Stage 1+2 winning vLLM flags
        model_id : str
        gpu_type : str
        model_meta : dict, optional
            Entry from configs/models.yaml for this model
        gpu_profile : dict, optional
            Entry from configs/gpu_profiles.yaml for this GPU

        Returns
        -------
        ResearchReport with ranked recommendations
        """
        log.info(
            "ResearchAgent: analysing trace for %s on %s (bottleneck=%s)",
            model_id, gpu_type, trace.bottleneck_type,
        )

        # Build the top kernels table
        top_kernels_lines = []
        for k in trace.top_kernels[:10]:
            top_kernels_lines.append(
                f"  {k.gpu_time_pct:5.1f}%  {k.avg_time_us:8.1f} us  "
                f"x{k.call_count:4d}  [{k.category:9s}]  {k.name[:80]}"
            )
        top_kernels_table = "\n".join(top_kernels_lines) if top_kernels_lines else "  (no kernel data)"

        vendor = "amd" if gpu_type in {"MI300X", "MI325X", "MI350X"} else "nvidia"

        user_msg = _CONTEXT_PROMPT_TEMPLATE.format(
            model_id=model_id,
            gpu_type=gpu_type,
            vendor=vendor,
            bottleneck_type=trace.bottleneck_type,
            bottleneck_kernel=trace.bottleneck_kernel or "(unknown)",
            attention_pct=trace.attention_pct,
            gemm_pct=trace.gemm_pct,
            moe_pct=trace.moe_pct,
            rope_pct=trace.rope_pct,
            norm_pct=trace.norm_pct,
            comm_pct=trace.comm_pct,
            other_pct=trace.other_pct,
            python_overhead_pct=trace.python_overhead_pct,
            top_kernels_table=top_kernels_table,
            winner_flags_json=json.dumps(winner_flags, indent=2),
            stage2_strategy_json=json.dumps(stage2_strategy or {}, indent=2),
            architecture_json=json.dumps(model_meta or {}, indent=2),
            gpu_profile_json=json.dumps(gpu_profile or {}, indent=2),
        )

        report = ResearchReport(
            model_id=model_id,
            gpu_type=gpu_type,
            bottleneck_type=trace.bottleneck_type,
        )

        try:
            raw_text = await self._client.chat(
                messages=[{"role": "user", "content": user_msg}],
                system=_RESEARCH_SYSTEM_PROMPT,
                json_mode=True,
            )
            import json as _json
            parsed = _json.loads(_strip_json_fences(raw_text))
            report.raw_response = raw_text

            report.bottleneck_explanation = parsed.get("bottleneck_explanation", "")
            report.architecture_notes = parsed.get("architecture_notes", "")
            report.custom_kernel_warranted = parsed.get("custom_kernel_warranted", False)
            report.custom_kernel_rationale = parsed.get("custom_kernel_rationale", "")

            for raw_rec in parsed.get("recommendations", []):
                rec = OptimizationRecommendation(
                    rank=raw_rec.get("rank", 99),
                    title=raw_rec.get("title", ""),
                    category=raw_rec.get("category", "unknown"),
                    description=raw_rec.get("description", ""),
                    expected_improvement_pct=float(raw_rec.get("expected_improvement_pct", 0.0)),
                    confidence=raw_rec.get("confidence", "low"),
                    evidence=raw_rec.get("evidence", ""),
                    implementation=raw_rec.get("implementation", ""),
                    stage=raw_rec.get("stage", "stage3_flag"),
                    requires_custom_code=raw_rec.get("requires_custom_code", False),
                    vllm_flags=(
                        raw_rec.get("vllm_flags")
                        or _parse_vllm_flags_from_impl(raw_rec.get("implementation", ""))
                    ),
                )
                report.recommendations.append(rec)

            # Sort by rank
            report.recommendations.sort(key=lambda r: r.rank)

            log.info(
                "ResearchAgent: %d recommendations, custom_kernel_warranted=%s",
                len(report.recommendations), report.custom_kernel_warranted,
            )

        except (DOClientError, Exception) as exc:
            log.warning("ResearchAgent LLM call failed: %s", exc)
            # Return fallback recommendations based on bottleneck type
            report = self._fallback_report(report, trace, winner_flags)

        return report

    def _fallback_report(
        self,
        report: ResearchReport,
        trace: ProfileTrace,
        winner_flags: Dict[str, Any],
    ) -> ResearchReport:
        """Return heuristic recommendations when LLM is unavailable."""
        recs = []
        rank = 1

        # Always worth trying fp8 KV cache if not already set
        if winner_flags.get("kv_cache_dtype", "auto") == "auto":
            recs.append(OptimizationRecommendation(
                rank=rank, title="Enable FP8 KV Cache",
                category="vllm_flag",
                description="FP8 KV cache halves memory footprint, allowing 2x sequences.",
                expected_improvement_pct=15.0,
                confidence="high",
                evidence="Published vLLM benchmarks show 10-25% throughput gain.",
                implementation="--kv-cache-dtype fp8",
                stage="stage3_flag",
                vllm_flags={"kv_cache_dtype": "fp8"},
            ))
            rank += 1

        # Attention backend switch
        if winner_flags.get("attention_backend", "FLASH_ATTN") == "FLASH_ATTN":
            recs.append(OptimizationRecommendation(
                rank=rank, title="Try FlashInfer Attention Backend",
                category="kernel_flag",
                description="FlashInfer has paged-attention GQA optimizations.",
                expected_improvement_pct=8.0,
                confidence="medium",
                evidence="FlashInfer benchmarks show 5-15% gain for GQA models.",
                implementation="--attention-backend FLASHINFER",
                stage="stage3_flag",
                vllm_flags={"attention_backend": "FLASHINFER"},
            ))
            rank += 1

        if trace.attention_pct > 40:
            recs.append(OptimizationRecommendation(
                rank=rank, title="Attention is primary bottleneck (>40% GPU time)",
                category="triton_kernel",
                description="Attention dominates — kernel-level tuning warranted.",
                expected_improvement_pct=10.0,
                confidence="medium",
                evidence="Profile shows attention at {:.0f}% of GPU time.".format(trace.attention_pct),
                implementation="Profile with nvprof/rocprof to identify specific attention variant.",
                stage="stage4_custom_kernel",
                requires_custom_code=True,
                vllm_flags={},
            ))

        report.recommendations = recs
        report.bottleneck_explanation = (
            f"LLM unavailable. Heuristic analysis: {trace.bottleneck_type} bottleneck "
            f"with attention={trace.attention_pct:.1f}% GEMM={trace.gemm_pct:.1f}%."
        )
        return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Mapping of CLI flag strings → (VLLMFlags field name, value)
# Used as fallback when the LLM returns vllm_flags: {} but the implementation
# string contains parseable flag names.
_CLI_TO_FIELD: Dict[str, tuple] = {
    "--kv-cache-dtype":           ("kv_cache_dtype",          str),
    "--attention-backend":        ("attention_backend",        str),
    "--enable-chunked-prefill":   ("enable_chunked_prefill",   lambda _: True),
    "--enable-prefix-caching":    ("enable_prefix_caching",    lambda _: True),
    "--gpu-memory-utilization":   ("gpu_memory_utilization",   float),
    "--max-num-batched-tokens":   ("max_num_batched_tokens",   int),
    "--max-num-seqs":             ("max_num_seqs",             int),
    "--block-size":               ("block_size",               int),
    "--quantization":             ("quantization",             str),
    "--dtype":                    ("dtype",                    str),
    "--scheduler-delay-factor":   ("scheduler_delay_factor",   float),
    "--enable-dbo":               ("enable_dbo",               lambda _: True),
    "--enforce-eager":            ("enforce_eager",            lambda _: True),
}


def _parse_vllm_flags_from_impl(implementation: str) -> Dict[str, Any]:
    """
    Best-effort parse of vllm_flags from an implementation string.

    Handles two formats that the LLM commonly produces:

    1. CLI format:   "--kv-cache-dtype fp8 --attention-backend FLASHINFER"
    2. Assignment:   "attention_backend='FLASHINFER'" / "kv_cache_dtype='fp8'"
                     "max_num_seqs=512" / "enable_prefix_caching=True"

    Returns {} when nothing useful is found or the string is too complex.
    """
    result: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Pass 1: CLI format  (--flag-name value)
    # ------------------------------------------------------------------
    cli_skip = {"nsys", "profile", "python -m"}
    if not any(s in implementation for s in cli_skip):
        tokens = implementation.split()
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in _CLI_TO_FIELD:
                field_name, coerce = _CLI_TO_FIELD[tok]
                if coerce.__name__ == "<lambda>":
                    result[field_name] = True
                    i += 1
                else:
                    if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                        try:
                            result[field_name] = coerce(tokens[i + 1])
                        except (ValueError, TypeError):
                            pass
                        i += 2
                    else:
                        i += 1
            else:
                i += 1

    # ------------------------------------------------------------------
    # Pass 2: assignment / description format  (key=value or key='value')
    # Only runs for fields not already found in pass 1.
    # ------------------------------------------------------------------
    for pattern, extractor in _ASSIGN_PATTERNS:
        m = pattern.search(implementation)
        if m:
            try:
                partial = extractor(m)
                for k, v in partial.items():
                    result.setdefault(k, v)   # don't override CLI-parsed values
            except Exception:
                pass

    return result


# ---------------------------------------------------------------------------
# Regex patterns for assignment-format implementation strings
# ---------------------------------------------------------------------------

def _first_int(m: re.Match) -> Dict[str, Any]:
    """Return the first captured integer group."""
    field_name = m.lastgroup or ""
    # The named group is the field_name; group(1) is the value
    return {field_name: int(m.group(1))}


_ASSIGN_PATTERNS: List[tuple] = [
    # attention_backend='flashinfer' / attention_backend=FLASHINFER / VLLM_ATTENTION_BACKEND=FLASHINFER
    (
        re.compile(r"(?:attention[_-]backend|VLLM_ATTENTION_BACKEND)\s*[=:]\s*['\"]?(FLASHINFER|flashinfer|AUTO|auto|FLASH_ATTN|flash_attn)['\"]?", re.I),
        lambda m: {"attention_backend": m.group(1).upper()},
    ),
    # kv_cache_dtype='fp8' / kv_cache_dtype='fp8_e4m3'
    (
        re.compile(r"kv[_-]cache[_-]dtype\s*[=:]\s*['\"]?(fp8_e4m3|fp8_e5m2|fp8|auto)['\"]?", re.I),
        lambda m: {"kv_cache_dtype": m.group(1).lower()},
    ),
    # enable_prefix_caching=True
    (
        re.compile(r"enable[_-]prefix[_-]caching\s*[=:]\s*(true|True|1|yes)", re.I),
        lambda _: {"enable_prefix_caching": True},
    ),
    # enable_chunked_prefill=True
    (
        re.compile(r"enable[_-]chunked[_-]prefill\s*[=:]\s*(true|True|1|yes)", re.I),
        lambda _: {"enable_chunked_prefill": True},
    ),
    # max_num_seqs=512  (take first number found)
    (
        re.compile(r"max[_-]num[_-]seqs\s*[=:]\s*(\d+)"),
        lambda m: {"max_num_seqs": int(m.group(1))},
    ),
    # max_num_batched_tokens=16384
    (
        re.compile(r"max[_-]num[_-]batched[_-]tokens\s*[=:]\s*(\d+)"),
        lambda m: {"max_num_batched_tokens": int(m.group(1))},
    ),
    # block_size=16
    (
        re.compile(r"block[_-]size\s*[=:]\s*(\d+)"),
        lambda m: {"block_size": int(m.group(1))},
    ),
    # gpu_memory_utilization=0.95
    (
        re.compile(r"gpu[_-]memory[_-]util(?:ization)?\s*[=:]\s*(0?\.\d+|1\.0)"),
        lambda m: {"gpu_memory_utilization": float(m.group(1))},
    ),
    # scheduler_delay_factor=0.1
    (
        re.compile(r"scheduler[_-]delay[_-]factor\s*[=:]\s*(0?\.\d+)"),
        lambda m: {"scheduler_delay_factor": float(m.group(1))},
    ),
    # num_scheduler_steps=4
    (
        re.compile(r"num[_-]scheduler[_-]steps\s*[=:]\s*(\d+)"),
        lambda m: {"num_scheduler_steps": int(m.group(1))},
    ),
    # enforce_eager=True
    (
        re.compile(r"enforce[_-]eager\s*[=:]\s*(true|True|1|yes)", re.I),
        lambda _: {"enforce_eager": True},
    ),
]
