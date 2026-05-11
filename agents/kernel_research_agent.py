"""
agents/kernel_research_agent.py
---------------------------------
Kernel Research Agent — Stage 4, Part 1.

Given a BottleneckAnalysis, this agent:
  1. Identifies the specific kernel variant that is the primary bottleneck
  2. Research the best-known implementations for this kernel on the target GPU:
       - FlashAttention v2/v3 variants
       - FlashInfer (chunked prefill, paged KV cache, GQA, MLA)
       - CUTLASS / CuTe tiled GEMM templates
       - ThunderKittens (H100 TMA-based kernel library)
       - Triton autotune configs from vLLM / FlashInfer / OpenAI repos
       - ROCm AITER kernels (AMD-specific)
  3. Produces a KernelResearchReport with:
       - Specific implementation to try first (with code reference / flag)
       - Whether a custom Triton kernel is warranted vs existing flags
       - Tile size recommendations based on the profiler bottleneck class
       - Exact Triton code structure or pseudo-code for the LLM to generate from

The agent uses the LLM to synthesize knowledge about the kernel ecosystem.
It does NOT search the internet — instead it uses the LLM's training knowledge
about published benchmarks and kernel repositories.

Usage
-----
    researcher = KernelResearchAgent(do_client=client)
    report = await researcher.research(
        bottleneck=bottleneck_analysis,
        trace=profile_trace,
        model_id="Qwen/Qwen2.5-7B-Instruct",
        gpu_type="H100",
    )
    for approach in report.approaches:
        print(approach.title, approach.expected_speedup_pct)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.bottleneck_reasoning_agent import BottleneckAnalysis
from agents.do_client import DOClient, DOClientError, _strip_json_fences
from agents.profiler_agent import ProfileTrace

log = logging.getLogger("agents.kernel_research_agent")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class KernelApproach:
    """A single research-backed optimization approach for the bottleneck kernel."""
    rank: int
    title: str
    approach_type: str          # existing_flag | triton_rewrite | cutlass_variant | aiter_flag
    target_kernel: str          # Which kernel this targets
    expected_speedup_pct: float
    confidence: str             # high | medium | low
    evidence: str               # benchmark citation or first-principles reasoning

    # Implementation details
    can_use_existing_impl: bool = False  # Can a vLLM flag unlock this without writing code?
    existing_impl_flag: str = ""        # If yes, what flag to set
    triton_approach: str = ""           # If Triton rewrite: description of algorithm
    tile_config: Dict[str, int] = field(default_factory=dict)  # Suggested tile sizes

    # Architecture-specific notes
    nvidia_notes: str = ""
    amd_notes: str = ""

    # Whether this warrants KernelGenerationAgent
    warrants_custom_kernel: bool = False
    custom_kernel_complexity: str = ""  # low | medium | high


@dataclass
class KernelResearchReport:
    """Output of the Kernel Research Agent."""
    model_id: str
    gpu_type: str
    bottleneck: str
    target_kernel: str

    approaches: List[KernelApproach] = field(default_factory=list)

    # Top recommendation summary
    top_approach: Optional[KernelApproach] = None

    # Whether to proceed to custom kernel generation
    proceed_to_generation: bool = False
    generation_priority: str = ""  # What to generate: "triton_flash_attention" etc.

    # Architecture context the researcher extracted
    model_architecture_notes: str = ""

    raw_response: Optional[str] = None


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_RESEARCH_SYSTEM_PROMPT = """\
You are a world-class GPU kernel optimization researcher specializing in LLM inference.
Your expertise covers:
  - FlashAttention v1/v2/v3 algorithms and implementation variants
  - FlashInfer: chunked prefill, paged KV, GQA/MLA, online softmax
  - CUTLASS / CuTe: tiled GEMM templates for H100 Hopper, warp specialization
  - ThunderKittens: H100 TMA-based persistent kernel library
  - Triton: autotune, warp grouping, scratchpad tiling, persistent loops
  - ROCm AITER: AMD-specific attention, GEMM, and MoE kernels
  - vLLM internals: attention backends, prefill/decode split, continuous batching

You will receive:
  - A detailed bottleneck analysis with profiler data
  - The target kernel causing the bottleneck
  - Model architecture and GPU hardware specifications
  - Current vLLM configuration

Your task: identify the best concrete optimization approaches, ranked by expected impact.

For each approach, answer:
  1. Can an existing vLLM/FlashInfer/AITER flag unlock this? (flag name?)
  2. Does this require a custom Triton rewrite? (algorithm description?)
  3. What specific tile/block sizes work best for this hardware+model combination?
  4. What is the evidence (benchmark results, papers, or first-principles reasoning)?

## Key optimization patterns by bottleneck class:

### compute_tensor (attention or GEMM compute-bound):
- H100: Try FlashAttention v3 with warp specialization + TMA (--attention-backend FA3)
  Expected: 15-30% over FA2 on H100 for seqlen≥2048
- H100: ThunderKittens persistent attention kernel (if available in vLLM)
- Both: BLOCK_M=128, BLOCK_N=64, BLOCK_K=64 for fa2 decode; BLOCK_N=128 for prefill
- GQA: FlashInfer native GQA is 10-20% faster than FA2's GQA for large num_kv_heads gaps

### memory_bandwidth (attention or KV cache memory-bound — typical for decode):
- FP8 KV cache: halves KV cache bytes, allows 2x sequences or 2x context
  (--kv-cache-dtype fp8, HIGH CONFIDENCE, published vLLM benchmarks show 15-25% throughput gain)
- GQA with fewer KV heads: reduces KV bytes proportional to kv_head_reduction_ratio
- Paged attention page size: default 16 is often sub-optimal; 32 or 64 reduces padding waste
- Flash-decode: separate decode attention for single-token generation (lower mem pressure)

### communication (TP all-reduce bottleneck):
- Fuse all-reduce with the output projection (GEMM + reduce)
- Reduce TP degree if possible, or try expert parallelism for MoE
- NVLINK BW saturated? → reduce TP, use pipeline parallelism instead

### moe compute:
- CUTLASS grouped GEMM vs looped individual GEMMs: 20-40% faster
- Routing: fused top-k gating + dispatch
- AMD: AITER MoE kernel (--enable-aiter-moe)

Output format (strict JSON):
{
  "target_kernel": "<specific kernel name>",
  "model_architecture_notes": "<1-2 sentences about this model's arch that matter>",
  "approaches": [
    {
      "rank": 1,
      "title": "<short title>",
      "approach_type": "existing_flag|triton_rewrite|cutlass_variant|aiter_flag",
      "target_kernel": "<kernel name>",
      "expected_speedup_pct": <float>,
      "confidence": "high|medium|low",
      "evidence": "<citation or reasoning>",
      "can_use_existing_impl": true/false,
      "existing_impl_flag": "<flag string if applicable>",
      "triton_approach": "<algorithm description if custom triton>",
      "tile_config": {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64},
      "nvidia_notes": "<H100/A100 specific notes>",
      "amd_notes": "<MI300X specific notes>",
      "warrants_custom_kernel": true/false,
      "custom_kernel_complexity": "low|medium|high"
    }
  ],
  "proceed_to_generation": true/false,
  "generation_priority": "<triton_flash_attention|triton_gemm|triton_moe|none>"
}

Do not include any text outside the JSON object.
"""


class KernelResearchAgent:
    """
    Researches best-known kernel optimization approaches for a given bottleneck.
    Uses LLM knowledge of published benchmarks, GitHub repos, and technical papers.
    """

    def __init__(self, do_client: DOClient) -> None:
        self._client = do_client

    async def research(
        self,
        *,
        bottleneck: BottleneckAnalysis,
        trace: ProfileTrace,
        model_id: str,
        gpu_type: str,
        winner_flags: Optional[Dict[str, Any]] = None,
        model_meta: Optional[Dict[str, Any]] = None,
        gpu_profile: Optional[Dict[str, Any]] = None,
    ) -> KernelResearchReport:
        """
        Research optimization approaches for the identified bottleneck.
        """
        log.info(
            "KernelResearchAgent: researching %s on %s (bottleneck=%s component=%s)",
            model_id, gpu_type, bottleneck.primary_bottleneck, bottleneck.primary_component,
        )

        report = KernelResearchReport(
            model_id=model_id,
            gpu_type=gpu_type,
            bottleneck=bottleneck.primary_bottleneck,
            target_kernel=bottleneck.primary_kernel or bottleneck.primary_component,
        )

        user_msg = self._build_context(
            bottleneck=bottleneck,
            trace=trace,
            model_id=model_id,
            gpu_type=gpu_type,
            winner_flags=winner_flags or {},
            model_meta=model_meta or {},
            gpu_profile=gpu_profile or {},
        )

        try:
            raw = await self._client.chat(
                messages=[{"role": "user", "content": user_msg}],
                system=_RESEARCH_SYSTEM_PROMPT,
                json_mode=True,
            )
            import json as _json
            parsed = _json.loads(_strip_json_fences(raw))
            report.raw_response = raw

            report.target_kernel = parsed.get("target_kernel", report.target_kernel)
            report.model_architecture_notes = parsed.get("model_architecture_notes", "")
            report.proceed_to_generation = parsed.get("proceed_to_generation", False)
            report.generation_priority = parsed.get("generation_priority", "none")

            for raw_a in parsed.get("approaches", []):
                approach = KernelApproach(
                    rank=int(raw_a.get("rank", 99)),
                    title=raw_a.get("title", ""),
                    approach_type=raw_a.get("approach_type", "unknown"),
                    target_kernel=raw_a.get("target_kernel", ""),
                    expected_speedup_pct=float(raw_a.get("expected_speedup_pct", 0.0)),
                    confidence=raw_a.get("confidence", "low"),
                    evidence=raw_a.get("evidence", ""),
                    can_use_existing_impl=raw_a.get("can_use_existing_impl", False),
                    existing_impl_flag=raw_a.get("existing_impl_flag", ""),
                    triton_approach=raw_a.get("triton_approach", ""),
                    tile_config=raw_a.get("tile_config", {}),
                    nvidia_notes=raw_a.get("nvidia_notes", ""),
                    amd_notes=raw_a.get("amd_notes", ""),
                    warrants_custom_kernel=raw_a.get("warrants_custom_kernel", False),
                    custom_kernel_complexity=raw_a.get("custom_kernel_complexity", "medium"),
                )
                report.approaches.append(approach)

            report.approaches.sort(key=lambda a: a.rank)
            if report.approaches:
                report.top_approach = report.approaches[0]

            log.info(
                "KernelResearchAgent: %d approaches, top=%s (speedup=%.1f%%), generate=%s",
                len(report.approaches),
                report.top_approach.title if report.top_approach else "none",
                report.top_approach.expected_speedup_pct if report.top_approach else 0,
                report.proceed_to_generation,
            )

        except (DOClientError, Exception) as exc:
            log.warning("KernelResearchAgent LLM call failed: %s", exc)
            report = self._fallback_research(report, bottleneck, gpu_type)

        return report

    def _build_context(
        self,
        *,
        bottleneck: BottleneckAnalysis,
        trace: ProfileTrace,
        model_id: str,
        gpu_type: str,
        winner_flags: Dict[str, Any],
        model_meta: Dict[str, Any],
        gpu_profile: Dict[str, Any],
    ) -> str:
        vendor = "amd" if gpu_type in {"MI300X", "MI325X", "MI350X"} else "nvidia"
        sections = [
            f"Model: {model_id}",
            f"GPU: {gpu_type} (vendor={vendor})",
            "",
            "=== BOTTLENECK ANALYSIS ===",
            f"Primary bottleneck: {bottleneck.primary_bottleneck}",
            f"Primary component:  {bottleneck.primary_component}",
            f"Primary kernel:     {bottleneck.primary_kernel or '(unknown)'}",
            f"Roofline bound:     {bottleneck.roofline_bound}",
            f"Arithmetic intens:  {bottleneck.arithmetic_intensity:.1f} FLOP/byte",
            "",
            "Evidence chain:",
        ]
        for e in bottleneck.evidence_chain:
            sections.append(f"  - {e}")
        sections.append(f"\nExplanation: {bottleneck.explanation}")

        sections.append("\n=== PROFILER SUMMARY ===")
        sections.extend([
            f"Attention: {trace.attention_pct:.1f}%",
            f"GEMM:      {trace.gemm_pct:.1f}%",
            f"MoE:       {trace.moe_pct:.1f}%",
            f"Comm:      {trace.comm_pct:.1f}%",
        ])
        if trace.top_kernels:
            sections.append("Top 5 kernels:")
            for k in trace.top_kernels[:5]:
                sections.append(f"  {k.gpu_time_pct:.1f}%  {k.avg_time_us:.0f}us  {k.name[:70]}")

        if bottleneck.components:
            sections.append("\n=== COMPONENT BREAKDOWN ===")
            for c in bottleneck.components[:5]:
                sections.append(
                    f"  {c.component}: {c.time_pct:.1f}%  [{c.bottleneck_class}]  "
                    f"hint={c.optimization_hint[:60]}"
                )

        sections.append(f"\n=== CURRENT vLLM FLAGS ===")
        sections.append(json.dumps(winner_flags, indent=2))

        if model_meta:
            sections.append(f"\n=== MODEL METADATA ===")
            sections.append(json.dumps(model_meta, indent=2))

        if gpu_profile:
            sections.append(f"\n=== GPU PROFILE ===")
            sections.append(json.dumps(gpu_profile, indent=2))

        return "\n".join(sections)

    def _fallback_research(
        self,
        report: KernelResearchReport,
        bottleneck: BottleneckAnalysis,
        gpu_type: str,
    ) -> KernelResearchReport:
        """Heuristic approaches when LLM is unavailable."""
        vendor = "amd" if gpu_type in {"MI300X", "MI325X", "MI350X"} else "nvidia"

        if bottleneck.primary_bottleneck in ("compute_tensor", "memory_bandwidth"):
            if bottleneck.primary_component == "attention":
                report.approaches.append(KernelApproach(
                    rank=1,
                    title="Try FlashInfer attention backend",
                    approach_type="existing_flag",
                    target_kernel="attention",
                    expected_speedup_pct=8.0,
                    confidence="medium",
                    evidence="FlashInfer native GQA/paged-attn outperforms FA2 for GQA models.",
                    can_use_existing_impl=True,
                    existing_impl_flag="--attention-backend FLASHINFER",
                    warrants_custom_kernel=False,
                ))
                if vendor == "nvidia":
                    report.approaches.append(KernelApproach(
                        rank=2,
                        title="FlashAttention v3 (H100 TMA + warp specialization)",
                        approach_type="existing_flag",
                        target_kernel="flash_attn_fwd",
                        expected_speedup_pct=20.0,
                        confidence="high",
                        evidence="FA3 paper: 1.5-2x over FA2 on H100 for seqlen≥2048.",
                        can_use_existing_impl=True,
                        existing_impl_flag="--attention-backend FLASH_ATTN_V3",
                        nvidia_notes="Requires H100+ and vLLM>=0.6.0",
                        warrants_custom_kernel=False,
                    ))

            elif bottleneck.primary_component == "gemm":
                report.approaches.append(KernelApproach(
                    rank=1,
                    title="FP8 weight quantization for GEMM",
                    approach_type="existing_flag",
                    target_kernel="gemm",
                    expected_speedup_pct=25.0,
                    confidence="high",
                    evidence="FP8 GEMM on H100 achieves near 2x throughput vs FP16.",
                    can_use_existing_impl=True,
                    existing_impl_flag="--quantization fp8",
                    warrants_custom_kernel=False,
                ))

        elif bottleneck.primary_bottleneck == "memory_bandwidth":
            report.approaches.append(KernelApproach(
                rank=1,
                title="FP8 KV cache to halve KV memory bandwidth",
                approach_type="existing_flag",
                target_kernel="kv_cache",
                expected_speedup_pct=15.0,
                confidence="high",
                evidence="FP8 KV: 2x compression, 15-25% throughput gain (vLLM benchmarks).",
                can_use_existing_impl=True,
                existing_impl_flag="--kv-cache-dtype fp8",
                warrants_custom_kernel=False,
            ))

        if report.approaches:
            report.top_approach = report.approaches[0]
        report.proceed_to_generation = any(
            a.warrants_custom_kernel for a in report.approaches
        )
        return report
