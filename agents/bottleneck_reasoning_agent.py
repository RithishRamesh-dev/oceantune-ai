"""
agents/bottleneck_reasoning_agent.py
--------------------------------------
Bottleneck Reasoning Agent — Stage 3, Part 1 (replaces simple profiler classification).

Given:
  - A ProfileTrace (Torch profiler: attention/GEMM/MoE time fractions)
  - Optional HardwareCounters (Nsight Compute / rocprof: SM occupancy, Tensor Core
    utilisation, DRAM BW, warp stall reasons)
  - RooflineAnalysis (if microbenchmarks were run)
  - Winner flags

Produces a BottleneckAnalysis with:
  - Primary bottleneck class (one of 8 categories below)
  - Evidence chain: which metrics led to this classification
  - Per-component breakdown with confidence
  - Specific diagnostic questions to guide kernel research

The 8 bottleneck classes (from PRAGMA's Conductor taxonomy + vLLM-specific):
  1. compute_tensor      — Tensor Core saturated (attention/GEMM compute-bound)
  2. compute_scalar      — Scalar math heavy (exp, sqrt in softmax, activations)
  3. memory_bandwidth    — DRAM BW saturated (large KV cache reads at low batch)
  4. memory_capacity     — OOM / KV cache spilling / fragmentation
  5. scheduling_overhead — Python dispatch overhead / micro-batch scheduling
  6. communication       — TP all-reduce / all-gather dominant
  7. occupancy_limited   — Low SM occupancy (register pressure / shared mem pressure)
  8. launch_overhead     — Short kernel duration → CUDA launch overhead dominant

Usage
-----
    reasoner = BottleneckReasoningAgent(do_client=client)
    analysis = await reasoner.analyse(
        trace=profile_trace,
        hw_counters=hardware_counters,
        winner_flags=winner_flags,
        model_id="deepseek-ai/DeepSeek-V3",
        gpu_type="H100",
    )
    print(analysis.primary_bottleneck)
    for finding in analysis.evidence_chain:
        print(" -", finding)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.do_client import DOClient, DOClientError, _strip_json_fences
from agents.profiler_agent import ProfileTrace
from profiling.hardware_counters import HardwareCounters
from microbench.roofline import RooflineAnalysis

log = logging.getLogger("agents.bottleneck_reasoning_agent")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ComponentAnalysis:
    """Bottleneck analysis for one pipeline component."""
    component: str          # attention | gemm | moe | norm | rope | comm | scheduler
    time_pct: float         # % of total time (from trace)
    bottleneck_class: str   # compute_tensor | memory_bandwidth | occupancy_limited | ...
    confidence: str         # high | medium | low
    evidence: str           # what metric(s) support this classification
    optimization_hint: str  # concrete what-to-do


@dataclass
class BottleneckAnalysis:
    """
    Full bottleneck analysis for the winning configuration.
    This feeds into KernelResearchAgent and KernelGenerationAgent.
    """
    model_id: str
    gpu_type: str
    session_id: str

    # Primary bottleneck
    primary_bottleneck: str = "unknown"
    # One of: compute_tensor | compute_scalar | memory_bandwidth |
    #          memory_capacity | scheduling_overhead | communication |
    #          occupancy_limited | launch_overhead

    primary_component: str = ""      # Which pipeline stage is the bottleneck
    primary_kernel: str = ""         # Specific kernel name if known

    # Evidence chain (ordered from strongest to weakest signal)
    evidence_chain: List[str] = field(default_factory=list)

    # Per-component breakdown
    components: List[ComponentAnalysis] = field(default_factory=list)

    # Roofline position
    arithmetic_intensity: float = 0.0
    ridge_point: float = 0.0
    roofline_bound: str = "unknown"   # compute | memory

    # LLM-generated narrative explanation
    explanation: str = ""
    diagnostic_questions: List[str] = field(default_factory=list)

    # What stage to proceed to
    recommended_action: str = ""
    # e.g. "kernel_generation:triton_attention" | "vllm_flag:kv_cache_dtype=fp8" | "done"

    raw_response: Optional[str] = None


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_BOTTLENECK_SYSTEM_PROMPT = """\
You are an expert GPU inference bottleneck analyst for LLM serving workloads.

You will receive:
  1. PyTorch profiler trace summary (kernel timing breakdown by category)
  2. Hardware counter data (Nsight Compute or rocprof) if available
  3. Roofline analysis results if available
  4. The current vLLM configuration and model metadata

Your task: classify the primary performance bottleneck with high precision.

## Bottleneck taxonomy:
- compute_tensor: Tensor Core saturated (attention or GEMM at >70% utilisation)
- compute_scalar: Scalar compute dominates (softmax, RoPE, activations, tokenization)
- memory_bandwidth: DRAM BW saturated — kernel is memory-bound (typical for decode)
- memory_capacity: OOM, KV cache overflow, fragmentation, or spilling
- scheduling_overhead: Python/vLLM scheduling latency > 15% of total, queue management
- communication: Tensor-parallel all-reduce/all-gather > 20% of GPU time
- occupancy_limited: Achieved SM occupancy < 50% — register spilling or shared memory overuse
- launch_overhead: Most kernels are <10μs — CUDA launch latency dominates

## Classification rules:
1. A profiler showing attention=60%+ with Tensor Core util>70%? → compute_tensor
2. Attention=60%+ but Tensor Core util<40%, DRAM BW util>60%? → memory_bandwidth
3. MoE>40% of GPU time? → compute_tensor (grouped GEMM) or memory_bandwidth (large expert weights)
4. comm_pct>25%? → communication (TP bottleneck)
5. All kernels <50μs each, high per-request latency? → launch_overhead or scheduling_overhead
6. OOM errors in logs, KV cache at 99%? → memory_capacity

## Output format (strict JSON):
{
  "primary_bottleneck": "<class from taxonomy>",
  "primary_component": "<attention|gemm|moe|comm|scheduler>",
  "primary_kernel": "<kernel name if known, else empty string>",
  "evidence_chain": [
    "<strongest signal first>",
    "<second signal>",
    ...
  ],
  "components": [
    {
      "component": "<name>",
      "time_pct": <float>,
      "bottleneck_class": "<class>",
      "confidence": "<high|medium|low>",
      "evidence": "<metrics that support this>",
      "optimization_hint": "<concrete next step>"
    }
  ],
  "arithmetic_intensity": <float>,
  "roofline_bound": "<compute|memory|unknown>",
  "explanation": "<2-4 sentence narrative>",
  "diagnostic_questions": [
    "<question 1 for kernel researcher>",
    "<question 2>",
    ...
  ],
  "recommended_action": "<kernel_generation:triton_attention|vllm_flag:...|stage4_custom|done>"
}

Be precise. If hardware counters are absent, say so and reduce confidence to medium/low.
Do not include any text outside the JSON object.
"""


class BottleneckReasoningAgent:
    """
    Deep bottleneck analysis combining profiler traces with hardware counters.
    """

    def __init__(self, do_client: DOClient) -> None:
        self._client = do_client

    async def analyse(
        self,
        *,
        trace: ProfileTrace,
        hw_counters: Optional[HardwareCounters] = None,
        roofline: Optional[RooflineAnalysis] = None,
        winner_flags: Dict[str, Any],
        model_id: str,
        gpu_type: str,
        session_id: str = "",
        model_meta: Optional[Dict[str, Any]] = None,
    ) -> BottleneckAnalysis:
        """
        Classify the bottleneck with LLM reasoning over multi-source profiler data.
        """
        analysis = BottleneckAnalysis(
            model_id=model_id,
            gpu_type=gpu_type,
            session_id=session_id,
        )

        user_msg = self._build_context(
            trace=trace,
            hw_counters=hw_counters,
            roofline=roofline,
            winner_flags=winner_flags,
            model_id=model_id,
            gpu_type=gpu_type,
            model_meta=model_meta or {},
        )

        try:
            raw = await self._client.chat(
                messages=[{"role": "user", "content": user_msg}],
                system=_BOTTLENECK_SYSTEM_PROMPT,
                json_mode=True,
            )
            import json as _json
            parsed = _json.loads(_strip_json_fences(raw))
            analysis.raw_response = raw

            analysis.primary_bottleneck = parsed.get("primary_bottleneck", "unknown")
            analysis.primary_component = parsed.get("primary_component", "")
            analysis.primary_kernel = parsed.get("primary_kernel", "")
            analysis.evidence_chain = parsed.get("evidence_chain", [])
            analysis.arithmetic_intensity = float(parsed.get("arithmetic_intensity", 0.0))
            analysis.roofline_bound = parsed.get("roofline_bound", "unknown")
            analysis.explanation = parsed.get("explanation", "")
            analysis.diagnostic_questions = parsed.get("diagnostic_questions", [])
            analysis.recommended_action = parsed.get("recommended_action", "")

            for c in parsed.get("components", []):
                analysis.components.append(ComponentAnalysis(
                    component=c.get("component", ""),
                    time_pct=float(c.get("time_pct", 0.0)),
                    bottleneck_class=c.get("bottleneck_class", "unknown"),
                    confidence=c.get("confidence", "low"),
                    evidence=c.get("evidence", ""),
                    optimization_hint=c.get("optimization_hint", ""),
                ))

            log.info(
                "BottleneckReasoningAgent: %s primary=%s component=%s",
                model_id, analysis.primary_bottleneck, analysis.primary_component,
            )

        except (DOClientError, Exception) as exc:
            log.warning("BottleneckReasoningAgent LLM call failed: %s", exc)
            # Heuristic fallback from trace data
            analysis = self._heuristic_classification(analysis, trace, hw_counters)

        return analysis

    def _build_context(
        self,
        *,
        trace: ProfileTrace,
        hw_counters: Optional[HardwareCounters],
        roofline: Optional[RooflineAnalysis],
        winner_flags: Dict[str, Any],
        model_id: str,
        gpu_type: str,
        model_meta: Dict[str, Any],
    ) -> str:
        sections = [
            f"Model: {model_id}",
            f"GPU: {gpu_type}",
            "",
            "=== TORCH PROFILER TRACE ===",
            f"Bottleneck (simple):   {trace.bottleneck_type}",
            f"Top kernel:            {trace.bottleneck_kernel}",
            f"Attention:       {trace.attention_pct:5.1f}%",
            f"GEMM:            {trace.gemm_pct:5.1f}%",
            f"MoE:             {trace.moe_pct:5.1f}%",
            f"RoPE:            {trace.rope_pct:5.1f}%",
            f"Norm:            {trace.norm_pct:5.1f}%",
            f"Communication:   {trace.comm_pct:5.1f}%",
            f"Python overhead: {trace.python_overhead_pct:5.1f}%",
            f"Other:           {trace.other_pct:5.1f}%",
        ]

        if trace.top_kernels:
            sections.append("\nTop 10 kernels by GPU time:")
            for k in trace.top_kernels[:10]:
                sections.append(
                    f"  {k.gpu_time_pct:5.1f}%  {k.avg_time_us:8.1f}us  "
                    f"x{k.call_count:4d}  [{k.category:9s}]  {k.name[:70]}"
                )

        if hw_counters is not None:
            sections.append("\n=== HARDWARE COUNTERS ===")
            if hw_counters.vendor == "nvidia" and hw_counters.nvidia:
                c = hw_counters.nvidia
                sections.extend([
                    f"SM throughput:       {c.sm_throughput_pct:.1f}% of peak",
                    f"Tensor Core active:  {c.tensor_active_pct:.1f}% of peak",
                    f"FP16 pipe active:    {c.fp16_active_pct:.1f}% of peak",
                    f"DRAM BW util:        {c.dram_bw_utilisation_pct:.1f}% of peak",
                    f"DRAM read:           {c.dram_read_gbps:.0f} GB/s",
                    f"L2 hit rate:         {c.l2_read_hit_rate_pct:.1f}%",
                    f"L1 hit rate:         {c.l1_hit_rate_pct:.1f}%",
                    f"Achieved occupancy:  {c.achieved_occupancy_pct:.1f}%",
                    f"Theoretical occ:     {c.theoretical_occupancy_pct:.1f}%",
                    f"Warp stall mem_dep:  {c.stall_memory_dependency_pct:.1f}%",
                    f"Warp stall long_sb:  {c.stall_long_scoreboard_pct:.1f}%",
                    f"Warp stall math:     {c.stall_math_pipe_throttle_pct:.1f}%",
                    f"Derived bottleneck:  {c.bottleneck}",
                ])
            elif hw_counters.vendor == "amd" and hw_counters.amd:
                c = hw_counters.amd
                sections.extend([
                    f"MFMA utilisation:    {c.mfma_utilisation_pct:.1f}%",
                    f"VALU utilisation:    {c.valu_utilisation_pct:.1f}%",
                    f"HBM BW util:         {c.hbm_bw_utilisation_pct:.1f}% of peak",
                    f"HBM read:            {c.hbm_read_gbps:.0f} GB/s",
                    f"L2 hit rate:         {c.l2_cache_hit_rate_pct:.1f}%",
                    f"Wavefront occupancy: {c.wavefront_occupancy:.1f}",
                    f"LDS bank conflicts:  {c.lds_bank_conflicts:.2f}x",
                    f"Derived bottleneck:  {c.bottleneck}",
                ])
            sections.append(f"HW bottleneck summary: {hw_counters.bottleneck_detail}")
        else:
            sections.append("\n=== HARDWARE COUNTERS: NOT AVAILABLE ===")
            sections.append("ncu/rocprof not run — reduce confidence in compute/memory distinction")

        if roofline is not None:
            sections.append("\n=== ROOFLINE ANALYSIS ===")
            sections.append(roofline.summary)
            for rec in roofline.recommendations[:2]:
                sections.append(f"  Rec: {rec[:120]}")

        sections.append(f"\n=== vLLM CONFIGURATION ===")
        sections.append(json.dumps(winner_flags, indent=2))

        if model_meta:
            sections.append(f"\n=== MODEL METADATA ===")
            sections.append(json.dumps(model_meta, indent=2))

        return "\n".join(sections)

    def _heuristic_classification(
        self,
        analysis: BottleneckAnalysis,
        trace: ProfileTrace,
        hw_counters: Optional[HardwareCounters],
    ) -> BottleneckAnalysis:
        """Fallback: rule-based bottleneck classification."""
        # Determine primary bottleneck from trace
        max_pct = 0.0
        primary_cat = "other"
        for cat, pct in [
            ("attention", trace.attention_pct),
            ("gemm", trace.gemm_pct),
            ("moe", trace.moe_pct),
            ("comm", trace.comm_pct),
            ("norm", trace.norm_pct),
            ("rope", trace.rope_pct),
        ]:
            if pct > max_pct:
                max_pct = pct
                primary_cat = cat

        analysis.primary_component = primary_cat

        # Check hardware counters for compute vs memory disambiguation
        if hw_counters and hw_counters.vendor == "nvidia" and hw_counters.nvidia:
            c = hw_counters.nvidia
            if c.dram_bw_utilisation_pct > 60 and c.tensor_active_pct < 40:
                analysis.primary_bottleneck = "memory_bandwidth"
            elif c.tensor_active_pct > 60:
                analysis.primary_bottleneck = "compute_tensor"
            elif c.achieved_occupancy_pct < 50:
                analysis.primary_bottleneck = "occupancy_limited"
            else:
                analysis.primary_bottleneck = "compute_tensor" if primary_cat in ("attention", "gemm", "moe") else "memory_bandwidth"
        elif hw_counters and hw_counters.vendor == "amd" and hw_counters.amd:
            c = hw_counters.amd
            if c.hbm_bw_utilisation_pct > 60:
                analysis.primary_bottleneck = "memory_bandwidth"
            elif c.mfma_utilisation_pct > 60:
                analysis.primary_bottleneck = "compute_tensor"
            else:
                analysis.primary_bottleneck = "compute_tensor"
        else:
            # No hardware counters: use trace heuristics
            if primary_cat == "comm":
                analysis.primary_bottleneck = "communication"
            elif primary_cat in ("attention", "gemm", "moe"):
                analysis.primary_bottleneck = "compute_tensor"  # assume compute-bound without counter data
            else:
                analysis.primary_bottleneck = "memory_bandwidth"

        analysis.evidence_chain = [
            f"Torch profiler: {primary_cat}={max_pct:.1f}% of GPU time (dominant category)",
            "Hardware counters: " + ("available" if hw_counters else "not available"),
        ]
        analysis.explanation = (
            f"Heuristic classification (LLM unavailable): primary={analysis.primary_bottleneck} "
            f"based on {primary_cat}={max_pct:.1f}% of GPU time."
        )
        analysis.recommended_action = (
            f"kernel_generation:triton_{primary_cat}"
            if primary_cat in ("attention", "gemm")
            else "done"
        )
        return analysis
