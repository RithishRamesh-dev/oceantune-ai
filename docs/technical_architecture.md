# OceanTune AI — Technical Architecture Document

**Version:** 5.0  
**Last Updated:** 2026-05-12  
**Status:** Active Development

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Pipeline Stages](#3-pipeline-stages)
   - [Stage 1 — vLLM Config Search](#31-stage-1--vllm-config-search)
   - [Stage 2 — Inference Strategy Search](#32-stage-2--inference-strategy-search)
   - [Stage 3 — Deep Profiling & Bottleneck Reasoning](#33-stage-3--deep-profiling--bottleneck-reasoning)
   - [Stage 4 — Autonomous Kernel Engineering](#34-stage-4--autonomous-kernel-engineering)
4. [Core Components](#4-core-components)
5. [Agent System](#5-agent-system)
6. [Data Model](#6-data-model)
7. [Configuration Reference](#7-configuration-reference)
8. [Hardware Support](#8-hardware-support)
9. [Fitness Scoring](#9-fitness-scoring)
10. [Output Artefacts](#10-output-artefacts)
11. [Deployment](#11-deployment)
12. [Component Interaction Diagrams](#12-component-interaction-diagrams)

---

## 1. Overview

OceanTune AI is an **autonomous LLM inference optimisation engine** that discovers the highest-throughput vLLM configuration for a given model and GPU without requiring any manual tuning. It wraps a multi-stage pipeline of AI agents that iteratively benchmark, analyse, and improve both server-level configuration flags and GPU kernel implementations.

### Design Goals

| Goal | How OceanTune achieves it |
|------|--------------------------|
| Zero-expert tuning | LLM agent proposes flag changes; human never touches vLLM flags |
| Reproducible results | Every config is fingerprinted, stored in MongoDB, emits shell scripts |
| Hardware-aware | Per-GPU profiles gate illegal flag combinations before benchmarking |
| Progressive depth | Run stops at Stage 1/2/3/4 depending on time budget and config |
| Safe by default | Correctness firewall validates custom kernels before deployment |

### What OceanTune Optimises

```
Input:  model_id="Qwen/Qwen2.5-7B-Instruct", gpu_type="H200"
Output: highest-fitness vLLM configuration + optional custom Triton kernels

Fitness = f(throughput, latency, memory_efficiency)
        = weighted harmonic mean of normalised per-metric scores
```

---

## 2. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         OceanTune AI                                │
│                                                                     │
│   CLI: oceantune run --model Qwen/Qwen2.5-7B-Instruct --gpu H200   │
│                           │                                         │
│                    ┌──────▼──────┐                                  │
│                    │  Controller │  ← OceanTuneConfig (YAML + env)  │
│                    │   Agent     │                                  │
│                    └──────┬──────┘                                  │
│                           │                                         │
│          ┌────────────────┼─────────────────┐                       │
│          │                │                 │                       │
│    ┌─────▼─────┐   ┌──────▼──────┐  ┌──────▼──────┐                │
│    │  Stage 1  │   │   Stage 2   │  │   Stage 3   │                │
│    │ vLLM Flag │   │  Inference  │  │  Profiling  │                │
│    │  Search   │   │  Strategy   │  │ Bottleneck  │                │
│    └─────┬─────┘   └──────┬──────┘  └──────┬──────┘                │
│          │                │                 │                       │
│          └────────────────┼─────────────────┘                       │
│                           │                                         │
│                    ┌──────▼──────┐                                  │
│                    │   Stage 4   │  (optional, stage4_enabled)      │
│                    │   Kernel    │                                  │
│                    │ Engineering │                                  │
│                    └──────┬──────┘                                  │
│                           │                                         │
│                    ┌──────▼──────┐                                  │
│                    │   Report    │  recipe.yaml + launch.sh +       │
│                    │ Generator   │  report.md                       │
│                    └─────────────┘                                  │
└─────────────────────────────────────────────────────────────────────┘
          │                                      │
   ┌──────▼──────┐                      ┌────────▼────────┐
   │  MongoDB    │                      │  DO Serverless  │
   │  (sessions  │                      │  Inference API  │
   │   configs   │                      │  (LLM agents)   │
   │  benchmarks)│                      └─────────────────┘
   └─────────────┘
```

### Component Layers

```
┌──────────────────────────────────────────────────────────────┐
│  Presentation Layer                                          │
│  oceantune.py (Click CLI)  ·  show_results.py               │
├──────────────────────────────────────────────────────────────┤
│  Orchestration Layer                                         │
│  ControllerAgent  ·  Coordinator  ·  NodeClient             │
├──────────────────────────────────────────────────────────────┤
│  Agent Layer (LLM-powered)                                   │
│  PlannerAgent  ·  ExecutorAgent  ·  AnalystAgent            │
│  StrategyOptimizerAgent  ·  ProfilerAgent                   │
│  BottleneckReasoningAgent  ·  ResearchAgent                  │
│  KernelResearchAgent  ·  KernelGenerationAgent              │
│  CorrectnessFirewallAgent  ·  KernelEvolutionAgent          │
├──────────────────────────────────────────────────────────────┤
│  Infrastructure Layer                                        │
│  VLLMServer  ·  BenchmarkEngine  ·  GPUSlotAllocator        │
│  PortAllocator  ·  Database  ·  DOClient                    │
├──────────────────────────────────────────────────────────────┤
│  Analysis Layer                                              │
│  MetricsCollector  ·  LogAnalyzer  ·  NcuProfiler           │
│  RocprofProfiler  ·  OperatorBench  ·  RooflineAnalyzer     │
├──────────────────────────────────────────────────────────────┤
│  Configuration Layer                                         │
│  OceanTuneConfig  ·  SearchSpace  ·  VLLMFlags              │
│  ConfigValidator  ·  gpu_profiles.yaml                      │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Pipeline Stages

### Full Pipeline Sequence

```
oceantune run
      │
      ▼
ControllerAgent._run_async()
      │
      ├──▶ Stage 1: vLLM Config Search
      │         │
      │         ├── Iteration 0: Baseline (bare minimum flags)
      │         ├── Iteration 1..N: PlannerAgent proposes → ExecutorAgent benchmarks
      │         │                   AnalystAgent evaluates → feeds next proposal
      │         └── Returns: (winner_flags, fingerprint, stage1_fitness)
      │
      ├──▶ Stage 2: Inference Strategy Search
      │         │
      │         ├── StrategyOptimizerAgent.run() — up to 12 iterations
      │         │   Each iter: LLM proposes strategy → benchmark → keep if better
      │         └── Returns: (best_strategy, winner_metrics, stage2_fitness)
      │
      ├──▶ Stage 3: Deep Profiling + Bottleneck Reasoning
      │         │
      │         ├── 3a. ProfilerAgent — PyTorch profiler trace
      │         ├── 3b. NcuProfiler / RocprofProfiler — hardware counters
      │         ├── 3c. BottleneckReasoningAgent — LLM classifies bottleneck
      │         ├── 3d. ResearchAgent — ranked vLLM flag recommendations
      │         ├── 3e. _try_flag_recommendations() — benchmark each, keep if better
      │         └── 3f. KernelResearchAgent — deep kernel research (if warranted)
      │         Returns: (report, bottleneck, kernel_research, stage3_fitness,
      │                   applied_recs, updated_flags)
      │
      ├──▶ Stage 4: Autonomous Kernel Engineering  [optional]
      │         │
      │         ├── KernelGenerationAgent — generate Triton kernel
      │         ├── CorrectnessFirewallAgent — validate vs PyTorch reference
      │         └── KernelEvolutionAgent — keep/revert loop
      │         Returns: EvolutionResult
      │
      └──▶ ReportGenerator
                │
                ├── recipe_*.yaml  — ready-to-use vLLM config
                ├── launch_*.sh    — docker run command
                └── report_*.md   — full analysis report
```

---

### 3.1 Stage 1 — vLLM Config Search

**Purpose:** Find the best combination of low-level vLLM server flags through iterative LLM-guided search.

**Search Space:** 21 tunable parameters across parallel config, cache config, model config, scheduler, attention backend, MoE, and speculative decoding.

#### Stage 1 Detailed Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: vLLM Config Search                                    │
│                                                                 │
│  Iteration 0 (Baseline)                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  VLLMFlags(tensor_parallel_size=1,                      │   │
│  │            gpu_memory_utilization=0.90,                 │   │
│  │            ..all other fields at defaults..)            │   │
│  │                      │                                  │   │
│  │               ExecutorAgent.run()                       │   │
│  │                      │                                  │   │
│  │         ┌────────────▼───────────────┐                  │   │
│  │         │  VLLMServer (Docker)        │                  │   │
│  │         │  vllm serve <model>         │                  │   │
│  │         │  --tensor-parallel-size 1   │                  │   │
│  │         │  --gpu-memory-util 0.90     │                  │   │
│  │         │  ...                        │                  │   │
│  │         └────────────┬───────────────┘                  │   │
│  │                      │                                  │   │
│  │         ┌────────────▼───────────────┐                  │   │
│  │         │  BenchmarkEngine           │                  │   │
│  │         │  concurrency ramp:         │                  │   │
│  │         │  [1,2,4,8,16,32,64,128]    │                  │   │
│  │         │  × context_configs         │                  │   │
│  │         └────────────┬───────────────┘                  │   │
│  │                      │                                  │   │
│  │         ┌────────────▼───────────────┐                  │   │
│  │         │  MetricsCollector          │                  │   │
│  │         │  → fitness_score: 0.6924   │                  │   │
│  │         │  → throughput: 5385 tok/s  │                  │   │
│  │         └────────────┬───────────────┘                  │   │
│  │                      │                                  │   │
│  │              MongoDB: insert benchmark_run              │   │
│  └──────────────────────────────────────────────────────── ┘   │
│                                                                 │
│  Iteration 1..N                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  AnalystAgent.evaluate_iteration() → analyst_eval dict   │  │
│  │  PlannerAgent.propose_next(                              │  │
│  │      current_best=best_flags,                            │  │
│  │      current_best_metrics=best_metrics,                  │  │
│  │      history=[{iteration, flags, fitness, rationale}...] │  │
│  │      analyst_eval={"bottleneck": ..., "recommendation":} │  │
│  │  ) → (new_flags: VLLMFlags, rationale: str)              │  │
│  │                                                          │  │
│  │  [same ExecutorAgent.run() → MetricsCollector loop]      │  │
│  │                                                          │  │
│  │  if fitness > best_fitness:                              │  │
│  │      best_flags = new_flags                              │  │
│  │      best_fitness = fitness                              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Returns: (winner_flags_dict, fingerprint, stage1_fitness)      │
└─────────────────────────────────────────────────────────────────┘
```

#### PlannerAgent Logic

```
PlannerAgent.propose_next()
      │
      ├── Build LLM context:
      │     - Model architecture (from models.yaml)
      │     - GPU hardware profile (from gpu_profiles.yaml)
      │     - Full search history (all prior iterations)
      │     - AnalystAgent evaluation of last iteration
      │     - Current best config + metrics
      │
      ├── LLM call → JSON response:
      │     {
      │       "change_type": "single_flag",
      │       "flag_name": "kv_cache_dtype",
      │       "new_value": "fp8",
      │       "rationale": "H200 supports FP8 natively ...",
      │       "expected_improvement_pct": 15.0
      │     }
      │
      ├── Parse → mutate current_best → VLLMFlags
      │
      └── Fallback (no LLM key):
            random mutation via SearchSpace.mutate()
```

#### VLLMFlags Fingerprint

Every configuration is identified by a deterministic SHA-256 fingerprint:

```python
fingerprint = SHA256(
    sorted(flags.to_dict().items())
).hexdigest()[:12]
# Example: "3278005fb230"
```

This prevents re-benchmarking the same config across iterations and sessions.

---

### 3.2 Stage 2 — Inference Strategy Search

**Purpose:** Layer higher-level inference strategies on top of the Stage 1 winner. Explores a different parameter class — not raw vLLM flags, but serving strategies that interact with the scheduler, memory management, and attention kernels.

**Key strategies explored:**

| Strategy | vLLM Flag | Expected Impact |
|----------|-----------|-----------------|
| FP8 KV cache | `--kv-cache-dtype fp8` | 10–25% throughput |
| FlashInfer backend | `--attention-backend FLASHINFER` | 5–15% on GQA models |
| Chunked prefill | `--enable-chunked-prefill` | Latency variance reduction |
| Prefix caching | `--enable-prefix-caching` | 15–25% if prefix reuse |
| Speculative decoding | `--speculative-model` | 30–50% for short outputs |
| AMD AITER kernels | env var | AMD-specific gains |

#### Stage 2 Detailed Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: StrategyOptimizerAgent.run()  (max 12 iterations)     │
│                                                                 │
│  Input: Stage 1 winner_flags + winner_metrics                   │
│                                                                 │
│  Each iteration:                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  LLM proposes strategy delta (one or more flag changes):  │  │
│  │  {                                                        │  │
│  │    "strategy_type": "kv_cache",                          │  │
│  │    "flags": {"kv_cache_dtype": "fp8"},                   │  │
│  │    "rationale": "...",                                    │  │
│  │    "expected_improvement_pct": 15.0                       │  │
│  │  }                                                        │  │
│  │                      │                                    │  │
│  │  Merge with Stage 1 winner_flags                          │  │
│  │  → trial_flags = {**winner_flags, **proposed_delta}       │  │
│  │                      │                                    │  │
│  │  ConfigValidator.validate(trial_flags, gpu_type)          │  │
│  │       if invalid → skip                                   │  │
│  │                      │                                    │  │
│  │  ExecutorAgent.run() → fitness                            │  │
│  │                      │                                    │  │
│  │  if fitness > best_fitness:                               │  │
│  │      best_strategy = proposed_delta                       │  │
│  │      best_fitness = fitness                               │  │
│  │      winner_flags = trial_flags  (cumulative)             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Fallback (no LLM): deterministic strategy sweep               │
│  [fp8 → prefix_caching → FLASHINFER → chunked_prefill → ...]   │
│                                                                 │
│  Returns: (best_strategy_delta, winner_metrics, stage2_fitness) │
└─────────────────────────────────────────────────────────────────┘
```

---

### 3.3 Stage 3 — Deep Profiling & Bottleneck Reasoning

**Purpose:** Understand *why* the current best config performs as it does, identify the dominant bottleneck with hardware-level evidence, then benchmark targeted flag changes and kernel improvements derived from the bottleneck analysis.

#### Stage 3 Detailed Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 3: Deep Profiling + Bottleneck Reasoning                     │
│                                                                     │
│  Input: Stage 1+2 merged winner_flags, stage2_fitness               │
│                                                                     │
│  ─── 3a. PyTorch Profiler Trace ─────────────────────────────────  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  VLLMServer starts with:                                     │  │
│  │    extra_docker_args: ["-v", "trace_dir:/tmp/vllm_profile"]  │  │
│  │    extra_vllm_args:   ["--profiler-config", '{"profiler":    │  │
│  │                         "torch", "torch_profiler_dir":       │  │
│  │                         "/tmp/vllm_profile"}']               │  │
│  │    env: VLLM_RPC_TIMEOUT=1800000                             │  │
│  │                                                              │  │
│  │  Warmup: 10 requests at optimal_concurrency                  │  │
│  │  POST /start_profile                                         │  │
│  │  Profile: 30 requests at optimal_concurrency                 │  │
│  │  POST /stop_profile                                          │  │
│  │  Wait for trace flush (retry 3s → 5s → 10s)                 │  │
│  │                                                              │  │
│  │  Parse Chrome trace JSON:                                    │  │
│  │  {"traceEvents": [{"ph":"X", "cat":"kernel", "dur":us, ...}]}│  │
│  │                                                              │  │
│  │  Classify kernels → ProfileTrace:                            │  │
│  │    attention_pct, gemm_pct, moe_pct, rope_pct, norm_pct,    │  │
│  │    comm_pct, python_overhead_pct                             │  │
│  │    top_kernels: List[KernelTiming]                           │  │
│  │    bottleneck_kernel: str (top kernel by GPU time)           │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ─── 3b. Hardware Counters ──────────────────────────────────────  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  if NVIDIA GPU:                                              │  │
│  │    NcuProfiler (nsight compute CLI)                         │  │
│  │    Metrics: SM throughput, Tensor Core active %,            │  │
│  │             DRAM BW util, warp stall reasons,               │  │
│  │             L1/L2 cache hit rates, occupancy                │  │
│  │    → NvidiaCounters → HardwareCounters                      │  │
│  │                                                              │  │
│  │  if AMD GPU:                                                 │  │
│  │    RocprofProfiler (omniperf > rocprofv2 > rocprof)         │  │
│  │    Metrics: MFMA util, VALU util, HBM BW,                   │  │
│  │             L2 hit rate, wavefront occupancy,               │  │
│  │             LDS bank conflicts                               │  │
│  │    → AmdCounters → HardwareCounters                         │  │
│  │                                                              │  │
│  │  if tool unavailable: hw_counters = None                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ─── 3c. Bottleneck Reasoning ──────────────────────────────────  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  BottleneckReasoningAgent.analyse(                           │  │
│  │      trace, hw_counters, winner_flags,                       │  │
│  │      model_id, gpu_type                                      │  │
│  │  ) → BottleneckAnalysis:                                     │  │
│  │      primary_bottleneck: one of 8 classes:                  │  │
│  │        compute_tensor | compute_scalar |                     │  │
│  │        memory_bandwidth | memory_capacity |                  │  │
│  │        scheduling_overhead | communication |                 │  │
│  │        occupancy_limited | launch_overhead                   │  │
│  │      primary_component: str (e.g. "FlashAttention-GQA")     │  │
│  │      primary_kernel: str                                     │  │
│  │      evidence_chain: List[str]                               │  │
│  │      recommended_action: str                                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ─── 3d. Research Agent ────────────────────────────────────────  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  ResearchAgent.analyse(                                      │  │
│  │      trace, winner_flags,                                    │  │
│  │      stage2_strategy,  ← "don't re-recommend these"         │  │
│  │      model_id, gpu_type                                      │  │
│  │  ) → ResearchReport:                                         │  │
│  │      recommendations: List[OptimizationRecommendation]       │  │
│  │        Each recommendation includes:                         │  │
│  │          rank, title, category, description                  │  │
│  │          expected_improvement_pct, confidence, evidence      │  │
│  │          implementation (CLI string)                         │  │
│  │          stage: "stage3_flag" | "stage4_custom_kernel"       │  │
│  │          vllm_flags: {"field": value}  ← machine-readable   │  │
│  │          (fallback: parsed from implementation string)       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ─── 3e. Flag Trials ───────────────────────────────────────────  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  For each recommendation with stage3_flag and vllm_flags:   │  │
│  │                                                              │  │
│  │    Skip if vllm_flags values already in winner_flags         │  │
│  │    Skip if fingerprint already benchmarked (DB check)        │  │
│  │                                                              │  │
│  │    trial_flags = merge(current_flags, rec.vllm_flags)        │  │
│  │    benchmark(trial_flags) → trial_fitness                    │  │
│  │                                                              │  │
│  │    if trial_fitness > current_fitness:                       │  │
│  │        current_flags = trial_flags     ← cumulative          │  │
│  │        current_fitness = trial_fitness                       │  │
│  │        applied_recs.append(rec + delta)                      │  │
│  │    else:                                                     │  │
│  │        revert (next rec uses previous best)                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ─── 3f. Kernel Research ───────────────────────────────────────  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Only if: custom_kernel_warranted = true                     │  │
│  │        OR bottleneck.recommended_action starts with          │  │
│  │           "kernel_generation"                                │  │
│  │                                                              │  │
│  │  KernelResearchAgent.research(                               │  │
│  │      bottleneck, trace, model_id, gpu_type,                  │  │
│  │      winner_flags=updated_flags  ← post-trial best           │  │
│  │  ) → KernelResearchReport:                                   │  │
│  │      approaches: List[KernelApproach]                        │  │
│  │        approach_type: existing_flag | triton_rewrite |       │  │
│  │                        cutlass_variant | aiter_flag          │  │
│  │        expected_speedup_pct, confidence                      │  │
│  │        can_use_existing_impl, existing_impl_flag             │  │
│  │        triton_approach, tile_config                          │  │
│  │      proceed_to_generation: bool                             │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Returns: (research_report, bottleneck_analysis, kernel_research,  │
│            stage3_fitness, applied_recs, updated_winner_flags)      │
└─────────────────────────────────────────────────────────────────────┘
```

#### Kernel Classification

```
Kernel name → category mapping:

"flash_attn*", "paged_attn*", "self_attn*"  → attention
"cutlass*", "cublas*", "gemm*", "matmul*"    → gemm
"rmsnorm*", "layernorm*"                     → norm
"rope*", "rotary*", "apply_rotary*"          → rope
"moe*", "expert*", "grouped_gemm*"           → moe
"nccl*", "all_reduce*", "rccl*"              → comm
everything else                              → other
```

---

### 3.4 Stage 4 — Autonomous Kernel Engineering

**Purpose:** When Stage 3 identifies a kernel bottleneck that cannot be addressed by flag changes alone, generate a custom Triton (or CUDA) kernel, validate it for correctness, and evolve it through a keep/revert benchmarking loop.

**Activation condition:** `stage4_enabled: true` in config AND Stage 3 returns `kernel_research` with `proceed_to_generation: true`.

#### Stage 4 Detailed Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 4: Autonomous Kernel Engineering                             │
│                                                                     │
│  Input: bottleneck_analysis, kernel_research, stage3_flags          │
│                                                                     │
│  KernelEvolutionAgent.evolve()  (max stage4_iterations=3)           │
│                                                                     │
│  For each iteration:                                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                                                              │  │
│  │  ① KernelGenerationAgent.generate()                         │  │
│  │     LLM generates complete Triton kernel file:              │  │
│  │     - @triton.autotune with hardware-specific tile configs   │  │
│  │     - TMA (H100/H200) or LDS padding (AMD) as appropriate   │  │
│  │     - Wrapper function matching PyTorch signature            │  │
│  │     - Built-in correctness test                              │  │
│  │     - Built-in microbenchmark                                │  │
│  │     Saved to: kernels/generated/<session_id>/<op>_v<n>.py   │  │
│  │                                                              │  │
│  │  ② CorrectnessFirewallAgent.validate()                      │  │
│  │     Shape sweep against PyTorch reference:                   │  │
│  │       Checks: max_abs_error, rms_error, NaN/Inf, determinism │  │
│  │       Thresholds per op:                                     │  │
│  │         attention: max_abs ≤ 1e-2                           │  │
│  │         rmsnorm:   max_abs ≤ 1e-3                           │  │
│  │         gemm:      max_abs ≤ 1e-2                           │  │
│  │     if fails → _attempt_repair() (LLM fixes kernel)         │  │
│  │              → if still fails: decision = "failed_correctness"│  │
│  │                continue to next iteration                    │  │
│  │                                                              │  │
│  │  ③ OperatorBench.run()   (microbenchmark)                   │  │
│  │     Isolated subprocess with CUDA events                    │  │
│  │     Roofline model: arithmetic_intensity vs ridge_point      │  │
│  │     speedup_pct = (custom_latency - reference_latency)      │  │
│  │                    / reference_latency × 100                 │  │
│  │                                                              │  │
│  │  ④ Keep/Revert decision:                                    │  │
│  │     if speedup_pct > 1.0%:   decision = "kept"              │  │
│  │                               best_kernel = this kernel      │  │
│  │     else:                    decision = "reverted"           │  │
│  │                               revert to previous best        │  │
│  │                                                              │  │
│  │  Persist: experiments/kernel_experiments.json               │  │
│  │           experiments/best_kernels.json                      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Returns: EvolutionResult:                                          │
│    best_kernel: GeneratedKernel                                     │
│    best_speedup_pct: float                                          │
│    iterations_run, total_kept, total_reverted: int                  │
└─────────────────────────────────────────────────────────────────────┘
```

#### Kernel Generation Targets (by operation type)

| Op Type | Reference | Hardware Optimisation |
|---------|-----------|----------------------|
| Attention (GQA/MHA) | `flash_attn_func` / SDPA | TMA tiling (H100+), paged cache fusion |
| GEMM | `torch.mm` / cuBLAS | CUTLASS stream-K, mixed precision |
| RMSNorm | `torch.nn.RMSNorm` | Fused kernel, vectorised loads |
| RoPE | manual application | Fused with Q/K projection |
| MoE dispatch | `torch.topk` + scatter | Grouped GEMM, EP fusion |

---

## 4. Core Components

### 4.1 VLLMServer

Manages the full lifecycle of a vLLM process running inside a Docker container.

```
VLLMServer lifecycle:
                                    
  start()                           
    │                               
    ├── docker rm -f oceantune-vllm-{port}   ← clean up stale containers
    │                               
    ├── _build_command() →           
    │   docker run                  
    │     --gpus device={CUDA_VISIBLE_DEVICES}  (NVIDIA)
    │     --device /dev/kfd          (AMD)
    │     -v {hf_cache}:/root/.cache/huggingface
    │     {extra_docker_args}        ← e.g. profiler volume mount
    │     {docker_image}             
    │     {model_id}                 ← positional vllm serve arg
    │     --host 0.0.0.0             
    │     --port {port}              
    │     {flags.to_vllm_args()}     ← VLLMFlags → CLI args
    │     {gpu_profile.vllm_extra_args}
    │     {extra_vllm_args}          ← e.g. --profiler-config JSON
    │                               
    ├── _capture_logs() task         ← async streaming into deque(maxlen=500)
    │                               
    ├── _wait_healthy()              ← poll GET /health (exp backoff, cap 10s)
    │     monitor logs for OOM / CUDA errors (fail-fast)
    │     classify: OOMError | StartupTimeout | PortConflict | CUDAError
    │                               
    └── state = HEALTHY              
                                    
  stop()                            
    ├── SIGTERM → process group      
    ├── wait grace_sec=10            
    ├── SIGKILL if still running     
    └── docker stop oceantune-vllm-{port}  ← belt-and-suspenders
```

**Failure hierarchy:**
- `OOMError` — CUDA OOM detected in logs
- `StartupTimeout` — /health never returned 200
- `PortConflict` — address already in use
- `CUDAError` — CUDA runtime error
- `ProcessCrash` — non-zero exit code

### 4.2 BenchmarkEngine

Runs the concurrency ramp benchmark against a running vLLM server.

```
BenchmarkEngine.run_full_ramp()
      │
      ├── For each concurrency level in [1,2,4,8,16,32,64,128]:
      │     For each (input_len, output_len) in context_configs:
      │         │
      │         ├── vllm bench serve
      │         │     --backend vllm
      │         │     --model {model_id}
      │         │     --num-prompts {num_prompts}
      │         │     --request-rate {concurrency}
      │         │     --input-len {input_len}
      │         │     --output-len {output_len}
      │         │
      │         └── Parse stdout → BenchmarkResult:
      │               requests_per_sec
      │               output_tokens_per_sec  ← primary throughput
      │               mean/p95/p99 latency_ms
      │               mean/p95/p99 ttft_ms   ← time to first token
      │               mean/p95/p99 tpot_ms   ← time per output token
      │               error_rate
      │
      └── Aggregate → RampResult:
            peak_throughput = max(output_tokens_per_sec across levels)
            best_concurrency = level at peak
            summary = {fitness-relevant aggregate metrics}
```

### 4.3 MetricsCollector & Fitness Score

The fitness score is a weighted harmonic mean of normalised per-metric scores, bounded to [0, 1].

```
fitness_score = weighted_harmonic_mean([
    throughput_score   × weight_throughput,
    latency_score      × weight_latency,
    ttft_score         × weight_ttft,
])

where:
  throughput_score = min(peak_throughput / BASELINE_THROUGHPUT, 1.0)
  latency_score    = 1 - min(p95_latency / MAX_ACCEPTABLE_LATENCY, 1.0)
  ttft_score       = 1 - min(mean_ttft / MAX_ACCEPTABLE_TTFT, 1.0)

BASELINE_THROUGHPUT = 5000 tok/s  (normalisation reference)
MAX_ACCEPTABLE_LATENCY = 10000 ms
MAX_ACCEPTABLE_TTFT = 2000 ms

primary_metric weighting (default: "throughput"):
  throughput:  0.70
  latency:     0.20
  ttft:        0.10
```

**Penalties applied before fitness calculation:**
- OOM failure: `fitness = 0.0`
- Startup timeout: `fitness = 0.0`
- Error rate > 5%: `fitness *= 0.5`

### 4.4 Database Schema (MongoDB)

```
Database: oceantune
├── sessions                ← one document per optimisation run
│   ├── _id: ObjectId
│   ├── model_id: str
│   ├── gpu_type: str
│   ├── strategy: str
│   ├── status: "running" | "done" | "error"
│   ├── context_configs: [[1024,1024], [1024,4096]]
│   ├── created_at: datetime
│   └── metadata: dict
│
├── configs                 ← one document per candidate VLLMFlags config
│   ├── _id: ObjectId
│   ├── session_id: str
│   ├── fingerprint: str   ← SHA-256[:12] of sorted flags
│   ├── flags: dict        ← full VLLMFlags dict
│   ├── generation: int    ← which iteration (-1 = Stage 3 trial)
│   ├── priority: int
│   ├── status: "pending" | "running" | "done" | "failed"
│   ├── fitness_score: float
│   ├── enriched_metrics: dict
│   └── error: str
│
├── benchmark_runs          ← one document per (config × context_config) run
│   ├── _id: ObjectId
│   ├── session_id: str
│   ├── config_id: str
│   ├── input_len, output_len: int
│   ├── raw_metrics: dict  ← BenchmarkResult fields
│   ├── enriched_metrics: dict  ← EnrichedMetrics fields
│   └── timestamp: datetime
│
├── kernel_runs             ← Stage 4 kernel evolution results
│   ├── session_id: str
│   ├── op_type: str
│   ├── iteration: int
│   ├── decision: "kept" | "reverted" | "failed_correctness"
│   ├── speedup_pct: float
│   ├── kernel_path: str
│   └── reason: str
│
└── nodes                   ← GPU node heartbeats (multi-node mode)
    ├── host: str
    ├── port: int
    ├── gpu_type: str
    ├── status: "idle" | "busy"
    └── last_seen: datetime
```

### 4.5 Resource Allocators

#### GPUSlotAllocator

```
GPUSlotAllocator(gpu_indices=[0,1,2,3,4,5,6,7], gpu_type="H100")

acquire(tensor_parallel_size=2) → [0, 1]  ← contiguous slot
    │
    ├── NVIDIA: CUDA_VISIBLE_DEVICES=0,1
    └── AMD:    ROCR_VISIBLE_DEVICES=0,1

release([0, 1]) → returns slot to pool
```

#### PortAllocator

```
PortAllocator(start=8000, end=8099)

acquire() → 8001   ← next free port from pool
release(8001) → returns to pool
```

Both allocators use `asyncio.Lock` for thread-safe access.

---

## 5. Agent System

### 5.1 DOClient (LLM Client)

All LLM calls go through `DOClient` which wraps the DigitalOcean Serverless Inference API (OpenAI-compatible):

```
DOClient.chat(
    messages=[{"role": "user", "content": prompt}],
    system=system_prompt,
    json_mode=True  ← force JSON output
) → str  (raw LLM response)

Model selection:
  1. OCEANTUNE_MODEL_ID env var (explicit override)
  2. DO_INFERENCE_MODEL env var
  3. "auto" → pick highest suitability_score from inference_models.yaml
  4. Fallback: empty string → agent uses heuristic fallback

Credentials:
  DO_INFERENCE_KEY    ← API key (required for LLM features)
  DO_INFERENCE_ENDPOINT  ← default: https://inference.do-ai.run/v1
```

**All agents are designed to degrade gracefully when `DO_INFERENCE_KEY` is absent.** They fall back to heuristic or random behaviour rather than crashing.

### 5.2 Agent Prompting Strategy

All LLM calls use a two-part prompt structure:

```
System prompt:  Role definition + output schema
                "You are a world-class GPU inference optimization researcher..."
                "Respond with a JSON object: {...schema...}"

User prompt:    Factual context (no instructions)
                - Current config flags
                - Benchmark metrics
                - GPU hardware profile
                - Model architecture
                - Search history
```

Requiring JSON output via `json_mode=True` (or via prompt instruction as fallback) makes responses machine-parseable. Every agent has a `_fallback_*()` method for the LLM-unavailable case.

### 5.3 Agent Responsibility Matrix

| Agent | Stage | Input | Output | LLM? | Fallback |
|-------|-------|-------|--------|------|----------|
| `PlannerAgent` | 1 | history, metrics | next VLLMFlags | Yes | random mutation |
| `ExecutorAgent` | 1 | VLLMFlags | benchmark results | Optional | skip sanity |
| `AnalystAgent` | 1 | session results | AnalysisResult | Yes | basic winner |
| `StrategyOptimizerAgent` | 2 | winner flags | strategy delta | Yes | deterministic sweep |
| `ProfilerAgent` | 3 | winner flags | ProfileTrace | No | log fallback |
| `BottleneckReasoningAgent` | 3 | trace + hw_counters | BottleneckAnalysis | Yes | heuristic 8-class |
| `ResearchAgent` | 3 | trace, flags | ResearchReport | Yes | heuristic recs |
| `KernelResearchAgent` | 3 | bottleneck | KernelResearchReport | Yes | skip Stage 4 |
| `KernelGenerationAgent` | 4 | bottleneck, research | GeneratedKernel | Yes | skeleton kernel |
| `CorrectnessFirewallAgent` | 4 | kernel, op_type | CorrectnessReport | No | subprocess test |
| `KernelEvolutionAgent` | 4 | bottleneck, research | EvolutionResult | Yes (repair) | no-op |

---

## 6. Data Model

### Key Dataclass Hierarchy

```
OceanTuneConfig
├── model_id: str
├── gpu_type: str
├── hf_token: str
├── agent: AgentConfig
│     ├── model: str
│     ├── max_tokens: int = 4096
│     ├── timeout_sec: int = 120
│     ├── max_turns: int = 6
│     ├── temperature: float = 0.3
│     ├── inference_key: str       ← DO_INFERENCE_KEY
│     └── inference_endpoint: str  ← DO_INFERENCE_ENDPOINT
├── database: DatabaseConfig
│     ├── uri: str                 ← MONGO_URI
│     ├── name: str = "oceantune"
│     └── collections: Dict[str, str]
├── nodes: List[NodeConfig]
│     ├── host: str = "localhost"
│     ├── node_port: int = 9000
│     ├── gpu_type: str = "H100"
│     └── gpu_indices: List[int] = [0..7]
├── coordinator: CoordinatorConfig
├── spaces: SpacesConfig  (DigitalOcean Spaces S3 storage)
├── vllm: VLLMConfig
│     ├── port: int = 8000
│     ├── startup_timeout_sec: int = 300
│     └── docker_image: str
├── benchmark: BenchmarkConfig
│     ├── concurrency_levels: [1,2,4,8,16,32,64]
│     ├── num_prompts: int = 200
│     ├── input_len: int = 1024
│     ├── output_len: int = 1024
│     └── duration_sec: int = 60
├── optimiser: OptimiserConfig
│     ├── strategy: str = "evolutionary"
│     ├── population_size: int = 10
│     ├── generations: int = 5
│     ├── mutation_rate: float = 0.2
│     ├── elite_fraction: float = 0.2
│     └── primary_metric: str = "throughput"
├── context_configs: List[Tuple[int,int]]
│     = [(1024,1024),(1024,4096),(1024,8192),(2048,8192),(4096,16384),(8192,32768)]
├── stage4_enabled: bool = False
└── stage4_iterations: int = 3
```

### VLLMFlags Parameters

```
VLLMFlags (21+ tunable parameters)
├── ParallelConfig
│     ├── tensor_parallel_size: int = 1        # [1,2,4,8]
│     ├── pipeline_parallel_size: int = 1      # [1,2,4]
│     ├── enable_expert_parallel: bool = False
│     ├── data_parallel_size: int = 1
│     └── distributed_executor_backend: str = "mp"  # "mp" | "ray"
├── CacheConfig
│     ├── gpu_memory_utilization: float = 0.90  # [0.70..0.95] step 0.05
│     ├── block_size: int = 16                  # [1,8,16,32]
│     ├── kv_cache_dtype: str = "auto"          # auto|fp8|fp8_e4m3|bfloat16
│     ├── enable_prefix_caching: bool = False
│     ├── max_num_seqs: int = 256               # [32,64,128,256,512]
│     └── max_num_batched_tokens: int = 8192    # [2048..65536]
├── ModelConfig
│     ├── dtype: str = "auto"                  # auto|bfloat16|float16
│     ├── quantization: Optional[str] = None   # fp8|awq|gptq|marlin
│     ├── max_model_len: int = 32768
│     ├── enforce_eager: bool = False
│     └── load_format: str = "auto"
├── Scheduler
│     ├── scheduler_delay_factor: float = 0.0  # [0.0..0.5]
│     └── enable_chunked_prefill: bool = False
├── AttentionConfig
│     └── attention_backend: str = "auto"     # auto|FLASH_ATTN|FLASHINFER|TRITON
├── MoE
│     ├── all2all_backend: str = "allgather_reducescatter"
│     └── enable_dbo: bool = False
├── Stage-2
│     ├── speculative_model: Optional[str] = None
│     └── num_speculative_tokens: Optional[int] = None
└── Metadata
      ├── cpu_offload_gb: int = 0
      ├── prefix_caching_hash_algo: str = "sha256"  # sha256|xxhash
      └── run_id: str  # internal, not passed to vLLM
```

---

## 7. Configuration Reference

### 7.1 Config File Hierarchy

```
Priority (later overrides earlier):
  1. Code-level defaults (dataclass fields)
  2. configs/oceantune.yaml
  3. Environment variables (OCEANTUNE_* prefix)
  4. CLI flags (--model, --gpu, --strategy)
```

### 7.2 Required Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `MONGO_URI` | MongoDB Atlas / self-hosted connection string | **Yes** |
| `DO_INFERENCE_KEY` | DigitalOcean Serverless Inference API key | No (agents fall back) |
| `HF_TOKEN` | Hugging Face token for gated models | Model-dependent |
| `VLLM_IMAGE` | Docker image override for vLLM | No (from gpu_profiles.yaml) |

### 7.3 Optional Environment Variables

| Variable | Description |
|----------|-------------|
| `DO_INFERENCE_ENDPOINT` | Override inference API URL |
| `DO_INFERENCE_MODEL` | Override LLM model selection |
| `DO_SPACES_KEY` | DigitalOcean Spaces S3 key |
| `DO_SPACES_SECRET` | DigitalOcean Spaces S3 secret |
| `OCEANTUNE_MODEL_ID` | Override model_id |
| `OCEANTUNE_GPU_TYPE` | Override gpu_type |
| `OCEANTUNE_PORT` | Override vLLM port |
| `OCEANTUNE_STRATEGY` | Override optimiser strategy |
| `OCEANTUNE_PRIMARY_METRIC` | Override primary_metric |

### 7.4 oceantune.yaml Quick Reference

```yaml
model_id: Qwen/Qwen2.5-7B-Instruct
gpu_type: H200  # H100 | H200 | B300 | MI300X | MI325X | MI350X

agent:
  model: auto
  max_tokens: 4096
  temperature: 0.3
  timeout_sec: 120
  max_turns: 6

benchmark:
  concurrency_levels: [1, 2, 4, 8, 16, 32, 64, 128]
  num_prompts: 30
  input_len: 1024
  output_len: 1024

optimiser:
  strategy: evolutionary  # uses LLM-guided search in v4+
  population_size: 10
  generations: 10
  primary_metric: throughput  # throughput | p95_latency | ttft | tpot

context_configs:
  - [1024, 1024]
  - [1024, 4096]

nodes:
  - host: localhost
    node_port: 9000
    gpu_type: H200
    gpu_indices: [0]

stage4_enabled: false   # requires ncu/rocprof + PyTorch+Triton
stage4_iterations: 3
```

---

## 8. Hardware Support

### 8.1 Supported GPU SKUs

| GPU | Vendor | VRAM | Compute | FP8 | Notes |
|-----|--------|------|---------|-----|-------|
| H100 | NVIDIA | 80 GB HBM3 | Hopper (sm_90) | Native | Max TP=8 |
| H200 | NVIDIA | 141 GB HBM3e | Hopper (sm_90) | Native | Max TP=8 |
| B300 | NVIDIA | 192 GB HBM3e | Blackwell (sm_100) | + NVFP4 | Max TP=8 |
| MI300X | AMD | 192 GB HBM3 | CDNA3 (gfx942) | Yes | AITER kernels, block_size=1 for MLA |
| MI325X | AMD | 256 GB HBM3e | CDNA3 (gfx942) | Yes | AITER kernels |
| MI350X | AMD | 288 GB HBM3e | CDNA4 (gfx950) | Yes | AITER kernels |

### 8.2 Vendor-Specific Behaviour

```
NVIDIA:
  GPU slot isolation:  CUDA_VISIBLE_DEVICES=0,1,...
  Docker GPU flag:     --gpus device={comma-separated-indices}
  Profiler:            ncu --csv --metrics {19 counter set}
  Kernel codegen:      Triton with TMA (H100/H200) or standard

AMD:
  GPU slot isolation:  ROCR_VISIBLE_DEVICES=0,1,...
  Docker GPU flag:     --device /dev/kfd --device /dev/dri
                       --group-add video
  Profiler:            omniperf (preferred) > rocprofv2 > rocprof
  Kernel codegen:      Triton with LDS padding for bank conflict avoidance
  Extra flags:         AITER environment variables via kernel_search_space.yaml
```

### 8.3 Hardware Counter Sets

**NVIDIA (ncu)**

| Counter | Meaning |
|---------|---------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM utilisation % |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak` | Tensor Core active % |
| `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` | Global load bytes |
| `dram__bytes.sum` | DRAM traffic |
| `smsp__sass_average_branch_targets_threads_uniform.pct` | Warp divergence |
| `smsp__warp_issue_stalled_mio_throttle_per_issue_active.pct` | Warp stall: mem throttle |

**AMD (rocprof/omniperf)**

| Counter | Meaning |
|---------|---------|
| `SQ_INSTS_MFMA` | MFMA (matrix) instructions |
| `SQ_INSTS_VALU` | VALU instructions |
| `TCC_EA_RDREQ` | HBM read requests |
| `TCC_HIT` | L2 cache hits |
| `LDS_BANK_CONFLICT` | LDS bank conflicts |
| `WAVE_OCCUPANCY` | Average wavefront occupancy |

---

## 9. Fitness Scoring

### 9.1 Fitness Formula

```
fitness_score ∈ [0.0, 1.0]

Step 1 — raw metric extraction from RampResult:
  peak_throughput = max(output_tokens_per_sec over all concurrency levels)
  best_concurrency = concurrency level at peak_throughput
  p95_latency = p95_latency_ms at best_concurrency
  mean_ttft = mean_ttft_ms at best_concurrency

Step 2 — normalise each metric to [0, 1]:
  throughput_score = min(peak_throughput / 5000, 1.0)
  latency_score    = 1 - min(p95_latency / 10000, 1.0)
  ttft_score       = 1 - min(mean_ttft / 2000, 1.0)

Step 3 — weighted sum (primary_metric="throughput"):
  fitness = 0.70 × throughput_score
          + 0.20 × latency_score
          + 0.10 × ttft_score

Step 4 — apply penalties:
  if oom_detected:               fitness = 0.0
  if startup_timeout:            fitness = 0.0
  if error_rate > 0.05:          fitness *= 0.5
  if peak_throughput < 100:      fitness *= 0.1   ← suspicious
```

### 9.2 Primary Metric Weights

| primary_metric | throughput | latency | ttft |
|----------------|-----------|---------|------|
| `throughput` | 0.70 | 0.20 | 0.10 |
| `p95_latency` | 0.20 | 0.70 | 0.10 |
| `ttft` | 0.20 | 0.10 | 0.70 |
| `tpot` | 0.50 | 0.40 | 0.10 |

### 9.3 Convergence Detection

The `AnalystAgent` reads the fitness time-series from MongoDB and reports:
- **Converged** — top-3 configs within 0.5% of each other
- **Improving** — monotonic increase over last 5 iterations
- **Stalled** — no improvement in last 3 iterations
- **Diverged** — mid-search collapse (exploration noise)

---

## 10. Output Artefacts

After every run, three files are written to `storage/results/`:

### 10.1 YAML Recipe (`recipe_*.yaml`)

Machine-readable optimised configuration, ready to paste into any deployment:

```yaml
model_id: Qwen/Qwen2.5-7B-Instruct
gpu_type: H200
fitness_score: 0.6925
fingerprint: 3278005fb230

vllm_flags:
  attention_backend: FLASHINFER
  gpu_memory_utilization: 0.9
  kv_cache_dtype: auto
  enable_chunked_prefill: false
  ...

stage2_strategy:
  attention_backend: FLASHINFER

stage3_research:
  bottleneck_type: unknown
  recommendations:
    - rank: 1
      title: "Enable FP8 KV Cache"
      vllm_flags: {kv_cache_dtype: fp8}
      expected_improvement_pct: 35.0
      ...

# Optional Stage 4 section
stage4_kernel_engineering:
  op_type: attention
  best_speedup_pct: 12.3
  best_kernel_path: kernels/generated/...
```

### 10.2 Shell Script (`launch_*.sh`)

Production-ready `docker run` command with all optimised flags merged across all stages:

```bash
#!/usr/bin/env bash
# OceanTune AI — Optimised vLLM launch script
# Fitness: 0.6925

MODEL=Qwen/Qwen2.5-7B-Instruct
IMAGE=vllm/vllm-openai:latest

docker run --gpus all --ipc=host \
  -p 8000:8000 \
  "$IMAGE" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --attention-backend FLASHINFER \   # ← Stage 2 flag included
  --kv-cache-dtype fp8               # ← Stage 3 flag (if applied)
```

### 10.3 Markdown Report (`report_*.md`)

Human-readable report sections:

```
## Pipeline Performance Summary
| Stage | Fitness | vs Previous |
|-------|---------|-------------|
| Stage 1 — vLLM Config Search  | 0.6924 | baseline       |
| Stage 2 — Inference Strategy  | 0.7103 | +0.0179 (+2.6%)|
| Stage 3 — Profiling + Trials  | 0.7241 | +0.0138 (+1.9%)|
| Stage 4 — Kernel Engineering  | —      | +8.3% speedup  |

## Stage 1 Winner Configuration
  Top 5 configs, key flags, analyst explanation

## Stage 2 — Inference Strategy
  Strategy flags applied, delta vs Stage 1

## Stage 3 — Profiling & Research
  Applied Flag Changes (validated in Stage 3)
  All Optimization Recommendations (with rank, confidence, evidence)

## Stage 4 — Autonomous Kernel Engineering
  Evolution history table (per-iteration decision + speedup)
```

---

## 11. Deployment

### 11.1 Prerequisites

```bash
# Required
Python >= 3.10
Docker with NVIDIA Container Toolkit (or ROCm for AMD)
MongoDB Atlas or self-hosted MongoDB 6+

# Environment variables
export MONGO_URI="mongodb+srv://user:pass@host/oceantune?tls=true"
export DO_INFERENCE_KEY="dop_v1_..."       # for LLM features
export HF_TOKEN="hf_..."                    # for gated models

# Optional (for Stage 3/4 profiling)
# NVIDIA: nsight-systems-cli, nsight-compute (ncu)
# AMD: rocprof / rocprofv2 / omniperf
# Kernel generation: torch, triton
```

### 11.2 Installation

```bash
git clone https://github.com/RithishRamesh-dev/oceantune-ai
cd oceantune-ai
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 11.3 Running

```bash
# Minimal run (Stage 1 only, uses heuristic fallback without LLM)
oceantune run --model Qwen/Qwen2.5-7B-Instruct --gpu H200

# Full pipeline with LLM-guided search
DO_INFERENCE_KEY=dop_v1_... oceantune run \
  --model Qwen/Qwen2.5-7B-Instruct \
  --gpu H200 \
  --config configs/oceantune.yaml

# Custom config
oceantune run --config my_config.yaml

# Dry run (validate config, no GPU required)
oceantune run --dry-run --config configs/oceantune.yaml

# Validate config file
oceantune validate-config --config configs/oceantune.yaml

# Show recent results
python show_results.py
```

### 11.4 Single-Node Architecture

```
GPU Server (e.g. DigitalOcean H200 Droplet)
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Host OS                                                │
│  ├── oceantune run  (Python process)                    │
│  │     ├── ControllerAgent                             │
│  │     ├── GPUSlotAllocator  (GPU indices [0..n])      │
│  │     ├── PortAllocator     (ports 8000–8099)         │
│  │     └── MongoDB client    (remote Atlas)            │
│  │                                                     │
│  └── Docker containers  (launched per experiment)      │
│       ├── oceantune-vllm-8000  (active benchmark)      │
│       └── oceantune-vllm-8001  (parallel, if tp=1)     │
│                                                         │
│  GPU: H200 (141 GB)                                     │
│  NVIDIA Container Toolkit                               │
└─────────────────────────────────────────────────────────┘
          │                          │
   MongoDB Atlas              DO Serverless
   (sessions, configs,        Inference API
    benchmark_runs)           (LLM agent calls)
```

### 11.5 Multi-Node Architecture (Coordinator Mode)

```
Control Node
┌───────────────────────────────────┐
│  ControllerAgent                  │
│  Coordinator ──── NodeClient      │
└───────────────────────────────────┘
         │              │
   ┌─────▼─────┐  ┌─────▼─────┐
   │ GPU Node 1│  │ GPU Node 2│
   │ node_server│  │ node_server│
   │  :9000    │  │  :9000    │
   │  NodeWorker│  │  NodeWorker│
   └───────────┘  └───────────┘
```

---

## 12. Component Interaction Diagrams

### 12.1 Full Session Sequence

```
User           CLI         Controller    Stage1        Stage2        Stage3        Stage4        DB            LLM
 │              │               │           │              │             │             │            │             │
 │  run(...)   │               │           │              │             │             │            │             │
 ├─────────────▶               │           │              │             │             │            │             │
 │              │  __init__()  │           │              │             │             │            │             │
 │              ├──────────────▶           │              │             │             │            │             │
 │              │  .run()      │           │              │             │             │            │             │
 │              ├──────────────▶           │              │             │             │            │             │
 │              │               │          │              │             │             │            │             │
 │              │               │ create_session()        │             │             │            │             │
 │              │               ├──────────────────────────────────────────────────────────────────▶            │
 │              │               │          │              │             │             │            │             │
 │              │               │ _stage1()│              │             │             │            │             │
 │              │               ├──────────▶              │             │             │            │             │
 │              │               │          │  (N iterations)            │             │            │             │
 │              │               │          │  propose_next()            │             │            │             │
 │              │               │          ├──────────────────────────────────────────────────────────────────▶ │
 │              │               │          │              │             │             │            │             │
 │              │               │          │  executor.run() (per config)             │            │             │
 │              │               │          │  VLLMServer + BenchmarkEngine            │            │             │
 │              │               │          │  → insert benchmark_run                 │            │             │
 │              │               │          ├──────────────────────────────────────────────────────▶            │
 │              │               │          │              │             │             │            │             │
 │              │               │◀─────────┘ (winner_flags, s1_fitness)│             │            │             │
 │              │               │          │              │             │             │            │             │
 │              │               │ _stage2()│              │             │             │            │             │
 │              │               ├────────────────────────▶             │             │            │             │
 │              │               │          │              │  propose strategy          │            │             │
 │              │               │          │              ├─────────────────────────────────────────────────────▶
 │              │               │          │              │  benchmark each            │            │             │
 │              │               │◀─────────────────────────────────────────────────────────────────────────────│
 │              │               │          │              │             │             │            │             │
 │              │               │ _stage3()│              │             │             │            │             │
 │              │               ├──────────────────────────────────────▶             │            │             │
 │              │               │          │              │  profiler + ncu           │            │             │
 │              │               │          │              │  bottleneck reasoning      │            │             │
 │              │               │          │              │  research + flag trials    │            │             │
 │              │               │◀──────────────────────────────────────────────────│             │            │
 │              │               │          │              │             │             │            │             │
 │              │               │ _stage4()│              │             │             │            │             │
 │              │               ├──────────────────────────────────────────────────▶│             │            │
 │              │               │          │              │             │  generate   │            │             │
 │              │               │          │              │             │  validate   │            │             │
 │              │               │          │              │             │  evolve     │            │             │
 │              │               │◀──────────────────────────────────────────────────┘             │            │
 │              │               │          │              │             │             │            │             │
 │              │               │ generate_report()       │             │             │            │             │
 │              │               ├──────────────────────────────────────────────────────────────────▶           │
 │              │               │ recipe.yaml + launch.sh + report.md                │            │             │
 │◀─────────────────────────────┘          │              │             │             │            │             │
```

### 12.2 ExecutorAgent Detail

```
ExecutorAgent.run(session_id, config_doc, context_configs)
│
├── await gpu_alloc.acquire(tp_size)     → slot=[0,1]
├── await port_alloc.acquire()           → port=8001
│
├── VLLMServer(flags, port=8001,
│             extra_env={CUDA_VISIBLE_DEVICES="0,1"})
│    ├── .start()
│    │    ├── docker rm -f oceantune-vllm-8001
│    │    ├── docker run --gpus device=0,1 ...
│    │    └── poll GET /health (timeout=1200s)
│    │         ├── OOM detected in logs → raise OOMError
│    │         └── healthy → proceed
│    │
│    ├── For each context in context_configs:
│    │    ├── BenchmarkEngine.run_full_ramp()
│    │    │    └── vllm bench serve ...
│    │    │         → parse stdout → BenchmarkResult
│    │    │
│    │    ├── LogAnalyzer.analyze(server.log_tail)
│    │    │    → LogAnalysis (load_time, kv_cache_blocks, errors)
│    │    │
│    │    ├── MetricsCollector.collect(ramp, analysis, flags, gpu_profile)
│    │    │    → EnrichedMetrics (fitness_score=0.6924)
│    │    │
│    │    └── db.create_benchmark_run(session_id, config_id, ...)
│    │
│    └── .stop()
│         ├── SIGTERM process group
│         └── docker stop oceantune-vllm-8001
│
├── await gpu_alloc.release(slot)
└── await port_alloc.release(port)
```

### 12.3 Stage 3 Flag Trial Loop

```
Stage 3e: _try_flag_recommendations()

winner_flags = {attention_backend: FLASHINFER, gpu_memory_util: 0.9, ...}
current_fitness = 0.7103

recommendation 1: {title: "FP8 KV Cache", vllm_flags: {kv_cache_dtype: fp8}}
  │
  ├── Skip check: winner_flags["kv_cache_dtype"] = "auto" ≠ "fp8"  → proceed
  │
  ├── trial_flags = {**winner_flags, kv_cache_dtype: "fp8"}
  ├── fingerprint = SHA256(trial_flags)
  ├── db.insert_config(fingerprint) → config_id (or None if duplicate)
  │
  ├── executor.run(config_id, context_configs)
  │    → fitness = 0.7241
  │
  ├── 0.7241 > 0.7103  → KEEP
  │    winner_flags["kv_cache_dtype"] = "fp8"
  │    current_fitness = 0.7241
  │    applied.append({title, flags, before=0.7103, after=0.7241, delta=+0.0138})
  │
recommendation 2: {title: "Prefix Caching", vllm_flags: {enable_prefix_caching: true}}
  │
  ├── trial_flags = {**winner_flags, enable_prefix_caching: true}
  ├── executor.run() → fitness = 0.7109
  │
  └── 0.7109 < 0.7241  → REVERT
       winner_flags unchanged (keep fp8, reject prefix_caching)

Returns: (updated_flags, 0.7241, [rec1_applied])
```

### 12.4 Stage 4 Kernel Evolution Loop

```
KernelEvolutionAgent.evolve()

get_reference_latency(op_type="attention")
  → reference_ms = 4.2 ms  (PyTorch SDPA / FlashAttn)

iteration 1:
  KernelGenerationAgent.generate()
    → triton_code (200+ lines)
    → saved: kernels/generated/<session>/attention_v1.py

  CorrectnessFirewallAgent.validate()
    → shape sweep: [B,H,N,D] across 12 shapes
    → max_abs_error = 0.003 ≤ 0.01  ✓
    → deterministic ✓
    → passed = True

  OperatorBench.run()
    → custom_ms = 3.7 ms
    → speedup_pct = (4.2 - 3.7) / 4.2 × 100 = 11.9%

  11.9% > 1.0%  → KEPT
    best_kernel = attention_v1.py
    best_speedup_pct = 11.9%

iteration 2:
  KernelGenerationAgent.generate()
    → different tile config (128×64 vs 64×64)

  CorrectnessFirewallAgent.validate()
    → max_abs_error = 0.018 > 0.01  ✗
    → _attempt_repair() → LLM fixes accumulation type
    → re-validate → passed = True

  OperatorBench.run()
    → custom_ms = 4.1 ms
    → speedup_pct = 2.4%

  2.4% > 1.0%  → KEPT
    best_speedup_pct = max(11.9%, 2.4%) = 11.9%  (v1 is still best)

persistence:
  experiments/kernel_experiments.json  ← all iterations
  experiments/best_kernels.json        ← current best per op_type
```

---

*Generated by OceanTune AI documentation toolchain.*  
*For issues and contributions: [github.com/RithishRamesh-dev/oceantune-ai](https://github.com/RithishRamesh-dev/oceantune-ai)*
