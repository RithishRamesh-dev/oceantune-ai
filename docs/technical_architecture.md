# OceanTune AI — Technical Architecture Document

**Version:** 6.0  
**Last Updated:** 2026-09-29  
**Status:** Active Development

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [End-to-End Pipeline](#3-end-to-end-pipeline)
4. [Pipeline Phases & Stages](#4-pipeline-phases--stages)
   - [Enablement](#41-enablement--boot-repair)
   - [Prelude](#42-prelude--warm-recipe--reject-seed)
   - [Stage 1](#43-stage-1--serving-config-search)
   - [Macro Stage 2↔3](#44-macro-cycle--stage-2--stage-3)
   - [Stage 2](#45-stage-2--inference-strategy-search)
   - [Stage 3](#46-stage-3--deep-profiling--bottleneck-reasoning)
   - [Operating-Point Sweep](#47-operating-point-sweep)
   - [Stage 4](#48-stage-4--autonomous-kernel-engineering)
   - [Stage 4b Campaign](#49-stage-4b--fusion-campaign--ledger)
   - [CLOSE](#410-close--sedimentation--artefacts)
5. [Core Components](#5-core-components)
6. [Agent System](#6-agent-system)
7. [Data Model](#7-data-model)
8. [Configuration Reference](#8-configuration-reference)
9. [Hardware Support](#9-hardware-support)
10. [Fitness Scoring](#10-fitness-scoring)
11. [Output Artefacts](#11-output-artefacts)
12. [Deployment](#12-deployment)
13. [Component Interaction Diagrams](#13-component-interaction-diagrams)
14. [Design Invariants](#14-design-invariants)

---

## 1. Overview

OceanTune AI is an **autonomous LLM inference optimisation engine**. Given a Hugging Face `model_id` and GPU SKU, it discovers a high-fitness serving configuration (vLLM or SGLang), optionally engineers fused kernels, and sediments reusable recipes — without manual flag tuning.

Hyperloom (MIT) is used only as a **design reference**. OceanTune never imports or calls Hyperloom at runtime; portable techniques are reimplemented under `core/` and `agents/`.

### Design Goals

| Goal | How OceanTune achieves it |
|------|---------------------------|
| Zero-expert tuning | LLM agents propose; humans do not edit serving flags |
| Measurement-owned KEEP | `MeasurementGate` + `CriticAgent` — predicted gains never decide KEEP |
| Reproducible results | Fingerprinted configs, MongoDB, recipes, session breakdown, checkpoints |
| Hardware-aware | GPU profiles + capability detection + quantization scheme gating |
| Progressive depth | Enablement → Prelude → S1 → (S2↔S3)×macro → Sweep → S4 → CLOSE |
| Multi-framework | `FrameworkBackend` for **vLLM** and **SGLang** |
| Safe kernels | CorrectnessFirewall (abs/RMS + SNR) → microbench → shadow E2E |
| Cross-session learning | Recipe KB, experience constraints, attempt ledger, search policy |

### What OceanTune Optimises

```
Input:  model_id, gpu_type, framework ∈ {vllm, sglang}
Output: winning flags + optional Triton kernels + recipe + session_breakdown

Fitness modes (optimiser.primary_metric):
  throughput | p95_latency | ttft | tpot | prefill_heavy | decode_heavy | cost_aware
```

---

## 2. System Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            OceanTune AI                                  │
│                                                                          │
│  CLI: oceantune run --model … --gpu H200                                 │
│                           │                                              │
│                    ┌──────▼──────┐                                       │
│                    │ Controller  │  OceanTuneConfig + PolicyGate         │
│                    │   Agent     │  SessionCheckpoint + AttemptLedger    │
│                    └──────┬──────┘                                       │
│                           │                                              │
│   Enablement → Prelude → Stage1 → (Stage2 ↔ Stage3)×macro                │
│                → Sweep → Stage4 → Campaign/Ledger → CLOSE                │
│                           │                                              │
│                    ┌──────▼──────┐                                       │
│                    │   Report    │  recipe.yaml · launch.sh · report.md  │
│                    │ Generator   │  session_breakdown · checkpoint       │
│                    └─────────────┘                                       │
└──────────────────────────────────────────────────────────────────────────┘
          │                         │                      │
   ┌──────▼──────┐         ┌────────▼────────┐    ┌───────▼────────┐
   │  MongoDB    │         │ DO Inference API│    │ Docker serving │
   │ sessions ·  │         │ (LLM agents)    │    │ vLLM / SGLang  │
   │ configs ·   │         └─────────────────┘    └────────────────┘
   │ recipes ·   │
   │ benchmarks  │
   └─────────────┘
```

### Component Layers

```
┌─────────────────────────────────────────────────────────────────┐
│  Presentation                                                    │
│  oceantune.py (Click CLI) · show_results.py                      │
├─────────────────────────────────────────────────────────────────┤
│  Orchestration                                                   │
│  ControllerAgent · PolicyGate · SessionCheckpoint · MacroCycle   │
│  Enablement · Prelude · Coordinator · NodeClient                 │
├─────────────────────────────────────────────────────────────────┤
│  Agent Layer (LLM-powered)                                       │
│  Planner · Executor · Analyst · Critic · StrategyOptimizer       │
│  Profiler · BottleneckReasoning · Research                       │
│  KernelResearch · KernelGeneration · CorrectnessFirewall         │
│  KernelEvolution                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Measurement & Learning                                          │
│  MeasurementGate · PairedBench · Convergence · SearchPolicy      │
│  RecipeKB · ExperienceConstraints · AttemptLedger · DraftRegistry│
├─────────────────────────────────────────────────────────────────┤
│  Kernel / Fusion Plane                                           │
│  Fusion diagnose/patterns · SNR · NumericalPipeline              │
│  KernelHarness · KernelIntegration (shadow E2E)                  │
│  KernelCampaign · KernelLedger · WorkspacePolicy · ServingPatches│
├─────────────────────────────────────────────────────────────────┤
│  Serving Infrastructure                                          │
│  FrameworkBackend (vLLM|SGLang) · VLLMServer · BenchmarkEngine   │
│  GPUSlotAllocator · PortAllocator · Database · DOClient          │
├─────────────────────────────────────────────────────────────────┤
│  Analysis                                                        │
│  MetricsCollector · LogAnalyzer · NcuProfiler · RocprofProfiler  │
│  OperatorBench · RooflineAnalyzer · AttentionBench               │
├─────────────────────────────────────────────────────────────────┤
│  Configuration & Knowledge                                       │
│  OceanTuneConfig · SearchSpace · VLLMFlags · gpu_profiles        │
│  knowledge/{nvidia,amd,common} · QuantizationSchemes             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. End-to-End Pipeline

Canonical order executed by `ControllerAgent._run_async()`:

```
oceantune run
      │
      ▼
ControllerAgent._run_async()
      │
      ├── Session create / resume (SessionCheckpoint)
      ├── FrameworkBackend validate (vllm | sglang)
      ├── Serving patches → env gates (optional)
      ├── Quantization scheme → seed flag delta (optional)
      │
      ├──▶ ENABLEMENT — boot repair ladder
      │         └── winning safer flags (or fail session early)
      │
      ├──▶ PRELUDE — Recipe KB warm-replay plan + reject seed
      │         └── confidence-gated seed flags for Stage 1
      │
      ├──▶ STAGE 1 — Serving config search (Planner → Executor → Analyst)
      │         └── (winner_flags, fingerprint, stage1_fitness)
      │
      ├──▶ MACRO CYCLE (Stage 2 ↔ Stage 3) × macro_cycle_max
      │         ├── Stage 2: Inference strategy search + Critic/MeasurementGate
      │         ├── Stage 3: Profile → bottleneck → research → flag trials
      │         │            + fusion diagnose + kernel harness + kernel research
      │         └── Reloop if budget + headroom remain
      │
      ├──▶ SWEEP — Operating-point matrix (concurrency × context)
      │
      ├──▶ STAGE 4 — Kernel generate → SNR firewall → evolve  [optional]
      │         └── Shadow E2E rebench (MeasurementGate)
      │
      ├──▶ STAGE 4b — Fusion campaign + KernelLedger  [optional]
      │
      └──▶ CLOSE
                ├── ReportGenerator (recipe.yaml, launch.sh, report.md)
                ├── Recipe KB sedimentation
                ├── Session breakdown JSON
                └── Checkpoint phase=done
```

### Phase → Module Map

| Phase | Primary modules |
|-------|-----------------|
| Enablement | `core/enablement.py`, `PolicyGate` |
| Prelude | `core/prelude.py`, `warmstart_policy.py`, `recipe_kb.py` |
| Stage 1 | `PlannerAgent`, `ExecutorAgent`, `AnalystAgent`, `search_policy`, `attempt_ledger` |
| Stage 2 | `StrategyOptimizerAgent`, `MeasurementGate`, `CriticAgent`, `draft_registry` |
| Stage 3 | Profiler / NCU / Rocprof / Bottleneck / Research / fusion / harness |
| Sweep | `core/operating_point_sweep.py` |
| Stage 4 | Kernel* agents, `snr_contract`, `numerical_pipeline`, `kernel_integration` |
| Stage 4b | `kernel_campaign.py`, `kernel_ledger.py`, `workspace_policy.py` |
| CLOSE | `report_generator`, `recipe_kb.sediment`, `session_breakdown` |

---

## 4. Pipeline Phases & Stages

### 4.1 Enablement — Boot Repair

**Purpose:** Make a broken stack runnable before search. If the baseline (or first warm config) fails with OOM / startup timeout / attention / MLA errors, walk a deterministic repair ladder.

**Module:** `core/enablement.py`

```
baseline probe
    │ fail
    ▼
classify_boot_failure(error) → oom | startup_timeout | attention_backend | …
    │
    ▼
repair_ladder(base_flags, error, moe=, mla=)
    ├── lower util / FP8 KV / max_seqs
    ├── enforce_eager
    ├── FLASH_ATTN fallback
    └── MoE-safe eager path
    │
    ▼
MeasurementGate-style probe each repair until boot succeeds
    └── seed Stage 1 / Prelude with winning_flags
```

**Config:** `enablement_enabled`, `enablement_max_repairs`

---

### 4.2 Prelude — Warm Recipe + Reject Seed

**Purpose:** Before Stage 1 search, optionally seed from Recipe KB and pre-load failure denylists.

**Module:** `core/prelude.py` + `core/warmstart_policy.py`

```
RecipeKnowledgeBase.lookup(model, gpu, framework)
    │
    ▼
decide_warmstart(confidence ≥ prelude_min_confidence, fitness floor, trial budget)
    │ accept
    ▼
build_prelude_plan → replay_flags + reject_fingerprints + reject_pitfalls
    │
    ▼
accept_warm_replay (retention vs claimed fitness)
    └── seed_flags → merged into Stage 1 enablement seed
```

**Config:** `prelude_enabled`, `prelude_min_confidence`, `warmstart_max_trials`

---

### 4.3 Stage 1 — Serving Config Search

**Purpose:** Find the best low-level serving flags (parallelism, memory util, KV dtype, attention, scheduler, …) via iterative LLM-guided search.

**Agents:** `PlannerAgent` → `ExecutorAgent` → `AnalystAgent`  
**Backend:** `FrameworkBackend` normalizes flags; Executor launches **vLLM** or **SGLang** via `VLLMServer` (optional `launch_override_cli`).

#### Stage 1 Flow

```
Iteration 0: baseline (defaults)
Iteration 1..N:
  ├── warm-start seeds (enablement + prelude + Recipe KB + GPU seeds)
  ├── or PlannerAgent.propose_next(
  │         history, analyst_eval, recipe_context,
  │         experience_constraints + attempt_ledger denylist,
  │         search_policy EXPLOIT|DIVERSIFY hint
  │     )
  ├── PolicyGate.sanitize_flags("stage1", …)
  ├── ExecutorAgent → Docker serve → BenchmarkEngine ramp → MetricsCollector
  └── keep if fitness improves
Returns: (winner_flags, fingerprint, stage1_fitness)
```

#### Planner context sources

- `models.yaml` architecture metadata  
- `gpu_profiles.yaml` legality  
- Recipe KB cascade warm-start  
- Experience constraints + attempt ledger  
- Vendor knowledge packs (`knowledge/`)  
- Search policy (stall → diversify flag families)  
- Speculative draft hint (`draft_registry` / `configs/draft_models.yaml`)

#### Fingerprint

```python
fingerprint = SHA256(sorted(flags.items())).hexdigest()[:12]
```

Prevents duplicate benchmarks within and across sessions.

---

### 4.4 Macro-Cycle — Stage 2 ↔ Stage 3

**Purpose:** Budgeted reloop between strategy search and profiling so Stage 2 can consume Stage 3 bottleneck / fusion hints.

**Module:** `core/macro_cycle.py`

```
macro_cycle = 0
cycle_flags = stage1_winner
loop:
  Stage 2(cycle_flags, profiler_hints if macro>0)
  Stage 3(merged_flags)
  decision = should_reloop_stage2_3(
      gain headroom, macro_cycle_max,
      session_remaining_sec from SessionCheckpoint,
      macro_cycle_min_remaining_sec
  )
  if not decision.reloop: break
  cycle_flags = stage3_flags
  macro_cycle += 1
```

**Config:** `macro_cycle_enabled`, `macro_cycle_max`, `macro_cycle_min_remaining_sec`, `session_max_minutes`

---

### 4.5 Stage 2 — Inference Strategy Search

**Purpose:** Layer serving strategies on the Stage 1 winner (KV dtype, attention backend, chunked prefill, prefix cache, speculative decode, scheduler knobs).

| Strategy | Typical flag | Notes |
|----------|--------------|-------|
| FP8 KV | `kv_cache_dtype=fp8` | Capacity / bandwidth |
| FlashInfer / FA | `attention_backend` | GPU-gated |
| Chunked prefill | `enable_chunked_prefill` | Prefill/decode balance |
| Prefix caching | `enable_prefix_caching` | Shared prefixes |
| Speculative decode | `speculative_model` + tokens | **Draft registry gated** |
| Scheduler | `num_scheduler_steps`, … | Throughput vs latency |

**KEEP path:** benchmark → `MeasurementGate.decide` → `CriticAgent` sign-off.  
**PolicyGate:** Stage 2 may only mutate strategy flag keys (TP frozen).

---

### 4.6 Stage 3 — Deep Profiling & Bottleneck Reasoning

**Purpose:** Explain *why* the winner is slow and try validated flag remedies; prepare Stage 4 handoff.

```
3a   ProfilerAgent — PyTorch profiler trace
3a′  KernelHarness — YAML phase cases (prefill/decode shapes)
3b   NcuProfiler / RocprofProfiler — hardware counters + roofline microbench
3c   BottleneckReasoningAgent — classify compute / memory / launch / …
3c′  Fusion diagnose — match residual+RMSNorm, SwiGLU, QK+RoPE, …
3d   ResearchAgent — ranked flag recommendations (+ knowledge pack)
3e   Flag trials — MeasurementGate + Critic per recommendation
3f   KernelResearchAgent — if kernel path warranted
```

Returns: research report, bottleneck, kernel research, stage3 fitness, applied recs, updated flags, fusion diagnosis.

---

### 4.7 Operating-Point Sweep

**Purpose:** After the macro winner, sweep concurrency × sequence-length operating points to find the real peak (not a single-ramp artifact).

**Module:** `core/operating_point_sweep.py`  
**PolicyGate action:** `sweep / operating_point_sweep`

---

### 4.8 Stage 4 — Autonomous Kernel Engineering

**Activation:** `stage4_enabled: true` and Stage 3 produced a bottleneck / kernel research handoff.

```
KernelGenerationAgent
        │
        ▼
CorrectnessFirewallAgent
  · abs / RMS thresholds per op
  · SNR estimate gate (≥ ~30 dB) via numerical_pipeline
        │ pass
        ▼
KernelEvolutionAgent — microbench keep/revert (SNR + speedup contract)
        │
        ▼
KernelIntegrationLayer.run_e2e_rebench  [stage4_e2e_enabled]
  · shadow package + env-gated hooks
  · Docker serve + MeasurementGate vs incumbent fitness
```

---

### 4.9 Stage 4b — Fusion Campaign & Ledger

**Purpose:** Bind evolved kernels to fusion patterns, record provenance, sediment lessons.

**Modules:** `core/kernel_campaign.py`, `core/kernel_ledger.py`, `core/workspace_policy.py`

```
run_fusion_campaign(patterns, kernels, …)
    ├── per-pattern research / bind / optional E2E
    └── KernelLedger.record(decision, snr_db, speedup, identity)
         └── lessons_for_recipe → Recipe what_worked on CLOSE
```

**Config:** `stage4_campaign_enabled`

---

### 4.10 CLOSE — Sedimentation & Artefacts

```
ReportGenerator → storage/results/
  recipe_*.yaml · launch_*.sh · report_*.md

RecipeKnowledgeBase.sediment(
  best_flags, fitness, lessons, pitfalls, what_worked/failed
)

build_session_breakdown → session_breakdown_<id>.json
  (enablement, prelude, macro, sweep, campaign, policy_denials, …)

SessionCheckpoint.phase = done
```

---

## 5. Core Components

### 5.1 FrameworkBackend & VLLMServer

| Piece | Role |
|-------|------|
| `FrameworkBackend` | Abstract launch/normalize for `vllm` \| `sglang` |
| `VLLMBackend` | `VLLMFlags.to_vllm_args` → Docker `vllm serve` |
| `SGLangBackend` | Remap OceanTune flags → `python -m sglang.launch_server` |
| `VLLMServer` | Docker lifecycle; optional `launch_override_cli` for SGLang |
| `ExecutorAgent` | Acquires GPU/port, builds backend launch, runs BenchmarkEngine |

### 5.2 Measurement Stack

| Module | Role |
|--------|------|
| `MeasurementGate` | Single KEEP/REVERT API from measured fitness |
| `CriticAgent` | Mission-grounded sign-off on KEEP |
| `measurement_convergence` | Reject high-spread / monotonic-climb ramps |
| `paired_bench` | Multi-probe sign agreement before KEEP |
| `snr_contract` / `numerical_pipeline` | Kernel numerical + statistical KEEP |

### 5.3 Learning Stack

| Module | Role |
|--------|------|
| `RecipeKnowledgeBase` | Cascade lookup + CLOSE sediment (Mongo `recipes`) |
| `experience_constraints` | Negative priors for Planner / Stage 2 |
| `AttemptLedger` | Typed failure rows → Planner denylist text |
| `search_policy` | EXPLOIT vs DIVERSIFY on stall |
| `draft_registry` | Speculative draft pairs (incl. MoE targets) |
| `knowledge_pack` | NVIDIA / AMD methodology levers |

### 5.4 Kernel / Fusion Stack

| Module | Role |
|--------|------|
| `core/fusion/` | Pattern library + diagnose from profiler shares |
| `kernel_harness` | YAML phase-attributed micro cases |
| `kernel_integration` | Shadow hooks + E2E rebench |
| `serving_patches` | Versioned manifests + env gates |
| `kernel_campaign` / `kernel_ledger` | Stage 4b provenance |
| `workspace_policy` | Isolated writable paths per session |

### 5.5 BenchmarkEngine & MetricsCollector

- Concurrency ramp over `benchmark.concurrency_levels`  
- × each `context_configs` pair `(input_len, output_len)`  
- Fitness via `MetricsCollector` (see §10)  
- Failures (OOM, timeout) → fitness `0` + attempt ledger / pitfalls

### 5.6 PolicyGate

Stage-allowed actions and flag keys:

| Stage | May mutate serving flags? | Example actions |
|-------|---------------------------|-----------------|
| enablement / prelude | Yes (boot set) | `baseline_probe`, `warm_replay` |
| stage1 | Yes (full Stage 1 set) | `propose_flags`, `benchmark` |
| stage2 | Strategy keys only | `propose_strategy` |
| stage3 | Limited set | `profile`, `flag_trial`, `harness` |
| stage4 | **Frozen** | `generate_kernel`, `shadow_e2e` |
| sweep / close | Frozen | `operating_point_sweep`, `sediment_recipe` |

### 5.7 SessionCheckpoint

Persists under `storage/sessions/<id>/checkpoint.json`:

- `phase` cursor for `--resume` / `resume_session_id` / `OCEANTUNE_RESUME_SESSION`  
- winner flags, fitnesses, macro cycle  
- `RoundBudget.session_max_sec` → feeds macro `session_remaining_sec`

### 5.8 Database (MongoDB)

```
oceantune
├── sessions
├── configs          (fingerprinted candidates)
├── benchmark_runs
├── recipes          (Recipe KB)
├── kernel_runs
├── kernel_benchmark_runs / kernel_metadata
└── nodes            (multi-node heartbeats)
```

### 5.9 Resource Allocators

- **GPUSlotAllocator** — contiguous TP slots; `CUDA_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`  
- **PortAllocator** — port pool for parallel Docker serves  

---

## 6. Agent System

### 6.1 DOClient

All LLM calls use DigitalOcean Serverless Inference (OpenAI-compatible). Without `DO_INFERENCE_KEY`, agents fall back to heuristic / evolutionary mutations.

### 6.2 Responsibility Matrix

| Agent / Module | Phase | Input | Output | LLM? |
|----------------|-------|-------|--------|------|
| Enablement | Boot | Error class | Safer flags | No |
| Prelude | Warm | Recipe | Seed + rejects | No |
| PlannerAgent | S1 | History + constraints | Flag proposal | Yes |
| ExecutorAgent | S1–3 | Flags | Benchmark + fitness | No |
| AnalystAgent | S1 | Metrics | Bottleneck note | Yes |
| StrategyOptimizer | S2 | Winner flags | Strategy delta | Yes |
| CriticAgent | S2/S3 KEEP | GateDecision | KEEP/REVERT | Optional |
| ProfilerAgent | S3 | Winner | Trace | No |
| BottleneckReasoning | S3 | Trace + counters | Classification | Yes |
| ResearchAgent | S3 | Bottleneck | Flag recs | Yes |
| KernelResearch | S3/S4 | Op bottleneck | Research report | Yes |
| KernelGeneration | S4 | Spec | Triton source | Yes |
| CorrectnessFirewall | S4 | Kernel | Pass/fail + SNR | No |
| KernelEvolution | S4 | Kernel | Keep/revert tree | Yes |
| ReportGenerator | CLOSE | Session | Artefacts | No |

---

## 7. Data Model

### Key types

```
OceanTuneConfig
VLLMFlags / BackendLaunchSpec
GateDecision / ConvergenceAssessment / PairedBenchResult
Recipe / Lesson / Pitfall
EnablementResult / PreludePlan / PreludeResult
MacroCycleState
IntegrationPlan / ShadowHook
CampaignResult / LedgerEntry
SessionCheckpoint / AttemptRow
CorrectnessReport (incl. snr_db_estimate)
EnrichedMetrics / fitness_score
```

### VLLMFlags (representative)

Parallelism · `gpu_memory_utilization` · `max_num_seqs` · `max_num_batched_tokens` ·  
`block_size` · `kv_cache_dtype` · `dtype` · `quantization` · `attention_backend` ·  
`enable_prefix_caching` · `enable_chunked_prefill` · speculative fields · scheduler knobs ·  
`enforce_eager` · MoE-related options (search-space gated)

SGLang receives a **remapped subset** via `SGLangBackend.normalize_flags`.

---

## 8. Configuration Reference

### 8.1 Hierarchy

1. `configs/oceantune.yaml`  
2. Env vars (`OCEANTUNE_*`, `MONGO_URI`, `DO_*`, `HF_TOKEN`, `VLLM_IMAGE`)  
3. Dataclass defaults in `core/config.py`

### 8.2 Required Environment

| Variable | Purpose |
|----------|---------|
| `MONGO_URI` | MongoDB connection |
| `HF_TOKEN` | Model download (private models) |
| `DO_INFERENCE_KEY` | LLM agents (optional but recommended) |
| `DO_INFERENCE_ENDPOINT` | Inference API base URL |

### 8.3 Pipeline knobs (oceantune.yaml)

```yaml
framework: "vllm"                 # vllm | sglang
framework_version: "0.6.0"

enablement_enabled: true
enablement_max_repairs: 4

prelude_enabled: true
prelude_min_confidence: 0.7
warmstart_max_trials: 3

macro_cycle_enabled: true
macro_cycle_max: 2
macro_cycle_min_remaining_sec: 1800

session_max_minutes: 0            # 0 = unlimited
resume_session_id: ""

search_stall_limit: 2
convergence_max_spread_pct: 15.0
snr_threshold_db: 30.0
serving_patches_enabled: true
quantization_scheme: "none"       # none|fp8|fp8_kv|awq|gptq|nvfp4|bitsandbytes

attention_e2e_enabled: false

stage4_enabled: true
stage4_iterations: 5
stage4_e2e_enabled: true
stage4_campaign_enabled: true

optimiser:
  generations: 15
  primary_metric: "throughput"
  # throughput | p95_latency | ttft | tpot
  # prefill_heavy | decode_heavy | cost_aware

context_configs:
  - [1024, 1024]
  - [1024, 4096]
  - [512, 64]      # decode-heavy
```

### 8.4 Related config files

| File | Role |
|------|------|
| `configs/search_space.yaml` | Stage 1 flag space |
| `configs/stage2_search_space.yaml` | Strategy space |
| `configs/gpu_profiles.yaml` | Per-SKU Docker / legality |
| `configs/models.yaml` | Architecture aliases (MoE, MLA, NVFP4) |
| `configs/draft_models.yaml` | Speculative draft pairs |
| `configs/kernel_harness_cases.yaml` | Stage 3a′ harness |
| `configs/kernel_registry/` | Known kernel metadata |
| `data/serving_patches/` | Versioned patch manifests |
| `knowledge/` | Vendor lever packs |

---

## 9. Hardware Support

### 9.1 Supported GPU SKUs

| GPU | Vendor | Notes |
|-----|--------|-------|
| H100 | NVIDIA | FP8 native |
| H200 | NVIDIA | Large HBM3e |
| B300 | NVIDIA | Blackwell; **NVFP4** scheme gated |
| MI300X / MI325X / MI350X | AMD | AITER levers via knowledge + profiles |

### 9.2 Vendor Behaviour

```
NVIDIA: CUDA_VISIBLE_DEVICES · --gpus device=… · ncu counters · Triton/TMA
AMD:    ROCR_VISIBLE_DEVICES · /dev/kfd+/dev/dri · rocprof/omniperf · LDS-aware Triton
```

Quantization schemes are GPU-gated in `core/quantization_schemes.py` (e.g. `nvfp4` → B300 only).

---

## 10. Fitness Scoring

### 10.1 Formula (default `throughput`)

```
peak_throughput, p95_latency, mean_ttft, mean_tpot  ← from ramp

throughput_score = min(peak_throughput / 5000, 1.0)
latency_score    = 1 - min(p95_latency / 10000, 1.0)
ttft_score       = 1 - min(mean_ttft / 2000, 1.0)
tpot_score       = 1 - min(mean_tpot / ref, 1.0)

fitness = Σ (weight_i × score_i)   # weights depend on primary_metric
```

### 10.2 Primary metric weights

| Mode | Emphasis |
|------|----------|
| `throughput` | Peak tok/s |
| `p95_latency` / `ttft` / `tpot` | Latency-first |
| `prefill_heavy` | Throughput + TTFT |
| `decode_heavy` | Throughput + TPOT (short decode contexts helped by `[512,64]`) |
| `cost_aware` | Throughput per GB VRAM (DO cost proxy) |

**Penalties:** OOM / startup failure → `0`; high error rate → multiplicative penalty.

### 10.3 Convergence & paired fidelity

- `assess_convergence` — discard warmup; reject high spread or strict monotonic climb  
- `evaluate_paired_probes` — require agreeing positive deltas before KEEP helpers  

---

## 11. Output Artefacts

| Artefact | Location | Contents |
|----------|----------|----------|
| YAML recipe | `storage/results/recipe_*.yaml` | Winner flags + metadata |
| Launch script | `storage/results/launch_*.sh` | Docker run command |
| Markdown report | `storage/results/report_*.md` | Stage-by-stage summary |
| Session breakdown | `session_breakdown_<id>.json` | Full phase extras |
| Checkpoint | `storage/sessions/<id>/checkpoint.json` | Resume cursor |
| Attempts | `storage/sessions/<id>/attempts.jsonl` | Failure ledger |
| Kernel ledger | under workspace / storage | Stage 4b provenance |
| Mongo recipes | `recipes` collection | Cross-session warm-start |

---

## 12. Deployment

### 12.1 Prerequisites

- Docker (GPU-enabled)  
- MongoDB (`MONGO_URI`)  
- Python 3.11+ / project `.venv`  
- Optional: `ncu` / `rocprof`, PyTorch + Triton (Stage 4)  
- Optional: `DO_INFERENCE_KEY` for LLM-guided search  

### 12.2 Typical commands

```bash
# Full pipeline
oceantune run --model deepseek-ai/DeepSeek-V3.2 --gpu H200

# Framework override
OCEANTUNE_FRAMEWORK=sglang oceantune run …

# Resume interrupted session
OCEANTUNE_RESUME_SESSION=<session_id> oceantune run …

# Wall-clock budget (minutes)
OCEANTUNE_SESSION_MAX_MINUTES=180 oceantune run …
```

### 12.3 Single-node vs multi-node

- **Single-node:** Controller + local GPUSlotAllocator / PortAllocator  
- **Multi-node:** Coordinator dispatches configs to NodeClient workers  

---

## 13. Component Interaction Diagrams

### 13.1 Full session sequence

```
CLI → Controller
        ├─ Checkpoint / FrameworkBackend / Patches / Quant scheme
        ├─ Enablement ──probe──▶ VLLMServer/SGLang
        ├─ Prelude ──lookup──▶ RecipeKB
        ├─ Stage1 loop
        │     Planner → Executor → BenchmarkEngine → MetricsCollector → Analyst
        │     SearchPolicy / AttemptLedger / ExperienceConstraints
        ├─ Macro
        │     Stage2 → MeasurementGate → Critic
        │     Stage3 → Profiler → Fusion → Research → FlagTrials → KernelResearch
        ├─ Sweep
        ├─ Stage4 → Generate → Firewall(SNR) → Evolve → E2E rebench
        ├─ Campaign → Ledger
        └─ CLOSE → Report + Recipe sediment + Breakdown
```

### 13.2 KEEP decision path (Stage 2/3/4)

```
Agent proposes
      │
      ▼
Executor / microbench / E2E measures fitness
      │
      ▼
MeasurementGate.decide(incumbent, candidate)
      │
      ▼
CriticAgent (optional LLM / hard rules)
      │
      ▼
KEEP → update incumbent   |   REVERT → discard
```

### 13.3 Stage 4 kernel path

```
Bottleneck / Fusion pattern
      │
      ▼
Generate Triton → CorrectnessFirewall (abs/RMS + SNR)
      │
      ▼
Evolution microbench (snr_contract.evaluate_keep)
      │
      ▼
ShadowDocker E2E (MeasurementGate vs serving incumbent)
      │
      ▼
Campaign bind + KernelLedger → Recipe lessons
```

---

## 14. Design Invariants

1. **Agents propose; measurements decide.** No KEEP from LLM-claimed speedups alone.  
2. **Hyperloom is reference-only** — no runtime dependency.  
3. **PolicyGate bounds power** per phase (Stage 4 freezes serving flags).  
4. **Fingerprints dedupe** expensive benchmarks.  
5. **Recipes sediment only measured winners** (+ typed failures/pitfalls).  
6. **Kernels need numerical + performance + (optional) E2E** before trust.  
7. **Session checkpoints** make long runs resumable under wall-clock budgets.

### Related docs

- [`docs/research/hyperloom_oceantune_integration.md`](./research/hyperloom_oceantune_integration.md)  
- [`docs/research/oceantune_capability_inventory.md`](./research/oceantune_capability_inventory.md)  

---

*End of Technical Architecture Document v6.0*
