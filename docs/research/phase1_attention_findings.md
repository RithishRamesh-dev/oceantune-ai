# Phase 1 — Attention Kernel Research

**Status:** Research complete (awaiting implementation approval)  
**OceanTune baseline:** [phase 0 plan](../../.cursor/plans/phase_0_architecture_62a57e35.plan.md), `core/flag_merge.py`, `configs/stage2_search_space.yaml` (`attention_backend`)

---

## Section 1 — Research Summary

### Key papers and innovations

| Work | Innovation | OceanTune relevance |
|------|------------|---------------------|
| FlashAttention (Dao et al.) | IO-aware tiling; softmax in SRAM | Default vLLM path via `FLASH_ATTN` |
| FlashAttention-2 | Better parallelism, less non-matmul work | Same backend family; wins on long seq |
| FlashAttention-3 | FP8 forward, async Tensor Cores (Hopper+) | Needs FA3-capable stack + dtype flags |
| FlashInfer | Paged KV + batch decode APIs, GQA/MoE | `FLASHINFER` in Stage 2 search space |
| FlashMLA (DeepSeek) | MLA latent cache, fused decode | `block_size=1` + model-specific paths |
| PagedAttention (vLLM) | Block-table KV; continuous batching | `block_size`, `max_num_seqs` Stage 1 |
| Flash-Decoding | Split KV across SMs for small batch decode | Not exposed as flag — workload gap |
| TensorRT-LLM / SGLang | Fused FMHA + custom schedulers | Reference targets for E2E fitness |

### Key projects

- **vLLM:** `attention_backend` enum — primary OceanTune lever today  
- **FlashInfer:** decode/prefill wrappers, good for variable-length + GQA  
- **SGLang:** RadixAttention + fused kernels — benchmark competitor, not integrated  
- **TensorRT-LLM:** production FMHA plugins — informs ceiling, not directly tunable in OceanTune

### State of the art (2026 serving)

Production stacks combine **paged KV** + **fused attention** + **FP8 KV** + **CUDA graphs** on decode. Attention is rarely the only bottleneck at high concurrency; **memory bandwidth** (KV) and **scheduling** dominate. Custom attention wins appear at **low batch decode** and **long context** where Flash-Decoding-style splitting matters.

---

## Section 2 — Kernel Architecture Analysis

### FlashAttention-2/3 (conceptual)

| Dimension | Typical structure |
|-----------|-------------------|
| Thread/warp | Warps own tile of Q; K/V streamed through shared mem |
| Shared memory | Q/K/V tiles; softmax stats per row |
| Registers | Per-thread accumulators for softmax + output |
| Tensor Cores | MMA on QK^T and PV (Hopper: FP8 in FA3) |
| Occupancy | Limited by smem per block; head_dim drives tile shape |
| Memory hierarchy | HBM ← L2 ← smem; goal is minimize HBM rereads |

### FlashInfer

- Separates **prefill** vs **decode** APIs; paged KV indices passed explicitly  
- Better when batch has **mixed lengths** and **GQA** (shared KV heads)  
- OceanTune: select via `--attention-backend FLASHINFER`

### PagedAttention (vLLM)

- KV stored in fixed **blocks** (`block_size`); logical seq → block table  
- Not a separate backend — infrastructure under all backends  
- OceanTune tunes `block_size`, `kv_cache_dtype`, `max_num_seqs`

### Bottleneck shift by phase

- **Prefill:** compute-bound (large QK^T) → FA2/FA3, Tensor Cores  
- **Decode:** memory-bound (KV read per token) → FP8 KV, Flash-Decoding, MLA  
- **High concurrency:** scheduling + KV capacity → batching flags, not kernel swap alone

---

## Section 3 — Remaining Bottlenecks

| Limitation | Why it persists |
|------------|-----------------|
| Decode at batch=1 | KV bandwidth ceiling; FA2 does not split seq across SMs by default |
| Long context | KV size grows linearly; kernel choice secondary to dtype/blocking |
| GQA / MLA | Requires specialized kernels (FlashInfer, FlashMLA, AITER on AMD) |
| CUDA graphs vs eager | Graphs hide kernel names in Stage 3 traces |
| Cross-backend fairness | OceanTune E2E fitness conflates attention with scheduler/KV |
| FP8 accuracy | Model-dependent; needs correctness checks beyond fitness |

### What could outperform current approaches

- **Fused decode** with paged FP8 KV + split-KV (Flash-Decoding lineage)  
- **FA3 FP8** on Hopper when model supports it  
- **FlashMLA** for DeepSeek-class MLA models  
- **Custom Triton** only when trace shows attention >40% and backend swap exhausted (Stage 4)

---

## Section 4 — OceanTune Gap Analysis

| Area | OceanTune Today | SOTA | Gap |
|------|-----------------|------|-----|
| Backend selection | `attention_backend` enum | Per-phase backend (prefill vs decode) | Single flag for both phases |
| Benchmark | E2E `vllm bench serve` | Prefill-only + decode-only microbench | No split metrics |
| Profiling | Trace % attention | NCU on real kernel name | Was hardcoded `--op attention` (fixed: `infer_microbench_op`) |
| Registry | None | Kernel metadata per backend/version | `kernel_metadata` stub only |
| MLA models | `block_size=1` validator | FlashMLA / AITER MLA | Flag-only, no MLA-specific benchmark matrix |
| Validation | Fitness + optional Stage 4 microbench | Backend A/B at fixed concurrency | No isolated attention bench in loop |

---

## Section 5 — Improvement Opportunities

| Rank | Opportunity | Impact | Complexity | Risk |
|------|-------------|--------|------------|------|
| 1 | Attention benchmark matrix in `kernel_benchmark_runs` | High | Medium | Low |
| 2 | Extend Stage 2 sweep: FLASHINFER + ROCM_FLASH per model arch | High | Low | Low |
| 3 | Prefill vs decode context configs in `oceantune.yaml` | High | Low | Low |
| 4 | `KernelRegistry` entries for FA2/FA3/FlashInfer | Medium | Medium | Low |
| 5 | Stage 4 attention Triton only after backend sweep | Medium | High | High |

---

## Section 6 — OceanTune Design Proposal

### New search dimensions

```yaml
# Proposed: configs/attention_search_space.yaml (Phase 1 implementation)
attention_backend: [FLASH_ATTN, FLASHINFER, ROCM_FLASH]  # existing
attention_prefill_backend: [...]   # future vLLM split APIs
attention_decode_backend: [...]
kv_cache_dtype: [auto, fp8, fp8_e4m3]  # interacts with attention BW
block_size: [1, 8, 16, 32]            # MLA vs dense
```

### New components

- **`AttentionBenchmarkSuite`** — wraps `OperatorBench` + optional `vllm bench` short runs  
- **`KernelRegistry`** — maps `FLASHINFER` → {min vLLM version, GPU arch, GQA support}  
- **`CapabilityDetector`** — `flashinfer_available()`, `fa3_fp8_available(gpu_type)`

### Agent changes

| Agent | Change |
|-------|--------|
| `StrategyOptimizerAgent` | Prioritize `attention_backend` when Analyst/Profiler bottleneck is attention |
| `ProfilerAgent` | Tag trace with prefill vs decode phases (when vLLM exposes markers) |
| `ResearchAgent` | Recommend backend from registry + model arch (GQA, MLA) |
| `ControllerAgent` | Persist attention microbench to `kernel_benchmark_runs` (done in Stage 3 roofline path) |

### Benchmarking requirements

- Matrix: `(backend × context_config × concurrency)` with **decode-heavy** row: `[512, 64]`  
- Metrics: `output_tokens_per_sec`, `mean_tpot_ms`, `mean_ttft_ms`, isolated `attention` microbench TFLOPS/BW  
- Store in `kernel_benchmark_runs` with `op_type=attention`

---

## Section 7 — Implementation Workflow

### Step 7: Attention benchmark suite

- **Objective:** Run isolated + short E2E attention comparisons per backend  
- **Files:** `core/attention_bench.py`, `configs/attention_benchmark_matrix.yaml`  
- **Components:** `KernelBenchmarkEngine`, `Database.insert_kernel_benchmark_run`  
- **Dependencies:** Step 6 schema (done), `infer_microbench_op`  
- **Validation:** 3 backends produce rows in `kernel_benchmark_runs` on H200  
- **Outcome:** Comparable attention numbers independent of full ramp cost

### Step 8: Kernel registry (attention slice)

- **Objective:** YAML + Mongo `kernel_metadata` for FA2, FA3, FlashInfer  
- **Files:** `configs/kernel_registry/attention.yaml`, `core/kernel_registry.py`  
- **Validation:** `CapabilityDetector` skips invalid backends for GPU/model  
- **Outcome:** StrategyOptimizer never proposes FLASHINFER on unsupported stacks

### Step 9: Context matrix extension

- **Objective:** Add decode-heavy `context_configs` to `oceantune.yaml`  
- **Files:** `configs/oceantune.yaml`, docs  
- **Validation:** Stage 1/2 fitness differs measurably vs chat-only configs  
- **Outcome:** Attention/KV bottlenecks visible in Analyst curves

### Step 10: StrategyOptimizer attention policy

- **Objective:** Category sweep always includes `attention_backend` when `gemm_pct + attention_pct > 50`  
- **Files:** `agents/strategy_optimizer.py`  
- **Validation:** Logs show attention category trial for attention-bound sessions  
- **Outcome:** Backend sweep is systematic, not LLM-luck-dependent

### Step 11: Report + recipe attention section

- **Objective:** Report lists backend trials from `kernel_benchmark_runs` + Stage 2 delta  
- **Files:** `core/report_generator.py`  
- **Validation:** `report_*.md` includes attention benchmark table  
- **Outcome:** Operators see evidence for `attention_backend` choice

---

## Section 8 — Exit Criteria

Phase 1 implementation is complete when:

- [x] `kernel_benchmark_runs` populated after Stage 2 via `AttentionBenchmarkSuite.run_microbench_suite()`  
- [x] Stage 2 queues `attention_backend` trials when GQA + VRAM ≥ 0.85 or trace attention+gemm > 50%  
- [x] Decode-heavy context `[512, 64]` in `configs/oceantune.yaml`  
- [x] `kernel_metadata` seeded from `configs/kernel_registry/attention.yaml` (6 implementations)  
- [ ] E2E fitness improvement ≥1% — validated on GPU runs (not CI)

**Implemented modules:** `core/attention_bench.py`, `core/kernel_benchmark_engine.py`, `core/kernel_registry.py`, `core/capability_detector.py`

---

## STOP

Reply **Continue** for **Phase 2 — KV Cache Kernel Research**.
