# OceanTune AI

Automated vLLM inference optimisation system. Finds the best combination of vLLM flags and kernel settings for a given model, GPU, and context-length profile — without manual tuning.

Benchmarks run **in parallel** across GPU Droplets. All results are stored in **MongoDB** for cross-session analytics and deduplication. A **4-agent LLM pipeline** (powered by DigitalOcean Serverless Inference) replaces mechanical search with genuine hardware reasoning.

---

## How it works

OceanTune runs a two-stage closed-loop optimisation pipeline:

**Stage 1 — vLLM Config Search**
1. `SearchSpace` samples `population_size × generations` candidate `VLLMFlags` combinations
2. `PlannerAgent` validates hardware compatibility and LLM-ranks the candidates; skips any fingerprints already benchmarked in prior sessions (cross-session deduplication via MongoDB)
3. Ranked configs are inserted into MongoDB as `pending` documents
4. `Coordinator` polls MongoDB, checks node capacity, and dispatches configs in parallel to GPU Droplet Node Servers
5. Each `ExecutorAgent` acquires a GPU slot (`CUDA_VISIBLE_DEVICES`) and port, starts a vLLM instance, runs the full benchmark ramp, computes fitness via `MetricsCollector`, and writes the result to MongoDB
6. `AnalystAgent` reads `benchmark_runs`, runs aggregation pipelines, picks the winner, and produces an LLM explanation

**Stage 2 — Kernel-Level Search**

7. `ControllerAgent` passes the Stage 1 winner config as the Stage 2 baseline
8. `KernelOptimizerAgent` iteratively proposes low-level kernel/attention/NCCL flag combinations via DO Serverless Inference; each proposal is benchmarked and the LLM learns from the result history
9. `ReportGenerator` emits a YAML recipe, ready-to-run shell script, and Markdown summary

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         oceantune.py run                             │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        ControllerAgent                               │
│                   agents/controller_agent.py                         │
│                                                                      │
│  1. calls _stage1()  →  returns winner_flags                         │
│  2. passes winner_flags as baseline to _stage2()                     │
│  3. calls _generate_report() with best_kernel merged over winner     │
└──────────────────┬────────────────────────────┬─────────────────────┘
                   │                            │
       ┌───────────▼────────────┐   ┌───────────▼────────────┐
       │       STAGE 1          │   │       STAGE 2          │
       │   vLLM Config Search   │   │  Kernel-Level Search   │
       └───────────┬────────────┘   └───────────┬────────────┘
                   │                            │
       ┌───────────▼────────────┐   ┌───────────▼────────────┐
       │     SearchSpace        │   │  KernelOptimizerAgent  │
       │  core/search_space.py  │   │ agents/kernel_          │
       │                        │   │   optimizer.py          │
       │  samples N candidates  │   │                        │
       │  (VLLMFlags objects)   │   │ • vendor-filtered      │
       └───────────┬────────────┘   │   kernel search space  │
                   │                │ • LLM proposes next    │
       ┌───────────▼────────────┐   │   kernel config        │
       │      PlannerAgent      │   │ • benchmarks proposal  │
       │   agents/planner.py    │   │ • learns from history  │
       │                        │   │ • 10 iterations        │
       │ • ConfigValidator      │   └───────────┬────────────┘
       │   (hardware check)     │               │
       │ • skip seen            │   ┌───────────▼────────────┐
       │   fingerprints         │   │   ReportGenerator      │
       │ • LLM-rank remainder   │   │ core/report_generator  │
       └───────────┬────────────┘   │                        │
                   │                │ • YAML recipe          │
       ┌───────────▼────────────┐   │ • Shell script         │
       │        MongoDB         │   │ • Markdown report      │
       │      core/db.py        │   └────────────────────────┘
       │                        │
       │  configs → "pending"   │
       │  (priority-ordered)    │
       └───────────┬────────────┘
                   │
       ┌───────────▼────────────┐
       │      Coordinator       │
       │  core/coordinator.py   │
       │                        │
       │ • poll pending configs │
       │ • check node capacity  │
       │ • dispatch via HTTP    │
       │ • retry on failure     │
       │   (max_retries)        │
       └───────────┬────────────┘
                   │  HTTP  (core/node_client.py)
       ┌───────────▼────────────────────────────────────────┐
       │                GPU Droplet Node Server              │
       │                 node/node_server.py                 │
       │                                                     │
       │  POST /jobs  ──►  NodeWorker                        │
       │                   node/node_worker.py               │
       │                          │                          │
       │               ┌──────────▼──────────┐              │
       │               │    ExecutorAgent     │              │
       │               │  agents/executor.py  │              │
       │               │                      │              │
       │               │  GPUSlotAllocator    │              │
       │               │  core/gpu_allocator  │              │
       │               │  (CUDA_VISIBLE_      │              │
       │               │   DEVICES per slot)  │              │
       │               │                      │              │
       │               │  PortAllocator       │              │
       │               │  core/port_allocator │              │
       │               │  (pool 8000–8099)    │              │
       │               │          │           │              │
       │               │    VLLMServer        │              │
       │               │  core/vllm_server.py │              │
       │               │  (GPU-profile env)   │              │
       │               │          │           │              │
       │               │   BenchmarkEngine    │              │
       │               │  core/benchmark_     │              │
       │               │    runner.py         │              │
       │               │  (22 regex patterns) │              │
       │               │          │           │              │
       │               │  LogAnalyzer         │              │
       │               │  core/log_analyzer   │              │
       │               │  (OOM / crash detect)│              │
       │               │          │           │              │
       │               │  MetricsCollector    │              │
       │               │  core/metrics_       │              │
       │               │    collector.py      │              │
       │               │  (fitness scoring)   │              │
       │               │          │           │              │
       │               │  LLM metric parse    │              │
       │               │  (DO Inference)      │              │
       │               │          │           │              │
       │               │  MongoDB write       │              │
       │               │  benchmark_runs      │              │
       └───────────────┴──────────────────────┴──────────────┘
                   │
       ┌───────────▼────────────┐
       │      AnalystAgent      │
       │   agents/analyst.py    │
       │                        │
       │ • top_configs()        │
       │ • oom_patterns()       │
       │ • perf_over_time()     │
       │ • LLM winner analysis  │
       │                        │
       │  returns winner_flags  │
       │  → Stage 2 baseline    │
       └────────────────────────┘
```

### Stage 1 → Stage 2 handoff

```
AnalystAgent.analyse()
        │
        │  returns AnalysisResult.winner_flags (dict)
        ▼
ControllerAgent._stage1() → (winner_flags, winner_fingerprint)
        │
        │  winner_flags passed directly as baseline
        ▼
KernelOptimizerAgent.run(baseline_flags=winner_flags)
        │
        │  each iteration: merged_flags = winner_flags + kernel_override
        ▼
ReportGenerator.generate(analysis, best_kernel_config)
        │  final recipe = winner_flags merged with best_kernel_config
```

### Data flow

```
configs/search_space.yaml
        │
        │  SearchSpace.sample_random() × (population_size × generations)
        ▼
PlannerAgent  ──── DO Serverless Inference ───►  ranked order + rationale
        │                                        (skips cross-session seen)
        │  insert priority-ordered "pending" docs
        ▼
MongoDB: configs collection
        │
        │  Coordinator claims pending, checks node /capacity
        ▼
Node Servers  (one per GPU Droplet)
        │
        │  parallel ExecutorAgents on GPU subsets
        ▼
MongoDB: benchmark_runs  ◄── fitness_score from MetricsCollector
        │
        ▼
AnalystAgent  ──── DO Serverless Inference ───►  winner explanation
        │
        │  winner_flags → Stage 2 baseline
        ▼
KernelOptimizerAgent  ──── DO Serverless Inference ───►  kernel proposals
        │
        │  10 iterations → MongoDB: kernel_runs
        ▼
ReportGenerator  ──►  storage/results/  (YAML + shell + Markdown)
```

---

## Quick start

### Local (Mac / Linux)

```bash
# 1. Clone and create a virtual environment
git clone https://github.com/RithishRamesh-dev/oceantune-ai
cd oceantune-ai
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Set secrets (.env file or export)
export HF_TOKEN=hf_...
export DO_INFERENCE_KEY=...                                    # DigitalOcean Serverless Inference key
export DO_INFERENCE_ENDPOINT=https://inference.do-ai.run/v1   # DO Inference base URL
export MONGO_URI=mongodb+srv://user:pass@your-cluster/oceantune?tls=true  # required — no local MongoDB

# 3. Validate config
python3 oceantune.py validate-config

# 4. Run the full pipeline
python3 oceantune.py run --gpu H100
```

### GPU Droplet setup

```bash
# On each GPU Droplet — start the Node Server
ssh root@YOUR_DROPLET_IP

git clone https://github.com/RithishRamesh-dev/oceantune-ai.git /opt/oceantune-ai
cd /opt/oceantune-ai
pip install -r requirements.txt

export MONGO_URI=mongodb+srv://user:pass@your-cluster/oceantune?tls=true
export DO_INFERENCE_KEY=...
export DO_INFERENCE_ENDPOINT=https://inference.do-ai.run/v1
export HF_TOKEN=hf_...
export NODE_HOST=YOUR_DROPLET_IP     # reported back to the Coordinator

python3 -m node.node_server \
    --port 9000 \
    --gpu-type H100 \
    --gpu-indices 0,1,2,3,4,5,6,7

# Then add this node to configs/oceantune.yaml under nodes:
#   - host: YOUR_DROPLET_IP
#     node_port: 9000
#     gpu_type: H100
#     gpu_indices: [0,1,2,3,4,5,6,7]
```

### Docker

```bash
# Build once
docker build -t oceantune-ai:latest .

# Run the full test suite (no GPU needed — all tests are mocked)
docker compose run --rm tests

# Run with GPU passthrough
docker compose run --rm tests-gpu

# Launch a vLLM server on port 8000
docker compose up vllm-server

# Full optimisation pipeline
docker compose run --rm optimizer
```

Secrets are loaded from `.env` (see `.env.example` — never commit `.env`).

---

## Repository layout

```
oceantune-ai/
├── oceantune.py                    # CLI entry point
├── requirements.txt                # Pinned dependencies
├── Dockerfile
├── docker-compose.yml
│
├── agents/
│   ├── controller_agent.py         # Top-level orchestrator: Stage 1 → handoff → Stage 2
│   ├── planner.py                  # Hardware validation + cross-session dedup + LLM ranking
│   ├── executor.py                 # Single-config: vLLM + benchmark + LLM metric parse + MongoDB
│   ├── analyst.py                  # Winner selection + aggregation pipelines + LLM explanation
│   ├── kernel_optimizer.py         # Stage 2: iterative LLM-guided kernel search (10 iterations)
│   └── do_client.py                # DO Serverless Inference HTTP client (retry, json_mode)
│
├── core/
│   ├── config.py                   # OceanTuneConfig, DatabaseConfig, NodeConfig,
│   │                               #   CoordinatorConfig, AgentConfig, OptimiserConfig
│   ├── db.py                       # MongoDB async client — 5 collections + analytics pipelines
│   ├── coordinator.py              # Parallel dispatch: poll MongoDB → assign to nodes → retry
│   ├── node_client.py              # HTTP client for remote Node Servers
│   ├── gpu_allocator.py            # GPU slot partitioning via CUDA_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES
│   ├── port_allocator.py           # Port pool (default 8000–8099) for parallel vLLM instances
│   ├── report_generator.py         # Emits YAML recipe + shell script + Markdown report
│   ├── search_space.py             # VLLMFlags (25 fields), SearchSpace (candidate sampling),
│   │                               #   ConfigValidator (10 hardware-constraint checks)
│   ├── vllm_server.py              # Async vLLM process manager + GPU-profile env injection
│   ├── benchmark_runner.py         # BenchmarkEngine — concurrency ramp, 22 regex patterns
│   ├── metrics_collector.py        # EnrichedMetrics — fitness scoring, GPU efficiency, OOM penalty
│   ├── log_analyzer.py             # 14 error-class patterns, startup timing, OOM/crash detection
│   └── logger.py                   # Structured logging (console + JSONL)
│
├── node/
│   ├── node_server.py              # FastAPI on each GPU Droplet — job queue + capacity API
│   └── node_worker.py              # Executes benchmark jobs; threads primary_metric to ExecutorAgent
│
├── configs/
│   ├── oceantune.yaml              # Main config: model, GPU, database, nodes, coordinator, optimiser
│   ├── models.yaml                 # 7 supported models with MoE/MLA/NVFP4 metadata
│   ├── gpu_profiles.yaml           # 6 GPU profiles: H100, H200, B300, MI300X, MI325X, MI350X
│   ├── search_space.yaml           # 20 Stage 1 vLLM flag parameters with bounds and defaults
│   ├── kernel_search_space.yaml    # 15 Stage 2 kernel parameters (attention, NCCL, AITER, DBO)
│   └── inference_models.yaml       # DO Serverless Inference model registry (suitability scores)
│
├── scripts/
│   ├── run_vllm.sh                 # Shell wrapper for vLLM (ulimits, PID file, signals)
│   ├── benchmark.sh                # Manual benchmark runner for individual concurrency levels
│   └── docker_test.sh              # One-shot droplet bootstrap + test runner
│
├── storage/
│   ├── logs/                       # Per-session JSONL logs (gitignored)
│   └── results/                    # YAML recipes, shell scripts, Markdown reports (gitignored)
│
└── tests/
    ├── test_search_space.py        # 66 tests — VLLMFlags, SearchSpace, ConfigValidator
    ├── test_vllm_server.py         # 50 tests — profile-driven server runner, AMD env injection
    ├── test_benchmark_runner.py    # 53 tests — regex parsing, concurrency ramp, early abort
    ├── test_log_analyzer.py        # 36 tests — 14 error classes, startup timing
    └── test_metrics_collector.py   # 32 tests — fitness scoring, GPU efficiency, primary metrics
```

---

## Configuration

Edit [configs/oceantune.yaml](configs/oceantune.yaml). Secrets always come from environment variables — never from YAML.

### Key settings

| Key | Default | Description |
|-----|---------|-------------|
| `model_id` | `deepseek-ai/DeepSeek-V3.2` | Hugging Face model ID |
| `gpu_type` | `H100` | GPU profile key |
| `agent.model` | `auto` | `auto` picks highest `suitability_score` from `configs/inference_models.yaml`; or set a specific model ID |
| `agent.temperature` | `0.3` | LLM temperature for all 4 agents (lower = more deterministic) |
| `agent.max_tokens` | `4096` | Max completion tokens per agent reasoning turn |
| `agent.timeout_sec` | `120` | HTTP timeout per DO Inference call |
| `database.uri` | `mongodb://localhost:27017` | MongoDB connection string (override via `MONGO_URI`) |
| `database.name` | `oceantune` | MongoDB database name |
| `nodes` | `[localhost:9000]` | GPU Droplet node list — each entry needs `host`, `node_port`, `gpu_type`, `gpu_indices` |
| `coordinator.max_parallel_per_node` | `2` | Cap on concurrent vLLM instances per node (bounded by GPU count ÷ tensor_parallel_size) |
| `coordinator.port_pool_start` | `8000` | First port in the per-node pool |
| `coordinator.port_pool_end` | `8099` | Last port in the per-node pool |
| `coordinator.max_retries` | `2` | Times to re-queue a config after node failure before marking it failed |
| `optimiser.population_size` | `10` | Candidates sampled per generation (passed to PlannerAgent) |
| `optimiser.generations` | `5` | Number of search rounds (`population_size × generations` = total configs) |
| `optimiser.primary_metric` | `throughput` | Fitness metric: `throughput` / `p95_latency` / `ttft` / `tpot` — used by MetricsCollector and Analyst |
| `benchmark.concurrency_levels` | `[1,2,4,8,16,32,64]` | Concurrency ramp per benchmark run |

---

## CLI commands

```bash
python3 oceantune.py --help
python3 oceantune.py validate-config          # check YAML + env vars
python3 oceantune.py run --dry-run            # validate only, no GPU needed
python3 oceantune.py run --gpu H100           # run full two-stage pipeline
python3 oceantune.py run --strategy bayesian  # override search strategy label
python3 oceantune.py info                     # print system / GPU info
```

---

## Node Server API

Each GPU Droplet runs `node/node_server.py` (FastAPI). The Coordinator communicates with it over HTTP. It threads `optimiser.primary_metric` from the config down to each `ExecutorAgent`.

```bash
python3 -m node.node_server \
    --port 9000 \
    --gpu-type H100 \
    --gpu-indices 0,1,2,3,4,5,6,7 \
    --port-pool-start 8000 \
    --port-pool-end 8099
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check + free GPU / port counts |
| `/capacity` | GET | Total / free GPUs, free ports, in-use port list |
| `/jobs` | POST | Submit a benchmark job — returns `job_id` immediately (async) |
| `/jobs/{job_id}` | GET | Poll job status: `pending` / `running` / `done` / `failed` |

---

## MongoDB collections

| Collection | Key fields | Purpose |
|------------|-----------|---------|
| `sessions` | model_id, gpu_type, strategy, status, created_at | One document per optimisation run |
| `nodes` | host, node_port, gpu_type, gpu_count, last_seen | GPU Droplet heartbeats (liveness via `last_seen`) |
| `configs` | session_id, fingerprint, flags, status, priority, retry_count | Candidate configs queue (`pending→running→done/failed`) |
| `benchmark_runs` | session_id, config_id, context, raw_metrics, fitness_score | All benchmark results (all contexts per config) |
| `kernel_runs` | session_id, iteration, kernel_config, fitness_score, llm_reasoning | Stage 2 results with LLM rationale |

### Analytics pipelines (`core/db.py`)

| Method | Returns |
|--------|---------|
| `top_configs_by_throughput(session_id, n)` | Top-N configs by max throughput across all contexts |
| `kernel_impact_analysis(session_id)` | Kernel flags ranked by average fitness delta |
| `oom_patterns(session_id)` | Configs associated with OOM errors + their flag patterns |
| `performance_over_time(session_id)` | Fitness time-series — used by AnalystAgent convergence check |
| `cross_session_seen_fingerprints(model_id, gpu_type)` | Fingerprints from all prior sessions for deduplication |

---

## DO Serverless Inference

All four agents (`PlannerAgent`, `ExecutorAgent`, `AnalystAgent`, `KernelOptimizerAgent`) share a single `DOClient` instance that calls the DO Serverless Inference API.

```
Base URL:   https://inference.do-ai.run/v1   (override via DO_INFERENCE_ENDPOINT)
Auth:       Bearer DO_INFERENCE_KEY
Model:      auto-selected from configs/inference_models.yaml by suitability_score
            or pinned via DO_INFERENCE_MODEL env var
```

`agent.model: auto` in `oceantune.yaml` means the highest-rated model from `configs/inference_models.yaml` is used. Set `agent.model` to a specific model ID to pin it. All agents fall back to deterministic (non-LLM) behaviour when `DO_INFERENCE_KEY` is not set.

---

## Supported models

| Alias | Hugging Face ID | Params | Notes |
|-------|-----------------|--------|-------|
| `deepseek_v3_2` | `deepseek-ai/DeepSeek-V3.2` | 671B | MoE + MLA; requires `block_size=1` |
| `deepseek_v3_2_nvfp4` | `nvidia/DeepSeek-V3.2-NVFP4` | 671B | NVIDIA Hopper/Blackwell only (NVFP4) |
| `minimax_m2_5` | `MiniMaxAI/MiniMax-M2.5` | 229B | MoE; tool-call + reasoning parsers |
| `kimi_k2_5` | `moonshotai/Kimi-K2.5` | 1T | MoE; FP8 recommended |
| `kimi_k2_5_nvfp4` | `nvidia/Kimi-K2.5-NVFP4` | 1T | NVIDIA only |
| `qwen3_5_397b` | `Qwen/Qwen3.5-397B-A17B` | 397B | MoE; 17B active params |
| `qwen3_5_397b_nvfp4` | `nvidia/Qwen3.5-397B-A17B-NVFP4` | 397B | NVIDIA only |

---

## Supported GPUs

| Profile | VRAM | Vendor | Arch | FP8 | NVFP4 | Docker image |
|---------|------|--------|------|-----|-------|--------------|
| `H100` | 80 GB | NVIDIA | Hopper | ✓ | — | `vllm/vllm-openai:latest-cu130` |
| `H200` | 141 GB | NVIDIA | Hopper | ✓ | — | `vllm/vllm-openai:latest-cu130` |
| `B300` | 192 GB | NVIDIA | Blackwell | ✓ | ✓ | `vllm/vllm-openai:latest-cu130` |
| `MI300X` | 192 GB | AMD | CDNA3 | ✓ | — | `vllm/vllm-openai-rocm:v0.18.1` |
| `MI325X` | 256 GB | AMD | CDNA3 | ✓ | — | `vllm/vllm-openai-rocm:v0.18.1` |
| `MI350X` | 288 GB | AMD | CDNA4 | ✓ | — | `vllm/vllm-openai-rocm:v0.18.1` |

AMD profiles automatically inject 12 ROCm performance env vars (`VLLM_ROCM_USE_AITER`, `HSA_NO_SCRATCH_RECLAIM`, etc.) and append `--distributed-executor-backend mp`.

---

## Stage 2 kernel search space (`configs/kernel_search_space.yaml`)

15 parameters explored by `KernelOptimizerAgent`. Each is tagged by vendor so the LLM only proposes flags valid for the target GPU.

| Parameter | Vendor | Type | What it controls |
|-----------|--------|------|-----------------|
| `attention_backend` | All | choice | `FLASH_ATTN` / `FLASHINFER` / `ROCM_FLASH` / `XFORMERS` / `TORCH_SDPA` |
| `all2all_backend` | NVIDIA | choice | MoE expert dispatch: `deepep_normal` / `deepep_low_latency` / `vllm` |
| `enable_dbo` | NVIDIA | bool | Dynamic Batch Optimiser |
| `vllm_rocm_use_aiter` | AMD | bool | AIter fused-attention kernel |
| `vllm_rocm_use_aiter_mla` | AMD | bool | AIter MLA kernel (DeepSeek MLA) |
| `vllm_rocm_use_aiter_rmsnorm` | AMD | bool | AIter fused RMSNorm |
| `vllm_rocm_use_aiter_moe` | AMD | bool | AIter MoE fused-GEMM |
| `nccl_min_nchannels` | NVIDIA | range_int 1–16 | NCCL channels per NVLINK ring |
| `nccl_socket_nthreads` | NVIDIA | range_int 1–8 | NCCL socket threads |
| `rccl_enable_intranode` | AMD | bool | RCCL intra-node optimised path |
| `hsa_no_scratch_reclaim` | AMD | bool | Disable HSA scratch buffer reclamation |
| `quant_dtype` | All | choice | `auto` / `float16` / `bfloat16` |
| `kv_cache_dtype` | All | choice | `auto` / `fp8` / `fp8_e5m2` / `fp8_e4m3` |
| `scheduler_delay_factor` | All | range_float 0–1 | Scheduler token budget fraction (higher = more throughput, worse TTFT) |
| `enable_prefix_caching` | All | bool | KV-cache reuse for repeated prefixes |

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | Hugging Face access token (required for gated models) |
| `DO_INFERENCE_KEY` | DigitalOcean Serverless Inference API key — used by all 4 agents |
| `DO_INFERENCE_ENDPOINT` | Override inference base URL (default: `https://inference.do-ai.run/v1`) |
| `DO_INFERENCE_MODEL` | Pin a specific inference model ID (overrides `agent.model: auto`) |
| `MONGO_URI` | MongoDB connection string — **required**, no local MongoDB assumed |
| `DO_SPACES_KEY` | DigitalOcean Spaces access key |
| `DO_SPACES_SECRET` | DigitalOcean Spaces secret key |
| `NODE_HOST` | Hostname this Node Server reports to the Coordinator (important for multi-node routing) |
| `OCEANTUNE_MODEL_ID` | Override `model_id` from YAML |
| `OCEANTUNE_GPU_TYPE` | Override `gpu_type` from YAML |
| `OCEANTUNE_PORT` | Override vLLM port |
| `OCEANTUNE_STRATEGY` | Override optimisation strategy label |
| `OCEANTUNE_PRIMARY_METRIC` | Override primary fitness metric |

---

## Test suite

```
pytest tests/ --asyncio-mode=auto

tests/test_search_space.py       66 passed   VLLMFlags, SearchSpace, ConfigValidator
tests/test_vllm_server.py        50 passed   profile-driven server, AMD/NVIDIA env
tests/test_benchmark_runner.py   53 passed   regex parsing, concurrency ramp
tests/test_log_analyzer.py       36 passed   14 error classes, startup timing
tests/test_metrics_collector.py  32 passed   fitness scoring, GPU efficiency
─────────────────────────────────────────
Total                           238 passed
```

All tests are mocked — no GPU, no live server, no Hugging Face token required.

```bash
# Run locally
pytest tests/ --asyncio-mode=auto -v

# Run in Docker (no GPU needed)
docker compose run --rm tests
```
