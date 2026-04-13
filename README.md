# OceanTune AI

Automated vLLM inference optimisation system. Finds the best combination of vLLM flags and kernel settings for a given model, GPU, and context-length profile — without manual tuning.

Benchmarks run **in parallel** across GPU Droplets. All results are stored in **MongoDB** for cross-session analytics and deduplication. A **4-agent LLM pipeline** (powered by DigitalOcean Serverless Inference) replaces mechanical search with genuine hardware reasoning.

---

## How it works

OceanTune runs a two-stage closed-loop optimisation pipeline:

**Stage 1 — vLLM Config Search**
1. Sample candidate vLLM flag combinations from the search space
2. PlannerAgent validates hardware compatibility and LLM-ranks the candidates
3. Coordinator dispatches configs in parallel to GPU Droplet Node Servers
4. Each ExecutorAgent starts a vLLM instance, runs the benchmark ramp, and writes results to MongoDB
5. AnalystAgent reads MongoDB, picks the winner, and produces an LLM explanation

**Stage 2 — Kernel-Level Search**
6. KernelOptimizerAgent iteratively proposes low-level kernel/attention/NCCL flag combinations
7. Each proposal is benchmarked; the LLM learns from the history to guide the next iteration
8. ReportGenerator emits a YAML recipe, shell script, and Markdown report

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        oceantune.py run                             │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ControllerAgent                               │
│                  agents/controller_agent.py                         │
└──────────────┬──────────────────────────────────┬───────────────────┘
               │                                  │
    ┌──────────▼──────────┐            ┌──────────▼──────────┐
    │     STAGE 1         │            │     STAGE 2         │
    │  vLLM Config Search │            │  Kernel-Level Search│
    └──────────┬──────────┘            └──────────┬──────────┘
               │                                  │
    ┌──────────▼──────────┐            ┌──────────▼──────────┐
    │    PlannerAgent     │            │  KernelOptimizer    │
    │  agents/planner.py  │            │     Agent           │
    │                     │            │ agents/kernel_       │
    │ • Hardware validate │            │   optimizer.py      │
    │ • LLM-rank configs  │            │                     │
    │ • Skip seen configs │            │ • LLM proposes next │
    └──────────┬──────────┘            │   kernel flags      │
               │                       │ • Benchmarks each   │
    ┌──────────▼──────────┐            │ • Learns from hist  │
    │      MongoDB        │            └──────────┬──────────┘
    │     core/db.py      │                       │
    │                     │            ┌──────────▼──────────┐
    │ configs → "pending" │            │  ReportGenerator    │
    └──────────┬──────────┘            │ core/report_        │
               │                       │   generator.py      │
    ┌──────────▼──────────┐            │                     │
    │    Coordinator      │            │ • YAML recipe       │
    │ core/coordinator.py │            │ • Shell script      │
    │                     │            │ • Markdown report   │
    │ • Poll pending      │            └─────────────────────┘
    │ • Check node cap.   │
    │ • Dispatch jobs     │
    │ • Retry on failure  │
    └──────────┬──────────┘
               │  HTTP  (core/node_client.py)
    ┌──────────▼──────────────────────────────────┐
    │            GPU Droplet Node Server           │
    │             node/node_server.py              │
    │                                              │
    │  POST /jobs  ──►  NodeWorker                 │
    │                   node/node_worker.py        │
    │                         │                   │
    │              ┌──────────▼──────────┐        │
    │              │   ExecutorAgent     │        │
    │              │ agents/executor.py  │        │
    │              │                     │        │
    │              │ GPUSlotAllocator    │        │
    │              │  (CUDA_VISIBLE_     │        │
    │              │   DEVICES)          │        │
    │              │ PortAllocator       │        │
    │              │  (8000–8099)        │        │
    │              │        │            │        │
    │              │   VLLMServer  ◄─────┤        │
    │              │ core/vllm_server.py │        │
    │              │        │            │        │
    │              │   BenchmarkEngine   │        │
    │              │ core/benchmark_     │        │
    │              │   runner.py         │        │
    │              │        │            │        │
    │              │   MetricsCollector  │        │
    │              │ core/metrics_       │        │
    │              │   collector.py      │        │
    │              │        │            │        │
    │              │   MongoDB write     │        │
    │              │  benchmark_runs     │        │
    │              └─────────────────────┘        │
    └──────────────────────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │    AnalystAgent     │
    │  agents/analyst.py  │
    │                     │
    │ • top_configs()     │
    │ • oom_patterns()    │
    │ • perf_over_time()  │
    │ • LLM explanation   │
    └─────────────────────┘
```

### Data flow

```
Search Space (configs/search_space.yaml)
        │
        │  sample N candidates
        ▼
  PlannerAgent  ──── DO Serverless Inference ────►  ranked order + rationale
        │
        │  insert as "pending" documents
        ▼
     MongoDB  ◄──────────────────────────────────── all writes go here
   (5 collections)
        │
        │  Coordinator polls + claims pending configs
        ▼
  Node Servers  (one per GPU Droplet, FastAPI)
        │
        │  parallel vLLM instances on GPU slot subsets
        ▼
  Benchmark Results  ──► MongoDB  ──► AnalystAgent  ──► winner flags
        │
        │  Stage 2: winner flags as baseline
        ▼
  KernelOptimizer  ──── DO Serverless Inference ────►  kernel config proposals
        │
        │  iterate 10 rounds
        ▼
  ReportGenerator  ──►  YAML recipe + shell script + Markdown report
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
export DO_INFERENCE_KEY=...          # DigitalOcean Serverless Inference key
export MONGO_URI=mongodb://localhost:27017

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

export MONGO_URI=mongodb://YOUR_MONGO_HOST:27017
export DO_INFERENCE_KEY=...
export HF_TOKEN=hf_...

python3 -m node.node_server \
    --port 9000 \
    --gpu-type H100 \
    --gpu-indices 0,1,2,3,4,5,6,7

# Then on the orchestrator machine, point configs/oceantune.yaml nodes: to this droplet's IP
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
│   ├── controller_agent.py         # Top-level orchestrator — wires all agents + both stages
│   ├── planner.py                  # Hardware validation + LLM-ranked config ordering
│   ├── executor.py                 # Single-config: vLLM + benchmark + LLM metric parse + MongoDB
│   ├── analyst.py                  # Winner selection + aggregation pipelines + LLM explanation
│   ├── kernel_optimizer.py         # Stage 2: iterative LLM-guided kernel search
│   └── do_client.py                # DO Serverless Inference HTTP client (retry, json_mode)
│
├── core/
│   ├── config.py                   # Config dataclasses: OceanTuneConfig, DatabaseConfig,
│   │                               #   NodeConfig, CoordinatorConfig, AgentConfig
│   ├── db.py                       # MongoDB async client — 5 collections + analytics pipelines
│   ├── coordinator.py              # Parallel dispatch loop: poll MongoDB → assign to nodes
│   ├── node_client.py              # HTTP client for remote Node Servers
│   ├── gpu_allocator.py            # GPU slot partitioning via CUDA_VISIBLE_DEVICES
│   ├── port_allocator.py           # Port pool (8000–8099) for parallel vLLM instances
│   ├── report_generator.py         # YAML recipe + shell script + Markdown report
│   ├── search_space.py             # VLLMFlags (25 fields), SearchSpace, ConfigValidator
│   ├── vllm_server.py              # Async vLLM process manager + profile-driven env/args
│   ├── benchmark_runner.py         # BenchmarkEngine — 22 regex patterns, concurrency ramp
│   ├── metrics_collector.py        # EnrichedMetrics — fitness scoring, GPU efficiency
│   ├── log_analyzer.py             # 14 error classes, startup timing, OOM detection
│   └── logger.py                   # Structured logging (console + JSONL)
│
├── node/
│   ├── node_server.py              # FastAPI server running on each GPU Droplet
│   └── node_worker.py              # Executes vLLM + benchmark jobs for the Coordinator
│
├── configs/
│   ├── oceantune.yaml              # Main config: model, GPU, database, nodes, coordinator
│   ├── models.yaml                 # 7 supported models with MoE/MLA/NVFP4 metadata
│   ├── gpu_profiles.yaml           # 6 GPU profiles: H100, H200, B300, MI300X, MI325X, MI350X
│   ├── search_space.yaml           # 20 Stage 1 vLLM flag parameters
│   ├── kernel_search_space.yaml    # 15 Stage 2 kernel parameters (attention, NCCL, AITER, DBO)
│   └── inference_models.yaml       # DO Serverless Inference model registry
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
    ├── test_vllm_server.py         # 50 tests — profile-driven server runner
    ├── test_benchmark_runner.py    # 53 tests — regex parsing, concurrency ramp
    ├── test_log_analyzer.py        # 36 tests — error patterns, startup timing
    └── test_metrics_collector.py   # 32 tests — fitness scoring, GPU efficiency
```

---

## Configuration

Edit [configs/oceantune.yaml](configs/oceantune.yaml). Secrets always come from environment variables — never from YAML.

### Key settings

| Key | Default | Description |
|-----|---------|-------------|
| `model_id` | `deepseek-ai/DeepSeek-V3.2` | Hugging Face model ID |
| `gpu_type` | `H100` | GPU profile key |
| `agent.model` | `auto` | DO Inference model (`auto` = highest suitability score) |
| `agent.temperature` | `0.3` | LLM temperature for all agents |
| `database.uri` | `mongodb://localhost:27017` | MongoDB connection string |
| `nodes` | `[localhost:9000]` | GPU Droplet node list |
| `coordinator.max_parallel_per_node` | `2` | Max concurrent vLLM instances per node |
| `optimiser.population_size` | `10` | Candidates per generation |
| `optimiser.generations` | `5` | Number of optimisation generations |
| `optimiser.primary_metric` | `throughput` | `throughput` / `p95_latency` / `ttft` / `tpot` |
| `benchmark.concurrency_levels` | `[1,2,4,8,16,32,64]` | Concurrency ramp |

---

## CLI commands

```bash
python3 oceantune.py --help
python3 oceantune.py validate-config          # check YAML + env vars
python3 oceantune.py run --dry-run            # validate only, no GPU needed
python3 oceantune.py run --gpu H100           # run full pipeline
python3 oceantune.py run --strategy bayesian  # override search strategy
python3 oceantune.py info                     # print system / GPU info
```

---

## Node Server API

Each GPU Droplet runs `node/node_server.py` (FastAPI). The Coordinator communicates with it over HTTP.

```bash
python3 -m node.node_server --port 9000 --gpu-type H100 --gpu-indices 0,1,2,3,4,5,6,7
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check + free GPU / port counts |
| `/capacity` | GET | Total / free GPUs, free ports, in-use port list |
| `/jobs` | POST | Submit a benchmark job — returns `job_id` immediately (async) |
| `/jobs/{job_id}` | GET | Poll job status: `pending` / `running` / `done` / `failed` |

---

## MongoDB collections

| Collection | Document | Purpose |
|------------|----------|---------|
| `sessions` | model, gpu, strategy, status, timestamps | One per optimisation run |
| `nodes` | host, gpu_type, gpu_count, last_seen | GPU Droplet heartbeats |
| `configs` | fingerprint, flags, status, retry_count | Candidate configs queue |
| `benchmark_runs` | raw_metrics, enriched_metrics, fitness_score | All benchmark results |
| `kernel_runs` | kernel_config, fitness_score, llm_reasoning | Stage 2 results |

### Analytics pipelines (`core/db.py`)

| Method | Returns |
|--------|---------|
| `top_configs_by_throughput(session_id, n)` | Top-N configs ranked by max throughput |
| `kernel_impact_analysis(session_id)` | Kernel flags ranked by average fitness impact |
| `oom_patterns(session_id)` | Configs associated with OOM errors |
| `performance_over_time(session_id)` | Fitness time-series for progress charts |
| `cross_session_seen_fingerprints(model_id, gpu_type)` | Deduplication across restarts |

---

## Supported models

| Alias | Hugging Face ID | Params | Notes |
|-------|-----------------|--------|-------|
| `deepseek_v3_2` | `deepseek-ai/DeepSeek-V3.2` | 671B | MoE + MLA; requires block_size=1 |
| `deepseek_v3_2_nvfp4` | `nvidia/DeepSeek-V3.2-NVFP4` | 671B | NVIDIA Hopper/Blackwell only |
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

## Stage 2 kernel search space

The `configs/kernel_search_space.yaml` defines 15 parameters explored in Stage 2:

| Parameter | Scope | What it controls |
|-----------|-------|-----------------|
| `attention_backend` | All | `FLASH_ATTN` / `FLASHINFER` / `ROCM_FLASH` / `XFORMERS` / `TORCH_SDPA` |
| `all2all_backend` | NVIDIA | MoE expert dispatch: `deepep_normal` / `deepep_low_latency` / `vllm` |
| `enable_dbo` | NVIDIA | Dynamic Batch Optimiser |
| `vllm_rocm_use_aiter` | AMD | AIter fused-attention kernel |
| `vllm_rocm_use_aiter_mla` | AMD | AIter MLA kernel (DeepSeek MLA) |
| `vllm_rocm_use_aiter_rmsnorm` | AMD | AIter fused RMSNorm |
| `vllm_rocm_use_aiter_moe` | AMD | AIter MoE fused-GEMM |
| `nccl_min_nchannels` | NVIDIA | NCCL channels per NVLINK ring |
| `nccl_socket_nthreads` | NVIDIA | NCCL socket threads |
| `rccl_enable_intranode` | AMD | RCCL intra-node optimised path |
| `hsa_no_scratch_reclaim` | AMD | Disable HSA scratch reclamation |
| `quant_dtype` | All | `auto` / `float16` / `bfloat16` |
| `kv_cache_dtype` | All | `auto` / `fp8` / `fp8_e5m2` / `fp8_e4m3` |
| `scheduler_delay_factor` | All | Scheduler token budget fraction |
| `enable_prefix_caching` | All | KV-cache reuse for repeated prefixes |

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | Hugging Face access token (required for gated models) |
| `DO_INFERENCE_KEY` | DigitalOcean Serverless Inference API key |
| `DO_INFERENCE_ENDPOINT` | Inference base URL (default: `https://inference.do-ai.run/v1`) |
| `DO_INFERENCE_MODEL` | Override the agent LLM model ID |
| `MONGO_URI` | MongoDB connection string (default: `mongodb://localhost:27017`) |
| `DO_SPACES_KEY` | DigitalOcean Spaces access key |
| `DO_SPACES_SECRET` | DigitalOcean Spaces secret key |
| `OCEANTUNE_MODEL_ID` | Override `model_id` from YAML |
| `OCEANTUNE_GPU_TYPE` | Override `gpu_type` from YAML |
| `OCEANTUNE_PORT` | Override vLLM port |
| `OCEANTUNE_STRATEGY` | Override optimisation strategy |
| `OCEANTUNE_PRIMARY_METRIC` | Override primary metric |
| `NODE_HOST` | Hostname reported by a Node Server to the Coordinator |

---

## Test suite

```
pytest tests/ --asyncio-mode=auto

tests/test_search_space.py       66 passed
tests/test_vllm_server.py        50 passed
tests/test_benchmark_runner.py   53 passed
tests/test_log_analyzer.py       36 passed
tests/test_metrics_collector.py  32 passed
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
