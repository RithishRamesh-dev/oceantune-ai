# OceanTune AI

Automated vLLM inference optimisation system. Finds the best combination of vLLM flags and kernel settings for a given model, GPU, and context-length profile — without manual tuning.

A **4-agent closed-loop LLM pipeline** (powered by DigitalOcean Serverless Inference) replaces mechanical hyperparameter search with genuine hardware reasoning. After every benchmark the Analyst diagnoses the bottleneck from the concurrency curve, and that diagnosis feeds directly into the next Planner proposal. All results are stored in **MongoDB** for cross-session analytics and deduplication.

---

## Results

| Model | GPU | Best Throughput | Best Fitness | Winner Config |
|-------|-----|-----------------|--------------|---------------|
| Qwen/Qwen2.5-7B-Instruct | H200 | 5407 tok/s | 0.693 | `gpu_memory_utilization=0.9, max_num_batched_tokens=8192` |

---

## How it works

OceanTune runs a two-stage closed-loop optimisation pipeline:

**Stage 1 — Iterative Agent-Guided vLLM Config Search**

Stage 1 is a feedback loop. Each iteration the Analyst diagnoses *why* the last config performed the way it did, and that diagnosis is passed directly to the Planner's next proposal.

```
Iteration 0: bare minimum VLLMFlags (vLLM defaults) → establishes baseline
     │
     ▼
ExecutorAgent: start vLLM in Docker → run concurrency ramp → compute fitness
     │
     ▼
AnalystAgent.evaluate_iteration(): read concurrency curve → diagnose bottleneck
     │  returns: { bottleneck, diagnosis, flag_insights, recommendation }
     │
     ▼
PlannerAgent.propose_next(analyst_eval=...): LLM reads diagnosis → proposes
     │  one targeted flag change to address the bottleneck
     │
     ▼
Iteration 1: benchmarks the proposed config
     │  ... repeat for N generations ...
     ▼
Stage 1 winner: highest fitness_score config → passed to Stage 2
```

**Stage 2 — Kernel-Level Search**

`KernelOptimizerAgent` applies the same iterative LLM loop to low-level kernel flags (attention backend, KV cache dtype, scheduler parameters, NVIDIA/AMD-specific settings) on top of the Stage 1 winner. 10 iterations. Results stored in MongoDB `kernel_runs`.

`ReportGenerator` emits a YAML recipe, ready-to-run shell script, and Markdown summary.

---

## Architecture

> **[Full Stage 1 architecture diagram with component details →](docs/architecture_stage1.md)**

```mermaid
flowchart LR
    CLI([oceantune.py run])

    CLI --> CTRL["ControllerAgent"]

    subgraph S1["Stage 1 — vLLM Config Search (N iterations)"]
        direction TB
        PLAN["PlannerAgent\n────────────\nReads analyst_eval\nDetects plateau/regression\nLLM proposes next VLLMFlags"]
        EXEC["ExecutorAgent\n────────────\nDocker vLLM server\nConcurrency ramp benchmark\nMetricsCollector fitness score"]
        ANALYST["AnalystAgent\n────────────\nevaluate_iteration()\nDiagnose bottleneck\nfrom concurrency curve"]
        DB1[("MongoDB\nconfigs +\nbenchmark_runs")]

        PLAN -->|"VLLMFlags"| DB1
        DB1 -->|"config_doc"| EXEC
        EXEC -->|"EnrichedMetrics\n+ fitness_score"| DB1
        DB1 -->|"best run\nfor config"| ANALYST
        ANALYST -->|"bottleneck +\nrecommendation"| PLAN
    end

    subgraph S2["Stage 2 — Kernel Search (10 iterations)"]
        direction TB
        KO["KernelOptimizerAgent\n────────────\nLLM kernel proposals\non top of winner_flags"]
        DB2[("MongoDB\nkernel_runs")]
        KO --> DB2
        DB2 --> KO
    end

    CTRL -->|"Iteration 0\nbaseline flags"| S1
    S1 -->|"winner_flags"| S2
    S2 -->|"best_kernel"| RG["ReportGenerator"]
    RG --> OUT([storage/results/\nYAML + shell + Markdown])

    DO["DO Serverless Inference\nanthropic-claude-4.5-sonnet\n────────────────────────\nPlanner proposals\nAnalyst diagnosis\nKernel proposals\nWinner explanation"]

    PLAN -.- DO
    ANALYST -.- DO
    KO -.- DO
```

### The closed-loop feedback signal

The critical path that makes Stage 1 genuinely LLM-guided (not random search):

```
benchmark completes
     │
     ▼  concurrency curve: tok/s at [1, 2, 4, 8, 16, 32, 64, 128]
AnalystAgent.evaluate_iteration()
     │  sends curve + flags to DO Serverless Inference
     │
     ▼
analyst_eval = {
    "bottleneck":      "memory",
    "diagnosis":       "throughput scales to concurrency 64 but flattens — KV cache fills
                        before GPU compute saturates",
    "flag_insights":   "gpu_memory_utilization=0.90 leaves 10% VRAM unused",
    "recommendation":  "try kv_cache_dtype=fp8 to halve KV memory footprint"
}
     │
     ▼  passed to next iteration
PlannerAgent.propose_next(analyst_eval=analyst_eval)
     │  plateau detected? → force aggressive exploration
     │  regression in history? → flag it explicitly to LLM
     │
     ▼
Iteration N+1: VLLMFlags with kv_cache_dtype=fp8
```

### Stage 1 → Stage 2 handoff

```
AnalystAgent.analyse()   (full session analysis)
     │  returns AnalysisResult.winner_flags
     ▼
ControllerAgent._stage1() → (winner_flags, winner_fingerprint)
     │
     ▼
KernelOptimizerAgent.run(baseline_flags=winner_flags)
     │  each iteration: merged_flags = winner_flags + kernel_override
     ▼
ReportGenerator.generate(analysis, best_kernel_config)
     │  final recipe = winner_flags merged with best_kernel_config
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
├── show_results.py                 # CLI results viewer: table / CSV / per-level breakdown
├── requirements.txt                # Pinned dependencies
├── Dockerfile
├── docker-compose.yml
│
├── agents/
│   ├── controller_agent.py         # Top-level orchestrator: iterative Stage 1 loop → Stage 2
│   ├── planner.py                  # Proposes next VLLMFlags using analyst_eval + plateau/regression detection
│   ├── executor.py                 # Single-config: vLLM + benchmark + MetricsCollector + MongoDB write
│   ├── analyst.py                  # Per-iteration bottleneck diagnosis + full session winner analysis
│   ├── kernel_optimizer.py         # Stage 2: iterative LLM-guided kernel search (10 iterations)
│   └── do_client.py                # DO Serverless Inference HTTP client (retry, json_mode, fence-strip)
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
│   ├── benchmark_runner.py         # BenchmarkEngine — concurrency ramp, asyncio.wait partial results
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
├── docs/
│   └── architecture_stage1.md     # Detailed Stage 1 Mermaid diagram + fitness formula + feedback loop
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

## Viewing results

```bash
# Summary table of all runs in the latest session
python3 show_results.py

# Include per-concurrency-level breakdown
python3 show_results.py --levels

# Export to CSV
python3 show_results.py --csv > results.csv

# Specific session
python3 show_results.py --session 69fe1b7ef7ca80b8a87b2dd5

# All sessions, top 20 configs
python3 show_results.py --all --top 20
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
| `database.uri` | `""` | MongoDB connection string (set via `MONGO_URI`) |
| `database.name` | `oceantune` | MongoDB database name |
| `nodes` | `[localhost:9000]` | GPU Droplet node list — each entry needs `host`, `node_port`, `gpu_type`, `gpu_indices` |
| `coordinator.max_parallel_per_node` | `1` | Cap on concurrent vLLM instances per node |
| `coordinator.port_pool_start` | `8000` | First port in the per-node pool |
| `coordinator.port_pool_end` | `8099` | Last port in the per-node pool |
| `coordinator.max_retries` | `2` | Times to re-queue a config after node failure |
| `optimiser.population_size` | `10` | Candidates sampled per generation |
| `optimiser.generations` | `10` | Number of search iterations |
| `optimiser.primary_metric` | `throughput` | Fitness metric: `throughput` / `p95_latency` / `ttft` / `tpot` |
| `benchmark.concurrency_levels` | `[1,2,4,8,16,32,64,128]` | Concurrency ramp per benchmark run |

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

Each GPU Droplet runs `node/node_server.py` (FastAPI). The Coordinator communicates with it over HTTP.

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
| `nodes` | host, node_port, gpu_type, gpu_count, last_seen | GPU Droplet heartbeats |
| `configs` | session_id, fingerprint, flags, status, priority, retry_count | Candidate configs queue (`pending→running→done/failed`) |
| `benchmark_runs` | session_id, config_id, flags, levels[], enriched_metrics, fitness_score | All benchmark results with per-concurrency level data |
| `kernel_runs` | session_id, iteration, kernel_config, fitness_score, llm_reasoning | Stage 2 results with LLM rationale |

### Analytics pipelines (`core/db.py`)

| Method | Returns |
|--------|---------|
| `top_configs_by_throughput(session_id, n)` | Top-N configs by max throughput across all contexts |
| `get_top_configs(session_id, n)` | Top-N configs by fitness score |
| `get_best_run_for_config(config_id)` | Best benchmark run for a single config (used by per-iteration analyst) |
| `kernel_impact_analysis(session_id)` | Kernel flags ranked by average fitness delta |
| `oom_patterns(session_id)` | Configs associated with OOM errors + their flag patterns |
| `performance_over_time(session_id)` | Fitness time-series — used by AnalystAgent convergence check |
| `cross_session_seen_fingerprints(model_id, gpu_type)` | Fingerprints from all prior sessions for deduplication |

---

## DO Serverless Inference

All four agents (`PlannerAgent`, `ExecutorAgent`, `AnalystAgent`, `KernelOptimizerAgent`) share a single `DOClient` instance.

```
Base URL:   https://inference.do-ai.run/v1   (override via DO_INFERENCE_ENDPOINT)
Auth:       Bearer DO_INFERENCE_KEY          (or AGENT_API_KEY)
Model:      anthropic-claude-4.5-sonnet      (override via DO_INFERENCE_MODEL or AGENT_MODEL)
```

All agents fall back to deterministic (non-LLM) behaviour when `DO_INFERENCE_KEY` is not set.

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
| `scheduler_delay_factor` | All | range_float 0–1 | Scheduler token budget fraction |
| `enable_prefix_caching` | All | bool | KV-cache reuse for repeated prefixes |

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | Hugging Face access token (required for gated models) |
| `DO_INFERENCE_KEY` | DigitalOcean Serverless Inference API key (also: `AGENT_API_KEY`) |
| `DO_INFERENCE_ENDPOINT` | Override inference base URL (also: `AGENT_ENDPOINT`) |
| `DO_INFERENCE_MODEL` | Pin a specific inference model ID (also: `AGENT_MODEL`) |
| `MONGO_URI` | MongoDB connection string — **required**, no local MongoDB assumed |
| `DO_SPACES_KEY` | DigitalOcean Spaces access key |
| `DO_SPACES_SECRET` | DigitalOcean Spaces secret key |
| `NODE_HOST` | Hostname this Node Server reports to the Coordinator |
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
