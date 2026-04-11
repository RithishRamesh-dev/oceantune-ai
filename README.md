# OceanTune AI

Automated vLLM inference optimisation system. Finds the best combination of vLLM flags for a given model, GPU, and context-length profile — without manual tuning.

---

Design considerations:
Run the benchmark in parallel
Store all the benchmark results in Database that helps in Analytics for future


## How it works

OceanTune runs a closed-loop optimisation pipeline:

1. Start a vLLM server with a candidate set of flags
2. Benchmark it (throughput, p95 latency, TTFT) across concurrency levels and context lengths
3. Feed results to an optimisation engine (evolutionary / Bayesian / grid / random)
4. Repeat until convergence, then emit a ready-to-use recipe

---

## Quick start

### Local (Mac / Linux)

```bash
# 1. Clone and create a virtual environment
git clone https://github.com/RithishRamesh-dev/oceantune-ai
cd oceantune-ai
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Set secrets
export HF_TOKEN=hf_...
export DO_SPACES_KEY=...
export DO_SPACES_SECRET=...

# 3. Validate config
python3 oceantune.py validate-config

# 4. Run the optimiser
python3 oceantune.py run --gpu H100
```

### GPU Droplet (Docker — recommended)

```bash
ssh root@YOUR_DROPLET_IP

export REPO_URL=https://github.com/RithishRamesh-dev/oceantune-ai.git
export HF_TOKEN=hf_...
git clone $REPO_URL /opt/oceantune-ai
cd /opt/oceantune-ai
bash scripts/docker_test.sh      # installs Docker, builds image, runs tests

# Day-to-day:
docker compose run --rm tests            # run test suite (no GPU needed)
docker compose run --rm tests-gpu        # run tests with GPU passthrough
docker compose up vllm-server            # launch vLLM on port 8000
docker compose run --rm optimizer        # full optimisation pipeline
```

---

## Repository layout

```
oceantune-ai/
├── oceantune.py              # CLI entry point
├── requirements.txt          # Pinned dependencies
├── Dockerfile                # Extends vllm/vllm-openai; ENTRYPOINT [] override
├── docker-compose.yml        # Services: tests, tests-gpu, vllm-server, optimizer
├── core/
│   ├── logger.py             # Structured logging (console + JSONL file)
│   ├── config.py             # YAML + env-var config loader
│   ├── search_space.py       # Typed params, VLLMFlags (25 fields), SearchSpace, ConfigValidator
│   ├── vllm_server.py        # Async vLLM process manager + profile-driven env/args
│   ├── benchmark_runner.py   # BenchmarkEngine, RampResult, parse_benchmark_output
│   ├── metrics_collector.py  # MetricsCollector, EnrichedMetrics, fitness scoring
│   ├── log_analyzer.py       # LogAnalyzer — 14 error classes, startup timing
│   ├── storage.py            # RunRecord, ResultStorage (CSV+JSON+Spaces), ResultLoader
│   ├── optimizer.py          # EvolutionaryOptimiser, GridOptimiser, RandomOptimiser, Bayesian
│   ├── experiment_runner.py  # stub — Step 7
│   └── recipe_generator.py   # stub — Step 9
├── agents/
│   ├── controller_agent.py   # stub — Step 8
│   └── research_agent.py     # stub — Step 13
├── configs/
│   ├── oceantune.yaml        # Main config (edit this)
│   ├── models.yaml           # 7 supported models with MoE/MLA/NVFP4 metadata
│   ├── gpu_profiles.yaml     # 6 GPU profiles: H100, H200, B300, MI300X, MI325X, MI350X
│   └── search_space.yaml     # 20 optimisation parameters with vLLM flag mappings
├── scripts/
│   ├── run_vllm.sh           # Shell wrapper for vLLM (ulimits, PID file, signals)
│   ├── benchmark.sh          # Manual benchmark runner for individual concurrency levels
│   └── docker_test.sh        # One-shot droplet bootstrap + test runner
├── storage/
│   ├── logs/                 # Per-session JSONL logs (gitignored)
│   └── results/              # Benchmark results and recipes (gitignored)
└── tests/
    ├── test_search_space.py      # 66 tests — Step 2 + GPU expansion
    ├── test_vllm_server.py       # 50 tests — Step 3 + profile-driven env
    ├── test_benchmark_runner.py  # 53 tests — Step 4
    ├── test_log_analyzer.py      # 36 tests — Step 5
    ├── test_metrics_collector.py # 32 tests — Step 5
    ├── test_storage.py           # 33 tests — Step 5
    └── test_optimizer.py         # 61 tests — Step 6
```

---

## Configuration

Edit [configs/oceantune.yaml](configs/oceantune.yaml) to change the model, GPU, and search settings. Secrets are always read from environment variables — never stored in YAML.

| Key | Default | Description |
|-----|---------|-------------|
| `model_id` | `deepseek-ai/DeepSeek-V3.2` | Hugging Face model ID |
| `gpu_type` | `H100` | GPU profile — see Supported GPUs below |
| `optimiser.strategy` | `evolutionary` | Search strategy (`evolutionary`, `grid`, `random`, `bayesian`) |
| `optimiser.primary_metric` | `throughput` | Metric to maximise (`throughput`, `p95_latency`, `ttft`, `tpot`) |
| `optimiser.generations` | `5` | Number of optimisation generations |
| `optimiser.population_size` | `10` | Candidates per generation |

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

## Supported models

| Alias | Hugging Face ID | Params | Notes |
|-------|-----------------|--------|-------|
| `deepseek_v3_2` | `deepseek-ai/DeepSeek-V3.2` | 671B | MoE + MLA; block_size=1 |
| `deepseek_v3_2_nvfp4` | `nvidia/DeepSeek-V3.2-NVFP4` | 671B | NVIDIA only (NVFP4) |
| `minimax_m2_5` | `MiniMaxAI/MiniMax-M2.5` | 229B | MoE; tool-call + reasoning parsers |
| `kimi_k2_5` | `moonshotai/Kimi-K2.5` | 1T | MoE; FP8 recommended |
| `kimi_k2_5_nvfp4` | `nvidia/Kimi-K2.5-NVFP4` | 1T | NVIDIA only (NVFP4) |
| `qwen3_5_397b` | `Qwen/Qwen3.5-397B-A17B` | 397B | MoE; 17B active params |
| `qwen3_5_397b_nvfp4` | `nvidia/Qwen3.5-397B-A17B-NVFP4` | 397B | NVIDIA only (NVFP4) |

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

AMD profiles automatically inject ROCm env vars (`VLLM_ROCM_USE_AITER`, `HSA_NO_SCRATCH_RECLAIM`, etc.) and append `--distributed-executor-backend mp` via the GPU profile.

---

## Docker

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

**Note:** `vllm/vllm-openai:latest` sets `ENTRYPOINT ["vllm"]`. The Dockerfile overrides this with `ENTRYPOINT []` so `CMD` works normally. Only genuinely missing packages are installed to avoid downgrading vLLM's own dependencies.

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | Hugging Face access token (required for gated models) |
| `DO_SPACES_KEY` | DigitalOcean Spaces access key |
| `DO_SPACES_SECRET` | DigitalOcean Spaces secret key |
| `OCEANTUNE_MODEL_ID` | Override `model_id` from config |
| `OCEANTUNE_GPU_TYPE` | Override `gpu_type` from config |
| `OCEANTUNE_PORT` | Override vLLM port |
| `OCEANTUNE_STRATEGY` | Override optimisation strategy |
| `OCEANTUNE_PRIMARY_METRIC` | Override primary metric |

---

## Architecture (v4)

OceanTune AI uses a **4-agent parallel system** backed by MongoDB for analytics and DigitalOcean Serverless Inference for all LLM calls.

```
oceantune.py run
      │
      ▼
ControllerAgent
      │
      ├─ Stage 1: vLLM Config Search
      │     ├─ PlannerAgent       — hardware validation + LLM-ranked config ordering
      │     ├─ MongoDB            — configs queued as "pending" documents
      │     ├─ Coordinator        — parallel dispatch to N GPU Droplet Node Servers
      │     │     └─ NodeServer (FastAPI on each droplet)
      │     │           └─ ExecutorAgent  — vLLM + benchmark + LLM metric parse
      │     └─ AnalystAgent       — picks winner, LLM explanation
      │
      └─ Stage 2: Kernel-Level Search
            └─ KernelOptimizerAgent — iterative LLM-guided attention/NCCL/AITER tuning
                  └─ ReportGenerator  — YAML recipe + shell script + Markdown report
```

All agents call **DigitalOcean Serverless Inference** (`DO_INFERENCE_KEY`). MongoDB stores sessions, configs, benchmark results, and kernel results for cross-session analytics.

---

## Repository layout

```
oceantune-ai/
├── oceantune.py               # CLI entry point
├── requirements.txt           # Pinned dependencies (motor, fastapi, uvicorn added)
├── Dockerfile
├── docker-compose.yml
├── agents/
│   ├── controller_agent.py    # Top-level pipeline orchestrator (v4)
│   ├── planner.py             # Hardware validation + LLM config ranking
│   ├── executor.py            # Single-config benchmark agent + MongoDB writes
│   ├── analyst.py             # Post-search winner selection + LLM analysis
│   ├── kernel_optimizer.py    # Stage 2 LLM-guided kernel search
│   ├── do_client.py           # DO Serverless Inference HTTP client
│   └── research_agent.py      # stub — future arXiv/GitHub flag scraper
├── core/
│   ├── config.py              # Config dataclasses (DatabaseConfig, NodeConfig, CoordinatorConfig added)
│   ├── db.py                  # MongoDB async client + 5 collections + analytics pipelines
│   ├── coordinator.py         # Parallel dispatch loop across GPU Droplets
│   ├── node_client.py         # HTTP client for remote Node Servers
│   ├── gpu_allocator.py       # GPU slot partitioning (CUDA_VISIBLE_DEVICES)
│   ├── port_allocator.py      # Port pool for parallel vLLM instances
│   ├── report_generator.py    # YAML recipe + shell script + Markdown report
│   ├── search_space.py        # VLLMFlags (25 fields), SearchSpace, ConfigValidator
│   ├── vllm_server.py         # Async vLLM process manager
│   ├── benchmark_runner.py    # BenchmarkEngine (22 regex patterns)
│   ├── metrics_collector.py   # EnrichedMetrics + fitness scoring
│   ├── log_analyzer.py        # 14 error-class patterns + startup timing
│   ├── optimizer.py           # Evolutionary/Grid/Random/Bayesian optimisers
│   ├── storage.py             # Legacy CSV+JSON+Spaces storage (kept for compatibility)
│   └── logger.py              # Structured logging
├── node/
│   ├── node_server.py         # FastAPI server running on each GPU Droplet
│   └── node_worker.py         # Executes vLLM + benchmark jobs on behalf of Coordinator
├── configs/
│   ├── oceantune.yaml         # Main config (model, GPU, database, nodes, coordinator)
│   ├── models.yaml            # 7 supported models
│   ├── gpu_profiles.yaml      # 6 GPU profiles (H100/H200/B300/MI300X/MI325X/MI350X)
│   ├── search_space.yaml      # 20 Stage 1 vLLM flag parameters
│   ├── kernel_search_space.yaml # Stage 2 kernel parameters (attention, NCCL, AITER, DBO)
│   └── inference_models.yaml  # DO Serverless Inference model registry
├── scripts/
│   ├── run_vllm.sh
│   ├── benchmark.sh
│   └── docker_test.sh
├── storage/
│   ├── logs/
│   └── results/               # YAML recipes, shell scripts, Markdown reports
└── tests/                     # 331 mocked tests (no GPU required)
```

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | Hugging Face access token (required for gated models) |
| `DO_INFERENCE_KEY` | DigitalOcean Serverless Inference API key |
| `DO_INFERENCE_ENDPOINT` | Inference base URL (default: `https://inference.do-ai.run/v1`) |
| `DO_INFERENCE_MODEL` | Override agent model ID |
| `MONGO_URI` | MongoDB connection string (default: `mongodb://localhost:27017`) |
| `DO_SPACES_KEY` | DigitalOcean Spaces access key |
| `DO_SPACES_SECRET` | DigitalOcean Spaces secret key |
| `OCEANTUNE_MODEL_ID` | Override `model_id` from config |
| `OCEANTUNE_GPU_TYPE` | Override `gpu_type` from config |
| `OCEANTUNE_PORT` | Override vLLM port |
| `OCEANTUNE_STRATEGY` | Override optimisation strategy |
| `OCEANTUNE_PRIMARY_METRIC` | Override primary metric |
| `NODE_HOST` | Hostname reported by a Node Server to the Coordinator |

---

## Node Server (GPU Droplet)

Each GPU Droplet runs a FastAPI Node Server that the Coordinator dispatches jobs to:

```bash
# On each GPU Droplet:
python3 -m node.node_server \
    --port 9000 \
    --gpu-type H100 \
    --gpu-indices 0,1,2,3,4,5,6,7 \
    --port-pool-start 8000 \
    --port-pool-end 8099
```

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Node liveness + free GPU/port count |
| `GET /capacity` | Detailed GPU slot + port availability |
| `POST /jobs` | Submit a benchmark job (async, returns `job_id`) |
| `GET /jobs/{job_id}` | Poll job status (`pending`/`running`/`done`/`failed`) |

---

## MongoDB Collections

| Collection | Purpose |
|------------|---------|
| `sessions` | One document per optimisation run (model, GPU, strategy, timestamps) |
| `nodes` | Heartbeat / capacity records for each GPU Droplet |
| `configs` | Candidate VLLMFlags configs with `pending`/`running`/`done`/`failed` status |
| `benchmark_runs` | Raw + enriched benchmark results per config per context length |
| `kernel_runs` | Stage 2 kernel-level results with LLM reasoning |

Analytics aggregation pipelines available via `core.db.Database`:
- `top_configs_by_throughput()` — rank configs by max throughput across contexts
- `kernel_impact_analysis()` — rank kernel flags by average fitness impact
- `oom_patterns()` — configs associated with OOM failures
- `performance_over_time()` — fitness time-series for progress charts
- `cross_session_seen_fingerprints()` — deduplication across restarts

---

## Build steps

### Steps 1–6 — Core infrastructure ✅

| Step | Description | Tests |
|------|-------------|-------|
| 1 | Repo skeleton, logging, config loader | — |
| 2 | VLLMFlags (25 fields), SearchSpace, ConfigValidator | 66 |
| 3 | vLLM server runner (profile-driven, AMD/NVIDIA) | 50 |
| 4 | Benchmark engine (22 regex patterns, concurrency ramp) | 53 |
| 5 | LogAnalyzer, MetricsCollector, ResultStorage | 101 |
| 6 | Evolutionary / Grid / Random / Bayesian optimisers | 61 |

**Test suite — 331/331 passed**

---

### Steps 7–12 — v4 Multi-Agent System ✅

| Step | File(s) | Description |
|------|---------|-------------|
| 7 | `core/db.py` | MongoDB schema + async CRUD + 5 analytics pipelines |
| 8 | `agents/do_client.py` | DO Serverless Inference client (retry, json_mode) |
| 9 | `agents/planner.py` | Hardware validation + LLM config ranking |
| 10 | `core/gpu_allocator.py`, `core/port_allocator.py` | GPU slot partitioning + port pool |
| 11 | `agents/executor.py` | Single-config benchmark + LLM metric parse + MongoDB writes |
| 12 | `node/node_server.py`, `node/node_worker.py` | FastAPI node server on each GPU Droplet |
| 13 | `core/coordinator.py`, `core/node_client.py` | Parallel dispatch coordinator + HTTP node client |
| 14 | `agents/analyst.py` | Winner selection + LLM analysis |
| 15 | `configs/kernel_search_space.yaml`, `agents/kernel_optimizer.py` | Stage 2 kernel search |
| 16 | `core/report_generator.py` | YAML recipe + shell script + Markdown report |

---

## Test suite summary

```
pytest tests/ --asyncio-mode=auto

tests/test_search_space.py       66 passed
tests/test_vllm_server.py        50 passed
tests/test_benchmark_runner.py   53 passed
tests/test_log_analyzer.py       36 passed
tests/test_metrics_collector.py  32 passed
tests/test_storage.py            33 passed
tests/test_optimizer.py          61 passed
─────────────────────────────────────────
Total                           331 passed
```

All tests are mocked — no GPU, no live server, no Hugging Face token required.
