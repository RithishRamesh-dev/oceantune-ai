# OceanTune AI

Automated vLLM inference optimisation system. Finds the best combination of vLLM flags for a given model, GPU, and context-length profile — without manual tuning.

---

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

## Build steps

The system is built incrementally. Each step is independently testable.

### Step 1 — Repository setup & project skeleton ✅
Creates the full directory tree, logging system (`core/logger.py`), config loader (`core/config.py`), placeholder modules, `requirements.txt`, CLI entry point.

Key files: [core/logger.py](core/logger.py), [core/config.py](core/config.py), [oceantune.py](oceantune.py)

---

### Step 2 — Search space & typed config system ✅
Typed parameter system (`ChoiceParam`, `RangeIntParam`, `RangeFloatParam`, `BoolFlagParam`). `VLLMFlags` dataclass with 25 fields, SHA1 fingerprinting, and full `to_vllm_args()` CLI generation. `SearchSpace` with random/seeded/grid sampling and evolutionary crossover. `ConfigValidator` with 10 hardware-constraint checks covering AMD ROCm, MoE flags, FP8, block size, and speculative decoding.

Key files: [core/search_space.py](core/search_space.py), [configs/search_space.yaml](configs/search_space.yaml), [tests/test_search_space.py](tests/test_search_space.py)

**Test suite — 66/66 passed**

---

### Step 3 — vLLM server runner ✅
Fully async process manager. Launches vLLM as a subprocess with flags from `VLLMFlags.to_vllm_args()`, injects GPU-profile env vars (including all 12 ROCm perf vars for AMD), appends `vllm_extra_args` from the GPU profile, streams stdout into a 500-line rotating log buffer, polls `/health` with exponential backoff, and tears down via SIGTERM → SIGKILL. Five-class failure hierarchy: `OOMError`, `StartupTimeout`, `PortConflict`, `CUDAError`, `ProcessCrash`.

Key files: [core/vllm_server.py](core/vllm_server.py), [configs/gpu_profiles.yaml](configs/gpu_profiles.yaml), [tests/test_vllm_server.py](tests/test_vllm_server.py)

**Test suite — 50/50 passed**

---

### Step 4 — Benchmark engine ✅
Wraps `vllm bench serve` to run the full concurrency ramp (1→2→4→8→16→32→64) against a live vLLM server. Captures throughput, p50/p90/p95/p99 latency, TTFT, TPOT, ITL, and error rate per level. 22 compiled regex patterns cover vLLM 0.4–0.18 output formats. `RampResult.summary` is the 12-key fitness signal fed to the optimiser. Early abort after 3 consecutive failed levels.

Key files: [core/benchmark_runner.py](core/benchmark_runner.py), [scripts/benchmark.sh](scripts/benchmark.sh), [tests/test_benchmark_runner.py](tests/test_benchmark_runner.py)

**Test suite — 53/53 passed**

```bash
# Manual benchmark against a live server
./scripts/benchmark.sh \
    --base-url http://localhost:8000 \
    --model deepseek-ai/DeepSeek-V3.2 \
    --input-len 1024 --output-len 1024 \
    --concurrency 16 --num-prompts 200
```

---

### Step 5 — Metrics collector & log analyser ✅
Three components that turn raw benchmark output into persistent, comparable records:

- **LogAnalyzer** — 14 error-class patterns (OOM, NCCL, CUDA kernel, tokenizer, ROCm, FlashInfer, process group), 8 single-value extractors (model load time, KV cache blocks/GB, GPU memory, TP init time, CUDA graph time), wall-clock startup estimation from log timestamps.
- **MetricsCollector** — GPU efficiency (tokens/sec per GB VRAM), memory headroom fraction (accounts for tensor parallel), normalised fitness score with log-scaled throughput, inverted latency scoring, 4 primary metric modes, and penalties for errors/OOM.
- **ResultStorage** — append-only CSV + per-run JSON + async DigitalOcean Spaces upload. `ResultLoader` provides history query, top-N by metric, and failed fingerprint set for cross-session deduplication.

Key files: [core/log_analyzer.py](core/log_analyzer.py), [core/metrics_collector.py](core/metrics_collector.py), [core/storage.py](core/storage.py)

**Test suite — 101/101 passed** (36 log_analyzer + 32 metrics_collector + 33 storage)

---

### Step 6 — Optimisation engine ✅
Four search strategies sharing a common interface:

- **Evolutionary** (primary) — tournament selection (k=3), crossover, mutation, elitism carry-forward, diversity guard with random injection when top-5 configs converge.
- **Grid** — hill-climbing via single-parameter neighbour exploration; advances seed to all-time best after each generation.
- **Random** — uniform sampling baseline.
- **Bayesian** — placeholder with surrogate-model hook; delegates to evolutionary until 20 results available (Stage 3+).

`OptimiserState` is fully serialisable (generation, population, scored history sorted best-first, failed + seen fingerprints) for crash recovery. `PopulationManager` loads cross-session failures from `runs.csv` and validates every candidate with `ConfigValidator` before queuing. `_filter_and_cap` falls back to random valid configs if breeding produces too few candidates.

Key files: [core/optimizer.py](core/optimizer.py), [tests/test_optimizer.py](tests/test_optimizer.py)

**Test suite — 61/61 passed**

---

### Step 7 — Experiment runner ⏳
Composes the vLLM server (Step 3) + benchmark engine (Step 4) + metrics collector (Step 5) + storage (Step 5) into a single `run_experiment(flags) -> RunRecord` call. Handles retries on startup failure.

Key files: `core/experiment_runner.py`

---

### Step 8 — Controller agent ⏳
Top-level orchestrator wiring Steps 3–7 into the full pipeline loop. Entry point for `python3 oceantune.py run`.

Key files: `agents/controller_agent.py`

---

### Step 9 — Recipe generator ⏳
Takes the best `RunRecord` and produces a copy-paste-ready vLLM launch recipe: shell script, `docker run` command, and JSON summary.

Key files: `core/recipe_generator.py`

---

### Step 10 — GitHub Actions CI ⏳
Workflow: validate-config, unit tests, and (on `main`) real optimisation run on self-hosted GPU runner.

Key files: `ci/oceantune.yml`

---

### Step 11 — Multi-model & multi-GPU matrix ⏳
Fan out across all `configs/models.yaml` × `configs/gpu_profiles.yaml` combinations using `asyncio` task groups.

---

### Step 12 — Speculative decoding ⏳
Add draft-model selection to the search space. Measure acceptance rate vs. latency trade-off.

---

### Step 13 — Research agent ⏳
Scrapes arXiv and vLLM's GitHub for new flags. Summarises with Claude and proposes `search_space.yaml` additions as a PR.

Key files: `agents/research_agent.py`

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
