# OceanTune AI

OceanTune is an automated vLLM inference optimisation system. It benchmarks your model on your GPU across hundreds of flag combinations and uses an LLM-guided search loop to find the configuration that maximises throughput — without any manual tuning.

**What you get:** a ready-to-run shell script and YAML recipe with the optimal `--gpu-memory-utilization`, `--max-num-batched-tokens`, `--kv-cache-dtype`, attention backend, and 20+ other vLLM flags for your specific model and GPU.

---

## Benchmark results

| Model | GPU | Peak throughput | Fitness | Winner flags |
|-------|-----|-----------------|---------|--------------|
| `Qwen/Qwen2.5-7B-Instruct` | H200 141 GB | **5407 tok/s** | 0.693 | `gpu_memory_utilization=0.9, max_num_batched_tokens=8192` |

---

## How OceanTune works

OceanTune runs a two-stage pipeline driven by four LLM agents powered by [DigitalOcean Serverless Inference](https://www.digitalocean.com/products/ai-ml/serverless-inference).

### Stage 1 — Iterative vLLM config search

Each iteration is a closed feedback loop:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   PlannerAgent  ──proposes next VLLMFlags──►  ExecutorAgent     │
│       ▲                                           │             │
│       │                                     (Docker vLLM)       │
│       │                                           │             │
│       │   bottleneck + recommendation       BenchmarkEngine     │
│       │                                           │             │
│   AnalystAgent  ◄──concurrency curve──────  MetricsCollector    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

1. **Iteration 0** — baseline run with bare minimum vLLM flags (no tuning). Establishes a fitness score to improve from.
2. **ExecutorAgent** starts vLLM in Docker, runs a concurrency ramp `[1, 2, 4, 8, 16, 32, 64, 128]`, and computes a fitness score from the resulting throughput and latency curve.
3. **AnalystAgent** reads the concurrency curve, calls the LLM, and diagnoses the bottleneck: *compute-bound*, *memory-bound*, or *scheduling-bound*.
4. **PlannerAgent** receives the diagnosis and recommendation, then calls the LLM to propose the single most impactful flag change for the next iteration.
5. Repeat for N generations. The planner detects fitness plateaus and regression configs and signals the LLM to explore more aggressively.

### Stage 2 — Kernel-level search

`KernelOptimizerAgent` takes the Stage 1 winner and iteratively tunes lower-level settings: attention backend (`FLASH_ATTN`, `FLASHINFER`), KV cache dtype, scheduler parameters, and GPU-vendor-specific flags (NVIDIA DBO, AMD AITER kernels). 10 iterations.

### Output

`ReportGenerator` writes three files to `storage/results/`:

| File | Contents |
|------|----------|
| `recipe_*.yaml` | Optimal flags as a YAML dict — import into your deployment config |
| `run_*.sh` | Ready-to-run `vllm serve` shell script with all flags pre-filled |
| `report_*.md` | Markdown report: winner config, LLM analysis, top-5 table, OOM insights |

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | Tested on 3.11 and 3.12 |
| Docker with GPU passthrough | vLLM runs in Docker; `nvidia-docker` or `rocm-docker` required on GPU server |
| MongoDB | Managed MongoDB (DO, Atlas) or self-hosted. No local MongoDB assumed. |
| DigitalOcean Serverless Inference key | Powers all 4 LLM agents. Agents fall back to deterministic search without it. |
| Hugging Face token | Required for gated models (Llama, Gemma, etc.) |

---

## Quick start

### Step 1 — Clone and install

```bash
git clone https://github.com/RithishRamesh-dev/oceantune-ai
cd oceantune-ai
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2 — Set environment variables

Create a `.env` file in the project root (never commit it):

```bash
# ── Required ─────────────────────────────────────────────────────────────────

# MongoDB connection string (DO Managed MongoDB, Atlas, or self-hosted)
MONGO_URI=mongodb+srv://user:password@your-cluster.mongodb.net/oceantune?tls=true&authSource=admin

# ── LLM agents (strongly recommended — agents fall back to deterministic search without this) ──

# DigitalOcean Serverless Inference API key
DO_INFERENCE_KEY=dop_v1_...
# Inference base URL
DO_INFERENCE_ENDPOINT=https://inference.do-ai.run/v1
# Model to use for all 4 agents
DO_INFERENCE_MODEL=anthropic-claude-4.5-sonnet

# ── Optional ──────────────────────────────────────────────────────────────────

# Hugging Face token — required for gated models (Llama, Gemma, Mistral)
HF_TOKEN=hf_...
```

> **Note:** `DO_INFERENCE_KEY` / `DO_INFERENCE_ENDPOINT` / `DO_INFERENCE_MODEL` can also be set as `AGENT_API_KEY` / `AGENT_ENDPOINT` / `AGENT_MODEL` — both names are checked.

### Step 3 — Set your model and GPU

Edit `configs/oceantune.yaml` (or pass flags to the CLI — see below):

```yaml
# The model you want to optimise
model_id: "Qwen/Qwen2.5-7B-Instruct"

# The GPU type of your server — must match a key in configs/gpu_profiles.yaml
# Options: H100 | H200 | B300 | MI300X | MI325X | MI350X
gpu_type: "H200"
```

### Step 4 — Validate the setup

```bash
python3 oceantune.py validate-config
```

Expected output:
```
✅  Config valid

  Model ID   : Qwen/Qwen2.5-7B-Instruct
  GPU type   : H200
  Metric     : throughput
  Generations: 10
  Concurrency: [1, 2, 4, 8, 16, 32, 64, 128]

  HF_TOKEN set       : ✅
  DO_SPACES_KEY set  : ⚠️  NOT SET
```

### Step 5 — Run the optimisation

```bash
# Run with model and GPU set in oceantune.yaml
python3 oceantune.py run

# Or override model and GPU from the command line
python3 oceantune.py run --model "Qwen/Qwen2.5-7B-Instruct" --gpu H200

# Validate config without starting any GPU workload
python3 oceantune.py run --dry-run
```

The pipeline logs iteration-by-iteration progress. A full 10-generation run on a 7B model takes approximately 2–4 hours depending on vLLM startup time and benchmark duration.

### Step 6 — View results

```bash
# Summary table of the latest session
python3 show_results.py

# Per-concurrency-level breakdown (shows the full throughput curve)
python3 show_results.py --levels

# Export to CSV for further analysis
python3 show_results.py --csv > results.csv

# View a specific past session
python3 show_results.py --session 69fe1b7ef7ca80b8a87b2dd5

# All sessions, top 20 configs by fitness
python3 show_results.py --all --top 20
```

Reports and recipes are also written to `storage/results/` automatically at the end of each run.

---

## Changing model and GPU

### Command-line overrides (no YAML edit needed)

```bash
# Different model, same GPU
python3 oceantune.py run --model "meta-llama/Llama-3.1-8B-Instruct" --gpu H100

# AMD GPU
python3 oceantune.py run --model "Qwen/Qwen2.5-72B-Instruct" --gpu MI300X

# Optimise for latency instead of throughput
export OCEANTUNE_PRIMARY_METRIC=p95_latency
python3 oceantune.py run --model "mistralai/Mistral-7B-Instruct-v0.3" --gpu H100
```

### Environment variable overrides

Any config value can be overridden without touching YAML:

```bash
export OCEANTUNE_MODEL_ID="deepseek-ai/DeepSeek-V3.2"
export OCEANTUNE_GPU_TYPE="H200"
export OCEANTUNE_PRIMARY_METRIC="throughput"   # throughput | p95_latency | ttft | tpot
python3 oceantune.py run
```

### Optimiser settings

Edit the `optimiser` section in `configs/oceantune.yaml`:

```yaml
optimiser:
  generations: 10           # iterations of the search loop — more = better result, longer runtime
  population_size: 10       # candidates sampled per generation (used by search space)
  primary_metric: "throughput"  # what the fitness score maximises

benchmark:
  concurrency_levels: [1, 2, 4, 8, 16, 32, 64, 128]  # concurrency ramp tested each run
  num_prompts: 30           # requests per concurrency level (higher = more stable measurement)

context_configs:
  - [1024, 1024]    # [input_tokens, output_tokens] — one benchmark pass per pair
  - [1024, 4096]    # add more pairs to test different context profiles
```

---

## Model configuration guide

### Small to mid-size models (≤ 13B) — any GPU

These run on a single GPU with default settings. No special flags required.

```bash
python3 oceantune.py run \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --gpu H200

python3 oceantune.py run \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --gpu H100

python3 oceantune.py run \
  --model "mistralai/Mistral-7B-Instruct-v0.3" \
  --gpu H100
```

For gated models (Llama, Gemma) set `HF_TOKEN` in `.env`.

### Large dense models (30B–70B) — multi-GPU

Tensor parallel size is set in `configs/oceantune.yaml` under `nodes.gpu_indices`. Add more GPU indices:

```yaml
nodes:
  - host: "localhost"
    node_port: 9000
    gpu_type: "H100"
    gpu_indices: [0, 1, 2, 3]   # 4 GPUs → tensor_parallel_size=4 available to the planner
```

```bash
python3 oceantune.py run \
  --model "meta-llama/Llama-3.1-70B-Instruct" \
  --gpu H100
```

### DeepSeek V3.2 (671B MoE + MLA) — 8× H200 or 8× MI300X

DeepSeek uses Multi-head Latent Attention (MLA) which requires `block_size=1`. OceanTune's `ConfigValidator` enforces this automatically. Set 8 GPUs:

```yaml
model_id: "deepseek-ai/DeepSeek-V3.2"
gpu_type: "H200"

nodes:
  - host: "localhost"
    node_port: 9000
    gpu_type: "H200"
    gpu_indices: [0, 1, 2, 3, 4, 5, 6, 7]   # 8× H200
```

```bash
# Requires HF_TOKEN — DeepSeek-V3.2 is gated
python3 oceantune.py run --model "deepseek-ai/DeepSeek-V3.2" --gpu H200
```

For NVIDIA-only NVFP4 variant (~half VRAM, best on B300):

```bash
python3 oceantune.py run --model "nvidia/DeepSeek-V3.2-NVFP4" --gpu B300
```

### AMD GPUs (MI300X / MI325X / MI350X)

OceanTune injects all required ROCm environment variables automatically from `configs/gpu_profiles.yaml`. Just set the GPU type:

```bash
python3 oceantune.py run \
  --model "Qwen/Qwen2.5-72B-Instruct" \
  --gpu MI300X

python3 oceantune.py run \
  --model "deepseek-ai/DeepSeek-V3.2" \
  --gpu MI325X
```

AMD profiles automatically set: `VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_USE_AITER_MLA=1`, `VLLM_ROCM_USE_AITER_MOE=1`, `HSA_NO_SCRATCH_RECLAIM=1`, `NCCL_MIN_NCHANNELS=112`, and 6 other ROCm perf vars.

### Fitness metric per use case

| Use case | Primary metric | Command |
|----------|---------------|---------|
| Batch processing / throughput | `throughput` | `export OCEANTUNE_PRIMARY_METRIC=throughput` |
| Interactive chat (low latency) | `p95_latency` | `export OCEANTUNE_PRIMARY_METRIC=p95_latency` |
| Streaming / time-to-first-token | `ttft` | `export OCEANTUNE_PRIMARY_METRIC=ttft` |
| Per-token generation speed | `tpot` | `export OCEANTUNE_PRIMARY_METRIC=tpot` |

---

## Model + GPU compatibility matrix

| Model | Size | H100 80G | H200 141G | B300 192G | MI300X 192G | MI325X 256G | Min GPUs |
|-------|------|----------|-----------|-----------|-------------|-------------|----------|
| Any ≤ 13B | 7–13B | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| Llama-3.1-70B | 70B BF16 | 4× | 2× | 1× | 2× | 1× | 2 |
| Qwen2.5-72B | 72B BF16 | 4× | 2× | 1× | 2× | 1× | 2 |
| Qwen3.5-397B | 397B MoE | 8× fp8 | 4× fp8 | 2× fp8 | 4× fp8 | 2× fp8 | 4 |
| DeepSeek-V3.2 | 671B MoE | 8× fp8 | 8× bf16 | 4× fp8 | 8× bf16 | 4× bf16 | 8 |
| nvidia/DeepSeek-V3.2-NVFP4 | 671B nvfp4 | 4× | 4× | 2× | ✗ | ✗ | 4 |
| Kimi-K2.5 | 1T MoE | 8× fp8 | 8× fp8 | 8× fp8 | 8× fp8 | 4× fp8 | 8 |
| nvidia/Kimi-K2.5-NVFP4 | 1T nvfp4 | 8× | 4× | 4× | ✗ | ✗ | 4 |
| MiniMax-M2.5 | 229B MoE | 4× | 2× | 2× | 4× | 2× | 4 |

> **MoE models** with `mla: true` (DeepSeek) require `block_size=1` — OceanTune enforces this automatically.  
> **NVFP4 models** require NVIDIA Hopper or Blackwell architecture (H100, H200, B300).

---

## Architecture

> **[Detailed Stage 1 Mermaid diagram →](docs/architecture_stage1.md)**

```mermaid
flowchart LR
    CLI([oceantune.py run])
    CLI --> CTRL["ControllerAgent"]

    subgraph S1["Stage 1 — vLLM Config Search  (N iterations)"]
        direction TB
        PLAN["PlannerAgent\n────────────────\nReads: analyst_eval, history\nDetects: plateau, regressions\nLLM call → next VLLMFlags\nFallback: curated variations"]
        EXEC["ExecutorAgent\n────────────────\nDockerised vLLM server\nConcurrency ramp benchmark\nMetricsCollector fitness score\nWrites flags + levels to MongoDB"]
        ANALYST["AnalystAgent\n────────────────\nevaluate_iteration()\nSends concurrency curve to LLM\nReturns: bottleneck type,\ndiagnosis, recommendation"]
        DB1[("MongoDB\nconfigs\nbenchmark_runs")]

        PLAN -->|"VLLMFlags proposal"| DB1
        DB1 -->|"config_doc"| EXEC
        EXEC -->|"EnrichedMetrics + fitness_score"| DB1
        DB1 -->|"best run for config"| ANALYST
        ANALYST -->|"analyst_eval dict"| PLAN
    end

    subgraph S2["Stage 2 — Kernel Search  (10 iterations)"]
        direction TB
        KO["KernelOptimizerAgent\n────────────────\nLLM kernel flag proposals\natop winner_flags\nAttention backend, KV dtype,\nNVIDIA DBO, AMD AITER"]
        DB2[("MongoDB\nkernel_runs")]
        KO <-->|"propose + record"| DB2
    end

    CTRL -->|"Iteration 0: baseline flags"| S1
    S1 -->|"winner_flags"| S2
    S2 -->|"best_kernel"| RG["ReportGenerator"]
    RG --> OUT(["storage/results/\nYAML recipe\nShell script\nMarkdown report"])

    DO(["DO Serverless Inference\nanthropic-claude-4.5-sonnet\n─────────────────\nAll 4 agent LLM calls"])
    PLAN -.- DO
    ANALYST -.- DO
    KO -.- DO
```

### Fitness score formula

`MetricsCollector` computes a single `[0, 1]` fitness score per benchmark run:

```
fitness = (0.70 × throughput_score + 0.30 × latency_score) × (1 − penalties)

throughput_score = log(tok_s / 100) / log(50000 / 100)   # log-scaled, 100→50000 range
latency_score    = (30000 − p95_ms) / (30000 − 10)       # inverted, 10ms→30000ms range

penalties:
  error_rate_penalty  = min(50%, error_rate × 50%)
  failed_level_penalty = 10% per failed concurrency level
  oom_penalty          = 30% if OOM/crash detected in logs
```

For other primary metrics, the weights shift: `p95_latency` → `(30%, 70%)`, `ttft` → `(20%, 0%, 80%, 0%)`.

---

## CLI reference

```bash
# Show all commands
python3 oceantune.py --help

# Validate YAML config and environment variables — no GPU required
python3 oceantune.py validate-config

# Validate only (parse config, check vars, then exit — no benchmarks)
python3 oceantune.py run --dry-run

# Run with YAML defaults
python3 oceantune.py run

# Override model and GPU from the command line
python3 oceantune.py run --model "Qwen/Qwen2.5-7B-Instruct" --gpu H200

# Point to a different config file
python3 oceantune.py run --config /path/to/my-config.yaml

# Print Python version, platform, GPU from nvidia-smi
python3 oceantune.py info
```

### `run` flags

| Flag | Short | Description |
|------|-------|-------------|
| `--model TEXT` | `-m` | Override `model_id` from YAML |
| `--gpu TEXT` | `-g` | Override `gpu_type` (H100, H200, B300, MI300X, MI325X, MI350X) |
| `--strategy` | `-s` | Override strategy label: `evolutionary` / `grid` / `random` / `bayesian` |
| `--config PATH` | `-c` | Use a different YAML config file |
| `--dry-run` | — | Validate config and exit without running any benchmarks |

### `show_results.py` flags

```bash
python3 show_results.py [options]

  --session ID   Show a specific session by MongoDB ObjectId
  --all          Show all sessions (not just the latest)
  --top N        Limit to top N rows by fitness (default: 50)
  --levels       Show per-concurrency-level breakdown instead of per-run summary
  --csv          Print CSV to stdout instead of a table
```

---

## Repository layout

```
oceantune-ai/
├── oceantune.py                    # CLI entry point (click)
├── show_results.py                 # Results viewer: table / CSV / per-level
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── agents/
│   ├── controller_agent.py         # Orchestrates Stage 1 loop and Stage 2
│   ├── planner.py                  # Proposes next VLLMFlags; plateau + regression detection
│   ├── executor.py                 # Runs one config: vLLM Docker + benchmark + MongoDB write
│   ├── analyst.py                  # Per-iteration bottleneck diagnosis + session winner analysis
│   ├── kernel_optimizer.py         # Stage 2: 10-iteration LLM kernel flag search
│   └── do_client.py                # DO Serverless Inference HTTP client (retry, JSON mode)
│
├── core/
│   ├── config.py                   # Config dataclasses + YAML loader
│   ├── db.py                       # MongoDB async client — 5 collections + analytics pipelines
│   ├── search_space.py             # VLLMFlags dataclass, SearchSpace sampler, ConfigValidator
│   ├── vllm_server.py              # Starts/stops vLLM in Docker; injects GPU env vars
│   ├── benchmark_runner.py         # Concurrency ramp; asyncio.wait partial-result collection
│   ├── metrics_collector.py        # EnrichedMetrics; fitness scoring; OOM penalty
│   ├── log_analyzer.py             # 14 error-class patterns; startup timing; OOM detection
│   ├── report_generator.py         # Writes YAML recipe + shell script + Markdown report
│   ├── coordinator.py              # Multi-node HTTP dispatch (not used in single-node mode)
│   ├── gpu_allocator.py            # CUDA_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES slot management
│   ├── port_allocator.py           # Port pool for concurrent vLLM instances
│   └── logger.py                   # Structured logging (console + JSONL)
│
├── configs/
│   ├── oceantune.yaml              # Main config — edit this to change model, GPU, search settings
│   ├── gpu_profiles.yaml           # Per-GPU: VRAM, Docker image, env vars, recommended settings
│   ├── models.yaml                 # Per-model: architecture, GPU requirements, extra vLLM flags
│   ├── search_space.yaml           # Stage 1: 20 tunable vLLM flag parameters
│   ├── kernel_search_space.yaml    # Stage 2: 15 kernel-level parameters
│   └── inference_models.yaml       # DO Serverless Inference model registry
│
├── docs/
│   └── architecture_stage1.md      # Detailed Mermaid flowchart of the Stage 1 loop
│
├── node/
│   ├── node_server.py              # FastAPI node server for multi-GPU-droplet deployments
│   └── node_worker.py              # Job runner on each node
│
├── storage/
│   ├── logs/                       # Per-session JSONL logs (gitignored)
│   └── results/                    # Output recipes, scripts, reports (gitignored)
│
└── tests/
    ├── test_search_space.py        # 66 tests
    ├── test_vllm_server.py         # 50 tests
    ├── test_benchmark_runner.py    # 53 tests
    ├── test_log_analyzer.py        # 36 tests
    └── test_metrics_collector.py   # 32 tests — total: 238 passed
```

---

## Configuration reference

All settings live in `configs/oceantune.yaml`. Secrets must come from environment variables.

### Full annotated config

```yaml
# ── Target model ──────────────────────────────────────────────────────────────
# Any Hugging Face model ID. Override with: --model flag  or  OCEANTUNE_MODEL_ID
model_id: "Qwen/Qwen2.5-7B-Instruct"

# ── Target GPU ────────────────────────────────────────────────────────────────
# Must match a key in configs/gpu_profiles.yaml
# Options: H100 | H200 | B300 | MI300X | MI325X | MI350X
# Override with: --gpu flag  or  OCEANTUNE_GPU_TYPE
gpu_type: "H200"

# ── LLM agent settings ────────────────────────────────────────────────────────
agent:
  model: "auto"          # "auto" = highest-rated from inference_models.yaml
                         # or set a specific model: "anthropic-claude-4.5-sonnet"
  max_tokens: 4096       # token budget per LLM call
  temperature: 0.3       # lower = more deterministic proposals
  timeout_sec: 120       # HTTP timeout per LLM call

# ── MongoDB ───────────────────────────────────────────────────────────────────
database:
  uri: ""                # set via MONGO_URI env var
  name: "oceantune"      # database name (override via OCEANTUNE_DB)

# ── GPU nodes ─────────────────────────────────────────────────────────────────
# For single-GPU: leave gpu_indices as [0]
# For multi-GPU: add all GPU indices, e.g. [0, 1, 2, 3] for 4 GPUs
nodes:
  - host: "localhost"
    node_port: 9000
    gpu_type: "H200"
    gpu_indices: [0]

# ── vLLM server ───────────────────────────────────────────────────────────────
vllm:
  startup_timeout_sec: 1200   # large models can take 10–20 min to load
  docker_image: ""            # leave empty to use the GPU profile default
                              # override: export VLLM_IMAGE="vllm/vllm-openai:v0.18.1-cu130"

# ── Benchmark settings ────────────────────────────────────────────────────────
benchmark:
  concurrency_levels: [1, 2, 4, 8, 16, 32, 64, 128]  # requests in flight per level
  num_prompts: 30         # total requests per concurrency level
  duration_sec: 60        # time budget per level

# ── Optimiser settings ────────────────────────────────────────────────────────
optimiser:
  generations: 10         # number of search iterations (more = better results)
  population_size: 10     # candidates sampled per generation
  primary_metric: "throughput"   # throughput | p95_latency | ttft | tpot
                                 # override: export OCEANTUNE_PRIMARY_METRIC=p95_latency

# ── Context configs ───────────────────────────────────────────────────────────
# Each [input_tokens, output_tokens] pair is benchmarked independently.
# Add more pairs to test different workload profiles.
context_configs:
  - [1024, 1024]    # typical chat
  - [1024, 4096]    # long generation
```

### Key environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MONGO_URI` | **Yes** | Full MongoDB connection string |
| `DO_INFERENCE_KEY` | Recommended | DO Serverless Inference API key (also: `AGENT_API_KEY`) |
| `DO_INFERENCE_ENDPOINT` | Recommended | Inference base URL (also: `AGENT_ENDPOINT`) |
| `DO_INFERENCE_MODEL` | Recommended | Model for agents, e.g. `anthropic-claude-4.5-sonnet` (also: `AGENT_MODEL`) |
| `HF_TOKEN` | For gated models | Hugging Face access token |
| `OCEANTUNE_MODEL_ID` | No | Override `model_id` from YAML |
| `OCEANTUNE_GPU_TYPE` | No | Override `gpu_type` from YAML |
| `OCEANTUNE_PRIMARY_METRIC` | No | Override fitness metric |
| `VLLM_IMAGE` | No | Override Docker image for vLLM |

> Without `DO_INFERENCE_KEY`, all four agents fall back to deterministic (non-LLM) behaviour: the Planner cycles through a curated list of single-parameter variations instead of reasoning about the bottleneck.

---

## MongoDB collections

| Collection | Purpose | Key fields |
|------------|---------|------------|
| `sessions` | One document per optimisation run | `model_id`, `gpu_type`, `status`, `created_at` |
| `configs` | Candidate configs queue | `fingerprint`, `flags`, `status` (`pending→running→done/failed`) |
| `benchmark_runs` | All benchmark results | `flags`, `levels[]`, `enriched_metrics`, `fitness_score` |
| `kernel_runs` | Stage 2 kernel search results | `kernel_config`, `fitness_score`, `llm_reasoning` |
| `nodes` | GPU droplet heartbeats (multi-node) | `host`, `gpu_type`, `last_seen` |

`benchmark_runs.levels` is a list with one entry per concurrency level tested:

```json
{
  "concurrency": 64,
  "output_tokens_per_sec": 5407.3,
  "p95_latency_ms": 5647.2,
  "mean_ttft_ms": 4876.1,
  "error_rate": 0.0,
  "failed": false
}
```

---

## Test suite

All tests are mocked — no GPU, no live server, no Hugging Face token required.

```bash
# Run locally
pytest tests/ --asyncio-mode=auto -v

# Run in Docker (identical environment to CI)
docker compose run --rm tests
```

```
tests/test_search_space.py       66 passed   VLLMFlags, SearchSpace, ConfigValidator
tests/test_vllm_server.py        50 passed   Profile-driven server, AMD/NVIDIA env injection
tests/test_benchmark_runner.py   53 passed   Regex parsing, concurrency ramp, partial results
tests/test_log_analyzer.py       36 passed   14 error classes, startup timing, OOM detection
tests/test_metrics_collector.py  32 passed   Fitness scoring, GPU efficiency, primary metrics
─────────────────────────────────────────────────────────────────────────────────────────────
Total                           238 passed
```

---

## Multi-node GPU Droplet setup

For models that need more GPUs than a single machine has, run a Node Server on each GPU Droplet and add them to `configs/oceantune.yaml`:

```bash
# On each GPU Droplet
ssh root@YOUR_DROPLET_IP
git clone https://github.com/RithishRamesh-dev/oceantune-ai.git /opt/oceantune-ai
cd /opt/oceantune-ai && pip install -r requirements.txt

export MONGO_URI=mongodb+srv://...
export DO_INFERENCE_KEY=dop_v1_...
export HF_TOKEN=hf_...
export NODE_HOST=YOUR_DROPLET_IP    # reported back to the Coordinator for routing

python3 -m node.node_server \
    --port 9000 \
    --gpu-type H100 \
    --gpu-indices 0,1,2,3,4,5,6,7
```

Then in `configs/oceantune.yaml`:

```yaml
nodes:
  - host: 10.0.0.1
    node_port: 9000
    gpu_type: H100
    gpu_indices: [0, 1, 2, 3, 4, 5, 6, 7]
  - host: 10.0.0.2
    node_port: 9000
    gpu_type: H100
    gpu_indices: [0, 1, 2, 3, 4, 5, 6, 7]
```

Node Server API:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness + free GPU count |
| `/capacity` | GET | Total/free GPUs, free ports |
| `/jobs` | POST | Submit benchmark job, returns `job_id` |
| `/jobs/{job_id}` | GET | Poll status: `pending / running / done / failed` |
