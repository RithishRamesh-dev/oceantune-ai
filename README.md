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
python oceantune.py validate-config

# 4. Run the optimiser
python oceantune.py run --model mistral --gpu A100
```

---

## Repository layout

```
oceantune-ai/
├── oceantune.py          # CLI entry point
├── requirements.txt      # Pinned dependencies
├── core/                 # Shared modules
│   ├── logger.py         # Structured logging (console + JSONL file)
│   ├── config.py         # YAML + env-var config loader
│   ├── experiment_runner.py
│   ├── benchmark_runner.py
│   ├── metrics_collector.py
│   ├── log_analyzer.py
│   ├── optimizer.py
│   └── recipe_generator.py
├── agents/
│   ├── controller_agent.py   # Orchestrates the full pipeline
│   └── research_agent.py     # Scrapes papers / GitHub for new flag ideas
├── configs/
│   ├── oceantune.yaml        # Main config (edit this)
│   ├── models.yaml           # Supported model registry
│   ├── gpu_profiles.yaml     # Per-GPU vLLM defaults
│   └── search_space.yaml     # Parameter ranges for optimisation
├── storage/
│   ├── logs/                 # Per-session JSONL logs
│   └── results/              # Benchmark results and recipes
├── scripts/                  # Shell helpers
└── ci/                       # GitHub Actions workflows
```

---

## Configuration

Edit [configs/oceantune.yaml](configs/oceantune.yaml) to change the model, GPU, and search settings. Secrets are always read from environment variables — never stored in YAML.

| Key | Default | Description |
|-----|---------|-------------|
| `model_id` | `mistralai/Mistral-7B-Instruct-v0.2` | Hugging Face model ID |
| `gpu_type` | `A100` | GPU profile (`A100`, `H100`, `A10G`, `RTX4090`, `MI300X`) |
| `optimiser.strategy` | `evolutionary` | Search strategy |
| `optimiser.primary_metric` | `throughput` | Metric to maximise (`throughput`, `p95_latency`, `ttft`) |
| `optimiser.generations` | `5` | Number of optimisation generations |
| `optimiser.population_size` | `10` | Candidates per generation |

---

## CLI commands

```bash
python oceantune.py --help
python oceantune.py validate-config          # check YAML + env vars
python oceantune.py run --dry-run            # validate only, no GPU needed
python oceantune.py run --model mistral      # run full pipeline
python oceantune.py run --strategy bayesian  # override search strategy
python oceantune.py info                     # print system / GPU info
```

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

The system is built incrementally. Each step is independently testable and leaves all prior steps intact.

### Step 1 — Repository setup & project skeleton
**Status: complete**

Creates the full directory tree, logging system, config loader, placeholder modules, `requirements.txt`, and this README. Every subsequent step imports from `core.logger` and `core.config`.

Key files: [core/logger.py](core/logger.py), [core/config.py](core/config.py), [configs/oceantune.yaml](configs/oceantune.yaml), [oceantune.py](oceantune.py)

#### Validation

**1. Directory tree — 25 files created**
```
$ find oceantune-ai -not -path '*/.git/*' -not -path '*/.venv/*' -type f | sort
agents/__init__.py
agents/controller_agent.py
agents/research_agent.py
configs/gpu_profiles.yaml
configs/models.yaml
configs/oceantune.yaml
configs/search_space.yaml
core/__init__.py
core/benchmark_runner.py
core/config.py
core/experiment_runner.py
core/log_analyzer.py
core/logger.py
core/metrics_collector.py
core/optimizer.py
core/recipe_generator.py
oceantune.py
requirements.txt
storage/logs/oceantune_20260402_013946.jsonl
...
```

**2. Logger writes JSONL**
```
$ cat storage/logs/oceantune_20260402_013946.jsonl | head -2
{"ts": "2026-04-02T01:39:46.004091+00:00", "level": "INFO", "logger": "core.logger", "msg": "Logging initialised", "log_file": "storage/logs/oceantune_20260402_013946.jsonl"}
{"ts": "2026-04-02T01:39:46.013446+00:00", "level": "INFO", "logger": "core.config", "msg": "Loaded config", "path": "configs/oceantune.yaml", "keys": ["model_id", "gpu_type", "spaces", "vllm", "benchmark", "optimiser", "context_configs"]}
```

**3. Config loads from YAML — exits 0**
```
$ python oceantune.py validate-config

Config valid

  Model ID   : mistralai/Mistral-7B-Instruct-v0.2
  GPU type   : A100
  Strategy   : evolutionary
  Metric     : throughput
  Generations: 5
  Pop. size  : 10
  Concurrency: [1, 2, 4, 8, 16, 32, 64]
  Contexts   : 6 pairs (...)

  Spaces bucket : oceantune-results (nyc3)
  vLLM port     : 8000

  HF_TOKEN set       : NOT SET
  DO_SPACES_KEY set  : NOT SET
```

**4. Env var override**
```
$ OCEANTUNE_MODEL_ID=foo python oceantune.py validate-config | grep "Model ID"
  Model ID   : foo
```

**5. Invalid strategy rejected**
```
# Set strategy: "bad" in oceantune.yaml
$ python oceantune.py validate-config
ValueError: Unknown optimisation strategy 'bad'. Choose one of: {'grid', 'random', 'evolutionary', 'bayesian'}
```

---

### Step 2 — Core config system
**Status: complete**

Builds the typed config system the optimiser operates on: four parameter types (`ChoiceParam`, `RangeIntParam`, `RangeFloatParam`, `BoolFlagParam`), a `VLLMFlags` dataclass with fingerprinting and CLI-arg generation, `SearchSpace` with random/seeded/grid sampling and evolutionary crossover, and `ConfigValidator` with 6 hardware-constraint checks. The optimiser (Step 6) calls all of these.

Key files: [core/search_space.py](core/search_space.py), [tests/test_search_space.py](tests/test_search_space.py)

#### Validation

**1. Search space summary**
```
$ python -c "from core.search_space import SearchSpace; ss = SearchSpace.load(); print(ss.summary())"

Search space (13 parameters):
  tensor_parallel_size: choices=[1, 2, 4, 8] default=1
  pipeline_parallel_size: choices=[1, 2, 4] default=1
  gpu_memory_utilization: [0.7..0.95 step=0.05] default=0.9
  max_num_seqs: choices=[64, 128, 256, 512] default=256
  max_num_batched_tokens: choices=[2048, 4096, 8192, 16384, 32768] default=8192
  kv_cache_dtype: choices=['auto', 'fp8_e5m2'] default=auto
  block_size: choices=[8, 16, 32] default=16
  quantization: choices=[None, 'awq', 'gptq', 'squeezellm'] default=None
  scheduler_delay_factor: [0.0..0.5 step=0.1] default=0.0
  enable_chunked_prefill: bool default=False
  use_v2_block_manager: bool default=False
  speculative_draft_model: choices=[None] default=None
  dtype: choices=['float16', 'bfloat16'] default=float16
  Total discrete configs: ~1,658,880
```

**2. Test suite — 49/49 passed**
```
$ pytest tests/test_search_space.py -v

tests/test_search_space.py::TestChoiceParam::test_sample_is_in_values PASSED
tests/test_search_space.py::TestChoiceParam::test_mutate_rate_zero_returns_current PASSED
tests/test_search_space.py::TestChoiceParam::test_mutate_rate_one_always_changes PASSED
tests/test_search_space.py::TestChoiceParam::test_neighbours_excludes_current PASSED
tests/test_search_space.py::TestRangeIntParam::test_sample_in_range PASSED
tests/test_search_space.py::TestRangeIntParam::test_neighbours_step PASSED
tests/test_search_space.py::TestRangeIntParam::test_neighbours_at_min PASSED
tests/test_search_space.py::TestRangeFloatParam::test_sample_in_range PASSED
tests/test_search_space.py::TestRangeFloatParam::test_values_includes_endpoints PASSED
tests/test_search_space.py::TestRangeFloatParam::test_neighbours PASSED
tests/test_search_space.py::TestBoolFlagParam::test_mutate_flips PASSED
tests/test_search_space.py::TestBoolFlagParam::test_neighbours PASSED
tests/test_search_space.py::TestVLLMFlags::test_fingerprint_is_stable PASSED
tests/test_search_space.py::TestVLLMFlags::test_fingerprint_changes_on_mutation PASSED
tests/test_search_space.py::TestVLLMFlags::test_copy_is_independent PASSED
tests/test_search_space.py::TestVLLMFlags::test_equality PASSED
tests/test_search_space.py::TestVLLMFlags::test_to_dict_roundtrip PASSED
tests/test_search_space.py::TestVLLMFlags::test_to_vllm_args_basic PASSED
tests/test_search_space.py::TestVLLMFlags::test_to_vllm_args_quantization PASSED
tests/test_search_space.py::TestVLLMFlags::test_to_vllm_args_amd PASSED
tests/test_search_space.py::TestVLLMFlags::test_chunked_prefill_flag PASSED
tests/test_search_space.py::TestVLLMFlags::test_no_quantization_no_flag PASSED
tests/test_search_space.py::TestSearchSpace::test_loads_from_yaml PASSED
tests/test_search_space.py::TestSearchSpace::test_has_expected_params PASSED
tests/test_search_space.py::TestSearchSpace::test_default_flags_are_valid PASSED
tests/test_search_space.py::TestSearchSpace::test_sample_random_returns_valid_object PASSED
tests/test_search_space.py::TestSearchSpace::test_sample_population_size PASSED
tests/test_search_space.py::TestSearchSpace::test_sample_population_unique PASSED
tests/test_search_space.py::TestSearchSpace::test_mutate_returns_new_object PASSED
tests/test_search_space.py::TestSearchSpace::test_mutate_run_id_updated PASSED
tests/test_search_space.py::TestSearchSpace::test_crossover_inherits_parents PASSED
tests/test_search_space.py::TestSearchSpace::test_grid_neighbours_nonempty PASSED
tests/test_search_space.py::TestSearchSpace::test_grid_neighbours_differ_by_one_param PASSED
tests/test_search_space.py::TestSearchSpace::test_size_is_positive PASSED
tests/test_search_space.py::TestSearchSpace::test_seeded_population_respects_gpu_limits PASSED
tests/test_search_space.py::TestSearchSpace::test_summary_contains_param_names PASSED
tests/test_search_space.py::TestConfigValidator::test_valid_config_passes PASSED
tests/test_search_space.py::TestConfigValidator::test_tp_exceeds_gpu_max_fails PASSED
tests/test_search_space.py::TestConfigValidator::test_tp_not_power_of_two_fails PASSED
tests/test_search_space.py::TestConfigValidator::test_memory_util_too_high_fails PASSED
tests/test_search_space.py::TestConfigValidator::test_memory_util_too_low_fails PASSED
tests/test_search_space.py::TestConfigValidator::test_batched_tokens_less_than_seqs_fails PASSED
tests/test_search_space.py::TestConfigValidator::test_speculative_without_v2_block_manager_fails PASSED
tests/test_search_space.py::TestConfigValidator::test_speculative_with_v2_passes PASSED
tests/test_search_space.py::TestConfigValidator::test_bf16_on_a10g_fails PASSED
tests/test_search_space.py::TestConfigValidator::test_is_valid_shortcut PASSED
tests/test_search_space.py::TestFlagsFromDict::test_roundtrip PASSED
tests/test_search_space.py::TestFlagsFromDict::test_unknown_keys_ignored PASSED
tests/test_search_space.py::TestFlagsFromDict::test_missing_keys_use_defaults PASSED

49 passed in 0.06s
```

---

### Step 3 — vLLM server runner
**Status: complete**

Fully async process manager for the vLLM server lifecycle. Launches vLLM with flags from `VLLMFlags.to_vllm_args()`, streams stdout/stderr into a 500-line rotating log buffer, polls `/health` with exponential backoff, and tears down cleanly via SIGTERM → SIGKILL. A 5-class failure hierarchy (`OOMError`, `StartupTimeout`, `PortConflict`, `CUDAError`, `ProcessCrash`) turns cryptic CUDA stderr into actionable exceptions. `scripts/run_vllm.sh` wraps the process with ulimits, NCCL tuning, PID files, and signal forwarding.

Key files: [core/vllm_server.py](core/vllm_server.py), [scripts/run_vllm.sh](scripts/run_vllm.sh), [tests/test_vllm_server.py](tests/test_vllm_server.py)

#### Validation

**Test suite — 49/49 new tests, 98/98 total (zero regressions)**

Ran on the DigitalOcean droplet (165.245.171.90) inside `python:3.12-slim` Docker container — no GPU, no vLLM installation required (all subprocess and HTTP calls are mocked):

```
$ docker run --rm -v /opt/oceantune-ai:/workspace -w /workspace python:3.12-slim \
    bash -c 'pip install pytest pytest-asyncio pyyaml httpx click --quiet && pytest tests/ -v'

platform linux -- Python 3.12.13, pytest-9.0.2
collected 98 items

tests/test_search_space.py::TestChoiceParam::test_sample_is_in_values PASSED
...
tests/test_vllm_server.py::TestClassifyLogFailure::test_oom_detected PASSED
tests/test_vllm_server.py::TestClassifyLogFailure::test_oom_takes_priority_over_cuda PASSED
tests/test_vllm_server.py::TestVLLMServerProperties::test_endpoint PASSED
tests/test_vllm_server.py::TestBuildCommand::test_tensor_parallel_in_args PASSED
tests/test_vllm_server.py::TestBuildEnv::test_amd_env_for_mi300x PASSED
tests/test_vllm_server.py::TestIsHealthy::test_healthy_on_200 PASSED
tests/test_vllm_server.py::TestIsHealthy::test_unhealthy_on_connection_error PASSED
tests/test_vllm_server.py::TestLogCapture::test_buffer_respects_maxlen PASSED
tests/test_vllm_server.py::TestStartStop::test_start_reaches_healthy_state PASSED
tests/test_vllm_server.py::TestStartStop::test_oom_in_logs_raises_oom_error PASSED
tests/test_vllm_server.py::TestStartStop::test_port_conflict_in_logs_raises_port_conflict PASSED
tests/test_vllm_server.py::TestStartStop::test_context_manager_stops_on_exception PASSED
tests/test_vllm_server.py::TestMakeServer::test_factory_port_override PASSED
tests/test_vllm_server.py::TestExceptionHierarchy::test_formatted_tail_truncates PASSED

98 passed in 0.21s
```

---

### Step 3 — Experiment runner
**Status: pending**

Composes the vLLM manager (Step 2) and benchmark runner (Step 4) into a single `run_experiment(config, vllm_flags) -> ExperimentResult` call. Handles retries on startup failure and persists raw results to `storage/results/`.

Key files: `core/experiment_runner.py`

---

### Step 4 — Benchmark engine
**Status: pending**

Drives the vLLM OpenAI-compatible endpoint with controlled concurrency ramps. Measures throughput (tokens/s), p50/p95/p99 latency, and time-to-first-token across all `context_configs`. Uses `asyncio` + `httpx` for accurate concurrent load.

Key files: `core/benchmark_runner.py`

---

### Step 5 — Metrics collector & log analyser
**Status: pending**

Parses benchmark output into structured `MetricsSnapshot` objects. Also tails vLLM's own logs to extract KV-cache hit rate, scheduler queue depth, and OOM events — signals the optimiser uses to avoid dead regions of the search space.

Key files: `core/metrics_collector.py`, `core/log_analyzer.py`

---

### Step 6 — Optimisation engine
**Status: pending**

Implements four search strategies over the parameter space defined in `configs/search_space.yaml`:

- **Evolutionary** — selection, crossover, mutation across generations
- **Random** — uniform sampling baseline
- **Grid** — exhaustive over a reduced grid
- **Bayesian** — Gaussian-process surrogate with expected-improvement acquisition

Key files: `core/optimizer.py`

---

### Step 7 — Controller agent
**Status: pending**

Top-level orchestrator that wires Steps 2–6 into the full pipeline loop. Reads config, initialises the search strategy, drives generations, emits progress to the logger, and hands the best result to the recipe generator. Entry point for `python oceantune.py run`.

Key files: `agents/controller_agent.py`

---

### Step 8 — DigitalOcean Spaces storage
**Status: pending**

Uploads experiment results, logs, and final recipes to the configured DO Spaces bucket using `boto3`. Implements a local-first cache so re-runs skip already-completed experiments.

Key files: `core/storage.py`

---

### Step 9 — Recipe generator
**Status: pending**

Takes the best `ExperimentResult` and produces a human-readable, copy-paste-ready vLLM launch recipe: a shell script, a `docker run` command, and a JSON summary. Recipes are versioned by model + GPU + context profile.

Key files: `core/recipe_generator.py`

---

### Step 10 — GitHub Actions CI
**Status: pending**

Workflow that runs on push: installs dependencies, runs `validate-config`, executes the unit-test suite, and (on `main`) triggers a real optimisation run on a self-hosted GPU runner.

Key files: `ci/oceantune.yml`

---

### Step 11 — Multi-model & multi-GPU matrix
**Status: pending**

Extends the controller to fan out experiments across all entries in `configs/models.yaml` and `configs/gpu_profiles.yaml` in parallel, using `asyncio` task groups. Produces a comparison table of best recipes per (model, GPU) pair.

Key files: `agents/controller_agent.py` (extended), `core/matrix_runner.py`

---

### Step 12 — Speculative decoding support
**Status: pending**

Adds draft-model selection to the search space. The optimiser can now evaluate `speculative_draft_model` + `num_speculative_tokens` combinations and measure the acceptance rate vs. latency trade-off.

Key files: `core/spec_decoding.py`, `configs/search_space.yaml` (extended)

---

### Step 13 — Research agent
**Status: pending**

Periodically scrapes arXiv, vLLM's GitHub issues, and the vLLM changelog for new flags or techniques. Summarises findings using Claude and proposes additions to `configs/search_space.yaml` as a pull request.

Key files: `agents/research_agent.py`

---

## Supported models

| Alias | Hugging Face ID | Params |
|-------|-----------------|--------|
| `mistral` | `mistralai/Mistral-7B-Instruct-v0.2` | 7B |
| `deepseek_v3` | `deepseek-ai/DeepSeek-V3` | 671B |
| `qwen3_235b` | `Qwen/Qwen3-235B-Instruct` | 235B |
| `kimi_k2` | `moonshotai/Kimi-K2-Instruct` | 1T |
| `minimax_m2` | `MiniMaxAI/MiniMax-M2-Instruct` | 456B |

## Supported GPUs

| Profile | VRAM | Vendor |
|---------|------|--------|
| `A100` | 80 GB | NVIDIA |
| `H100` | 80 GB | NVIDIA |
| `A10G` | 24 GB | NVIDIA |
| `RTX4090` | 24 GB | NVIDIA |
| `MI300X` | 192 GB | AMD |
