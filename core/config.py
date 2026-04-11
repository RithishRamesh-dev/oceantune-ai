"""
core/config.py
--------------
Loads and validates OceanTune AI configuration from YAML files and
environment variables.

Config hierarchy (later overrides earlier):
  1. YAML file in configs/
  2. Environment variables  (prefixed OCEANTUNE_)
  3. Code-level defaults in this file

Usage
-----
    from core.config import load_config, OceanTuneConfig
    cfg = load_config()
    print(cfg.spaces.bucket)
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import yaml

from core.logger import get_logger, log_dict

log = get_logger("core.config")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
STORAGE_DIR = REPO_ROOT / "storage"


# ---------------------------------------------------------------------------
# Dataclasses — one per logical section
# ---------------------------------------------------------------------------

@dataclass
class DatabaseConfig:
    """MongoDB connection settings."""
    uri: str = field(default_factory=lambda: os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    name: str = "oceantune"
    collections: Dict[str, str] = field(default_factory=lambda: {
        "sessions": "sessions",
        "nodes": "nodes",
        "configs": "configs",
        "benchmark_runs": "benchmark_runs",
        "kernel_runs": "kernel_runs",
    })


@dataclass
class NodeConfig:
    """A single GPU droplet node the Coordinator can dispatch work to."""
    host: str = "localhost"
    node_port: int = 9000
    gpu_type: str = "H100"
    gpu_indices: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7])

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.node_port}"


@dataclass
class CoordinatorConfig:
    """Parallel-dispatch coordinator settings."""
    max_parallel_per_node: int = 2
    port_pool_start: int = 8000
    port_pool_end: int = 8099
    poll_interval_sec: int = 5
    max_retries: int = 2


@dataclass
class AgentConfig:
    """AI agent configuration — DigitalOcean Serverless Inference."""
    # "auto" = pick highest suitability_score from registry
    model: str = "auto"
    max_tokens: int = 4096
    timeout_sec: int = 120
    max_turns: int = 6
    temperature: float = 0.3
    # Credentials — ALWAYS from env vars
    inference_key: str = field(
        default_factory=lambda: os.getenv("DO_INFERENCE_KEY", "")
    )
    inference_endpoint: str = field(
        default_factory=lambda: os.getenv(
            "DO_INFERENCE_ENDPOINT",
            "https://inference.do-ai.run/v1"
        )
    )
    # Explicit model override (takes priority over "auto")
    inference_model: str = field(
        default_factory=lambda: os.getenv("DO_INFERENCE_MODEL", "")
    )


@dataclass
class SpacesConfig:
    """DigitalOcean Spaces (S3-compatible) settings."""
    bucket: str = "oceantune-results"
    region: str = "nyc3"
    endpoint_url: str = "https://nyc3.digitaloceanspaces.com"
    # Credentials are ALWAYS sourced from env vars — never stored in YAML
    access_key: str = field(default_factory=lambda: os.getenv("DO_SPACES_KEY", ""))
    secret_key: str = field(default_factory=lambda: os.getenv("DO_SPACES_SECRET", ""))


@dataclass
class VLLMConfig:
    """Defaults for the vLLM server process."""
    host: str = "0.0.0.0"
    port: int = 8000
    startup_timeout_sec: int = 300       # 5 min; large models take time
    health_check_interval_sec: int = 5


@dataclass
class BenchmarkConfig:
    """Benchmark engine defaults."""
    concurrency_levels: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64]
    )
    num_prompts: int = 200               # total prompts per concurrency level
    input_len: int = 1024                # tokens in
    output_len: int = 1024               # tokens out
    duration_sec: int = 60               # max run time per concurrency level


@dataclass
class OptimiserConfig:
    """Optimisation engine settings."""
    strategy: str = "evolutionary"       # evolutionary | grid | random | bayesian
    population_size: int = 10
    generations: int = 5
    mutation_rate: float = 0.2
    elite_fraction: float = 0.2
    primary_metric: str = "throughput"   # throughput | p95_latency | ttft


@dataclass
class OceanTuneConfig:
    """Top-level config object passed throughout the system."""
    model_id: str = "deepseek-ai/DeepSeek-V3.2"
    gpu_type: str = "H100"
    hf_token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))

    agent: AgentConfig = field(default_factory=AgentConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    nodes: List[NodeConfig] = field(default_factory=lambda: [NodeConfig()])
    coordinator: CoordinatorConfig = field(default_factory=CoordinatorConfig)
    spaces: SpacesConfig = field(default_factory=SpacesConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    optimiser: OptimiserConfig = field(default_factory=OptimiserConfig)

    # Context-length pairs to evaluate  [(input_tokens, output_tokens), ...]
    context_configs: List[tuple] = field(default_factory=lambda: [
        (1024, 1024),
        (1024, 4096),
        (1024, 8192),
        (2048, 8192),
        (4096, 16384),
        (8192, 32768),
    ])


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    if not path.exists():
        log_dict(log, "warning", "Config file not found, using defaults", path=str(path))
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    log_dict(log, "info", "Loaded config", path=str(path), keys=list(data.keys()))
    return data


def _apply_env_overrides(cfg: OceanTuneConfig) -> None:
    """
    Apply environment-variable overrides.

    Supported env vars:
        OCEANTUNE_MODEL_ID
        OCEANTUNE_GPU_TYPE
        OCEANTUNE_PORT
        OCEANTUNE_STRATEGY
        OCEANTUNE_PRIMARY_METRIC
        DO_SPACES_KEY
        DO_SPACES_SECRET
        HF_TOKEN
    """
    if v := os.getenv("OCEANTUNE_MODEL_ID"):
        cfg.model_id = v
        log_dict(log, "info", "Env override: model_id", value=v)
    if v := os.getenv("OCEANTUNE_GPU_TYPE"):
        cfg.gpu_type = v
        log_dict(log, "info", "Env override: gpu_type", value=v)
    if v := os.getenv("OCEANTUNE_PORT"):
        cfg.vllm.port = int(v)
    if v := os.getenv("OCEANTUNE_STRATEGY"):
        cfg.optimiser.strategy = v
    if v := os.getenv("OCEANTUNE_PRIMARY_METRIC"):
        cfg.optimiser.primary_metric = v


def load_config(override_path: Optional[Path] = None) -> OceanTuneConfig:
    """
    Build and return an OceanTuneConfig.

    Parameters
    ----------
    override_path : Path, optional
        Path to a custom YAML config file.  Defaults to
        configs/oceantune.yaml if it exists.

    Returns
    -------
    OceanTuneConfig
    """
    cfg = OceanTuneConfig()

    # Try loading the main YAML
    yaml_path = override_path or (CONFIGS_DIR / "oceantune.yaml")
    raw = _load_yaml(yaml_path)

    # Apply YAML values — only the keys that exist in the file
    if "model_id" in raw:
        cfg.model_id = raw["model_id"]
    if "gpu_type" in raw:
        cfg.gpu_type = raw["gpu_type"]

    if "agent" in raw:
        a = raw["agent"]
        cfg.agent.model = a.get("model", cfg.agent.model)
        cfg.agent.max_tokens = a.get("max_tokens", cfg.agent.max_tokens)
        cfg.agent.timeout_sec = a.get("timeout_sec", cfg.agent.timeout_sec)
        cfg.agent.max_turns = a.get("max_turns", cfg.agent.max_turns)
        cfg.agent.temperature = a.get("temperature", cfg.agent.temperature)

    if "spaces" in raw:
        s = raw["spaces"]
        cfg.spaces.bucket = s.get("bucket", cfg.spaces.bucket)
        cfg.spaces.region = s.get("region", cfg.spaces.region)
        cfg.spaces.endpoint_url = s.get("endpoint_url", cfg.spaces.endpoint_url)

    if "vllm" in raw:
        v = raw["vllm"]
        cfg.vllm.port = v.get("port", cfg.vllm.port)
        cfg.vllm.startup_timeout_sec = v.get(
            "startup_timeout_sec", cfg.vllm.startup_timeout_sec
        )

    if "benchmark" in raw:
        b = raw["benchmark"]
        cfg.benchmark.concurrency_levels = b.get(
            "concurrency_levels", cfg.benchmark.concurrency_levels
        )
        cfg.benchmark.num_prompts = b.get("num_prompts", cfg.benchmark.num_prompts)
        cfg.benchmark.input_len = b.get("input_len", cfg.benchmark.input_len)
        cfg.benchmark.output_len = b.get("output_len", cfg.benchmark.output_len)

    if "optimiser" in raw:
        o = raw["optimiser"]
        cfg.optimiser.strategy = o.get("strategy", cfg.optimiser.strategy)
        cfg.optimiser.population_size = o.get(
            "population_size", cfg.optimiser.population_size
        )
        cfg.optimiser.generations = o.get("generations", cfg.optimiser.generations)
        cfg.optimiser.primary_metric = o.get(
            "primary_metric", cfg.optimiser.primary_metric
        )

    if "database" in raw:
        d = raw["database"]
        cfg.database.uri = d.get("uri", cfg.database.uri)
        cfg.database.name = d.get("name", cfg.database.name)
        if "collections" in d:
            cfg.database.collections.update(d["collections"])

    if "nodes" in raw:
        cfg.nodes = [
            NodeConfig(
                host=n.get("host", "localhost"),
                node_port=n.get("node_port", 9000),
                gpu_type=n.get("gpu_type", "H100"),
                gpu_indices=n.get("gpu_indices", [0]),
            )
            for n in raw["nodes"]
        ]

    if "coordinator" in raw:
        c = raw["coordinator"]
        cfg.coordinator.max_parallel_per_node = c.get(
            "max_parallel_per_node", cfg.coordinator.max_parallel_per_node
        )
        cfg.coordinator.port_pool_start = c.get(
            "port_pool_start", cfg.coordinator.port_pool_start
        )
        cfg.coordinator.port_pool_end = c.get(
            "port_pool_end", cfg.coordinator.port_pool_end
        )
        cfg.coordinator.poll_interval_sec = c.get(
            "poll_interval_sec", cfg.coordinator.poll_interval_sec
        )
        cfg.coordinator.max_retries = c.get("max_retries", cfg.coordinator.max_retries)

    if "context_configs" in raw:
        cfg.context_configs = [tuple(pair) for pair in raw["context_configs"]]

    # Env vars always win
    _apply_env_overrides(cfg)

    _validate(cfg)
    log_dict(
        log, "info", "Config ready",
        model=cfg.model_id,
        gpu=cfg.gpu_type,
        strategy=cfg.optimiser.strategy,
    )
    return cfg


def _validate(cfg: OceanTuneConfig) -> None:
    """Raise ValueError for obviously invalid configs."""
    valid_strategies = {"evolutionary", "grid", "random", "bayesian"}
    if cfg.optimiser.strategy not in valid_strategies:
        raise ValueError(
            f"Unknown optimisation strategy '{cfg.optimiser.strategy}'. "
            f"Choose one of: {valid_strategies}"
        )

    # All supported GPU SKUs
    valid_gpus = {
        # NVIDIA
        "H100", "H200", "B300",
        # AMD
        "MI300X", "MI325X", "MI350X",
    }
    if cfg.gpu_type not in valid_gpus:
        raise ValueError(
            f"Unknown gpu_type '{cfg.gpu_type}'. "
            f"Choose one of: {sorted(valid_gpus)}"
        )

    if cfg.vllm.port < 1024 or cfg.vllm.port > 65535:
        raise ValueError(f"vLLM port {cfg.vllm.port} is out of range [1024, 65535]")

    if not cfg.model_id:
        raise ValueError("model_id must not be empty")

    log.debug("Config validation passed")