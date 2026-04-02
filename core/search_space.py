"""
core/search_space.py
--------------------
Typed search-space system for OceanTune AI.

Handles:
  1. Loading search_space.yaml into strongly-typed Parameter objects.
  2. Representing a single vLLM configuration as a VLLMFlags dataclass.
  3. Sampling initial populations (random, grid, or GPU-profile-seeded).
  4. Mutating configs for evolutionary search.
  5. Validating configs to reject hardware-impossible combinations.

Supported GPU targets:
  NVIDIA: H100, H200, B300
  AMD:    MI300X, MI325X, MI350X
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, List, Optional

import yaml

from core.logger import get_logger, log_dict

log = get_logger("core.search_space")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SEARCH_SPACE_YAML = REPO_ROOT / "configs" / "search_space.yaml"
GPU_PROFILES_YAML = REPO_ROOT / "configs" / "gpu_profiles.yaml"

# ---------------------------------------------------------------------------
# GPU type sets
# ---------------------------------------------------------------------------
_AMD_GPU_TYPES = {"MI300X", "MI325X", "MI350X"}
_NVIDIA_GPU_TYPES = {"H100", "H200", "B300"}


# ===========================================================================
# 1.  Parameter types
# ===========================================================================

@dataclass
class ChoiceParam:
    """A parameter that takes one of a fixed list of values."""
    name: str
    values: List[Any]
    default: Any

    def sample(self) -> Any:
        return random.choice(self.values)

    def mutate(self, current: Any, mutation_rate: float) -> Any:
        if random.random() < mutation_rate:
            return random.choice(self.values)
        return current

    def neighbours(self, current: Any) -> List[Any]:
        return [v for v in self.values if v != current]


@dataclass
class RangeIntParam:
    """An integer parameter within [min, max] with a step size."""
    name: str
    min: int
    max: int
    step: int
    default: int

    def _values(self) -> List[int]:
        return list(range(self.min, self.max + 1, self.step))

    def sample(self) -> int:
        return random.choice(self._values())

    def mutate(self, current: int, mutation_rate: float) -> int:
        if random.random() < mutation_rate:
            return random.choice(self._values())
        delta = random.gauss(0, (self.max - self.min) * 0.1)
        raw = current + delta
        snapped = round((raw - self.min) / self.step) * self.step + self.min
        return int(max(self.min, min(self.max, snapped)))

    def neighbours(self, current: int) -> List[int]:
        vals = self._values()
        idx = vals.index(current) if current in vals else 0
        result = []
        if idx > 0:
            result.append(vals[idx - 1])
        if idx < len(vals) - 1:
            result.append(vals[idx + 1])
        return result


@dataclass
class RangeFloatParam:
    """A float parameter within [min, max] with a step size."""
    name: str
    min: float
    max: float
    step: float
    default: float

    def _values(self) -> List[float]:
        result, v = [], self.min
        while v <= self.max + 1e-9:
            result.append(round(v, 6))
            v += self.step
        return result

    def sample(self) -> float:
        return random.choice(self._values())

    def mutate(self, current: float, mutation_rate: float) -> float:
        if random.random() < mutation_rate:
            return random.choice(self._values())
        delta = random.gauss(0, (self.max - self.min) * 0.1)
        raw = current + delta
        steps = round((raw - self.min) / self.step)
        snapped = self.min + steps * self.step
        return round(max(self.min, min(self.max, snapped)), 6)

    def neighbours(self, current: float) -> List[float]:
        vals = self._values()
        closest = min(vals, key=lambda v: abs(v - current))
        idx = vals.index(closest)
        result = []
        if idx > 0:
            result.append(vals[idx - 1])
        if idx < len(vals) - 1:
            result.append(vals[idx + 1])
        return result


@dataclass
class BoolFlagParam:
    """A boolean flag (True / False)."""
    name: str
    default: bool

    def sample(self) -> bool:
        return random.choice([True, False])

    def mutate(self, current: bool, mutation_rate: float) -> bool:
        if random.random() < mutation_rate:
            return not current
        return current

    def neighbours(self, current: bool) -> List[bool]:
        return [not current]


AnyParam = ChoiceParam | RangeIntParam | RangeFloatParam | BoolFlagParam


# ===========================================================================
# 2.  VLLMFlags — a single configuration point in the search space
# ===========================================================================

@dataclass
class VLLMFlags:
    """
    Represents one set of vLLM server flags.

    Fields are grouped by vLLM config class:
      ParallelConfig, CacheConfig, ModelConfig, Scheduler, MoE, Stage-2
    """
    # ── ParallelConfig ────────────────────────────────────────────────────
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    enable_expert_parallel: bool = False
    data_parallel_size: int = 1
    distributed_executor_backend: str = "mp"

    # ── CacheConfig ───────────────────────────────────────────────────────
    gpu_memory_utilization: float = 0.90
    block_size: int = 16
    kv_cache_dtype: str = "auto"
    enable_prefix_caching: bool = False
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192

    # ── ModelConfig ───────────────────────────────────────────────────────
    dtype: str = "auto"
    quantization: Optional[str] = None
    max_model_len: int = 32768
    enforce_eager: bool = False
    load_format: str = "auto"
    trust_remote_code: bool = False

    # ── Scheduler ─────────────────────────────────────────────────────────
    scheduler_delay_factor: float = 0.0
    enable_chunked_prefill: bool = False

    # ── AttentionConfig ───────────────────────────────────────────────────
    attention_backend: str = "auto"

    # ── MoE / EP communication ────────────────────────────────────────────
    all2all_backend: str = "allgather_reducescatter"
    enable_dbo: bool = False

    # ── CPU offload ───────────────────────────────────────────────────────
    cpu_offload_gb: int = 0

    # ── Prefix caching ────────────────────────────────────────────────────
    prefix_caching_hash_algo: str = "sha256"

    # ── Stage-2: Speculative Decoding (Step 13) ───────────────────────────
    speculative_model: Optional[str] = None
    num_speculative_tokens: Optional[int] = None

    # ── Metadata (not part of vLLM CLI) ───────────────────────────────────
    run_id: str = field(default="", repr=False)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_vllm_args(self, model_id: str, gpu_type: str = "H100") -> List[str]:
        """Convert this config into a list of vLLM CLI arguments."""
        args = [
            "--model", model_id,
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--pipeline-parallel-size", str(self.pipeline_parallel_size),
            "--data-parallel-size", str(self.data_parallel_size),
            "--distributed-executor-backend", self.distributed_executor_backend,
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--block-size", str(self.block_size),
            "--kv-cache-dtype", self.kv_cache_dtype,
            "--max-num-seqs", str(self.max_num_seqs),
            "--max-num-batched-tokens", str(self.max_num_batched_tokens),
            "--dtype", self.dtype,
            "--max-model-len", str(self.max_model_len),
            "--load-format", self.load_format,
            "--scheduler-delay-factor", str(self.scheduler_delay_factor),
        ]

        # Bool flags — only emit when True
        if self.enable_expert_parallel:
            args.append("--enable-expert-parallel")
        if self.enable_prefix_caching:
            args.append("--enable-prefix-caching")
        if self.enforce_eager:
            args.append("--enforce-eager")
        if self.trust_remote_code:
            args.append("--trust-remote-code")
        if self.enable_chunked_prefill:
            args.append("--enable-chunked-prefill")
        if self.enable_dbo:
            args.append("--enable-dbo")

        # Optional value flags
        if self.quantization:
            args += ["--quantization", self.quantization]
        if self.attention_backend != "auto":
            args += ["--attention-backend", self.attention_backend]
        if self.all2all_backend != "allgather_reducescatter":
            args += ["--all2all-backend", self.all2all_backend]
        if self.cpu_offload_gb > 0:
            args += ["--cpu-offload-gb", str(self.cpu_offload_gb)]
        if self.enable_prefix_caching and self.prefix_caching_hash_algo != "sha256":
            args += ["--prefix-caching-hash-algo", self.prefix_caching_hash_algo]

        # Stage-2: speculative decoding
        if self.speculative_model:
            args += ["--speculative-model", self.speculative_model]
        if self.num_speculative_tokens is not None:
            args += ["--num-speculative-tokens", str(self.num_speculative_tokens)]

        # AMD ROCm device flag
        if gpu_type in _AMD_GPU_TYPES:
            args += ["--device", "rocm"]

        return args

    def fingerprint(self) -> str:
        """Return a short, stable hash of this config's flags."""
        d = {k: v for k, v in asdict(self).items() if k != "run_id"}
        canonical = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha1(canonical.encode()).hexdigest()[:12]

    def copy(self) -> "VLLMFlags":
        return copy.deepcopy(self)

    def __hash__(self):
        return hash(self.fingerprint())

    def __eq__(self, other):
        if not isinstance(other, VLLMFlags):
            return False
        return self.fingerprint() == other.fingerprint()


# ===========================================================================
# 3.  SearchSpace
# ===========================================================================

class SearchSpace:
    """
    Loads search_space.yaml and provides typed access to every parameter.

    Usage
    -----
        ss = SearchSpace.load()
        flags = ss.sample_random()
        mutated = ss.mutate(flags, mutation_rate=0.2)
    """

    def __init__(self, params: dict[str, AnyParam]):
        self._params: dict[str, AnyParam] = params
        log_dict(log, "info", "SearchSpace ready", num_params=len(params))

    @classmethod
    def load(cls, path: Path = SEARCH_SPACE_YAML) -> "SearchSpace":
        """Parse search_space.yaml and return a SearchSpace instance."""
        if not path.exists():
            raise FileNotFoundError(f"search_space.yaml not found: {path}")

        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        params: dict[str, AnyParam] = {}
        space = raw.get("search_space", {})

        for name, spec in space.items():
            ptype = spec.get("type")
            default = spec.get("default")

            if ptype == "choice":
                params[name] = ChoiceParam(name=name, values=spec["values"], default=default)
            elif ptype == "range_int":
                params[name] = RangeIntParam(
                    name=name, min=int(spec["min"]), max=int(spec["max"]),
                    step=int(spec.get("step", 1)), default=int(default),
                )
            elif ptype == "range_float":
                params[name] = RangeFloatParam(
                    name=name, min=float(spec["min"]), max=float(spec["max"]),
                    step=float(spec.get("step", 0.05)), default=float(default),
                )
            elif ptype == "bool_flag":
                params[name] = BoolFlagParam(name=name, default=bool(default))
            else:
                log.warning(f"Unknown parameter type '{ptype}' for '{name}' — skipping")

        log_dict(log, "info", "Loaded search space", path=str(path), params=list(params))
        return cls(params)

    def default_flags(self) -> VLLMFlags:
        """Return a VLLMFlags with every parameter set to its YAML default."""
        flags = VLLMFlags()
        for name, param in self._params.items():
            if hasattr(flags, name):
                setattr(flags, name, param.default)
        flags.run_id = flags.fingerprint()
        return flags

    def sample_random(self) -> VLLMFlags:
        """Return a uniformly random VLLMFlags from the full search space."""
        flags = VLLMFlags()
        for name, param in self._params.items():
            if hasattr(flags, name):
                setattr(flags, name, param.sample())
        flags.run_id = flags.fingerprint()
        return flags

    def sample_population(self, size: int) -> List[VLLMFlags]:
        """Return *size* unique random configs."""
        seen: set[str] = set()
        population: List[VLLMFlags] = []
        max_attempts = size * 20

        for _ in range(max_attempts):
            if len(population) >= size:
                break
            candidate = self.sample_random()
            fp = candidate.fingerprint()
            if fp not in seen:
                seen.add(fp)
                population.append(candidate)

        if len(population) < size:
            log.warning(
                f"Could only generate {len(population)} unique configs "
                f"(requested {size}) — search space may be small."
            )
        log_dict(log, "info", "Population sampled", size=len(population))
        return population

    def sample_seeded(
        self,
        gpu_type: str,
        size: int,
        gpu_profiles_path: Path = GPU_PROFILES_YAML,
    ) -> List[VLLMFlags]:
        """Return a population seeded from GPU-profile hints, lightly perturbed."""
        profile = _load_gpu_profile(gpu_type, gpu_profiles_path)
        seed = self.default_flags()

        if "default_gpu_memory_utilization" in profile:
            seed.gpu_memory_utilization = float(profile["default_gpu_memory_utilization"])
        if "max_tensor_parallel" in profile:
            max_tp = int(profile["max_tensor_parallel"])
            seed.tensor_parallel_size = 1
            tp_param = self._params.get("tensor_parallel_size")
            if isinstance(tp_param, ChoiceParam):
                tp_param.values = [v for v in tp_param.values if v <= max_tp]

        seed.run_id = seed.fingerprint()
        population: List[VLLMFlags] = [seed]

        seen = {seed.fingerprint()}
        for _ in range(size * 30):
            if len(population) >= size:
                break
            candidate = self.mutate(seed, mutation_rate=0.3)
            fp = candidate.fingerprint()
            if fp not in seen:
                seen.add(fp)
                population.append(candidate)

        log_dict(log, "info", "Seeded population ready", gpu=gpu_type, size=len(population))
        return population

    def mutate(self, flags: VLLMFlags, mutation_rate: float = 0.2) -> VLLMFlags:
        """Return a mutated copy of *flags*. Original is never modified."""
        mutated = flags.copy()
        for name, param in self._params.items():
            if hasattr(mutated, name):
                current = getattr(mutated, name)
                setattr(mutated, name, param.mutate(current, mutation_rate))
        mutated.run_id = mutated.fingerprint()
        return mutated

    def crossover(self, parent_a: VLLMFlags, parent_b: VLLMFlags) -> VLLMFlags:
        """Single-point crossover between two parents."""
        child = VLLMFlags()
        for name in self._params:
            if hasattr(child, name):
                src = parent_a if random.random() < 0.5 else parent_b
                setattr(child, name, getattr(src, name))
        child.run_id = child.fingerprint()
        return child

    def grid_neighbours(self, flags: VLLMFlags) -> List[VLLMFlags]:
        """Return all single-step neighbours for grid search."""
        neighbours: List[VLLMFlags] = []
        seen: set[str] = {flags.fingerprint()}

        for name, param in self._params.items():
            if not hasattr(flags, name):
                continue
            current = getattr(flags, name)
            for new_val in param.neighbours(current):
                candidate = flags.copy()
                setattr(candidate, name, new_val)
                candidate.run_id = candidate.fingerprint()
                if candidate.run_id not in seen:
                    seen.add(candidate.run_id)
                    neighbours.append(candidate)

        return neighbours

    def size(self) -> int:
        """Return the total number of configs in the discrete search space."""
        total = 1
        for param in self._params.values():
            if isinstance(param, ChoiceParam):
                total *= len(param.values)
            elif isinstance(param, RangeIntParam):
                total *= (param.max - param.min) // param.step + 1
            elif isinstance(param, RangeFloatParam):
                total *= round((param.max - param.min) / param.step) + 1
            elif isinstance(param, BoolFlagParam):
                total *= 2
        return total

    def summary(self) -> str:
        """Return a human-readable summary of the search space."""
        lines = [f"Search space ({len(self._params)} parameters):"]
        for name, param in self._params.items():
            if isinstance(param, ChoiceParam):
                lines.append(f"  {name}: choices={param.values} default={param.default}")
            elif isinstance(param, (RangeIntParam, RangeFloatParam)):
                lines.append(
                    f"  {name}: [{param.min}..{param.max} step={param.step}]"
                    f" default={param.default}"
                )
            elif isinstance(param, BoolFlagParam):
                lines.append(f"  {name}: bool default={param.default}")
        lines.append(f"  Total discrete configs: ~{self.size():,}")
        return "\n".join(lines)


# ===========================================================================
# 4.  ConfigValidator
# ===========================================================================

class ConfigValidator:
    """
    Validates VLLMFlags against hardware constraints and vLLM rules.

    Covers all 6 supported GPU types:
      NVIDIA: H100, H200, B300
      AMD:    MI300X, MI325X, MI350X

    Returns a list of human-readable violation strings (empty = valid).
    """

    def __init__(
        self,
        gpu_type: str = "H100",
        gpu_profiles_path: Path = GPU_PROFILES_YAML,
    ):
        self.gpu_type = gpu_type
        self.profile = _load_gpu_profile(gpu_type, gpu_profiles_path)
        self.is_amd = gpu_type in _AMD_GPU_TYPES
        self.is_nvidia = gpu_type in _NVIDIA_GPU_TYPES

    def validate(self, flags: VLLMFlags) -> List[str]:
        """Return a list of violation strings. Empty = valid."""
        violations: List[str] = []
        violations.extend(self._check_tensor_parallel(flags))
        violations.extend(self._check_memory(flags))
        violations.extend(self._check_batching(flags))
        violations.extend(self._check_quantization_compat(flags))
        violations.extend(self._check_fp8_support(flags))
        violations.extend(self._check_dtype_compat(flags))
        violations.extend(self._check_speculative_decoding(flags))
        violations.extend(self._check_amd_specific(flags))
        violations.extend(self._check_moe_flags(flags))
        violations.extend(self._check_block_size(flags))
        return violations

    def is_valid(self, flags: VLLMFlags) -> bool:
        return len(self.validate(flags)) == 0

    def _check_tensor_parallel(self, f: VLLMFlags) -> List[str]:
        violations = []
        max_tp = int(self.profile.get("max_tensor_parallel", 8))
        if f.tensor_parallel_size > max_tp:
            violations.append(
                f"tensor_parallel_size={f.tensor_parallel_size} exceeds "
                f"GPU max ({max_tp}) for {self.gpu_type}"
            )
        tp = f.tensor_parallel_size
        if tp > 0 and (tp & (tp - 1) != 0):
            violations.append(
                f"tensor_parallel_size={tp} is not a power of 2 (valid: 1, 2, 4, 8)"
            )
        if f.pipeline_parallel_size < 1:
            violations.append("pipeline_parallel_size must be >= 1")
        return violations

    def _check_memory(self, f: VLLMFlags) -> List[str]:
        violations = []
        if not (0.50 <= f.gpu_memory_utilization <= 0.99):
            violations.append(
                f"gpu_memory_utilization={f.gpu_memory_utilization} must be in [0.50, 0.99]"
            )
        return violations

    def _check_batching(self, f: VLLMFlags) -> List[str]:
        violations = []
        if f.max_num_batched_tokens < f.max_num_seqs:
            violations.append(
                f"max_num_batched_tokens ({f.max_num_batched_tokens}) must be "
                f">= max_num_seqs ({f.max_num_seqs})"
            )
        if f.max_model_len < 512:
            violations.append(f"max_model_len={f.max_model_len} is unusably small (min 512)")
        return violations

    def _check_quantization_compat(self, f: VLLMFlags) -> List[str]:
        violations = []
        if f.kv_cache_dtype == "fp8_e5m2" and f.quantization == "gptq":
            violations.append("kv_cache_dtype=fp8_e5m2 is incompatible with quantization=gptq")
        if f.enable_chunked_prefill and f.quantization == "squeezellm":
            violations.append(
                "enable_chunked_prefill=True is incompatible with squeezellm quantization"
            )
        if f.quantization == "bitsandbytes" and f.load_format not in ("auto", "bitsandbytes"):
            violations.append(
                "quantization=bitsandbytes requires load_format=auto or bitsandbytes"
            )
        return violations

    def _check_fp8_support(self, f: VLLMFlags) -> List[str]:
        violations = []
        fp8_dtypes = {"fp8", "fp8_e4m3", "fp8_e5m2"}
        supports_fp8 = bool(self.profile.get("fp8", True))
        if f.kv_cache_dtype in fp8_dtypes and not supports_fp8:
            violations.append(
                f"kv_cache_dtype={f.kv_cache_dtype} requires FP8 support "
                f"(not available on {self.gpu_type})"
            )
        if f.quantization == "fp8" and not supports_fp8:
            violations.append(f"quantization=fp8 not supported on {self.gpu_type}")
        return violations

    def _check_dtype_compat(self, f: VLLMFlags) -> List[str]:
        violations = []
        supports_bf16 = bool(self.profile.get("bf16", True))
        if f.dtype == "bfloat16" and not supports_bf16:
            violations.append(f"dtype=bfloat16 is not supported on {self.gpu_type}")
        return violations

    def _check_speculative_decoding(self, f: VLLMFlags) -> List[str]:
        violations = []
        if f.speculative_model and f.num_speculative_tokens is None:
            violations.append(
                "speculative_model is set but num_speculative_tokens is None. "
                "Set num_speculative_tokens (e.g. 3, 5, or 8)."
            )
        if f.num_speculative_tokens is not None and not f.speculative_model:
            violations.append(
                "num_speculative_tokens is set but speculative_model is None."
            )
        return violations

    def _check_amd_specific(self, f: VLLMFlags) -> List[str]:
        violations = []
        if not self.is_amd:
            return violations
        if f.distributed_executor_backend == "ray":
            violations.append(
                f"{self.gpu_type} (AMD ROCm) requires distributed_executor_backend=mp, "
                f"not ray. Ray is not supported on ROCm."
            )
        if f.attention_backend == "flash_attn":
            violations.append(
                f"{self.gpu_type}: use attention_backend=auto or aiter, "
                f"not flash_attn (CUDA-only)."
            )
        return violations

    def _check_moe_flags(self, f: VLLMFlags) -> List[str]:
        violations = []
        deepep_backends = {"deepep_high_throughput", "deepep_low_latency"}
        if f.all2all_backend in deepep_backends and not f.enable_expert_parallel:
            violations.append(
                f"all2all_backend={f.all2all_backend} requires enable_expert_parallel=True"
            )
        if f.enable_dbo and not f.enable_expert_parallel:
            violations.append("enable_dbo=True requires enable_expert_parallel=True")
        return violations

    def _check_block_size(self, f: VLLMFlags) -> List[str]:
        violations = []
        valid_block_sizes = {1, 8, 16, 32}
        if f.block_size not in valid_block_sizes:
            violations.append(
                f"block_size={f.block_size} is not valid. "
                f"Must be one of {sorted(valid_block_sizes)}."
            )
        return violations


# ===========================================================================
# 5.  Helpers
# ===========================================================================

def _load_gpu_profile(gpu_type: str, path: Path = GPU_PROFILES_YAML) -> dict:
    """Load a single GPU profile from gpu_profiles.yaml."""
    if not path.exists():
        log.warning(f"gpu_profiles.yaml not found at {path}, using empty profile")
        return {}
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    profiles = raw.get("gpu_profiles", {})
    if gpu_type not in profiles:
        log.warning(f"GPU type '{gpu_type}' not in gpu_profiles.yaml — using defaults")
        return {}
    return profiles[gpu_type]


def flags_from_dict(d: dict) -> VLLMFlags:
    """
    Reconstruct a VLLMFlags from a plain dict (e.g. loaded from CSV/JSON).
    Unknown keys are silently ignored; missing keys fall back to defaults.
    """
    flags = VLLMFlags()
    for k, v in d.items():
        if hasattr(flags, k):
            setattr(flags, k, v)
    if not flags.run_id:
        flags.run_id = flags.fingerprint()
    return flags
