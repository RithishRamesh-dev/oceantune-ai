"""
core/search_space.py
--------------------
Typed search-space system for OceanTune AI.

This module is the backbone of the optimisation engine.  It handles:

  1. Loading search_space.yaml into strongly-typed Parameter objects.
  2. Representing a single vLLM configuration as a VLLMFlags dataclass.
  3. Sampling initial populations (random, grid, or GPU-profile-seeded).
  4. Mutating configs for evolutionary search.
  5. Validating configs to reject hardware-impossible combinations
     before we waste a GPU spin-up on them.

Key design decisions
--------------------
- Every parameter is one of four types (choice, range_int, range_float,
  bool_flag).  All sampling / mutation logic is type-dispatched, so adding
  a new parameter to search_space.yaml requires zero code changes here.
- VLLMFlags is a plain dataclass (not Pydantic) so it can be copied,
  mutated, hashed, and serialised cheaply millions of times.
- ConfigValidator returns a list of violations rather than raising, so the
  optimiser can log failures without try/except overhead.
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
        """Return all values except the current one (for grid search)."""
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


# Union type for convenience
AnyParam = ChoiceParam | RangeIntParam | RangeFloatParam | BoolFlagParam


# ===========================================================================
# 2.  VLLMFlags — a single configuration point in the search space
# ===========================================================================

@dataclass
class VLLMFlags:
    """
    Represents one set of vLLM server flags.

    All fields mirror the keys in search_space.yaml.
    Defaults match the YAML defaults so the object is always valid
    even if partially constructed.
    """
    # Parallelism
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    # Memory
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192

    # KV cache
    kv_cache_dtype: str = "auto"
    block_size: int = 16

    # Quantisation
    quantization: Optional[str] = None

    # Scheduler
    scheduler_delay_factor: float = 0.0
    enable_chunked_prefill: bool = False

    # Stage-2 flags (populated later)
    use_v2_block_manager: bool = False
    speculative_draft_model: Optional[str] = None

    # Dtype
    dtype: str = "float16"

    # Metadata (not part of vLLM command line)
    run_id: str = field(default="", repr=False)

    def to_dict(self) -> dict:
        """Return all fields as a plain dict (for CSV / JSON serialisation)."""
        return asdict(self)

    def to_vllm_args(self, model_id: str, gpu_type: str = "A100") -> List[str]:
        """
        Convert this config into a list of vLLM CLI arguments.

        Parameters
        ----------
        model_id : str
            Hugging Face model ID.
        gpu_type : str
            Used to select AMD-specific flags.
        """
        args = [
            "--model", model_id,
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--pipeline-parallel-size", str(self.pipeline_parallel_size),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-num-seqs", str(self.max_num_seqs),
            "--max-num-batched-tokens", str(self.max_num_batched_tokens),
            "--kv-cache-dtype", self.kv_cache_dtype,
            "--block-size", str(self.block_size),
            "--dtype", self.dtype,
            "--scheduler-delay-factor", str(self.scheduler_delay_factor),
        ]

        if self.quantization:
            args += ["--quantization", self.quantization]

        if self.enable_chunked_prefill:
            args.append("--enable-chunked-prefill")

        if self.use_v2_block_manager:
            args.append("--use-v2-block-manager")

        if self.speculative_draft_model:
            args += ["--speculative-model", self.speculative_draft_model]

        if gpu_type == "MI300X":
            args += ["--device", "rocm"]

        return args

    def fingerprint(self) -> str:
        """
        Return a short, stable hash of this config's flags.

        Used as a unique run identifier and to detect duplicate configs
        before spending GPU time on them.
        """
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
# 3.  SearchSpace — loads YAML, owns all parameter objects
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

    # ── Loader ────────────────────────────────────────────────────────────

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
                params[name] = ChoiceParam(
                    name=name,
                    values=spec["values"],
                    default=default,
                )
            elif ptype == "range_int":
                params[name] = RangeIntParam(
                    name=name,
                    min=int(spec["min"]),
                    max=int(spec["max"]),
                    step=int(spec.get("step", 1)),
                    default=int(default),
                )
            elif ptype == "range_float":
                params[name] = RangeFloatParam(
                    name=name,
                    min=float(spec["min"]),
                    max=float(spec["max"]),
                    step=float(spec.get("step", 0.05)),
                    default=float(default),
                )
            elif ptype == "bool_flag":
                params[name] = BoolFlagParam(
                    name=name,
                    default=bool(default),
                )
            else:
                log.warning(f"Unknown parameter type '{ptype}' for '{name}' — skipping")

        log_dict(log, "info", "Loaded search space", path=str(path), params=list(params))
        return cls(params)

    # ── Default config ────────────────────────────────────────────────────

    def default_flags(self) -> VLLMFlags:
        """Return a VLLMFlags with every parameter set to its YAML default."""
        flags = VLLMFlags()
        for name, param in self._params.items():
            if hasattr(flags, name):
                setattr(flags, name, param.default)
        flags.run_id = flags.fingerprint()
        return flags

    # ── Random sampling ───────────────────────────────────────────────────

    def sample_random(self) -> VLLMFlags:
        """Return a uniformly random VLLMFlags from the full search space."""
        flags = VLLMFlags()
        for name, param in self._params.items():
            if hasattr(flags, name):
                setattr(flags, name, param.sample())
        flags.run_id = flags.fingerprint()
        return flags

    def sample_population(self, size: int) -> List[VLLMFlags]:
        """
        Return *size* random configs, guaranteed to be unique.

        If the search space is smaller than *size*, returns as many
        unique configs as possible (with a warning).
        """
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

    # ── GPU-profile-seeded sampling ───────────────────────────────────────

    def sample_seeded(
        self,
        gpu_type: str,
        size: int,
        gpu_profiles_path: Path = GPU_PROFILES_YAML,
    ) -> List[VLLMFlags]:
        """
        Return a population seeded with sensible GPU-profile defaults,
        then lightly perturbed so we explore near the hint.
        """
        profile = _load_gpu_profile(gpu_type, gpu_profiles_path)
        seed = self.default_flags()

        if "default_gpu_memory_utilization" in profile:
            seed.gpu_memory_utilization = float(
                profile["default_gpu_memory_utilization"]
            )
        if "max_tensor_parallel" in profile:
            max_tp = int(profile["max_tensor_parallel"])
            seed.tensor_parallel_size = 1
            tp_param = self._params.get("tensor_parallel_size")
            if isinstance(tp_param, ChoiceParam):
                tp_param.values = [v for v in tp_param.values if v <= max_tp]

        seed.run_id = seed.fingerprint()
        population: List[VLLMFlags] = [seed]

        seen = {seed.fingerprint()}
        max_attempts = size * 30
        for _ in range(max_attempts):
            if len(population) >= size:
                break
            candidate = self.mutate(seed, mutation_rate=0.3)
            fp = candidate.fingerprint()
            if fp not in seen:
                seen.add(fp)
                population.append(candidate)

        log_dict(
            log, "info", "Seeded population ready",
            gpu=gpu_type, size=len(population),
        )
        return population

    # ── Mutation ──────────────────────────────────────────────────────────

    def mutate(self, flags: VLLMFlags, mutation_rate: float = 0.2) -> VLLMFlags:
        """
        Return a mutated copy of *flags*.

        Each parameter is independently mutated with probability
        *mutation_rate*.  The original object is never modified.
        """
        mutated = flags.copy()
        for name, param in self._params.items():
            if hasattr(mutated, name):
                current = getattr(mutated, name)
                new_val = param.mutate(current, mutation_rate)
                setattr(mutated, name, new_val)
        mutated.run_id = mutated.fingerprint()
        return mutated

    # ── Crossover ─────────────────────────────────────────────────────────

    def crossover(self, parent_a: VLLMFlags, parent_b: VLLMFlags) -> VLLMFlags:
        """
        Single-point crossover: for each parameter, randomly pick from
        parent A or parent B.  Used by the evolutionary optimiser.
        """
        child = VLLMFlags()
        for name in self._params:
            if hasattr(child, name):
                if random.random() < 0.5:
                    setattr(child, name, getattr(parent_a, name))
                else:
                    setattr(child, name, getattr(parent_b, name))
        child.run_id = child.fingerprint()
        return child

    # ── Grid expansion ────────────────────────────────────────────────────

    def grid_neighbours(self, flags: VLLMFlags) -> List[VLLMFlags]:
        """
        Return all single-step neighbours for grid search.

        For each parameter, generate configs where only that parameter
        changes by one step in either direction.
        """
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

    # ── Introspection ─────────────────────────────────────────────────────

    def size(self) -> int:
        """Return the total number of configs in the discrete search space."""
        total = 1
        for param in self._params.values():
            if isinstance(param, ChoiceParam):
                total *= len(param.values)
            elif isinstance(param, RangeIntParam):
                steps = (param.max - param.min) // param.step + 1
                total *= steps
            elif isinstance(param, RangeFloatParam):
                steps = round((param.max - param.min) / param.step) + 1
                total *= steps
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

    Returns a list of human-readable violation strings.
    An empty list means the config is valid.

    Usage
    -----
        validator = ConfigValidator(gpu_type="A100", gpu_profiles_path=...)
        violations = validator.validate(flags)
        if violations:
            log.warning("Invalid config", violations=violations)
    """

    def __init__(
        self,
        gpu_type: str = "A100",
        gpu_profiles_path: Path = GPU_PROFILES_YAML,
    ):
        self.gpu_type = gpu_type
        self.profile = _load_gpu_profile(gpu_type, gpu_profiles_path)

    def validate(self, flags: VLLMFlags) -> List[str]:
        """Return a list of violation strings.  Empty = valid."""
        violations: List[str] = []
        violations.extend(self._check_tensor_parallel(flags))
        violations.extend(self._check_memory(flags))
        violations.extend(self._check_batching(flags))
        violations.extend(self._check_quantization_compat(flags))
        violations.extend(self._check_speculative_decoding(flags))
        violations.extend(self._check_dtype_compat(flags))
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
        if f.tensor_parallel_size > 0 and (
            f.tensor_parallel_size & (f.tensor_parallel_size - 1) != 0
        ):
            violations.append(
                f"tensor_parallel_size={f.tensor_parallel_size} is not a power of 2"
            )
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
        return violations

    def _check_quantization_compat(self, f: VLLMFlags) -> List[str]:
        violations = []
        if f.kv_cache_dtype == "fp8_e5m2" and f.quantization in ("gptq",):
            violations.append(
                f"kv_cache_dtype=fp8_e5m2 is incompatible with quantization={f.quantization}"
            )
        if f.enable_chunked_prefill and f.quantization in ("squeezellm",):
            violations.append(
                "enable_chunked_prefill=True is incompatible with squeezellm quantization"
            )
        return violations

    def _check_speculative_decoding(self, f: VLLMFlags) -> List[str]:
        violations = []
        if f.speculative_draft_model and not f.use_v2_block_manager:
            violations.append(
                "Speculative decoding requires use_v2_block_manager=True"
            )
        return violations

    def _check_dtype_compat(self, f: VLLMFlags) -> List[str]:
        violations = []
        supports_bf16 = bool(self.profile.get("bf16", True))
        if f.dtype == "bfloat16" and not supports_bf16:
            violations.append(
                f"dtype=bfloat16 is not supported on {self.gpu_type}"
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

    Unknown keys are silently ignored, missing keys fall back to defaults.
    """
    flags = VLLMFlags()
    for k, v in d.items():
        if hasattr(flags, k):
            setattr(flags, k, v)
    if not flags.run_id:
        flags.run_id = flags.fingerprint()
    return flags
