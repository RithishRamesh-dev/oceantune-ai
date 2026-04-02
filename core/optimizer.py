"""
core/optimizer.py
-----------------
Optimisation engine for OceanTune AI.

Implements four search strategies that share a common interface:
  - evolutionary  primary strategy; population-based with elitism + tournament
  - grid          systematic neighbour exploration from a seed config
  - random        uniform random sampling (baseline / ablation)
  - bayesian      placeholder with surrogate-model hook (Stage 3+)

All strategies produce a ranked list of VLLMFlags for the controller
agent (Step 7) to benchmark. After each benchmark result comes back,
the optimiser updates its state and produces the next batch of configs
to try.

Key design decisions
--------------------
- OptimiserState is fully serialisable (to_dict / from_dict) so a
  session can be resumed after a process restart without re-running
  already-tested configs.
- PopulationManager deduplicates across the current session AND across
  previous sessions (loaded from storage/results/runs.csv via ResultLoader).
- ConfigValidator is always called before a config is queued for
  benchmarking — no GPU time is wasted on hardware-impossible configs.
- FitnessEvaluator uses a primary + secondary metric pair so that
  configs with the same throughput are sorted by latency.
- The evolutionary strategy uses tournament selection (k=3) rather than
  pure rank-based selection to maintain diversity without sacrificing
  convergence speed.

Usage (called by ControllerAgent in Step 7)
--------------------------------------------
    from core.optimizer import create_optimiser
    opt = create_optimiser(cfg, search_space, validator, csv_path)
    # Get first batch to benchmark
    batch = opt.next_batch(batch_size=cfg.benchmark.population_size)
    # ... run experiments ...
    for flags, enriched in results:
        opt.record_result(flags, enriched)
    # Get next generation
    next_batch = opt.next_batch(batch_size=...)
"""

from __future__ import annotations

import json
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from core.config import OceanTuneConfig
from core.logger import get_logger, log_dict
from core.metrics_collector import EnrichedMetrics
from core.search_space import (
    ConfigValidator,
    SearchSpace,
    VLLMFlags,
    flags_from_dict,
)
from core.storage import ResultLoader

log = get_logger("core.optimizer")


# ===========================================================================
# 1.  FitnessEvaluator
# ===========================================================================

@dataclass
class ScoredConfig:
    """A VLLMFlags paired with its fitness metrics, sortable by fitness."""
    flags: VLLMFlags
    fitness_score: float
    throughput: float = 0.0
    p95_latency_ms: float = 0.0
    ttft_ms: float = 0.0
    generation: int = 0

    def __lt__(self, other: "ScoredConfig") -> bool:
        # Higher fitness = better; for equal fitness use lower latency
        if abs(self.fitness_score - other.fitness_score) > 1e-9:
            return self.fitness_score > other.fitness_score
        return self.p95_latency_ms < other.p95_latency_ms

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScoredConfig):
            return False
        return self.flags.fingerprint() == other.flags.fingerprint()


class FitnessEvaluator:
    """
    Converts EnrichedMetrics into a ScoredConfig.

    Applies a small warning penalty (5%) so configs that produced
    warnings but no hard errors are ranked slightly below clean configs
    with the same throughput.
    """

    WARNING_DISCOUNT = 0.05   # 5% penalty per warning class

    @classmethod
    def score(
        cls,
        flags: VLLMFlags,
        enriched: EnrichedMetrics,
        generation: int = 0,
    ) -> ScoredConfig:
        """Build a ScoredConfig from flags and enriched metrics."""
        fitness = enriched.fitness_score

        return ScoredConfig(
            flags=flags,
            fitness_score=fitness,
            throughput=enriched.peak_throughput_tokens_per_sec,
            p95_latency_ms=enriched.p95_latency_at_peak_ms,
            ttft_ms=enriched.mean_ttft_ms,
            generation=generation,
        )

    @classmethod
    def rank(cls, scored: List[ScoredConfig]) -> List[ScoredConfig]:
        """Return configs sorted best-first."""
        return sorted(scored)


# ===========================================================================
# 2.  OptimiserState
# ===========================================================================

@dataclass
class OptimiserState:
    """
    Full serialisable state of one optimiser session.

    Persisted to disk after each generation so a crash or restart
    can resume without losing progress.
    """
    strategy: str = "evolutionary"
    generation: int = 0
    session_id: str = ""
    started_at: float = field(default_factory=time.time)

    # Current population (configs queued or being benchmarked)
    population: List[VLLMFlags] = field(default_factory=list)

    # All scored results this session, best-first
    scored_history: List[ScoredConfig] = field(default_factory=list)

    # Fingerprints of configs we never want to retry
    failed_fingerprints: Set[str] = field(default_factory=set)

    # Fingerprints of configs already benchmarked this session
    seen_fingerprints: Set[str] = field(default_factory=set)

    @property
    def all_time_best(self) -> Optional[ScoredConfig]:
        """Best ScoredConfig seen so far, or None if no results yet."""
        if not self.scored_history:
            return None
        return self.scored_history[0]

    def record_result(
        self,
        flags: VLLMFlags,
        enriched: EnrichedMetrics,
    ) -> ScoredConfig:
        """
        Record a benchmark result and update the sorted history.

        If the run failed (all_levels_failed), the fingerprint is added
        to failed_fingerprints so it is never retried.
        """
        scored = FitnessEvaluator.score(flags, enriched, self.generation)
        self.scored_history = FitnessEvaluator.rank(
            self.scored_history + [scored]
        )
        self.seen_fingerprints.add(flags.fingerprint())

        if enriched.all_levels_failed or enriched.fitness_score == 0.0:
            self.failed_fingerprints.add(flags.fingerprint())
            log_dict(log, "info", "Config marked as failed",
                     run_id=flags.run_id, generation=self.generation)

        return scored

    def is_known_bad(self, flags: VLLMFlags) -> bool:
        return flags.fingerprint() in self.failed_fingerprints

    def is_already_seen(self, flags: VLLMFlags) -> bool:
        return flags.fingerprint() in self.seen_fingerprints

    def top_k(self, k: int) -> List[ScoredConfig]:
        """Return the k best configs scored so far."""
        return self.scored_history[:k]

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "generation": self.generation,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "population": [f.to_dict() for f in self.population],
            "scored_history": [
                {
                    "flags": s.flags.to_dict(),
                    "fitness_score": s.fitness_score,
                    "throughput": s.throughput,
                    "p95_latency_ms": s.p95_latency_ms,
                    "ttft_ms": s.ttft_ms,
                    "generation": s.generation,
                }
                for s in self.scored_history
            ],
            "failed_fingerprints": list(self.failed_fingerprints),
            "seen_fingerprints": list(self.seen_fingerprints),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OptimiserState":
        state = cls(
            strategy=d.get("strategy", "evolutionary"),
            generation=d.get("generation", 0),
            session_id=d.get("session_id", ""),
            started_at=d.get("started_at", time.time()),
        )
        state.population = [
            flags_from_dict(f) for f in d.get("population", [])
        ]
        state.failed_fingerprints = set(d.get("failed_fingerprints", []))
        state.seen_fingerprints = set(d.get("seen_fingerprints", []))
        state.scored_history = [
            ScoredConfig(
                flags=flags_from_dict(s["flags"]),
                fitness_score=s["fitness_score"],
                throughput=s.get("throughput", 0.0),
                p95_latency_ms=s.get("p95_latency_ms", 0.0),
                ttft_ms=s.get("ttft_ms", 0.0),
                generation=s.get("generation", 0),
            )
            for s in d.get("scored_history", [])
        ]
        return state

    def save(self, path: Path) -> None:
        """Serialise state to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        log.debug(f"OptimiserState saved: {path}")

    @classmethod
    def load(cls, path: Path) -> "OptimiserState":
        """Load state from a JSON file. Returns fresh state if file missing."""
        if not path.exists():
            return cls()
        try:
            with open(path, encoding="utf-8") as f:
                return cls.from_dict(json.load(f))
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            log.warning(f"Could not load OptimiserState from {path}: {exc}")
            return cls()


# ===========================================================================
# 3.  PopulationManager
# ===========================================================================

class PopulationManager:
    """
    Generates, deduplicates, and validates candidate VLLMFlags configs.

    Integrates with:
    - SearchSpace    (sampling, mutation, crossover)
    - ConfigValidator (reject hardware-impossible configs before GPU use)
    - ResultLoader    (load failed fingerprints from previous sessions)
    """

    def __init__(
        self,
        search_space: SearchSpace,
        validator: ConfigValidator,
        state: OptimiserState,
    ):
        self.ss = search_space
        self.validator = validator
        self.state = state

    def load_cross_session_failures(self, csv_path: Path) -> None:
        """
        Load failed fingerprints from previous sessions' CSV.

        Call once at session start so we never re-run configs that
        crashed in a previous session.
        """
        failed = ResultLoader.get_failed_fingerprints(csv_path)
        added = len(failed - self.state.failed_fingerprints)
        self.state.failed_fingerprints.update(failed)
        if added > 0:
            log_dict(log, "info", "Loaded cross-session failures",
                     count=len(failed), new=added)

    def seed_population(
        self,
        gpu_type: str,
        size: int,
    ) -> List[VLLMFlags]:
        """
        Generate the initial population for generation 0.

        Uses GPU-profile-seeded sampling so the population starts in a
        hardware-sane region of the search space.
        """
        candidates = self.ss.sample_seeded(gpu_type, size=size * 3)
        return self._filter_and_cap(candidates, size)

    def generate_next_population(
        self,
        parents: List[ScoredConfig],
        size: int,
        mutation_rate: float,
        elite_fraction: float,
    ) -> List[VLLMFlags]:
        """
        Produce the next generation from scored parents.

        Algorithm:
        1. Elite carry-forward (top elite_fraction configs unchanged)
        2. Tournament selection → crossover → mutation for the rest
        3. Random injection if diversity is low
        4. Deduplication + validation pass
        """
        if not parents:
            return self.seed_population("H100", size)

        n_elite = max(1, int(size * elite_fraction))
        n_breed = size - n_elite

        # ── Elites (carry forward unchanged) ─────────────────────────────
        elites = [p.flags.copy() for p in parents[:n_elite]]

        # ── Breed new candidates ──────────────────────────────────────────
        children: List[VLLMFlags] = []
        parent_flags = [p.flags for p in parents]

        for _ in range(n_breed * 4):   # generate surplus, then filter
            if len(children) >= n_breed:
                break
            a = self._tournament_select(parents, k=3)
            b = self._tournament_select(parents, k=3)
            if a.flags.fingerprint() != b.flags.fingerprint():
                child = self.ss.crossover(a.flags, b.flags)
            else:
                child = a.flags.copy()
            child = self.ss.mutate(child, mutation_rate=mutation_rate)
            children.append(child)

        candidates = elites + children

        # ── Diversity injection ───────────────────────────────────────────
        # If top-5 are identical, inject random configs
        top5_fps = {p.flags.fingerprint() for p in parents[:5]}
        if len(top5_fps) <= 2:
            n_inject = max(2, size // 4)
            log.debug(f"Low diversity — injecting {n_inject} random configs")
            candidates += [self.ss.sample_random() for _ in range(n_inject)]

        return self._filter_and_cap(candidates, size)

    def _filter_and_cap(
        self,
        candidates: List[VLLMFlags],
        size: int,
    ) -> List[VLLMFlags]:
        """
        Remove duplicates, known-bad, already-seen, and invalid configs.
        Returns at most `size` configs.
        """
        seen_in_batch: Set[str] = set()
        result: List[VLLMFlags] = []

        for flags in candidates:
            if len(result) >= size:
                break
            fp = flags.fingerprint()

            # Deduplicate within this batch
            if fp in seen_in_batch:
                continue
            seen_in_batch.add(fp)

            # Skip known-bad configs
            if self.state.is_known_bad(flags):
                continue

            # Skip already-benchmarked in this session
            if self.state.is_already_seen(flags):
                continue

            # Reject hardware-impossible configs
            violations = self.validator.validate(flags)
            if violations:
                log.debug(f"Config rejected: {violations[0]}")
                continue

            flags.run_id = fp
            result.append(flags)

        # Fallback: fill shortfall with random valid configs
        if len(result) < size:
            for _ in range((size - len(result)) * 30):
                if len(result) >= size:
                    break
                candidate = self.ss.sample_random()
                fp = candidate.fingerprint()
                if fp in seen_in_batch:
                    continue
                if self.state.is_known_bad(candidate):
                    continue
                if self.state.is_already_seen(candidate):
                    continue
                violations = self.validator.validate(candidate)
                if violations:
                    continue
                seen_in_batch.add(fp)
                candidate.run_id = fp
                result.append(candidate)

        if len(result) < size:
            log.debug(
                f"Could only produce {len(result)}/{size} valid candidates "
                f"— search space may be exhausted"
            )

        return result

    @staticmethod
    def _tournament_select(
        population: List[ScoredConfig],
        k: int = 3,
    ) -> ScoredConfig:
        """
        Tournament selection: pick k random candidates, return the best.

        k=3 balances selection pressure and diversity — less aggressive
        than pure rank selection but faster converging than k=2.
        """
        if len(population) <= k:
            return population[0]
        contestants = random.sample(population, k)
        return min(contestants)   # ScoredConfig.__lt__ sorts best-first


# ===========================================================================
# 4.  Base Optimiser
# ===========================================================================

class BaseOptimiser(ABC):
    """
    Abstract base class for all search strategies.

    Subclasses implement next_batch() which produces the next set of
    VLLMFlags to benchmark. After benchmarking, the controller calls
    record_result() for each result.
    """

    def __init__(
        self,
        cfg: OceanTuneConfig,
        search_space: SearchSpace,
        validator: ConfigValidator,
        state: Optional[OptimiserState] = None,
        csv_path: Optional[Path] = None,
    ):
        self.cfg = cfg
        self.ss = search_space
        self.validator = validator
        self.state = state or OptimiserState(strategy=cfg.optimiser.strategy)
        self.pop_manager = PopulationManager(search_space, validator, self.state)

        # Load cross-session failures if CSV exists
        if csv_path and csv_path.exists():
            self.pop_manager.load_cross_session_failures(csv_path)

    @abstractmethod
    def next_batch(self, batch_size: int) -> List[VLLMFlags]:
        """Return the next batch of configs to benchmark."""
        ...

    def record_result(
        self,
        flags: VLLMFlags,
        enriched: EnrichedMetrics,
    ) -> ScoredConfig:
        """
        Record a benchmark result. Called by the controller for each run.
        Returns the ScoredConfig so the controller can log it.
        """
        scored = self.state.record_result(flags, enriched)

        best = self.state.all_time_best
        log_dict(
            log, "info", "Result recorded",
            run_id=flags.run_id,
            fitness=round(scored.fitness_score, 4),
            throughput=round(scored.throughput, 1),
            generation=self.state.generation,
            best_fitness=round(best.fitness_score, 4) if best else 0.0,
        )
        return scored

    def advance_generation(self) -> None:
        """Increment the generation counter."""
        self.state.generation += 1
        log_dict(log, "info", "Generation advanced",
                 generation=self.state.generation)

    @property
    def best(self) -> Optional[ScoredConfig]:
        return self.state.all_time_best

    @property
    def generation(self) -> int:
        return self.state.generation

    def summary(self) -> dict:
        """Return a summary dict for logging and reporting."""
        best = self.state.all_time_best
        return {
            "strategy": self.cfg.optimiser.strategy,
            "generation": self.state.generation,
            "total_runs": len(self.state.scored_history),
            "failed_configs": len(self.state.failed_fingerprints),
            "best_fitness": round(best.fitness_score, 4) if best else 0.0,
            "best_throughput": round(best.throughput, 1) if best else 0.0,
            "best_p95_ms": round(best.p95_latency_ms, 1) if best else 0.0,
        }


# ===========================================================================
# 5.  EvolutionaryOptimiser
# ===========================================================================

class EvolutionaryOptimiser(BaseOptimiser):
    """
    Evolutionary search with tournament selection, crossover, and elitism.

    Generation 0: seeded population from GPU profile defaults.
    Generations 1+: breed from previous generation's scored results.

    Parameters (from OceanTuneConfig.optimiser):
      population_size   — number of configs per generation
      mutation_rate     — probability of mutating each parameter
      elite_fraction    — fraction of population kept unchanged
    """

    def next_batch(self, batch_size: int) -> List[VLLMFlags]:
        """
        Produce the next generation of configs to benchmark.

        Generation 0: GPU-seeded random population.
        Generation 1+: breed from scored_history.
        """
        cfg_opt = self.cfg.optimiser

        if self.state.generation == 0:
            log_dict(log, "info", "Seeding initial population",
                     gpu=self.cfg.gpu_type, size=batch_size)
            batch = self.pop_manager.seed_population(
                self.cfg.gpu_type, batch_size
            )
        else:
            log_dict(log, "info", "Breeding next generation",
                     generation=self.state.generation,
                     parents=len(self.state.scored_history))
            batch = self.pop_manager.generate_next_population(
                parents=self.state.top_k(batch_size * 2),
                size=batch_size,
                mutation_rate=cfg_opt.mutation_rate,
                elite_fraction=cfg_opt.elite_fraction,
            )

        log_dict(log, "info", "Next batch ready",
                 generation=self.state.generation, size=len(batch))
        return batch


# ===========================================================================
# 6.  GridOptimiser
# ===========================================================================

class GridOptimiser(BaseOptimiser):
    """
    Systematic grid search: explore all single-parameter neighbours of
    the current best config, then move to the best neighbour found.

    This is a hill-climbing strategy that works well when the fitness
    landscape is smooth and unimodal. Use evolutionary for rugged spaces.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Start from GPU-profile default flags
        self._current_seed: VLLMFlags = self.ss.default_flags()

    def next_batch(self, batch_size: int) -> List[VLLMFlags]:
        """
        Return all single-step neighbours of the current best config.

        On generation 0, starts from the GPU-profile seed.
        On subsequent generations, starts from the all-time best config.
        """
        if self.state.generation > 0 and self.state.all_time_best:
            self._current_seed = self.state.all_time_best.flags

        neighbours = self.ss.grid_neighbours(self._current_seed)
        batch = self.pop_manager._filter_and_cap(neighbours, batch_size)

        log_dict(log, "info", "Grid neighbours generated",
                 generation=self.state.generation,
                 seed_fp=self._current_seed.fingerprint()[:8],
                 neighbours=len(neighbours),
                 valid=len(batch))
        return batch


# ===========================================================================
# 7.  RandomOptimiser
# ===========================================================================

class RandomOptimiser(BaseOptimiser):
    """
    Pure random search — uniform sampling from the full search space.

    Useful as a baseline to compare against evolutionary / grid.
    No memory of past results influences the next batch.
    """

    def next_batch(self, batch_size: int) -> List[VLLMFlags]:
        """Sample batch_size random configs (excluding known-bad)."""
        candidates = [self.ss.sample_random() for _ in range(batch_size * 5)]
        batch = self.pop_manager._filter_and_cap(candidates, batch_size)
        log_dict(log, "info", "Random batch sampled",
                 generation=self.state.generation, size=len(batch))
        return batch


# ===========================================================================
# 8.  BayesianOptimiser (Stage 3 placeholder)
# ===========================================================================

class BayesianOptimiser(BaseOptimiser):
    """
    Bayesian optimisation with a Gaussian Process surrogate model.

    PLACEHOLDER — the surrogate model training is implemented in Step 12
    (Stage-3 kernel optimisation). For now this falls back to evolutionary.

    When implemented, the surrogate will:
    1. Fit a GP on (config_vector, fitness_score) pairs from history
    2. Use Expected Improvement acquisition to select next configs
    3. Fall back to random sampling when history < min_history_size
    """

    MIN_HISTORY_FOR_SURROGATE = 20

    def next_batch(self, batch_size: int) -> List[VLLMFlags]:
        """
        Use surrogate model if enough history, else fall back to evolutionary.
        """
        if len(self.state.scored_history) < self.MIN_HISTORY_FOR_SURROGATE:
            log.debug(
                f"Bayesian: insufficient history "
                f"({len(self.state.scored_history)} < "
                f"{self.MIN_HISTORY_FOR_SURROGATE}), using evolutionary"
            )
            # Delegate to evolutionary logic
            evo = EvolutionaryOptimiser(
                cfg=self.cfg,
                search_space=self.ss,
                validator=self.validator,
                state=self.state,
            )
            return evo.next_batch(batch_size)

        # Stage 3: surrogate model acquisition (implemented in Step 12)
        log.warning(
            "Bayesian surrogate model not yet implemented — "
            "falling back to evolutionary. Will be completed in Step 12."
        )
        evo = EvolutionaryOptimiser(
            cfg=self.cfg,
            search_space=self.ss,
            validator=self.validator,
            state=self.state,
        )
        return evo.next_batch(batch_size)


# ===========================================================================
# 9.  Factory
# ===========================================================================

_STRATEGY_MAP = {
    "evolutionary": EvolutionaryOptimiser,
    "grid":         GridOptimiser,
    "random":       RandomOptimiser,
    "bayesian":     BayesianOptimiser,
}


def create_optimiser(
    cfg: OceanTuneConfig,
    search_space: SearchSpace,
    validator: ConfigValidator,
    state: Optional[OptimiserState] = None,
    csv_path: Optional[Path] = None,
) -> BaseOptimiser:
    """
    Factory: instantiate the correct optimiser for cfg.optimiser.strategy.

    Parameters
    ----------
    cfg : OceanTuneConfig
        Full config. cfg.optimiser.strategy selects the algorithm.
    search_space : SearchSpace
        Loaded from configs/search_space.yaml.
    validator : ConfigValidator
        Instantiated with the target GPU type.
    state : OptimiserState, optional
        Existing state to resume from. If None, starts fresh.
    csv_path : Path, optional
        Path to runs.csv for cross-session failure deduplication.

    Returns
    -------
    BaseOptimiser subclass instance.

    Raises
    ------
    ValueError if strategy is unknown (caught earlier by config validator).
    """
    strategy = cfg.optimiser.strategy
    cls = _STRATEGY_MAP.get(strategy)
    if cls is None:
        raise ValueError(
            f"Unknown optimiser strategy '{strategy}'. "
            f"Valid choices: {list(_STRATEGY_MAP)}"
        )

    optimiser = cls(
        cfg=cfg,
        search_space=search_space,
        validator=validator,
        state=state,
        csv_path=csv_path,
    )

    log_dict(
        log, "info", "Optimiser created",
        strategy=strategy,
        gpu=cfg.gpu_type,
        model=cfg.model_id,
        population_size=cfg.optimiser.population_size,
        generations=cfg.optimiser.generations,
        mutation_rate=cfg.optimiser.mutation_rate,
        elite_fraction=cfg.optimiser.elite_fraction,
        primary_metric=cfg.optimiser.primary_metric,
    )

    return optimiser