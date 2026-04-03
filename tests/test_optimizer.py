"""
tests/test_optimizer.py
-----------------------
Unit tests for core/optimizer.py.

No real GPU or vLLM required — everything is mocked or uses
the real search_space.yaml / gpu_profiles.yaml files.

Run with:
    pytest tests/test_optimizer.py -v
"""

from __future__ import annotations

import sys
import json
import random
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.optimizer import (
    BaseOptimiser,
    BayesianOptimiser,
    EvolutionaryOptimiser,
    FitnessEvaluator,
    GridOptimiser,
    OptimiserState,
    PopulationManager,
    RandomOptimiser,
    ScoredConfig,
    create_optimiser,
)
from core.config import OceanTuneConfig, OptimiserConfig
from core.metrics_collector import EnrichedMetrics
from core.search_space import ConfigValidator, SearchSpace, VLLMFlags


# ===========================================================================
# Fixtures
# ===========================================================================

def make_cfg(
    strategy="evolutionary",
    population_size=6,
    generations=3,
    mutation_rate=0.3,
    elite_fraction=0.2,
    gpu_type="H100",
) -> OceanTuneConfig:
    cfg = OceanTuneConfig()
    cfg.model_id = "test/model"
    cfg.gpu_type = gpu_type
    cfg.optimiser.strategy = strategy
    cfg.optimiser.population_size = population_size
    cfg.optimiser.generations = generations
    cfg.optimiser.mutation_rate = mutation_rate
    cfg.optimiser.elite_fraction = elite_fraction
    cfg.optimiser.primary_metric = "throughput"
    return cfg


def make_flags(tp=1, gpu_mem=0.90, run_id=None) -> VLLMFlags:
    f = VLLMFlags(tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem)
    f.run_id = run_id or f.fingerprint()
    return f


def make_enriched(throughput=2000.0, fitness=0.75, failed=False) -> EnrichedMetrics:
    return EnrichedMetrics(
        peak_throughput_tokens_per_sec=throughput,
        p95_latency_at_peak_ms=500.0,
        mean_ttft_ms=80.0,
        mean_tpot_ms=5.0,
        valid_levels=4 if not failed else 0,
        fitness_score=fitness if not failed else 0.0,
        all_levels_failed=failed,
        has_errors=failed,
    )


def make_scored(throughput=2000.0, fitness=0.75, flags=None) -> ScoredConfig:
    f = flags or make_flags(tp=1)
    return ScoredConfig(
        flags=f,
        fitness_score=fitness,
        throughput=throughput,
        p95_latency_ms=500.0,
        ttft_ms=80.0,
        generation=0,
    )


@pytest.fixture(scope="module")
def ss() -> SearchSpace:
    return SearchSpace.load()


@pytest.fixture(scope="module")
def validator() -> ConfigValidator:
    return ConfigValidator(gpu_type="H100")


@pytest.fixture
def fresh_state() -> OptimiserState:
    return OptimiserState(strategy="evolutionary", session_id="test-session")


@pytest.fixture
def cfg() -> OceanTuneConfig:
    return make_cfg()


# ===========================================================================
# 1. ScoredConfig ordering
# ===========================================================================

class TestScoredConfig:
    def test_higher_fitness_sorts_first(self):
        low = make_scored(fitness=0.4)
        high = make_scored(fitness=0.8)
        assert sorted([low, high])[0] == high

    def test_equal_fitness_lower_latency_wins(self):
        fast = ScoredConfig(make_flags(), 0.75, 2000, 300.0, 80.0)
        slow = ScoredConfig(make_flags(gpu_mem=0.85), 0.75, 2000, 800.0, 80.0)
        assert sorted([slow, fast])[0] == fast

    def test_equality_by_fingerprint(self):
        f = make_flags()
        a = make_scored(flags=f)
        b = make_scored(flags=f.copy())
        assert a == b

    def test_inequality_different_flags(self):
        a = make_scored(flags=make_flags(tp=1))
        b = make_scored(flags=make_flags(tp=2))
        assert a != b


# ===========================================================================
# 2. FitnessEvaluator
# ===========================================================================

class TestFitnessEvaluator:
    def test_score_returns_scored_config(self):
        flags = make_flags()
        enriched = make_enriched(throughput=3000.0, fitness=0.85)
        scored = FitnessEvaluator.score(flags, enriched, generation=2)
        assert isinstance(scored, ScoredConfig)
        assert scored.fitness_score == 0.85
        assert scored.throughput == 3000.0
        assert scored.generation == 2

    def test_rank_sorts_best_first(self):
        configs = [
            make_scored(fitness=0.3),
            make_scored(fitness=0.9),
            make_scored(fitness=0.6),
        ]
        ranked = FitnessEvaluator.rank(configs)
        assert ranked[0].fitness_score == 0.9
        assert ranked[-1].fitness_score == 0.3

    def test_rank_empty_list(self):
        assert FitnessEvaluator.rank([]) == []

    def test_failed_run_gets_zero_fitness(self):
        flags = make_flags()
        enriched = make_enriched(failed=True)
        scored = FitnessEvaluator.score(flags, enriched)
        assert scored.fitness_score == 0.0


# ===========================================================================
# 3. OptimiserState
# ===========================================================================

class TestOptimiserState:
    def test_all_time_best_none_initially(self, fresh_state):
        assert fresh_state.all_time_best is None

    def test_record_result_updates_history(self, fresh_state):
        flags = make_flags(tp=1)
        enriched = make_enriched(fitness=0.7)
        fresh_state.record_result(flags, enriched)
        assert len(fresh_state.scored_history) == 1

    def test_record_result_best_first(self, fresh_state):
        for tp, fitness in [(1, 0.5), (2, 0.9), (4, 0.3)]:
            fresh_state.record_result(make_flags(tp=tp), make_enriched(fitness=fitness))
        assert fresh_state.scored_history[0].fitness_score == 0.9

    def test_all_time_best_updates(self, fresh_state):
        fresh_state.record_result(make_flags(tp=1), make_enriched(fitness=0.5))
        fresh_state.record_result(make_flags(tp=2), make_enriched(fitness=0.9))
        assert abs(fresh_state.all_time_best.fitness_score - 0.9) < 1e-9

    def test_failed_config_added_to_failed_set(self, fresh_state):
        flags = make_flags(tp=4)
        enriched = make_enriched(failed=True)
        fresh_state.record_result(flags, enriched)
        assert flags.fingerprint() in fresh_state.failed_fingerprints

    def test_successful_config_not_in_failed_set(self, fresh_state):
        flags = make_flags(tp=1, gpu_mem=0.85)
        enriched = make_enriched(fitness=0.7)
        fresh_state.record_result(flags, enriched)
        assert flags.fingerprint() not in fresh_state.failed_fingerprints

    def test_is_known_bad(self, fresh_state):
        flags = make_flags(tp=8)
        fresh_state.failed_fingerprints.add(flags.fingerprint())
        assert fresh_state.is_known_bad(flags) is True

    def test_is_not_known_bad(self, fresh_state):
        flags = make_flags(tp=1)
        assert fresh_state.is_known_bad(flags) is False

    def test_is_already_seen(self, fresh_state):
        flags = make_flags()
        fresh_state.seen_fingerprints.add(flags.fingerprint())
        assert fresh_state.is_already_seen(flags) is True

    def test_top_k_returns_k_best(self, fresh_state):
        for i in range(5):
            fresh_state.record_result(
                make_flags(tp=1, gpu_mem=0.70 + i * 0.05),
                make_enriched(fitness=i * 0.2),
            )
        top3 = fresh_state.top_k(3)
        assert len(top3) == 3
        assert top3[0].fitness_score >= top3[1].fitness_score >= top3[2].fitness_score

    def test_advance_generation(self, fresh_state):
        assert fresh_state.generation == 0
        fresh_state.generation += 1
        assert fresh_state.generation == 1

    def test_serialise_deserialise(self, fresh_state):
        fresh_state.record_result(make_flags(tp=2), make_enriched(fitness=0.6))
        fresh_state.failed_fingerprints.add("deadbeef0001")
        d = fresh_state.to_dict()
        restored = OptimiserState.from_dict(d)
        assert restored.generation == fresh_state.generation
        assert len(restored.scored_history) == len(fresh_state.scored_history)
        assert "deadbeef0001" in restored.failed_fingerprints

    def test_save_and_load(self, tmp_path, fresh_state):
        fresh_state.record_result(make_flags(tp=1), make_enriched(fitness=0.55))
        path = tmp_path / "state.json"
        fresh_state.save(path)
        loaded = OptimiserState.load(path)
        assert loaded.generation == fresh_state.generation
        assert len(loaded.scored_history) == 1

    def test_load_missing_file_returns_fresh(self, tmp_path):
        state = OptimiserState.load(tmp_path / "nonexistent.json")
        assert isinstance(state, OptimiserState)
        assert state.generation == 0


# ===========================================================================
# 4. PopulationManager
# ===========================================================================

class TestPopulationManager:
    def test_seed_population_returns_valid_configs(self, ss, validator, fresh_state):
        pm = PopulationManager(ss, validator, fresh_state)
        pop = pm.seed_population("H100", size=5)
        assert len(pop) > 0
        assert len(pop) <= 5
        for flags in pop:
            assert validator.is_valid(flags)

    def test_seed_excludes_known_bad(self, ss, validator, fresh_state):
        pm = PopulationManager(ss, validator, fresh_state)
        # Mark all possible TP=2 configs as failed
        for _ in range(20):
            f = ss.sample_random()
            f.tensor_parallel_size = 2
            fresh_state.failed_fingerprints.add(f.fingerprint())
        pop = pm.seed_population("H100", size=5)
        for flags in pop:
            assert flags.fingerprint() not in fresh_state.failed_fingerprints

    def test_seed_no_duplicates(self, ss, validator, fresh_state):
        pm = PopulationManager(ss, validator, fresh_state)
        pop = pm.seed_population("H100", size=8)
        fps = [f.fingerprint() for f in pop]
        assert len(fps) == len(set(fps))

    def test_filter_and_cap_respects_size(self, ss, validator, fresh_state):
        pm = PopulationManager(ss, validator, fresh_state)
        candidates = [ss.sample_random() for _ in range(50)]
        result = pm._filter_and_cap(candidates, size=5)
        assert len(result) <= 5

    def test_filter_and_cap_skips_already_seen(self, ss, validator, fresh_state):
        pm = PopulationManager(ss, validator, fresh_state)
        flags = ss.sample_random()
        fresh_state.seen_fingerprints.add(flags.fingerprint())
        result = pm._filter_and_cap([flags], size=5)
        assert all(f.fingerprint() != flags.fingerprint() for f in result)

    def test_tournament_select_returns_scored_config(self):
        pop = [make_scored(fitness=0.3 + i * 0.1) for i in range(5)]
        selected = PopulationManager._tournament_select(pop, k=3)
        assert isinstance(selected, ScoredConfig)

    def test_tournament_select_single_candidate(self):
        pop = [make_scored(fitness=0.5)]
        selected = PopulationManager._tournament_select(pop, k=3)
        assert selected == pop[0]

    def test_generate_next_population_size(self, ss, validator, fresh_state):
        pm = PopulationManager(ss, validator, fresh_state)
        parents = [
            make_scored(fitness=0.9 - i * 0.1, flags=ss.sample_random())
            for i in range(5)
        ]
        batch = pm.generate_next_population(parents, size=4,
                                            mutation_rate=0.3,
                                            elite_fraction=0.2)
        assert len(batch) <= 4

    def test_generate_next_population_no_duplicates(self, ss, validator, fresh_state):
        pm = PopulationManager(ss, validator, fresh_state)
        parents = [
            make_scored(fitness=0.9 - i * 0.1, flags=ss.sample_random())
            for i in range(6)
        ]
        batch = pm.generate_next_population(parents, size=4,
                                            mutation_rate=0.5,
                                            elite_fraction=0.25)
        fps = [f.fingerprint() for f in batch]
        assert len(fps) == len(set(fps))

    def test_load_cross_session_failures(self, ss, validator, fresh_state, tmp_path):
        # Create a minimal CSV with failed run IDs
        csv_path = tmp_path / "runs.csv"
        import csv
        with open(csv_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=["run_id", "run_successful"])
            writer.writeheader()
            writer.writerow({"run_id": "deadbeef0001", "run_successful": "False"})
            writer.writerow({"run_id": "cafebabe0002", "run_successful": "False"})
            writer.writerow({"run_id": "goodrun00001", "run_successful": "True"})
        pm = PopulationManager(ss, validator, fresh_state)
        pm.load_cross_session_failures(csv_path)
        assert "deadbeef0001" in fresh_state.failed_fingerprints
        assert "cafebabe0002" in fresh_state.failed_fingerprints
        assert "goodrun00001" not in fresh_state.failed_fingerprints


# ===========================================================================
# 5. EvolutionaryOptimiser
# ===========================================================================

class TestEvolutionaryOptimiser:
    def test_generation_0_returns_seeded_batch(self, ss, validator, cfg):
        opt = EvolutionaryOptimiser(cfg, ss, validator)
        batch = opt.next_batch(batch_size=4)
        assert len(batch) > 0
        assert opt.generation == 0

    def test_next_batch_all_valid(self, ss, validator, cfg):
        opt = EvolutionaryOptimiser(cfg, ss, validator)
        batch = opt.next_batch(batch_size=4)
        for flags in batch:
            assert validator.is_valid(flags), f"Invalid config: {flags}"

    def test_record_result_updates_state(self, ss, validator, cfg):
        opt = EvolutionaryOptimiser(cfg, ss, validator)
        flags = ss.sample_random()
        enriched = make_enriched(fitness=0.8)
        opt.record_result(flags, enriched)
        assert len(opt.state.scored_history) == 1

    def test_best_updates_after_record(self, ss, validator, cfg):
        opt = EvolutionaryOptimiser(cfg, ss, validator)
        for i in range(3):
            opt.record_result(ss.sample_random(), make_enriched(fitness=i * 0.3))
        assert opt.best is not None
        assert abs(opt.best.fitness_score - 0.6) < 1e-6

    def test_generation_1_uses_parents(self, ss, validator, cfg):
        opt = EvolutionaryOptimiser(cfg, ss, validator)
        # Simulate gen 0 results
        for _ in range(4):
            opt.record_result(ss.sample_random(), make_enriched(fitness=0.5))
        opt.advance_generation()
        batch = opt.next_batch(batch_size=4)
        assert len(batch) > 0
        assert opt.generation == 1

    def test_no_duplicate_configs_in_batch(self, ss, validator, cfg):
        opt = EvolutionaryOptimiser(cfg, ss, validator)
        batch = opt.next_batch(batch_size=6)
        fps = [f.fingerprint() for f in batch]
        assert len(fps) == len(set(fps))

    def test_advance_generation_increments(self, ss, validator, cfg):
        opt = EvolutionaryOptimiser(cfg, ss, validator)
        opt.advance_generation()
        assert opt.generation == 1
        opt.advance_generation()
        assert opt.generation == 2

    def test_summary_dict_keys(self, ss, validator, cfg):
        opt = EvolutionaryOptimiser(cfg, ss, validator)
        s = opt.summary()
        assert "strategy" in s
        assert "generation" in s
        assert "total_runs" in s
        assert "best_fitness" in s


# ===========================================================================
# 6. GridOptimiser
# ===========================================================================

class TestGridOptimiser:
    def test_batch_contains_valid_neighbours(self, ss, validator, cfg):
        cfg.optimiser.strategy = "grid"
        opt = GridOptimiser(cfg, ss, validator)
        batch = opt.next_batch(batch_size=10)
        assert len(batch) > 0
        for flags in batch:
            assert validator.is_valid(flags)

    def test_gen0_starts_from_default_seed(self, ss, validator, cfg):
        opt = GridOptimiser(cfg, ss, validator)
        # Just confirm it runs without error
        batch = opt.next_batch(batch_size=5)
        assert isinstance(batch, list)

    def test_gen1_uses_best_as_seed(self, ss, validator, cfg):
        opt = GridOptimiser(cfg, ss, validator)
        best_flags = ss.sample_random()
        best_flags.run_id = best_flags.fingerprint()
        opt.state.scored_history = [
            make_scored(fitness=0.9, flags=best_flags)
        ]
        opt.advance_generation()
        # Should not raise
        batch = opt.next_batch(batch_size=5)
        assert isinstance(batch, list)


# ===========================================================================
# 7. RandomOptimiser
# ===========================================================================

class TestRandomOptimiser:
    def test_returns_valid_batch(self, ss, validator, cfg):
        cfg.optimiser.strategy = "random"
        opt = RandomOptimiser(cfg, ss, validator)
        batch = opt.next_batch(batch_size=5)
        assert len(batch) > 0
        for flags in batch:
            assert validator.is_valid(flags)

    def test_no_duplicates(self, ss, validator, cfg):
        opt = RandomOptimiser(cfg, ss, validator)
        batch = opt.next_batch(batch_size=5)
        fps = [f.fingerprint() for f in batch]
        assert len(fps) == len(set(fps))

    def test_skips_already_seen(self, ss, validator, cfg):
        opt = RandomOptimiser(cfg, ss, validator)
        # Pre-fill seen set with many configs
        for _ in range(100):
            opt.state.seen_fingerprints.add(ss.sample_random().fingerprint())
        batch = opt.next_batch(batch_size=3)
        for flags in batch:
            assert flags.fingerprint() not in opt.state.seen_fingerprints


# ===========================================================================
# 8. BayesianOptimiser (placeholder)
# ===========================================================================

class TestBayesianOptimiser:
    def test_falls_back_to_evolutionary_with_little_history(self, ss, validator, cfg):
        cfg.optimiser.strategy = "bayesian"
        opt = BayesianOptimiser(cfg, ss, validator)
        # No history yet → should fall back to evolutionary seed
        batch = opt.next_batch(batch_size=4)
        assert len(batch) > 0

    def test_returns_valid_configs(self, ss, validator, cfg):
        opt = BayesianOptimiser(cfg, ss, validator)
        batch = opt.next_batch(batch_size=3)
        for flags in batch:
            assert validator.is_valid(flags)


# ===========================================================================
# 9. create_optimiser factory
# ===========================================================================

class TestCreateOptimiser:
    @pytest.mark.parametrize("strategy", [
        "evolutionary", "grid", "random", "bayesian"
    ])
    def test_creates_correct_type(self, ss, validator, strategy):
        cfg = make_cfg(strategy=strategy)
        opt = create_optimiser(cfg, ss, validator)
        assert isinstance(opt, BaseOptimiser)

    def test_evolutionary_creates_evolutionary(self, ss, validator):
        cfg = make_cfg(strategy="evolutionary")
        opt = create_optimiser(cfg, ss, validator)
        assert isinstance(opt, EvolutionaryOptimiser)

    def test_grid_creates_grid(self, ss, validator):
        cfg = make_cfg(strategy="grid")
        opt = create_optimiser(cfg, ss, validator)
        assert isinstance(opt, GridOptimiser)

    def test_random_creates_random(self, ss, validator):
        cfg = make_cfg(strategy="random")
        opt = create_optimiser(cfg, ss, validator)
        assert isinstance(opt, RandomOptimiser)

    def test_unknown_strategy_raises(self, ss, validator):
        cfg = make_cfg(strategy="evolutionary")  # start valid
        cfg.optimiser.strategy = "genetic_programming"   # override to invalid
        with pytest.raises(ValueError, match="Unknown optimiser strategy"):
            create_optimiser(cfg, ss, validator)

    def test_loads_cross_session_failures_from_csv(self, ss, validator, tmp_path):
        import csv
        csv_path = tmp_path / "runs.csv"
        with open(csv_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=["run_id", "run_successful"])
            writer.writeheader()
            writer.writerow({"run_id": "old_fail_001", "run_successful": "False"})
        cfg = make_cfg()
        opt = create_optimiser(cfg, ss, validator, csv_path=csv_path)
        assert "old_fail_001" in opt.state.failed_fingerprints

    def test_resumes_from_existing_state(self, ss, validator):
        state = OptimiserState(strategy="evolutionary", generation=3)
        state.scored_history = [make_scored(fitness=0.88)]
        cfg = make_cfg()
        opt = create_optimiser(cfg, ss, validator, state=state)
        assert opt.generation == 3
        assert opt.best is not None
        assert abs(opt.best.fitness_score - 0.88) < 1e-6


# ===========================================================================
# 10. End-to-end mini optimisation loop
# ===========================================================================

class TestMiniOptimisationLoop:
    def test_two_generations_evolutionary(self, ss, validator):
        """
        Simulate two full generations of evolutionary search.
        Verifies the loop runs without error and produces improving results.
        """
        cfg = make_cfg(population_size=4, generations=2, mutation_rate=0.4)
        opt = create_optimiser(cfg, ss, validator)

        # Generation 0
        batch = opt.next_batch(batch_size=4)
        assert len(batch) > 0

        # Simulate benchmark results with increasing fitness
        for i, flags in enumerate(batch):
            score = 0.3 + i * 0.15
            opt.record_result(flags, make_enriched(fitness=score))

        best_after_gen0 = opt.best.fitness_score if opt.best else 0.0
        opt.advance_generation()

        # Generation 1
        batch = opt.next_batch(batch_size=4)
        assert len(batch) > 0

        for i, flags in enumerate(batch):
            score = 0.5 + i * 0.1
            opt.record_result(flags, make_enriched(fitness=score))

        best_after_gen1 = opt.best.fitness_score if opt.best else 0.0

        # Overall best should improve or stay same
        assert best_after_gen1 >= best_after_gen0
        assert opt.generation == 1
        assert len(opt.state.scored_history) == 8   # 4 + 4

    def test_failed_configs_not_retried(self, ss, validator):
        """Configs that fail should never appear in subsequent batches."""
        cfg = make_cfg(population_size=4)
        opt = create_optimiser(cfg, ss, validator)

        batch = opt.next_batch(batch_size=4)
        failed_fps = set()

        # Mark first 2 as failed
        for flags in batch[:2]:
            opt.record_result(flags, make_enriched(failed=True))
            failed_fps.add(flags.fingerprint())

        for flags in batch[2:]:
            opt.record_result(flags, make_enriched(fitness=0.6))

        opt.advance_generation()
        next_batch = opt.next_batch(batch_size=4)

        for flags in next_batch:
            assert flags.fingerprint() not in failed_fps, \
                "Failed config was retried in next generation"

    def test_state_round_trips_correctly(self, ss, validator, tmp_path):
        """Save and restore state, then continue optimising."""
        cfg = make_cfg(population_size=3)
        opt = create_optimiser(cfg, ss, validator)

        batch = opt.next_batch(batch_size=3)
        for flags in batch:
            opt.record_result(flags, make_enriched(fitness=0.65))
        opt.advance_generation()

        state_path = tmp_path / "opt_state.json"
        opt.state.save(state_path)

        # Restore
        restored_state = OptimiserState.load(state_path)
        opt2 = create_optimiser(cfg, ss, validator, state=restored_state)

        assert opt2.generation == 1
        assert len(opt2.state.scored_history) == 3
        assert abs(opt2.best.fitness_score - 0.65) < 1e-6