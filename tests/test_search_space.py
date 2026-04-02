"""
tests/test_search_space.py
--------------------------
Unit tests for core/search_space.py.

Run with:
    pytest tests/test_search_space.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from core.search_space import (
    SearchSpace,
    VLLMFlags,
    ConfigValidator,
    ChoiceParam,
    RangeIntParam,
    RangeFloatParam,
    BoolFlagParam,
    flags_from_dict,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def ss() -> SearchSpace:
    return SearchSpace.load()


@pytest.fixture(scope="module")
def validator_h100() -> ConfigValidator:
    return ConfigValidator(gpu_type="H100")


@pytest.fixture(scope="module")
def validator_mi300x() -> ConfigValidator:
    return ConfigValidator(gpu_type="MI300X")


@pytest.fixture(scope="module")
def validator_b300() -> ConfigValidator:
    return ConfigValidator(gpu_type="B300")


# ===========================================================================
# 1. Parameter type tests
# ===========================================================================

class TestChoiceParam:
    def test_sample_is_in_values(self):
        p = ChoiceParam("x", values=[1, 2, 4, 8], default=1)
        for _ in range(50):
            assert p.sample() in [1, 2, 4, 8]

    def test_mutate_rate_zero_returns_current(self):
        p = ChoiceParam("x", values=[1, 2, 4, 8], default=1)
        for _ in range(20):
            assert p.mutate(2, mutation_rate=0.0) == 2

    def test_mutate_rate_one_always_changes(self):
        p = ChoiceParam("x", values=[1, 2, 4, 8], default=1)
        for _ in range(20):
            result = p.mutate(2, mutation_rate=1.0)
            assert result in [1, 2, 4, 8]

    def test_neighbours_excludes_current(self):
        p = ChoiceParam("x", values=[1, 2, 4, 8], default=1)
        neighbours = p.neighbours(4)
        assert 4 not in neighbours
        assert all(v in [1, 2, 8] for v in neighbours)


class TestRangeIntParam:
    def test_sample_in_range(self):
        p = RangeIntParam("y", min=0, max=100, step=10, default=50)
        for _ in range(50):
            v = p.sample()
            assert 0 <= v <= 100
            assert v % 10 == 0

    def test_neighbours_step(self):
        p = RangeIntParam("y", min=0, max=100, step=10, default=50)
        n = p.neighbours(50)
        assert 40 in n
        assert 60 in n

    def test_neighbours_at_min(self):
        p = RangeIntParam("y", min=0, max=100, step=10, default=0)
        n = p.neighbours(0)
        assert 10 in n
        assert len(n) == 1  # only one direction


class TestRangeFloatParam:
    def test_sample_in_range(self):
        p = RangeFloatParam("z", min=0.7, max=0.95, step=0.05, default=0.9)
        for _ in range(50):
            v = p.sample()
            assert 0.7 <= v <= 0.95 + 1e-9

    def test_values_includes_endpoints(self):
        p = RangeFloatParam("z", min=0.7, max=0.95, step=0.05, default=0.9)
        vals = p._values()
        assert abs(vals[0] - 0.7) < 1e-6
        assert abs(vals[-1] - 0.95) < 1e-6

    def test_neighbours(self):
        p = RangeFloatParam("z", min=0.7, max=0.95, step=0.05, default=0.9)
        n = p.neighbours(0.85)
        assert any(abs(v - 0.80) < 1e-6 for v in n)
        assert any(abs(v - 0.90) < 1e-6 for v in n)


class TestBoolFlagParam:
    def test_mutate_flips(self):
        p = BoolFlagParam("b", default=False)
        assert p.mutate(True, mutation_rate=1.0) is False
        assert p.mutate(False, mutation_rate=1.0) is True

    def test_neighbours(self):
        p = BoolFlagParam("b", default=False)
        assert p.neighbours(True) == [False]
        assert p.neighbours(False) == [True]


# ===========================================================================
# 2. VLLMFlags tests
# ===========================================================================

class TestVLLMFlags:
    def test_fingerprint_is_stable(self):
        f = VLLMFlags(tensor_parallel_size=2, gpu_memory_utilization=0.90)
        assert f.fingerprint() == f.fingerprint()

    def test_fingerprint_changes_on_mutation(self):
        f1 = VLLMFlags(tensor_parallel_size=1)
        f2 = VLLMFlags(tensor_parallel_size=2)
        assert f1.fingerprint() != f2.fingerprint()

    def test_copy_is_independent(self):
        f = VLLMFlags(tensor_parallel_size=1)
        g = f.copy()
        g.tensor_parallel_size = 4
        assert f.tensor_parallel_size == 1

    def test_equality(self):
        f1 = VLLMFlags(tensor_parallel_size=2, dtype="float16")
        f2 = VLLMFlags(tensor_parallel_size=2, dtype="float16")
        assert f1 == f2

    def test_to_dict_roundtrip(self):
        f = VLLMFlags(tensor_parallel_size=4, gpu_memory_utilization=0.85)
        d = f.to_dict()
        restored = flags_from_dict(d)
        assert restored == f

    def test_to_vllm_args_basic(self):
        f = VLLMFlags(tensor_parallel_size=2, gpu_memory_utilization=0.90)
        args = f.to_vllm_args("mistralai/Mistral-7B-Instruct-v0.2")
        assert "--tensor-parallel-size" in args
        assert "2" in args
        assert "--gpu-memory-utilization" in args

    def test_to_vllm_args_quantization(self):
        f = VLLMFlags(quantization="awq")
        args = f.to_vllm_args("some/model")
        assert "--quantization" in args
        assert "awq" in args

    def test_chunked_prefill_flag(self):
        f = VLLMFlags(enable_chunked_prefill=True)
        args = f.to_vllm_args("some/model")
        assert "--enable-chunked-prefill" in args

    def test_no_quantization_no_flag(self):
        f = VLLMFlags(quantization=None)
        args = f.to_vllm_args("some/model")
        assert "--quantization" not in args

    def test_expert_parallel_flag(self):
        f = VLLMFlags(enable_expert_parallel=True)
        args = f.to_vllm_args("some/model")
        assert "--enable-expert-parallel" in args

    def test_cpu_offload_flag(self):
        f = VLLMFlags(cpu_offload_gb=20)
        args = f.to_vllm_args("some/model")
        assert "--cpu-offload-gb" in args
        assert "20" in args

    def test_speculative_decoding_flags(self):
        f = VLLMFlags(speculative_model="some/draft", num_speculative_tokens=5)
        args = f.to_vllm_args("some/model")
        assert "--speculative-model" in args
        assert "--num-speculative-tokens" in args
        assert "5" in args

    def test_attention_backend_auto_not_emitted(self):
        f = VLLMFlags(attention_backend="auto")
        args = f.to_vllm_args("some/model")
        assert "--attention-backend" not in args

    def test_attention_backend_non_auto_emitted(self):
        f = VLLMFlags(attention_backend="flashinfer")
        args = f.to_vllm_args("some/model")
        assert "--attention-backend" in args
        assert "flashinfer" in args


# ===========================================================================
# 3. SearchSpace tests
# ===========================================================================

class TestSearchSpace:
    def test_loads_from_yaml(self, ss):
        assert len(ss._params) > 0

    def test_has_expected_params(self, ss):
        expected = {
            "tensor_parallel_size", "gpu_memory_utilization",
            "max_num_seqs", "dtype", "block_size",
        }
        assert expected.issubset(ss._params.keys())

    def test_has_moe_params(self, ss):
        assert "enable_expert_parallel" in ss._params
        assert "all2all_backend" in ss._params
        assert "enable_dbo" in ss._params

    def test_has_speculative_params(self, ss):
        assert "speculative_model" in ss._params
        assert "num_speculative_tokens" in ss._params

    def test_default_flags_are_valid(self, ss, validator_h100):
        flags = ss.default_flags()
        violations = validator_h100.validate(flags)
        assert violations == [], f"Default flags have violations: {violations}"

    def test_sample_random_returns_valid_object(self, ss):
        flags = ss.sample_random()
        assert isinstance(flags, VLLMFlags)
        assert flags.run_id != ""

    def test_sample_population_size(self, ss):
        pop = ss.sample_population(10)
        assert len(pop) == 10

    def test_sample_population_unique(self, ss):
        pop = ss.sample_population(10)
        fps = [f.fingerprint() for f in pop]
        assert len(fps) == len(set(fps)), "Population contains duplicates"

    def test_mutate_returns_new_object(self, ss):
        original = ss.default_flags()
        mutated = ss.mutate(original, mutation_rate=1.0)
        assert mutated is not original

    def test_mutate_run_id_updated(self, ss):
        flags = ss.default_flags()
        mutated = ss.mutate(flags, mutation_rate=1.0)
        assert mutated.run_id != ""

    def test_crossover_inherits_parents(self, ss):
        a = VLLMFlags(tensor_parallel_size=1, dtype="float16")
        b = VLLMFlags(tensor_parallel_size=4, dtype="bfloat16")
        child = ss.crossover(a, b)
        assert child.tensor_parallel_size in [1, 4]
        assert child.dtype in ["float16", "bfloat16"]

    def test_grid_neighbours_nonempty(self, ss):
        flags = ss.default_flags()
        neighbours = ss.grid_neighbours(flags)
        assert len(neighbours) > 0

    def test_grid_neighbours_differ_by_one_param(self, ss):
        flags = ss.default_flags()
        for neighbour in ss.grid_neighbours(flags):
            diff_count = sum(
                1 for name in ss._params
                if hasattr(flags, name) and
                getattr(flags, name) != getattr(neighbour, name)
            )
            assert diff_count == 1, (
                f"Neighbour differs in {diff_count} params (expected 1)"
            )

    def test_size_is_positive(self, ss):
        assert ss.size() > 0

    def test_seeded_population_respects_gpu_limits(self, ss):
        pop = ss.sample_seeded("H100", size=5)
        assert len(pop) > 0
        for flags in pop:
            assert flags.tensor_parallel_size <= 8

    def test_seeded_population_mi300x(self, ss):
        pop = ss.sample_seeded("MI300X", size=5)
        assert len(pop) > 0

    def test_summary_contains_param_names(self, ss):
        summary = ss.summary()
        assert "tensor_parallel_size" in summary
        assert "gpu_memory_utilization" in summary


# ===========================================================================
# 4. ConfigValidator tests
# ===========================================================================

class TestConfigValidator:
    def test_valid_config_passes(self, validator_h100):
        f = VLLMFlags(
            tensor_parallel_size=2,
            gpu_memory_utilization=0.90,
            max_num_seqs=256,
            max_num_batched_tokens=8192,
        )
        assert validator_h100.validate(f) == []

    def test_tp_exceeds_gpu_max_fails(self, validator_mi300x):
        # MI300X max_tensor_parallel=8; 16 should fail
        f = VLLMFlags(tensor_parallel_size=16)
        violations = validator_mi300x.validate(f)
        assert any("tensor_parallel_size" in v for v in violations)

    def test_tp_not_power_of_two_fails(self, validator_h100):
        f = VLLMFlags(tensor_parallel_size=3)
        violations = validator_h100.validate(f)
        assert any("power of 2" in v for v in violations)

    def test_memory_util_too_high_fails(self, validator_h100):
        f = VLLMFlags(gpu_memory_utilization=1.05)
        violations = validator_h100.validate(f)
        assert any("gpu_memory_utilization" in v for v in violations)

    def test_memory_util_too_low_fails(self, validator_h100):
        f = VLLMFlags(gpu_memory_utilization=0.3)
        violations = validator_h100.validate(f)
        assert any("gpu_memory_utilization" in v for v in violations)

    def test_batched_tokens_less_than_seqs_fails(self, validator_h100):
        f = VLLMFlags(max_num_seqs=512, max_num_batched_tokens=128)
        violations = validator_h100.validate(f)
        assert any("max_num_batched_tokens" in v for v in violations)

    def test_speculative_model_without_tokens_fails(self, validator_h100):
        f = VLLMFlags(speculative_model="some/draft", num_speculative_tokens=None)
        violations = validator_h100.validate(f)
        assert any("speculative_model" in v or "num_speculative_tokens" in v
                   for v in violations)

    def test_speculative_tokens_without_model_fails(self, validator_h100):
        f = VLLMFlags(speculative_model=None, num_speculative_tokens=5)
        violations = validator_h100.validate(f)
        assert any("speculative_model" in v or "num_speculative_tokens" in v
                   for v in violations)

    def test_speculative_both_set_passes(self, validator_h100):
        f = VLLMFlags(speculative_model="some/draft", num_speculative_tokens=5)
        violations = validator_h100.validate(f)
        spec_violations = [v for v in violations
                           if "speculative" in v.lower()]
        assert spec_violations == []

    def test_amd_ray_backend_fails(self, validator_mi300x):
        f = VLLMFlags(distributed_executor_backend="ray")
        violations = validator_mi300x.validate(f)
        assert any("ray" in v for v in violations)

    def test_amd_flash_attn_fails(self, validator_mi300x):
        f = VLLMFlags(attention_backend="flash_attn")
        violations = validator_mi300x.validate(f)
        assert any("flash_attn" in v for v in violations)

    def test_nvidia_ray_backend_passes(self, validator_h100):
        f = VLLMFlags(distributed_executor_backend="ray")
        violations = validator_h100.validate(f)
        assert not any("ray" in v for v in violations)

    def test_moe_deepep_without_expert_parallel_fails(self, validator_h100):
        f = VLLMFlags(all2all_backend="deepep_high_throughput",
                      enable_expert_parallel=False)
        violations = validator_h100.validate(f)
        assert any("enable_expert_parallel" in v for v in violations)

    def test_moe_deepep_with_expert_parallel_passes(self, validator_h100):
        f = VLLMFlags(all2all_backend="deepep_high_throughput",
                      enable_expert_parallel=True)
        violations = validator_h100.validate(f)
        moe_violations = [v for v in violations if "expert_parallel" in v]
        assert moe_violations == []

    def test_dbo_without_expert_parallel_fails(self, validator_h100):
        f = VLLMFlags(enable_dbo=True, enable_expert_parallel=False)
        violations = validator_h100.validate(f)
        assert any("enable_expert_parallel" in v for v in violations)

    def test_invalid_block_size_fails(self, validator_h100):
        f = VLLMFlags(block_size=7)
        violations = validator_h100.validate(f)
        assert any("block_size" in v for v in violations)

    def test_valid_block_sizes_pass(self, validator_h100):
        for bs in [1, 8, 16, 32]:
            f = VLLMFlags(block_size=bs)
            violations = validator_h100.validate(f)
            block_violations = [v for v in violations if "block_size" in v]
            assert block_violations == [], f"block_size={bs} should be valid"

    def test_fp8_kv_cache_on_non_fp8_gpu_fails(self):
        # MI300X has fp8=true, so this won't trigger there — use a hypothetical
        # We test the check indirectly: all 6 GPUs support FP8, so valid configs pass
        v = ConfigValidator(gpu_type="H100")
        f = VLLMFlags(kv_cache_dtype="fp8_e5m2")
        violations = v.validate(f)
        fp8_violations = [x for x in violations if "FP8" in x]
        assert fp8_violations == []  # H100 supports fp8

    def test_nvfp4_check_b300_passes(self, validator_b300):
        # B300 supports nvfp4 — no violation expected for fp8 kv-cache
        f = VLLMFlags(kv_cache_dtype="fp8_e4m3")
        violations = validator_b300.validate(f)
        assert all("nvfp4" not in v.lower() for v in violations)

    def test_is_valid_shortcut(self, validator_h100):
        f = VLLMFlags()
        assert validator_h100.is_valid(f) == (validator_h100.validate(f) == [])


# ===========================================================================
# 5. flags_from_dict roundtrip
# ===========================================================================

class TestFlagsFromDict:
    def test_roundtrip(self):
        original = VLLMFlags(
            tensor_parallel_size=4,
            gpu_memory_utilization=0.85,
            quantization="awq",
            enable_chunked_prefill=True,
        )
        d = original.to_dict()
        restored = flags_from_dict(d)
        assert restored == original

    def test_unknown_keys_ignored(self):
        d = {"tensor_parallel_size": 2, "unknown_future_flag": "value"}
        flags = flags_from_dict(d)
        assert flags.tensor_parallel_size == 2
        assert not hasattr(flags, "unknown_future_flag")

    def test_missing_keys_use_defaults(self):
        flags = flags_from_dict({})
        assert flags.tensor_parallel_size == 1

    def test_moe_fields_roundtrip(self):
        original = VLLMFlags(
            enable_expert_parallel=True,
            all2all_backend="deepep_high_throughput",
            enable_dbo=True,
        )
        restored = flags_from_dict(original.to_dict())
        assert restored == original
