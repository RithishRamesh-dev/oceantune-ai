"""Tests for prelude, convergence, paired bench, search policy, patches, session, quant."""

from pathlib import Path

from core.measurement_convergence import assess_convergence, stable_fitness
from core.paired_bench import PairedProbe, evaluate_paired_probes
from core.warmstart_policy import decide_warmstart
from core.prelude import build_prelude_plan, accept_warm_replay
from core.search_policy import (
    MODE_DIVERSIFY,
    MODE_EXPLOIT,
    SearchPolicyState,
    update_search_policy,
)
from core.quantization_schemes import (
    resolve_quantization,
    schemes_for_gpu,
    build_quantization_prompt,
    QuantizationConfig,
)
from core.serving_patches import (
    select_patches,
    version_in_range,
    apply_verification_marker,
    DEFAULT_PATCH_ROOT,
)
from core.session_checkpoint import (
    RoundBudget,
    new_checkpoint,
    save_checkpoint,
    load_checkpoint,
    session_remaining_sec,
    should_skip_phase,
)
from core.attempt_ledger import AttemptLedger
from core.framework_backend import remap_flags_for_sglang, SGLangBackend
from core.policy_gate import PolicyGate


def test_convergence_ok():
    a = assess_convergence([100, 110, 108, 109, 111], warmup_discard=1)
    assert a.converged
    assert a.reason == "ok"


def test_convergence_monotonic():
    a = assess_convergence([10, 20, 30, 40], warmup_discard=0, min_samples=3)
    assert not a.converged
    assert a.reason == "monotonic_climb"


def test_convergence_spread():
    a = assess_convergence([100, 100, 200], warmup_discard=0, max_spread_pct=10)
    assert not a.converged
    assert a.reason == "spread_too_high"


def test_stable_fitness():
    assert stable_fitness([1, 10, 12, 11], warmup_discard=1) == 11.0


def test_paired_bench_keep():
    probes = [
        PairedProbe(100, 110, "a"),
        PairedProbe(100, 112, "b"),
    ]
    r = evaluate_paired_probes(probes, noise_band=0.01, min_agreeing_pairs=2)
    assert r.keep
    assert r.sign_agreement


def test_paired_bench_disagreement():
    probes = [
        PairedProbe(100, 110, "a"),
        PairedProbe(100, 90, "b"),
    ]
    r = evaluate_paired_probes(probes, min_agreeing_pairs=2)
    assert not r.keep


def test_warmstart_confidence_gate():
    d = decide_warmstart(
        flags={"gpu_memory_utilization": 0.9},
        claimed_fitness=100,
        confidence=0.4,
        min_confidence=0.7,
    )
    assert not d.accept
    assert "confidence" in d.reason


def test_prelude_plan_and_accept():
    recipe = {
        "best_flags": {"gpu_memory_utilization": 0.9, "max_num_seqs": 128},
        "best_fitness": 100.0,
        "confidence": 0.9,
        "what_failed": [{"fingerprint": "abc123", "error": "OOM"}],
        "pitfalls": [{"description": "fp8 broke MLA"}],
    }
    plan = build_prelude_plan(recipe=recipe, min_confidence=0.7)
    assert plan.should_benchmark_replay
    assert "abc123" in plan.reject_fingerprints
    result = accept_warm_replay(plan=plan, measured_fitness=90.0)
    assert result.replay_accepted
    assert result.seed_flags["max_num_seqs"] == 128


def test_search_policy_diversify_on_stall():
    st = SearchPolicyState()
    d1 = update_search_policy(st, current_fitness=100.0, stall_limit=2)
    assert d1.mode == MODE_EXPLOIT
    update_search_policy(st, current_fitness=100.1, stall_epsilon_pct=0.5, stall_limit=2)
    d3 = update_search_policy(st, current_fitness=100.2, stall_epsilon_pct=0.5, stall_limit=2)
    assert d3.mode == MODE_DIVERSIFY
    assert d3.force_new_families


def test_quant_nvfp4_blackwell_only():
    bad = resolve_quantization(scheme="nvfp4", gpu_type="H100")
    assert not bad.allowed
    ok = resolve_quantization(scheme="nvfp4", gpu_type="B300")
    assert ok.allowed
    assert "quantization" in ok.flags_delta
    assert "nvfp4" in schemes_for_gpu("B300")
    prompt = build_quantization_prompt(QuantizationConfig(global_scheme="fp8_kv", kv_cache_dtype="fp8"))
    assert "fp8" in prompt.lower()


def test_serving_patches_manifest_loads():
    assert DEFAULT_PATCH_ROOT.exists()
    sel = select_patches(framework="vllm", framework_version="0.8.0")
    # default off without env
    assert isinstance(sel.selected, list)
    assert version_in_range("0.7.1", "0.6.0", "1.0.0")
    assert apply_verification_marker("fused_residual_rmsnorm").startswith("OCEANTUNE_SHADOW")


def test_session_checkpoint_roundtrip(tmp_path, monkeypatch):
    import core.session_checkpoint as sc
    monkeypatch.setattr(sc, "SESSIONS_DIR", tmp_path)
    cp = new_checkpoint("sess-test", budget=RoundBudget(session_max_sec=3600))
    save_checkpoint(cp)
    loaded = load_checkpoint("sess-test")
    assert loaded is not None
    assert loaded.session_id == "sess-test"
    rem = session_remaining_sec(loaded)
    assert rem is not None and rem > 0
    loaded.phase = "stage2"
    assert should_skip_phase(loaded, "stage1")
    assert not should_skip_phase(loaded, "stage2")


def test_attempt_ledger(tmp_path, monkeypatch):
    import core.attempt_ledger as al
    monkeypatch.setattr(al, "REPO_ROOT", tmp_path)
    led = AttemptLedger(session_id="s1", persist=True)
    led.record(phase="stage1", fingerprint="deadbeef", error="CUDA out of memory", kept=False)
    assert led.failures()[0].error_class == "oom"
    assert "do not repeat" in led.planner_denylist_text().lower()


def test_sglang_remap_expanded():
    norm = remap_flags_for_sglang({
        "tensor_parallel_size": 2,
        "block_size": 16,
        "speculative_model": "tiny",
        "num_speculative_tokens": 5,
        "unknown_x": 1,
    })
    assert "block_size" in norm
    assert "speculative_model" in norm
    assert "unknown_x" not in norm
    spec = SGLangBackend().build_launch_spec(
        model_id="m",
        flags=norm,
        port=8001,
        gpu_type="H200",
    )
    assert "sglang.launch_server" in spec.cli_args
    assert any("--page-size" in a or a == "--page-size" for a in spec.cli_args) or "--page-size" in spec.cli_args


def test_policy_prelude_stage():
    g = PolicyGate()
    assert g.check_action("prelude", "warm_replay").allowed
    assert not g.check_action("prelude", "generate_kernel").allowed
