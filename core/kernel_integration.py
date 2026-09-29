"""
core/kernel_integration.py
--------------------------
KernelIntegrationLayer — bridge Stage 4 generated kernels into serving validation.

Hyperloom never trusts microbench-only KEPT kernels: it re-measures under the
session protocol. OceanTune Stage 4 historically stopped at OperatorBench.

This module provides:
  1. IntegrationPlan — what would need to change in vLLM to load a custom kernel
  2. validate_kernel_artifact — filesystem + correctness metadata checks
  3. ShadowPatchSpec — Docker volume + PYTHONPATH + env gate for a kernel
  4. run_e2e_rebench — start shadow vLLM, ramp vs incumbent fitness, MeasurementGate

No Hyperloom source is vendored. Design is OceanTune-native.
"""

from __future__ import annotations

import logging
import os
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.measurement_gate import MeasurementGate
from core.snr_contract import evaluate_keep

log = logging.getLogger("core.kernel_integration")

_SHADOW_ROOT = Path(__file__).resolve().parent.parent / "storage" / "shadow_kernels"


# ---------------------------------------------------------------------------
# Shadow hook registry — documented vLLM import-time patch targets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShadowHook:
    """
    One env-gated monkey-patch site in the serving stack.

    ``import_targets`` are module paths that *would* be patched once a real
    adapter is implemented for a given vLLM version. Until then, sitecustomize
    logs the intended targets when the env gate is set (smoke path).
    """

    hook_id: str
    op_type: str
    env_flag: str
    import_targets: Tuple[str, ...]
    fusion_pattern_id: str = ""
    notes: str = ""
    min_vllm_version: str = "0.6.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hook_id": self.hook_id,
            "op_type": self.op_type,
            "env_flag": self.env_flag,
            "import_targets": list(self.import_targets),
            "fusion_pattern_id": self.fusion_pattern_id,
            "notes": self.notes,
            "min_vllm_version": self.min_vllm_version,
        }


# OceanTune-owned registry. Targets are documentation + future patch sites,
# not live hooks until a version-specific adapter lands.
SHADOW_HOOKS: Tuple[ShadowHook, ...] = (
    ShadowHook(
        hook_id="fused_residual_rmsnorm",
        op_type="rmsnorm",
        env_flag="OCEANTUNE_FUSED_RESIDUAL",
        fusion_pattern_id="residual_add_rmsnorm",
        import_targets=(
            "vllm.model_executor.layers.layernorm",
            "vllm.model_executor.models",
        ),
        notes="Replace RMSNorm(+residual) with fused Triton when available.",
    ),
    ShadowHook(
        hook_id="fused_silu_mul",
        op_type="activation",
        env_flag="OCEANTUNE_FUSED_SILU",
        fusion_pattern_id="swiglu_silu_mul",
        import_targets=(
            "vllm.model_executor.layers.activation",
        ),
        notes="SiluAndMul fusion site.",
    ),
    ShadowHook(
        hook_id="fused_qk_rope",
        op_type="rope",
        env_flag="OCEANTUNE_FUSED_QK",
        fusion_pattern_id="qk_norm_rope",
        import_targets=(
            "vllm.model_executor.layers.rotary_embedding",
            "vllm.attention",
        ),
        notes="Q/K norm + RoPE fuse site.",
    ),
    ShadowHook(
        hook_id="attention_custom",
        op_type="attention",
        env_flag="OCEANTUNE_SHADOW_KERNEL",
        import_targets=(
            "vllm.attention.backends",
            "vllm.v1.attention",
        ),
        notes="Generic Stage-4 attention kernel shadow (backend-specific).",
    ),
    ShadowHook(
        hook_id="moe_dispatch_custom",
        op_type="moe",
        env_flag="OCEANTUNE_SHADOW_MOE",
        import_targets=(
            "vllm.model_executor.layers.fused_moe",
        ),
        notes="MoE dispatch/combine custom kernel gate.",
    ),
)


class ShadowHookRegistry:
    """Lookup hooks by op_type, env_flag, or fusion pattern id."""

    def __init__(self, hooks: Optional[List[ShadowHook]] = None) -> None:
        self._hooks: List[ShadowHook] = list(hooks or SHADOW_HOOKS)

    def all(self) -> List[ShadowHook]:
        return list(self._hooks)

    def by_env_flag(self, env_flag: str) -> Optional[ShadowHook]:
        for h in self._hooks:
            if h.env_flag == env_flag:
                return h
        return None

    def by_fusion_pattern(self, pattern_id: str) -> Optional[ShadowHook]:
        for h in self._hooks:
            if h.fusion_pattern_id == pattern_id:
                return h
        return None

    def by_op_type(self, op_type: str) -> List[ShadowHook]:
        return [h for h in self._hooks if h.op_type == op_type]

    def resolve_for_plan(
        self,
        *,
        op_type: str,
        env_flag: str = "",
        fusion_pattern_id: str = "",
    ) -> Optional[ShadowHook]:
        if env_flag:
            hit = self.by_env_flag(env_flag)
            if hit:
                return hit
        if fusion_pattern_id:
            hit = self.by_fusion_pattern(fusion_pattern_id)
            if hit:
                return hit
        matches = self.by_op_type(op_type)
        return matches[0] if matches else None


@dataclass
class IntegrationPlan:
    """Describes how a Stage 4 kernel would enter the serving path."""

    op_type: str
    kernel_path: str
    status: str  # planned | artifact_ok | blocked | e2e_pending | integrated | e2e_rejected
    blockers: List[str] = field(default_factory=list)
    recommended_steps: List[str] = field(default_factory=list)
    micro_speedup_pct: float = 0.0
    e2e_verified: bool = False
    e2e_fitness: float = 0.0
    incumbent_fitness: float = 0.0
    shadow_env: Dict[str, str] = field(default_factory=dict)
    keep_eval: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShadowPatchSpec:
    """
    How to inject a generated kernel into a vLLM Docker container.

    Mechanism (OceanTune-native):
      - Host bind-mount of a shadow package directory
      - PYTHONPATH prepend inside the container
      - sitecustomize.py that imports the shadow kernel when env gate is set
      - Env flag (fusion pattern or OCEANTUNE_SHADOW_KERNEL=1) to enable
    """

    kernel_path: str
    op_type: str
    container_mount: str = "/opt/oceantune_shadow"
    env_flag: str = "OCEANTUNE_SHADOW_KERNEL"
    package_name: str = "oceantune_shadow"

    def host_shadow_dir(self, session_id: str) -> Path:
        return _SHADOW_ROOT / session_id[:16] / self.op_type

    def docker_volume_args(self, session_id: str) -> List[str]:
        host = str(self.host_shadow_dir(session_id).resolve())
        return ["-v", f"{host}:{self.container_mount}:ro"]

    def docker_env(self) -> Dict[str, str]:
        return {
            self.env_flag: "1",
            "PYTHONPATH": f"{self.container_mount}:{os.environ.get('PYTHONPATH', '')}".rstrip(":"),
            "OCEANTUNE_SHADOW_OP": self.op_type,
            "OCEANTUNE_SHADOW_KERNEL_FILE": f"{self.container_mount}/{self.package_name}/kernel.py",
        }


def validate_kernel_artifact(kernel_path: str) -> List[str]:
    """Return list of blocker strings; empty means artifact looks loadable."""
    blockers: List[str] = []
    if not kernel_path:
        blockers.append("empty_kernel_path")
        return blockers
    p = Path(kernel_path)
    if not p.is_file():
        blockers.append(f"missing_file:{kernel_path}")
        return blockers
    text = p.read_text(encoding="utf-8", errors="replace")
    if "TODO: Implement" in text or "LLM generation failed" in text:
        blockers.append("skeleton_or_failed_generation")
    if "@triton.jit" not in text and "torch." not in text:
        blockers.append("no_triton_or_torch_kernel_detected")
    return blockers


def plan_integration(
    *,
    op_type: str,
    kernel_path: str,
    micro_speedup_pct: float = 0.0,
    min_speedup_pct: float = 1.0,
    env_flag: str = "OCEANTUNE_SHADOW_KERNEL",
) -> IntegrationPlan:
    """Build an IntegrationPlan for a Stage 4 kernel."""
    blockers = validate_kernel_artifact(kernel_path)
    steps = [
        "Confirm CorrectnessFirewall + SNR ≥ 30 dB for target shapes",
        "Materialize shadow package (kernel.py + sitecustomize hook)",
        "Start vLLM Docker with bind-mount + PYTHONPATH + env gate",
        "Run MeasurementGate E2E ramp vs Stage 3 winner flags",
        "KEEP only if E2E fitness improves; sediment into Recipe KB",
    ]
    if micro_speedup_pct < min_speedup_pct:
        blockers.append(f"micro_speedup_below_{min_speedup_pct}%")

    if blockers:
        return IntegrationPlan(
            op_type=op_type,
            kernel_path=kernel_path,
            status="blocked",
            blockers=blockers,
            recommended_steps=steps,
            micro_speedup_pct=micro_speedup_pct,
            e2e_verified=False,
        )

    return IntegrationPlan(
        op_type=op_type,
        kernel_path=kernel_path,
        status="e2e_pending",
        blockers=[],
        recommended_steps=steps,
        micro_speedup_pct=micro_speedup_pct,
        e2e_verified=False,
        shadow_env={env_flag: "1"},
    )


def materialize_shadow_package(
    *,
    kernel_path: str,
    op_type: str,
    session_id: str,
    env_flag: str = "OCEANTUNE_SHADOW_KERNEL",
    fusion_pattern_id: str = "",
    registry: Optional[ShadowHookRegistry] = None,
) -> ShadowPatchSpec:
    """Copy kernel into a shadow package with a hook-aware sitecustomize."""
    reg = registry or ShadowHookRegistry()
    hook = reg.resolve_for_plan(
        op_type=op_type,
        env_flag=env_flag,
        fusion_pattern_id=fusion_pattern_id,
    )
    effective_flag = hook.env_flag if hook else env_flag
    targets = list(hook.import_targets) if hook else []

    spec = ShadowPatchSpec(
        kernel_path=kernel_path,
        op_type=op_type,
        env_flag=effective_flag,
    )
    dest = spec.host_shadow_dir(session_id)
    pkg = dest / spec.package_name
    pkg.mkdir(parents=True, exist_ok=True)

    src = Path(kernel_path)
    (pkg / "kernel.py").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    (pkg / "__init__.py").write_text(
        f'"""OceanTune shadow kernel package for op={op_type}."""\n',
        encoding="utf-8",
    )
    # Persist hook metadata for operators / Stage 4b campaign runners
    import json as _json
    (pkg / "hook.json").write_text(
        _json.dumps(
            {
                "op_type": op_type,
                "env_flag": effective_flag,
                "fusion_pattern_id": fusion_pattern_id or (
                    hook.fusion_pattern_id if hook else ""
                ),
                "import_targets": targets,
                "hook_id": hook.hook_id if hook else "",
                "status": "smoke_import_only",
                "notes": (
                    "Targets are documented patch sites. Real monkey-patch "
                    "adapters are version-gated and not applied in smoke mode."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    targets_repr = repr(targets)
    (dest / "sitecustomize.py").write_text(
        textwrap.dedent(
            f"""\
            import os, logging
            log = logging.getLogger("oceantune.shadow")
            if os.environ.get("{effective_flag}") == "1":
                _targets = {targets_repr}
                log.warning(
                    "OceanTune shadow kernel active op=%s file=%s "
                    "documented_targets=%s (smoke: import only)",
                    os.environ.get("OCEANTUNE_SHADOW_OP"),
                    os.environ.get("OCEANTUNE_SHADOW_KERNEL_FILE"),
                    _targets,
                )
                try:
                    import oceantune_shadow.kernel  # noqa: F401
                except Exception as exc:
                    log.error("Shadow kernel import failed: %s", exc)
                # Future: version-specific adapters patch `_targets` here.
            """
        ),
        encoding="utf-8",
    )
    log.info(
        "Materialized shadow package at %s (hook=%s targets=%s)",
        dest,
        hook.hook_id if hook else "none",
        targets,
    )
    return spec


async def run_e2e_rebench(
    *,
    plan: IntegrationPlan,
    session_id: str,
    model_id: str,
    gpu_type: str,
    flags: Dict[str, Any],
    incumbent_fitness: float,
    gpu_alloc: Any,
    port_alloc: Any,
    docker_image: str = "",
    startup_timeout_sec: int = 1200,
    concurrency_levels: Optional[List[int]] = None,
    num_prompts: int = 20,
    input_len: int = 1024,
    output_len: int = 1024,
    primary_metric: str = "throughput",
    env_flag: str = "OCEANTUNE_SHADOW_KERNEL",
    dry_run: bool = False,
) -> IntegrationPlan:
    """
    Docker shadow-patch E2E rebench against incumbent fitness.

    When ``dry_run=True`` (unit tests / no GPU), materializes the package and
    returns ``e2e_pending`` with shadow paths populated — no Docker start.
    """
    if plan.status == "blocked":
        return plan

    blockers = validate_kernel_artifact(plan.kernel_path)
    if blockers:
        plan.status = "blocked"
        plan.blockers = blockers
        return plan

    spec = materialize_shadow_package(
        kernel_path=plan.kernel_path,
        op_type=plan.op_type,
        session_id=session_id,
        env_flag=env_flag,
    )
    plan.shadow_env = spec.docker_env()

    if dry_run:
        plan.status = "e2e_pending"
        plan.blockers = list(plan.blockers) + ["dry_run_no_docker"]
        plan.incumbent_fitness = incumbent_fitness
        return plan

    from core.benchmark_runner import BenchmarkEngine
    from core.log_analyzer import LogAnalyzer
    from core.metrics_collector import MetricsCollector
    from core.search_space import VLLMFlags
    from core.vllm_server import VLLMServer, _load_gpu_profile

    known = set(VLLMFlags.__dataclass_fields__)
    clean = {k: v for k, v in (flags or {}).items() if k in known}
    try:
        vf = VLLMFlags(**clean)
    except Exception as exc:
        plan.status = "blocked"
        plan.blockers.append(f"bad_flags:{exc}")
        return plan

    tp = vf.tensor_parallel_size or 1
    slot = await gpu_alloc.acquire(tp)
    if slot is None:
        plan.status = "e2e_pending"
        plan.blockers.append("no_gpu_slot_for_e2e")
        return plan
    port = await port_alloc.acquire()
    if port is None:
        await gpu_alloc.release(slot)
        plan.status = "e2e_pending"
        plan.blockers.append("no_port_for_e2e")
        return plan

    device_env = gpu_alloc.build_device_env(slot)
    extra_env = {**device_env, **spec.docker_env()}
    server = VLLMServer(
        model_id=model_id,
        flags=vf,
        gpu_type=gpu_type,
        port=port,
        startup_timeout=startup_timeout_sec,
        extra_env=extra_env,
        docker_image=docker_image,
        extra_docker_args=spec.docker_volume_args(session_id),
    )

    fitness = 0.0
    error: Optional[str] = None
    try:
        await server.start()
        levels = concurrency_levels or [1, 4, 16, 64]
        engine = BenchmarkEngine(
            base_url=f"http://localhost:{port}",
            model_id=model_id,
            concurrency_levels=levels,
            num_prompts=num_prompts,
            input_len=input_len,
            output_len=output_len,
        )
        ramp = await engine.run()
        analysis = LogAnalyzer.analyze(server.log_tail)
        gpu_profile = _load_gpu_profile(gpu_type)
        enriched = MetricsCollector.collect(
            ramp=ramp,
            analysis=analysis,
            flags=vf,
            gpu_profile=gpu_profile,
            primary_metric=primary_metric,
        )
        fitness = float(enriched.fitness_score or 0.0)
    except Exception as exc:
        error = str(exc)[:300]
        log.warning("E2E shadow rebench failed: %s", exc)
    finally:
        try:
            await server.stop()
        except Exception:
            pass
        await gpu_alloc.release(slot)
        await port_alloc.release(port)

    plan.incumbent_fitness = float(incumbent_fitness or 0.0)
    plan.e2e_fitness = fitness

    if error:
        plan.status = "e2e_rejected"
        plan.e2e_verified = False
        plan.blockers.append(f"e2e_error:{error}")
        return plan

    gate = MeasurementGate()
    decision = gate.decide(
        incumbent_fitness=plan.incumbent_fitness,
        candidate_fitness=fitness,
        label=f"stage4_e2e:{plan.op_type}",
        metadata={"kernel_path": plan.kernel_path, "shadow": True},
    )
    speedup = (
        fitness / plan.incumbent_fitness
        if plan.incumbent_fitness > 0
        else 0.0
    )
    keep_eval = evaluate_keep(
        [speedup],
        incumbent_mean_speedup=1.0,
        min_speedup=1.01,
    )
    plan.keep_eval = {
        "gate": decision.to_dict(),
        "snr_keep": keep_eval.to_dict(),
    }

    if decision.keep and keep_eval.keep:
        plan.status = "integrated"
        plan.e2e_verified = True
        plan.blockers = []
        log.info(
            "Stage 4 E2E KEEP: op=%s fitness %.4f → %.4f",
            plan.op_type, plan.incumbent_fitness, fitness,
        )
    else:
        plan.status = "e2e_rejected"
        plan.e2e_verified = False
        plan.blockers.append(
            f"e2e_no_gain:gate={decision.reason},keep={keep_eval.reason}"
        )
        log.info(
            "Stage 4 E2E REVERT: op=%s fitness %.4f → %.4f (%s)",
            plan.op_type, plan.incumbent_fitness, fitness, decision.reason,
        )
    return plan


async def e2e_rebench_placeholder(
    *,
    plan: IntegrationPlan,
    session_id: str,
) -> IntegrationPlan:
    """
    Materialize shadow package without Docker (tests / offline).

    Prefer ``run_e2e_rebench`` from the controller when GPU resources exist.
    """
    if plan.status == "blocked":
        return plan
    try:
        spec = materialize_shadow_package(
            kernel_path=plan.kernel_path,
            op_type=plan.op_type,
            session_id=session_id,
        )
        plan.shadow_env = spec.docker_env()
        plan.status = "e2e_pending"
        plan.blockers = [
            b for b in plan.blockers
            if "e2e_rebench_not_implemented" not in b
        ]
        plan.blockers.append("awaiting_docker_e2e_rebench")
    except Exception as exc:
        plan.blockers.append(f"shadow_materialize_failed:{exc}")
        plan.status = "blocked"
    return plan
