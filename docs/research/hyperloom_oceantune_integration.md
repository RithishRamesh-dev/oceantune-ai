# Hyperloom → OceanTune Integration Study

**Date:** 2026-09-29 (updated)  
**Sources:** Hyperloom-main (AMD AGI, MIT), OceanTune codebase  
**License note:** Hyperloom is MIT. OceanTune reimplements *techniques*; it does not vendor Hyperloom source.

---

## Critical framing

Hyperloom is a **design reference only**. OceanTune never imports or calls Hyperloom at runtime.

| Dimension | Hyperloom | OceanTune |
|-----------|-----------|-----------|
| Primary GPU | AMD Instinct + ROCm | NVIDIA + AMD |
| Frameworks | vLLM, SGLang, xDiT | **vLLM + SGLang** (`FrameworkBackend`) |
| Loop | PRELUDE→ENABLEMENT→FRAMEWORK→KERNEL→SWEEP→CLOSE + macro | **Enablement → Prelude → S1 → (S2↔S3)×macro → Sweep → S4/campaign → CLOSE** |
| Keep decision | Coordinator + Critic | **MeasurementGate + CriticAgent + paired/convergence guards** |
| Learning | Recipe KB + experience | **Recipe KB + experience constraints + attempt ledger** |
| Kernel path | Fusion + shadow + SNR + E2E | **Fusion patterns + shadow E2E + SNR firewall + campaign/ledger** |

---

## Pipeline (current)

```
Enablement → Prelude (warm-replay) → Stage1 → (Stage2 → Stage3)×macro
  → Sweep → Stage4 → Campaign/Ledger → CLOSE (+ checkpoint)
```

Config knobs: `framework`, `enablement_*`, `prelude_*`, `macro_cycle_*`, `session_max_minutes`,
`resume_session_id`, `stage4_*`, `quantization_scheme`, `serving_patches_enabled`,
`optimiser.primary_metric` (`throughput|prefill_heavy|decode_heavy|cost_aware`).

---

## Gap matrix (portable status)

| Capability | Status |
|------------|--------|
| Recipe KB + cascade | **Done** |
| MeasurementGate + Critic | **Done** |
| Enablement ladder | **Done** |
| Prelude warm-replay + reject seed | **Done** (`core/prelude.py`) |
| Macro-cycle + wall-clock budget | **Done** |
| Operating-point sweep | **Done** |
| Experience constraints + attempt ledger | **Done** |
| Fusion patterns + SNR + numerical pipeline | **Done** (firewall SNR gate) |
| Shadow E2E + serving patch manifests | **Done** |
| PolicyGate (all stages + prelude) | **Done** |
| FrameworkBackend vLLM \| SGLang | **Done** (Executor launch override) |
| Draft registry + fitness modes | **Done** |
| Search EXPLOIT/DIVERSIFY | **Done** |
| Session resume + checkpoint | **Done** |
| Quantization scheme registry | **Done** |
| Convergence + paired A/B helpers | **Done** |
| TraceLens / Magpie / GEAK / FlyDSL / xDiT | **External — not ported** |

See [`oceantune_capability_inventory.md`](./oceantune_capability_inventory.md).

---

## Best-in-class modules

- `core/prelude.py`, `core/warmstart_policy.py`, `core/search_policy.py`
- `core/measurement_convergence.py`, `core/paired_bench.py`
- `core/session_checkpoint.py`, `core/attempt_ledger.py`
- `core/serving_patches.py`, `data/serving_patches/`
- `core/quantization_schemes.py`
- `core/enablement.py`, `core/policy_gate.py`, `core/framework_backend.py`
- `core/macro_cycle.py`, `core/fusion/`, `core/snr_contract.py`, `core/numerical_pipeline.py`
- `core/kernel_integration.py`, `core/kernel_campaign.py`, `core/kernel_ledger.py`
- `agents/controller_agent.py`

---

## Beyond Hyperloom (OceanTune improvisations)

- Prefill / decode / **cost_aware** fitness modes
- NVFP4 / Blackwell scheme gating
- DigitalOcean-oriented session budgets + resume
- MoE draft pairs in `configs/draft_models.yaml`
- Short-ramp paired fidelity helpers before KEEP

---

## External (honest limits)

Magpie, TraceLens, IntelliKit, GEAK, FlyDSL, full KernelForge git forge, xDiT multimodal —
require separate AMD/internal toolchains. OceanTune absorbs *techniques* only.
