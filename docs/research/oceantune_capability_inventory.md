# OceanTune capability inventory & extension points

Canonical snapshot of OceanTune architecture (repo-native; Hyperloom source excluded).  
Use with [`hyperloom_oceantune_integration.md`](./hyperloom_oceantune_integration.md).

**Bottom line:** OceanTune is a **self-contained** best-in-class vLLM + SGLang optimizer. Portable Hyperloom techniques are reimplemented under `core/` and `agents/`.

---

## What OceanTune owns (implemented)

| Capability | Module |
|------------|--------|
| Enablement → Prelude → Stages 1–4 + macro | `agents/controller_agent.py`, `core/macro_cycle.py`, `core/prelude.py` |
| Recipe KB + warmstart policy | `core/recipe_kb.py`, `core/warmstart_policy.py` |
| MeasurementGate + Critic + convergence/paired | `core/measurement_gate.py`, `agents/critic_agent.py`, `core/measurement_convergence.py`, `core/paired_bench.py` |
| Session breakdown + checkpoint resume | `core/session_breakdown.py`, `core/session_checkpoint.py` |
| Experience constraints + attempt ledger | `core/experience_constraints.py`, `core/attempt_ledger.py` |
| Search EXPLOIT/DIVERSIFY | `core/search_policy.py` |
| Fusion diagnose + patterns | `core/fusion/` |
| SNR + numerical pipeline (+ firewall gate) | `core/snr_contract.py`, `core/numerical_pipeline.py`, `agents/correctness_firewall_agent.py` |
| YAML kernel harness | `core/kernel_harness.py` |
| Shadow hooks + E2E rebench | `core/kernel_integration.py` |
| Serving patch manifests | `core/serving_patches.py`, `data/serving_patches/` |
| Stage 4b fusion campaign + ledger | `core/kernel_campaign.py`, `core/kernel_ledger.py` |
| Workspace isolation | `core/workspace_policy.py` |
| Enablement / boot repair | `core/enablement.py` |
| PolicyGate (incl. prelude) | `core/policy_gate.py` |
| FrameworkBackend vLLM \| SGLang (+ Executor) | `core/framework_backend.py`, `agents/executor.py`, `core/vllm_server.py` |
| Draft model registry (incl. MoE) | `core/draft_registry.py`, `configs/draft_models.yaml` |
| Quantization schemes | `core/quantization_schemes.py` |
| Vendor knowledge packs | `knowledge/`, `core/knowledge_pack.py` |
| Fitness modes | `throughput`, `prefill_heavy`, `decode_heavy`, `cost_aware` |

---

## Pipeline order

```
Enablement → Prelude → Stage1 → (Stage2 → Stage3)×macro → Sweep → Stage4 → Campaign/Ledger → CLOSE
```

---

## External / not portable

| Item | Why not in OceanTune |
|------|----------------------|
| TraceLens / Magpie / IntelliKit | Proprietary/external profilers |
| GEAK / FlyDSL / CK rewrite forge | Separate AMD toolchains |
| Full git worktree KernelForge | Campaign + workspace policy instead |
| xDiT multimodal campaigns | Out of current serving scope |

---

*Updated after depth wave: prelude, convergence, paired bench, search policy, session resume, serving patches, quantization schemes, SGLang Executor wiring, firewall SNR.*
