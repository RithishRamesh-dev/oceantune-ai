"""
agents/strategy_optimizer.py
-----------------------------
Strategy Optimizer Agent — Stage 2.

Explores higher-level inference algorithm strategies on top of the Stage 1
winner configuration. Covers five strategy categories:

  1. KV Cache         — fp8 dtype, prefix caching, hash algorithm
  2. Speculative Dec  — draft tokens count, draft TP size
  3. Prefill          — chunked prefill, multi-step scheduling, delay factor
  4. Attention Kernel — backend selection (FLASH_ATTN, FLASHINFER, ROCM_FLASH)
  5. MoE / AMD        — vendor-specific dispatch and fused kernels

The LLM receives the Stage 1 winner metrics, the full experiment history,
and per-category notes to guide targeted proposals. Includes a deterministic
fallback sweep when the LLM is unavailable.

Usage
-----
    so = StrategyOptimizerAgent(
        do_client=client,
        db=db,
        gpu_alloc=alloc,
        port_alloc=pool,
        model_id="Qwen/Qwen2.5-7B-Instruct",
        gpu_type="H200",
        ...
    )
    best = await so.run(
        session_id="...",
        baseline_flags=winner_flags,
        baseline_metrics=winner_metrics,
        context_configs=[(1024, 1024), (1024, 4096)],
        max_iterations=12,
    )
"""

from __future__ import annotations

import copy
import json
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents.do_client import DOClient, DOClientError
from core.db import Database
from core.gpu_allocator import GPUSlotAllocator
from core.port_allocator import PortAllocator
from core.search_space import VLLMFlags
from core.vllm_server import VLLMServer, _load_gpu_profile
from core.benchmark_runner import BenchmarkEngine
from core.metrics_collector import MetricsCollector
from core.log_analyzer import LogAnalyzer

log = logging.getLogger("agents.strategy_optimizer")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STAGE2_SS_YAML = _REPO_ROOT / "configs" / "stage2_search_space.yaml"

_AMD_GPU_TYPES = {"MI300X", "MI325X", "MI350X"}

_PROPOSE_SYSTEM_PROMPT = """\
You are an expert vLLM inference optimization engineer running Stage 2 of OceanTune.

You have the Stage 1 winner (best vLLM flag configuration). Your job is to explore
higher-level inference strategies that can improve performance further:

  - KV Cache strategies (fp8 KV dtype, prefix caching)
  - Speculative decoding (if a suitable draft model exists)
  - Prefill strategies (chunked prefill, multi-step scheduling)
  - Attention kernel selection (FLASH_ATTN vs FLASHINFER vs ROCM_FLASH)
  - MoE/AMD vendor-specific kernel flags

You will receive:
  - Stage 1 winner flags and metrics (throughput, latency, concurrency curve)
  - The strategy search space with parameter descriptions
  - Full experiment history with what was tried and the result

Your task: propose the SINGLE most impactful strategy change not yet tried.

Output format (strict JSON):
{
  "strategy_config": { "<param_name>": <value>, ... },
  "category": "<kv_cache|speculative_decoding|prefill|kernel|moe|amd_kernel|communication>",
  "rationale": "<2-3 sentences: what bottleneck you're targeting and expected mechanism>"
}

Strategy selection guidelines:
1. If best_concurrency == max_concurrency_tested: the GPU can handle more — try
   increasing max_num_batched_tokens or enabling multi-step scheduling.
2. If p95_latency > 5000ms and throughput is high: the queue is saturated — try
   chunked prefill to reduce head-of-line blocking.
3. If gpu_memory_utilization > 0.90: memory is the bottleneck — try fp8 KV cache
   to free VRAM for more sequences.
4. If enable_prefix_caching is false: test it — it's lossless for chat workloads.
5. If attention_backend is FLASH_ATTN and model has GQA: try FLASHINFER which
   has native paged-attention GQA support (often 5-15% faster).
6. If on AMD GPU: verify all AITER kernels are enabled — each is a free win.
7. Do NOT propose speculative_model unless you know a compatible draft model exists.

Do not include any text outside the JSON object.
"""

_CATEGORY_SWEEP_PROMPT = """\
You are an expert vLLM inference optimization engineer running Stage 2 of OceanTune.

Your task: propose ONE representative strategy from EACH of the following categories.
This enables a broad initial sweep across all strategy dimensions in a single LLM call,
before follow-up per-iteration proposals refine the best direction.

Categories to cover (propose one per category):
  1. kv_cache        — fp8 KV dtype, prefix caching, or combination
  2. prefill         — chunked prefill, multi-step scheduling, or scheduler_delay_factor
  3. kernel          — attention backend (FLASH_ATTN vs FLASHINFER vs ROCM_FLASH)
  4. moe             — MoE/AMD vendor-specific dispatch (skip if not applicable: output empty {})
  5. batching        — max_num_batched_tokens, max_num_seqs, or num_scheduler_steps

For each category, choose the configuration most likely to improve performance
given the Stage 1 metrics. Skip categories where no meaningful change is possible
(output empty strategy_config {} for that category).

Output format (strict JSON):
{
  "sweep": [
    {"category": "kv_cache",  "strategy_config": {...}, "rationale": "..."},
    {"category": "prefill",   "strategy_config": {...}, "rationale": "..."},
    {"category": "kernel",    "strategy_config": {...}, "rationale": "..."},
    {"category": "moe",       "strategy_config": {},    "rationale": "Not applicable"},
    {"category": "batching",  "strategy_config": {...}, "rationale": "..."}
  ]
}

Do not include any text outside the JSON object.
"""


class StrategyOptimizerAgent:
    """
    LLM-guided Stage 2: inference strategy search.

    Parameters mirror KernelOptimizerAgent for drop-in replacement.
    """

    # Deterministic fallback sweep when LLM is unavailable.
    # Ordered by expected impact. Skipped if already tried.
    _FALLBACK_SWEEP: List[Dict[str, Any]] = [
        {"kv_cache_dtype": "fp8"},
        {"enable_prefix_caching": True},
        {"attention_backend": "FLASHINFER"},
        {"enable_chunked_prefill": True, "max_num_chunked_tokens": 2048},
        {"num_scheduler_steps": 8},
        {"scheduler_delay_factor": 0.2},
        {"kv_cache_dtype": "fp8", "enable_prefix_caching": True},
        {"attention_backend": "FLASH_ATTN"},
        {"num_scheduler_steps": 4},
        {"enable_chunked_prefill": False, "max_num_chunked_tokens": 4096},
    ]

    def __init__(
        self,
        *,
        do_client: DOClient,
        db: Database,
        gpu_alloc: GPUSlotAllocator,
        port_alloc: PortAllocator,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        gpu_type: str = "H200",
        concurrency_levels: Optional[List[int]] = None,
        num_prompts: int = 30,
        startup_timeout_sec: int = 1200,
        node_host: str = "localhost",
        docker_image: str = "",
        primary_metric: str = "throughput",
    ) -> None:
        self._client = do_client
        self._db = db
        self._gpu_alloc = gpu_alloc
        self._port_alloc = port_alloc
        self._model_id = model_id
        self._gpu_type = gpu_type
        self._concurrency_levels = concurrency_levels or [1, 2, 4, 8, 16, 32, 64, 128]
        self._num_prompts = num_prompts
        self._startup_timeout_sec = startup_timeout_sec
        self._node_host = node_host
        self._docker_image = docker_image
        self._primary_metric = primary_metric
        self._vendor = "amd" if gpu_type in _AMD_GPU_TYPES else "nvidia"
        self._search_space = self._load_search_space()
        # Pre-fetched category sweep proposals (consumed before per-iteration LLM calls)
        self._batch_queue: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        *,
        session_id: str,
        baseline_flags: Dict[str, Any],
        baseline_metrics: Dict[str, Any],
        context_configs: List[Tuple[int, int]],
        max_iterations: int = 12,
    ) -> Dict[str, Any]:
        """
        Run Stage 2 strategy search.

        Returns the best strategy_config dict (to merge with baseline_flags).
        Always records all experiments in MongoDB for visibility, regardless of
        whether any improve on the baseline.
        """
        log.info(
            "StrategyOptimizer starting: session=%s iterations=%d",
            session_id, max_iterations,
        )

        # 1. Benchmark baseline (Stage 1 winner, no overrides)
        baseline_fitness, baseline_em = await self._benchmark_strategy(
            session_id=session_id,
            iteration=0,
            baseline_flags=baseline_flags,
            strategy_override={},
            context_configs=context_configs,
            llm_reasoning="Baseline (Stage 1 winner — no strategy overrides)",
            category="baseline",
        )
        log.info("Stage 2 baseline fitness: %.4f", baseline_fitness)

        history: List[Dict[str, Any]] = [{
            "iteration": 0,
            "strategy_config": {},
            "category": "baseline",
            "fitness_score": baseline_fitness,
            "metrics": baseline_em,
        }]
        best_config: Dict[str, Any] = {}
        best_fitness = baseline_fitness
        best_category = "baseline"
        fallback_idx = 0

        # 2. Category sweep: one proposal per strategy dimension in a single LLM call.
        # This seeds the queue so the first N iterations cover all strategy categories
        # before follow-up per-iteration proposals refine the best direction.
        await self._do_category_sweep(
            baseline_flags=baseline_flags,
            baseline_metrics=baseline_metrics,
        )
        log.info(
            "Stage 2 category sweep seeded %d proposals into queue",
            len(self._batch_queue),
        )

        for iteration in range(1, max_iterations + 1):
            # 3. Propose next strategy — consume sweep queue first, then LLM per-iteration
            proposal: Optional[Dict[str, Any]] = None

            if self._batch_queue:
                proposal = self._batch_queue.pop(0)
                log.info(
                    "Stage 2 iteration %d [sweep/%s]: %s — %s",
                    iteration,
                    proposal.get("category", "?"),
                    proposal.get("strategy_config", {}),
                    proposal.get("rationale", "")[:100],
                )
            else:
                proposal = await self._propose_next(
                    baseline_flags=baseline_flags,
                    baseline_metrics=baseline_metrics,
                    history=history,
                )

            if proposal is None:
                # LLM unavailable: use deterministic fallback sweep
                proposal = self._fallback_proposal(history, fallback_idx)
                fallback_idx += 1
                if proposal is None:
                    log.info("Stage 2 fallback sweep exhausted — stopping")
                    break
                log.info("Stage 2 fallback proposal: %s", proposal.get("strategy_config"))
            elif not self._batch_queue:
                log.info(
                    "Stage 2 iteration %d [%s]: %s — %s",
                    iteration,
                    proposal.get("category", "?"),
                    proposal.get("strategy_config", {}),
                    proposal.get("rationale", "")[:100],
                )

            strategy_cfg = proposal.get("strategy_config", {})
            category = proposal.get("category", "unknown")
            rationale = proposal.get("rationale", "")

            # 3. Benchmark the proposed strategy
            fitness, em = await self._benchmark_strategy(
                session_id=session_id,
                iteration=iteration,
                baseline_flags=baseline_flags,
                strategy_override=strategy_cfg,
                context_configs=context_configs,
                llm_reasoning=rationale,
                category=category,
            )

            delta = fitness - baseline_fitness
            log.info(
                "Stage 2 iteration %d: fitness=%.4f delta=%+.4f [%s]",
                iteration, fitness, delta, category,
            )

            history.append({
                "iteration": iteration,
                "strategy_config": strategy_cfg,
                "category": category,
                "fitness_score": fitness,
                "delta_vs_baseline": delta,
                "metrics": em,
            })

            if fitness > best_fitness:
                best_fitness = fitness
                best_config = strategy_cfg
                best_category = category
                log.info(
                    "New best strategy (iteration %d, %s): fitness=%.4f (+%.4f)",
                    iteration, category, fitness, delta,
                )

        improvement = best_fitness - baseline_fitness
        log.info(
            "Stage 2 done: best_fitness=%.4f improvement=%+.4f category=%s config=%s",
            best_fitness, improvement, best_category, best_config,
        )
        return best_config, best_fitness

    # ------------------------------------------------------------------
    # LLM proposal
    # ------------------------------------------------------------------

    async def _do_category_sweep(
        self,
        baseline_flags: Dict[str, Any],
        baseline_metrics: Dict[str, Any],
    ) -> None:
        """
        Single LLM call asking for one proposal per strategy category.
        Populates self._batch_queue. Silently skips on LLM failure.
        """
        vendor_ss = {
            name: params
            for name, params in self._search_space.items()
            if params.get("vendor", "all") in ("all", self._vendor)
        }
        user_msg = (
            f"Model: {self._model_id}\n"
            f"GPU: {self._gpu_type} (vendor={self._vendor})\n\n"
            f"Stage 1 winner flags:\n{json.dumps(baseline_flags, indent=2)}\n\n"
            f"Stage 1 winner metrics:\n{json.dumps(baseline_metrics, indent=2)}\n\n"
            f"Strategy search space:\n{json.dumps(vendor_ss, indent=2)}\n\n"
            "Return the category sweep JSON now."
        )
        try:
            result = await self._client.chat_json(
                messages=[{"role": "user", "content": user_msg}],
                system=_CATEGORY_SWEEP_PROMPT,
            )
            if not isinstance(result, dict) or "sweep" not in result:
                log.warning("Stage 2 category sweep: unexpected format %s", result)
                return
            for item in result["sweep"]:
                cfg = item.get("strategy_config", {})
                if not cfg:
                    continue  # skip empty / non-applicable categories
                self._batch_queue.append({
                    "strategy_config": cfg,
                    "category": item.get("category", "unknown"),
                    "rationale": item.get("rationale", "Category sweep proposal"),
                })
            log.info(
                "Stage 2 category sweep: %d non-empty proposals queued",
                len(self._batch_queue),
            )
        except DOClientError as exc:
            log.warning("Stage 2 category sweep failed (%s); will use per-iteration proposals", exc)

    async def _propose_next(
        self,
        baseline_flags: Dict[str, Any],
        baseline_metrics: Dict[str, Any],
        history: List[Dict],
    ) -> Optional[Dict[str, Any]]:
        """Ask the LLM to propose the next strategy. Returns None on failure."""
        # Filter search space to vendor-appropriate params only
        vendor_ss = {
            name: params
            for name, params in self._search_space.items()
            if params.get("vendor", "all") in ("all", self._vendor)
        }

        # Already-tried strategy configs (to avoid repeats)
        tried = [h["strategy_config"] for h in history if h.get("strategy_config")]

        # Compact history for the prompt
        history_summary = [
            {
                "iteration": h["iteration"],
                "category": h.get("category"),
                "strategy_config": h.get("strategy_config", {}),
                "fitness": h.get("fitness_score"),
                "delta_vs_baseline": h.get("delta_vs_baseline"),
                "peak_tok_s": (h.get("metrics") or {}).get("peak_throughput_tokens_per_sec"),
                "p95_ms": (h.get("metrics") or {}).get("p95_latency_at_peak_ms"),
            }
            for h in history
        ]

        user_msg = (
            f"Model: {self._model_id}\n"
            f"GPU: {self._gpu_type} (vendor={self._vendor})\n\n"
            f"Stage 1 winner flags:\n{json.dumps(baseline_flags, indent=2)}\n\n"
            f"Stage 1 winner metrics:\n{json.dumps(baseline_metrics, indent=2)}\n\n"
            f"Strategy search space:\n{json.dumps(vendor_ss, indent=2)}\n\n"
            f"Already tried ({len(tried)}):\n{json.dumps(tried, indent=2)}\n\n"
            f"Experiment history ({len(history_summary)} runs):\n"
            f"{json.dumps(history_summary, indent=2)}\n\n"
            "Propose the next strategy to try."
        )

        try:
            result = await self._client.chat_json(
                messages=[{"role": "user", "content": user_msg}],
                system=_PROPOSE_SYSTEM_PROMPT,
            )
            if not isinstance(result, dict) or "strategy_config" not in result:
                log.warning("Stage 2 LLM response malformed: %s", result)
                return None
            return result
        except DOClientError as exc:
            log.warning("Stage 2 LLM proposal failed: %s", exc)
            return None

    def _fallback_proposal(
        self,
        history: List[Dict],
        idx: int,
    ) -> Optional[Dict[str, Any]]:
        """Return the next untried fallback variation, or None if exhausted."""
        tried_configs = [
            json.dumps(h.get("strategy_config", {}), sort_keys=True)
            for h in history
        ]
        checked = 0
        pos = idx
        while checked < len(self._FALLBACK_SWEEP):
            candidate = self._FALLBACK_SWEEP[pos % len(self._FALLBACK_SWEEP)]
            candidate_key = json.dumps(candidate, sort_keys=True)
            if candidate_key not in tried_configs:
                return {
                    "strategy_config": candidate,
                    "category": "fallback",
                    "rationale": f"Fallback sweep #{pos + 1}: {candidate}",
                }
            pos += 1
            checked += 1
        return None

    # ------------------------------------------------------------------
    # Benchmark a strategy config
    # ------------------------------------------------------------------

    async def _benchmark_strategy(
        self,
        *,
        session_id: str,
        iteration: int,
        baseline_flags: Dict[str, Any],
        strategy_override: Dict[str, Any],
        context_configs: List[Tuple[int, int]],
        llm_reasoning: str,
        category: str,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Start vLLM with baseline_flags + strategy_override, benchmark it,
        write a kernel_run document, return (fitness_score, enriched_metrics).
        Always returns a result — never raises.
        """
        # Separate env-var overrides from vLLM CLI flag overrides
        env_overrides: Dict[str, str] = {}
        vllm_overrides: Dict[str, Any] = {}
        for name, val in strategy_override.items():
            param_def = self._search_space.get(name, {})
            if "env_var" in param_def:
                env_overrides[param_def["env_var"]] = (
                    str(val).lower() if isinstance(val, bool) else str(val)
                )
            else:
                vllm_overrides[name] = val

        # Build merged VLLMFlags: baseline + CLI overrides
        merged_dict = {**baseline_flags, **vllm_overrides}
        known_fields = set(VLLMFlags.__dataclass_fields__.keys())
        flags = VLLMFlags(**{k: v for k, v in merged_dict.items() if k in known_fields})
        tp_size = flags.tensor_parallel_size or 1

        slot = await self._gpu_alloc.acquire(tp_size)
        if slot is None:
            log.warning("Stage 2 iteration %d: no GPU slot available", iteration)
            await self._db.insert_kernel_run(
                session_id=session_id, iteration=iteration,
                kernel_config={**strategy_override, "_category": category},
                raw_metrics={}, fitness_score=0.0,
                llm_reasoning=llm_reasoning,
                error="No GPU slot available",
            )
            return 0.0, {}

        port = await self._port_alloc.acquire()
        if port is None:
            await self._gpu_alloc.release(slot)
            log.warning("Stage 2 iteration %d: no port available", iteration)
            await self._db.insert_kernel_run(
                session_id=session_id, iteration=iteration,
                kernel_config={**strategy_override, "_category": category},
                raw_metrics={}, fitness_score=0.0,
                llm_reasoning=llm_reasoning,
                error="No port available",
            )
            return 0.0, {}

        device_env = self._gpu_alloc.build_device_env(slot)
        combined_env = {**device_env, **env_overrides}

        server = VLLMServer(
            model_id=self._model_id,
            flags=flags,
            gpu_type=self._gpu_type,
            port=port,
            startup_timeout=self._startup_timeout_sec,
            extra_env=combined_env,
            docker_image=self._docker_image,
        )

        fitness = 0.0
        em_dict: Dict[str, Any] = {}
        error: Optional[str] = None

        try:
            await server.start()

            # Benchmark all context configs (not just the first)
            best_fitness_this_run = 0.0
            best_em_this_run: Dict[str, Any] = {}
            for input_len, output_len in context_configs:
                engine = BenchmarkEngine(
                    base_url=f"http://localhost:{port}",
                    model_id=self._model_id,
                    concurrency_levels=self._concurrency_levels,
                    num_prompts=self._num_prompts,
                    input_len=input_len,
                    output_len=output_len,
                )
                ramp = await engine.run()
                analysis = LogAnalyzer.analyze(server.log_tail)
                gpu_profile = _load_gpu_profile(self._gpu_type)
                enriched = MetricsCollector.collect(
                    ramp=ramp,
                    analysis=analysis,
                    flags=flags,
                    gpu_profile=gpu_profile,
                    primary_metric=self._primary_metric,
                )
                if enriched.fitness_score > best_fitness_this_run:
                    best_fitness_this_run = enriched.fitness_score
                    best_em_this_run = enriched.to_dict()

            fitness = best_fitness_this_run
            em_dict = best_em_this_run

        except Exception as exc:
            error = str(exc)
            log.warning("Stage 2 iteration %d error: %s", iteration, exc)

        finally:
            try:
                await server.stop()
            except Exception:
                pass
            await self._gpu_alloc.release(slot)
            await self._port_alloc.release(port)

        # Always write to MongoDB — visibility even when no improvement
        await self._db.insert_kernel_run(
            session_id=session_id,
            iteration=iteration,
            kernel_config={**strategy_override, "_category": category},
            raw_metrics={
                "fitness_score": fitness,
                "enriched_metrics": em_dict,
            },
            fitness_score=fitness,
            llm_reasoning=llm_reasoning,
            error=error,
        )

        return fitness, em_dict

    # ------------------------------------------------------------------
    # Search space loader
    # ------------------------------------------------------------------

    def _load_search_space(self) -> Dict[str, Any]:
        with open(_STAGE2_SS_YAML, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
