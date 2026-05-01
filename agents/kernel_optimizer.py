"""
agents/kernel_optimizer.py
--------------------------
Kernel Optimizer Agent — Stage 2.

Starting from the winning Stage 1 vLLM config, the Kernel Optimizer
iteratively explores low-level kernel and runtime parameters (attention
backend, NCCL/RCCL flags, AMD AITER flags, MoE all2all backend, DBO,
quantization kernels) to squeeze out additional throughput.

Algorithm (LLM-guided iterative search):
  1. Benchmark the Stage 1 winner as the baseline.
  2. Call the LLM to propose the next kernel config to try.
  3. Benchmark the proposed config on the same GPU/model.
  4. Record the result in MongoDB (kernel_runs collection).
  5. Repeat for ``max_iterations`` rounds.
  6. Return the best kernel config found.

Usage
-----
    ko = KernelOptimizerAgent(
        do_client=client,
        db=db,
        gpu_alloc=alloc,
        port_alloc=pool,
        model_id="deepseek-ai/DeepSeek-V3.2",
        gpu_type="H100",
    )
    best = await ko.run(
        session_id="...",
        baseline_flags=winner_flags,
        context_configs=[(1024, 1024)],
        max_iterations=10,
    )
"""

from __future__ import annotations

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
from core.vllm_server import VLLMServer
from core.benchmark_runner import BenchmarkEngine
from core.metrics_collector import MetricsCollector

log = logging.getLogger("agents.kernel_optimizer")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_KERNEL_SS_YAML = _REPO_ROOT / "configs" / "kernel_search_space.yaml"

_AMD_GPU_TYPES = {"MI300X", "MI325X", "MI350X"}

_PROPOSE_SYSTEM_PROMPT = """\
You are an expert vLLM kernel tuning engineer.
You will receive:
  - The baseline vLLM flag configuration (Stage 1 winner).
  - The Stage 2 kernel search space definition.
  - The history of kernel experiments so far (config → fitness score).
  - The GPU vendor and model type.

Your task: propose the SINGLE most promising kernel configuration to try next.

Rules:
  - Only suggest parameters from the search space.
  - Do not repeat configurations already in the history.
  - For AMD GPUs, only suggest AMD-specific flags.
  - For NVIDIA GPUs, only suggest NVIDIA-specific flags.
  - Vendor-agnostic flags are always in scope.
  - Return a JSON object with two keys:
      "kernel_config": { <param_name>: <value>, ... }
      "rationale": "<one sentence>"

Example:
{
  "kernel_config": {
    "attention_backend": "FLASHINFER",
    "enable_prefix_caching": true
  },
  "rationale": "FLASHINFER has shown throughput gains on MoE models with TP>4."
}
"""


class KernelOptimizerAgent:
    """
    LLM-guided Stage 2 kernel search.

    Parameters
    ----------
    do_client : DOClient
    db : Database
    gpu_alloc : GPUSlotAllocator
    port_alloc : PortAllocator
    model_id : str
    gpu_type : str
    concurrency_levels : list of int
    num_prompts : int
    startup_timeout_sec : int
    node_host : str
    """

    def __init__(
        self,
        *,
        do_client: DOClient,
        db: Database,
        gpu_alloc: GPUSlotAllocator,
        port_alloc: PortAllocator,
        model_id: str = "deepseek-ai/DeepSeek-V3.2",
        gpu_type: str = "H100",
        concurrency_levels: Optional[List[int]] = None,
        num_prompts: int = 200,
        startup_timeout_sec: int = 1200,
        node_host: str = "localhost",
        docker_image: str = "",
    ) -> None:
        self._client = do_client
        self._db = db
        self._gpu_alloc = gpu_alloc
        self._port_alloc = port_alloc
        self._model_id = model_id
        self._gpu_type = gpu_type
        self._concurrency_levels = concurrency_levels or [1, 2, 4, 8, 16, 32, 64]
        self._num_prompts = num_prompts
        self._startup_timeout_sec = startup_timeout_sec
        self._node_host = node_host
        self._docker_image = docker_image
        self._vendor = "amd" if gpu_type in _AMD_GPU_TYPES else "nvidia"
        self._search_space = self._load_search_space()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        *,
        session_id: str,
        baseline_flags: Dict[str, Any],
        context_configs: List[Tuple[int, int]],
        max_iterations: int = 10,
    ) -> Dict[str, Any]:
        """
        Run Stage 2 iterative kernel search.

        Returns the best kernel_config dict found (to merge with baseline_flags).
        """
        log.info(
            "KernelOptimizer starting: session=%s iterations=%d",
            session_id, max_iterations,
        )

        # 1. Benchmark baseline to establish reference fitness
        baseline_fitness = await self._benchmark_kernel_config(
            session_id=session_id,
            iteration=0,
            baseline_flags=baseline_flags,
            kernel_override={},
            context_configs=context_configs,
            llm_reasoning="Baseline (Stage 1 winner)",
        )
        log.info("Baseline fitness: %.4f", baseline_fitness)

        history: List[Dict[str, Any]] = [
            {"kernel_config": {}, "fitness_score": baseline_fitness, "iteration": 0}
        ]
        best_config: Dict[str, Any] = {}
        best_fitness = baseline_fitness

        for iteration in range(1, max_iterations + 1):
            # 2. Ask LLM to propose the next kernel config
            proposal = await self._propose_next(
                baseline_flags=baseline_flags,
                history=history,
            )
            if proposal is None:
                log.info("LLM could not propose a new config — stopping early")
                break

            kernel_cfg = proposal.get("kernel_config", {})
            rationale = proposal.get("rationale", "")
            log.info("Iteration %d: %s — %s", iteration, kernel_cfg, rationale)

            # 3. Benchmark the proposed config
            fitness = await self._benchmark_kernel_config(
                session_id=session_id,
                iteration=iteration,
                baseline_flags=baseline_flags,
                kernel_override=kernel_cfg,
                context_configs=context_configs,
                llm_reasoning=rationale,
            )
            history.append({
                "kernel_config": kernel_cfg,
                "fitness_score": fitness,
                "iteration": iteration,
            })

            if fitness > best_fitness:
                best_fitness = fitness
                best_config = kernel_cfg
                log.info(
                    "New best kernel config (iteration %d): fitness=%.4f %s",
                    iteration, fitness, kernel_cfg,
                )

        log.info(
            "KernelOptimizer done: best_fitness=%.4f improvement=%.4f",
            best_fitness, best_fitness - baseline_fitness,
        )
        return best_config

    # ------------------------------------------------------------------
    # LLM proposal
    # ------------------------------------------------------------------

    async def _propose_next(
        self,
        baseline_flags: Dict[str, Any],
        history: List[Dict],
    ) -> Optional[Dict[str, Any]]:
        """Ask the LLM to propose the next kernel config. Returns None on failure."""
        vendor_filtered_ss = {
            name: params
            for name, params in self._search_space.items()
            if params.get("vendor", "all") in ("all", self._vendor)
        }

        user_msg = (
            f"GPU vendor: {self._vendor} ({self._gpu_type})\n"
            f"Model: {self._model_id}\n\n"
            f"Baseline vLLM flags:\n{json.dumps(baseline_flags, indent=2)}\n\n"
            f"Kernel search space:\n{json.dumps(vendor_filtered_ss, indent=2)}\n\n"
            f"History ({len(history)} experiments):\n{json.dumps(history, indent=2)}\n\n"
            "Propose the next kernel config:"
        )

        try:
            result = await self._client.chat_json(
                messages=[{"role": "user", "content": user_msg}],
                system=_PROPOSE_SYSTEM_PROMPT,
            )
            if not isinstance(result, dict) or "kernel_config" not in result:
                log.warning("LLM proposal malformed: %s", result)
                return None
            return result
        except DOClientError as exc:
            log.warning("LLM proposal failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Benchmark a kernel config
    # ------------------------------------------------------------------

    async def _benchmark_kernel_config(
        self,
        *,
        session_id: str,
        iteration: int,
        baseline_flags: Dict[str, Any],
        kernel_override: Dict[str, Any],
        context_configs: List[Tuple[int, int]],
        llm_reasoning: str,
    ) -> float:
        """
        Start vLLM with baseline_flags + kernel_override, benchmark it,
        write a kernel_run document, and return the fitness score.
        """
        merged_flags = {**baseline_flags, **kernel_override}

        # Separate env vars from vLLM CLI flags
        env_overrides: Dict[str, str] = {}
        vllm_flag_overrides: Dict[str, Any] = {}
        for name, val in kernel_override.items():
            param_def = self._search_space.get(name, {})
            if "env_var" in param_def:
                env_overrides[param_def["env_var"]] = str(val).lower() if isinstance(val, bool) else str(val)
            elif "vllm_flag" in param_def:
                vllm_flag_overrides[name] = val

        # Build VLLMFlags from baseline, patching CLI-flag overrides
        flags_fields = {
            k: v for k, v in baseline_flags.items()
            if k in VLLMFlags.__dataclass_fields__
        }
        flags_fields.update({
            k: v for k, v in vllm_flag_overrides.items()
            if k in VLLMFlags.__dataclass_fields__
        })
        flags = VLLMFlags(**flags_fields)
        tp_size = flags.tensor_parallel_size or 1

        slot = await self._gpu_alloc.acquire(tp_size)
        if slot is None:
            log.warning("KernelOptimizer: no GPU slots for iteration %d", iteration)
            return 0.0

        port = await self._port_alloc.acquire()
        if port is None:
            await self._gpu_alloc.release(slot)
            log.warning("KernelOptimizer: no ports for iteration %d", iteration)
            return 0.0

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
        error: Optional[str] = None

        try:
            await server.start()

            # Benchmark on the first context config only (speed vs thoroughness)
            input_len, output_len = context_configs[0]
            engine = BenchmarkEngine(
                base_url=f"http://localhost:{port}",
                model_id=self._model_id,
                concurrency_levels=self._concurrency_levels,
                num_prompts=self._num_prompts,
                input_len=input_len,
                output_len=output_len,
            )
            ramp = await engine.run()
            collector = MetricsCollector(
                ramp_result=ramp,
                gpu_type=self._gpu_type,
                primary_metric="throughput",
            )
            enriched = collector.collect()
            fitness = enriched.fitness_score

        except Exception as exc:
            error = str(exc)
            log.warning("KernelOptimizer iteration %d error: %s", iteration, exc)

        finally:
            try:
                await server.stop()
            except Exception:
                pass
            await self._gpu_alloc.release(slot)
            await self._port_alloc.release(port)

        # Write to MongoDB
        await self._db.insert_kernel_run(
            session_id=session_id,
            iteration=iteration,
            kernel_config=kernel_override,
            raw_metrics={"fitness_score": fitness},
            fitness_score=fitness,
            llm_reasoning=llm_reasoning,
            error=error,
        )

        return fitness

    # ------------------------------------------------------------------
    # Search space loader
    # ------------------------------------------------------------------

    def _load_search_space(self) -> Dict[str, Any]:
        with open(_KERNEL_SS_YAML, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
