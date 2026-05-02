"""
agents/executor.py
------------------
Executor Agent — runs one vLLM config, benchmarks it, and writes results
to MongoDB.

Each Executor is assigned a single candidate config by the Coordinator.
It is responsible for:
  1. Starting a vLLM server on an assigned GPU slot + port.
  2. Running the benchmark ramp (all concurrency levels × all context lengths).
  3. Using the DO Serverless Inference LLM to parse any ambiguous benchmark
     output and sanity-check the extracted metrics.
  4. Writing a benchmark_run document to MongoDB.
  5. Marking the config as done (or failed) in MongoDB.
  6. Releasing the GPU slot and port back to their allocators.

Usage
-----
    executor = ExecutorAgent(
        do_client=client,
        db=db,
        gpu_alloc=alloc,
        port_alloc=pool,
        gpu_type="H100",
        model_id="deepseek-ai/DeepSeek-V3.2",
    )
    await executor.run(
        session_id="...",
        config_doc={"_id": "...", "flags": {...}, "fingerprint": "..."},
        context_configs=[(1024, 1024), (1024, 4096)],
    )
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from agents.do_client import DOClient, DOClientError
from core.db import Database
from core.gpu_allocator import GPUSlotAllocator
from core.port_allocator import PortAllocator
from core.search_space import VLLMFlags
from core.vllm_server import VLLMServer, OOMError, StartupTimeout, PortConflict, CUDAError, _load_gpu_profile
from core.benchmark_runner import BenchmarkEngine
from core.metrics_collector import MetricsCollector
from core.log_analyzer import LogAnalyzer

log = logging.getLogger("agents.executor")

_PARSE_SYSTEM_PROMPT = """\
You are a vLLM benchmark output parser.
Given raw benchmark stdout, extract the following metrics as a JSON object:
  throughput_tok_s   : float  (tokens per second, summed across all requests)
  p50_latency_ms     : float  (p50 end-to-end latency in milliseconds)
  p95_latency_ms     : float  (p95 end-to-end latency in milliseconds)
  p99_latency_ms     : float  (p99 end-to-end latency in milliseconds)
  ttft_ms            : float  (time-to-first-token in milliseconds, median)
  tpot_ms            : float  (time-per-output-token in milliseconds, median)
  error_rate         : float  (fraction of failed requests, 0.0–1.0)
  concurrency        : int    (concurrency level this measurement was taken at)

Return ONLY the JSON object. If a metric cannot be found, use null.
"""


class ExecutorAgent:
    """
    Single-config benchmarking agent.

    Parameters
    ----------
    do_client : DOClient
        Shared DO Serverless Inference client.
    db : Database
        Shared MongoDB client.
    gpu_alloc : GPUSlotAllocator
        Node-level GPU slot allocator.
    port_alloc : PortAllocator
        Node-level port allocator.
    gpu_type : str
        GPU SKU for this node (e.g. "H100").
    model_id : str
        Hugging Face model ID to serve.
    concurrency_levels : list of int
        Concurrency ramp for benchmarking.
    num_prompts : int
        Total prompts per concurrency level.
    startup_timeout_sec : int
        Max seconds to wait for vLLM to become healthy.
    node_host : str
        Hostname of this node (recorded in the benchmark_run document).
    primary_metric : str
        Fitness metric: throughput | p95_latency | ttft | tpot.
        Passed to MetricsCollector — comes from optimiser.primary_metric in config.
    """

    def __init__(
        self,
        *,
        do_client: DOClient,
        db: Database,
        gpu_alloc: GPUSlotAllocator,
        port_alloc: PortAllocator,
        gpu_type: str = "H100",
        model_id: str = "deepseek-ai/DeepSeek-V3.2",
        concurrency_levels: Optional[List[int]] = None,
        num_prompts: int = 200,
        startup_timeout_sec: int = 1200,
        node_host: str = "localhost",
        primary_metric: str = "throughput",
        docker_image: str = "",
    ) -> None:
        self._client = do_client
        self._db = db
        self._gpu_alloc = gpu_alloc
        self._port_alloc = port_alloc
        self._gpu_type = gpu_type
        self._model_id = model_id
        self._concurrency_levels = concurrency_levels or [1, 2, 4, 8, 16, 32, 64]
        self._num_prompts = num_prompts
        self._startup_timeout_sec = startup_timeout_sec
        self._node_host = node_host
        self._primary_metric = primary_metric
        self._docker_image = docker_image

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        *,
        session_id: str,
        config_doc: Dict[str, Any],
        context_configs: List[Tuple[int, int]],
    ) -> None:
        """
        Benchmark one config across all context lengths and write results.

        On any unrecoverable error, marks the config as failed in MongoDB
        and releases resources before returning.
        """
        config_id = str(config_doc["_id"])
        fingerprint = config_doc["fingerprint"]
        flags_dict = config_doc["flags"]

        # Reconstruct VLLMFlags from stored dict
        flags = VLLMFlags(**{
            k: v for k, v in flags_dict.items()
            if hasattr(VLLMFlags, k) or k in VLLMFlags.__dataclass_fields__
        })
        tp_size = flags.tensor_parallel_size or 1

        # Acquire GPU slot and port
        slot = await self._gpu_alloc.acquire(tp_size)
        if slot is None:
            log.warning("Executor: no GPU slots available for tp=%d, re-queuing", tp_size)
            await self._db.requeue_config(config_id)
            return

        port = await self._port_alloc.acquire()
        if port is None:
            log.warning("Executor: no ports available, releasing GPU slot")
            await self._gpu_alloc.release(slot)
            await self._db.requeue_config(config_id)
            return

        device_env = self._gpu_alloc.build_device_env(slot)
        log.info(
            "Executor starting: config=%s port=%d slot=%s tp=%d",
            fingerprint[:8], port, slot, tp_size,
        )

        server = VLLMServer(
            model_id=self._model_id,
            flags=flags,
            gpu_type=self._gpu_type,
            port=port,
            startup_timeout=self._startup_timeout_sec,
            extra_env=device_env,
            docker_image=self._docker_image,
        )

        best_fitness = 0.0
        total_error: Optional[str] = None

        try:
            await server.start()

            # Run benchmark for each context configuration
            for input_len, output_len in context_configs:
                context = {"input_len": input_len, "output_len": output_len}
                try:
                    run_id, fitness = await self._benchmark_context(
                        session_id=session_id,
                        config_id=config_id,
                        fingerprint=fingerprint,
                        server=server,
                        port=port,
                        context=context,
                        flags=flags,
                    )
                    best_fitness = max(best_fitness, fitness)
                    log.info(
                        "Context %s done: fitness=%.4f run_id=%s",
                        context, fitness, run_id,
                    )
                except Exception as exc:
                    log.warning("Context %s failed: %s", context, exc)
                    await self._db.insert_benchmark_run(
                        session_id=session_id,
                        config_id=config_id,
                        fingerprint=fingerprint,
                        context=context,
                        raw_metrics={},
                        fitness_score=0.0,
                        error=str(exc),
                        node_host=self._node_host,
                    )

            await self._db.mark_config_done(config_id, best_fitness)

        except (OOMError, StartupTimeout, PortConflict, CUDAError, Exception) as exc:
            # Include log tail in the stored error so the planner can read it
            from core.vllm_server import ServerFailure
            if isinstance(exc, ServerFailure) and exc.log_tail:
                tail = exc.formatted_tail(30)
                total_error = f"{exc}\n--- vLLM log tail ---\n{tail}"
            else:
                total_error = str(exc)
            log.error("Executor fatal error for config %s: %s", fingerprint[:8], exc)
            await self._db.mark_config_failed(config_id, total_error)

        finally:
            try:
                await server.stop()
            except Exception:
                pass
            await self._gpu_alloc.release(slot)
            await self._port_alloc.release(port)
            log.info(
                "Executor done: config=%s fitness=%.4f error=%s",
                fingerprint[:8], best_fitness, total_error,
            )

    # ------------------------------------------------------------------
    # Benchmark a single context configuration
    # ------------------------------------------------------------------

    async def _benchmark_context(
        self,
        *,
        session_id: str,
        config_id: str,
        fingerprint: str,
        server: VLLMServer,
        port: int,
        context: Dict[str, int],
        flags: VLLMFlags,
    ) -> Tuple[str, float]:
        """Run the benchmark ramp for one context length. Returns (run_id, fitness)."""
        engine = BenchmarkEngine(
            base_url=f"http://localhost:{port}",
            model_id=self._model_id,
            concurrency_levels=self._concurrency_levels,
            num_prompts=self._num_prompts,
            input_len=context["input_len"],
            output_len=context["output_len"],
        )

        ramp_result = await engine.run()
        summary = ramp_result.summary

        # Use LLM to verify/enrich the parsed metrics (best-effort)
        best_result = ramp_result.best_throughput_result()
        raw_stdout = best_result.raw_output if best_result else ""
        enriched = await self._llm_parse_metrics(
            raw_stdout=raw_stdout,
            summary=summary,
            context=context,
        )

        # Compute fitness score using server log analysis + GPU profile
        analysis = LogAnalyzer.analyze(server.log_tail)
        gpu_profile = _load_gpu_profile(self._gpu_type)
        enriched_metrics = MetricsCollector.collect(
            ramp=ramp_result,
            analysis=analysis,
            flags=flags,
            gpu_profile=gpu_profile,
            primary_metric=self._primary_metric,
        )
        fitness = enriched_metrics.fitness_score

        # Per-concurrency-level breakdown for table display
        levels_data = [r.to_dict() for r in ramp_result.levels]

        from dataclasses import asdict as _asdict
        run_id = await self._db.insert_benchmark_run(
            session_id=session_id,
            config_id=config_id,
            fingerprint=fingerprint,
            flags=_asdict(flags),
            context=context,
            raw_metrics=summary,
            levels=levels_data,
            enriched_metrics={
                **enriched_metrics.__dict__,
                "llm_parsed": enriched,
            },
            fitness_score=fitness,
            node_host=self._node_host,
        )
        return run_id, fitness

    # ------------------------------------------------------------------
    # LLM-assisted metric parsing
    # ------------------------------------------------------------------

    async def _llm_parse_metrics(
        self,
        raw_stdout: str,
        summary: Dict[str, Any],
        context: Dict[str, int],
    ) -> Dict[str, Any]:
        """
        Ask the LLM to parse or verify benchmark metrics.
        Falls back to the regex-extracted summary on failure.
        """
        if not raw_stdout or not self._client.api_key:
            return {}

        # Truncate stdout to avoid exceeding token limits
        truncated = raw_stdout[-4000:] if len(raw_stdout) > 4000 else raw_stdout

        user_msg = (
            f"Context: input_len={context['input_len']}, output_len={context['output_len']}\n"
            f"Regex-extracted summary:\n{json.dumps(summary, indent=2)}\n\n"
            f"Raw benchmark stdout (last 4000 chars):\n{truncated}"
        )

        try:
            parsed = await self._client.chat_json(
                messages=[{"role": "user", "content": user_msg}],
                system=_PARSE_SYSTEM_PROMPT,
            )
            return parsed if isinstance(parsed, dict) else {}
        except DOClientError as exc:
            log.debug("LLM metric parse failed: %s", exc)
            return {}
