"""
agents/kernel_evolution_agent.py
----------------------------------
Kernel Evolution Agent — Stage 4, Part 4.

Implements the autonomous keep/revert loop for iterative kernel optimization:

  Generate → Benchmark → Validate → Keep or Revert → Repeat

The agent:
  1. Maintains an experiment tree in experiments/kernel_experiments.json
  2. Tracks the historical best kernel per (model_id, gpu_type, op_type) tuple
  3. Evaluates each generated kernel against:
       a. CorrectnessFirewallAgent (must pass ALL checks)
       b. KernelBenchmarkAgent (must show >1% speedup over current best)
  4. Keeps a kernel only if it passes correctness AND improves performance
  5. Reverts to the previous best if the new kernel fails either check
  6. Checkpoints the best kernel to experiments/best_kernels.json
  7. Runs up to `max_iterations` evolution rounds, then returns the best found

Experiment tree format (experiments/kernel_experiments.json):
{
  "<session_id>": {
    "<op_type>": {
      "best_kernel_path": "kernels/generated/.../kernel.py",
      "best_speedup_pct": 12.5,
      "iterations": [
        {
          "iteration": 1,
          "kernel_path": "...",
          "correctness": {"passed": true, ...},
          "benchmark": {"speedup_pct": 12.5, ...},
          "decision": "kept|reverted|skipped",
          "reason": "..."
        }
      ]
    }
  }
}

Usage
-----
    evolver = KernelEvolutionAgent(
        do_client=client,
        device="cuda:0",
    )
    best_kernel = await evolver.evolve(
        bottleneck=bottleneck_analysis,
        research=kernel_research_report,
        model_id="Qwen/Qwen2.5-7B-Instruct",
        gpu_type="H100",
        session_id="...",
        model_meta={...},
        max_iterations=5,
    )
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents.bottleneck_reasoning_agent import BottleneckAnalysis
from agents.correctness_firewall_agent import CorrectnessFirewallAgent, CorrectnessReport
from agents.do_client import DOClient
from agents.kernel_generation_agent import GeneratedKernel, KernelGenerationAgent
from agents.kernel_research_agent import KernelResearchReport
from microbench.operator_bench import BenchResult, OperatorBench

log = logging.getLogger("agents.kernel_evolution_agent")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXPERIMENTS_FILE = _REPO_ROOT / "experiments" / "kernel_experiments.json"
_BEST_KERNELS_FILE = _REPO_ROOT / "experiments" / "best_kernels.json"

# Minimum speedup required to accept a new kernel
_MIN_SPEEDUP_PCT = 1.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Microbenchmark result for a generated kernel."""
    kernel_path: str
    op_type: str
    latency_us_p50: float = 0.0
    latency_us_ref: float = 0.0     # PyTorch/vLLM reference
    speedup_pct: float = 0.0
    tflops_achieved: float = 0.0
    tflops_efficiency_pct: float = 0.0
    roofline_bound: str = "unknown"
    success: bool = False
    error: str = ""


@dataclass
class EvolutionIteration:
    """One round in the kernel evolution loop."""
    iteration: int
    kernel_path: str
    correctness: Optional[CorrectnessReport] = None
    benchmark: Optional[BenchmarkResult] = None
    decision: str = "pending"   # kept | reverted | skipped | failed_correctness | failed_benchmark
    reason: str = ""
    timestamp: str = ""


@dataclass
class EvolutionResult:
    """Final result of the kernel evolution session."""
    session_id: str
    op_type: str
    model_id: str
    gpu_type: str

    best_kernel: Optional[GeneratedKernel] = None
    best_speedup_pct: float = 0.0
    iterations_run: int = 0
    total_kept: int = 0
    total_reverted: int = 0

    iterations: List[EvolutionIteration] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Evolution [{self.op_type}] on {self.gpu_type}: "
            f"{self.iterations_run} iterations, "
            f"best_speedup={self.best_speedup_pct:.1f}%, "
            f"kept={self.total_kept}, reverted={self.total_reverted}"
        )


# ---------------------------------------------------------------------------
# Kernel Evolution Agent
# ---------------------------------------------------------------------------

class KernelEvolutionAgent:
    """
    Autonomous kernel optimization loop: generate → benchmark → keep/revert.
    """

    def __init__(
        self,
        do_client: DOClient,
        device: str = "cuda:0",
        bench_timeout_sec: int = 120,
    ) -> None:
        self._client = do_client
        self._device = device
        self._bench_timeout = bench_timeout_sec

    async def evolve(
        self,
        *,
        bottleneck: BottleneckAnalysis,
        research: KernelResearchReport,
        model_id: str,
        gpu_type: str,
        session_id: str,
        model_meta: Optional[Dict[str, Any]] = None,
        max_iterations: int = 5,
        generate_cuda: bool = False,
    ) -> EvolutionResult:
        """
        Run the kernel evolution loop.

        Returns the best kernel found across all iterations.
        """
        op_type = bottleneck.primary_component or research.target_kernel or "attention"
        result = EvolutionResult(
            session_id=session_id,
            op_type=op_type,
            model_id=model_id,
            gpu_type=gpu_type,
        )

        log.info(
            "KernelEvolutionAgent: evolving %s on %s for %s (%d iterations)",
            op_type, gpu_type, model_id, max_iterations,
        )

        generator = KernelGenerationAgent(do_client=self._client)
        firewall = CorrectnessFirewallAgent(device=self._device, timeout_sec=self._bench_timeout)
        bencher = OperatorBench(gpu_type=gpu_type, device=self._device)

        best_kernel: Optional[GeneratedKernel] = None
        best_speedup = 0.0
        best_latency_us: Optional[float] = None

        # Get reference latency from existing implementation
        ref_params = self._get_bench_params(op_type, model_meta or {})
        ref_result = await bencher.run(
            op_type=op_type,
            params=ref_params,
            num_warmup=10,
            num_iters=50,
            timeout_sec=self._bench_timeout,
        )
        if ref_result.success:
            best_latency_us = ref_result.latency_us_p50
            log.info(
                "Reference latency: %.1f us (%.1f TFLOP/s, %s-bound)",
                best_latency_us, ref_result.tflops_achieved, ref_result.roofline_bound,
            )

        for i in range(1, max_iterations + 1):
            log.info("Evolution iteration %d/%d", i, max_iterations)
            ev_iter = EvolutionIteration(
                iteration=i,
                kernel_path="",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            # 1. Generate kernel
            kernel = await generator.generate(
                research=research,
                bottleneck=bottleneck,
                model_id=model_id,
                gpu_type=gpu_type,
                session_id=session_id,
                model_meta=model_meta,
                generate_cuda=generate_cuda,
            )
            ev_iter.kernel_path = kernel.file_path or ""

            if not kernel.success:
                ev_iter.decision = "skipped"
                ev_iter.reason = f"Generation failed: {kernel.error}"
                result.iterations.append(ev_iter)
                log.warning("Iteration %d: generation failed — %s", i, kernel.error[:100])
                continue

            # 2. Correctness validation
            correctness = await firewall.validate(
                kernel=kernel,
                op_type=op_type,
                model_meta=model_meta,
            )
            ev_iter.correctness = correctness

            if not correctness.passed:
                ev_iter.decision = "failed_correctness"
                ev_iter.reason = correctness.failure_reason
                result.iterations.append(ev_iter)
                result.total_reverted += 1
                log.warning(
                    "Iteration %d: correctness FAILED — %s",
                    i, correctness.failure_reason,
                )

                # Try LLM-guided repair before giving up on this iteration
                repaired = await self._attempt_repair(
                    kernel=kernel,
                    correctness_report=correctness,
                    firewall=firewall,
                    op_type=op_type,
                    model_meta=model_meta or {},
                )
                if repaired is not None:
                    log.info("Iteration %d: repair succeeded", i)
                    kernel = repaired
                    # Re-validate the repaired kernel
                    correctness = await firewall.validate(
                        kernel=kernel, op_type=op_type, model_meta=model_meta
                    )
                    ev_iter.correctness = correctness
                    ev_iter.kernel_path = kernel.file_path or ""
                    if not correctness.passed:
                        log.warning("Iteration %d: repair failed correctness too", i)
                        continue
                else:
                    continue

            # 3. Microbenchmark
            bench = await self._benchmark_kernel(
                kernel=kernel,
                op_type=op_type,
                params=ref_params,
                ref_latency_us=best_latency_us,
                bench_timeout=self._bench_timeout,
            )
            ev_iter.benchmark = bench

            # 4. Keep or revert decision
            speedup = bench.speedup_pct

            if bench.success and speedup > _MIN_SPEEDUP_PCT:
                # Keep!
                ev_iter.decision = "kept"
                ev_iter.reason = (
                    f"Speedup +{speedup:.1f}% over reference "
                    f"({bench.latency_us_p50:.1f}us vs {bench.latency_us_ref:.1f}us ref)"
                )
                best_kernel = kernel
                best_speedup = max(best_speedup, speedup)
                best_latency_us = bench.latency_us_p50
                result.total_kept += 1
                log.info(
                    "Iteration %d KEPT: speedup=+%.1f%% latency=%.1fus",
                    i, speedup, bench.latency_us_p50,
                )
            else:
                ev_iter.decision = "reverted"
                if not bench.success:
                    ev_iter.reason = f"Benchmark failed: {bench.error}"
                else:
                    ev_iter.reason = (
                        f"No improvement: speedup={speedup:.1f}% < threshold={_MIN_SPEEDUP_PCT}%"
                    )
                result.total_reverted += 1
                log.info(
                    "Iteration %d REVERTED: speedup=%.1f%% (threshold=%.1f%%)",
                    i, speedup, _MIN_SPEEDUP_PCT,
                )

            result.iterations.append(ev_iter)
            result.iterations_run = i

        result.best_kernel = best_kernel
        result.best_speedup_pct = best_speedup

        # Save experiment records
        self._save_experiment(session_id, op_type, result, best_kernel)

        log.info("Evolution complete: %s", result.summary())
        return result

    async def _benchmark_kernel(
        self,
        *,
        kernel: GeneratedKernel,
        op_type: str,
        params: Dict[str, Any],
        ref_latency_us: Optional[float],
        bench_timeout: int,
    ) -> BenchmarkResult:
        """
        Benchmark the kernel by running its built-in benchmark() function.
        Falls back to OperatorBench if the built-in is unavailable.
        """
        bench_result = BenchmarkResult(
            kernel_path=kernel.file_path or "",
            op_type=op_type,
        )

        if not kernel.file_path or not Path(kernel.file_path).exists():
            bench_result.error = "Kernel file not found"
            return bench_result

        # Try to run the kernel's built-in benchmark
        import asyncio, os, sys, json as _json, tempfile

        run_script = f"""\
import sys, json, importlib.util, time
spec = importlib.util.spec_from_file_location("kernel", "{kernel.file_path}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
if hasattr(mod, "benchmark"):
    mod.benchmark()
else:
    print(json.dumps({{"mean_us": 0, "p50_us": 0, "tflops": 0}}))
"""
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", run_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": self._device.replace("cuda:", "")},
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=bench_timeout)
            output = stdout.decode().strip()
            if output:
                try:
                    metrics = _json.loads(output)
                    bench_result.latency_us_p50 = float(metrics.get("p50_us", metrics.get("mean_us", 0)))
                    bench_result.tflops_achieved = float(metrics.get("tflops", 0))
                    bench_result.success = bench_result.latency_us_p50 > 0

                    if ref_latency_us and bench_result.latency_us_p50 > 0:
                        bench_result.latency_us_ref = ref_latency_us
                        bench_result.speedup_pct = (
                            (ref_latency_us - bench_result.latency_us_p50) / ref_latency_us
                        ) * 100
                except _json.JSONDecodeError:
                    bench_result.error = f"JSON parse failed: {output[:200]}"
        except asyncio.TimeoutError:
            bench_result.error = f"Benchmark timed out after {bench_timeout}s"
        except Exception as exc:
            bench_result.error = str(exc)

        return bench_result

    async def _attempt_repair(
        self,
        *,
        kernel: GeneratedKernel,
        correctness_report: CorrectnessReport,
        firewall: CorrectnessFirewallAgent,
        op_type: str,
        model_meta: Dict[str, Any],
    ) -> Optional[GeneratedKernel]:
        """
        Ask the LLM to repair a failing kernel.
        Returns a new GeneratedKernel or None if repair fails.
        """
        if not kernel.triton_code:
            return None

        failure = correctness_report.failure_reason
        failed_tests = [
            f"params={t.shape_params}, error={t.error_message or f'max_abs={t.max_abs_error:.2e}'}"
            for t in correctness_report.failed_tests[:3]
        ]

        repair_prompt = (
            f"The following Triton kernel has a correctness failure.\n\n"
            f"FAILURE: {failure}\n\n"
            f"FAILED TEST CASES:\n" + "\n".join(f"  - {t}" for t in failed_tests) + "\n\n"
            f"KERNEL CODE:\n```python\n{kernel.triton_code[:3000]}\n```\n\n"
            f"Fix the kernel. Output ONLY the corrected Python code."
        )

        try:
            fixed_code = await self._client.chat(
                messages=[{"role": "user", "content": repair_prompt}],
                system="You are an expert Triton kernel engineer. Fix the bug in the kernel. Output only the corrected Python code.",
                json_mode=False,
            )
            from agents.kernel_generation_agent import _strip_code_fences
            fixed_code = _strip_code_fences(fixed_code)

            # Write repaired kernel
            from pathlib import Path
            original_path = Path(kernel.file_path)
            repaired_path = original_path.parent / (original_path.stem + "_repaired.py")
            repaired_path.write_text(fixed_code, encoding="utf-8")

            repaired_kernel = GeneratedKernel(
                session_id=kernel.session_id,
                kernel_name=kernel.kernel_name + "_repaired",
                target_op=kernel.target_op,
                gpu_type=kernel.gpu_type,
                model_id=kernel.model_id,
                triton_code=fixed_code,
                file_path=str(repaired_path),
                success=True,
            )
            return repaired_kernel

        except Exception as exc:
            log.warning("Kernel repair failed: %s", exc)
            return None

    def _get_bench_params(
        self, op_type: str, model_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get default benchmark parameters for the operation type."""
        if op_type == "attention":
            return {
                "batch_size": 1,
                "seq_len": 2048,
                "num_heads": model_meta.get("num_heads", 32),
                "num_kv_heads": model_meta.get("num_kv_heads", model_meta.get("num_heads", 32)),
                "head_dim": model_meta.get("head_dim", 128),
            }
        if op_type == "gemm":
            return {
                "M": 1,
                "N": model_meta.get("hidden_dim", 4096),
                "K": model_meta.get("hidden_dim", 4096),
            }
        return {"batch_size": 1, "seq_len": 512, "hidden_dim": model_meta.get("hidden_dim", 4096)}

    def _save_experiment(
        self,
        session_id: str,
        op_type: str,
        result: EvolutionResult,
        best_kernel: Optional[GeneratedKernel],
    ) -> None:
        """Persist experiment records to disk."""
        _EXPERIMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Load existing
        records: Dict[str, Any] = {}
        if _EXPERIMENTS_FILE.exists():
            try:
                records = json.loads(_EXPERIMENTS_FILE.read_text())
            except Exception:
                records = {}

        # Add this session's record
        if session_id not in records:
            records[session_id] = {}

        records[session_id][op_type] = {
            "best_kernel_path": best_kernel.file_path if best_kernel else None,
            "best_speedup_pct": result.best_speedup_pct,
            "iterations_run": result.iterations_run,
            "kept": result.total_kept,
            "reverted": result.total_reverted,
            "iterations": [
                {
                    "iteration": it.iteration,
                    "kernel_path": it.kernel_path,
                    "decision": it.decision,
                    "reason": it.reason,
                    "timestamp": it.timestamp,
                    "correctness_passed": it.correctness.passed if it.correctness else None,
                    "speedup_pct": it.benchmark.speedup_pct if it.benchmark else None,
                }
                for it in result.iterations
            ],
        }

        _EXPERIMENTS_FILE.write_text(json.dumps(records, indent=2))

        # Update best kernels checkpoint
        best_records: Dict[str, Any] = {}
        if _BEST_KERNELS_FILE.exists():
            try:
                best_records = json.loads(_BEST_KERNELS_FILE.read_text())
            except Exception:
                best_records = {}

        if best_kernel and result.best_speedup_pct > 0:
            key = f"{result.model_id}_{result.gpu_type}_{op_type}"
            existing = best_records.get(key, {})
            if result.best_speedup_pct > existing.get("speedup_pct", 0):
                best_records[key] = {
                    "session_id": session_id,
                    "kernel_path": best_kernel.file_path,
                    "speedup_pct": result.best_speedup_pct,
                    "model_id": result.model_id,
                    "gpu_type": result.gpu_type,
                    "op_type": op_type,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                _BEST_KERNELS_FILE.write_text(json.dumps(best_records, indent=2))
                log.info(
                    "New best kernel checkpoint: %s → %.1f%% speedup",
                    key, result.best_speedup_pct,
                )
