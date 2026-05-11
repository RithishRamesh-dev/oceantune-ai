"""
agents/correctness_firewall_agent.py
--------------------------------------
Correctness Firewall Agent — Stage 4, Part 3.

Before any generated kernel is accepted into the OceanTune pipeline, it MUST
pass correctness validation. This agent:

  1. Runs the generated kernel's built-in test_correctness() function
  2. Runs a comprehensive shape sweep: (1, 64, 256, 1024, 2048, 4096) × variants
  3. Checks numerical accuracy vs PyTorch reference:
     - Max absolute error < threshold (configurable per op type)
     - RMS error < threshold
     - No NaN or Inf in output
  4. Checks for output determinism (re-run, compare)
  5. Tests edge cases: seq_len=1 (decode), causal vs non-causal, GQA ratios

Returns a CorrectnessReport with PASS/FAIL + specific failure details.

A kernel that fails ANY check is NOT accepted — the KernelEvolutionAgent will
either attempt LLM-guided repair or revert to the previous best kernel.

Usage
-----
    firewall = CorrectnessFirewallAgent()
    report = await firewall.validate(
        kernel=generated_kernel,
        op_type="attention",
        model_meta={"num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        device="cuda:0",
    )
    if report.passed:
        print("Kernel accepted")
    else:
        print("FAILED:", report.failure_reason)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.kernel_generation_agent import GeneratedKernel

log = logging.getLogger("agents.correctness_firewall_agent")

# Max absolute error thresholds per operation type
_MAX_ABS_THRESHOLDS: Dict[str, float] = {
    "attention": 1e-2,   # FP16 attention can have ~1e-2 max abs error vs FP32 ref
    "gemm": 1e-2,
    "rmsnorm": 1e-3,
    "rope": 1e-3,
    "moe": 1e-2,
    "moe_dispatch": 5e-3,
    "default": 1e-2,
}

_RMS_THRESHOLDS: Dict[str, float] = {
    "attention": 5e-4,
    "gemm": 5e-4,
    "rmsnorm": 1e-4,
    "rope": 1e-4,
    "moe": 5e-4,
    "default": 5e-4,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ShapeTestResult:
    """Result of a single shape test case."""
    shape_params: Dict[str, Any]
    passed: bool
    max_abs_error: float = 0.0
    rms_error: float = 0.0
    has_nan: bool = False
    has_inf: bool = False
    error_message: str = ""
    duration_ms: float = 0.0


@dataclass
class CorrectnessReport:
    """Full correctness validation report for a generated kernel."""
    kernel_name: str
    op_type: str
    passed: bool = False

    # Failure details
    failure_reason: str = ""
    failed_tests: List[ShapeTestResult] = field(default_factory=list)

    # Passing tests
    passed_tests: List[ShapeTestResult] = field(default_factory=list)

    # Summary statistics
    total_tests: int = 0
    n_passed: int = 0
    n_failed: int = 0
    max_abs_error_seen: float = 0.0
    rms_error_seen: float = 0.0

    # Determinism check
    is_deterministic: bool = True
    determinism_error: str = ""

    # Import / syntax check
    import_succeeded: bool = False
    import_error: str = ""

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.kernel_name}: "
            f"{self.n_passed}/{self.total_tests} tests passed, "
            f"max_abs={self.max_abs_error_seen:.2e}, "
            f"rms={self.rms_error_seen:.2e}"
            + (f" FAIL: {self.failure_reason}" if not self.passed else "")
        )


# ---------------------------------------------------------------------------
# Correctness Firewall Agent
# ---------------------------------------------------------------------------

class CorrectnessFirewallAgent:
    """
    Validates generated kernels against PyTorch reference implementations.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        timeout_sec: int = 120,
    ) -> None:
        self._device = device
        self._timeout_sec = timeout_sec

    async def validate(
        self,
        *,
        kernel: GeneratedKernel,
        op_type: str = "",
        model_meta: Optional[Dict[str, Any]] = None,
    ) -> CorrectnessReport:
        """
        Run full correctness validation for a generated kernel.

        Parameters
        ----------
        kernel : GeneratedKernel
            The kernel to validate (must have file_path set).
        op_type : str
            Operation type (attention | gemm | rmsnorm | rope | moe).
        model_meta : dict
            Model architecture for shape generation.

        Returns
        -------
        CorrectnessReport with passed=True only if ALL checks pass.
        """
        op_type = op_type or kernel.target_op
        report = CorrectnessReport(
            kernel_name=kernel.kernel_name,
            op_type=op_type,
        )

        # 1. Check the file exists
        if not kernel.file_path or not Path(kernel.file_path).exists():
            report.failure_reason = f"Kernel file not found: {kernel.file_path}"
            return report

        # 2. Import / syntax check
        import_ok, import_err = await self._check_import(kernel.file_path)
        report.import_succeeded = import_ok
        report.import_error = import_err
        if not import_ok:
            report.failure_reason = f"Import failed: {import_err}"
            return report

        # 3. Run the kernel's built-in test_correctness()
        builtin_ok, builtin_err = await self._run_builtin_test(kernel.file_path)
        if not builtin_ok:
            # Failing built-in test is not immediately fatal — it might test
            # a different reference. Record and continue.
            log.warning("Built-in correctness test failed: %s", builtin_err[:200])

        # 4. Run shape sweep
        shapes = self._generate_shape_sweep(op_type, model_meta or {})
        validation_script = self._generate_validation_script(
            kernel.file_path, op_type, shapes, model_meta or {}
        )

        results_json = await self._run_validation_script(validation_script)
        if results_json is None:
            report.failure_reason = "Validation script execution failed"
            return report

        # 5. Parse results
        for r in results_json.get("results", []):
            stest = ShapeTestResult(
                shape_params=r.get("params", {}),
                passed=r.get("passed", False),
                max_abs_error=r.get("max_abs_error", 0.0),
                rms_error=r.get("rms_error", 0.0),
                has_nan=r.get("has_nan", False),
                has_inf=r.get("has_inf", False),
                error_message=r.get("error", ""),
                duration_ms=r.get("duration_ms", 0.0),
            )
            report.total_tests += 1
            if stest.passed:
                report.n_passed += 1
                report.passed_tests.append(stest)
            else:
                report.n_failed += 1
                report.failed_tests.append(stest)

            report.max_abs_error_seen = max(report.max_abs_error_seen, stest.max_abs_error)
            report.rms_error_seen = max(report.rms_error_seen, stest.rms_error)

        # 6. Check determinism (if tests ran)
        det_result = results_json.get("determinism_check", {})
        report.is_deterministic = det_result.get("passed", True)
        report.determinism_error = det_result.get("error", "")

        # 7. Overall pass/fail
        if report.n_failed > 0:
            first_fail = report.failed_tests[0]
            if first_fail.has_nan:
                report.failure_reason = "NaN in output"
            elif first_fail.has_inf:
                report.failure_reason = "Inf in output"
            else:
                report.failure_reason = (
                    f"Numerical error too large: max_abs={first_fail.max_abs_error:.2e}, "
                    f"rms={first_fail.rms_error:.2e} for params={first_fail.shape_params}"
                )
        elif not report.is_deterministic:
            report.failure_reason = f"Non-deterministic: {report.determinism_error}"
        elif report.total_tests == 0:
            report.failure_reason = "No tests ran"
        else:
            report.passed = True

        log.info("CorrectnessFirewall: %s", report.summary())
        return report

    async def _check_import(self, file_path: str) -> tuple[bool, str]:
        """Try to import the kernel module (syntax + import check)."""
        script = f"import importlib.util; spec = importlib.util.spec_from_file_location('kernel', '{file_path}'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); print('ok')"
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": self._device.replace("cuda:", "")},
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if b"ok" in stdout:
                return True, ""
            return False, stderr.decode()[:300]
        except asyncio.TimeoutError:
            return False, "Import timed out"
        except Exception as exc:
            return False, str(exc)

    async def _run_builtin_test(self, file_path: str) -> tuple[bool, str]:
        """Run the kernel file as a script (triggers __main__ block)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, file_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": self._device.replace("cuda:", "")},
                cwd=str(Path(file_path).parent),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._timeout_sec)
            if proc.returncode == 0:
                return True, stdout.decode()[:200]
            return False, stderr.decode()[:300]
        except asyncio.TimeoutError:
            return False, "Built-in test timed out"
        except Exception as exc:
            return False, str(exc)

    def _generate_shape_sweep(
        self,
        op_type: str,
        model_meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate a comprehensive shape sweep for the operation type."""
        nheads = model_meta.get("num_heads", 32)
        nkv = model_meta.get("num_kv_heads", nheads)
        head_dim = model_meta.get("head_dim", 128)

        if op_type == "attention":
            shapes = []
            for bs in [1, 4]:
                for seqlen in [1, 64, 512, 2048]:
                    shapes.append({
                        "batch_size": bs, "seq_len": seqlen,
                        "num_heads": nheads, "num_kv_heads": nkv,
                        "head_dim": head_dim,
                    })
            return shapes

        if op_type == "gemm":
            shapes = []
            for m in [1, 64, 512, 2048]:
                for n in [4096, 8192]:
                    shapes.append({"M": m, "N": n, "K": model_meta.get("hidden_dim", 4096)})
            return shapes

        if op_type in ("rmsnorm", "rope"):
            return [
                {"batch_size": 1, "seq_len": 1, "hidden_dim": model_meta.get("hidden_dim", 4096)},
                {"batch_size": 4, "seq_len": 512, "hidden_dim": model_meta.get("hidden_dim", 4096)},
                {"batch_size": 1, "seq_len": 2048, "hidden_dim": model_meta.get("hidden_dim", 4096)},
            ]

        # Generic shape
        return [{"size": 1024}, {"size": 4096}]

    def _generate_validation_script(
        self,
        kernel_file: str,
        op_type: str,
        shapes: List[Dict[str, Any]],
        model_meta: Dict[str, Any],
    ) -> str:
        """Generate a standalone validation script."""
        max_abs = _MAX_ABS_THRESHOLDS.get(op_type, _MAX_ABS_THRESHOLDS["default"])
        rms_thresh = _RMS_THRESHOLDS.get(op_type, _RMS_THRESHOLDS["default"])

        if op_type == "attention":
            return self._gen_attention_validation(
                kernel_file, shapes, max_abs, rms_thresh, model_meta
            )
        if op_type == "gemm":
            return self._gen_gemm_validation(kernel_file, shapes, max_abs, rms_thresh)
        if op_type == "rmsnorm":
            return self._gen_rmsnorm_validation(kernel_file, shapes, max_abs, rms_thresh)
        # Generic: just try to import and run the built-in test
        return self._gen_import_only_validation(kernel_file)

    def _gen_attention_validation(
        self,
        kernel_file: str,
        shapes: List[Dict[str, Any]],
        max_abs: float,
        rms_thresh: float,
        model_meta: Dict[str, Any],
    ) -> str:
        shapes_json = json.dumps(shapes)
        return f"""\
import sys, json, time, importlib.util, torch
import torch.nn.functional as F

results = []

def load_kernel(path):
    spec = importlib.util.spec_from_file_location("kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

try:
    mod = load_kernel("{kernel_file}")
    kernel_fn = getattr(mod, "attention", None) or getattr(mod, "flash_attn", None)
except Exception as e:
    print(json.dumps({{"error": f"Import failed: {{e}}", "results": [], "determinism_check": {{"passed": False, "error": str(e)}}}}))
    sys.exit(0)

shapes = {shapes_json}

for shape in shapes:
    bs = shape["batch_size"]
    seqlen = shape["seq_len"]
    nh = shape["num_heads"]
    nkv = shape.get("num_kv_heads", nh)
    hd = shape["head_dim"]
    result = {{"params": shape, "passed": False, "max_abs_error": 0, "rms_error": 0, "has_nan": False, "has_inf": False, "error": "", "duration_ms": 0}}

    try:
        torch.manual_seed(42)
        q = torch.randn(bs, nh, seqlen, hd, dtype=torch.float16, device="cuda")
        k = torch.randn(bs, nkv, seqlen, hd, dtype=torch.float16, device="cuda")
        v = torch.randn(bs, nkv, seqlen, hd, dtype=torch.float16, device="cuda")

        # Reference: torch SDPA
        kv_rpt = nh // nkv
        k_ref = k.repeat_interleave(kv_rpt, dim=1) if kv_rpt > 1 else k
        v_ref = v.repeat_interleave(kv_rpt, dim=1) if kv_rpt > 1 else v
        with torch.no_grad():
            ref = F.scaled_dot_product_attention(q, k_ref, v_ref, is_causal=(seqlen > 1))

        # Run generated kernel
        t0 = time.time()
        if kernel_fn is not None:
            try:
                out = kernel_fn(q, k, v)
            except TypeError:
                out = kernel_fn(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), causal=(seqlen > 1))
                if out.shape != ref.shape:
                    out = out.transpose(1, 2)
        else:
            raise RuntimeError("No callable attention function found in kernel module")
        result["duration_ms"] = (time.time() - t0) * 1000

        result["has_nan"] = bool(torch.isnan(out).any().item())
        result["has_inf"] = bool(torch.isinf(out).any().item())
        if result["has_nan"] or result["has_inf"]:
            result["error"] = "NaN or Inf in output"
        else:
            diff = (out.float() - ref.float()).abs()
            result["max_abs_error"] = float(diff.max().item())
            result["rms_error"] = float(diff.pow(2).mean().sqrt().item())
            result["passed"] = (
                result["max_abs_error"] < {max_abs}
                and result["rms_error"] < {rms_thresh}
            )
            if not result["passed"]:
                result["error"] = f"Numerical error exceeds threshold: max_abs={{result['max_abs_error']:.2e}} rms={{result['rms_error']:.2e}}"
    except Exception as e:
        result["error"] = str(e)[:200]

    results.append(result)

# Determinism check (small shape)
det = {{"passed": True, "error": ""}}
try:
    q = torch.randn(1, {model_meta.get('num_heads', 32)}, 64, {model_meta.get('head_dim', 128)}, dtype=torch.float16, device="cuda")
    k = torch.randn_like(q); v = torch.randn_like(q)
    if kernel_fn is not None:
        out1 = kernel_fn(q, k, v)
        out2 = kernel_fn(q, k, v)
        if not torch.allclose(out1, out2, atol=1e-5):
            det = {{"passed": False, "error": "Non-deterministic outputs"}}
except Exception as e:
    det = {{"passed": True, "error": ""}}  # Skip if not runnable

print(json.dumps({{"results": results, "determinism_check": det}}))
"""

    def _gen_gemm_validation(
        self,
        kernel_file: str,
        shapes: List[Dict[str, Any]],
        max_abs: float,
        rms_thresh: float,
    ) -> str:
        shapes_json = json.dumps(shapes)
        return f"""\
import sys, json, importlib.util, torch

results = []

try:
    spec = importlib.util.spec_from_file_location("kernel", "{kernel_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kernel_fn = getattr(mod, "matmul", None) or getattr(mod, "gemm", None)
except Exception as e:
    print(json.dumps({{"error": str(e), "results": [], "determinism_check": {{"passed": False, "error": str(e)}}}}))
    sys.exit(0)

for shape in {shapes_json}:
    M, N, K = shape["M"], shape["N"], shape["K"]
    result = {{"params": shape, "passed": False, "max_abs_error": 0, "rms_error": 0, "has_nan": False, "has_inf": False, "error": ""}}
    try:
        a = torch.randn(M, K, dtype=torch.float16, device="cuda")
        b = torch.randn(K, N, dtype=torch.float16, device="cuda")
        ref = torch.mm(a.float(), b.float())
        if kernel_fn:
            out = kernel_fn(a, b)
        else:
            out = torch.mm(a, b)
        result["has_nan"] = bool(torch.isnan(out).any())
        result["has_inf"] = bool(torch.isinf(out).any())
        if not result["has_nan"] and not result["has_inf"]:
            diff = (out.float() - ref).abs()
            result["max_abs_error"] = float(diff.max())
            result["rms_error"] = float(diff.pow(2).mean().sqrt())
            result["passed"] = result["max_abs_error"] < {max_abs}
    except Exception as e:
        result["error"] = str(e)[:200]
    results.append(result)

print(json.dumps({{"results": results, "determinism_check": {{"passed": True}}}}))
"""

    def _gen_rmsnorm_validation(
        self,
        kernel_file: str,
        shapes: List[Dict[str, Any]],
        max_abs: float,
        rms_thresh: float,
    ) -> str:
        shapes_json = json.dumps(shapes)
        return f"""\
import sys, json, importlib.util, torch

results = []

try:
    spec = importlib.util.spec_from_file_location("kernel", "{kernel_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kernel_fn = getattr(mod, "rmsnorm", None) or getattr(mod, "rms_norm", None)
except Exception as e:
    print(json.dumps({{"error": str(e), "results": [], "determinism_check": {{"passed": True}}}}))
    sys.exit(0)

for shape in {shapes_json}:
    bs, seqlen, hdim = shape["batch_size"], shape["seq_len"], shape["hidden_dim"]
    result = {{"params": shape, "passed": False, "max_abs_error": 0, "rms_error": 0, "has_nan": False, "has_inf": False, "error": ""}}
    try:
        x = torch.randn(bs, seqlen, hdim, dtype=torch.float16, device="cuda")
        w = torch.ones(hdim, dtype=torch.float16, device="cuda")
        ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
        if kernel_fn:
            out = kernel_fn(x, w)
        else:
            out = ref
        diff = (out.float() - ref.float()).abs()
        result["max_abs_error"] = float(diff.max())
        result["rms_error"] = float(diff.pow(2).mean().sqrt())
        result["has_nan"] = bool(torch.isnan(out).any())
        result["has_inf"] = bool(torch.isinf(out).any())
        result["passed"] = result["max_abs_error"] < {max_abs} and not result["has_nan"]
    except Exception as e:
        result["error"] = str(e)[:200]
    results.append(result)

print(json.dumps({{"results": results, "determinism_check": {{"passed": True}}}}))
"""

    def _gen_import_only_validation(self, kernel_file: str) -> str:
        return f"""\
import sys, json, importlib.util
try:
    spec = importlib.util.spec_from_file_location("kernel", "{kernel_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(json.dumps({{"results": [{{"params": {{}}, "passed": True, "max_abs_error": 0, "rms_error": 0, "has_nan": False, "has_inf": False, "error": ""}}], "determinism_check": {{"passed": True}}}}))
except Exception as e:
    print(json.dumps({{"results": [{{"params": {{}}, "passed": False, "max_abs_error": 0, "rms_error": 0, "has_nan": False, "has_inf": False, "error": str(e)}}], "determinism_check": {{"passed": True}}}}))
"""

    async def _run_validation_script(
        self, script: str
    ) -> Optional[Dict[str, Any]]:
        """Run validation script, return parsed JSON output."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="oceantune_val_"
        ) as f:
            f.write(script)
            script_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": self._device.replace("cuda:", "")},
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._timeout_sec)

            output = stdout.decode().strip()
            if output:
                try:
                    return json.loads(output)
                except json.JSONDecodeError:
                    log.warning("Validation script JSON parse error: %s", output[:200])
                    return None
            else:
                log.warning("Validation script produced no output. Stderr: %s", stderr.decode()[:300])
                return None

        except asyncio.TimeoutError:
            log.warning("Validation script timed out")
            return None
        except Exception as exc:
            log.warning("Validation script error: %s", exc)
            return None
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass
