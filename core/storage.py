"""
core/storage.py
---------------
Persistent result storage for OceanTune AI.

Responsibilities
----------------
1. Define RunRecord — the canonical record of one complete experiment run.
   Contains VLLMFlags, RampResult summary, LogAnalysis, EnrichedMetrics,
   timestamps, git commit hash, and context info.

2. ResultStorage — writes RunRecords to:
   a) storage/results/runs.csv     — appended row (one per run, all fields flat)
   b) storage/results/runs/{id}.json — full record per run (no data loss)
   c) DigitalOcean Spaces          — async upload, non-blocking

3. ResultLoader — reads historical run records back from CSV/JSON for the
   optimiser to analyse across sessions.

Design notes
------------
- CSV is append-only. The optimiser reads the full CSV at startup to
  recover history from previous sessions.
- JSON preserves nested structure (e.g. full levels list from RampResult).
- Spaces upload is fire-and-forget; local write is always attempted first.
- RunRecord is JSON-serialisable (all fields are plain Python types).
- The git commit hash is included so experiments can be traced to code.

Usage
-----
    from core.storage import ResultStorage, RunRecord, make_run_record

    record = make_run_record(flags=flags, ramp=ramp, analysis=analysis,
                              enriched=enriched, cfg=cfg)
    storage = ResultStorage(cfg)
    storage.save(record)

    # Load history for optimiser
    from core.storage import ResultLoader
    history = ResultLoader.load_csv(storage.csv_path)
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from core.benchmark_runner import RampResult
from core.config import OceanTuneConfig
from core.log_analyzer import LogAnalysis
from core.metrics_collector import EnrichedMetrics
from core.search_space import VLLMFlags
from core.logger import get_logger, log_dict

log = get_logger("core.storage")

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "storage" / "results"
RUNS_DIR = RESULTS_DIR / "runs"


# ===========================================================================
# 1.  RunRecord — the canonical experiment record
# ===========================================================================

@dataclass
class RunRecord:
    """
    Complete record of one OceanTune experiment run.

    One record = one (VLLMFlags config, context_length) pair.
    All fields are JSON-serialisable plain Python types.
    """

    # ── Identity ─────────────────────────────────────────────────────────
    run_id: str = ""                    # from VLLMFlags.fingerprint()
    session_id: str = ""                # shared across all runs in one session
    timestamp_utc: float = field(default_factory=time.time)
    git_commit: str = ""                # short hash for traceability

    # ── Config ────────────────────────────────────────────────────────────
    model_id: str = ""
    gpu_type: str = ""
    input_len: int = 0
    output_len: int = 0
    context_label: str = ""             # e.g. "1K/4K"
    optimiser_strategy: str = ""
    generation: int = 0                 # evolutionary generation number

    # ── vLLM flags (flat dict from VLLMFlags.to_dict()) ──────────────────
    flags: Dict = field(default_factory=dict)

    # ── Benchmark summary (flat dict from RampResult.summary) ────────────
    benchmark_summary: Dict = field(default_factory=dict)

    # ── Full benchmark levels (list of BenchmarkResult.to_dict()) ────────
    benchmark_levels: List[Dict] = field(default_factory=list)

    # ── Log analysis ──────────────────────────────────────────────────────
    log_analysis: Dict = field(default_factory=dict)

    # ── Enriched metrics ──────────────────────────────────────────────────
    enriched_metrics: Dict = field(default_factory=dict)

    # ── Run status ────────────────────────────────────────────────────────
    run_successful: bool = False
    failure_reason: str = ""
    fitness_score: float = 0.0

    # ── Timing ────────────────────────────────────────────────────────────
    total_run_duration_sec: float = 0.0
    benchmark_duration_sec: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_csv_row(self) -> dict:
        """
        Flat dict suitable for a CSV row.
        Nested dicts are flattened with double-underscore separators.
        Lists are stored as JSON strings.
        """
        row: dict = {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "timestamp_utc": self.timestamp_utc,
            "git_commit": self.git_commit,
            "model_id": self.model_id,
            "gpu_type": self.gpu_type,
            "input_len": self.input_len,
            "output_len": self.output_len,
            "context_label": self.context_label,
            "optimiser_strategy": self.optimiser_strategy,
            "generation": self.generation,
            "run_successful": self.run_successful,
            "failure_reason": self.failure_reason,
            "fitness_score": self.fitness_score,
            "total_run_duration_sec": self.total_run_duration_sec,
            "benchmark_duration_sec": self.benchmark_duration_sec,
        }
        # Flatten flags dict
        for k, v in self.flags.items():
            row[f"flag__{k}"] = v
        # Flatten benchmark summary
        for k, v in self.benchmark_summary.items():
            row[f"bench__{k}"] = v
        # Flatten enriched metrics
        for k, v in self.enriched_metrics.items():
            if not isinstance(v, (dict, list)):
                row[f"metric__{k}"] = v
        # Log analysis highlights
        for k in ("model_load_time_sec", "kv_cache_blocks", "kv_cache_gb",
                  "total_startup_sec", "has_errors", "has_oom"):
            row[f"log__{k}"] = self.log_analysis.get(k)
        return row


# ===========================================================================
# 2.  Factory function
# ===========================================================================

def make_run_record(
    flags: VLLMFlags,
    ramp: RampResult,
    analysis: LogAnalysis,
    enriched: EnrichedMetrics,
    cfg: OceanTuneConfig,
    session_id: str = "",
    generation: int = 0,
    total_run_duration_sec: float = 0.0,
) -> RunRecord:
    """
    Build a RunRecord from all the pieces produced in Steps 2–5.

    Parameters
    ----------
    flags : VLLMFlags
        The config that was benchmarked.
    ramp : RampResult
        Full concurrency ramp output from BenchmarkEngine.
    analysis : LogAnalysis
        Structured log output from LogAnalyzer.
    enriched : EnrichedMetrics
        Derived metrics from MetricsCollector.
    cfg : OceanTuneConfig
        Global config for model_id, gpu_type, etc.
    session_id : str
        UUID shared across all runs in one optimiser session.
    generation : int
        Evolutionary generation number (0 = random init).
    total_run_duration_sec : float
        Wall-clock time for the full run including server startup.
    """
    input_len = ramp.input_len
    output_len = ramp.output_len
    context_label = f"{input_len // 1024}K/{output_len // 1024}K"

    return RunRecord(
        run_id=flags.run_id or flags.fingerprint(),
        session_id=session_id,
        timestamp_utc=time.time(),
        git_commit=_get_git_hash(),
        model_id=cfg.model_id,
        gpu_type=cfg.gpu_type,
        input_len=input_len,
        output_len=output_len,
        context_label=context_label,
        optimiser_strategy=cfg.optimiser.strategy,
        generation=generation,
        flags=flags.to_dict(),
        benchmark_summary=ramp.summary,
        benchmark_levels=[r.to_dict() for r in ramp.levels],
        log_analysis=analysis.to_dict(),
        enriched_metrics=enriched.to_dict(),
        run_successful=enriched.is_usable,
        failure_reason=_extract_failure_reason(ramp, analysis),
        fitness_score=enriched.fitness_score,
        total_run_duration_sec=round(total_run_duration_sec, 2),
        benchmark_duration_sec=ramp.total_duration_sec,
    )


def _get_git_hash() -> str:
    """Return the current short git commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
            cwd=REPO_ROOT,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _extract_failure_reason(ramp: RampResult, analysis: LogAnalysis) -> str:
    """Best-effort single-line failure summary for the CSV."""
    if analysis.has_oom:
        return "OOM: CUDA out of memory"
    if analysis.has_nccl_error:
        return "NCCL communication error"
    if analysis.error_classes:
        return f"Errors: {', '.join(sorted(analysis.error_classes))}"
    if ramp.all_failed():
        failed_reasons = [
            r.failure_reason for r in ramp.levels
            if r.failed and r.failure_reason
        ]
        return failed_reasons[0][:200] if failed_reasons else "All levels failed"
    return ""


# ===========================================================================
# 3.  ResultStorage
# ===========================================================================

class ResultStorage:
    """
    Writes RunRecords to local disk and optionally to DigitalOcean Spaces.

    Parameters
    ----------
    cfg : OceanTuneConfig
        Provides Spaces credentials and bucket config.
    results_dir : Path, optional
        Override the default storage/results/ directory.
    """

    def __init__(
        self,
        cfg: OceanTuneConfig,
        results_dir: Optional[Path] = None,
    ):
        self.cfg = cfg
        self.results_dir = results_dir or RESULTS_DIR
        self.runs_dir = self.results_dir / "runs"
        self.csv_path = self.results_dir / "runs.csv"

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        self._csv_fieldnames: Optional[List[str]] = None

    def save(self, record: RunRecord) -> None:
        """
        Save a RunRecord to disk (CSV + JSON) and upload to Spaces.

        Always attempts local write first. Spaces upload failure is
        logged as a warning but does not raise.
        """
        self._save_json(record)
        self._append_csv(record)
        self._upload_to_spaces(record)

        log_dict(
            log, "info", "Run record saved",
            run_id=record.run_id,
            fitness=round(record.fitness_score, 4),
            successful=record.run_successful,
        )

    def _save_json(self, record: RunRecord) -> None:
        """Write full run record as JSON to runs/{run_id}.json."""
        json_path = self.runs_dir / f"{record.run_id}.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(record.to_dict(), f, indent=2, default=str)
            log.debug(f"JSON saved: {json_path}")
        except (OSError, IOError) as exc:
            log.error(f"Failed to save JSON for run {record.run_id}: {exc}")

    def _append_csv(self, record: RunRecord) -> None:
        """Append a flat row to runs.csv. Creates with header if new."""
        row = record.to_csv_row()

        # Determine fieldnames from first row or from existing CSV header
        if self._csv_fieldnames is None:
            if self.csv_path.exists():
                with open(self.csv_path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    self._csv_fieldnames = reader.fieldnames or list(row.keys())
            else:
                self._csv_fieldnames = list(row.keys())

        # Add any new fields from this run (schema can grow)
        for k in row:
            if k not in self._csv_fieldnames:
                self._csv_fieldnames.append(k)

        write_header = not self.csv_path.exists()
        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=self._csv_fieldnames,
                    extrasaction="ignore",
                )
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            log.debug(f"CSV appended: {self.csv_path}")
        except (OSError, IOError) as exc:
            log.error(f"Failed to append CSV for run {record.run_id}: {exc}")

    def _upload_to_spaces(self, record: RunRecord) -> None:
        """
        Upload the JSON run record to DigitalOcean Spaces.

        Skips silently if credentials are not configured.
        """
        if not (self.cfg.spaces.access_key and self.cfg.spaces.secret_key):
            log.debug("Spaces credentials not set — skipping upload")
            return

        try:
            import boto3
            from botocore.exceptions import BotoCoreError, ClientError

            s3 = boto3.client(
                "s3",
                region_name=self.cfg.spaces.region,
                endpoint_url=self.cfg.spaces.endpoint_url,
                aws_access_key_id=self.cfg.spaces.access_key,
                aws_secret_access_key=self.cfg.spaces.secret_key,
            )

            key = (
                f"runs/{self.cfg.gpu_type}/{self.cfg.model_id.replace('/', '__')}/"
                f"{record.run_id}.json"
            )

            s3.put_object(
                Bucket=self.cfg.spaces.bucket,
                Key=key,
                Body=json.dumps(record.to_dict(), default=str).encode("utf-8"),
                ContentType="application/json",
            )
            log_dict(log, "info", "Uploaded to Spaces",
                     bucket=self.cfg.spaces.bucket, key=key)

        except ImportError:
            log.debug("boto3 not installed — skipping Spaces upload")
        except Exception as exc:
            log.warning(f"Spaces upload failed for {record.run_id}: {exc}")


# ===========================================================================
# 4.  ResultLoader
# ===========================================================================

class ResultLoader:
    """
    Reads historical run records back from disk for the optimiser.
    """

    @staticmethod
    def load_csv(csv_path: Path) -> List[dict]:
        """
        Load all run records from the CSV file.

        Returns an empty list if the file doesn't exist.
        Fields are returned as strings — caller handles type conversion.
        """
        if not csv_path.exists():
            return []
        try:
            rows = []
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(dict(row))
            log_dict(log, "info", "Loaded run history",
                     csv=str(csv_path), rows=len(rows))
            return rows
        except (OSError, IOError) as exc:
            log.warning(f"Could not load CSV {csv_path}: {exc}")
            return []

    @staticmethod
    def load_json(run_id: str, runs_dir: Optional[Path] = None) -> Optional[dict]:
        """
        Load a single full run record from its JSON file.

        Parameters
        ----------
        run_id : str
            The run fingerprint (VLLMFlags.fingerprint()).
        runs_dir : Path, optional
            Override default storage/results/runs/ directory.

        Returns
        -------
        dict or None if not found.
        """
        directory = runs_dir or RUNS_DIR
        json_path = directory / f"{run_id}.json"
        if not json_path.exists():
            return None
        try:
            with open(json_path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning(f"Could not load JSON for run {run_id}: {exc}")
            return None

    @staticmethod
    def get_best_runs(
        csv_path: Path,
        n: int = 10,
        metric: str = "fitness_score",
    ) -> List[dict]:
        """
        Return the top-n runs sorted by a metric (descending).

        Parameters
        ----------
        n : int
            Number of top runs to return.
        metric : str
            CSV column to sort by. Default: "fitness_score".
        """
        rows = ResultLoader.load_csv(csv_path)
        if not rows:
            return []

        def safe_float(row: dict, key: str) -> float:
            try:
                return float(row.get(key, 0) or 0)
            except (ValueError, TypeError):
                return 0.0

        sorted_rows = sorted(
            rows,
            key=lambda r: safe_float(r, metric),
            reverse=True,
        )
        return sorted_rows[:n]

    @staticmethod
    def get_failed_fingerprints(csv_path: Path) -> set:
        """
        Return the set of run_ids that failed.

        The optimiser uses this to avoid re-running known-bad configs.
        """
        rows = ResultLoader.load_csv(csv_path)
        return {
            row["run_id"]
            for row in rows
            if row.get("run_successful", "True").lower() == "false"
            and row.get("run_id")
        }