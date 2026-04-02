#!/usr/bin/env python3
"""
oceantune.py
------------
OceanTune AI — Main CLI entry point.

Usage
-----
    python3 oceantune.py --help
    python3 oceantune.py run --model mistral --gpu H100
    python3 oceantune.py run --config configs/oceantune.yaml
    python3 oceantune.py validate-config
"""

import sys
import os
from pathlib import Path

# Ensure repo root is on the import path regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

import click
from dotenv import load_dotenv

load_dotenv()  # load .env before any config is read

from core.logger import get_logger
from core.config import load_config, CONFIGS_DIR

log = get_logger("oceantune.cli")


# ─────────────────────────────────────────────────────────────────────────────
# CLI group
# ─────────────────────────────────────────────────────────────────────────────

@click.group()
@click.version_option("0.1.0", prog_name="OceanTune AI")
def cli():
    """OceanTune AI — vLLM inference optimisation system."""


# ─────────────────────────────────────────────────────────────────────────────
# validate-config
# ─────────────────────────────────────────────────────────────────────────────

@cli.command("validate-config")
@click.option(
    "--config", "-c",
    default=None,
    help="Path to YAML config file. Defaults to configs/oceantune.yaml.",
)
def validate_config(config):
    """Load and validate the configuration, then print a summary."""
    path = Path(config) if config else None
    try:
        cfg = load_config(override_path=path)
    except (ValueError, FileNotFoundError) as exc:
        click.echo(f"❌  Config validation FAILED: {exc}", err=True)
        sys.exit(1)

    click.echo("\n✅  Config valid\n")
    click.echo(f"  Model ID   : {cfg.model_id}")
    click.echo(f"  GPU type   : {cfg.gpu_type}")
    click.echo(f"  Strategy   : {cfg.optimiser.strategy}")
    click.echo(f"  Metric     : {cfg.optimiser.primary_metric}")
    click.echo(f"  Generations: {cfg.optimiser.generations}")
    click.echo(f"  Pop. size  : {cfg.optimiser.population_size}")
    click.echo(f"  Concurrency: {cfg.benchmark.concurrency_levels}")
    click.echo(
        f"  Contexts   : {len(cfg.context_configs)} pairs "
        f"({cfg.context_configs})"
    )
    click.echo(f"\n  Spaces bucket : {cfg.spaces.bucket} ({cfg.spaces.region})")
    click.echo(f"  vLLM port     : {cfg.vllm.port}")
    has_hf = bool(cfg.hf_token)
    has_spaces = bool(cfg.spaces.access_key and cfg.spaces.secret_key)
    click.echo(f"\n  HF_TOKEN set       : {'✅' if has_hf else '⚠️  NOT SET'}")
    click.echo(f"  DO_SPACES_KEY set  : {'✅' if has_spaces else '⚠️  NOT SET'}")


# ─────────────────────────────────────────────────────────────────────────────
# run  (wired to ControllerAgent — fully operational in Step 8)
# ─────────────────────────────────────────────────────────────────────────────

@cli.command("run")
@click.option("--config", "-c", default=None, help="Path to YAML config.")
@click.option("--model", "-m", default=None, help="Override model_id.")
@click.option("--gpu", "-g", default=None, help="Override gpu_type.")
@click.option(
    "--strategy", "-s",
    default=None,
    type=click.Choice(["evolutionary", "grid", "random", "bayesian"]),
    help="Override optimisation strategy.",
)
@click.option("--dry-run", is_flag=True, help="Validate config and exit.")
def run(config, model, gpu, strategy, dry_run):
    """Run the full OceanTune optimisation pipeline."""
    if model:
        os.environ["OCEANTUNE_MODEL_ID"] = model
    if gpu:
        os.environ["OCEANTUNE_GPU_TYPE"] = gpu
    if strategy:
        os.environ["OCEANTUNE_STRATEGY"] = strategy

    path = Path(config) if config else None
    try:
        cfg = load_config(override_path=path)
    except (ValueError, FileNotFoundError) as exc:
        click.echo(f"❌  Config error: {exc}", err=True)
        sys.exit(1)

    if dry_run:
        click.echo("✅  Dry run — config valid. Exiting.")
        return

    log.info(
        "Starting OceanTune pipeline",
        model=cfg.model_id,
        gpu=cfg.gpu_type,
        strategy=cfg.optimiser.strategy,
    )

    from agents.controller_agent import ControllerAgent
    agent = ControllerAgent()
    agent.run()


# ─────────────────────────────────────────────────────────────────────────────
# info — print system information
# ─────────────────────────────────────────────────────────────────────────────

@cli.command("info")
def info():
    """Print system and environment information."""
    import platform
    click.echo(f"\nOceanTune AI v0.1.0")
    click.echo(f"Python  : {sys.version.split()[0]}")
    click.echo(f"Platform: {platform.platform()}")
    click.echo(f"Repo    : {Path(__file__).resolve().parent}")
    click.echo(f"Configs : {CONFIGS_DIR}")

    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                click.echo(f"GPU     : {line.strip()}")
        else:
            click.echo("GPU     : nvidia-smi not available")
    except Exception:
        click.echo("GPU     : (could not query nvidia-smi)")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
