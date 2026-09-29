# OceanTune serving patches

Versioned manifests under `vllm/<name>/manifest.json` (and future `sglang/`).

Selected by `core/serving_patches.select_patches` using detected framework version
and env gates (`OCEANTUNE_PATCH_*` or `enabled_by_default`).

OceanTune does **not** vendor Hyperloom patch diffs. Optional `.patch` files may
live beside a manifest for operators who maintain private serving forks.
