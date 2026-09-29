# NVIDIA Triton / CUTLASS notes

kind: lever
scope: nvidia
title: Triton on NVIDIA — Stage 4 authoring notes

## What to change

- Prefer Triton `@triton.jit` for residual+RMSNorm and SiluAndMul fusions
- Use torch reference for CorrectnessFirewall; SNR ≥ 30 dB before KEEP
- CUTLASS / torch.compile paths: treat as research; E2E via shadow PYTHONPATH

## Failure modes

- Autotune configs that pass microbench but regress TTFT under serving load
- Forgetting contiguous layouts → silent SNR failure
