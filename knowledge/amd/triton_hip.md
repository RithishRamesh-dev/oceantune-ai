# AMD Triton / HIP notes

kind: lever
scope: amd
title: Triton on AMD — deltas vs NVIDIA

## Route here

Stage 4 kernel authoring on MI300X-class GPUs.

## What changes vs NVIDIA

- Wavefront 64 vs warp 32 — tile sizes that win on CUDA often lose on CDNA
- Prefer MFMA-friendly shapes; avoid bank-conflict-heavy LDS patterns
- HIP vs CUDA traps: device pointers, stream defaults, cooperative groups gaps

## Verify

SNR ≥ 30 dB vs PyTorch reference on target shapes; then E2E shadow rebench.

## Expected magnitude

Fused residual+RMSNorm: often 1.05–1.20× kernel; E2E usually lower — do not KEEP on microbench alone.
