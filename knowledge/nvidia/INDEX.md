# NVIDIA knowledge pack

kind: index
scope: nvidia
title: NVIDIA (Hopper / Blackwell) optimization levers

## Route here

1. Attention backend selection (FLASH_ATTN / FLASHINFER / TRITON_ATTN)
2. FP8 KV cache and prefix caching
3. NVFP4 / FP8 weight paths when model supports them
4. Fusion candidates: residual+RMSNorm, SwiGLU, QK-Norm+RoPE

## GPU SKUs

- H100 / H200 — Hopper; strong FlashAttention-3 / FlashInfer
- B300 — Blackwell; prefer FA3/FlashInfer and NVFP4 when available

## Verify

CapabilityDetector + short ramp; never enable speculative decoding without a known draft model.
