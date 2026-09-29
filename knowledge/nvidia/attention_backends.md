# NVIDIA attention backend levers

kind: lever
scope: nvidia
title: Attention backend A/B on Hopper/Blackwell

## Route here

When profiler shows attention ≥ ~25% or Stage 2 kernel category is unexplored.

## What to change

| Backend | Prefer when |
|---------|-------------|
| FLASH_ATTN | Default Hopper; GQA models; stable baseline |
| FLASHINFER | Paged attention + GQA; often +5–15% vs FA when available |
| TRITON_ATTN | Experimental / custom; validate SNR before KEEP |

## Verify

1. CapabilityDetector lists backend as supported
2. Stage 2 MeasurementGate KEEP vs incumbent
3. Watch startup errors — FlashInfer missing → constrain planner

## Failure modes

- FlashInfer import / JIT failure → revert to FLASH_ATTN
- MLA models: keep `block_size=1`
