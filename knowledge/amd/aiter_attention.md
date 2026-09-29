# AMD AITER / ROCm attention levers

kind: lever
scope: amd
title: AITER and ROCM_FLASH attention

## What to change

| Lever | Prefer when |
|-------|-------------|
| `attention_backend=ROCM_FLASH` | Default AMD serving path |
| AITER fused attention / MoE flags | MoE models; Stage 2 amd_kernel category |
| Disable experimental all2all | NCCL/RCCL instability in history |

## Verify

One flag family per trial. OOM → lower util or enable FP8 KV before raising TP.

## Failure modes

- Mixing NVIDIA-only backends (FLASHINFER) on ROCm images
- Raising TP while enabling experimental kernels in one step
