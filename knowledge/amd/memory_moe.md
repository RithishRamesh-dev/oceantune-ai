# AMD memory & MoE levers

kind: lever
scope: amd
title: VRAM, MoE, and scheduling on Instinct

## What to change

- `gpu_memory_utilization` conservative first (0.85–0.90) on large MoE
- FP8 KV when decode concurrency saturates
- AITER fused MoE kernels when search space exposes them
- `max_num_seqs` / `max_num_batched_tokens` after memory is stable

## Failure modes

- Startup timeout after aggressive EP/TP + quantization combo
- RCCL hangs — reduce TP, disable experimental dispatch
