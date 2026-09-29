# NVIDIA memory & KV levers

kind: lever
scope: nvidia
title: FP8 KV, prefix cache, utilization

## What to change

- `kv_cache_dtype=fp8` when VRAM-bound or concurrency saturates early
- `enable_prefix_caching=true` for multi-turn / shared-prefix workloads
- Raise `gpu_memory_utilization` only after FP8 KV; never jump past prior OOM util

## Expected magnitude

- FP8 KV: +10–30% concurrency headroom typical on H100/H200
- Prefix cache: large win on chat; near-zero on unique long prompts

## Verify

Operating-point sweep after KEEP; sediment peak concurrency into Recipe KB.
