# OceanTune common methodology

kind: index
scope: all_vendors
title: Common optimization methodology

## Route here

Use these levers before vendor-specific packs. Always confirm with MeasurementGate.

## Iron rules

1. **Measure, don't claim** — agents propose; fitness / SNR decide KEEP.
2. **One axis per trial** — change one strategy or flag family at a time.
3. **SNR then speedup** — numerical parity (≈30 dB) is a pre-filter; KEEP needs noise-adjusted speedup.
4. **Microbench ≠ serving** — Stage 4 wins require E2E rebench under winner flags.

## Prefill vs decode

- Prefill-heavy: watch attention + GEMM; try chunked prefill, larger `max_num_batched_tokens`.
- Decode-heavy: watch KV bandwidth + scheduler; try FP8 KV, prefix caching, attention backend A/B.
