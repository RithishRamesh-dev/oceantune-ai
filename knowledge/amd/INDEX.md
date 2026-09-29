# AMD knowledge pack

kind: index
scope: amd
title: AMD Instinct (MI300X / MI325X / MI350X) optimization levers

## Route here

1. ROCM_FLASH / AITER attention paths
2. AITER fused MoE / RMSNorm / SiLU when flags exist in Stage 2 search space
3. RCCL / TP stability — change TP separately from quantization
4. Fusion: residual+RMSNorm and SwiGLU are high-value on CDNA

## GPU SKUs

- MI300X / MI325X — CDNA3; prioritize AITER env toggles
- MI350X — newer stack; re-validate FlashAttention / AITER versions

## Verify

CapabilityDetector vendor filter; MeasurementGate; avoid FLASHINFER unless explicitly supported on the image.
