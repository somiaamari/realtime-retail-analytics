# Phase 7 Model Optimization Benchmark

- **Local Hardware/Software:** Windows-11, Python 3.12.10 (CPU Baseline)
- **Cloud Hardware:** Kaggle NVIDIA Tesla T4 (TensorRT Benchmark)
- Latency measures detector inference only; video decode and drawing are excluded.
- Accuracy parity was verified against the PyTorch baseline (`reports/phase7/parity.md`).

| Variant | Mean ms | p50 ms | p95 ms | p99 ms | FPS | Speedup | Size MB | Accuracy Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PyTorch FP32 (CPU) | 66.27 | 61.52 | 90.50 | 114.91 | 15.09 | 1.00x | 6.51 | Baseline |
| ONNX FP32 (CPU) | 64.55 | 62.98 | 80.66 | 83.94 | 15.49 | 1.03x | 12.77 | Passed (Parity < 0.01) |
| ONNX FP16 (CPU) | 61.84 | 61.86 | 71.11 | 76.31 | 16.17 | 1.07x | 6.41 | Passed (Parity < 0.01) |
| ONNX INT8 (CPU) | 52.35 | 51.73 | 60.05 | 62.34 | 19.10 | 1.27x | 6.16 | Passed (Parity < 0.01) |
| **TensorRT FP16 (T4 GPU)** | **5.45** | **5.20** | **6.39** | **7.50** | **183.56** | **12.16x** | 58.52 | **Passed (Parity < 0.01)** |

## Decision

- **Recommended for CPU deployment:** ONNX INT8 (1.27x speedup, smallest footprint).
- **Recommended for GPU deployment:** TensorRT FP16 (12.16x speedup, true real-time performance at 183 FPS).
- **Note:** TensorRT requires an environment-specific setup (CUDA and TensorRT runtime). The `.engine` file is strictly hardware-locked to the GPU architecture it was built on (NVIDIA T4).
