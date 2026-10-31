# Phase 6 & 7 System Baseline

Measured, unoptimized results vs. optimized results. 

| Model | Device | imgsz | Baseline FPS | Baseline Mean / p95 (ms) | Date | Hardware | Optimized FPS | Optimized p95 (ms) |
|---|---|---:|---:|---:|---|---|---:|---:|
| retail_people_clean (PyTorch) | CPU | 640 | 11.76 | 85.02 / 112.58 | 2026-10-24 | Intel64 Family 6 Model 142 | N/A | N/A |
| retail_people_clean (ONNX INT8) | CPU | 640 | 15.09 | 66.27 / 90.50 | 2026-10-31 | Intel64 Family 6 Model 142 | 19.10 | 60.05 |
| retail_people_clean (TensorRT FP16) | NVIDIA T4 GPU | 640 | N/A | N/A | 2026-10-31 | Kaggle Tesla T4 | 183.56 | 6.39 |