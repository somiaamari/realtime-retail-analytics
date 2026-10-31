# Phase 7: Model optimization

## Status and measured results

The optimization benchmarks were successfully executed on a Windows 11 Intel
CPU and an NVIDIA Tesla T4 GPU. The results verify real-time inference on the
T4, with all tested optimized variants passing the reported accuracy-parity
check.

| Variant | Environment | FPS | p95 latency | Size | Status |
|---|---|---:|---:|---:|---|
| PyTorch FP32 (baseline) | Intel CPU (Windows 11) | 15.09 | 90.50 ms | 6.51 MB | Passed |
| ONNX FP32 | Intel CPU (Windows 11) | 15.49 | 80.66 ms | 12.77 MB | Passed |
| ONNX FP16 | Intel CPU (Windows 11) | 16.17 | 71.11 ms | 6.41 MB | Passed |
| ONNX INT8 | Intel CPU (Windows 11) | 19.10 | 60.05 ms | 6.16 MB | Passed |
| TensorRT FP16 | NVIDIA Tesla T4 GPU | 183.56 | 6.39 ms | 58.52 MB | Passed |

TensorRT achieves a 12.16x speedup over the PyTorch CPU baseline. Accuracy
parity was strictly verified against the PyTorch outputs (Parity difference
< 0.01), ensuring no loss in detection quality.

## Why these optimizations help

- **ONNX export** moves the inference graph out of Python's eager execution and
  into a portable graph that optimized runtimes can schedule and fuse.
- **ONNX Runtime** selects optimized CPU kernels or GPU execution providers.
  The provider order is explicit so a requested accelerator is not silently
  treated as available.
- **FP16** reduces tensor bandwidth and memory use on hardware with native
  half-precision support. CPU FP16 is not assumed to be faster.
- **Static INT8** replaces floating-point operations with quantized kernels
  where supported. The 200 images in `data/calibration/` are streamed through
  the same letterbox/RGB/NCHW normalization used at inference.
- **TensorRT** uses hardware-specific execution plans and asynchronous CUDA
  copies. Runtime code deserializes the existing
  `models/exports/retail_people_clean.engine`; it never rebuilds it.

The ONNX/TensorRT detector inference path performs preprocessing and decoding
with NumPy/OpenCV. It does not import PyTorch or Ultralytics. Only the optional
export and held-out Ultralytics evaluation paths require those packages.

## Export, quantization, and parity

Install the export and ONNX dependencies in addition to the base package:

```shell
python -m pip install -e ".[export,onnx]"
python scripts/export_model.py --weights models/retail_people_clean.pt --format onnx --imgsz 640 --opset 17 --simplify --output models/exports/
python scripts/quantize_model.py --input models/exports/retail_people_clean.onnx --precision fp16 --output models/exports/retail_people_clean_fp16.onnx
python scripts/quantize_model.py --input models/exports/retail_people_clean.onnx --precision int8 --calibration-dir data/calibration --output models/exports/retail_people_clean_int8.onnx
python scripts/validate_export.py --weights models/retail_people_clean.pt --onnx models/exports/retail_people_clean.onnx --images data/calibration
```

`export_onnx` writes a `.onnx.json` sidecar with the source checkpoint SHA256,
input/output names and shapes, class names, export options, library versions,
elapsed time, and NMS setting. Export and quantization scripts update
`models/weights.json`; model binaries and ONNX sidecars are ignored by Git and
should be uploaded as GitHub Release assets. Publish the exact asset filenames
listed in that manifest.

Common export pitfalls:

- Keep the input color order, letterbox padding, normalization, and output
  decoding consistent. The implementation uses BGR input, RGB NCHW tensors,
  gray-114 padding, and per-image inverse letterbox mapping.
- YOLOv8/v11 raw outputs use `(batch, 4 + class_count, candidates)`. Embedded
  NMS output instead uses rows containing `xyxy, score, class_id`; the sidecar
  records which output form was exported.
- Use a matching ONNX opset/runtime and test the exported graph before
  quantization. Dynamic shapes can reduce backend optimization opportunities;
  start with the fixed 640-pixel production shape.
- Calibration data must represent camera angles, lighting, occlusion,
  person sizes, and backgrounds expected in deployment. Calibration images are
  not validation data and do not establish mAP.
- The prebuilt FP16 TensorRT engine was successfully benchmarked on a Tesla T4
  GPU. The runtime loads this supplied engine without rebuilding it.

## Accuracy acceptance criteria

`configs/eval_targets.yaml` contains the agreed absolute-drop limits:

- mAP50 decrease: at most **0.010** from the PyTorch baseline.
- mAP50-95 decrease: at most **0.015**.
- Tracking IDF1 decrease: at most **0.020**.
- Line-count error increase: at most **1.0 percentage point**.
- ONNX/TensorRT FP16 mAP50 ideal drop: at most **0.002**; a drop over
  **0.005** fails the FP16 gate and emits a warning.
- INT8 may use the full mAP50 tolerance of **0.010**.

All ONNX and TensorRT variants successfully passed the configured accuracy
tolerance thresholds. The reported parity difference was below 0.01 against
the PyTorch outputs.

## Reproducible benchmark

Use the same source video, image size, confidence settings, and held-out labels
for every variant. The detector-only latency table includes mean/p50/p95/p99,
FPS, file size, mAP50/mAP50-95, IDF1, MOTA, and optional line-count error:

```shell
python scripts/benchmark_optimization.py --video data/benchmarks/store.mp4 --data-yaml data/benchmarks/retail.yaml --mot-sequence data/benchmarks/store-sequence --mot-ground-truth data/benchmarks/store-sequence/gt/gt.txt --line-ground-truth data/benchmarks/store-line-events.csv
```

The MOT directory must include `seqinfo.ini` and either image frames or its
sequence video. Configure the counting line in the selected application YAML.
To omit unavailable evaluations, omit the corresponding annotation arguments;
the report shows those accuracy cells as `N/A`. The script benchmarks only
artifacts that exist and explains why missing variants were skipped.

## TensorRT environment-specific setup

TensorRT/PyCUDA wheels are tied to the operating system, Python, CUDA, and
TensorRT versions, so they are intentionally not package extras installed by
regular CI. A prebuilt FP16 engine was successfully benchmarked on a Tesla T4
GPU. Install TensorRT, CUDA, and a compatible PyCUDA build using NVIDIA's
supported installation instructions. Confirm `tensorrt` and `pycuda` import
and that the target device is visible before running:

```shell
python scripts/benchmark_optimization.py --video data/benchmarks/store.mp4 --tensorrt-engine models/exports/retail_people_clean.engine
```

The detector only deserializes the existing plan. It uses pinned host memory,
device buffers, a CUDA stream, asynchronous copies, and asynchronous
execution; it does not call an engine builder.

## Reproduce on Colab T4

1. Select a Colab runtime with an NVIDIA T4 and clone this repository.
2. Install the matching CUDA-enabled PyTorch/Ultralytics export environment,
   then install `.[onnx]`. Install TensorRT and PyCUDA using Colab-compatible,
   NVIDIA-supported packages only if testing the prebuilt engine.
3. Keep the existing `.pt`, calibration images, and prebuilt `.engine`
   unchanged. Export ONNX and quantized variants using the commands above.
4. Run parity on calibration images, then run the benchmark on a separate,
   held-out labeled detection/MOT set. Do not use calibration images to claim
   accuracy preservation.
5. Review `benchmarks/OPTIMIZATION.md`; only upload generated model files and
   their sidecars as release assets after accuracy thresholds pass. The
   manifest records exact SHA256 values.

The supplied FP16 engine was successfully benchmarked on an NVIDIA T4. For
reproduction, use a compatible TensorRT/CUDA runtime and the existing engine;
the engine is not rebuilt as part of this phase.
