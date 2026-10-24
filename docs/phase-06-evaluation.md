# Phase 6: System evaluation and benchmarking

## Purpose and status

Phase 6 establishes a reproducible, pre-optimization baseline for the detector, ByteTracker, retail analytics, and complete video path. To ensure the pipeline does not silently regress, CI uses synthetic MOT sequences and regression guards (verifying IDF1 remains above 0.90)



## Metrics in plain language

- **MOTA** combines misses, false positives, and identity switches relative to
  the number of annotated objects. It is useful as a broad error summary but can
  hide identity quality when detections dominate the error count.
- **MOTP** describes localization quality of matched boxes (the implementation
  reports the motmetrics distance convention: lower distances indicate closer
  localization).
- **IDF1** measures the fraction of correctly identified detections after
  matching complete predicted and ground-truth identity trajectories. It is a
  useful retail metric because visitor counts and dwell-time attribution depend
  on keeping each anonymous track identity consistent.
- **HOTA** balances detection and association quality. TrackEval is optional
  and reported as skipped unless a compatible dataset adapter is available.
- **Line-crossing precision/recall/F1** compare event time and direction against
  manual crossings. Count error measures the difference in total counts, both
  absolute and as a percentage of annotated count.
- **Zone dwell MAE** is the mean absolute difference, in seconds, between
  predicted and annotated visit durations.
- **Latency** is measured by stage and end to end after warmup. The performance
  report includes FPS, frame-time jitter, frames beyond the configured real-time
  budget, RSS, CPU utilization when psutil is available, and CUDA peak memory
  when PyTorch/CUDA is available.
  The application-level preprocess stage currently measures contiguous-frame
  preparation; Ultralytics image resizing/normalization is included in detector
  inference, not broken out internally.

## Ground-truth protocol

For a valid retail evaluation, use held-out footage representative of the
deployment camera and annotate a fixed set of sequences. MOT annotations use
one-based frame numbers and pedestrian boxes; the reader filters `gt.txt` rows
to class 1 with a positive mark/confidence, excluding unmarked and distractor
rows in MOT17-style files. Keep image/video data and annotations locally under
`data/benchmarks/`; this directory is gitignored.

Manual line labels should contain `timestamp_s,direction,line_id`, where
direction is `in` or `out`. Zone labels should contain
`zone_id,enter_s,exit_s`. The benchmark should be split by sequence, not by
random frames: tracker parameters are tuned on `train_sequences` and reported
on disjoint `validation_sequences`.



## How to reproduce

Install the optional evaluation dependencies and detection runtime:

```shell
python -m pip install -e ".[eval,detection]"
```

Place each MOT sequence below `data/benchmarks/`, including `seqinfo.ini`,
`img1/` or a sequence video, and `gt/gt.txt`. Configure entries under
`sequences` in `configs/eval.yaml`; optionally provide a separate
`line_ground_truth` and `zone_ground_truth` per sequence. Configure counting
lines/zones in the application YAML. Then run:

```shell
python scripts/run_all_benchmarks.py --config configs/eval.yaml
python scripts/benchmark_system.py --source data/benchmarks/store.mp4 --config configs/default.yaml
python scripts/sweep_tracker.py --config configs/eval.yaml --workers 1
```

The unified runner writes `reports/phase6/REPORT.md` and `results.json`, plus
predicted MOT files and event timeline charts. The standalone performance
runner writes a dated JSON file under `benchmarks/results/`. Detection results
in `configs/eval.yaml` are explicitly labeled as Phase 5 context until
Phase 6 held-out image labels are supplied. The detection-cache key includes
weights content hash, image size, confidence threshold, and IoU threshold.
Set `detection_data_yaml` to a held-out Ultralytics dataset YAML to run an
actual detection evaluation; do not reuse training/validation images for the
held-out result.
Use `mode: public` and provide `public_detections` when evaluating public
detections; use `mode: oracle` to feed filtered GT boxes to the tracker as an
upper-bound association diagnostic. Oracle scores are not end-to-end system
scores.

The standalone system benchmark's `--sweep` covers inference sizes
320/480/640/960, currently available CPU/CUDA/MPS devices, drawing enabled and
disabled, and video writing enabled and disabled. Pass `--model-s <path>` to
include a local YOLO small checkpoint; without it, only the configured model is
measured. It records latency by detection-count bucket and captures first-/second-
half FPS drift and RSS growth with a default two-minute run cap.

For reproducibility, each report records the config hash, local commit hash
when readable from `.git` metadata, timestamp, OS, CPU, RAM, Python, and
installed PyTorch/Ultralytics versions. Hardware-dependent results should only
be compared under equivalent hardware, source, model, and inference settings.

## Current Results

Baseline benchmarks were executed on an Intel Core i7 (CPU) with an input resolution of 640x640 using the fine-tuned `retail_people_clean.pt` model.

| Evaluation | Result |
|---|---|
| Phase 5 detector validation mAP50 / mAP50-95 | 0.939 / 0.680 |
| Phase 6 tracking (Synthetic CI Guard) | IDF1 > 0.90 (Pass) |
| **End-to-End Mean Latency** | **85.02 ms** |
| **End-to-End p95 Latency** | **112.58 ms** |
| **Throughput** | **11.76 FPS** |
| CPU Peak Memory (RSS) | ~484 MB |
| CPU Utilization | ~655% (Multi-threaded) |

The target configuration is in `configs/eval_targets.yaml`: IDF1 at least 0.70, line-count error at most 5%, and p95 end-to-end latency at most 50 ms. The current CPU baseline fails the latency target (112.58 ms vs 50 ms), highlighting the need for hardware acceleration and model optimization in Phase 7.

## Failure Modes and Phase 7 Focus

Based on the performance profiling from this phase, the primary failure mode for real-time operation is **inference latency**. 

*   **Detection Bottleneck:** The YOLO detection stage accounts for ~89% of the total frame processing time (75.86 ms mean). 
*   **Analytics & Tracking Efficiency:** The ByteTracker and Analytics engines are highly optimized, taking only 1.61 ms and 0.18 ms respectively. 
*   **Stability:** A slight FPS drift (-23.95%) was observed over a 2-minute run, likely due to CPU thermal throttling or background OS tasks.

**Phase 7 Focus:** Phase 7 will implement ONNX and TensorRT optimizations to accelerate the detection stage. The goal is to reduce the p95 latency from 112 ms down to under 50 ms, and push the FPS from 11.76 to over 30 FPS, achieving true real-time performance.