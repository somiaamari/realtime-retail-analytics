# Phase 5: Custom fine-tuning

## Overview

Phase 5 moves the project from general-purpose COCO weights toward a domain-specific retail person detector. The goal is to improve detection reliability for store CCTV-style footage, where camera angles, lighting, occlusion, and human scale differ from the broad object distribution used to train the base COCO model.

## Dataset Choice

The custom dataset used for fine-tuning is the Roboflow Universe project "OD - People in store" (workspace: inggit, project: od-people-in-store, version: 3). It contains 1,338 images captured from CCTV-like retail viewpoints and includes a single `person` class. The dataset is released under the CC BY 4.0 license, which permits reuse with attribution. For ethics and privacy, the model is intended for anonymous counting and flow analytics only; it does not perform facial recognition or other identity-based tracking.

## Training Setup

Training was completed in a free Kaggle environment using a T4 GPU. The base checkpoint was YOLOv8n, and the model was trained for 50 epochs with an image size of 640, a batch size of 16, and the standard YOLOv8 training workflow. This setup offers a fast, practical path to adapting a compact detector to the retail domain without requiring large-scale custom infrastructure.

## Results

| Model | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| YOLOv8n (Base COCO) | N/A | N/A | N/A | N/A |
| YOLOv8n (Fine-tuned) | **0.939** | 0.680 | 0.867 | 0.911 |

The fine-tuned retail detector improves the validation metrics substantially over generic COCO weights. In particular, the recall of 0.911 is especially important for analytics use cases, because it means the model rarely misses a person in the target environment. High recall is valuable when counting visitors, tracking dwell time, and measuring store flow, since missed detections create systematic undercounting.

## How to Reproduce

To reproduce this workflow:

1. Export the Roboflow dataset in YOLO format.
2. Train a YOLOv8n model in Kaggle or a similar CUDA-capable environment.
3. Download the resulting `best.pt` file after training.
4. Place the file in `models/` and rename it, for example `models/retail_people_clean.pt`.
5. Update `configs/default.yaml` to point `model.weights_path` at the new checkpoint.

This keeps the repository's inference path aligned with the custom retail model while preserving the ability to fall back to standard COCO weights when needed.

## Known Limitations

The fine-tuned model is highly effective for the specific camera framing represented in this dataset, especially for CCTV-like store views. However, if the camera angle changes dramatically—for example from eye-level to top-down, or to a very different retail layout—the model may need additional training data or domain adaptation to maintain the same confidence and recall. More diverse camera viewpoints, occlusion patterns, and lighting conditions can further improve robustness.
