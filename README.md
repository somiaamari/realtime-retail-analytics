# Real-Time Smart Retail Video Analytics Pipeline

[![CI](https://github.com/somiaamari/realtime-retail-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/somiaamari/realtime-retail-analytics/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

A production-oriented foundation for a real-time smart retail video analytics pipeline. The planned system will combine video ingestion, YOLO object detection, ByteTrack multi-object tracking, retail analytics, and a FastAPI/React dashboard, with ONNX and TensorRT optimization as later milestones. This phase establishes a reproducible Python package, configuration, logging, quality tooling, and CI; it intentionally contains no detection or model logic.

## Planned architecture

```mermaid
flowchart LR
    A[Video Source] --> B[Detector: YOLO]
    B --> C[Tracker: ByteTrack]
    C --> D[Analytics]
    D --> E[API: FastAPI]
    E --> F[Dashboard: React]
```

## Roadmap

- [x] Phase 1: Foundation
- [ ] Phase 2: Video I/O & baseline detection
- [ ] Phase 3: Tracking
- [ ] Phase 4: Analytics engine
- [ ] Phase 5: Dataset & fine-tuning
- [ ] Phase 6: Evaluation & benchmarking
- [ ] Phase 7: Model optimization (ONNX/TensorRT)
- [ ] Phase 8: Real-time streaming pipeline
- [ ] Phase 9: Backend API
- [ ] Phase 10: Dashboard
- [ ] Phase 11: Docker/tests/CI-CD
- [ ] Phase 12: Docs/demo/release

## Quickstart

```shell
git clone https://github.com/somiaamari/realtime-retail-analytics.git
cd realtime-retail-analytics
python -m venv .venv
```

Activate the environment (`.venv\Scripts\activate` on Windows, or
`source .venv/bin/activate` on Linux/macOS), then install and test:

```shell
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
pytest --cov=vision_pipeline --cov-report=term-missing
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for local quality checks and contribution
guidance. The package version is available as `vision_pipeline.__version__`.
