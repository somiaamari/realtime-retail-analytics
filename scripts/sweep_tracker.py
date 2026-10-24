"""Tune ByteTracker parameters against sequence-level MOT annotations."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from vision_pipeline.config import load_config
from vision_pipeline.evaluation.mot_io import read_mot
from vision_pipeline.evaluation.report import collect_environment, set_reproducible_seed
from vision_pipeline.evaluation.sweep import sweep_tracker
from vision_pipeline.evaluation.tracking_eval import DetectionCache
from vision_pipeline.logging_utils import setup_logging


def _resolve(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def main(argv: list[str] | None = None) -> int:
    """Load benchmark sequences and run the configured deterministic sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase6/sweep"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--best-config",
        type=Path,
        default=Path("configs/tracker.tuned.yaml"),
    )
    args = parser.parse_args(argv)
    setup_logging()
    set_reproducible_seed(args.seed)
    with args.config.open(encoding="utf-8") as stream:
        evaluation_config: dict[str, Any] = yaml.safe_load(stream) or {}
    sequence_config = evaluation_config.get("sequences", [])
    split = evaluation_config.get("tracker_sweep", {})
    if not split:
        raise SystemExit(
            "configs/eval.yaml must define tracker_sweep train/validation split"
        )
    root = args.config.resolve().parent.parent
    app_config = load_config(
        evaluation_config.get("app_config", "configs/default.yaml")
    )
    cache = DetectionCache(
        _resolve(
            root, evaluation_config.get("detection_cache", "data/benchmarks/cache")
        )
    )
    sequences = {}
    for item in sequence_config:
        name = item["name"]
        gt = read_mot(_resolve(root, item["ground_truth"]), ground_truth=True)
        cache_path = cache.cache_path(
            name,
            weights_path=app_config.model.weights_path,
            imgsz=app_config.model.imgsz,
            conf_threshold=app_config.model.conf_threshold,
            iou_threshold=app_config.model.iou_threshold,
        )
        if not cache_path.is_file():
            raise FileNotFoundError(
                "Detection cache is missing for "
                f"{name}; run tracking evaluation first: {cache_path}"
            )
        sequences[name] = (gt, cache.load(cache_path))
    result = sweep_tracker(
        sequences,
        train_sequences=list(split.get("train_sequences", [])),
        validation_sequences=list(split.get("validation_sequences", [])),
        parameter_grid=split.get("parameter_grid", {}),
        output_directory=args.output_dir,
        workers=args.workers,
        random_trials=split.get("random_trials"),
        seed=int(split.get("seed", args.seed)),
    )
    args.best_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(result["artifacts"]["config"], args.best_config)
    result["artifacts"]["best_config"] = str(args.best_config)
    environment = collect_environment(
        repository_root=root,
        config_paths=[args.config],
    )
    environment["seed"] = int(split.get("seed", args.seed))
    result_path = args.output_dir / "sweep.json"
    result_path.write_text(
        json.dumps(
            {"environment": environment, "results": result},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    result["artifacts"]["result_json"] = str(result_path)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
