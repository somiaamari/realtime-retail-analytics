"""Measure numerical and detection parity between PyTorch and ONNX."""

import argparse
import logging
import sys
from pathlib import Path

import cv2

from vision_pipeline.config import DetectorConfig, ModelConfig
from vision_pipeline.detection.onnx_detector import OnnxDetector
from vision_pipeline.detection.yolo_detector import YoloDetector
from vision_pipeline.logging_utils import setup_logging
from vision_pipeline.optimization.validate import compare_outputs

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the ONNX validation parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", type=Path, default=Path("models/retail_people_clean.pt")
    )
    parser.add_argument(
        "--onnx", type=Path, default=Path("models/exports/retail_people_clean.onnx")
    )
    parser.add_argument("--images", type=Path, default=Path("data/calibration"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "int8"],
        default="fp32",
        help="Precision recorded in the ONNX metadata sidecar",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("reports/phase7/parity.md"))
    return parser


def run(args: argparse.Namespace) -> Path:
    """Run parity over actual image files and persist a Markdown report."""
    image_paths = sorted(
        path
        for path in args.images.iterdir()
        if path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".bmp"}
    )[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No validation images found in {args.images}")
    config = ModelConfig(weights_path=args.weights, imgsz=args.imgsz)
    torch_detector = YoloDetector(config)
    onnx_detector = OnnxDetector(
        config,
        DetectorConfig(
            backend="onnx",
            precision=args.precision,
            model_path=args.onnx,
        ),
    )
    images = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise OSError(f"Could not read validation image {image_path}")
        images.append(image)
    result = compare_outputs(torch_detector, onnx_detector, images)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "# ONNX export parity\n\n"
        f"- Weights: `{args.weights}`\n"
        f"- ONNX: `{args.onnx}`\n"
        f"- Precision: {args.precision}\n"
        f"- Validation images: {result['images']}\n"
        f"- Maximum absolute raw-output difference: "
        f"{result['max_absolute_difference']:.8f}\n"
        f"- Mean absolute raw-output difference: "
        f"{result['mean_absolute_difference']:.8f}\n"
        f"- Detection parity (IoU >= 0.95 and score difference < 0.01): "
        f"{result['detection_parity']:.4%} "
        f"({result['parity_matches']} matched; "
        f"{result['torch_detections']} PyTorch / "
        f"{result['onnx_detections']} ONNX detections)\n",
        encoding="utf-8",
    )
    logger.info("Parity results written to %s", args.output)
    return args.output


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    setup_logging()
    try:
        run(build_parser().parse_args(argv))
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
