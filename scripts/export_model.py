"""Export an Ultralytics checkpoint to ONNX."""

import argparse
import logging
import sys
from pathlib import Path

from vision_pipeline.logging_utils import setup_logging
from vision_pipeline.optimization.export import export_onnx

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the ONNX export command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--format", choices=["onnx"], default="onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--nms-mode", choices=["none", "embedded"], default="none")
    parser.add_argument("--output", type=Path, default=Path("models/exports"))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run ONNX export and report actionable failures."""
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        export_onnx(
            args.weights,
            imgsz=args.imgsz,
            opset=args.opset,
            dynamic=args.dynamic,
            simplify=args.simplify,
            half=args.half,
            nms_mode=args.nms_mode,
            output_dir=args.output,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
