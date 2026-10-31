"""Convert an ONNX model to FP16 or calibrated static INT8."""

import argparse
import logging
import sys
from pathlib import Path

from vision_pipeline.logging_utils import setup_logging
from vision_pipeline.optimization.quantize import (
    convert_onnx_to_fp16,
    quantize_onnx_static,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the ONNX quantization parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp16", "int8"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--calibration-dir", type=Path, default=Path("data/calibration")
    )
    parser.add_argument("--imgsz", type=int, default=640)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the selected quantization operation."""
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        if args.precision == "fp16":
            convert_onnx_to_fp16(args.input, args.output)
        else:
            quantize_onnx_static(
                args.input,
                args.output,
                args.calibration_dir,
                imgsz=args.imgsz,
            )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
