"""Ultralytics-based one-off ONNX model export."""

import importlib
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any

from vision_pipeline.logging_utils import get_logger
from vision_pipeline.optimization.artifacts import register_artifact, sha256_file

logger = get_logger(__name__)


def _shape(value_info: Any) -> list[int | str | None]:
    result: list[int | str | None] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(int(dimension.dim_value))
        elif dimension.dim_param:
            result.append(str(dimension.dim_param))
        else:
            result.append(None)
    return result


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def export_onnx(
    weights: str | Path,
    imgsz: int = 640,
    opset: int = 17,
    dynamic: bool = False,
    simplify: bool = False,
    half: bool = False,
    nms_mode: bool | str = False,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    """Export Ultralytics weights to ONNX and write an adjacent metadata sidecar.

    Args:
        weights: Source Ultralytics checkpoint.
        imgsz: Square inference/export side length.
        opset: ONNX operator-set version.
        dynamic: Export dynamic batch/image dimensions.
        simplify: Simplify the resulting graph using onnxsim.
        half: Export with FP16 weights/activations where supported.
        nms_mode: Whether to embed NMS, or ``"embedded"``/``"none"``.
        output_dir: Destination directory; defaults to the current directory.

    Returns:
        The exported ONNX model path.
    """
    source = Path(weights)
    if not source.is_file():
        raise FileNotFoundError(f"Source weights not found: {source}")
    if imgsz <= 0 or opset < 1:
        raise ValueError("imgsz and opset must be positive")
    if isinstance(nms_mode, str) and nms_mode not in {"embedded", "none"}:
        raise ValueError("nms_mode must be true/false, 'embedded', or 'none'")
    embed_nms = nms_mode is True or nms_mode == "embedded"
    destination = Path(output_dir) if output_dir is not None else Path.cwd()
    destination.mkdir(parents=True, exist_ok=True)

    try:
        ultralytics = importlib.import_module("ultralytics")
        onnx = importlib.import_module("onnx")
    except ImportError as error:
        raise ImportError(
            "ONNX export requires `pip install -e '.[export,onnx]'`."
        ) from error
    model = ultralytics.YOLO(str(source))
    started = time.perf_counter()
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,
        simplify=simplify,
        half=half,
        nms=embed_nms,
        project=str(destination),
        name=source.stem,
        exist_ok=True,
    )
    export_seconds = time.perf_counter() - started
    reported_path = Path(str(exported)).resolve()
    if not reported_path.is_file():
        raise FileNotFoundError(
            f"Ultralytics returned an ONNX path that does not exist: {reported_path}"
        )
    exported_path = (destination / f"{source.stem}.onnx").resolve()
    if reported_path != exported_path:
        exported_path.parent.mkdir(parents=True, exist_ok=True)
        reported_path.replace(exported_path)
        reported_sidecar = reported_path.with_suffix(reported_path.suffix + ".json")
        if reported_sidecar.is_file():
            reported_sidecar.replace(
                exported_path.with_suffix(exported_path.suffix + ".json")
            )

    graph = onnx.load(str(exported_path)).graph
    metadata = {
        "source_weights": str(source.resolve()),
        "source_weights_sha256": sha256_file(source),
        "imgsz": imgsz,
        "class_names": model.names,
        "opset": opset,
        "input": [
            {"name": value.name, "shape": _shape(value)} for value in graph.input
        ],
        "output": [
            {"name": value.name, "shape": _shape(value)} for value in graph.output
        ],
        "export_time_seconds": export_seconds,
        "library_versions": {
            "ultralytics": _version("ultralytics"),
            "torch": _version("torch"),
            "onnx": _version("onnx"),
            "onnxruntime": _version("onnxruntime"),
            "onnxsim": _version("onnxsim"),
            "onnxconverter-common": _version("onnxconverter-common"),
        },
        "nms_embedded": embed_nms,
        "precision": "fp16" if half else "fp32",
    }
    sidecar_path = exported_path.with_suffix(exported_path.suffix + ".json")
    sidecar_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    register_artifact(source, exported_path, "onnx_fp16" if half else "onnx_fp32")
    logger.info("Exported %s and metadata %s", exported_path, sidecar_path)
    return exported_path
