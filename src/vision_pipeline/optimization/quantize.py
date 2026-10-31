"""ONNX FP16 conversion and static INT8 quantization."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from vision_pipeline.logging_utils import get_logger
from vision_pipeline.optimization.artifacts import register_artifact
from vision_pipeline.optimization.preprocess import letterbox

logger = get_logger(__name__)
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class CalibrationDataReader:
    """Lazily preprocess calibration images using the inference letterbox path."""

    def __init__(
        self,
        calibration_dir: str | Path,
        input_name: str,
        *,
        imgsz: int = 640,
        input_shape: tuple[int, int] | None = None,
    ) -> None:
        self.calibration_dir = Path(calibration_dir)
        if not self.calibration_dir.is_dir():
            raise FileNotFoundError(
                f"Calibration image directory not found: {self.calibration_dir}"
            )
        self.image_paths = sorted(
            path
            for path in self.calibration_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES
        )
        if not self.image_paths:
            raise ValueError(f"No calibration images found in {self.calibration_dir}")
        self.input_name = input_name
        self.imgsz = imgsz
        self.input_shape = input_shape or (imgsz, imgsz)
        self._iterator: Iterator[dict[str, NDArray[Any]]] | None = None

    def _samples(self) -> Iterator[dict[str, NDArray[Any]]]:
        for path in self.image_paths:
            image = cv2.imread(str(path))
            if image is None:
                raise OSError(f"Could not read calibration image {path}")
            tensor, _ = letterbox(image, self.input_shape, dtype=np.dtype(np.float32))
            yield {self.input_name: tensor}

    def get_next(self) -> dict[str, NDArray[Any]] | None:
        """Return the next preprocessed input dictionary, or ``None`` at EOF."""
        if self._iterator is None:
            self.rewind()
        assert self._iterator is not None
        return next(self._iterator, None)

    def rewind(self) -> None:
        """Restart calibration from the first image."""
        self._iterator = self._samples()


def convert_onnx_to_fp16(
    input_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Convert ONNX floating-point tensors to FP16 while keeping external I/O FP32."""
    source = Path(input_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"ONNX model not found: {source}")
    try:
        onnx = __import__("onnx")
        converter = __import__(
            "onnxconverter_common.float16",
            fromlist=["convert_float_to_float16"],
        )
    except ImportError as error:
        raise ImportError(
            "FP16 conversion requires `pip install -e '.[onnx]'`."
        ) from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(source))
    converted = converter.convert_float_to_float16(model, keep_io_types=True)
    del converted.graph.value_info[:]
    converted = onnx.shape_inference.infer_shapes(converted, strict_mode=True)
    onnx.checker.check_model(converted)
    onnx.save(converted, str(destination))
    _write_sidecar(source, destination, "fp16")
    _register_quantized_artifact(source, destination, "onnx_fp16")
    logger.info("Wrote FP16 ONNX model %s", destination)
    return destination


def quantize_onnx_static(
    input_path: str | Path,
    output_path: str | Path,
    calibration_dir: str | Path,
    *,
    imgsz: int = 640,
) -> Path:
    """Quantize an ONNX graph to static QDQ INT8 using calibration images."""
    source = Path(input_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"ONNX model not found: {source}")
    try:
        onnx = __import__("onnx")
        quantization = __import__(
            "onnxruntime.quantization",
            fromlist=[
                "QuantType",
                "QuantFormat",
                "CalibrationMethod",
                "quantize_static",
            ],
        )
    except ImportError as error:
        raise ImportError(
            "INT8 quantization requires `pip install -e '.[onnx]'`."
        ) from error
    model = onnx.load(str(source))
    if not model.graph.input:
        raise ValueError(f"ONNX graph has no input: {source}")
    model_input = model.graph.input[0]
    dimensions = model_input.type.tensor_type.shape.dim
    if len(dimensions) != 4:
        raise ValueError(
            f"Expected an NCHW image model input, got {len(dimensions)} dimensions"
        )
    height = dimensions[2].dim_value or imgsz
    width = dimensions[3].dim_value or imgsz
    reader = CalibrationDataReader(
        calibration_dir,
        model_input.name,
        imgsz=imgsz,
        input_shape=(height, width),
    )
    excluded_nodes = _output_class_head_nodes(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    quantization.quantize_static(
        str(source),
        str(destination),
        reader,
        quant_format=quantization.QuantFormat.QDQ,
        activation_type=quantization.QuantType.QUInt8,
        weight_type=quantization.QuantType.QInt8,
        calibrate_method=quantization.CalibrationMethod.MinMax,
        per_channel=True,
        nodes_to_exclude=excluded_nodes,
    )
    quantized_model = onnx.load(str(destination))
    restored_outputs = _restore_float_graph_outputs(quantized_model)
    onnx.checker.check_model(quantized_model)
    onnx.save(quantized_model, str(destination))
    _write_sidecar(
        source,
        destination,
        "int8",
        {
            "quantization_excluded_nodes": excluded_nodes,
            "output": [
                {
                    "name": value.name,
                    "shape": [
                        int(dimension.dim_value)
                        if dimension.HasField("dim_value")
                        else dimension.dim_param or None
                        for dimension in value.type.tensor_type.shape.dim
                    ],
                }
                for value in quantized_model.graph.output
            ],
        },
    )
    _register_quantized_artifact(source, destination, "onnx_int8")
    logger.info(
        "Wrote INT8 ONNX model %s using %d calibration images",
        destination,
        len(reader.image_paths),
    )
    logger.info(
        "Kept %d terminal detection-head convolution(s) in floating point",
        len(excluded_nodes),
    )
    logger.info(
        "Preserved float graph outputs: %s",
        ", ".join(restored_outputs) if restored_outputs else "none required",
    )
    return destination


def _restore_float_graph_outputs(model: Any) -> list[str]:
    """Remove output-only QDQ pairs that collapse YOLO's mixed-scale output."""
    nodes = list(model.graph.node)
    producers = {
        output_name: (index, node)
        for index, node in enumerate(nodes)
        for output_name in node.output
        if output_name
    }
    consumers: dict[str, int] = {}
    for node in nodes:
        for input_name in node.input:
            consumers[input_name] = consumers.get(input_name, 0) + 1
    remove_indices: set[int] = set()
    restored_names: list[str] = []
    for graph_output in model.graph.output:
        dequantized = producers.get(graph_output.name)
        if dequantized is None or dequantized[1].op_type != "DequantizeLinear":
            continue
        quantized = producers.get(dequantized[1].input[0])
        if (
            quantized is None
            or quantized[1].op_type != "QuantizeLinear"
            or consumers.get(quantized[1].output[0], 0) != 1
        ):
            continue
        graph_output.name = quantized[1].input[0]
        remove_indices.update((dequantized[0], quantized[0]))
        restored_names.append(graph_output.name)
    if remove_indices:
        kept_nodes = [
            node for index, node in enumerate(nodes) if index not in remove_indices
        ]
        del model.graph.node[:]
        model.graph.node.extend(kept_nodes)
        del model.graph.value_info[:]
    return restored_names


def _output_class_head_nodes(model: Any) -> list[str]:
    """Keep the terminal YOLO detection-head convolutions in floating point."""
    graph = model.graph
    producers = {
        output_name: node
        for node in graph.node
        for output_name in node.output
        if output_name
    }
    if not graph.output:
        return []
    output_producer = producers.get(graph.output[0].name)
    if output_producer is None:
        return []
    head_prefix = output_producer.name.rsplit("/", 1)[0]
    head_convs = sorted(
        node.name
        for node in graph.node
        if node.op_type == "Conv" and node.name.startswith(f"{head_prefix}/")
    )
    if head_convs:
        return head_convs
    if output_producer.op_type != "Concat":
        return []
    pending = [output_producer.input[-1]]
    visited: set[str] = set()
    excluded: set[str] = set()
    while pending:
        tensor_name = pending.pop()
        if tensor_name in visited:
            continue
        visited.add(tensor_name)
        node = producers.get(tensor_name)
        if node is None:
            continue
        if node.op_type == "Conv":
            excluded.add(node.name)
            continue
        pending.extend(node.input)
    return sorted(excluded)


def _register_quantized_artifact(
    source: Path,
    destination: Path,
    precision: str,
) -> None:
    sidecar = source.with_suffix(source.suffix + ".json")
    metadata = (
        json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
    )
    source_weights = Path(metadata.get("source_weights", source))
    register_artifact(source_weights, destination, precision)


def _write_sidecar(
    source: Path,
    destination: Path,
    precision: str,
    extra: dict[str, Any] | None = None,
) -> None:
    source_sidecar = source.with_suffix(source.suffix + ".json")
    if not source_sidecar.is_file():
        return
    metadata = json.loads(source_sidecar.read_text(encoding="utf-8"))
    metadata["precision"] = precision
    metadata["quantized_from"] = str(source)
    if extra:
        metadata.update(extra)
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
