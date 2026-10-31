"""Tests for numpy preprocessing, output decoding, and backend selection."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_pipeline.config import AppConfig
from vision_pipeline.detection.factory import select_backend
from vision_pipeline.optimization.postprocess import numpy_nms
from vision_pipeline.optimization.preprocess import letterbox, undo_letterbox


def _brute_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> list[int]:
    """Reference greedy NMS implementation for randomized fixtures."""
    order = np.argsort(scores, kind="stable")[::-1].tolist()
    keep: list[int] = []
    while order:
        current = order.pop(0)
        keep.append(current)
        survivors = []
        for candidate in order:
            left = max(boxes[current, 0], boxes[candidate, 0])
            top = max(boxes[current, 1], boxes[candidate, 1])
            right = min(boxes[current, 2], boxes[candidate, 2])
            bottom = min(boxes[current, 3], boxes[candidate, 3])
            intersection = max(0.0, right - left) * max(0.0, bottom - top)
            area_a = max(0.0, boxes[current, 2] - boxes[current, 0]) * max(
                0.0, boxes[current, 3] - boxes[current, 1]
            )
            area_b = max(0.0, boxes[candidate, 2] - boxes[candidate, 0]) * max(
                0.0, boxes[candidate, 3] - boxes[candidate, 1]
            )
            overlap = intersection / (area_a + area_b - intersection)
            if overlap <= threshold:
                survivors.append(candidate)
        order = survivors
    return keep


def test_letterbox_inverse_mapping_round_trip() -> None:
    """Letterbox padding and scale map boxes back to the original image."""
    image = np.zeros((60, 100, 3), dtype=np.uint8)
    _, geometry = letterbox(image, (128, 128))
    original = np.asarray([[10.0, 8.0, 80.0, 50.0]], dtype=np.float32)
    model_boxes = original * geometry.scale
    model_boxes[:, (0, 2)] += geometry.pad_left
    model_boxes[:, (1, 3)] += geometry.pad_top

    restored = undo_letterbox(model_boxes, geometry)

    np.testing.assert_allclose(restored, original, atol=1e-5)


def test_letterbox_output_layout_and_normalization() -> None:
    """Preprocessing returns contiguous normalized RGB NCHW tensors."""
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    image[:, :, 2] = 255

    tensor, _ = letterbox(image, (32, 32))

    assert tensor.shape == (1, 3, 32, 32)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    assert tensor[0, 0].max() == 1.0
    assert tensor[0, 2, 8:24, :].max() == 0.0


@pytest.mark.parametrize("seed", [4, 17, 81])
def test_numpy_nms_matches_brute_force(seed: int) -> None:
    """Vectorized overlap calculations preserve greedy NMS results."""
    random = np.random.default_rng(seed)
    starts = random.uniform(0, 80, size=(50, 2))
    sizes = random.uniform(2, 30, size=(50, 2))
    boxes = np.concatenate((starts, starts + sizes), axis=1).astype(np.float32)
    scores = random.uniform(0.1, 1.0, size=50).astype(np.float32)

    actual = numpy_nms(boxes, scores, 0.45).tolist()

    assert actual == _brute_nms(boxes, scores, 0.45)


def test_auto_backend_prefers_available_accelerated_artifact(tmp_path: Path) -> None:
    """Auto-selection requires both installed runtime and matching model file."""
    weights = tmp_path / "retail.pt"
    weights.touch()
    engine = tmp_path / "exports" / "retail.engine"
    engine.parent.mkdir()
    engine.touch()
    config = AppConfig.model_validate(
        {
            "model": {"weights_path": weights},
            "detector": {"backend": "auto", "precision": "fp16"},
        }
    )

    assert (
        select_backend(
            config,
            {"pytorch": True, "onnx": True, "tensorrt": True},
        )
        == "tensorrt"
    )
    fp32_config = config.model_copy(
        update={"detector": config.detector.model_copy(update={"precision": "fp32"})}
    )
    assert (
        select_backend(
            fp32_config,
            {"pytorch": True, "onnx": True, "tensorrt": False},
        )
        == "pytorch"
    )


def test_auto_backend_reports_missing_usable_model(tmp_path: Path) -> None:
    """The factory does not select an installed backend without an artifact."""
    config = AppConfig.model_validate(
        {
            "model": {"weights_path": tmp_path / "missing.pt"},
            "detector": {"backend": "auto"},
        }
    )

    with pytest.raises(RuntimeError, match="No usable detector backend/model"):
        select_backend(config, {"pytorch": True, "onnx": True, "tensorrt": True})


def test_factory_constructs_each_explicit_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The factory forwards validated config to the selected backend class."""
    import sys
    from types import SimpleNamespace

    import vision_pipeline.detection.factory as factory
    import vision_pipeline.detection.yolo_detector as yolo_module

    class FakeBackend:
        def __init__(self, *args: object) -> None:
            self.arguments = args

    weights = tmp_path / "model.pt"
    weights.touch()
    monkeypatch.setattr(yolo_module, "YoloDetector", FakeBackend)
    monkeypatch.setitem(
        sys.modules,
        "vision_pipeline.detection.onnx_detector",
        SimpleNamespace(OnnxDetector=FakeBackend),
    )
    monkeypatch.setitem(
        sys.modules,
        "vision_pipeline.detection.tensorrt_detector",
        SimpleNamespace(TensorRTDetector=FakeBackend),
    )

    for backend in ("pytorch", "onnx", "tensorrt"):
        config = AppConfig.model_validate(
            {
                "model": {"weights_path": weights},
                "detector": {"backend": backend},
            }
        )
        assert isinstance(factory.create_detector(config), FakeBackend)


def test_optimization_accuracy_gates_enforce_agreed_limits() -> None:
    """FP16, INT8, IDF1, mAP50-95, and line-count regression gates fail."""
    import importlib.util
    import sys

    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "benchmark_optimization.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_optimization", script_path)
    assert spec is not None and spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)
    baseline = benchmark.VariantResult(
        benchmark.Variant("PyTorch FP32", "pytorch", "fp32", Path("baseline.pt")),
        map50=0.939,
        map50_95=0.680,
        idf1=0.90,
        mota=0.80,
        line_count_error_percent=2.0,
    )
    fp16 = benchmark.VariantResult(
        benchmark.Variant("ONNX FP16", "onnx", "fp16", Path("fp16.onnx")),
        map50=0.933,
        map50_95=0.664,
        idf1=0.879,
        mota=0.79,
        line_count_error_percent=3.1,
    )
    int8 = benchmark.VariantResult(
        benchmark.Variant("ONNX INT8", "onnx", "int8", Path("int8.onnx")),
        map50=0.928,
        map50_95=0.670,
        idf1=0.89,
        mota=0.79,
        line_count_error_percent=2.5,
    )
    thresholds = {
        "max_map50_drop": 0.010,
        "max_map50_95_drop": 0.015,
        "max_idf1_drop": 0.020,
        "max_line_count_error_increase_percent": 1.0,
        "fp16_map50_ideal_drop": 0.002,
        "fp16_map50_failure_drop": 0.005,
    }

    benchmark._accuracy_status([baseline, fp16, int8], thresholds)

    assert fp16.accuracy_status == "FAILED ACCURACY TOLERANCE"
    assert len(fp16.accuracy_failures) == 4
    assert int8.accuracy_status == "FAILED ACCURACY TOLERANCE"
    assert int8.accuracy_failures == ["mAP50 drop 0.0110 exceeds tolerance 0.0100"]


@pytest.mark.requires_onnx
def test_onnx_detector_decodes_fixed_model_output(tmp_path: Path) -> None:
    """A tiny constant ONNX graph yields the expected person detection."""
    import json

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from vision_pipeline.config import DetectorConfig, ModelConfig
    from vision_pipeline.detection.onnx_detector import OnnxDetector

    model_path = tmp_path / "fixed.onnx"
    output = np.asarray([16, 16, 10, 10, 0.9], dtype=np.float32).reshape(1, 5, 1)
    constant = numpy_helper.from_array(output, name="constant_output")
    zeros = numpy_helper.from_array(
        np.zeros((1, 3, 32, 32), dtype=np.float32),
        name="zero_image",
    )
    graph = helper.make_graph(
        [
            helper.make_node("Mul", ["images", "zero_image"], ["zeroed_image"]),
            helper.make_node(
                "ReduceMean",
                ["zeroed_image"],
                ["offset"],
                axes=[0, 1, 2, 3],
                keepdims=0,
            ),
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=["constant_predictions"],
                value=constant,
            ),
            helper.make_node(
                "Add",
                ["constant_predictions", "offset"],
                ["predictions"],
            ),
        ],
        "fixed-detections",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("predictions", TensorProto.FLOAT, [1, 5, 1])],
        initializer=[zeros],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    onnx.save(model, model_path)
    model_path.with_suffix(".onnx.json").write_text(
        json.dumps({"class_names": ["person"], "nms_embedded": False}),
        encoding="utf-8",
    )
    detector = OnnxDetector(
        ModelConfig(imgsz=32),
        DetectorConfig(
            backend="onnx",
            providers=["CPUExecutionProvider"],
            model_path=model_path,
        ),
    )

    actual = detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))

    assert len(actual) == 1
    assert actual[0].bbox_xyxy == (11.0, 11.0, 21.0, 21.0)
    assert np.isclose(actual[0].confidence, 0.9)
    assert (actual[0].class_id, actual[0].class_name) == (0, "person")


@pytest.mark.requires_onnx
def test_onnx_fp16_conversion_and_calibration_reader(tmp_path: Path) -> None:
    """FP16 conversion and image calibration use the optional ONNX stack."""
    import onnx
    import onnxruntime
    from onnx import TensorProto, helper

    from vision_pipeline.optimization.quantize import (
        CalibrationDataReader,
        convert_onnx_to_fp16,
    )

    source = tmp_path / "identity.onnx"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "identity",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    onnx.save(model, source)
    fp16_path = convert_onnx_to_fp16(source, tmp_path / "identity_fp16.onnx")
    session = onnxruntime.InferenceSession(
        str(fp16_path),
        providers=["CPUExecutionProvider"],
    )
    result = session.run(
        ["output"],
        {"input": np.asarray([[2.0]], dtype=np.float32)},
    )[0]

    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    assert cv2.imwrite(str(calibration_dir / "calib.jpg"), image)
    reader = CalibrationDataReader(calibration_dir, "input", imgsz=32)

    assert fp16_path.is_file()
    np.testing.assert_array_equal(result, [[2.0]])
    sample = reader.get_next()
    assert sample is not None
    assert sample["input"].shape == (1, 3, 32, 32)
    assert reader.get_next() is None
    reader.rewind()
    assert reader.get_next() is not None


@pytest.mark.requires_onnx
def test_static_int8_quantization_writes_qdq_model(tmp_path: Path) -> None:
    """Static INT8 quantization consumes calibration images and writes a model."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from vision_pipeline.optimization.quantize import quantize_onnx_static

    source = tmp_path / "conv.onnx"
    weights = numpy_helper.from_array(
        np.ones((1, 3, 1, 1), dtype=np.float32),
        name="weights",
    )
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["input", "weights"], ["output"]),
        ],
        "conv",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 4, 4])],
        initializer=[weights],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    onnx.save(model, source)
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    assert cv2.imwrite(
        str(calibration_dir / "calib.jpg"),
        np.full((8, 8, 3), 127, dtype=np.uint8),
    )

    output = quantize_onnx_static(
        source,
        tmp_path / "conv_int8.onnx",
        calibration_dir,
        imgsz=4,
    )

    assert output.is_file()
    quantized = onnx.load(output)
    assert any(
        node.op_type.startswith("QuantizeLinear") for node in quantized.graph.node
    )
