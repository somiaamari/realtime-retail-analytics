"""TensorRT detector that loads an existing serialized engine."""

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vision_pipeline.config import DetectorConfig, ModelConfig
from vision_pipeline.detection.yolo_detector import BaseDetector
from vision_pipeline.logging_utils import get_logger
from vision_pipeline.optimization.postprocess import (
    decode_nms_output,
    decode_yolo_output,
)
from vision_pipeline.optimization.preprocess import LetterboxInfo, letterbox
from vision_pipeline.types import Detection

logger = get_logger(__name__)


class TensorRTDetector(BaseDetector):
    """Execute an existing TensorRT engine via PyCUDA asynchronous transfers."""

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        detector_config: DetectorConfig | None = None,
    ) -> None:
        self.model_config = model_config or ModelConfig()
        self.detector_config = detector_config or DetectorConfig(
            backend="tensorrt", precision="fp16"
        )
        if self.detector_config.precision != "fp16":
            raise ValueError(
                "The configured prebuilt TensorRT engine is FP16; use "
                "detector.precision=fp16."
            )
        self.engine_path = Path(
            self.detector_config.model_path
            or self.model_config.weights_path.parent
            / "exports"
            / f"{self.model_config.weights_path.stem}.engine"
        )
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")
        try:
            self._trt = importlib.import_module("tensorrt")
            self._cuda = importlib.import_module("pycuda.driver")
        except ImportError as error:
            raise ImportError(
                "TensorRT inference requires a CUDA-enabled TensorRT installation "
                "and PyCUDA built for the same CUDA runtime. See "
                "docs/phase-07-optimization.md for environment-specific setup."
            ) from error

        self._cuda.init()
        device = self._cuda.Device(0)
        self._device_name = device.name()
        self._cuda_context = device.make_context()
        self._stream: Any | None = None
        self._engine: Any | None = None
        self._execution_context: Any | None = None
        self._host_buffers: dict[str, NDArray[Any]] = {}
        self._device_buffers: dict[str, Any] = {}
        self._tensor_shapes: dict[str, tuple[int, ...]] = {}
        self._tensor_dtypes: dict[str, np.dtype[Any]] = {}
        self._input_name = ""
        self._closed = False
        context_popped = False
        try:
            logger_trt = self._trt.Logger(self._trt.Logger.WARNING)
            runtime = self._trt.Runtime(logger_trt)
            engine_bytes = self.engine_path.read_bytes()
            try:
                self._engine = runtime.deserialize_cuda_engine(engine_bytes)
            except Exception as error:
                raise RuntimeError(
                    f"Could not load TensorRT engine {self.engine_path} on GPU "
                    f"'{self._device_name}'. The serialized plan may target a "
                    "different GPU architecture or TensorRT/CUDA version."
                ) from error
            if self._engine is None:
                raise RuntimeError(
                    f"TensorRT could not deserialize {self.engine_path} on GPU "
                    f"'{self._device_name}'; check engine/GPU/runtime compatibility."
                )
            self._execution_context = self._engine.create_execution_context()
            if self._execution_context is None:
                raise RuntimeError("TensorRT failed to create an execution context")
            self._stream = self._cuda.Stream()
            self._configure_io()
            sidecar_path = self.engine_path.with_suffix(
                self.engine_path.suffix + ".json"
            )
            metadata = (
                json.loads(sidecar_path.read_text(encoding="utf-8"))
                if sidecar_path.is_file()
                else {}
            )
            self.class_names = metadata.get("class_names", ["person"])
            if isinstance(self.class_names, dict):
                self.class_names = {
                    int(class_id): str(class_name)
                    for class_id, class_name in self.class_names.items()
                }
            self.nms_embedded = bool(metadata.get("nms_embedded", False))
            self._class_ids = {
                value for value in self.model_config.classes if isinstance(value, int)
            }
            names_selected = {
                value.casefold()
                for value in self.model_config.classes
                if isinstance(value, str)
            }
            if isinstance(self.class_names, dict):
                self._class_ids.update(
                    class_id
                    for class_id, class_name in self.class_names.items()
                    if class_name.casefold() in names_selected
                )
            else:
                self._class_ids.update(
                    class_id
                    for class_id, class_name in enumerate(self.class_names)
                    if class_name.casefold() in names_selected
                )
        except Exception:
            self._cuda_context.pop()
            context_popped = True
            self.close()
            raise
        finally:
            if not context_popped:
                self._cuda_context.pop()
        logger.info(
            "Loaded TensorRT engine %s on GPU %s",
            self.engine_path,
            self._device_name,
        )

    def _configure_io(self) -> None:
        """Resolve fixed/dynamic tensor shapes and allocate pinned/device buffers."""
        assert self._engine is not None
        assert self._execution_context is not None
        if hasattr(self._engine, "num_io_tensors"):
            tensor_names = [
                self._engine.get_tensor_name(index)
                for index in range(self._engine.num_io_tensors)
            ]
            for name in tensor_names:
                shape = tuple(self._engine.get_tensor_shape(name))
                if any(dimension < 0 for dimension in shape):
                    if (
                        self._engine.get_tensor_mode(name)
                        == self._trt.TensorIOMode.INPUT
                    ):
                        shape = tuple(
                            self.model_config.imgsz if dimension < 0 else dimension
                            for dimension in shape
                        )
                        if len(shape) == 3:
                            shape = (1, *shape)
                        self._execution_context.set_input_shape(name, shape)
                resolved = tuple(self._execution_context.get_tensor_shape(name))
                dtype = np.dtype(self._trt.nptype(self._engine.get_tensor_dtype(name)))
                self._allocate(name, resolved, dtype)
                self._execution_context.set_tensor_address(
                    name, int(self._device_buffers[name])
                )
                if self._engine.get_tensor_mode(name) == self._trt.TensorIOMode.INPUT:
                    self._input_name = name
        else:
            for index in range(self._engine.num_bindings):
                name = self._engine.get_binding_name(index)
                shape = tuple(self._engine.get_binding_shape(index))
                if self._engine.binding_is_input(index) and any(
                    dimension < 0 for dimension in shape
                ):
                    shape = tuple(
                        self.model_config.imgsz if dimension < 0 else dimension
                        for dimension in shape
                    )
                    if len(shape) == 3:
                        shape = (1, *shape)
                    self._execution_context.set_binding_shape(index, shape)
                resolved = tuple(self._execution_context.get_binding_shape(index))
                dtype = np.dtype(
                    self._trt.nptype(self._engine.get_binding_dtype(index))
                )
                self._allocate(name, resolved, dtype)
                if self._engine.binding_is_input(index):
                    self._input_name = name
        if not self._input_name:
            raise RuntimeError("TensorRT engine has no image input binding")
        self._binding_names = (
            tensor_names
            if hasattr(self._engine, "num_io_tensors")
            else [
                self._engine.get_binding_name(index)
                for index in range(self._engine.num_bindings)
            ]
        )
        if not any(name != self._input_name for name in self._binding_names):
            raise RuntimeError("TensorRT engine has no output binding")

    def _allocate(
        self, name: str, shape: tuple[int, ...], dtype: np.dtype[Any]
    ) -> None:
        if any(dimension <= 0 for dimension in shape):
            raise RuntimeError(f"TensorRT resolved invalid shape {shape} for {name}")
        self._tensor_shapes[name] = shape
        self._tensor_dtypes[name] = dtype
        host = self._cuda.pagelocked_empty(int(np.prod(shape)), dtype)
        self._host_buffers[name] = host
        self._device_buffers[name] = self._cuda.mem_alloc(host.nbytes)

    def _input_tensor(self, frame: NDArray[Any]) -> tuple[NDArray[Any], LetterboxInfo]:
        shape = self._tensor_shapes[self._input_name]
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
            raise ValueError(f"Expected TensorRT NCHW image input, got {shape}")
        return letterbox(
            frame,
            (shape[2], shape[3]),
            dtype=self._tensor_dtypes[self._input_name],
        )

    def raw_output(self, frame: NDArray[Any]) -> NDArray[Any]:
        """Return the primary engine output for numerical parity checks."""
        return np.asarray(self._infer(frame)[0])

    def _infer(self, frame: NDArray[Any]) -> tuple[list[NDArray[Any]], LetterboxInfo]:
        if self._execution_context is None or self._stream is None:
            raise RuntimeError("TensorRT detector is closed")
        tensor, geometry = self._input_tensor(frame)
        self._cuda_context.push()
        try:
            input_host = self._host_buffers[self._input_name]
            np.copyto(input_host, tensor.reshape(-1))
            self._cuda.memcpy_htod_async(
                self._device_buffers[self._input_name],
                input_host,
                self._stream,
            )
            if hasattr(self._execution_context, "execute_async_v3"):
                succeeded = self._execution_context.execute_async_v3(
                    stream_handle=self._stream.handle
                )
            else:
                bindings = [
                    int(self._device_buffers[name]) for name in self._binding_names
                ]
                succeeded = self._execution_context.execute_async_v2(
                    bindings=bindings,
                    stream_handle=self._stream.handle,
                )
            if not succeeded:
                raise RuntimeError("TensorRT asynchronous inference execution failed")
            output_names = [
                name for name in self._binding_names if name != self._input_name
            ]
            for name in output_names:
                self._cuda.memcpy_dtoh_async(
                    self._host_buffers[name],
                    self._device_buffers[name],
                    self._stream,
                )
            self._stream.synchronize()
            outputs = [
                self._host_buffers[name].reshape(self._tensor_shapes[name]).copy()
                for name in output_names
            ]
            return outputs, geometry
        finally:
            self._cuda_context.pop()

    def detect(self, frame: NDArray[Any]) -> list[Detection]:
        """Run the loaded engine and map results into source-frame coordinates."""
        outputs, geometry = self._infer(frame)
        output = outputs[0]
        if self.nms_embedded or (output.ndim >= 2 and output.shape[-1] == 6):
            return decode_nms_output(
                output,
                geometry,
                class_names=self.class_names,
                confidence_threshold=self.model_config.conf_threshold,
                class_ids=self._class_ids or None,
            )
        return decode_yolo_output(
            output,
            geometry,
            class_names=self.class_names,
            confidence_threshold=self.model_config.conf_threshold,
            iou_threshold=self.model_config.iou_threshold,
            class_ids=self._class_ids or None,
        )

    def warmup(self) -> None:
        """Initialize the TensorRT execution path with a zero-valued frame."""
        self.detect(
            np.zeros(
                (self.model_config.imgsz, self.model_config.imgsz, 3),
                dtype=np.uint8,
            )
        )

    def close(self) -> None:
        """Release device allocations and detach the CUDA context."""
        if getattr(self, "_cuda_context", None) is None:
            return
        if self._closed:
            return
        self._cuda_context.push()
        try:
            if getattr(self, "_cuda", None) is not None:
                for buffer in getattr(self, "_device_buffers", {}).values():
                    buffer.free()
        finally:
            self._cuda_context.pop()
            self._cuda_context.detach()
            self._cuda_context = None
            self._closed = True

    def __enter__(self) -> "TensorRTDetector":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
