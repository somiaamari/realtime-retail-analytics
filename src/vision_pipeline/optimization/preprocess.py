"""Numpy/OpenCV preprocessing shared by optimized inference backends."""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

_DEFAULT_DTYPE = np.dtype(np.float32)


@dataclass(frozen=True)
class LetterboxInfo:
    """Geometry needed to map letterboxed model coordinates to the source frame."""

    original_height: int
    original_width: int
    input_height: int
    input_width: int
    scale: float
    pad_left: int
    pad_top: int


def letterbox(
    image: NDArray[Any],
    input_size: int | tuple[int, int] = 640,
    *,
    color: tuple[int, int, int] = (114, 114, 114),
    dtype: np.dtype[Any] = _DEFAULT_DTYPE,
) -> tuple[NDArray[Any], LetterboxInfo]:
    """Resize and normalize a BGR image into a batched RGB NCHW tensor.

    Args:
        image: Three-channel BGR image from OpenCV.
        input_size: Square side length or ``(height, width)`` model input size.
        color: BGR padding color; YOLO's conventional gray is 114.
        dtype: Tensor element dtype, normally float32 or float16.

    Returns:
        The contiguous ``(1, 3, height, width)`` tensor and inverse geometry.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] == 0:
        raise ValueError("image must be a non-empty three-channel BGR array")
    if isinstance(input_size, int):
        input_height = input_width = input_size
    else:
        input_height, input_width = input_size
    if input_height <= 0 or input_width <= 0:
        raise ValueError("input dimensions must be positive")

    original_height, original_width = image.shape[:2]
    scale = min(input_width / original_width, input_height / original_height)
    resized_width = min(input_width, int(round(original_width * scale)))
    resized_height = min(input_height, int(round(original_height * scale)))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    pad_width = input_width - resized_width
    pad_height = input_height - resized_height
    pad_left = pad_width // 2
    pad_top = pad_height // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_height - pad_top,
        pad_left,
        pad_width - pad_left,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    tensor = padded[:, :, ::-1].transpose(2, 0, 1)
    tensor = np.ascontiguousarray(tensor, dtype=dtype) / np.array(255.0, dtype=dtype)
    tensor = np.ascontiguousarray(tensor[None, ...])
    return tensor, LetterboxInfo(
        original_height=original_height,
        original_width=original_width,
        input_height=input_height,
        input_width=input_width,
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
    )


def undo_letterbox(
    boxes_xyxy: NDArray[Any], info: LetterboxInfo
) -> NDArray[np.float32]:
    """Map xyxy boxes from model-input pixels back to source-image pixels."""
    boxes: NDArray[np.float32] = (
        np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4).copy()
    )
    boxes[:, (0, 2)] = (boxes[:, (0, 2)] - info.pad_left) / info.scale
    boxes[:, (1, 3)] = (boxes[:, (1, 3)] - info.pad_top) / info.scale
    boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, info.original_width)
    boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, info.original_height)
    return boxes
