"""Downscaled Gaussian heatmap accumulation and image export."""

from collections import deque
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from numpy.typing import NDArray


class HeatmapAccumulator:
    """Accumulate Gaussian track-position heat on a small grid.

    Adding a frame costs O(tracks * kernel_area); memory is O(grid_area). The
    Gaussian patch is precomputed and clipped at boundaries rather than
    allocating or scanning a full-frame mask for every track.
    """

    def __init__(
        self,
        frame_size: tuple[int, int],
        grid_scale: int = 8,
        sigma: float = 3.0,
        decay: float = 0.0,
        *,
        mode: Literal["cumulative", "rolling"] = "cumulative",
        window_seconds: float | None = None,
    ) -> None:
        width, height = frame_size
        if width <= 0 or height <= 0 or grid_scale < 1 or sigma <= 0:
            raise ValueError(
                "Heatmap dimensions, grid_scale, and sigma must be positive."
            )
        if not 0 <= decay <= 1:
            raise ValueError("Heatmap decay must be between zero and one.")
        if mode not in ("cumulative", "rolling"):
            raise ValueError("Heatmap mode must be cumulative or rolling.")
        if window_seconds is not None and window_seconds <= 0:
            raise ValueError("window_seconds must be positive when provided.")
        self.frame_size = frame_size
        self.grid_scale = grid_scale
        self.sigma = sigma
        self.decay = decay
        self.mode = mode
        self.window_seconds = (
            60.0 if mode == "rolling" and window_seconds is None else window_seconds
        )
        self.grid = np.zeros(
            (
                max(1, (height + grid_scale - 1) // grid_scale),
                max(1, (width + grid_scale - 1) // grid_scale),
            ),
            dtype=np.float32,
        )
        self._kernel = self._make_kernel(sigma)
        self._last_timestamp_s: float | None = None
        self._patches: deque[tuple[float, int, int]] = deque()

    @staticmethod
    def _make_kernel(sigma: float) -> NDArray[np.float32]:
        """Create a normalized, truncated Gaussian patch."""
        radius = max(1, int(np.ceil(3.0 * sigma)))
        coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
        x_grid, y_grid = np.meshgrid(coordinates, coordinates)
        kernel = np.exp(-(x_grid * x_grid + y_grid * y_grid) / (2 * sigma * sigma))
        kernel /= float(kernel.sum())
        return kernel.astype(np.float32)

    def add(
        self,
        points: list[tuple[float, float]],
        *,
        timestamp_s: float | None = None,
    ) -> None:
        """Add one Gaussian contribution for each original-frame coordinate."""
        if timestamp_s is not None:
            if (
                self._last_timestamp_s is not None
                and timestamp_s < self._last_timestamp_s
            ):
                raise ValueError("Heatmap timestamps must be non-decreasing.")
            if self._last_timestamp_s is not None and self.decay:
                elapsed = timestamp_s - self._last_timestamp_s
                self.grid *= np.float32(np.exp(-self.decay * elapsed))
            self._last_timestamp_s = timestamp_s
            if self.mode == "rolling" and self.window_seconds is not None:
                self._expire(timestamp_s - self.window_seconds, timestamp_s)
        radius_y, radius_x = self._kernel.shape[0] // 2, self._kernel.shape[1] // 2
        for x, y in points:
            grid_x = round(x / self.grid_scale)
            grid_y = round(y / self.grid_scale)
            left = max(0, grid_x - radius_x)
            right = min(self.grid.shape[1], grid_x + radius_x + 1)
            top = max(0, grid_y - radius_y)
            bottom = min(self.grid.shape[0], grid_y + radius_y + 1)
            if left >= right or top >= bottom:
                continue
            kernel_left = left - (grid_x - radius_x)
            kernel_top = top - (grid_y - radius_y)
            kernel_right = kernel_left + right - left
            kernel_bottom = kernel_top + bottom - top
            patch = self._kernel[kernel_top:kernel_bottom, kernel_left:kernel_right]
            self.grid[top:bottom, left:right] += patch
            if self.mode == "rolling" and self.window_seconds is not None:
                self._patches.append((timestamp_s or 0.0, grid_x, grid_y))

    def _expire(self, cutoff_s: float, current_time_s: float) -> None:
        """Remove Gaussian contributions older than a configured rolling window."""
        radius_y, radius_x = self._kernel.shape[0] // 2, self._kernel.shape[1] // 2
        while self._patches and self._patches[0][0] < cutoff_s:
            timestamp_s, grid_x, grid_y = self._patches.popleft()
            left = max(0, grid_x - radius_x)
            right = min(self.grid.shape[1], grid_x + radius_x + 1)
            top = max(0, grid_y - radius_y)
            bottom = min(self.grid.shape[0], grid_y + radius_y + 1)
            kernel_left = left - (grid_x - radius_x)
            kernel_top = top - (grid_y - radius_y)
            kernel_right = kernel_left + right - left
            kernel_bottom = kernel_top + bottom - top
            decay_scale = np.float32(
                np.exp(-self.decay * (current_time_s - timestamp_s))
            )
            self.grid[top:bottom, left:right] -= (
                decay_scale
                * self._kernel[
                    kernel_top:kernel_bottom,
                    kernel_left:kernel_right,
                ]
            )
        np.maximum(self.grid, 0.0, out=self.grid)

    def normalize(self, percentile: float = 99.0) -> NDArray[np.uint8]:
        """Normalize values using percentile clipping to reduce hotspot bias."""
        if not 0 < percentile <= 100:
            raise ValueError("Normalization percentile must be in (0, 100].")
        positive = self.grid[self.grid > 0]
        result = np.zeros(self.grid.shape, dtype=np.uint8)
        if positive.size == 0:
            return result
        upper = float(np.percentile(positive, percentile))
        if upper <= 0:
            return result
        scaled = np.clip(self.grid / upper, 0.0, 1.0) * 255.0
        result[:] = np.rint(scaled)
        return result  # type: ignore[no-any-return]

    def render(
        self,
        background_frame: NDArray[Any] | None = None,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> NDArray[Any]:
        """Render the normalized heatmap, optionally blended over a BGR frame."""
        if not 0 <= alpha <= 1:
            raise ValueError("Heatmap alpha must be between zero and one.")
        small = self.normalize()
        heat = cv2.applyColorMap(small, colormap)
        width, height = self.frame_size
        full_size = cv2.resize(heat, (width, height), interpolation=cv2.INTER_LINEAR)
        if background_frame is None:
            return full_size
        if background_frame.shape[:2] != (height, width):
            raise ValueError("Background frame size does not match the heatmap size.")
        if background_frame.ndim != 3 or background_frame.shape[2] != 3:
            raise ValueError("Background frame must be a three-channel BGR image.")
        mask = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR) > 0
        result = np.array(background_frame, copy=True)
        if np.any(mask):
            blended = cv2.addWeighted(
                background_frame,
                1.0 - alpha,
                full_size,
                alpha,
                0,
            )
            result[mask] = blended[mask]
        return result  # type: ignore[no-any-return]

    def export_png(
        self,
        path: str | Path,
        background_frame: NDArray[Any] | None = None,
        *,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> None:
        """Render and save the heatmap as a PNG image."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(
            str(output_path), self.render(background_frame, alpha, colormap)
        ):
            raise OSError(f"Unable to write heatmap image: {output_path}")

    def export_npy(self, path: str | Path) -> None:
        """Save the unnormalized accumulation grid as a NumPy array."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, self.grid)

    def reset(self) -> None:
        """Clear activity and timestamp history."""
        self.grid.fill(0)
        self._last_timestamp_s = None
        self._patches.clear()
