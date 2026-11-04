"""Static and adaptive inference frame-rate control."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from vision_pipeline.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ControllerConfig:
    """Settings for detection stride and adaptive load control."""

    target_fps: float = 30.0
    latency_budget_ms: float = 100.0
    detect_every_n: int = 1
    adaptive: bool = False
    min_stride: int = 1
    max_stride: int = 8
    ema_alpha: float = 0.2
    hysteresis: float = 0.15
    hysteresis_samples: int = 3
    max_detection_age_frames: int | None = None

    def __post_init__(self) -> None:
        """Validate controller ranges."""
        if self.target_fps <= 0 or self.latency_budget_ms <= 0:
            raise ValueError("target_fps and latency_budget_ms must be positive")
        if self.detect_every_n < 1 or self.min_stride < 1 or self.max_stride < 1:
            raise ValueError("detection strides must be positive")
        if self.min_stride > self.max_stride:
            raise ValueError("min_stride cannot exceed max_stride")
        if not 0 < self.ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")
        if not 0 <= self.hysteresis < 1:
            raise ValueError("hysteresis must be in [0, 1)")
        if self.hysteresis_samples < 1:
            raise ValueError("hysteresis_samples must be positive")
        if self.max_detection_age_frames is not None:
            if self.max_detection_age_frames < 1:
                raise ValueError("max_detection_age_frames must be positive")


class AdaptiveController:
    """Adjust detector stride using latency and queue depth with hysteresis."""

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.config = config or ControllerConfig()
        self._lock = threading.Lock()
        self._stride = max(
            self.config.min_stride,
            min(self.config.max_stride, self.config.detect_every_n),
        )
        self._latency_ema_ms: float | None = None
        self._over_count = 0
        self._under_count = 0
        self._last_detection_frame: int | None = None
        self._last_detection_ts: float | None = None

    @property
    def stride(self) -> int:
        """Return the current detection stride."""
        with self._lock:
            return self._stride

    @property
    def latency_ema_ms(self) -> float | None:
        """Return the current smoothed latency estimate."""
        with self._lock:
            return self._latency_ema_ms

    def should_detect(
        self,
        frame_id: int,
        queue_depth: int = 0,
        capture_ts: float | None = None,
    ) -> bool:
        """Return whether ``frame_id`` should be sent to the detector."""
        del queue_depth
        with self._lock:
            if self._last_detection_frame is None:
                return True
            max_age = self.config.max_detection_age_frames
            if (
                max_age is not None
                and self._last_detection_frame is not None
                and frame_id - self._last_detection_frame >= max_age
            ):
                return True
            if (
                capture_ts is not None
                and self._last_detection_ts is not None
                and capture_ts - self._last_detection_ts < 1.0 / self.config.target_fps
            ):
                return False
            return frame_id % self._stride == 0

    def record_detection(self, frame_id: int, capture_ts: float | None = None) -> None:
        """Record the frame index of the most recent real detection."""
        with self._lock:
            self._last_detection_frame = frame_id
            if capture_ts is not None:
                self._last_detection_ts = capture_ts

    def observe(self, latency_ms: float, queue_depth: int = 0) -> int:
        """Update EMA and possibly change stride; return the current stride."""
        if latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        with self._lock:
            alpha = self.config.ema_alpha
            if self._latency_ema_ms is None:
                self._latency_ema_ms = float(latency_ms)
            else:
                self._latency_ema_ms = (
                    alpha * latency_ms + (1.0 - alpha) * self._latency_ema_ms
                )
            if not self.config.adaptive:
                return self._stride

            budget = self.config.latency_budget_ms
            overloaded = (
                self._latency_ema_ms > budget * (1.0 + self.config.hysteresis)
                or queue_depth > 0
            )
            comfortably_under = (
                self._latency_ema_ms < budget * (1.0 - self.config.hysteresis)
                and queue_depth == 0
            )
            self._over_count = self._over_count + 1 if overloaded else 0
            self._under_count = self._under_count + 1 if comfortably_under else 0
            new_stride = self._stride
            if (
                self._over_count >= self.config.hysteresis_samples
                and self._stride < self.config.max_stride
            ):
                new_stride += 1
                self._over_count = 0
                self._under_count = 0
            elif (
                self._under_count >= self.config.hysteresis_samples
                and self._stride > self.config.min_stride
            ):
                new_stride -= 1
                self._under_count = 0
                self._over_count = 0
            if new_stride != self._stride:
                logger.info(
                    "Adaptive detector stride changed %d -> %d (latency EMA %.2f ms)",
                    self._stride,
                    new_stride,
                    self._latency_ema_ms,
                )
                self._stride = new_stride
            return self._stride
