"""Pure-NumPy constant-velocity Kalman filter for bounding boxes."""

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[Any]


class KalmanFilter:
    """Track box center, shape, and their velocities with a linear Kalman filter.

    The state is ``[cx, cy, aspect_ratio, height, vcx, vcy, va, vh]``: the
    first four values describe a box and the last four describe how quickly
    those values change per frame. Measurements contain only the first four
    values. Position and velocity noise scale with box height, so larger
    objects are allowed proportionally larger motion uncertainty; small fixed
    noise terms keep aspect-ratio estimates numerically stable.
    """

    ndim = 4
    dt = 1.0
    std_weight_position = 1.0 / 20.0
    std_weight_velocity = 1.0 / 160.0

    def __init__(self) -> None:
        """Initialize the fixed motion and observation matrices."""
        self._motion_mat = np.eye(2 * self.ndim, dtype=np.float64)
        for index in range(self.ndim):
            self._motion_mat[index, self.ndim + index] = self.dt
        self._update_mat = np.eye(self.ndim, 2 * self.ndim, dtype=np.float64)

    def initiate(self, measurement: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Create a state mean and covariance from ``[cx, cy, a, h]``.

        Args:
            measurement: Four-element measured box vector.

        Returns:
            Eight-element state mean and its 8x8 covariance matrix.
        """
        measurement = np.asarray(measurement, dtype=np.float64).reshape(4)
        mean = np.r_[measurement, np.zeros(4, dtype=np.float64)]
        height = max(float(measurement[3]), 1.0)
        std = np.array(
            [
                2 * self.std_weight_position * height,
                2 * self.std_weight_position * height,
                1e-2,
                2 * self.std_weight_position * height,
                10 * self.std_weight_velocity * height,
                10 * self.std_weight_velocity * height,
                1e-5,
                10 * self.std_weight_velocity * height,
            ],
            dtype=np.float64,
        )
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(
        self, mean: FloatArray, cov: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Advance the box state by one frame and add motion uncertainty."""
        height = max(float(mean[3]), 1.0)
        std_position = self.std_weight_position * height
        std_velocity = self.std_weight_velocity * height
        motion_std = np.array(
            [
                std_position,
                std_position,
                1e-2,
                std_position,
                std_velocity,
                std_velocity,
                1e-5,
                std_velocity,
            ],
            dtype=np.float64,
        )
        motion_cov = np.diag(np.square(motion_std))
        predicted_mean = self._motion_mat @ mean
        predicted_cov = self._motion_mat @ cov @ self._motion_mat.T + motion_cov
        predicted_cov = (predicted_cov + predicted_cov.T) * 0.5
        return predicted_mean, predicted_cov

    def project(
        self, mean: FloatArray, cov: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Project state and covariance into the four-dimensional measurement space."""
        height = max(float(mean[3]), 1.0)
        std = np.array(
            [
                self.std_weight_position * height,
                self.std_weight_position * height,
                1e-1,
                self.std_weight_position * height,
            ],
            dtype=np.float64,
        )
        innovation_cov = np.diag(np.square(std))
        projected_mean = self._update_mat @ mean
        projected_cov = self._update_mat @ cov @ self._update_mat.T + innovation_cov
        projected_cov = (projected_cov + projected_cov.T) * 0.5
        return projected_mean, projected_cov

    def update(
        self,
        mean: FloatArray,
        cov: FloatArray,
        measurement: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """Correct a prediction using a measured ``[cx, cy, a, h]`` box."""
        measurement = np.asarray(measurement, dtype=np.float64).reshape(4)
        projected_mean, projected_cov = self.project(mean, cov)
        cross_cov = cov @ self._update_mat.T
        kalman_gain = np.linalg.solve(projected_cov, cross_cov.T).T
        updated_mean = mean + kalman_gain @ (measurement - projected_mean)
        updated_cov = cov - kalman_gain @ projected_cov @ kalman_gain.T
        updated_cov = (updated_cov + updated_cov.T) * 0.5
        return updated_mean, updated_cov

    def gating_distance(
        self,
        mean: FloatArray,
        cov: FloatArray,
        measurements: FloatArray,
    ) -> FloatArray:
        """Return squared Mahalanobis distances from measurements to a prediction."""
        projected_mean, projected_cov = self.project(mean, cov)
        residual = np.asarray(measurements, dtype=np.float64) - projected_mean
        solved = np.linalg.solve(projected_cov, residual.T)
        return cast(FloatArray, np.einsum("ij,ji->i", residual, solved))
