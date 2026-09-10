"""Continuous camera-only alignment estimates; no serial or motor connection."""
from dataclasses import dataclass
import math

import cv2
import numpy as np

from .observer import ArucoObserver


@dataclass(frozen=True)
class AlignmentMeasurement:
    tilt_deg: float
    horizontal_deg: float
    vertical_deg: float
    corner_error_deg: float
    opposite_edge_error: float
    polygon: np.ndarray


def measure_alignment(normal, polygon) -> AlignmentMeasurement:
    """Positive X/Y means aim the lens toward image-right/image-bottom."""
    n = np.asarray(normal, dtype=float)
    points = np.asarray(polygon, dtype=float)
    if n.shape != (3,) or points.shape != (4, 2):
        raise ValueError("Expected a floor normal and four image corners")
    if not np.isfinite(n).all() or not np.isfinite(points).all() or np.linalg.norm(n) < 1e-9:
        raise ValueError("Invalid camera alignment geometry")
    n = n / np.linalg.norm(n)
    if n[2] < 0:
        n = -n
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    if min(lengths) < 1e-6 or not cv2.isContourConvex(points.astype(np.float32)):
        raise ValueError("Degenerate camera workspace polygon")
    errors = []
    for i in range(4):
        a = points[i - 1] - points[i]
        b = points[(i + 1) % 4] - points[i]
        angle = math.degrees(math.acos(float(np.clip(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1
        ))))
        errors.append(abs(angle - 90.0))
    edge_error = max(
        abs(lengths[0] - lengths[2]) / max(lengths[0], lengths[2]),
        abs(lengths[1] - lengths[3]) / max(lengths[1], lengths[3]),
    )
    return AlignmentMeasurement(
        tilt_deg=math.degrees(math.acos(float(np.clip(n[2], -1, 1)))),
        horizontal_deg=math.degrees(math.atan2(n[0], n[2])),
        vertical_deg=math.degrees(math.atan2(n[1], n[2])),
        corner_error_deg=max(errors),
        opposite_edge_error=float(edge_error),
        polygon=points,
    )


class AlignmentMonitor:
    """Require fresh acceptable estimates for a continuous hold period."""

    def __init__(self, tilt_tolerance=3.0, corner_tolerance=3.0,
                 edge_tolerance=0.05, hold_seconds=1.5):
        values = (tilt_tolerance, corner_tolerance, edge_tolerance, hold_seconds)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("Alignment tolerances and hold time must be positive")
        self.tilt_tolerance = tilt_tolerance
        self.corner_tolerance = corner_tolerance
        self.edge_tolerance = edge_tolerance
        self.hold_seconds = hold_seconds
        self._since = None
        self._last_timestamp = None
        self.elapsed = 0.0

    def update(self, measurement, timestamp, now):
        if (measurement is None or not math.isfinite(timestamp)
                or not 0 <= now - timestamp <= 0.5):
            self._since = self._last_timestamp = None
            self.elapsed = 0.0
            return "waiting"
        acceptable = (
            measurement.tilt_deg <= self.tilt_tolerance
            and measurement.corner_error_deg <= self.corner_tolerance
            and measurement.opposite_edge_error <= self.edge_tolerance
        )
        if (self._last_timestamp is not None
                and (timestamp < self._last_timestamp
                     or timestamp - self._last_timestamp > 0.5)):
            self._since = None
        self._last_timestamp = timestamp
        if not acceptable:
            self._since = None
            self.elapsed = 0.0
            return "adjust"
        if self._since is None:
            self._since = timestamp
        # Repainting the same frame never advances the hold timer.
        self.elapsed = timestamp - self._since
        return "aligned" if self.elapsed >= self.hold_seconds else "hold"

    def guidance(self, measurement):
        directions = []
        threshold = self.tilt_tolerance / 2.0
        if abs(measurement.horizontal_deg) > threshold:
            direction = "RIGHT" if measurement.horizontal_deg > 0 else "LEFT"
            directions.append(f"{direction} ~{abs(measurement.horizontal_deg):.1f}°")
        if abs(measurement.vertical_deg) > threshold:
            direction = "DOWN" if measurement.vertical_deg > 0 else "UP"
            directions.append(f"{direction} ~{abs(measurement.vertical_deg):.1f}°")
        return (
            "Tilt the lens toward image " + " and ".join(directions)
            if directions else
            "Tilt is close. Check flat markers and calibration if the outline stays skewed."
        )


class CameraAlignmentObserver(ArucoObserver):
    """Reuse vision while deliberately estimating continuously instead of locking."""

    def _try_lock_workspace(self, car_markers, obstacle_markers):
        candidate = self._estimate_workspace_from_markers(car_markers, obstacle_markers)
        with self._workspace_lock:
            self._workspace = candidate
        return candidate

    def alignment_measurement(self):
        workspace = self.get_workspace_estimate()
        if not workspace.ready:
            return None, workspace.timestamp
        floor_corners = [
            (0, 0), (workspace.width_cm, 0),
            (workspace.width_cm, workspace.height_cm), (0, workspace.height_cm),
        ]
        camera_points = np.asarray([
            self._workspace_xy_to_camera(x, y, workspace) for x, y in floor_corners
        ], dtype=np.float64)
        pixels, _ = cv2.projectPoints(
            camera_points, np.zeros(3), np.zeros(3), self._K, self._D,
        )
        try:
            measurement = measure_alignment(workspace.normal_cam, pixels.reshape(4, 2))
        except ValueError:
            return None, workspace.timestamp
        return measurement, workspace.timestamp
