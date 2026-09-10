"""Floor-plane rectification for the live camera overlay."""
from typing import Callable

import cv2
import numpy as np


class BirdseyeProjection:
    """Map a calibrated floor rectangle to an image with uniform cm/pixel.

    The floor is rectified; objects above it retain perspective parallax.
    The same homography maps height-aware drawings to the warped image.
    """

    def __init__(self, width_cm: float, height_cm: float, project: Callable):
        if width_cm <= 0 or height_cm <= 0:
            raise ValueError("Bird's-eye view requires a locked workspace")
        self.width_cm = width_cm
        self.height_cm = height_cm
        self.project = project
        corners = [
            project(0.0, height_cm, 0.0),
            project(width_cm, height_cm, 0.0),
            project(width_cm, 0.0, 0.0),
            project(0.0, 0.0, 0.0),
        ]
        if any(point is None for point in corners):
            raise ValueError("Cannot project workspace corners for bird's-eye view")
        self.source = np.asarray(corners, dtype=np.float32)
        if not np.isfinite(self.source).all() or abs(cv2.contourArea(self.source)) < 1.0:
            raise ValueError("Invalid projected workspace for bird's-eye view")
        self.matrix = None
        self.output_size = None
        self._input_size = None

    def warp(self, frame):
        height, width = frame.shape[:2]
        if (width, height) != self._input_size:
            scale = min(width / self.width_cm, height / self.height_cm)
            out_w = max(2, int(self.width_cm * scale) // 2 * 2)
            out_h = max(2, int(self.height_cm * scale) // 2 * 2)
            target = np.float32([[0, 0], [out_w - 1, 0],
                                 [out_w - 1, out_h - 1], [0, out_h - 1]])
            self.matrix = cv2.getPerspectiveTransform(self.source, target)
            self.output_size = (out_w, out_h)
            self._input_size = (width, height)
        return cv2.warpPerspective(frame, self.matrix, self.output_size)

    def workspace_to_image(self, x: float, y: float, height_cm: float):
        if self.matrix is None:
            return None
        point = self.project(x, y, height_cm)
        if point is None:
            return None
        pixels = cv2.perspectiveTransform(
            np.array([[point]], dtype=np.float64), self.matrix
        )[0, 0]
        return float(pixels[0]), float(pixels[1])

    def image_to_workspace(self, px: float, py: float):
        if self.output_size is None:
            return None
        width, height = self.output_size
        return (
            px * self.width_cm / (width - 1),
            (1.0 - py / (height - 1)) * self.height_cm,
        )
