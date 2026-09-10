"""Live-camera canvas for perspective-aligned workspace drawings."""
from __future__ import annotations

import math
from pathlib import Path
import threading
from typing import Any, Callable, Dict, Optional, Tuple

from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QWidget,
)

from micromvp.core.models import CarState, WorkspaceConfig

from .canvas import MVPCanvas
from .window import MVPWindow


Point = Tuple[float, float]
WorkspaceProjector = Callable[[float, float, float], Optional[Point]]
ImageProjector = Callable[[float, float], Optional[Point]]


class CameraOverlayCanvas(MVPCanvas):
    """Render retained workspace drawings over a live camera frame.

    Workspace points are projected through the observer's calibrated camera
    model. The camera image is letterboxed without cropping, and the inverse
    projection is used for canvas goal clicks.
    """

    def __init__(
        self,
        workspace_config: WorkspaceConfig,
        workspace_to_image: WorkspaceProjector,
        image_to_workspace: ImageProjector,
        parent: Optional[QWidget] = None,
        workspace_boundary_height_cm: float = 0.0,
        birdseye: bool = False,
    ) -> None:
        self._birdseye = None
        if birdseye:
            from .birdseye import BirdseyeProjection

            self._birdseye = BirdseyeProjection(
                workspace_config.width, workspace_config.height, workspace_to_image,
            )
            workspace_to_image = self._birdseye.workspace_to_image
            image_to_workspace = self._birdseye.image_to_workspace
            workspace_boundary_height_cm = 0.0
        self._workspace_to_image = workspace_to_image
        self._image_to_workspace = image_to_workspace
        self._workspace_boundary_height_cm = max(
            0.0, float(workspace_boundary_height_cm)
        )
        self._camera_width = 0
        self._camera_height = 0
        self._camera_scale = 1.0
        self._camera_offset_x = 0.0
        self._camera_offset_y = 0.0
        super().__init__(workspace_config, parent)

        self.setBackgroundBrush(QColor(20, 20, 20))
        self._camera_item = QGraphicsPixmapItem()
        self._camera_item.setZValue(-1000.0)
        self._camera_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._camera_item.setTransformationMode(
            Qt.TransformationMode.SmoothTransformation
        )
        self._scene.addItem(self._camera_item)

    def set_camera_frame(self, frame: Any) -> None:
        """Replace the camera background from a BGR NumPy frame."""
        if frame is None or not hasattr(frame, "shape") or frame.ndim != 3:
            return
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0 or frame.shape[2] != 3:
            return

        if self._birdseye is not None:
            frame = self._birdseye.warp(frame)
            height, width = frame.shape[:2]

        image = QImage(
            frame.data,
            width,
            height,
            int(frame.strides[0]),
            QImage.Format.Format_BGR888,
        ).copy()
        dimensions_changed = (
            width != self._camera_width or height != self._camera_height
        )
        self._camera_width = width
        self._camera_height = height
        self._camera_item.setPixmap(QPixmap.fromImage(image))

        if dimensions_changed:
            self._update_transform_params()
            self._redraw_scene()

    def _update_transform_params(self) -> None:
        if self._camera_width <= 0 or self._camera_height <= 0:
            super()._update_transform_params()
            return

        view_rect = self.viewport().rect()
        view_width = view_rect.width()
        view_height = view_rect.height()
        if view_width <= 0 or view_height <= 0:
            return

        self._camera_scale = min(
            view_width / self._camera_width,
            view_height / self._camera_height,
        )
        displayed_width = self._camera_width * self._camera_scale
        displayed_height = self._camera_height * self._camera_scale
        self._camera_offset_x = (view_width - displayed_width) / 2.0
        self._camera_offset_y = (view_height - displayed_height) / 2.0
        self._view_h = float(view_height)
        self._scene.setSceneRect(0, 0, view_width, view_height)

        self._camera_item.setPos(self._camera_offset_x, self._camera_offset_y)
        self._camera_item.setScale(self._camera_scale)
        self._update_local_scale_estimate()

    def recording_image(self) -> Optional[QImage]:
        """Render only the camera rectangle, including retained overlays."""
        if self._camera_width < 2 or self._camera_height < 2:
            return None
        # Even dimensions for video codecs; no sidebar or letterbox margins.
        image = QImage(
            self._camera_width // 2 * 2,
            self._camera_height // 2 * 2,
            QImage.Format.Format_BGR888,
        )
        image.fill(QColor("black"))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._scene.render(
            painter, QRectF(image.rect()),
            self._camera_item.sceneBoundingRect(),
            Qt.AspectRatioMode.IgnoreAspectRatio,
        )
        painter.end()
        return image

    def _update_local_scale_estimate(self) -> None:
        """Estimate px/cm for inherited radius and car-size rendering."""
        center_x = self._ws_config.width / 2.0
        center_y = self._ws_config.height / 2.0
        center = self._workspace_to_image(center_x, center_y, 0.0)
        x_step = self._workspace_to_image(center_x + 1.0, center_y, 0.0)
        y_step = self._workspace_to_image(center_x, center_y + 1.0, 0.0)
        if center is None or x_step is None or y_step is None:
            return
        scales = [
            math.dist(center, x_step) * self._camera_scale,
            math.dist(center, y_step) * self._camera_scale,
        ]
        valid = [scale for scale in scales if math.isfinite(scale) and scale > 0]
        if valid:
            self._pixels_per_meter = sum(valid) / len(valid)

    def _image_to_scene_pixel(self, px: float, py: float) -> Point:
        return (
            self._camera_offset_x + px * self._camera_scale,
            self._camera_offset_y + py * self._camera_scale,
        )

    def workspace_to_pixel(self, wx: float, wy: float) -> Point:
        return self.workspace_to_pixel_at_height(wx, wy, 0.0)

    def workspace_to_pixel_at_height(
        self, wx: float, wy: float, height_cm: float
    ) -> Point:
        if self._camera_width <= 0 or self._camera_height <= 0:
            return super().workspace_to_pixel(wx, wy)
        projected = self._workspace_to_image(wx, wy, height_cm)
        if projected is None:
            return super().workspace_to_pixel(wx, wy)
        return self._image_to_scene_pixel(*projected)

    def set_workspace_boundary_height(self, height_cm: float) -> None:
        """Update the boundary projection plane and redraw it immediately."""
        self._workspace_boundary_height_cm = (
            0.0 if self._birdseye is not None else max(0.0, float(height_cm))
        )
        self._redraw_scene()

    def _update_drawing_geometry(self, item: Any, drawing: Dict[str, Any]) -> None:
        """Apply optional height to path and filled-region geometry."""
        super()._update_drawing_geometry(item, drawing)
        height_cm = float(drawing.get("projection_height_cm", 0.0))
        if not isinstance(item, QGraphicsPathItem):
            return
        if height_cm == 0.0:
            self._clip_region_to_workspace(item, drawing)
            return

        drawing_type = drawing.get("type")
        if drawing_type == "path":
            points = drawing.get("points", [])
            path = QPainterPath()
            if points:
                px, py = self.workspace_to_pixel_at_height(
                    points[0][0], points[0][1], height_cm
                )
                path.moveTo(px, py)
                for x, y in points[1:]:
                    px, py = self.workspace_to_pixel_at_height(x, y, height_cm)
                    path.lineTo(px, py)
            item.setPath(path)
            return

        if drawing_type != "region":
            return

        combined_path = QPainterPath()
        for region in drawing.get("regions", []):
            region_path = QPainterPath()
            region_path.setFillRule(Qt.FillRule.OddEvenFill)
            for points in [region.get("outer", [])] + list(
                region.get("holes", [])
            ):
                if len(points) < 3:
                    continue
                px, py = self.workspace_to_pixel_at_height(
                    points[0][0], points[0][1], height_cm
                )
                region_path.moveTo(px, py)
                for x, y in points[1:]:
                    px, py = self.workspace_to_pixel_at_height(x, y, height_cm)
                    region_path.lineTo(px, py)
                region_path.closeSubpath()
            if combined_path.isEmpty():
                combined_path = region_path
            else:
                combined_path = combined_path.united(region_path)
        item.setPath(combined_path)
        self._clip_region_to_workspace(item, drawing)

    def _clip_region_to_workspace(self, item, drawing) -> None:
        """Clip after projection so elevated scan geometry cannot spill out."""
        if (drawing.get("type") == "region" and drawing.get("clip_to_workspace")
                and isinstance(self._boundary_rect_item, QGraphicsPathItem)):
            item.setPath(item.path().intersected(self._boundary_rect_item.path()))

    def pixel_to_workspace(self, px: float, py: float) -> Point:
        if self._camera_width <= 0 or self._camera_height <= 0:
            return super().pixel_to_workspace(px, py)
        image_px = (px - self._camera_offset_x) / self._camera_scale
        image_py = (py - self._camera_offset_y) / self._camera_scale
        if not (
            0.0 <= image_px < self._camera_width
            and 0.0 <= image_py < self._camera_height
        ):
            return math.nan, math.nan
        projected = self._image_to_workspace(image_px, image_py)
        if projected is None:
            return math.nan, math.nan
        return projected

    def mousePressEvent(self, event: Any) -> None:
        """Ignore clicks outside the projected workspace."""
        if event.button() == Qt.MouseButton.LeftButton and self._camera_width > 0:
            scene_pos = self.mapToScene(event.pos())
            wx, wy = self.pixel_to_workspace(scene_pos.x(), scene_pos.y())
            if not (
                math.isfinite(wx)
                and math.isfinite(wy)
                and 0.0 <= wx <= self._ws_config.width
                and 0.0 <= wy <= self._ws_config.height
            ):
                event.ignore()
                return
        super().mousePressEvent(event)

    def _draw_workspace_boundary(self) -> None:
        """Draw the perspective workspace quadrilateral."""
        if self._boundary_rect_item is None:
            boundary = QGraphicsPathItem()
            boundary.setPen(
                QPen(QColor(255, 220, 0), 2, Qt.PenStyle.DashLine)
            )
            boundary.setZValue(100.0)
            self._scene.addItem(boundary)
            self._boundary_rect_item = boundary  # type: ignore[assignment]

        corners = [
            (0.0, 0.0),
            (self._ws_config.width, 0.0),
            (self._ws_config.width, self._ws_config.height),
            (0.0, self._ws_config.height),
        ]
        pixels = [
            self.workspace_to_pixel_at_height(
                x, y, self._workspace_boundary_height_cm
            )
            for x, y in corners
        ]
        if not all(math.isfinite(value) for point in pixels for value in point):
            return
        path = QPainterPath()
        path.moveTo(*pixels[0])
        for point in pixels[1:]:
            path.lineTo(*point)
        path.closeSubpath()
        self._boundary_rect_item.setPath(path)  # type: ignore[attr-defined]

    def update_cars(self, car_states: Dict[int, CarState]) -> None:
        """Keep state for the sidebar without covering the physical cars."""
        self._last_car_states = dict(car_states)
        for item in self._car_items.values():
            self._scene.removeItem(item)
        self._car_items.clear()


class CameraOverlayWindow(MVPWindow):
    """MVP window that consumes RealEnv frames without an OpenCV window."""

    def __init__(
        self,
        gui_config: Dict[str, Any],
        workspace_config: WorkspaceConfig,
        environment: Any,
        parent: Optional[QWidget] = None,
    ) -> None:
        self._environment = environment
        self._frame_lock = threading.Lock()
        self._latest_frame: Any = None
        super().__init__(gui_config, workspace_config, parent)
        view_name = (
            "MicroMVP Bird's-eye View"
            if gui_config.get("camera_view") == "birdseye"
            else "MicroMVP Camera Overlay"
        )
        self._view_name = view_name
        self._recorder = None
        recording_path = gui_config.get("recording_path")
        self._recording_path = Path(recording_path) if recording_path is not None else None
        self._recording_index = 0
        self.setWindowTitle(
            f"{view_name} — recording armed; click a goal"
            if self._recording_path is not None else view_name
        )

        self._camera_timer = QTimer(self)
        self._camera_timer.timeout.connect(self._consume_latest_frame)
        self._camera_timer.start(33)
        self._environment.set_frame_callback(self._receive_frame)

    def begin_goal_recording(self) -> None:
        """Start one video for an accepted goal click, closing any previous run."""
        if self._recording_path is None:
            return
        self.end_goal_recording()
        from .video_recorder import OverlayVideoRecorder

        while True:
            self._recording_index += 1
            path = (
                self._recording_path if self._recording_index == 1 else
                self._recording_path.with_name(
                    f"{self._recording_path.stem}_{self._recording_index:03d}"
                    f"{self._recording_path.suffix}"
                )
            )
            try:
                self._recorder = OverlayVideoRecorder(path)
                break
            except FileExistsError:
                continue
            except OSError as error:
                print(f"[Recording] Cannot start: {error}", flush=True)
                self.setWindowTitle(f"{self._view_name} — RECORDING FAILED")
                return
        self.setWindowTitle(f"{self._view_name} — recording {path}")
        self._consume_latest_frame()

    def end_goal_recording(self) -> None:
        """Capture the terminal overlay frame and finalize this run's MP4."""
        if self._recorder is None:
            return
        recorder = self._recorder
        self._recorder = None
        image = self._canvas.recording_image()
        if image is not None:
            recorder.submit(image)
        recorder.close()
        self.setWindowTitle(
            f"{self._view_name} — saved {recorder.path}; click another goal"
            if recorder.error is None else f"{self._view_name} — RECORDING FAILED"
        )

    def _create_canvas(self) -> CameraOverlayCanvas:
        canvas_config = self._gui_config.get("canvas", {})
        return CameraOverlayCanvas(
            self._ws_config,
            self._environment.workspace_to_image_pixel,
            self._environment.image_pixel_to_workspace,
            workspace_boundary_height_cm=float(
                canvas_config.get("workspace_boundary_height_cm", 0.0)
            ),
            birdseye=self._gui_config.get("camera_view") == "birdseye",
        )

    def set_workspace_boundary_on_floor(self, enabled: bool) -> None:
        """Toggle the boundary between its configured plane and the floor."""
        configured_height = float(
            self._gui_config.get("canvas", {}).get(
                "workspace_boundary_height_cm", 0.0
            )
        )
        self._canvas.set_workspace_boundary_height(
            0.0 if enabled else configured_height
        )

    def _receive_frame(self, frame: Any) -> None:
        # Keep only the newest frame so a busy GUI cannot accumulate latency.
        with self._frame_lock:
            self._latest_frame = frame.copy()

    def _consume_latest_frame(self) -> None:
        with self._frame_lock:
            frame = self._latest_frame
            self._latest_frame = None
        if frame is not None:
            self._canvas.set_camera_frame(frame)
        if self._recorder is not None:
            image = self._canvas.recording_image()
            if image is not None:
                self._recorder.submit(image)
                if self._recorder.error is not None:
                    print(f"[Recording] Failed: {self._recorder.error}", flush=True)
                    self.end_goal_recording()

    def closeEvent(self, event: Any) -> None:
        self._camera_timer.stop()
        self._environment.set_frame_callback(None)
        self.end_goal_recording()
        super().closeEvent(event)
