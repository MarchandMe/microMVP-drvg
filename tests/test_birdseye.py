"""Bird's-eye camera, overlays and clicks must share the floor transform."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from micromvp.gui.birdseye import BirdseyeProjection
from micromvp.gui.camera_overlay import CameraOverlayCanvas
from micromvp.core.models import WorkspaceConfig


def projector():
    floor = np.float32([[0, 0], [100, 0], [100, 50], [0, 50]])
    image = np.float32([[40, 30], [280, 10], [310, 190], [10, 180]])
    matrix = cv2.getPerspectiveTransform(floor, image)

    def project(x, y, height):
        # A height offset lets the test check non-floor overlay mapping too.
        p = cv2.perspectiveTransform(
            np.float64([[[x, 50 - y]]]), matrix
        )[0, 0]
        return float(p[0] + height), float(p[1] - height)
    return project


def test_rectifies_trapezoid_preserving_floor_scale_and_color():
    project = projector()
    view = BirdseyeProjection(100, 50, project)
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    frame[:] = (180, 90, 40)
    center = tuple(round(v) for v in project(50, 25, 0))
    cv2.circle(frame, center, 8, (0, 0, 255), -1)
    warped = view.warp(frame)
    assert warped.shape == (160, 320, 3)
    assert warped[80, 160, 2] > 240
    np.testing.assert_allclose(view.workspace_to_image(0, 50, 0), (0, 0), atol=1e-4)
    np.testing.assert_allclose(view.workspace_to_image(100, 0, 0), (319, 159), atol=1e-4)
    for point in [(0, 0), (20, 40), (100, 50)]:
        image_point = view.workspace_to_image(*point, 0)
        np.testing.assert_allclose(view.image_to_workspace(*image_point), point, atol=1e-4)
    elevated = view.workspace_to_image(50, 25, 4)
    expected = cv2.perspectiveTransform(
        np.float64([[project(50, 25, 4)]]), view.matrix
    )[0, 0]
    np.testing.assert_allclose(elevated, expected)
    assert elevated != view.workspace_to_image(50, 25, 0)


def test_rectangular_canvas_recording_and_clicks_survive_resize():
    app = QApplication.instance() or QApplication([])
    ws = WorkspaceConfig(100, 50, 4, 5, 2, 2, 4, 10, 30, [])
    canvas = CameraOverlayCanvas(
        ws, projector(), lambda x, y: None,
        workspace_boundary_height_cm=4, birdseye=True,
    )
    canvas.show()
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    canvas.set_camera_frame(frame)
    canvas.update_drawings([{
        "uuid": "goal", "type": "point", "position": (50, 25),
        "radius": 8, "color": "#ff0000", "fill": "#ff0000",
    }])
    for size in [(800, 600), (600, 800)]:
        canvas.resize(*size)
        app.processEvents()
        pixel = canvas.workspace_to_pixel(40, 20)
        np.testing.assert_allclose(canvas.pixel_to_workspace(*pixel), (40, 20), atol=1e-4)
        image = canvas.recording_image()
        assert (image.width(), image.height()) == (320, 160)
        assert image.pixelColor(160, 80).red() > 240
        top_left = canvas.workspace_to_pixel(0, 50)
        top_right = canvas.workspace_to_pixel(100, 50)
        assert top_left[1] == pytest.approx(top_right[1], abs=1e-3)
    canvas.set_workspace_boundary_height(4)
    assert canvas._workspace_boundary_height_cm == 0
    canvas.close()


def test_invalid_projection_fails_before_recording():
    with pytest.raises(ValueError, match="project workspace"):
        BirdseyeProjection(100, 50, lambda *args: None)
    with pytest.raises(ValueError, match="Invalid projected"):
        BirdseyeProjection(100, 50, lambda *args: (1, 1))


@pytest.mark.parametrize("birdseye", [False, True])
@pytest.mark.parametrize("height", [0.0, 4.0])
def test_scan_shading_is_clipped_after_projection_and_keeps_obstacle_holes(birdseye, height):
    from PyQt6.QtCore import QPointF

    app = QApplication.instance() or QApplication([])
    ws = WorkspaceConfig(100, 50, 4, 5, 2, 2, 4, 10, 30, [])
    canvas = CameraOverlayCanvas(
        ws, projector(), lambda x, y: None,
        workspace_boundary_height_cm=4, birdseye=birdseye,
    )
    canvas.show()
    canvas.set_camera_frame(np.zeros((200, 320, 3), dtype=np.uint8))
    scan = {
        "uuid": "scanned_area", "type": "region", "clip_to_workspace": True,
        "projection_height_cm": height,
        "regions": [{
            "outer": [(-10,-10),(110,-10),(110,60),(-10,60)],
            "holes": [[(40,20),(60,20),(60,30),(40,30)]],
        }],
        "color": "#D6A20B", "fill": "#40D6A20B", "width": 1,
    }
    canvas.update_drawings([scan])
    for size in [(800,600),(600,800)]:
        canvas.resize(*size)
        app.processEvents()
        region = canvas._drawing_items_cache["scanned_area"].path()
        boundary = canvas._boundary_rect_item.path()
        assert region.subtracted(boundary).isEmpty()
        hole = QPointF(*canvas.workspace_to_pixel_at_height(50,25,height))
        inside = QPointF(*canvas.workspace_to_pixel_at_height(20,20,height))
        assert not region.contains(hole)
        assert region.contains(inside)
    canvas.set_workspace_boundary_height(0)
    region = canvas._drawing_items_cache["scanned_area"].path()
    assert region.subtracted(canvas._boundary_rect_item.path()).isEmpty()
    canvas.close()
