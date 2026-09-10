"""Hardware-free checks for camera-only video recording."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import time

import cv2
import numpy as np
import pytest
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import QApplication

from micromvp.core.models import WorkspaceConfig
from micromvp.gui.camera_overlay import CameraOverlayCanvas
from micromvp.gui.video_recorder import OverlayVideoRecorder


def test_camera_crop_preserves_overlay_when_window_resizes():
    app = QApplication.instance() or QApplication([])
    ws = WorkspaceConfig(100, 60, 4, 5, 2, 2, 4, 10, 30, [])
    canvas = CameraOverlayCanvas(
        ws, lambda x, y, h: (x * 3.2, y * 3.0),
        lambda x, y: (x / 3.2, y / 3.0),
    )
    canvas.resize(800, 600)
    canvas.show()
    app.processEvents()
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    frame[:, :] = (200, 80, 30)
    canvas.set_camera_frame(frame)
    canvas.update_drawings([{
        "uuid": "goal", "type": "point", "position": (50, 30),
        "radius": 8, "color": "#ff0000", "fill": "#ff0000",
    }])
    for width, height in [(800, 600), (600, 800)]:
        canvas.resize(width, height)
        app.processEvents()
        image = canvas.recording_image()
        assert (image.width(), image.height()) == (320, 180)
        assert image.pixelColor(30, 30) == QColor(30, 80, 200)
        center = image.pixelColor(160, 90)
        assert center.red() > 240 and center.blue() < 20
    canvas.close()


def test_mp4_round_trip_repeats_frames_and_closes(tmp_path):
    path = tmp_path / "demo.mp4"
    recorder = OverlayVideoRecorder(path)
    image = QImage(320, 180, QImage.Format.Format_BGR888)
    image.fill(QColor(230, 30, 20))
    recorder.submit(image)
    time.sleep(0.3)
    recorder.close()
    assert recorder.error is None
    reader = cv2.VideoCapture(str(path))
    assert reader.isOpened()
    assert reader.get(cv2.CAP_PROP_FPS) == pytest.approx(30)
    assert reader.get(cv2.CAP_PROP_FRAME_COUNT) >= 2
    ok, frame = reader.read()
    reader.release()
    assert ok and frame.shape == (180, 320, 3)
    assert frame[90, 160, 2] > 210
    with pytest.raises(FileExistsError):
        OverlayVideoRecorder(path)


def test_encoder_failure_is_reported_and_can_close(tmp_path, monkeypatch):
    class UnavailableWriter:
        def isOpened(self):
            return False
        def release(self):
            pass
    monkeypatch.setattr(cv2, "VideoWriter", lambda *args: UnavailableWriter())
    recorder = OverlayVideoRecorder(tmp_path / "failed.mp4")
    image = QImage(32, 32, QImage.Format.Format_BGR888)
    image.fill(QColor("black"))
    recorder.submit(image)
    recorder._thread.join(timeout=2)
    recorder.close()
    assert isinstance(recorder.error, RuntimeError)


def test_close_flushes_the_final_frame(tmp_path):
    path = tmp_path / "last-frame.mp4"
    recorder = OverlayVideoRecorder(path)
    image = QImage(64, 48, QImage.Format.Format_BGR888)
    image.fill(QColor("red"))
    recorder.submit(image)
    time.sleep(0.08)
    image.fill(QColor("blue"))
    recorder.submit(image)
    recorder.close()
    assert recorder.error is None
    reader = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        frames.append(frame)
    reader.release()
    assert frames
    assert frames[-1][24, 32, 0] > 200
    assert frames[-1][24, 32, 2] < 40


def test_goal_recording_is_idle_until_started_and_numbers_later_runs(tmp_path):
    from types import SimpleNamespace
    from micromvp.gui.camera_overlay import CameraOverlayWindow

    app = QApplication.instance() or QApplication([])
    env = SimpleNamespace(
        workspace_to_image_pixel=lambda x, y, h: (x * 3.2, y * 3),
        image_pixel_to_workspace=lambda x, y: (x / 3.2, y / 3),
        set_frame_callback=lambda callback: None,
    )
    path = tmp_path / "goals.mp4"
    window = CameraOverlayWindow(
        {"canvas": {}, "control_panel": [], "recording_path": path},
        WorkspaceConfig(100, 60, 4, 5, 2, 2, 4, 10, 30, []), env,
    )
    window.show()
    app.processEvents()
    frame = np.full((180, 320, 3), 100, dtype=np.uint8)
    window._receive_frame(frame)
    window._consume_latest_frame()
    assert window._recorder is None and not path.exists()

    window.begin_goal_recording()
    first = window._recorder
    assert first.path == path
    window.end_goal_recording()
    assert window._recorder is None and not first._thread.is_alive()
    reader = cv2.VideoCapture(str(path))
    assert reader.read()[0]
    reader.release()
    window.end_goal_recording()  # Repeated idle updates do not create files.
    assert len(list(tmp_path.glob("*.mp4"))) == 1

    existing = tmp_path / "goals_002.mp4"
    existing.write_bytes(b"preserve")
    window.begin_goal_recording()
    second = window._recorder
    assert second.path.name == "goals_003.mp4"
    # Selecting another goal finalizes the interrupted run first.
    window.begin_goal_recording()
    assert not second._thread.is_alive()
    assert window._recorder.path.name == "goals_004.mp4"
    assert existing.read_bytes() == b"preserve"
    last = window._recorder
    window.close()
    assert not last._thread.is_alive()


def test_recording_disabled_does_not_create_output(tmp_path):
    from types import SimpleNamespace
    from micromvp.gui.camera_overlay import CameraOverlayWindow
    app = QApplication.instance() or QApplication([])
    env = SimpleNamespace(
        workspace_to_image_pixel=lambda x, y, h: (x, y),
        image_pixel_to_workspace=lambda x, y: (x, y),
        set_frame_callback=lambda callback: None,
    )
    window = CameraOverlayWindow(
        {"canvas": {}, "control_panel": []},
        WorkspaceConfig(100, 60, 4, 5, 2, 2, 4, 10, 30, []), env,
    )
    window.begin_goal_recording()
    assert window._recorder is None
    window.close()
