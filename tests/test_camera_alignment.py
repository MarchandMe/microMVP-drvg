"""Alignment direction, shape tests, and green-state freshness checks."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from micromvp.env.real_env.camera_alignment import (
    AlignmentMonitor, CameraAlignmentObserver, measure_alignment,
)
from micromvp.env.real_env.observer import ObserverConfig, WorkspaceEstimate
from micromvp.gui.camera_alignment import CameraAlignmentWindow


RECTANGLE = np.array([[20, 20], [300, 20], [300, 160], [20, 160]], dtype=float)


def measurement(normal=(0, 0, 1)):
    return measure_alignment(normal, RECTANGLE)


@pytest.mark.parametrize("normal,direction", [
    ((0.2, 0, 1), "RIGHT"), ((-0.2, 0, 1), "LEFT"),
    ((0, 0.2, 1), "DOWN"), ((0, -0.2, 1), "UP"),
])
def test_guidance_uses_image_axes(normal, direction):
    monitor = AlignmentMonitor()
    result = measurement(normal)
    assert direction in monitor.guidance(result)
    assert result.tilt_deg == pytest.approx(np.degrees(np.arctan(0.2)))
    # Flipping the plane-normal sign does not reverse physical guidance.
    assert monitor.guidance(measurement(-np.array(normal))) == monitor.guidance(result)


def test_rectangle_metrics_ignore_roll_but_reject_trapezoid():
    theta = np.radians(20)
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    rolled = measure_alignment((0, 0, 1), RECTANGLE @ rotation.T)
    assert rolled.corner_error_deg == pytest.approx(0)
    assert rolled.opposite_edge_error == pytest.approx(0)
    trapezoid = measure_alignment((0, 0, 1), [[80, 20], [240, 20], [300, 160], [20, 160]])
    assert trapezoid.opposite_edge_error > .05
    assert trapezoid.corner_error_deg > 3
    assert AlignmentMonitor().update(trapezoid, 10, 10) == "adjust"


def test_green_requires_continuous_fresh_frames_and_clears_on_motion_or_loss():
    monitor = AlignmentMonitor(hold_seconds=1.0)
    good = measurement()
    assert monitor.update(good, 10, 10) == "hold"
    assert monitor.update(good, 10, 10.4) == "hold"
    assert monitor.elapsed == 0  # A frozen image cannot advance the hold.
    assert monitor.update(good, 10.4, 10.4) == "hold"
    assert monitor.update(good, 10.8, 10.8) == "hold"
    assert monitor.update(good, 11.1, 11.1) == "aligned"
    assert monitor.update(measurement((.2, 0, 1)), 11.2, 11.2) == "adjust"
    assert monitor.update(good, 11.3, 11.3) == "hold"
    assert monitor.update(good, 11.3, 12) == "waiting"
    assert monitor.update(None, 0, 12.1) == "waiting"
    assert monitor.update(good, 12.2, 12.2) == "hold"


def test_gap_in_observations_restarts_hold():
    monitor = AlignmentMonitor(hold_seconds=1)
    good = measurement()
    monitor.update(good, 1, 1)
    assert monitor.update(good, 5, 5) == "hold"
    assert monitor.elapsed == 0


def test_alignment_observer_keeps_updating_after_good_pose(monkeypatch):
    observer = CameraAlignmentObserver(ObserverConfig())
    first = WorkspaceEstimate(ready=True, width_cm=100, height_cm=50, normal_cam=(0, 0, 1))
    second = replace(first, normal_cam=(.2, 0, .98), width_cm=90)
    estimator = Mock(side_effect=[first, second])
    monkeypatch.setattr(observer, "_estimate_workspace_from_markers", estimator)
    observer._try_lock_workspace({}, {})
    assert not observer.is_workspace_ready()
    observer._try_lock_workspace({}, {})
    assert observer.get_workspace_estimate().width_cm == 90


def test_gui_turns_green_then_loses_green_when_markers_disappear(monkeypatch):
    app = QApplication.instance() or QApplication([])
    observer = Mock()
    observer.alignment_measurement.return_value = (measurement(), 10)
    observer.get_workspace_corner_progress.return_value = (30, 30, False)
    clock = [10.0]
    monkeypatch.setattr("micromvp.gui.camera_alignment.time.time", lambda: clock[0])
    window = CameraAlignmentWindow(observer, AlignmentMonitor(hold_seconds=.1))
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    window.receive_frame(frame)
    window.update_view()
    assert "HOLD STILL" in window.status.text()
    clock[0] = 10.2
    observer.alignment_measurement.return_value = (measurement(), 10.2)
    window.receive_frame(frame)
    window.update_view()
    assert window.status.text() == "STOP — ALIGNED"
    image = window.preview.pixmap().toImage()
    center = image.pixelColor(image.width() // 4, image.height() // 2)
    assert center.green() > center.red()
    observer.alignment_measurement.return_value = (None, 0)
    clock[0] = 10.3
    window.update_view()
    assert "STOP" not in window.status.text()
    window.close()
    observer.stop.assert_called_once()


@pytest.mark.parametrize("normal,polygon", [
    ((0, 0, 0), RECTANGLE), ((np.nan, 0, 1), RECTANGLE),
    ((0, 0, 1), np.zeros((4, 2))),
])
def test_invalid_geometry_is_rejected(normal, polygon):
    with pytest.raises(ValueError):
        measure_alignment(normal, polygon)
