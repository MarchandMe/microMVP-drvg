"""Startup corner averaging must reduce jitter without mixing moving views."""
from dataclasses import replace

import cv2
import numpy as np

from micromvp.env.real_env.observer import (
    ArucoObserver, ObserverConfig, _MarkerInfo, _WorkspaceCornerFilter,
)


def corners(x=0.0):
    return np.array([[x, 0], [x + 20, 0], [x + 20, 20], [x, 20]], dtype=float)


def test_noise_is_averaged_and_car_obstacle_ids_are_distinct():
    smoother = _WorkspaceCornerFilter(3, 2.0)
    key = ("car", 3)
    obstacle = ("obstacle", 3)
    assert smoother.update({key: corners(-0.3), obstacle: corners(100)}) is None
    assert smoother.update({key: corners(0.3), obstacle: corners(100)}) is None
    result = smoother.update({key: corners(), obstacle: corners(100)})
    np.testing.assert_allclose(result[key], corners(), atol=1e-12)
    np.testing.assert_allclose(result[obstacle], corners(100))


def test_camera_motion_restarts_the_whole_window():
    smoother = _WorkspaceCornerFilter(3, 2.0)
    old = {("car", 3): corners(), ("obstacle", 6): corners(100)}
    for _ in range(3):
        smoother.update(old)
    moved = {key: points + 10 for key, points in old.items()}
    assert smoother.update(moved) is None
    assert smoother.motion_detected
    assert smoother.sample_count == 1
    assert smoother.update(moved) is None
    result = smoother.update(moved)
    for key in moved:
        np.testing.assert_allclose(result[key], moved[key])


def test_missing_marker_discards_previously_collected_view():
    smoother = _WorkspaceCornerFilter(3, 2.0)
    markers = {("car", 3): corners(), ("obstacle", 6): corners(100)}
    for _ in range(3):
        smoother.update(markers)
    assert smoother.update({("car", 3): corners()}) is None
    assert smoother.sample_count == 1
    assert smoother.update({}) is None
    assert smoother.sample_count == 0


def test_workspace_waits_for_averaged_corners_and_freezes_after_lock():
    config = ObserverConfig(workspace_corner_smoothing_frames=3, workspace_lock_frames=3)
    observer = ArucoObserver(config)
    observer._K = np.array([[960., 0, 640], [0, 960., 360], [0, 0, 1]])
    observer._D = np.zeros(5)
    observer._frame_size = (1280, 720)
    rvec = np.array([np.pi, 0.0, 0.0])
    tvec = np.array([0.0, 0.0, 1.0])
    points, _ = cv2.projectPoints(
        observer._marker_corners_in_marker_frame(config.car_marker_size_mm / 1000),
        rvec, tvec, observer._K, observer._D,
    )
    marker = _MarkerInfo(3, points.reshape(4, 2), rvec, tvec,
                         cv2.Rodrigues(rvec)[0][:, 2], 5.0)
    original_corners = marker.image_corners.copy()
    for _ in range(4):
        observer._try_lock_workspace({3: marker}, {})
        assert not observer.is_workspace_ready()
    locked = observer._try_lock_workspace({3: marker}, {})
    assert observer.is_workspace_ready()
    # Workspace filtering must not mutate the raw markers used for tracking.
    np.testing.assert_array_equal(marker.image_corners, original_corners)
    moved = replace(marker, image_corners=original_corners + 50)
    assert observer._try_lock_workspace({3: moved}, {}) is locked


def test_motion_clears_partial_workspace_lock():
    config = ObserverConfig(workspace_corner_smoothing_frames=2, workspace_lock_frames=3)
    observer = ArucoObserver(config)
    observer._K = np.array([[960., 0, 640], [0, 960., 360], [0, 0, 1]])
    observer._D = np.zeros(5)
    observer._frame_size = (1280, 720)
    rvec = np.array([np.pi, 0., 0.])
    tvec = np.array([0., 0., 1.])
    points, _ = cv2.projectPoints(
        observer._marker_corners_in_marker_frame(config.car_marker_size_mm / 1000),
        rvec, tvec, observer._K, observer._D,
    )
    marker = _MarkerInfo(3, points.reshape(4, 2), rvec, tvec,
                         cv2.Rodrigues(rvec)[0][:, 2], 5.)
    observer._try_lock_workspace({3: marker}, {})
    observer._try_lock_workspace({3: marker}, {})
    assert observer.get_workspace_lock_progress() == (1, 3)
    observer._try_lock_workspace({3: replace(marker, image_corners=marker.image_corners + 20)}, {})
    assert observer.get_workspace_lock_progress() == (0, 3)
    assert not observer.get_workspace_estimate().ready
