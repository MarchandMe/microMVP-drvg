"""Regression tests for real-navigation steering behavior."""
from __future__ import annotations

import pytest

from micromvp.controller import NavigationController
from micromvp.core.models import RobotObservation, WorkspaceConfig


@pytest.fixture
def workspace() -> WorkspaceConfig:
    return WorkspaceConfig(
        width=80.0,
        height=42.0,
        car_width=4.2,
        car_height=4.8,
        offset_w=2.1,
        offset_h=4.5,
        wheel_base=4.2,
        max_wheel_speed=10.0,
        frequency=30.0,
        car_id_list=[1],
    )


@pytest.mark.parametrize(
    ("y", "expected_sign"),
    [(0.6, -1.0), (-0.6, 1.0)],
)
def test_cross_track_term_reinforces_path_recovery(
    workspace: WorkspaceConfig,
    y: float,
    expected_sign: float,
) -> None:
    controller = NavigationController(
        1,
        workspace,
        lookahead_distance=2.4,
        max_speed=0.2,
    )
    controller.set_path([(0.0, 0.0), (20.0, 0.0)])

    action = controller.step(
        RobotObservation(
            robot_id=1,
            x=0.0,
            y=y,
            theta=0.0,
            timestamp=1.0,
        )
    )
    metadata = controller.car_state.metadata
    pure_pursuit = metadata["curvature_pp"]
    combined = metadata["curvature_cmd"]

    assert pure_pursuit * expected_sign > 0.0
    assert combined * expected_sign > pure_pursuit * expected_sign
    if y > 0.0:
        assert action.left_speed > action.right_speed
    else:
        assert action.right_speed > action.left_speed


def test_heading_filter_wraps_across_zero(workspace: WorkspaceConfig) -> None:
    controller = NavigationController(
        1,
        workspace,
        heading_filter_alpha=0.5,
    )
    controller.update(
        RobotObservation(robot_id=1, x=0.0, y=0.0, theta=359.0, timestamp=1.0)
    )
    controller.update(
        RobotObservation(robot_id=1, x=0.0, y=0.0, theta=1.0, timestamp=1.1)
    )

    assert controller.car_state.theta == pytest.approx(0.0)
    assert controller.car_state.metadata["measured_theta"] == 1.0
    assert controller.car_state.metadata["filtered_theta"] == pytest.approx(0.0)


def test_heading_filter_resets_after_observation_gap(
    workspace: WorkspaceConfig,
) -> None:
    controller = NavigationController(
        1,
        workspace,
        heading_filter_alpha=0.1,
    )
    controller.update(
        RobotObservation(robot_id=1, x=0.0, y=0.0, theta=10.0, timestamp=1.0)
    )
    controller.update(
        RobotObservation(robot_id=1, x=0.0, y=0.0, theta=80.0, timestamp=2.0)
    )

    assert controller.car_state.theta == pytest.approx(80.0)


@pytest.mark.parametrize(
    ("current_heading", "target_heading", "expected_direction"),
    [
        (359.0, 5.0, 1.0),
        (5.0, 359.0, -1.0),
    ],
)
def test_rotation_takes_shortest_direction_across_zero(
    workspace: WorkspaceConfig,
    current_heading: float,
    target_heading: float,
    expected_direction: float,
) -> None:
    controller = NavigationController(1, workspace)
    controller.rotate_to(target_heading)

    action = controller.step(
        RobotObservation(
            robot_id=1,
            x=20.0,
            y=20.0,
            theta=current_heading,
            timestamp=1.0,
        )
    )

    assert controller.car_state.metadata["rotation_error_deg"] == pytest.approx(
        6.0 * expected_direction
    )
    assert action.left_speed * expected_direction < 0.0
    assert action.right_speed * expected_direction > 0.0


def test_rotation_across_zero_respects_tolerance(
    workspace: WorkspaceConfig,
) -> None:
    controller = NavigationController(1, workspace)
    controller.rotate_to(0.0)

    action = controller.step(
        RobotObservation(
            robot_id=1,
            x=20.0,
            y=20.0,
            theta=359.0,
            timestamp=1.0,
        )
    )

    assert action.left_speed == 0.0
    assert action.right_speed == 0.0
    assert controller.car_state.status_label == "ROTATION_STABLE"
    assert controller.car_state.metadata["rotation_error_deg"] == pytest.approx(1.0)


def test_rotation_stops_at_measured_heading_without_waiting_for_ema(workspace):
    controller = NavigationController(1,workspace,heading_filter_alpha=.1)
    controller.update(RobotObservation(1,20,20,0,timestamp=1))
    controller.rotate_to(20)
    action = controller.step(RobotObservation(1,20,20,20,timestamp=1.033))
    assert action.left_speed == 0 and action.right_speed == 0
    assert controller.car_state.status_label == "ROTATION_STABLE"
    assert controller.car_state.theta == pytest.approx(20)
    assert controller.car_state.metadata["rotation_error_deg"] == pytest.approx(0)


def test_stale_filtered_heading_cannot_hide_rotation_overshoot(workspace):
    controller = NavigationController(1,workspace,heading_filter_alpha=.1)
    controller.update(RobotObservation(1,20,20,0,timestamp=1))
    controller.rotate_to(0)
    action = controller.step(RobotObservation(1,20,20,10,timestamp=1.033))
    assert controller.car_state.status_label == "ROTATING"
    assert controller.car_state.metadata["rotation_error_deg"] == pytest.approx(-10)
    assert action.left_speed > 0 and action.right_speed < 0
