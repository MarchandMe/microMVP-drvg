"""Path-planning tests that need the external RVG planner.

Excluded from the default run: `import rvg` succeeds against the bare rvg/
directory as a namespace package, so importorskip cannot tell whether the
compiled extension is really usable. Run explicitly with:

    pytest -m rvg
"""
import math

import pytest

pytestmark = pytest.mark.rvg

from micromvp.controller.base import Controller
from micromvp.coordinator.navigation_coordinator.navigation_coordinator import (
    NavigationCoordinator,
)
from micromvp.core.models import CarState, WorkspaceConfig
from micromvp.planner.dynamic_rvg import DynamicRVGSession, DynamicRVGSettings


class _DummyController(Controller):
    def __init__(self, car_id: int, x: float, y: float, theta: float) -> None:
        self._car_state = CarState(car_id=car_id, x=x, y=y, theta=theta)

    def step(self, observation):
        raise NotImplementedError

    @property
    def car_state(self) -> CarState:
        return self._car_state

    def update(self, observation):
        raise NotImplementedError

    def calculate_action(self):
        raise NotImplementedError


def _make_coordinator(monkeypatch: pytest.MonkeyPatch) -> NavigationCoordinator:
    monkeypatch.setattr(NavigationCoordinator, "_start_webserver", lambda self: None)
    ws = WorkspaceConfig(
        width=50.0,
        height=30.0,
        car_width=4.2,
        car_height=4.8,
        offset_w=2.1,
        offset_h=4.5,
        wheel_base=4.2,
        max_wheel_speed=10.0,
        frequency=30.0,
        car_id_list=[1],
    )
    controllers = {1: _DummyController(1, 10.0, 15.0, 0.0)}
    return NavigationCoordinator(
        ws,
        controllers,
        active_robot_id=1,
        webserver_port=0,
        robot_geometry_scale=1.0,
    )


def test_plan_path_preserves_initial_rvg_rotation(monkeypatch: pytest.MonkeyPatch):
    coord = _make_coordinator(monkeypatch)
    obstacles = [
        [(20.0, 0.0), (22.0, 0.0), (22.0, 12.0), (20.0, 12.0)],
        [(20.0, 18.0), (22.0, 18.0), (22.0, 30.0), (20.0, 30.0)],
    ]

    path = coord._plan_path((10.0, 15.0, 0.0), (40.0, 15.0, 0.0), obstacles)

    assert path is not None
    assert len(path) >= 3
    assert path[0] == (10.0, 15.0)
    assert path[-1] == (40.0, 15.0)

    first_heading_deg = math.degrees(
        math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0])
    )
    assert 3.0 <= first_heading_deg <= 7.0


def test_default_robot_geometry_uses_conservative_front_extent(
    monkeypatch: pytest.MonkeyPatch,
):
    coord = _make_coordinator(monkeypatch)

    front_extent = max(x for x, _ in coord._robot_geometry)
    upper_half_width = max(y for _, y in coord._robot_geometry)
    lower_half_width = abs(min(y for _, y in coord._robot_geometry))

    assert front_extent >= 2.1
    assert front_extent >= upper_half_width
    assert front_extent >= lower_half_width


def test_dynamic_rvg_merges_intersecting_world_obstacles() -> None:
    workspace = WorkspaceConfig(
        width=20.0,
        height=20.0,
        car_width=1.0,
        car_height=1.0,
        offset_w=0.5,
        offset_h=0.5,
        wheel_base=1.0,
        max_wheel_speed=1.0,
        frequency=30.0,
        car_id_list=[1],
    )
    overlapping_obstacles = [
        [(5.0, 5.0), (9.0, 5.0), (9.0, 9.0), (5.0, 9.0)],
        [(8.5, 5.0), (12.0, 5.0), (12.0, 9.0), (8.5, 9.0)],
    ]

    session = DynamicRVGSession(
        workspace,
        overlapping_obstacles,
        DynamicRVGSettings(resolution=4),
    )

    assert session.planner is not None


def test_dynamic_rvg_merges_visible_obstacle_into_workspace_border() -> None:
    workspace = WorkspaceConfig(
        width=20.0,
        height=20.0,
        car_width=1.0,
        car_height=1.0,
        offset_w=0.5,
        offset_h=0.5,
        wheel_base=1.0,
        max_wheel_speed=1.0,
        frequency=30.0,
        car_id_list=[1],
    )

    session = DynamicRVGSession(
        workspace,
        [[(-0.5, 5.0), (2.0, 5.0), (2.0, 8.0), (-0.5, 8.0)]],
        DynamicRVGSettings(resolution=4),
    )

    assert session.planner is not None
    assert min(x for x, _ in session._obstacle_points[0]) > 0.0

    session.initialize((5.0, 10.0, 0.0), (15.0, 10.0, 0.0))
    plan = session.step()

    assert plan.status.name == "GoalPathAvailable"
    assert plan.final_segment


def test_actual_size_pose_can_be_valid_inside_planning_margin() -> None:
    workspace = WorkspaceConfig(
        width=20.0,
        height=20.0,
        car_width=1.0,
        car_height=1.0,
        offset_w=0.5,
        offset_h=0.5,
        wheel_base=1.0,
        max_wheel_speed=1.0,
        frequency=30.0,
        car_id_list=[1],
    )
    session = DynamicRVGSession(
        workspace,
        [[(6.0, 4.0), (8.0, 4.0), (8.0, 6.0), (6.0, 6.0)]],
        DynamicRVGSettings(resolution=4, robot_geometry_scale=1.2),
    )
    pose = (5.45, 5.0, 0.0)

    assert session.pose_is_valid(pose, robot_geometry_scale=1.0)
    assert not session.pose_is_valid(pose, robot_geometry_scale=1.2)


def test_empty_world_uses_direct_final_goal_path() -> None:
    workspace = WorkspaceConfig(
        width=20.0,
        height=20.0,
        car_width=1.0,
        car_height=1.0,
        offset_w=0.5,
        offset_h=0.5,
        wheel_base=1.0,
        max_wheel_speed=1.0,
        frequency=30.0,
        car_id_list=[1],
    )
    session = DynamicRVGSession(
        workspace,
        [],
        DynamicRVGSettings(resolution=4),
    )
    start = (3.0, 10.0, 0.0)
    goal = (17.0, 10.0, 90.0)

    session.initialize(start, goal)
    plan = session.step()

    assert plan.status.name == "GoalPathAvailable"
    assert plan.final_segment
    assert plan.temporary_goal is None
    assert plan.controller_path == pytest.approx([start[:2], goal[:2]])
    assert len(plan.configurations) == 2
    assert plan.configurations[0].getTheta() == pytest.approx(0.0)
    assert plan.configurations[-1].getTheta() == pytest.approx(math.pi / 2.0)


def test_clear_direct_route_takes_precedence_over_rvg_detour() -> None:
    workspace = WorkspaceConfig(
        width=79.0,
        height=43.3,
        car_width=4.2,
        car_height=4.8,
        offset_w=2.1,
        offset_h=4.5,
        wheel_base=4.2,
        max_wheel_speed=10.0,
        frequency=30.0,
        car_id_list=[3],
    )
    session = DynamicRVGSession(
        workspace,
        [[(30.0, 5.0), (35.0, 5.0), (35.0, 10.0), (30.0, 10.0)]],
        DynamicRVGSettings(resolution=36, robot_geometry_scale=1.2),
    )
    start = (15.0, 20.0, 0.0)
    goal = (65.0, 20.0, 0.0)

    session.initialize(start, goal)
    plan = session.step()

    assert plan.status.name == "GoalPathAvailable"
    assert plan.final_segment
    assert plan.temporary_goal is None
    assert plan.controller_path == pytest.approx([start[:2], goal[:2]])


def test_blocked_direct_route_still_uses_temporary_goal() -> None:
    workspace = WorkspaceConfig(
        width=79.0,
        height=43.3,
        car_width=4.2,
        car_height=4.8,
        offset_w=2.1,
        offset_h=4.5,
        wheel_base=4.2,
        max_wheel_speed=10.0,
        frequency=30.0,
        car_id_list=[3],
    )
    session = DynamicRVGSession(
        workspace,
        [[(38.0, 15.0), (42.0, 15.0), (42.0, 25.0), (38.0, 25.0)]],
        DynamicRVGSettings(resolution=36, robot_geometry_scale=1.2),
    )

    session.initialize((15.0, 20.0, 0.0), (65.0, 20.0, 0.0))
    plan = session.step()

    assert plan.status.name == "TemporaryGoalPathAvailable"
    assert not plan.final_segment
    assert plan.temporary_goal is not None


def test_rvg_path_does_not_take_full_turn_across_angle_seam() -> None:
    workspace = WorkspaceConfig(
        width=79.0,
        height=43.3,
        car_width=4.2,
        car_height=4.8,
        offset_w=2.1,
        offset_h=4.5,
        wheel_base=4.2,
        max_wheel_speed=10.0,
        frequency=30.0,
        car_id_list=[3],
    )
    session = DynamicRVGSession(
        workspace,
        [[(38.0, 15.0), (42.0, 15.0), (42.0, 25.0), (38.0, 25.0)]],
        DynamicRVGSettings(resolution=36, robot_geometry_scale=1.2),
    )

    session.initialize((15.0, 20.0, 1.0), (65.0, 20.0, 359.0))
    plan = session.step()

    initial_rotation = [
        configuration
        for configuration in plan.configurations
        if configuration.getX() == pytest.approx(15.0)
        and configuration.getY() == pytest.approx(20.0)
    ]
    assert len(initial_rotation) == 2
    first_heading = math.degrees(initial_rotation[0].getTheta())
    last_heading = math.degrees(initial_rotation[-1].getTheta())
    wrapped_rotation = (last_heading - first_heading + 180.0) % 360.0 - 180.0
    assert first_heading == pytest.approx(1.0)
    assert abs(wrapped_rotation) < 30.0
    assert len(plan.controller_path) < 10
