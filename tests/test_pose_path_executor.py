"""Regressions for wrapped turns, swept clearance and pose-preserving execution."""
import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from micromvp.core.models import Action, RobotObservation, WorkspaceConfig
from micromvp.controller import NavigationController
from micromvp.controller.navigation_controller.pose_path_executor import (
    PosePathExecutor, differential_drive_motions, heading_delta,
)
from micromvp.planner.dynamic_rvg import DynamicRVGSession, DynamicRVGSettings


def vertex(x, y, theta):
    return SimpleNamespace(getX=lambda: x, getY=lambda: y, getTheta=lambda: math.radians(theta))


def collision_session(obstacles=()):
    session = DynamicRVGSession.__new__(DynamicRVGSession)
    session._workspace = WorkspaceConfig(100, 60, 4, 2, 1, 2, 4, 10, 30, [3])
    session._settings = DynamicRVGSettings(robot_geometry_scale=1)
    session._base_robot_geometry = [(-2, -1), (2, -1), (2, 1), (-2, 1)]
    session._obstacle_points = list(obstacles)
    return session


def test_pure_turn_across_seam_never_creates_translation():
    motions = differential_drive_motions([(20, 20, 359), (20, 20, 1)])
    assert len(motions) == 1 and motions[0].kind == "rotate"
    assert heading_delta(motions[0].start[2], motions[0].end[2]) == 2
    assert motions[0].start[:2] == motions[0].end[:2]


def test_swept_arc_detects_collision_between_clear_endpoint_poses():
    angle = math.radians(2.5)
    x = 20 + 2 * math.cos(angle) - math.sin(angle)
    y = 20 + 2 * math.sin(angle) + math.cos(angle)
    tiny = .0001
    obstacle = [(x-tiny,y-tiny),(x+tiny,y-tiny),(x+tiny,y+tiny),(x-tiny,y+tiny)]
    session = collision_session([obstacle])
    assert session.pose_is_valid((20, 20, 0))
    assert session.pose_is_valid((20, 20, 5))
    assert not session.motion_is_valid((20, 20, 0), (20, 20, 5))


def test_translation_sweep_detects_obstacle_between_endpoints():
    session = collision_session([[(29,19),(31,19),(31,21),(29,21)]])
    assert session.pose_is_valid((20,20,0))
    assert session.pose_is_valid((40,20,0))
    assert not session.motion_is_valid((20,20,0),(40,20,0))


def test_clear_short_rotation_collapses_but_blocked_shortcut_keeps_long_route():
    configs = [vertex(20,20,h) for h in [350,270,180,90,10]]
    session = collision_session()
    assert len(session._collapse_redundant_full_turns(configs)) == 2
    session._rotation_is_valid = lambda *args: False
    assert session._collapse_redundant_full_turns(configs) == configs


def test_executor_turns_then_sends_only_one_straight_edge_to_controller():
    controller = Mock()
    controller.car_state = SimpleNamespace(status_label="ROTATING")
    controller.step.return_value = Action(-.1,.1)
    validate = Mock(return_value=True)
    executor = PosePathExecutor(controller, validate)
    executor.load([vertex(20,20,90),vertex(40,20,0)])
    obs = RobotObservation(3,20,20,90)
    executor.step(obs)
    controller.rotate_to.assert_called_once_with(0)
    controller.set_path.assert_not_called()
    controller.car_state.status_label = "ROTATION_DONE"
    assert executor.step(RobotObservation(3,20,20,0)) == Action.stop()
    controller.car_state.status_label = "FOLLOWING"
    executor.step(RobotObservation(3,20,20,0))
    controller.set_path.assert_called_once_with([(20,20),(40,20)])
    controller.car_state.status_label = "FINISHED"
    executor.step(RobotObservation(3,39.9,20,0))
    assert executor.done


def test_measured_drift_is_rechecked_and_stops_execution():
    controller = Mock()
    controller.car_state = SimpleNamespace(status_label="ROTATING")
    validator = Mock(return_value=True)
    executor = PosePathExecutor(controller, validator)
    executor.load([vertex(20,20,0),vertex(20,20,90)])
    validator.return_value = False
    with pytest.raises(ValueError, match="Measured rotate"):
        executor.step(RobotObservation(3,20.6,20,5))
    assert validator.call_args.args == ((20.6,20,5),(20.6,20,90))
    controller.step.assert_not_called()
    controller.reset.assert_called_once()


def test_differential_drive_conversion_rejects_unavailable_turn_space():
    executor = PosePathExecutor(Mock(), lambda a,b: a[2] == b[2])
    with pytest.raises(ValueError, match="Unsafe rotate"):
        executor.load([vertex(20,20,0),vertex(20,40,0)])


def test_real_controller_takes_short_in_place_turn_in_simulation(monkeypatch):
    ws = WorkspaceConfig(100,60,4.2,4.8,2.1,4.5,4.2,10,30,[3])
    controller = NavigationController(3,ws)
    controller.ROTATION_TOLERANCE_DEG = 2
    controller.ROTATION_STABLE_TIME = .2
    clock = [100.]
    monkeypatch.setattr("micromvp.controller.navigation_controller.navigation_controller.time.time",
                        lambda: clock[0])
    executor = PosePathExecutor(controller, lambda a,b: True)
    executor.load([vertex(20,20,350),vertex(20,20,10)])
    theta = 350.
    travel = 0.
    for i in range(300):
        clock[0] += 1/30
        action = executor.step(RobotObservation(3,20,20,theta,timestamp=clock[0]))
        assert action.left_speed + action.right_speed == pytest.approx(0)
        delta = math.degrees((action.right_speed-action.left_speed)*10/4.2/30)
        travel += abs(delta)
        theta = (theta + delta) % 360
        if executor.done:
            break
    assert executor.done
    assert travel < 35
    assert abs(heading_delta(theta,10)) < 4


@pytest.mark.rvg
@pytest.mark.parametrize("mode", ["graph_merge","incremental","delta_graph_merge","buffered_delta_graph_merge"])
def test_optimal_binding_applies_before_graph_construction(mode):
    ws = WorkspaceConfig(100,60,4.2,4.8,2.1,4.5,4.2,10,30,[3])
    session = DynamicRVGSession(ws,[],DynamicRVGSettings(optimal=True,planner_mode=mode))
    assert session.planner.optimalRvgConstruction()
    session.initialize((20,20,359),(40,20,1))
    assert not session.planner.setOptimalRvgConstruction(False)
    plan = session.step()
    assert plan.final_segment
    assert plan.controller_path == [(20,20),(40,20)]


def test_disabled_execution_guard_does_not_reject_planned_motion():
    controller = Mock()
    controller.car_state = SimpleNamespace(status_label="ROTATING")
    controller.step.return_value = Action(-.1, .1)
    validator = Mock(return_value=False)
    executor = PosePathExecutor(controller, validator, reject_blocked_motion=False)
    executor.load([vertex(20,20,0),vertex(20,20,90)])
    validator.reset_mock()
    result = executor.step(RobotObservation(3,20.6,20,5))
    assert result == Action(-.1,.1)
    controller.rotate_to.assert_called_once_with(90)
    validator.assert_not_called()


def test_backward_drvg_hop_does_not_add_two_half_turns():
    motions = differential_drive_motions([
        (65.60, 30.45, 185.0), (66.68, 30.50, 185.0)
    ])
    drives = [m for m in motions if m.kind == "drive"]
    assert len(drives) == 1 and drives[0].reverse
    turning = sum(abs(heading_delta(m.start[2],m.end[2]))
                  for m in motions if m.kind == "rotate")
    assert turning < 10  # Previously roughly 350 degrees for a 1 cm hop.


def test_pre_turn_hands_off_at_tolerance_like_standard_navigation():
    controller = Mock()
    controller.car_state = SimpleNamespace(status_label="ROTATION_STABLE")
    controller.step.return_value = Action.stop()
    executor = PosePathExecutor(controller, lambda a,b: True)
    executor.load([vertex(20,20,90),vertex(40,20,0)])
    executor.step(RobotObservation(3,20,20,2))
    assert executor.motions[executor.index].kind == "drive"
    controller.car_state.status_label = "FOLLOWING"
    executor.step(RobotObservation(3,20,20,2))
    controller.set_path.assert_called_once_with([(20,20),(40,20)])


def test_final_rotation_still_waits_for_settled_completion():
    controller = Mock()
    controller.car_state = SimpleNamespace(status_label="ROTATION_STABLE")
    controller.step.return_value = Action.stop()
    executor = PosePathExecutor(controller,lambda a,b: True)
    executor.load([vertex(20,20,0),vertex(20,20,20)])
    executor.step(RobotObservation(3,20,20,19))
    assert not executor.done
    controller.car_state.status_label = "ROTATION_DONE"
    executor.step(RobotObservation(3,20,20,19))
    assert executor.done


def test_reverse_wheel_mapping_preserves_physical_heading_and_steering_sign():
    ws = WorkspaceConfig(100,60,4.2,4.8,2.1,4.5,4.2,10,30,[3])
    forward = NavigationController(3,ws)
    reverse = NavigationController(3,ws)
    path = [(20,20),(40,20)]
    forward.set_path(path)
    reverse.set_path(path,reverse=True)
    a = forward.step(RobotObservation(3,20,20.5,5,timestamp=100))
    b = reverse.step(RobotObservation(3,20,20.5,185,timestamp=100))
    assert b.left_speed == pytest.approx(-a.right_speed)
    assert b.right_speed == pytest.approx(-a.left_speed)
    assert reverse.car_state.theta == pytest.approx(185)
    assert b.right_speed-b.left_speed == pytest.approx(a.right_speed-a.left_speed)
    reverse.rotate_to(190)
    assert not reverse._reverse_path


def test_reverse_translation_reaches_goal_without_spinning(monkeypatch):
    ws = WorkspaceConfig(100,60,4.2,4.8,2.1,4.5,4.2,10,30,[3])
    controller = NavigationController(3,ws,max_speed=.2,goal_tolerance=.5)
    clock = [100.]
    monkeypatch.setattr("micromvp.controller.navigation_controller.navigation_controller.time.time",
                        lambda: clock[0])
    executor = PosePathExecutor(controller,lambda a,b: True)
    executor.load([vertex(20,20,180),vertex(30,20,180)])
    assert len(executor.motions) == 1
    assert executor.motions[0].reverse
    x,y,theta = 20.,20.,180.
    angular_travel = 0.
    for i in range(400):
        clock[0] += 1/30
        action = executor.step(RobotObservation(3,x,y,theta,timestamp=clock[0]))
        v = (action.left_speed+action.right_speed)*10/2
        w = (action.right_speed-action.left_speed)*10/4.2
        x += v*math.cos(math.radians(theta))/30
        y += v*math.sin(math.radians(theta))/30
        delta = math.degrees(w/30)
        angular_travel += abs(delta)
        theta = (theta+delta)%360
        if executor.done:
            break
    assert executor.done
    assert math.hypot(x-30,y-20) < .6
    assert angular_travel < 5


def test_same_signed_arc_merges_across_zero_without_changing_rotation_path():
    controller=Mock()
    validator=Mock(return_value=False)
    executor=PosePathExecutor(controller,validator,reject_blocked_motion=False)
    executor.load([vertex(54.05,27.99,h) for h in (85.2,355,345,294.8)])
    assert len(executor.motions)==1
    motion=executor.motions[0]
    assert heading_delta(motion.start[2],motion.end[2])==pytest.approx(-150.4)
    validator.assert_not_called()


def test_merging_does_not_reverse_a_positive_half_turn():
    controller=Mock()
    executor=PosePathExecutor(controller,lambda a,b:False,reject_blocked_motion=False)
    executor.load([vertex(20,20,h) for h in (0,90,180)])
    assert len(executor.motions)==2
    assert all(heading_delta(m.start[2],m.end[2])>0 for m in executor.motions)


def test_355_and_minus_5_are_identical_shortest_turn_targets():
    assert heading_delta(85.2,355)==pytest.approx(-90.2)
    assert heading_delta(85.2,-5)==pytest.approx(-90.2)
