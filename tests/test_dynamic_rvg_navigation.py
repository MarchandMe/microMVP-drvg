"""Tests for the real-hardware DynamicRVG navigation sequence."""

import importlib.util
import math
from pathlib import Path

import pytest

from micromvp.core.models import (
    Action,
    CarState,
    RobotObservation,
    WorkspaceConfig,
)
from micromvp.planner import (
    DynamicRVGPlan,
    DynamicRVGSettings,
    attach_obstacles_to_workspace_border,
    pad_obstacle_polygon,
)


_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "dynamic_rvg_navigation.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "dynamic_rvg_navigation_example",
    _EXAMPLE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
DynamicRVGRealNavigator = _MODULE.DynamicRVGRealNavigator
capture_fixed_obstacles = _MODULE.capture_fixed_obstacles
parse_args = _MODULE.parse_args


class _RecordingPlanner:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def initialize(self, start, goal) -> None:
        self.events.append(("initialize", (start, goal)))

    def update_pose(self, observation) -> None:
        self.events.append(("update_pose", observation.pose))

    def step(self) -> DynamicRVGPlan:
        self.events.append(("step", None))
        return DynamicRVGPlan(
            status="TemporaryGoalPathAvailable",
            configurations=[_Configuration(10, 10, 0), _Configuration(20, 15, 0)],
            controller_path=[(10.0, 10.0), (20.0, 15.0)],
            temporary_goal=(20.0, 15.0),
            final_segment=False,
        )

    def motion_is_valid(self, start, end) -> bool:
        return True

    def draw(self, _figure_path: Path) -> bool:
        return True

    def latest_visible_region(self):
        return (
            [(8.0, 8.0), (22.0, 8.0), (22.0, 18.0), (8.0, 18.0)],
            [[(12.0, 10.0), (14.0, 10.0), (14.0, 12.0), (12.0, 12.0)]],
        )


class _RecordingController:
    def __init__(self) -> None:
        self.car_state = CarState(car_id=3, status_label="IDLE")
        self.path = []

    def reset(self) -> None:
        self.car_state.status_label = "IDLE"

    def update(self, observation: RobotObservation) -> None:
        self.car_state.x = observation.x
        self.car_state.y = observation.y
        self.car_state.theta = observation.theta

    def set_path(self, path) -> None:
        self.path = list(path)
        self.car_state.status_label = "FOLLOWING"

    def step(self, _observation: RobotObservation) -> Action:
        return Action.stop()

    def rotate_to(self, heading) -> None:
        self.car_state.status_label = 'ROTATING'

    def set_speed(self, _speed: float) -> None:
        pass


class _Configuration:
    def __init__(self, x: float, y: float, theta_radians: float) -> None:
        self._x = x
        self._y = y
        self._theta = theta_radians

    def getX(self) -> float:
        return self._x

    def getY(self) -> float:
        return self._y

    def getTheta(self) -> float:
        return self._theta


@pytest.fixture(autouse=True)
def display_scan_without_native_binding(monkeypatch):
    class DisplayScan:
        def scan(self, pose):
            return _RecordingPlanner().latest_visible_region()
    monkeypatch.setattr(
        DynamicRVGRealNavigator, "_new_display_visibility", lambda self: DisplayScan()
    )


def test_obstacle_capture_default_is_one_second(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["dynamic_rvg_navigation.py"])

    args = parse_args()

    assert args.obstacle_capture_seconds == 1.0


def test_first_plan_uses_initialization_observation(monkeypatch, tmp_path) -> None:
    planner = _RecordingPlanner()
    monkeypatch.setattr(
        DynamicRVGRealNavigator,
        "_new_planner",
        lambda _self: planner,
    )
    workspace = WorkspaceConfig(
        width=50.0,
        height=30.0,
        car_width=4.2,
        car_height=4.8,
        offset_w=2.1,
        offset_h=4.5,
        wheel_base=4.2,
        max_wheel_speed=10.0,
        frequency=30.0,
        car_id_list=[3],
    )
    navigator = DynamicRVGRealNavigator(
        workspace,
        _RecordingController(),
        [],
        DynamicRVGSettings(),
        tmp_path,
        draw_planner_graphs=False,
        obstacle_padding_cm=0.5,
        obstacle_draw_height_cm=4.0,
    )
    observation = RobotObservation(
        robot_id=3,
        x=10.0,
        y=12.0,
        theta=25.0,
    )

    assert navigator.request_goal(40.0, 20.0)
    action = navigator.process_observation(observation)

    assert action == Action.stop()
    assert planner.events == []

    navigator.replace_obstacles(
        [[(5.0, 5.0), (7.0, 5.0), (7.0, 7.0), (5.0, 7.0)]]
    )
    assert navigator.fixed_obstacles[0] == pytest.approx(
        [(4.5, 4.5), (7.5, 4.5), (7.5, 7.5), (4.5, 7.5)]
    )
    action = navigator.process_observation(observation)

    assert action == Action.stop()
    assert planner.events == [
        ("initialize", (observation.pose, (40.0, 20.0, 0.0))),
        ("step", None),
    ]
    assert not navigator.executor.reject_blocked_motion
    assert navigator.planning_step == 1
    assert navigator.needs_plan is False
    scanned_area = next(
        drawing
        for drawing in navigator._gui_drawings()
        if drawing["uuid"] == "scanned_area"
    )
    assert scanned_area["type"] == "region"
    assert scanned_area["clip_to_workspace"] is True
    assert len(scanned_area["regions"]) == 1
    assert scanned_area["projection_height_cm"] == 4.0
    obstacle_drawing = next(
        drawing
        for drawing in navigator._gui_drawings()
        if drawing["uuid"] == "fixed_obstacle_0"
    )
    assert obstacle_drawing["projection_height_cm"] == 4.0
    assert obstacle_drawing["points"] == [
        (5.0, 5.0), (7.0, 5.0), (7.0, 7.0), (5.0, 7.0), (5.0, 5.0)
    ]

    navigator.set_overlay_projection_to_floor(True)
    floor_drawings = navigator._gui_drawings()
    floor_scan = next(
        drawing
        for drawing in floor_drawings
        if drawing["uuid"] == "scanned_area"
    )
    floor_obstacle = next(
        drawing
        for drawing in floor_drawings
        if drawing["uuid"] == "fixed_obstacle_0"
    )
    assert floor_scan["projection_height_cm"] == 0.0
    assert floor_obstacle["projection_height_cm"] == 0.0

    navigator.set_overlay_projection_to_floor(False)
    raised_drawings = navigator._gui_drawings()
    raised_scan = next(
        drawing
        for drawing in raised_drawings
        if drawing["uuid"] == "scanned_area"
    )
    raised_obstacle = next(
        drawing
        for drawing in raised_drawings
        if drawing["uuid"] == "fixed_obstacle_0"
    )
    assert raised_scan["projection_height_cm"] == 4.0
    assert raised_obstacle["projection_height_cm"] == 4.0

    next_observation = RobotObservation(
        robot_id=3,
        x=10.01,
        y=12.01,
        theta=25.1,
    )
    navigator.process_observation(next_observation)

    assert planner.events[-1] == ("update_pose", next_observation.pose)

    assert navigator.request_goal(30.0, 18.0)
    assert all(
        drawing["uuid"] != "scanned_area"
        for drawing in navigator._gui_drawings()
    )


def test_obstacle_padding_offsets_convex_and_concave_edges() -> None:
    rectangle = pad_obstacle_polygon(
        [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)],
        0.5,
    )
    assert rectangle == pytest.approx(
        [(-0.5, -0.5), (4.5, -0.5), (4.5, 2.5), (-0.5, 2.5)]
    )

    concave = pad_obstacle_polygon(
        [
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 2.0),
            (2.0, 2.0),
            (2.0, 4.0),
            (0.0, 4.0),
        ],
        0.5,
    )
    assert concave == pytest.approx(
        [
            (-0.5, -0.5),
            (4.5, -0.5),
            (4.5, 2.5),
            (2.5, 2.5),
            (2.5, 4.5),
            (-0.5, 4.5),
        ]
    )


def test_border_obstacles_are_attached_without_changing_interior_obstacles() -> None:
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
    border_obstacle = [
        (-0.5, 5.0),
        (2.0, 5.0),
        (2.0, 8.0),
        (-0.5, 8.0),
    ]
    interior_obstacle = [
        (10.0, 10.0),
        (12.0, 10.0),
        (12.0, 12.0),
        (10.0, 12.0),
    ]

    prepared, attached_count = attach_obstacles_to_workspace_border(
        workspace,
        [border_obstacle, interior_obstacle],
    )

    assert attached_count == 1
    assert [point[0] for point in prepared[0]] == pytest.approx(
        [0.0001, 2.0, 2.0, 0.0001]
    )
    assert [point[1] for point in prepared[0]] == pytest.approx(
        [5.0, 5.0, 8.0, 8.0]
    )
    assert prepared[1] == interior_obstacle


def test_obstacle_capture_clears_the_previous_snapshot() -> None:
    class _Environment:
        def __init__(self) -> None:
            self.obstacles = [
                [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            ]
            self.reset_called = False

        def stop_all(self) -> None:
            pass

        def reset_obstacles(self) -> None:
            self.reset_called = True
            self.obstacles = []

        def get_obstacles(self):
            return self.obstacles

    environment = _Environment()

    captured = capture_fixed_obstacles(environment, duration=0.0, frequency=30.0)

    assert environment.reset_called
    assert captured == []


def test_rejected_obstacle_replacement_keeps_previous_world(
    monkeypatch,
    tmp_path,
) -> None:
    initial_planner = _RecordingPlanner()
    planner_results = iter([initial_planner, RuntimeError("invalid geometry")])

    def make_planner(_self):
        result = next(planner_results)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(DynamicRVGRealNavigator, "_new_planner", make_planner)
    workspace = WorkspaceConfig(
        width=50.0,
        height=30.0,
        car_width=4.2,
        car_height=4.8,
        offset_w=2.1,
        offset_h=4.5,
        wheel_base=4.2,
        max_wheel_speed=10.0,
        frequency=30.0,
        car_id_list=[3],
    )
    navigator = DynamicRVGRealNavigator(
        workspace,
        _RecordingController(),
        [[(5.0, 5.0), (7.0, 5.0), (7.0, 7.0), (5.0, 7.0)]],
        DynamicRVGSettings(),
        tmp_path,
        draw_planner_graphs=False,
        obstacle_padding_cm=0.5,
    )
    previous_obstacles = navigator.fixed_obstacles
    previous_display = navigator._display_obstacles

    with pytest.raises(RuntimeError, match="invalid geometry"):
        navigator.replace_obstacles(
            [[(20.0, 20.0), (22.0, 20.0), (22.0, 22.0), (20.0, 22.0)]]
        )

    assert navigator.planner is initial_planner
    assert navigator.fixed_obstacles == previous_obstacles
    assert navigator._display_obstacles == previous_display


def test_replanning_uses_measured_pose_without_discarding_padding(
    monkeypatch,
    tmp_path,
) -> None:
    terminal = _Configuration(20.0, 15.0, math.radians(90.0))

    class _Planner(_RecordingPlanner):
        def update_pose(self, pose) -> None:
            measured = pose.pose if isinstance(pose, RobotObservation) else pose
            self.events.append(("update_pose", measured))

        def pose_is_valid(self, pose, robot_geometry_scale=1.0) -> bool:
            self.events.append(
                ("pose_is_valid", (pose, robot_geometry_scale))
            )
            return robot_geometry_scale == 1.0

        def complete_current_segment(self) -> bool:
            self.events.append(("complete_current_segment", None))
            return True

        def step(self) -> DynamicRVGPlan:
            self.events.append(("step", None))
            return DynamicRVGPlan(
                status="TemporaryGoalPathAvailable",
                configurations=[
                    _Configuration(10.0, 12.0, math.radians(25.0)),
                    terminal,
                ],
                controller_path=[(10.0, 12.0), (20.0, 15.0)],
                temporary_goal=(20.0, 15.0),
                final_segment=False,
            )

    planner = _Planner()
    monkeypatch.setattr(
        DynamicRVGRealNavigator,
        "_new_planner",
        lambda _self: planner,
    )
    workspace = WorkspaceConfig(
        width=50.0,
        height=30.0,
        car_width=4.2,
        car_height=4.8,
        offset_w=2.1,
        offset_h=4.5,
        wheel_base=4.2,
        max_wheel_speed=10.0,
        frequency=30.0,
        car_id_list=[3],
    )
    navigator = DynamicRVGRealNavigator(
        workspace,
        _RecordingController(),
        [],
        DynamicRVGSettings(),
        tmp_path,
        draw_planner_graphs=False,
    )
    observation = RobotObservation(
        robot_id=3,
        x=19.9,
        y=15.1,
        theta=87.0,
    )
    navigator.active = True
    navigator.current_plan = planner.step()

    navigator._finish_current_segment(observation.pose)

    assert navigator.needs_plan

    planner.events.clear()
    navigator.process_observation(observation)

    assert planner.events[0] == (
        "update_pose",
        pytest.approx(observation.pose),
    )
    assert planner.events[1] == ("step", None)


def test_recording_state_includes_pending_goal_and_stops_on_completion_or_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(DynamicRVGRealNavigator, "_new_planner", lambda self: _RecordingPlanner())
    navigator = DynamicRVGRealNavigator(
        WorkspaceConfig(50, 30, 4.2, 4.8, 2.1, 4.5, 4.2, 10, 30, [3]),
        _RecordingController(), [], DynamicRVGSettings(), tmp_path,
    )
    assert not navigator.recording_run_active()
    assert not navigator.request_goal(-1, 20)
    assert not navigator.recording_run_active()
    assert navigator.request_goal(40, 20)
    assert navigator.recording_run_active()  # Includes the obstacle rescan.
    navigator.report_tracking_lost()
    assert navigator.recording_run_active()  # A temporary pause does not end the run.
    navigator.report_obstacle_capture_failed(RuntimeError("bad scan"))
    assert not navigator.recording_run_active()
    navigator.request_goal(40, 20)
    navigator.cancel()
    assert not navigator.recording_run_active()
    navigator.request_goal(40, 20)
    navigator._finish_goal((40, 20, 0))
    assert not navigator.recording_run_active()


def test_preplanned_playback_never_replans_from_measured_drift(monkeypatch,tmp_path):
    from micromvp.planner.preplanned import FrozenConfiguration as V
    plans=[
        DynamicRVGPlan("TemporaryGoalPathAvailable",[V(10,10,0),V(20,10,0)],[(10,10),(20,10)],(20,10),False),
        DynamicRVGPlan("GoalPathAvailable",[V(20,10,0),V(30,10,0)],[(20,10),(30,10)],None,True),
    ]
    class Task:
        done=False
        cancelled=False
        def start(self):pass
        def poll(self):return self.done,plans if self.done else None,None,"Preparing"
        def cancel(self):self.cancelled=True
        def join(self):pass
    task=Task()
    planner=_RecordingPlanner()
    scans=[]
    class Display:
        def scan(self,pose):
            scans.append(pose)
            return planner.latest_visible_region()
    monkeypatch.setattr(DynamicRVGRealNavigator,"_new_planner",lambda self:planner)
    monkeypatch.setattr(DynamicRVGRealNavigator,"_make_preplan_task",lambda self,a,b:task)
    monkeypatch.setattr(DynamicRVGRealNavigator,"_new_display_visibility",lambda self:Display())
    controller=_RecordingController()
    nav=DynamicRVGRealNavigator(
        WorkspaceConfig(50,30,4.2,4.8,2.1,4.5,4.2,10,30,[3]),controller,[],
        DynamicRVGSettings(),tmp_path,execution_mode="preplanned",scan_pause=.5)
    nav.request_goal(30,10);nav.replace_obstacles([])
    obs=RobotObservation(3,10,10,0)
    assert nav.process_observation(obs)==Action.stop()
    assert nav.process_observation(obs)==Action.stop()
    assert controller.path==[]
    task.done=True
    assert nav.process_observation(obs)==Action.stop()
    assert nav.planning_step==1 and len(scans)==1
    assert nav.process_observation(obs)==Action.stop()
    assert controller.path==[]
    nav._scan_pause_until=0
    nav.process_observation(obs)
    controller.car_state.status_label="FINISHED"
    drifted=RobotObservation(3,19.7,10.1,1)
    assert nav.process_observation(drifted)==Action.stop()
    assert nav.needs_plan
    assert nav.process_observation(drifted)==Action.stop()
    assert nav.planning_step==2 and scans[-1]==drifted.pose
    assert nav.current_plan.configurations[0].getX()==20
    assert planner.events==[]
    assert list(tmp_path.glob("preplanned_run_*.json"))
    nav.cancel()
    assert not nav.recording_run_active()
    nav.close()

def test_new_goal_discards_cancelled_preplan(monkeypatch,tmp_path):
    class Task:
        cancelled=False
        def start(self):pass
        def poll(self):return False,None,None,"Preparing"
        def cancel(self):self.cancelled=True
        def join(self):pass
    tasks=[Task(),Task()]
    sequence=iter(tasks)
    monkeypatch.setattr(DynamicRVGRealNavigator,"_new_planner",lambda self:_RecordingPlanner())
    monkeypatch.setattr(DynamicRVGRealNavigator,"_make_preplan_task",lambda self,a,b:next(sequence))
    nav=DynamicRVGRealNavigator(
        WorkspaceConfig(50,30,4.2,4.8,2.1,4.5,4.2,10,30,[3]),_RecordingController(),[],
        DynamicRVGSettings(),tmp_path,execution_mode="preplanned")
    nav.request_goal(30,10);nav.replace_obstacles([])
    nav.process_observation(RobotObservation(3,10,10,0))
    nav.request_goal(35,10)
    assert tasks[0].cancelled
    nav.replace_obstacles([])
    nav.process_observation(RobotObservation(3,10,10,0))
    assert nav._preplan_task is tasks[1]
    nav.close()
    assert tasks[1].cancelled



def test_discovery_presentation_arguments(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "demo", "--discovery-view", "--setup-seconds", "0",
        "--blackout-seconds", "0.75", "--unknown-opacity", "0.8",
    ])
    args = parse_args(camera_recording=True)
    assert args.setup_seconds == 0
    assert args.blackout_seconds == .75
    assert args.unknown_opacity == .8


@pytest.mark.parametrize("flag,value", [
    ("--unknown-opacity", "1.1"), ("--unknown-opacity", "nan"),
    ("--blackout-seconds", "-1"), ("--blackout-seconds", "inf"),
])
def test_invalid_discovery_presentation_arguments(monkeypatch, flag, value):
    monkeypatch.setattr("sys.argv", ["demo", flag, value])
    with pytest.raises(SystemExit):
        parse_args(camera_recording=True)
