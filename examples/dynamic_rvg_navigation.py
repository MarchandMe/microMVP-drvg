"""Run the externally stepped DynamicRVG planner on real MicroMVP hardware.

The obstacle geometry is captured once and remains fixed for a planner
session.  Camera observations continue to provide the authoritative robot
pose on every control cycle.  Click the MicroMVP canvas to start a run.

Safety behaviour:
- the robot is stopped while planning or replacing the obstacle snapshot;
- loss of the active robot's camera observation produces a stop command;
- planner failures stop instead of falling back to a straight-line path;
- Space and C cancel the current run and stop the hardware immediately.
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, replace
import math
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from micromvp.controller import NavigationController
from micromvp.controller.navigation_controller.pose_path_executor import PosePathExecutor
from micromvp.planner.dynamic_rvg import PhysicalVisibility
from micromvp.planner.preplanned import PreplanTask
from micromvp.utils.motion_log import MotionLog
from micromvp.core.models import (
    Action,
    CarState,
    Point,
    Pose,
    RobotObservation,
    WorkspaceConfig,
)
from micromvp.planner import (
    PLANNER_MODES,
    SCAN_MODES,
    TEMPORARY_GOAL_STRATEGIES,
    DynamicRVGPlan,
    DynamicRVGSession,
    DynamicRVGSettings,
    pad_obstacles,
)


_NUMBER_PATTERN = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")


def parse_args(*, camera_recording: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DynamicRVG + MicroMVP real-hardware navigation"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/car_v4.yaml"),
        help="MicroMVP deployment config (default: config/car_v4.yaml)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for workspace and robot discovery (default: 10)",
    )
    parser.add_argument(
        "--execution-mode", choices=("live", "preplanned"), default="live",
        help="Preplanned computes the complete DRVG route before moving",
    )
    parser.add_argument(
        "--preplan-timeout", type=float, default=30.0,
        help="Maximum preplanning time in seconds (default: 30)",
    )
    parser.add_argument(
        "--scan-pause", type=float, default=0.5,
        help="Stopped display-update time between preplanned segments (default: 0.5)",
    )
    parser.add_argument(
        "--obstacle-capture-seconds",
        type=float,
        default=1.0,
        help="Time used to collect the fixed obstacle snapshot (default: 1)",
    )
    parser.add_argument(
        "--goal-heading",
        type=float,
        default=0.0,
        help="Initial heading for canvas goals, in degrees (default: 0)",
    )
    parser.add_argument(
        "--strategy",
        choices=TEMPORARY_GOAL_STRATEGIES,
        default="quadtree",
        help="Temporary-goal strategy (default: quadtree)",
    )
    parser.add_argument(
        "--mode",
        choices=PLANNER_MODES,
        default="graph_merge",
        help="DynamicRVG graph update mode (default: graph_merge)",
    )
    parser.add_argument(
        "--scan-mode",
        choices=SCAN_MODES,
        default="center",
        help="Software scan origin mode (default: center)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Override planner.rvg.resolution from the config",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="DynamicRVG worker threads (default: 1)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10000,
        help="Maximum DynamicRVG planning steps (default: 10000)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override planner.rvg.euclidean_weight from the config",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Override planner.rvg.rotational_weight from the config",
    )
    parser.add_argument(
        "--leaf-scale",
        "--minimum-leaf-width-scale",
        dest="leaf_scale",
        type=float,
        default=1.0,
        help="DynamicRVG quadtree minimum leaf scale (default: 1)",
    )
    parser.add_argument(
        "--bucket-capacity",
        type=int,
        default=32,
        help="DynamicRVG quadtree bucket capacity (default: 32)",
    )
    parser.add_argument(
        "--information-weight",
        type=float,
        default=1.0,
        help="Information-gain score weight (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("navigation_output/dynamic_rvg"),
        help="Directory for per-step planner graph drawings",
    )
    graph_output = parser.add_mutually_exclusive_group()
    graph_output.add_argument(
        "--planner-drawings", dest="no_planner_drawings", action="store_false",
        help="Save diagnostic graph PNGs; synchronous plotting pauses robot execution",
    )
    graph_output.add_argument(
        "--no-planner-drawings", dest="no_planner_drawings", action="store_true",
        help="Skip diagnostic graph PNGs (default); live overlays remain enabled",
    )
    parser.set_defaults(no_planner_drawings=True)
    if camera_recording:
        parser.add_argument(
            "--discovery-view", action="store_true",
            help="Darken unseen space and reveal observed obstacle edges",
        )
        parser.add_argument(
            "--setup-seconds", type=float, default=2.0,
            help="Full-setup intro before blackout; 0 starts black (default: 2)",
        )
        parser.add_argument(
            "--blackout-seconds", type=float, default=0.5,
            help="Fully black transition before revealing the first scan (default: 0.5)",
        )
        parser.add_argument(
            "--unknown-opacity", type=float, default=0.85,
            help="Black opacity of unknown areas: 0 transparent, 1 opaque (default: 0.85)",
        )
        parser.add_argument(
            "--view", choices=("perspective", "birdseye"), default="perspective",
            help="Camera view: original perspective or rectangular bird's-eye view",
        )
        parser.add_argument(
            "--record", type=Path, metavar="FILE.mp4",
            help="Record each goal run as a silent 30 fps MP4; idle time is excluded",
        )
    args = parser.parse_args()
    if camera_recording and (not math.isfinite(args.setup_seconds) or args.setup_seconds < 0):
        parser.error("--setup-seconds must be non-negative")

    if camera_recording:
        if not math.isfinite(args.blackout_seconds) or args.blackout_seconds < 0:
            parser.error("--blackout-seconds must be non-negative")
        if not 0.0 <= args.unknown_opacity <= 1.0:
            parser.error("--unknown-opacity must be between 0 and 1")

    if camera_recording and args.record is not None:
        if args.record.suffix.lower() != ".mp4":
            parser.error("--record requires an .mp4 file")
        if args.record.exists():
            parser.error(f"recording already exists: {args.record}")
    return args


def copy_obstacles(
    obstacles: Sequence[Sequence[Point]],
) -> list[list[Point]]:
    """Copy valid obstacle polygons out of the observer's mutable storage."""
    return [
        [(float(x), float(y)) for x, y in polygon]
        for polygon in obstacles
        if len(polygon) >= 3
    ]


def capture_fixed_obstacles(
    environment: Any,
    duration: float,
    frequency: float,
) -> list[list[Point]]:
    """Clear cached geometry and capture a fresh obstacle observation."""
    environment.stop_all()
    reset_obstacles = getattr(environment, "reset_obstacles", None)
    if callable(reset_obstacles):
        reset_obstacles()
    deadline = time.monotonic() + max(0.0, duration)
    snapshot = copy_obstacles(environment.get_obstacles())
    sample_period = 1.0 / max(1.0, frequency)

    while time.monotonic() < deadline:
        environment.stop_all()
        environment.observe()
        candidate = copy_obstacles(environment.get_obstacles())
        snapshot = candidate
        time.sleep(sample_period)

    environment.stop_all()
    return snapshot


class DynamicRVGRealNavigator:
    """Connect real pose observations to one fixed-world DynamicRVG session."""

    def __init__(
        self,
        workspace: WorkspaceConfig,
        controller: NavigationController,
        obstacles: Sequence[Sequence[Point]],
        settings: DynamicRVGSettings,
        output_dir: Path,
        draw_planner_graphs: bool = False,
        goal_heading: float = 0.0,
        obstacle_padding_cm: float = 0.0,
        obstacle_draw_height_cm: float = 0.0,
        execution_collision_check: bool = False,
        execution_mode: str = "live",
        preplan_timeout: float = 30.0,
        scan_pause: float = 0.5,
        discovery_view: bool = False,
    ) -> None:
        if execution_mode not in {"live", "preplanned"}:
            raise ValueError("execution_mode must be live or preplanned")
        if not math.isfinite(preplan_timeout) or preplan_timeout <= 0:
            raise ValueError("preplan_timeout must be positive")
        if not math.isfinite(scan_pause) or scan_pause < 0:
            raise ValueError("scan_pause must be non-negative")
        self.discovery_view = discovery_view
        self._discovered_edges = []
        self._fully_discovered_obstacles = []
        self._motion_log = None
        self.execution_mode = execution_mode
        self.preplan_timeout = preplan_timeout
        self.scan_pause = scan_pause
        self._preplan_task = None
        self._preplan_jobs = []
        self._prepared_plans = []
        self._prepared_index = 0
        self._scan_pause_until = 0.0
        self.workspace = workspace
        self.controller = controller
        self.settings = settings
        self.output_dir = output_dir
        self.draw_planner_graphs = draw_planner_graphs
        self.obstacle_padding_cm = max(0.0, float(obstacle_padding_cm))
        self.obstacle_draw_height_cm = max(
            0.0, float(obstacle_draw_height_cm)
        )
        self._draw_overlays_on_floor = False
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._display_obstacles = copy_obstacles(obstacles)
        self._fixed_obstacles = pad_obstacles(
            self._display_obstacles,
            self.obstacle_padding_cm,
        )
        self.planner = self._new_planner()
        self.executor = PosePathExecutor(
            controller, lambda a, b: self.planner.motion_is_valid(a, b),
            reject_blocked_motion=execution_collision_check,
            orientation_invariant=settings.robot_footprint == "circle",
        )
        self._display_visibility = None
        self._pending_goal: Pose | None = None
        self._goal_waiting_for_obstacle_scan = False
        self._goal_heading = goal_heading % 360.0
        self._last_observation: RobotObservation | None = None
        self._status_before_tracking_loss = "Idle; click the canvas to set a goal"

        self.goal: Pose | None = None
        self.start_pose: Pose | None = None
        self.trajectory: list[Point] = []
        self.planned_segments: list[list[Point]] = []
        self._scanned_regions: list[
            tuple[list[Point], list[list[Point]]]
        ] = []
        self.current_plan: DynamicRVGPlan | None = None
        self.run_number = 0
        self.planning_step = 0
        self.needs_plan = False
        self.segment_rotation_active = False
        self.final_segment_active = False
        self.active = False
        self.finished = False
        self.failed = False
        self.tracking_lost = True
        self.status_message = "Idle; waiting for the active robot"

    @property
    def fixed_obstacles(self) -> list[list[Point]]:
        with self._lock:
            return copy_obstacles(self._fixed_obstacles)

    def _new_planner(self) -> DynamicRVGSession:
        return DynamicRVGSession(
            self.workspace,
            self._fixed_obstacles,
            self.settings,
        )

    def _record_motion(self, observation, action, event="sample"):
        if self._motion_log is not None:
            self._motion_log.record(
                observation, action, self.planning_step,
                self.executor, self.controller.car_state, event,
            )

    def _close_motion_log(self, reason):
        if self._motion_log is not None:
            self._record_motion(self._last_observation, Action.stop(), reason)
            self._motion_log.close(reason)
            self._motion_log = None

    def _cancel_preplanning(self):
        if self._preplan_task is not None:
            self._preplan_task.cancel()
            self._preplan_task = None
        self._prepared_plans = []
        self._prepared_index = 0
        self._scan_pause_until = 0.0

    def _make_preplan_task(self, start, goal):
        workspace = replace(self.workspace, car_id_list=list(self.workspace.car_id_list))
        obstacles = copy_obstacles(self._fixed_obstacles)
        settings = self.settings
        return PreplanTask(
            lambda: DynamicRVGSession(workspace, obstacles, settings),
            start, goal, settings.max_iterations, self.preplan_timeout,
        )

    def close(self):
        self.cancel("Window closed")
        for task in self._preplan_jobs:
            task.cancel()
            task.join()

    def _new_display_visibility(self):
        return PhysicalVisibility(
            self.workspace, self._display_obstacles, self.settings.scan_mode
        )

    def request_goal(self, x: float, y: float) -> bool:
        """Queue a goal that remains stopped until a fresh obstacle scan."""
        with self._lock:
            if not (0.0 <= x <= self.workspace.width):
                self.status_message = f"Rejected goal: x={x:.2f} is outside workspace"
                return False
            if not (0.0 <= y <= self.workspace.height):
                self.status_message = f"Rejected goal: y={y:.2f} is outside workspace"
                return False
            self._close_motion_log("superseded")
            self._cancel_preplanning()
            self._pending_goal = (float(x), float(y), self._goal_heading)
            self._goal_waiting_for_obstacle_scan = True
            self.controller.reset()
            self.executor.reset()
            self.needs_plan = False
            self.segment_rotation_active = False
            self.final_segment_active = False
            self.active = False
            self.finished = False
            self.failed = False
            self._scanned_regions = []
            self.status_message = (
                f"Goal queued: ({x:.2f}, {y:.2f}, "
                f"{self._goal_heading:.1f} deg); rescanning obstacles"
            )
            return True

    def set_goal_heading_text(self, text: str) -> bool:
        """Set the heading used by future canvas clicks without exceptions."""
        value = text.strip()
        with self._lock:
            if _NUMBER_PATTERN.fullmatch(value) is None:
                self.status_message = f"Invalid goal heading: {text!r}"
                return False
            self._goal_heading = float(value) % 360.0
            self.status_message = (
                f"Future canvas goals will use {self._goal_heading:.1f} deg"
            )
            return True

    def set_controller_speed(self, speed: float) -> None:
        with self._lock:
            self.controller.set_speed(speed)

    def set_overlay_projection_to_floor(self, enabled: bool) -> None:
        """Switch scan and obstacle drawings between object and floor planes."""
        with self._lock:
            self._draw_overlays_on_floor = bool(enabled)

    def cancel(self, reason: str = "Run cancelled") -> None:
        with self._lock:
            self._close_motion_log(reason)
            self._cancel_preplanning()
            self.controller.reset()
            self.executor.reset()
            self._pending_goal = None
            self._goal_waiting_for_obstacle_scan = False
            self.needs_plan = False
            self.segment_rotation_active = False
            self.final_segment_active = False
            self.active = False
            self.finished = False
            self.failed = False
            self.status_message = reason

    def begin_obstacle_capture(self) -> None:
        self.cancel("Stopped; capturing a new fixed obstacle snapshot")

    def replace_obstacles(self, obstacles: Sequence[Sequence[Point]]) -> None:
        """Replace the fixed world while stopped, discarding planner history."""
        with self._lock:
            self.controller.reset()
            self.executor.reset()
            previous_obstacles = self._fixed_obstacles
            next_display_obstacles = copy_obstacles(obstacles)
            next_obstacles = pad_obstacles(
                next_display_obstacles,
                self.obstacle_padding_cm,
            )
            self._fixed_obstacles = next_obstacles
            try:
                next_planner = self._new_planner()
            except Exception:
                self._fixed_obstacles = previous_obstacles
                raise
            self._cancel_preplanning()
            self.planner = next_planner
            self._display_obstacles = next_display_obstacles
            self._display_visibility = None
            self.goal = None
            self.start_pose = None
            self.trajectory = []
            self.planned_segments = []
            self._scanned_regions = []
            self.current_plan = None
            self._goal_waiting_for_obstacle_scan = False
            self.planning_step = 0
            self.needs_plan = False
            self.segment_rotation_active = False
            self.final_segment_active = False
            self.active = False
            self.finished = False
            self.failed = False
            if self._pending_goal is None:
                self.status_message = (
                    f"Captured {len(self._fixed_obstacles)} fixed obstacle(s); "
                    "click the canvas to set a goal"
                )
            else:
                self.status_message = (
                    f"Captured {len(self._fixed_obstacles)} fixed obstacle(s); "
                    "initializing queued goal"
                )

    def report_obstacle_capture_failed(self, error: Exception) -> None:
        """Keep the previous safe world after a rejected rescan."""
        with self._lock:
            self.controller.reset()
            self.executor.reset()
            self.needs_plan = False
            self.active = False
            self.failed = True
            self.status_message = f"OBSTACLE SCAN REJECTED - robot stopped: {error}"

    def report_tracking_lost(self) -> None:
        with self._lock:
            if not self.tracking_lost:
                self._status_before_tracking_loss = self.status_message
            self.tracking_lost = True
            self.status_message = "TRACKING LOST - robot stopped"
            self._record_motion(None, Action.stop(), "tracking_lost")

    def process_observation(self, observation: RobotObservation) -> Action:
        with self._lock:
            action = self._process_observation(observation)
            self._record_motion(observation, action)
            if self.finished or self.failed:
                self._close_motion_log("goal_reached" if self.finished else "failed")
            return action

    def _process_observation(self, observation: RobotObservation) -> Action:
        """Consume one real pose and return the next normalized wheel command."""
        with self._lock:
            self._last_observation = observation
            if self.tracking_lost:
                self.tracking_lost = False
                self.status_message = self._status_before_tracking_loss

            if (
                self._pending_goal is not None
                and self._goal_waiting_for_obstacle_scan
            ):
                self.controller.update(observation)
                return Action.stop()

            if self._pending_goal is not None:
                self._initialize_run(observation, self._pending_goal)
                self._pending_goal = None
                # DRVG's first graph must be built from the exact pose used
                # for initialization.  A later camera frame differs even for
                # a stationary robot, which leaves the original start vertex
                # out of the first graph and is reported as CurrentPoseInvalid.
                if self.execution_mode == "live":
                    self._plan_next_segment(observation)
                return Action.stop()

            if not self.active:
                self.controller.update(observation)
                return Action.stop()

            if self.execution_mode == "preplanned":
                return self._process_preplanned(observation)

            self.planner.update_pose(observation)

            if self.needs_plan:
                self._plan_next_segment(observation)
                if self.failed or self.finished:
                    self.controller.update(observation)
                    return Action.stop()

            return self._execute_current_segment(observation)

    def _execute_current_segment(self, observation):
        try:
            action = self.executor.step(observation)
        except ValueError as error:
            self._reject_motion(error)
            return Action.stop()
        self.trajectory.append(observation.position)
        self.segment_rotation_active = (
            not self.executor.done
            and self.executor.motions[self.executor.index].kind == "rotate"
        )
        self.status_message = f"Segment {self.planning_step}: {self.executor.status}"
        if self.executor.done:
            self._finish_current_segment(observation.pose)
            return Action.stop()
        return action

    def _save_preplanned_route(self):
        path = self.output_dir / f"preplanned_run_{self.run_number:03d}_{time.time_ns()}.json"
        data = {
            "execution_mode": "preplanned",
            "start": self.start_pose, "goal": self.goal,
            "workspace": asdict(self.workspace), "settings": asdict(self.settings),
            "obstacles": self._display_obstacles,
            "obstacle_padding_cm": self.obstacle_padding_cm,
            "segments": [
                {
                    "status": plan.status, "final_segment": plan.final_segment,
                    "configurations": [
                        [v.getX(),v.getY(),math.degrees(v.getTheta())%360]
                        for v in plan.configurations
                    ],
                    "controller_path": plan.controller_path,
                }
                for plan in self._prepared_plans
            ],
        }
        path.write_text(json.dumps(data,indent=2))
        if self._motion_log is not None:
            self._motion_log.set_route_path(path)
        print(f"[Preplan] Complete route saved: {path}",flush=True)

    def _process_preplanned(self, observation):
        if self._preplan_task is not None:
            done,plans,error,progress = self._preplan_task.poll()
            self.status_message = progress + "; robot stopped"
            self.controller.update(observation)
            if not done:
                return Action.stop()
            self._preplan_task = None
            if error is not None:
                self._reject_motion(RuntimeError(error))
                return Action.stop()
            self._prepared_plans = plans
            self._prepared_index = 0
            self._save_preplanned_route()
            if not plans:
                self._finish_goal(observation.pose)
                return Action.stop()
            self.needs_plan = True
        if self.needs_plan:
            plan = self._prepared_plans[self._prepared_index]
            self.current_plan = plan
            self.planning_step = self._prepared_index + 1
            self.final_segment_active = plan.final_segment
            self.planned_segments.append(list(plan.controller_path))
            self._record_latest_scan_region(observation.pose)
            try:
                self.executor.load(plan.configurations)
            except ValueError as error:
                self._reject_motion(error)
                return Action.stop()
            self.needs_plan = False
            self._scan_pause_until = time.monotonic() + self.scan_pause
            self.status_message = (
                f"Preplanned segment {self.planning_step}/{len(self._prepared_plans)}: "
                "stopped; visible area updated"
            )
            print(f"[Replay] {self.status_message}; measured pose={observation.pose}",flush=True)
            self.controller.update(observation)
            return Action.stop()
        if time.monotonic() < self._scan_pause_until:
            self.controller.update(observation)
            return Action.stop()
        return self._execute_current_segment(observation)

    def report_execution_failed(self, error: Exception) -> None:
        with self._lock:
            self._reject_motion(error)
            self._close_motion_log("execution_error")

    def _reject_motion(self, error: Exception) -> None:
        self._cancel_preplanning()
        self._pending_goal = None
        self._goal_waiting_for_obstacle_scan = False
        self.controller.reset()
        self.executor.reset()
        self.active = False
        self.failed = True
        self.needs_plan = False
        self.status_message = f"MOTION BLOCKED - robot stopped: {error}"
        print(f"[DynamicRVG] {self.status_message}", flush=True)

    def _initialize_run(
        self,
        observation: RobotObservation,
        goal: Pose,
    ) -> None:
        self.controller.reset()
        self.executor.reset()
        self.controller.update(observation)
        if self.execution_mode == "live":
            self.planner.initialize(observation.pose, goal)
        self._goal_waiting_for_obstacle_scan = False
        self.run_number += 1
        self._close_motion_log("reinitialized")
        self._motion_log = MotionLog(self.output_dir,self.run_number,{
            "goal": list(goal), "start": list(observation.pose),
            "workspace": asdict(self.workspace), "settings": asdict(self.settings),
            "obstacles": self._display_obstacles,
            "obstacle_padding_cm": self.obstacle_padding_cm,
            "execution_mode": self.execution_mode,
            "heading_note": "Unwrapped heading restarts after tracking loss or an observation gap over 0.5 seconds.",
            "command_note": "Commands are generated wheel thrust, not measured motor feedback.",
        })
        self.planning_step = 0
        self.start_pose = observation.pose
        self.goal = goal
        self.trajectory = [observation.position]
        self.planned_segments = []
        self._scanned_regions = []
        self.current_plan = None
        self.needs_plan = True
        self.segment_rotation_active = False
        self.final_segment_active = False
        self.active = True
        self.finished = False
        self.failed = False
        self.status_message = "Goal initialized; robot stopped for planning"
        if self.execution_mode == "preplanned":
            self.status_message = "Preparing complete DRVG route; robot stopped"
            self._preplan_task = self._make_preplan_task(observation.pose, goal)
            self._preplan_jobs.append(self._preplan_task)
            self._preplan_task.start()


    def _save_invalid_pose_snapshot(self, observation, previous_plan) -> None:
        """Retain the exact failed scan and planner inputs without changing control."""
        native = self.planner.planner.latestObservation()
        terminal = None
        if previous_plan is not None and previous_plan.configurations:
            v = previous_plan.configurations[-1]
            terminal = [v.getX(),v.getY(),math.degrees(v.getTheta()) % 360]
        data = {
            "captured_at": time.time(),
            "workspace": asdict(self.workspace),
            "pose": list(observation.pose),
            "goal": self.goal,
            "previous_planned_terminal": terminal,
            "obstacles": self._display_obstacles,
            "obstacle_padding_cm": self.obstacle_padding_cm,
            "settings": asdict(self.settings),
            "native_observation": {
                "success": bool(native.success),
                "outer": self.planner._polygon_points(native.outerBoundary),
                "holes": [self.planner._polygon_points(hole) for hole in native.holes],
            },
        }
        directory = Path("diagnostics/drvg/failures")
        directory.mkdir(parents=True,exist_ok=True)
        path = directory / f"run_{self.run_number:03d}_step_{self.planning_step:03d}_{time.time_ns()}.json"
        path.write_text(json.dumps(data,indent=2))
        print(
            f"[DynamicRVG] Invalid pose={observation.pose}; previous terminal={terminal}; "
            f"snapshot={path.resolve()}", flush=True,
        )

    def _plan_next_segment(self, observation: RobotObservation) -> None:
        previous_plan = self.current_plan
        planning_started = time.perf_counter()
        plan = self.planner.step()
        planning_ms = (time.perf_counter() - planning_started) * 1000.0
        self._record_latest_scan_region(observation.pose)
        self.current_plan = plan
        self.planning_step += 1
        status_name = self._status_name(plan.status)
        print(
            f"[DynamicRVG] run={self.run_number} step={self.planning_step} "
            f"status={status_name} configurations={len(plan.configurations)} "
            f"controller_points={len(plan.controller_path)} planning_ms={planning_ms:.1f}"
        )

        if self.draw_planner_graphs:
            drawing_started = time.perf_counter()
            drawn = self.planner.draw(
                self.output_dir
                / (
                    f"run_{self.run_number:03d}_"
                    f"step_{self.planning_step:03d}.png"
                )
            )
            print(
                f"[DynamicRVG] diagnostic_plot_ms="
                f"{(time.perf_counter() - drawing_started) * 1000.0:.1f} saved={drawn}"
            )

        if status_name == "CurrentPoseInvalid":
            try:
                self._save_invalid_pose_snapshot(observation, previous_plan)
            except Exception as error:
                print(f"[DynamicRVG] Could not save invalid-pose diagnostic: {error}",flush=True)

        if status_name == "GoalReached":
            self._finish_goal(observation.pose)
            return

        planner_has_path = (
            status_name in {
                "GoalPathAvailable",
                "TemporaryGoalPathAvailable",
            }
            and len(plan.configurations) > 1
        )
        if not planner_has_path:
            self.controller.reset()
            self.executor.reset()
            self.active = False
            self.failed = True
            self.needs_plan = False
            if (
                status_name == "CurrentPoseInvalid"
                and self.planner.pose_is_valid(
                    observation.pose,
                    robot_geometry_scale=1.0,
                )
            ):
                self.status_message = (
                    "PLANNER SAFETY MARGIN REJECTED CURRENT POSE - "
                    "actual-size footprint is valid; robot stopped"
                )
            else:
                self.status_message = (
                    f"PLANNER FAILED - robot stopped: {status_name}"
                )
            return

        self.planned_segments.append(list(plan.controller_path))
        self.final_segment_active = plan.final_segment
        self.needs_plan = False

        try:
            self.executor.load(plan.configurations)
        except ValueError as error:
            self._reject_motion(error)
            return
        self.status_message = f"{status_name}: {self.executor.status}"

    def _finish_current_segment(self, pose: Pose) -> None:
        self.segment_rotation_active = False
        if self.final_segment_active:
            self._finish_goal(pose)
            return

        if self.execution_mode == "preplanned":
            self._prepared_index += 1
            self.needs_plan = True
            self.status_message = "Preplanned segment complete; stopped for visibility update"
            return

        if self.planner.complete_current_segment():
            self.needs_plan = True
            self.status_message = (
                f"Temporary segment {self.planning_step} complete; "
                "robot stopped for replanning"
            )
            return

        self.controller.reset()
        self.active = False
        self.failed = True
        self.status_message = (
            "PLANNER FAILED - controller completed a segment but DynamicRVG "
            "had no pending temporary goal"
        )

    def _finish_goal(self, pose: Pose) -> None:
        distance, heading_error = self.goal_errors(pose)
        self.active = False
        self.finished = True
        self.failed = False
        self.needs_plan = False
        self.segment_rotation_active = False
        self.status_message = (
            f"Goal reached: position error {distance:.3f} cm, "
            f"heading error {heading_error:.3f} deg"
        )

    def _record_latest_scan_region(self, pose: Pose) -> None:
        if self._display_visibility is None:
            self._display_visibility = self._new_display_visibility()
        outer_boundary, holes = self._display_visibility.scan(pose)
        if not outer_boundary:
            return
        self._scanned_regions.append(
            (
                list(outer_boundary),
                [list(hole) for hole in holes],
            )
        )

        if self.discovery_view:
            from micromvp.gui.discovery import discovered_obstacle_edges
            self._discovered_edges, complete = discovered_obstacle_edges(
                self._display_obstacles, self._scanned_regions
            )
            self._fully_discovered_obstacles = [
                self._display_obstacles[index] for index in complete
            ]

    def goal_errors(self, pose: Pose) -> tuple[float, float]:
        if self.goal is None:
            return math.inf, math.inf
        distance = math.hypot(pose[0] - self.goal[0], pose[1] - self.goal[1])
        heading_error = abs((pose[2] - self.goal[2] + 180.0) % 360.0 - 180.0)
        return distance, heading_error

    def snapshot(self) -> tuple[dict[int, CarState], list[dict[str, Any]]]:
        with self._lock:
            state = self.controller.car_state
            state_copy = CarState(
                car_id=state.car_id,
                x=state.x,
                y=state.y,
                theta=state.theta,
                linear_velocity=state.linear_velocity,
                angular_velocity=state.angular_velocity,
                status_label=state.status_label,
                metadata=dict(state.metadata),
            )
            return {state.car_id: state_copy}, self._gui_drawings()

    def recording_run_active(self) -> bool:
        """A goal owns its video through scanning, planning and execution."""
        with self._lock:
            return not (self.finished or self.failed) and (
                self.active or self._pending_goal is not None
            )

    def summary(self) -> tuple[str, str, str, str, str]:
        with self._lock:
            pose_text = "unavailable"
            if self._last_observation is not None:
                pose = self._last_observation.pose
                pose_text = f"({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.1f} deg)"
            if self.tracking_lost:
                pose_text += " [LOST]"

            planner_text = (
                f"run {self.run_number}, step {self.planning_step}"
                if self.run_number
                else "not initialized"
            )
            if self.execution_mode == "preplanned":
                planner_text = (
                    f"PREPLANNED playback: {self.planning_step}/{len(self._prepared_plans)}"
                    if self._prepared_plans else
                    "PREPLANNING full route" if self._preplan_task is not None else
                    "PREPLANNED mode: waiting for goal"
                )
            goal_text = "none"
            if self.goal is not None:
                goal_text = (
                    f"({self.goal[0]:.2f}, {self.goal[1]:.2f}, "
                    f"{self.goal[2]:.1f} deg)"
                )
            obstacle_text = f"{len(self._fixed_obstacles)} fixed polygon(s)"
            tracking_text = "LOST - STOPPED" if self.tracking_lost else "visible"
            return (
                self.status_message,
                planner_text,
                pose_text,
                goal_text,
                f"{tracking_text}; {obstacle_text}",
            )

    def _gui_drawings(self) -> list[dict[str, Any]]:
        drawings: list[dict[str, Any]] = []
        overlay_height_cm = (
            0.0
            if self._draw_overlays_on_floor
            else self.obstacle_draw_height_cm
        )
        if self._scanned_regions:
            drawings.append(
                {
                    "uuid": "scanned_area",
                    "type": "region",
                    "clip_to_workspace": True,
                    "regions": [
                        {"outer": outer, "holes": holes}
                        for outer, holes in self._scanned_regions
                    ],
                    "color": "transparent" if self.discovery_view else "#D6A20B",
                    "fill": "transparent" if self.discovery_view else "#40D6A20B",
                    "width": 0 if self.discovery_view else 1,
                    "discovered_edges": self._discovered_edges,
                    "discovered_obstacles": self._fully_discovered_obstacles,
                    "z": -10,
                    "projection_height_cm": overlay_height_cm,
                }
            )

        for index, obstacle in enumerate(self._display_obstacles):
            drawing = self._path_drawing(
                f"fixed_obstacle_{index}",
                self._closed(obstacle),
                "#FF6600",
                3,
            )
            drawing["projection_height_cm"] = overlay_height_cm
            drawings.append(drawing)

        for index, segment in enumerate(self.planned_segments):
            if len(segment) < 2:
                continue
            last_segment = index == len(self.planned_segments) - 1
            drawings.append(
                self._path_drawing(
                    f"planned_segment_{index}",
                    segment,
                    "#00AA33" if last_segment else "#77AA77",
                    3 if last_segment else 1,
                )
            )

        if len(self.trajectory) >= 2:
            drawings.append(
                self._path_drawing(
                    "measured_trajectory", self.trajectory, "#0055CC", 2
                )
            )

        if self.start_pose is not None:
            drawings.append(
                {
                    "uuid": "run_start",
                    "type": "point",
                    "position": self.start_pose[:2],
                    "radius": 6,
                    "color": "#00AA00",
                    "fill": "#00AA00",
                }
            )

        # Keep the selected goal visible while its initial scan is pending.
        display_goal = self._pending_goal or self.goal
        if display_goal is not None:
            drawings.extend(
                [
                    {
                        "uuid": "run_goal",
                        "type": "point",
                        "position": display_goal[:2],
                        "radius": 7,
                        "color": "#DD0000",
                        "fill": "#DD0000",
                        # Above both discovery shade (2000) and blackout (4000).
                        "z": 5001,
                    },
                    {
                        "uuid": "goal_heading",
                        "type": "line",
                        "start": display_goal[:2],
                        "end": (
                            display_goal[0]
                            + 7.0 * math.cos(math.radians(display_goal[2])),
                            display_goal[1]
                            + 7.0 * math.sin(math.radians(display_goal[2])),
                        ),
                        "color": "#DD0000",
                        "width": 2,
                        "z": 5000,
                    },
                ]
            )

        if (
            self.current_plan is not None
            and self.current_plan.temporary_goal is not None
        ):
            drawings.append(
                {
                    "uuid": "temporary_goal",
                    "type": "point",
                    "position": self.current_plan.temporary_goal,
                    "radius": 6,
                    "color": "#AA00DD",
                    "fill": "#AA00DD",
                }
            )

        state = self.controller.car_state
        target = state.metadata.get("target_point")
        if target is not None and state.status_label == "FOLLOWING":
            drawings.extend(
                [
                    {
                        "uuid": "controller_target",
                        "type": "point",
                        "position": target,
                        "radius": 5,
                        "color": "#FF0000",
                        "fill": "#FF0000",
                    },
                    {
                        "uuid": "controller_target_line",
                        "type": "line",
                        "start": (state.x, state.y),
                        "end": target,
                        "color": "#FF000080",
                        "width": 1,
                    },
                ]
            )
        return drawings

    @staticmethod
    def _path_drawing(
        uuid: str,
        points: Sequence[Point],
        color: str,
        width: int,
    ) -> dict[str, Any]:
        return {
            "uuid": uuid,
            "type": "path",
            "points": list(points),
            "color": color,
            "width": width,
        }

    @staticmethod
    def _closed(points: Sequence[Point]) -> list[Point]:
        return list(points) + [points[0]] if points else []

    @staticmethod
    def _status_name(status: Any) -> str:
        return str(status).split(".")[-1]


def main(
    *,
    window_factory: Callable[[dict[str, Any], WorkspaceConfig, Any], Any]
    | None = None,
    render_environment: bool = True,
    camera_recording: bool = False,
) -> None:
    args = parse_args(camera_recording=camera_recording)

    # Keep hardware-only imports after argument parsing so --help works on a
    # development machine without OpenCV, serial, or Qt installed.
    from PyQt6.QtCore import QTimer

    from micromvp.config import load_config
    from micromvp.env import RealEnv
    from micromvp.gui import MVPWindow

    rvg_module = importlib.import_module("rvg")
    if not hasattr(rvg_module, "PyDynamicRVG"):
        print(
            "[main] The imported rvg module does not contain PyDynamicRVG. "
            "Put DRVG's code/build-python directory on PYTHONPATH."
        )
        return

    cfg = load_config(args.config)
    environment = RealEnv(cfg)

    print("[main] Starting real environment...")
    from micromvp.gui.startup import wait_for_camera_workspace

    if not wait_for_camera_workspace(environment, timeout=args.timeout):
        print("[main] Startup cancelled or failed; hardware stopped")
        environment.stop_all()
        environment.close()
        return

    workspace = environment.workspace_config
    if workspace.width <= 0.0 or workspace.height <= 0.0:
        print("[main] Workspace did not lock; refusing to command hardware")
        environment.stop_all()
        environment.close()
        return
    if not workspace.car_id_list:
        print("[main] No robots detected; refusing to command hardware")
        environment.stop_all()
        environment.close()
        return

    configured_id = cfg.require(
        "navigation.active_robot_id", int, who="DynamicRVG real navigation"
    )
    active_robot_id = (
        workspace.car_id_list[0] if configured_id is None else configured_id
    )
    if active_robot_id not in workspace.car_id_list:
        print(
            f"[main] Configured robot {active_robot_id} was not detected; "
            "refusing to command hardware"
        )
        environment.stop_all()
        environment.close()
        return

    controller = NavigationController.from_config(active_robot_id, workspace, cfg)
    max_speed = cfg.require(
        "control.max_speed", float, who="DynamicRVG real navigation"
    )
    configured_resolution = cfg.require(
        "planner.rvg.resolution", int, who="DynamicRVG real navigation"
    )
    configured_alpha = cfg.require(
        "planner.rvg.euclidean_weight",
        float,
        who="DynamicRVG real navigation",
    )
    configured_beta = cfg.require(
        "planner.rvg.rotational_weight",
        float,
        who="DynamicRVG real navigation",
    )
    obstacle_padding_cm = cfg.require(
        "navigation.obstacle_padding_cm",
        float,
        who="DynamicRVG real navigation",
    )
    if obstacle_padding_cm < 0.0:
        raise ValueError("navigation.obstacle_padding_cm must be non-negative")
    obstacle_draw_height_cm = cfg.require(
        "obstacle.marker_height_cm",
        float,
        who="DynamicRVG real navigation",
    )
    settings = DynamicRVGSettings(
        robot_footprint=cfg.require(
            "navigation.robot_footprint", str, who="DynamicRVG real navigation"
        ),
        optimal=cfg.require("planner.rvg.optimal", bool, who="DynamicRVG real navigation"),
        resolution=max(
            1,
            configured_resolution
            if args.resolution is None
            else args.resolution,
        ),
        num_threads=max(1, args.num_threads),
        strategy=args.strategy,
        planner_mode=args.mode,
        scan_mode=args.scan_mode,
        max_iterations=max(1, args.max_iterations),
        euclidean_weight=max(
            0.0, configured_alpha if args.alpha is None else args.alpha
        ),
        rotational_weight=max(
            0.0, configured_beta if args.beta is None else args.beta
        ),
        minimum_leaf_width_scale=max(1e-9, args.leaf_scale),
        bucket_capacity=max(1, args.bucket_capacity),
        information_weight=max(0.0, args.information_weight),
        robot_geometry_scale=cfg.require(
            "navigation.robot_geometry_scale",
            float,
            who="DynamicRVG real navigation",
        ),
    )

    print(
        f"[main] Capturing fixed obstacles for "
        f"{args.obstacle_capture_seconds:.1f} seconds..."
    )
    fixed_obstacles = capture_fixed_obstacles(
        environment,
        args.obstacle_capture_seconds,
        workspace.frequency,
    )
    print(f"[main] Captured {len(fixed_obstacles)} obstacle polygon(s)")

    try:
        navigator = DynamicRVGRealNavigator(
            workspace,
            controller,
            fixed_obstacles,
            settings,
            args.output_dir,
            draw_planner_graphs=not args.no_planner_drawings,
            goal_heading=args.goal_heading,
            obstacle_padding_cm=obstacle_padding_cm,
            obstacle_draw_height_cm=obstacle_draw_height_cm,
            execution_mode=args.execution_mode,
            preplan_timeout=args.preplan_timeout,
            scan_pause=args.scan_pause,
            discovery_view=getattr(args, "discovery_view", False),
            execution_collision_check=cfg.require(
                "navigation.execution_collision_check", bool,
                who="DynamicRVG real navigation",
            ),
        )
    except Exception:
        print("[main] Planner startup failed; stopping hardware")
        environment.stop_all()
        environment.close()
        raise

    gui_config = {
        "canvas": {
            "click_canvas_callback": True,
            "workspace_boundary_height_cm": obstacle_draw_height_cm,
        },
        "control_panel": [
            {"type": "label", "text": "=== Preplanned DRVG replay ===" if args.execution_mode == "preplanned" else "=== Real DynamicRVG ==="},
            {
                "type": "dynamic_label",
                "title": "Status",
                "widget_name": "navigation_status",
                "default": "Waiting for robot",
            },
            {
                "type": "dynamic_label",
                "title": "Planner",
                "widget_name": "planner_status",
                "default": "not initialized",
            },
            {
                "type": "dynamic_label",
                "title": "Robot pose",
                "widget_name": "robot_pose",
                "default": "unavailable",
            },
            {
                "type": "dynamic_label",
                "title": "Goal",
                "widget_name": "goal_pose",
                "default": "none",
            },
            {
                "type": "dynamic_label",
                "title": "World snapshot",
                "widget_name": "world_status",
                "default": f"{len(fixed_obstacles)} fixed polygon(s)",
            },
            {
                "type": "continuous_slider",
                "label": "Robot speed",
                "range": [0.0, 1.0],
                "default": max_speed,
                "callback_name": "set_robot_speed",
            },
            {
                "type": "input",
                "label": "Goal heading:",
                "placeholder": f"current: {args.goal_heading % 360.0:.1f}",
                "callback_name": "set_goal_heading",
            },
            {
                "type": "button",
                "label": "STOP / cancel run",
                "callback_name": "cancel_run",
            },
            {
                "type": "button",
                "label": "Recapture fixed obstacles",
                "callback_name": "recapture_obstacles",
            },
            {
                "type": "toggle",
                "label": "Draw geometry on floor",
                "default": False,
                "callback_name": "set_overlay_projection_to_floor",
            },
            {
                "type": "label",
                "text": "Click canvas: rescan obstacles, then start",
            },
            {"type": "label", "text": "C or Space: stop/cancel"},
            {"type": "label", "text": "O: recapture obstacles   Esc: quit"},
            {
                "type": "label",
                "text": "Orange: physical obstacle outline",
            },
            {"type": "label", "text": "Dark: undiscovered; orange: observed edges" if getattr(args, "discovery_view", False) else "Gold: scanned area"},
            {"type": "label", "text": "Green: plan   Blue: measured path"},
        ],
    }
    gui_config["recording_path"] = getattr(args, "record", None)
    gui_config["camera_view"] = getattr(args, "view", "perspective")
    gui_config["discovery_view"] = getattr(args, "discovery_view", False)
    gui_config["setup_seconds"] = getattr(args, "setup_seconds", 2.0)
    gui_config["blackout_seconds"] = getattr(args, "blackout_seconds", 0.5)
    gui_config["unknown_opacity"] = getattr(args, "unknown_opacity", 0.85)

    try:
        gui = (
            MVPWindow(gui_config, workspace)
            if window_factory is None
            else window_factory(gui_config, workspace, environment)
        )
    except Exception:
        environment.stop_all()
        environment.close()
        raise
    running = threading.Event()
    running.set()
    recapture_requested = threading.Event()
    recapture_generation = 0
    hardware_action_lock = threading.Lock()

    begin_goal_recording = getattr(gui, "begin_goal_recording", lambda: None)
    end_goal_recording = getattr(gui, "end_goal_recording", lambda: None)

    def stop_and_cancel(reason: str) -> None:
        environment.stop_all()
        with hardware_action_lock:
            navigator.cancel(reason)
            environment.stop_all()
        end_goal_recording()

    def on_canvas_click(x: float, y: float) -> None:
        nonlocal recapture_generation
        environment.stop_all()
        with hardware_action_lock:
            if navigator.request_goal(x, y):
                end_goal_recording()
                # Draw the new goal before capturing the first video frame.
                gui.update(*navigator.snapshot())
                begin_goal_recording()
                recapture_generation += 1
                recapture_requested.set()
            environment.stop_all()

    def request_recapture() -> None:
        nonlocal recapture_generation
        environment.stop_all()
        with hardware_action_lock:
            navigator.begin_obstacle_capture()
            recapture_generation += 1
            recapture_requested.set()
            environment.stop_all()
        end_goal_recording()

    def set_overlay_projection_to_floor(enabled: bool) -> None:
        navigator.set_overlay_projection_to_floor(enabled)
        set_boundary_on_floor = getattr(
            gui, "set_workspace_boundary_on_floor", None
        )
        if callable(set_boundary_on_floor):
            set_boundary_on_floor(enabled)

    def on_key_press(key: str) -> None:
        if key == "escape":
            environment.stop_all()
            with hardware_action_lock:
                running.clear()
                environment.stop_all()
                gui.close_window()
        elif key in {"c", "space"}:
            stop_and_cancel("EMERGENCY STOP - run cancelled")
        elif key == "o":
            request_recapture()

    gui.register_callback("on_canvas_click", on_canvas_click)
    gui.register_callback("set_robot_speed", navigator.set_controller_speed)
    gui.register_callback("set_goal_heading", navigator.set_goal_heading_text)
    gui.register_callback(
        "cancel_run", lambda: stop_and_cancel("Run cancelled; robot stopped")
    )
    gui.register_callback("recapture_obstacles", request_recapture)
    gui.register_callback(
        "set_overlay_projection_to_floor",
        set_overlay_projection_to_floor,
    )
    gui.register_callback("on_key_press", on_key_press)

    latest_observations: dict[int, RobotObservation] = {}
    observations_lock = threading.Lock()

    def logic_loop() -> None:
        nonlocal latest_observations
        period = 1.0 / max(1.0, workspace.frequency)
        while running.is_set():
            started = time.monotonic()

            if recapture_requested.is_set():
                with hardware_action_lock:
                    # A goal selected during a capture increments this value;
                    # discard that partial capture and run a full new one.
                    capture_generation = recapture_generation
                    recapture_requested.clear()
                new_obstacles = capture_fixed_obstacles(
                    environment,
                    args.obstacle_capture_seconds,
                    workspace.frequency,
                )
                with hardware_action_lock:
                    if capture_generation != recapture_generation:
                        recapture_requested.set()
                        environment.stop_all()
                        continue
                    try:
                        navigator.replace_obstacles(new_obstacles)
                    except (RuntimeError, ValueError) as error:
                        navigator.report_obstacle_capture_failed(error)
                        print(f"[DynamicRVG] Obstacle rescan rejected: {error}")
                    environment.stop_all()

            with hardware_action_lock:
                observations = environment.observe()
                actions = {
                    robot_id: Action.stop()
                    for robot_id in workspace.car_id_list
                }
                observation = observations.get(active_robot_id)
                if observation is None:
                    navigator.report_tracking_lost()
                else:
                    try:
                        actions[active_robot_id] = navigator.process_observation(observation)
                    except Exception as error:
                        environment.stop_all()
                        navigator.report_execution_failed(error)
                environment.apply_actions(actions)

            with observations_lock:
                latest_observations = dict(observations)

            remaining = period - (time.monotonic() - started)
            if remaining > 0.0:
                time.sleep(remaining)

        environment.stop_all()

    logic_thread = threading.Thread(
        target=logic_loop,
        name="dynamic-rvg-real-navigation",
        daemon=True,
    )
    logic_thread.start()

    def main_thread_update() -> None:
        if not running.is_set():
            return
        if render_environment:
            environment.render()
        with observations_lock:
            observations_available = bool(latest_observations)
        car_states, drawings = navigator.snapshot()
        if observations_available:
            gui.update(car_states, drawings)
        status, planner_status, pose, goal, world = navigator.summary()
        gui.update_widget_text("navigation_status", status)
        gui.update_widget_text("planner_status", planner_status)
        gui.update_widget_text("robot_pose", pose)
        gui.update_widget_text("goal_pose", goal)
        gui.update_widget_text("world_status", world)
        if not navigator.recording_run_active():
            end_goal_recording()

    render_timer = QTimer()
    render_timer.timeout.connect(main_thread_update)
    render_timer.start(33)

    print()
    print("=" * 64)
    print("MicroMVP real-world DynamicRVG navigation")
    print("=" * 64)
    print(f"  Workspace       : {workspace.width:.1f} x {workspace.height:.1f} cm")
    print(f"  Active robot    : {active_robot_id}")
    print(f"  Optimal RVG     : {settings.optimal}")
    print(f"  Execution mode  : {args.execution_mode}")
    print(f"  Footprint       : {settings.robot_footprint}")
    if navigator.planner.planning_circle_radius is not None:
        radius = navigator.planner.planning_circle_radius
        print(f"  Circle radius   : {radius:.2f} cm (diameter {2 * radius:.2f} cm)")
    print(f"  Search weights  : translation={settings.euclidean_weight:g}, rotation={settings.rotational_weight:g}")
    print(f"  Execution guard : {navigator.executor.reject_blocked_motion}")
    print(f"  Fixed obstacles : {len(fixed_obstacles)}")
    print(f"  Obstacle padding: {obstacle_padding_cm:.2f} cm")
    print(f"  Goal heading    : {args.goal_heading % 360.0:.1f} deg")
    print(
        "  Planner graphs  : disabled (live overlays enabled)"
        if args.no_planner_drawings else
        f"  Planner graphs  : {args.output_dir} (synchronous diagnostic output)"
    )
    print("  Start a run     : click canvas (performs a full obstacle rescan)")
    print("  Emergency stop  : Space or C")
    print("  Recapture world : O")
    print("=" * 64)

    gui.run()

    running.clear()
    render_timer.stop()
    environment.stop_all()
    logic_thread.join()
    navigator.close()
    environment.stop_all()
    environment.close()
    print("[main] Hardware stopped and environment closed")


if __name__ == "__main__":
    main()
