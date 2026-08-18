"""Run DynamicRVG through MicroMVP's differential-drive simulation."""

from __future__ import annotations

import argparse
import math
import threading
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch

from micromvp.controller import NavigationController
from micromvp.core.models import Action, CarState, Pose
from micromvp.env import SimConfig, SimEnv
from micromvp.planner import (
    PLANNER_MODES,
    SCAN_MODES,
    TEMPORARY_GOAL_STRATEGIES,
    DynamicRVGPlan,
    DynamicRVGSession,
    DynamicRVGSettings,
)


ROBOT_ID = 1
START: Pose = (12.0, 40.0, 0.0)
GOAL: Pose = (108.0, 40.0, 0.0)
OBSTACLES = [
    [(50.0, 5.0), (62.0, 5.0), (62.0, 55.0), (50.0, 55.0)],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DynamicRVG + MicroMVP SimEnv run"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show the live MicroMVP GUI instead of running headlessly",
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
        default=36,
        help="DynamicRVG angular resolution (default: 36)",
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
        default=1.0,
        help="Euclidean path weight (default: 1)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Rotational path weight (default: 0.1)",
    )
    parser.add_argument(
        "--leaf-scale",
        type=float,
        default=1.0,
        help="Quadtree minimum leaf width scale (default: 1)",
    )
    parser.add_argument(
        "--bucket-capacity",
        type=int,
        default=32,
        help="Quadtree bucket capacity (default: 32)",
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
        default=Path("simulation_output/dynamic_rvg"),
        help="Directory for per-step planner graphs and the trajectory plot",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Maximum headless wall-clock run time in seconds",
    )
    parser.add_argument(
        "--speed-scale",
        type=float,
        default=4.0,
        help="SimEnv physics speed multiplier",
    )
    parser.add_argument(
        "--no-planner-drawings",
        action="store_true",
        help="Skip the C++ planner's per-step graph drawings",
    )
    parser.add_argument(
        "--auto-close",
        type=float,
        default=0.0,
        help="GUI only: close this many seconds after completion (0 stays open)",
    )
    return parser.parse_args()


def create_environment(speed_scale: float) -> SimEnv:
    return SimEnv(
        SimConfig(
            width=120.0,
            height=80.0,
            car_width=4.2,
            car_height=4.8,
            offset_w=2.1,
            offset_h=4.5,
            wheel_base=4.2,
            max_wheel_speed=10.0,
            frequency=60.0,
            speed_scale=speed_scale,
            max_dt=0.05,
            initial_poses=[(ROBOT_ID, *START)],
        )
    )


class DynamicRVGSimulation:
    """State machine shared by the headless and live-GUI runners."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        self.environment = create_environment(args.speed_scale)
        self.workspace = self.environment.workspace_config
        self.controller = NavigationController(
            ROBOT_ID,
            self.workspace,
            lookahead_distance=2.4,
            max_speed=0.45,
            goal_tolerance=1.0,
            max_point_gap_ratio=0.05,
            no_skip_ratio=0.75,
        )
        self.planner = DynamicRVGSession(
            self.workspace,
            OBSTACLES,
            DynamicRVGSettings(
                resolution=max(1, args.resolution),
                num_threads=max(1, args.num_threads),
                strategy=args.strategy,
                planner_mode=args.mode,
                scan_mode=args.scan_mode,
                max_iterations=max(1, args.max_iterations),
                euclidean_weight=max(0.0, args.alpha),
                rotational_weight=max(0.0, args.beta),
                minimum_leaf_width_scale=max(1e-9, args.leaf_scale),
                bucket_capacity=max(1, args.bucket_capacity),
                information_weight=max(0.0, args.information_weight),
                robot_geometry_scale=1.2,
            ),
        )
        self._lock = threading.RLock()
        self.goal = GOAL
        self.start_pose = START
        self.trajectory: list[tuple[float, float]] = []
        self.planned_segments: list[list[tuple[float, float]]] = []
        self.current_plan: DynamicRVGPlan | None = None
        self.planning_step = 0
        self.needs_plan = True
        self.final_segment_active = False
        self.final_rotation_active = False
        self.paused = False
        self.finished = False
        self.failed = False
        self.status_message = "Initializing"
        self._initialize_run(START, GOAL, reset_pose=True)

    def _initialize_run(
        self,
        start: Pose,
        goal: Pose,
        reset_pose: bool,
    ) -> None:
        self.environment.apply_actions({ROBOT_ID: Action.stop()})
        if reset_pose:
            self.environment.set_pose(ROBOT_ID, *start)
        self.environment.reset_time()
        self.controller.reset()
        observation = self.environment.observe()[ROBOT_ID]
        self.start_pose = observation.pose
        self.goal = goal
        self.planner.initialize(observation.pose, goal)
        self.trajectory = [(observation.x, observation.y)]
        self.planned_segments = []
        self.current_plan = None
        self.planning_step = 0
        self.needs_plan = True
        self.final_segment_active = False
        self.final_rotation_active = False
        self.paused = False
        self.finished = False
        self.failed = False
        self.status_message = "Waiting for first planning step"

    def restart(self) -> None:
        with self._lock:
            self._initialize_run(START, GOAL, reset_pose=True)

    def set_goal(self, x: float, y: float) -> None:
        with self._lock:
            observation = self.environment.observe()[ROBOT_ID]
            goal = (x, y, self.goal[2])
            self._initialize_run(observation.pose, goal, reset_pose=False)

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self.paused = paused
            if paused:
                self.environment.apply_actions({ROBOT_ID: Action.stop()})

    def toggle_paused(self) -> None:
        with self._lock:
            self.set_paused(not self.paused)

    def set_controller_speed(self, speed: float) -> None:
        with self._lock:
            self.controller.set_speed(speed)

    def tick(self) -> None:
        with self._lock:
            observation = self.environment.observe()[ROBOT_ID]
            self.planner.update_pose(observation)

            if self.paused or self.finished or self.failed:
                self.environment.apply_actions({ROBOT_ID: Action.stop()})
                self.controller.update(observation)
                return

            if self.needs_plan:
                self._plan_next_segment()
                if self.failed:
                    self.controller.update(observation)
                    return

            action = self.controller.step(observation)
            self.environment.apply_actions({ROBOT_ID: action})
            self.trajectory.append((observation.x, observation.y))

            if self.controller.car_state.status_label == "FINISHED":
                self.environment.apply_actions({ROBOT_ID: Action.stop()})
                if self.final_segment_active:
                    self.controller.rotate_to(self.goal[2])
                    self.final_segment_active = False
                    self.final_rotation_active = True
                    self.status_message = "Performing final heading rotation"
                elif self.planner.complete_current_segment():
                    self.needs_plan = True
                    self.status_message = "Temporary segment complete; replanning"
                else:
                    self.failed = True
                    self.status_message = (
                        "Controller completed a segment but DRVG had no pending "
                        "temporary goal"
                    )

            if (
                self.final_rotation_active
                and self.controller.car_state.status_label == "ROTATION_DONE"
            ):
                self.environment.apply_actions({ROBOT_ID: Action.stop()})
                distance, heading_error = self.goal_errors(observation.pose)
                self.finished = True
                self.status_message = (
                    f"Goal reached: {distance:.3f} cm, "
                    f"{heading_error:.3f} deg"
                )

    def _plan_next_segment(self) -> None:
        plan = self.planner.step()
        self.current_plan = plan
        self.planning_step += 1
        status_name = self._status_name(plan.status)
        print(
            f"planner step {self.planning_step}: status={status_name}, "
            f"configurations={len(plan.configurations)}, "
            f"controller_points={len(plan.controller_path)}"
        )
        if not plan.has_path:
            self.failed = True
            self.status_message = f"DynamicRVG returned no path: {status_name}"
            self.environment.apply_actions({ROBOT_ID: Action.stop()})
            return

        self.planned_segments.append(plan.controller_path)
        if not self.args.no_planner_drawings:
            self.planner.draw(
                self.args.output_dir
                / f"planner_step_{self.planning_step:02d}.png"
            )
        self.controller.set_path(plan.controller_path)
        self.final_segment_active = plan.final_segment
        self.needs_plan = False
        self.status_message = (
            f"{status_name}: executing segment {self.planning_step}"
        )

    def goal_errors(self, pose: Pose) -> tuple[float, float]:
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
            return {ROBOT_ID: state_copy}, self._gui_drawings()

    def summary(self) -> tuple[str, str, str]:
        with self._lock:
            pose = self.environment.get_pose(ROBOT_ID)
            pose_text = "unavailable"
            if pose is not None:
                pose_text = f"({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.1f}°)"
            planner_text = f"{self.planning_step} planning step(s)"
            if self.paused:
                return "Paused", planner_text, pose_text
            return self.status_message, planner_text, pose_text

    def _gui_drawings(self) -> list[dict[str, Any]]:
        drawings: list[dict[str, Any]] = []

        outer_boundary, holes = self.planner.latest_visible_region()
        if outer_boundary:
            drawings.append(
                self._path_drawing(
                    "visible_region",
                    self._closed(outer_boundary),
                    "#D6A20B",
                    1,
                )
            )
        for index, hole in enumerate(holes):
            drawings.append(
                self._path_drawing(
                    f"visible_hole_{index}",
                    self._closed(hole),
                    "#00AAAA",
                    1,
                )
            )

        for index, obstacle in enumerate(OBSTACLES):
            drawings.append(
                self._path_drawing(
                    f"obstacle_{index}",
                    self._closed(obstacle),
                    "#FF6600",
                    3,
                )
            )

        for index, segment in enumerate(self.planned_segments):
            last_segment = index == len(self.planned_segments) - 1
            color = "#00AA33" if last_segment else "#77AA77"
            width = 3 if last_segment else 1
            drawings.append(
                self._path_drawing(
                    f"planned_segment_{index}", segment, color, width
                )
            )

        if len(self.trajectory) >= 2:
            drawings.append(
                self._path_drawing(
                    "measured_trajectory", self.trajectory, "#0055CC", 2
                )
            )

        drawings.extend(
            [
                {
                    "uuid": "run_start",
                    "type": "point",
                    "position": self.start_pose[:2],
                    "radius": 6,
                    "color": "#00AA00",
                    "fill": "#00AA00",
                },
                {
                    "uuid": "run_goal",
                    "type": "point",
                    "position": self.goal[:2],
                    "radius": 7,
                    "color": "#DD0000",
                    "fill": "#DD0000",
                },
                {
                    "uuid": "goal_heading",
                    "type": "line",
                    "start": self.goal[:2],
                    "end": (
                        self.goal[0] + 7.0 * math.cos(math.radians(self.goal[2])),
                        self.goal[1] + 7.0 * math.sin(math.radians(self.goal[2])),
                    ),
                    "color": "#DD0000",
                    "width": 2,
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
        drawings.append(
            {
                "uuid": "selection_indicator",
                "type": "circle",
                "center": (state.x, state.y),
                "radius": max(
                    self.workspace.car_width, self.workspace.car_height
                )
                * 0.8,
                "color": "#00AA00",
                "width": 2,
            }
        )
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
        points: list[tuple[float, float]],
        color: str,
        width: int,
    ) -> dict[str, Any]:
        return {
            "uuid": uuid,
            "type": "path",
            "points": points,
            "color": color,
            "width": width,
        }

    @staticmethod
    def _closed(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return list(points) + [points[0]] if points else []

    @staticmethod
    def _status_name(status: Any) -> str:
        return str(status).split(".")[-1]

    def write_summary_drawing(self) -> Path:
        with self._lock:
            output_path = self.args.output_dir / "trajectory.png"
            draw_trajectory(
                output_path,
                self.trajectory,
                self.planned_segments,
                self.start_pose,
                self.goal,
                OBSTACLES,
            )
            return output_path

    def close(self) -> None:
        with self._lock:
            self.environment.apply_actions({ROBOT_ID: Action.stop()})
            self.environment.close()


def draw_trajectory(
    output_path: Path,
    trajectory: list[tuple[float, float]],
    planned_segments: list[list[tuple[float, float]]],
    start: Pose,
    goal: Pose,
    obstacles: list[list[tuple[float, float]]],
) -> None:
    figure, axis = plt.subplots(figsize=(12, 8))
    axis.set_xlim(0.0, 120.0)
    axis.set_ylim(0.0, 80.0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title("DynamicRVG driven by MicroMVP SimEnv")
    axis.set_xlabel("x (cm)")
    axis.set_ylabel("y (cm)")
    axis.grid(alpha=0.25)

    for index, obstacle in enumerate(obstacles):
        axis.add_patch(
            PolygonPatch(
                obstacle,
                closed=True,
                facecolor="0.35",
                edgecolor="black",
                label="obstacle" if index == 0 else None,
            )
        )

    for index, segment in enumerate(planned_segments):
        axis.plot(
            [point[0] for point in segment],
            [point[1] for point in segment],
            "--",
            linewidth=1.2,
            label="planned segments" if index == 0 else None,
        )

    axis.plot(
        [point[0] for point in trajectory],
        [point[1] for point in trajectory],
        color="tab:blue",
        linewidth=2.0,
        label="simulated robot trajectory",
    )
    axis.scatter(
        [start[0]], [start[1]], color="tab:green", s=60, label="start"
    )
    axis.scatter(
        [goal[0]], [goal[1]], color="tab:red", s=60, label="goal"
    )
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_headless(args: argparse.Namespace) -> None:
    simulation = DynamicRVGSimulation(args)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline and not simulation.finished:
        simulation.tick()
        if simulation.failed:
            raise RuntimeError(simulation.status_message)
        time.sleep(1.0 / simulation.workspace.frequency)

    if not simulation.finished:
        raise RuntimeError(
            f"simulation did not reach the goal within {args.timeout:.1f} seconds"
        )

    final_pose = simulation.environment.get_pose(ROBOT_ID)
    trajectory_path = simulation.write_summary_drawing()
    print(simulation.status_message)
    print(f"final pose: {final_pose}")
    print(f"planning steps: {simulation.planning_step}")
    print(f"trajectory samples: {len(simulation.trajectory)}")
    print(f"trajectory drawing: {trajectory_path}")
    simulation.close()


def run_gui(args: argparse.Namespace) -> None:
    from PyQt6.QtCore import QTimer

    from micromvp.gui import MVPWindow

    simulation = DynamicRVGSimulation(args)
    gui_config = {
        "canvas": {"click_canvas_callback": True},
        "control_panel": [
            {"type": "label", "text": "=== DynamicRVG Simulation ==="},
            {
                "type": "dynamic_label",
                "title": "Status",
                "widget_name": "simulation_status",
                "default": "Initializing",
            },
            {
                "type": "dynamic_label",
                "title": "Planner",
                "widget_name": "planner_status",
                "default": "0 planning steps",
            },
            {
                "type": "dynamic_label",
                "title": "Pose",
                "widget_name": "robot_pose",
                "default": "unavailable",
            },
            {
                "type": "continuous_slider",
                "label": "Robot Speed",
                "range": [0.0, 1.0],
                "default": 0.45,
                "callback_name": "set_robot_speed",
            },
            {
                "type": "toggle",
                "label": "Pause",
                "default": False,
                "callback_name": "set_paused",
            },
            {"type": "label", "text": "Click the canvas to set a new goal"},
            {"type": "label", "text": "R: restart   Space: pause   Esc: quit"},
            {"type": "label", "text": "Orange: obstacle   Gold: visible area"},
            {"type": "label", "text": "Green: plan   Blue: measured trajectory"},
        ],
    }
    gui = MVPWindow(gui_config, simulation.workspace)
    running = threading.Event()
    running.set()

    gui.register_callback("on_canvas_click", simulation.set_goal)
    gui.register_callback("set_robot_speed", simulation.set_controller_speed)
    gui.register_callback("set_paused", simulation.set_paused)

    def on_key_press(key: str) -> None:
        if key == "escape":
            running.clear()
            gui.close_window()
        elif key == "r":
            simulation.restart()
        elif key == "space":
            simulation.toggle_paused()

    gui.register_callback("on_key_press", on_key_press)

    def logic_loop() -> None:
        while running.is_set():
            started = time.perf_counter()
            simulation.tick()
            states, drawings = simulation.snapshot()
            status, planner_status, pose = simulation.summary()
            gui.update(states, drawings)
            gui.update_widget_text("simulation_status", status)
            gui.update_widget_text("planner_status", planner_status)
            gui.update_widget_text("robot_pose", pose)
            elapsed = time.perf_counter() - started
            time.sleep(max(0.0, 1.0 / simulation.workspace.frequency - elapsed))

    logic_thread = threading.Thread(target=logic_loop, daemon=True)
    logic_thread.start()

    completion_time: list[float | None] = [None]

    def check_auto_close() -> None:
        if args.auto_close <= 0.0:
            return
        if not simulation.finished and not simulation.failed:
            completion_time[0] = None
            return
        if completion_time[0] is None:
            completion_time[0] = time.monotonic()
            return
        if time.monotonic() - completion_time[0] >= args.auto_close:
            running.clear()
            gui.close_window()

    auto_close_timer = QTimer()
    auto_close_timer.timeout.connect(check_auto_close)
    auto_close_timer.start(100)

    print("DynamicRVG live simulation")
    print("  Click the canvas to choose a new goal")
    print("  Press R to restart, Space to pause, Escape to quit")
    gui.run()

    running.clear()
    auto_close_timer.stop()
    logic_thread.join(timeout=2.0)
    trajectory_path = simulation.write_summary_drawing()
    simulation.close()
    print(f"trajectory drawing: {trajectory_path}")


def main() -> None:
    args = parse_args()
    if args.gui:
        run_gui(args)
    else:
        run_headless(args)


if __name__ == "__main__":
    main()
