"""Persistent DynamicRVG session driven by MicroMVP pose observations."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from micromvp.core.models import Point, Pose, RobotObservation, WorkspaceConfig


PolygonPoints = Sequence[Point]


@dataclass(frozen=True, slots=True)
class DynamicRVGSettings:
    """Settings passed to the C++ DynamicRVG implementation."""

    resolution: int = 36
    num_threads: int = 1
    euclidean_weight: float = 1.0
    rotational_weight: float = 0.1
    minimum_leaf_width_scale: float = 1.0
    bucket_capacity: int = 32
    robot_geometry_scale: float = 1.2


@dataclass(slots=True)
class DynamicRVGPlan:
    """One planner segment and its controller-ready projection."""

    status: Any
    configurations: list[Any]
    controller_path: list[Point]
    temporary_goal: Point | None
    final_segment: bool

    @property
    def has_path(self) -> bool:
        return len(self.configurations) > 1 and len(self.controller_path) > 1


class DynamicRVGSession:
    """Own one fixed-world DynamicRVG instance for a navigation run.

    MicroMVP owns the robot pose and completion tolerances. The C++ planner
    owns the accumulated visibility graph and temporary-goal selection state.
    """

    def __init__(
        self,
        workspace: WorkspaceConfig,
        obstacles: Sequence[PolygonPoints],
        settings: DynamicRVGSettings = DynamicRVGSettings(),
        robot_geometry: PolygonPoints | None = None,
    ) -> None:
        self._rvg = importlib.import_module("rvg")
        self._workspace = workspace
        self._settings = settings
        self._heading_marker_length = max(
            workspace.car_width, workspace.car_height
        ) * 0.25

        geometry = list(robot_geometry or self.default_robot_geometry(workspace))
        geometry = [
            (
                x * settings.robot_geometry_scale,
                y * settings.robot_geometry_scale,
            )
            for x, y in geometry
        ]

        border = self._rvg.polygon(
            [
                self._rvg.vertex(0.0, 0.0),
                self._rvg.vertex(workspace.width, 0.0),
                self._rvg.vertex(workspace.width, workspace.height),
                self._rvg.vertex(0.0, workspace.height),
            ],
            False,
        )
        robot = self._rvg.polygon(
            [self._rvg.vertex(x, y) for x, y in geometry],
            self._rvg.vertex(0.0, 0.0),
            False,
        )
        world_obstacles = [
            self._rvg.polygon(
                [self._rvg.vertex(x, y) for x, y in obstacle], False
            )
            for obstacle in obstacles
            if len(obstacle) >= 3
        ]

        self._planner = self._rvg.PyDynamicRVG(
            robot=robot,
            border=border,
            worldObstacles=world_obstacles,
            resolution=settings.resolution,
            numThreads=settings.num_threads,
        )
        self._planner.setWeight(
            settings.euclidean_weight, settings.rotational_weight
        )
        self._planner.setQuadtreeOptions(
            settings.minimum_leaf_width_scale, settings.bucket_capacity
        )

    @staticmethod
    def default_robot_geometry(workspace: WorkspaceConfig) -> list[Point]:
        """Return the same axle-centred footprint used by NavigationCoordinator."""
        left_extent = workspace.offset_w
        right_extent = workspace.car_width - workspace.offset_w
        rear_extent = workspace.offset_h
        front_extent = workspace.car_height - workspace.offset_h
        planner_front_extent = max(front_extent, left_extent, right_extent)
        return [
            (-rear_extent, left_extent),
            (-rear_extent, -right_extent),
            (planner_front_extent, -right_extent),
            (planner_front_extent, left_extent),
        ]

    @property
    def planner(self) -> Any:
        """Expose the bound planner for graph inspection and experiments."""
        return self._planner

    def initialize(self, start: Pose, goal: Pose) -> None:
        self._planner.initialize(self._vertex(start), self._vertex(goal))

    def update_pose(self, pose: Pose | RobotObservation) -> None:
        measured_pose = pose.pose if isinstance(pose, RobotObservation) else pose
        self._planner.updateRobotPose(self._vertex(measured_pose))

    def scan(self) -> Any:
        return self._planner.scan(self._rvg.ScanMode.FootprintVertices)

    def step(self) -> DynamicRVGPlan:
        result = self._planner.step(self._rvg.ScanMode.FootprintVertices)
        configurations = list(result.path)
        temporary_goal = None
        if result.temporaryGoal is not None:
            temporary_goal = (
                result.temporaryGoal.getX(),
                result.temporaryGoal.getY(),
            )
        return DynamicRVGPlan(
            status=result.status,
            configurations=configurations,
            controller_path=self._to_controller_path(configurations),
            temporary_goal=temporary_goal,
            final_segment=(
                result.status == self._rvg.PyDynamicRVGStatus.GoalPathAvailable
            ),
        )

    def complete_current_segment(self) -> bool:
        """Acknowledge that the controller completed its current segment."""
        return self._planner.completeCurrentSegment()

    def draw(self, figure_path: str | Path) -> bool:
        return self._planner.draw(str(figure_path))

    def latest_visible_region(self) -> tuple[list[Point], list[list[Point]]]:
        """Return the most recent simulated scan as GUI-friendly polylines."""
        observation = self._planner.latestObservation()
        if not observation.success:
            return [], []
        outer_boundary = self._polygon_points(observation.outerBoundary)
        holes = [self._polygon_points(hole) for hole in observation.holes]
        return outer_boundary, holes

    def _vertex(self, pose: Pose) -> Any:
        x, y, theta_degrees = pose
        theta = math.radians(theta_degrees) % (2.0 * math.pi)
        return self._rvg.vertex(x, y, 0.0, 2.0 * math.pi, theta)

    @staticmethod
    def _polygon_points(polygon: Any) -> list[Point]:
        return [(vertex.getX(), vertex.getY()) for vertex in polygon.getVertices()]

    def _to_controller_path(self, configurations: Sequence[Any]) -> list[Point]:
        """Project an SE(2) RVG path to NavigationController waypoints."""
        path: list[Point] = []
        previous_xy: Point | None = None
        previous_theta: float | None = None

        for index, configuration in enumerate(configurations):
            xy = (configuration.getX(), configuration.getY())
            theta = configuration.getTheta()

            if previous_xy is None:
                path.append(xy)
                previous_xy = xy
                previous_theta = theta
                continue

            same_xy = (
                abs(xy[0] - previous_xy[0]) <= 1e-6
                and abs(xy[1] - previous_xy[1]) <= 1e-6
            )
            if same_xy:
                reference_theta = (
                    previous_theta if previous_theta is not None else 0.0
                )
                delta_theta = (
                    theta - reference_theta + math.pi
                ) % (2.0 * math.pi) - math.pi
                terminal_rotation = index == len(configurations) - 1
                if abs(delta_theta) > math.radians(1.0) and not terminal_rotation:
                    marker = (
                        xy[0] + self._heading_marker_length * math.cos(theta),
                        xy[1] + self._heading_marker_length * math.sin(theta),
                    )
                    if (
                        abs(marker[0] - path[-1][0]) > 1e-6
                        or abs(marker[1] - path[-1][1]) > 1e-6
                    ):
                        path.append(marker)
                previous_theta = theta
                continue

            if (
                abs(xy[0] - path[-1][0]) > 1e-6
                or abs(xy[1] - path[-1][1]) > 1e-6
            ):
                path.append(xy)
            previous_xy = xy
            previous_theta = theta

        return path
