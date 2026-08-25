"""Persistent DynamicRVG session driven by MicroMVP pose observations."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from micromvp.core.models import Point, Pose, RobotObservation, WorkspaceConfig


PolygonPoints = Sequence[Point]

TEMPORARY_GOAL_STRATEGIES = (
    "greedy",
    "quadtree",
    "astar",
    "information",
    "quadtree_astar",
    "quadtree_frontier_astar",
    "quadtree_information",
    "astar_information",
    "quadtree_frontier_astar_information",
)
PLANNER_MODES = (
    "graph_merge",
    "delta_graph_merge",
    "buffered_delta_graph_merge",
    "incremental",
)
SCAN_MODES = ("center", "footprint")
_GEOMETRY_EPSILON = 1e-9


def _cross_product(origin: Point, first: Point, second: Point) -> float:
    return (
        (first[0] - origin[0]) * (second[1] - origin[1])
        - (first[1] - origin[1]) * (second[0] - origin[0])
    )


def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
    if abs(_cross_product(start, end, point)) > _GEOMETRY_EPSILON:
        return False
    return (
        min(start[0], end[0]) - _GEOMETRY_EPSILON
        <= point[0]
        <= max(start[0], end[0]) + _GEOMETRY_EPSILON
        and min(start[1], end[1]) - _GEOMETRY_EPSILON
        <= point[1]
        <= max(start[1], end[1]) + _GEOMETRY_EPSILON
    )


def _segments_intersect(
    first_start: Point,
    first_end: Point,
    second_start: Point,
    second_end: Point,
) -> bool:
    first_side_start = _cross_product(first_start, first_end, second_start)
    first_side_end = _cross_product(first_start, first_end, second_end)
    second_side_start = _cross_product(second_start, second_end, first_start)
    second_side_end = _cross_product(second_start, second_end, first_end)
    if (
        first_side_start * first_side_end < -_GEOMETRY_EPSILON
        and second_side_start * second_side_end < -_GEOMETRY_EPSILON
    ):
        return True
    return (
        _point_on_segment(second_start, first_start, first_end)
        or _point_on_segment(second_end, first_start, first_end)
        or _point_on_segment(first_start, second_start, second_end)
        or _point_on_segment(first_end, second_start, second_end)
    )


def _point_in_polygon(point: Point, polygon: PolygonPoints) -> bool:
    inside = False
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        if _point_on_segment(point, start, end):
            return True
        if (start[1] > point[1]) == (end[1] > point[1]):
            continue
        crossing_x = (
            start[0]
            + (point[1] - start[1])
            * (end[0] - start[0])
            / (end[1] - start[1])
        )
        if crossing_x > point[0]:
            inside = not inside
    return inside


def _polygons_intersect(
    first: PolygonPoints,
    second: PolygonPoints,
) -> bool:
    for first_index, first_start in enumerate(first):
        first_end = first[(first_index + 1) % len(first)]
        for second_index, second_start in enumerate(second):
            second_end = second[(second_index + 1) % len(second)]
            if _segments_intersect(
                first_start,
                first_end,
                second_start,
                second_end,
            ):
                return True
    return _point_in_polygon(first[0], second) or _point_in_polygon(
        second[0], first
    )


def pad_obstacle_polygon(
    polygon: PolygonPoints,
    padding: float,
) -> list[Point]:
    """Offset a simple obstacle polygon outward with mitered corners."""
    points = [(float(x), float(y)) for x, y in polygon]
    distance = max(0.0, float(padding))
    if len(points) < 3 or distance == 0.0:
        return points

    signed_area = 0.5 * sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )
    if abs(signed_area) <= 1e-12:
        raise ValueError("cannot pad a zero-area obstacle polygon")
    if signed_area < 0.0:
        points.reverse()

    padded: list[Point] = []
    for index, current in enumerate(points):
        previous = points[index - 1]
        following = points[(index + 1) % len(points)]
        previous_edge = (
            current[0] - previous[0],
            current[1] - previous[1],
        )
        following_edge = (
            following[0] - current[0],
            following[1] - current[1],
        )
        previous_length = math.hypot(*previous_edge)
        following_length = math.hypot(*following_edge)
        if previous_length <= 1e-12 or following_length <= 1e-12:
            raise ValueError("cannot pad an obstacle polygon with duplicate vertices")

        previous_direction = (
            previous_edge[0] / previous_length,
            previous_edge[1] / previous_length,
        )
        following_direction = (
            following_edge[0] / following_length,
            following_edge[1] / following_length,
        )
        previous_normal = (previous_direction[1], -previous_direction[0])
        following_normal = (following_direction[1], -following_direction[0])
        previous_line = (
            current[0] + distance * previous_normal[0],
            current[1] + distance * previous_normal[1],
        )
        following_line = (
            current[0] + distance * following_normal[0],
            current[1] + distance * following_normal[1],
        )
        denominator = (
            previous_direction[0] * following_direction[1]
            - previous_direction[1] * following_direction[0]
        )
        if abs(denominator) <= 1e-12:
            normal_x = previous_normal[0] + following_normal[0]
            normal_y = previous_normal[1] + following_normal[1]
            normal_length = math.hypot(normal_x, normal_y)
            if normal_length <= 1e-12:
                raise ValueError("cannot pad a polygon with a 180-degree vertex")
            padded.append(
                (
                    current[0] + distance * normal_x / normal_length,
                    current[1] + distance * normal_y / normal_length,
                )
            )
            continue

        delta_x = following_line[0] - previous_line[0]
        delta_y = following_line[1] - previous_line[1]
        line_parameter = (
            delta_x * following_direction[1]
            - delta_y * following_direction[0]
        ) / denominator
        padded.append(
            (
                previous_line[0] + line_parameter * previous_direction[0],
                previous_line[1] + line_parameter * previous_direction[1],
            )
        )
    return padded


def pad_obstacles(
    obstacles: Sequence[PolygonPoints],
    padding: float,
) -> list[list[Point]]:
    """Return valid obstacle polygons with an outward padding in workspace units."""
    return [
        pad_obstacle_polygon(obstacle, padding)
        for obstacle in obstacles
        if len(obstacle) >= 3
    ]


def _merge_intersecting_rvg_polygons(polygons: Sequence[Any]) -> list[Any]:
    """Union polygons whose boundaries cross before DRVG builds its arrangement."""
    merged: list[Any] = []
    for polygon in polygons:
        candidate = polygon
        index = 0
        while index < len(merged):
            existing = merged[index]
            if not candidate.intersects(existing):
                index += 1
                continue

            union = candidate.merge(existing)
            if union.size() < 3:
                raise RuntimeError(
                    "DynamicRVG could not merge intersecting obstacle polygons"
                )
            candidate = union
            merged.pop(index)
            # The larger union can intersect a polygon checked previously.
            index = 0
        merged.append(candidate)
    return merged


@dataclass(frozen=True, slots=True)
class DynamicRVGSettings:
    """Settings passed to the C++ DynamicRVG implementation."""

    resolution: int = 36
    num_threads: int = 1
    strategy: str = "quadtree"
    planner_mode: str = "graph_merge"
    scan_mode: str = "center"
    max_iterations: int = 10000
    euclidean_weight: float = 1.0
    rotational_weight: float = 0.1
    minimum_leaf_width_scale: float = 1.0
    bucket_capacity: int = 32
    information_weight: float = 1.0
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

        geometry = [
            (float(x), float(y))
            for x, y in (
                robot_geometry or self.default_robot_geometry(workspace)
            )
        ]
        self._base_robot_geometry = list(geometry)
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
        obstacle_points = [
            [(float(x), float(y)) for x, y in obstacle]
            for obstacle in obstacles
            if len(obstacle) >= 3
        ]
        self._obstacle_points = obstacle_points
        self._validate_obstacle_bounds(workspace, obstacle_points)
        world_obstacles = [
            self._rvg.polygon(
                [self._rvg.vertex(x, y) for x, y in obstacle], False
            )
            for obstacle in obstacle_points
        ]
        original_obstacle_count = len(world_obstacles)
        world_obstacles = _merge_intersecting_rvg_polygons(world_obstacles)
        if len(world_obstacles) != original_obstacle_count:
            print(
                f"[DynamicRVG] Merged {original_obstacle_count} intersecting "
                f"obstacle polygons into {len(world_obstacles)}"
            )

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
        temporary_goal_options = self._temporary_goal_options(settings)
        if not self._planner.setTemporaryGoalOptions(temporary_goal_options):
            raise RuntimeError("DynamicRVG rejected temporary-goal options")
        planner_mode = self._planner_mode(settings.planner_mode)
        if not self._planner.setPlannerMode(planner_mode):
            raise RuntimeError("DynamicRVG rejected planner mode")
        self._scan_mode = self._scan_mode_value(settings.scan_mode)

    @staticmethod
    def _validate_obstacle_bounds(
        workspace: WorkspaceConfig,
        obstacles: Sequence[PolygonPoints],
    ) -> None:
        for polygon_index, obstacle in enumerate(obstacles):
            for point_index, (x, y) in enumerate(obstacle):
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError(
                        f"obstacle {polygon_index} point {point_index} is not finite"
                    )
                if 0.0 <= x <= workspace.width and 0.0 <= y <= workspace.height:
                    continue
                raise ValueError(
                    f"obstacle {polygon_index} crosses the workspace boundary at "
                    f"({x:.3f}, {y:.3f}); move it inward or reduce obstacle padding"
                )

    def _temporary_goal_options(
        self, settings: DynamicRVGSettings
    ) -> Any:
        if settings.strategy not in TEMPORARY_GOAL_STRATEGIES:
            raise ValueError(
                f"unsupported DynamicRVG strategy: {settings.strategy}"
            )
        options = self._rvg.TemporaryGoalOptions()
        options.useQuadtree = settings.strategy.startswith("quadtree")
        options.scoreMode = (
            self._rvg.TemporaryGoalScoreMode.AStar
            if "astar" in settings.strategy
            else self._rvg.TemporaryGoalScoreMode.GoalGreedy
        )
        options.useFrontierCandidates = "frontier" in settings.strategy
        options.useInformationGain = "information" in settings.strategy
        options.minimumLeafWidthScale = settings.minimum_leaf_width_scale
        options.bucketCapacity = settings.bucket_capacity
        options.informationWeight = settings.information_weight
        options.maxIterations = settings.max_iterations
        return options

    def _planner_mode(self, mode: str) -> Any:
        values = {
            "graph_merge": self._rvg.PyDynamicRVGMode.GraphMerge,
            "delta_graph_merge": (
                self._rvg.PyDynamicRVGMode.ExactObservationDelta
            ),
            "buffered_delta_graph_merge": (
                self._rvg.PyDynamicRVGMode.BufferedObservationDelta
            ),
            "incremental": self._rvg.PyDynamicRVGMode.IncrementalMapping,
        }
        if mode not in values:
            raise ValueError(f"unsupported DynamicRVG planner mode: {mode}")
        return values[mode]

    def _scan_mode_value(self, mode: str) -> Any:
        values = {
            "center": self._rvg.ScanMode.Center,
            "footprint": self._rvg.ScanMode.FootprintVertices,
        }
        if mode not in values:
            raise ValueError(f"unsupported DynamicRVG scan mode: {mode}")
        return values[mode]

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

    def pose_is_valid(
        self,
        pose: Pose | RobotObservation,
        robot_geometry_scale: float = 1.0,
    ) -> bool:
        """Check a measured pose against the exact footprint and fixed world."""
        measured_pose = pose.pose if isinstance(pose, RobotObservation) else pose
        x, y, theta_degrees = measured_pose
        scale = float(robot_geometry_scale)
        if scale <= 0.0 or not all(
            math.isfinite(value) for value in measured_pose
        ):
            return False

        cosine = math.cos(math.radians(theta_degrees))
        sine = math.sin(math.radians(theta_degrees))
        footprint = [
            (
                x + scale * (local_x * cosine - local_y * sine),
                y + scale * (local_x * sine + local_y * cosine),
            )
            for local_x, local_y in self._base_robot_geometry
        ]
        if any(
            point_x <= 0.0
            or point_x >= self._workspace.width
            or point_y <= 0.0
            or point_y >= self._workspace.height
            for point_x, point_y in footprint
        ):
            return False
        return not any(
            _polygons_intersect(footprint, obstacle)
            for obstacle in self._obstacle_points
        )

    def scan(self) -> Any:
        return self._planner.scan(self._scan_mode)

    def step(self) -> DynamicRVGPlan:
        result = self._planner.step(self._scan_mode)
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
