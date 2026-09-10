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
_BORDER_ATTACHMENT_EPSILON_CM = 1e-4


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


def _clip_polygon_to_axis(
    polygon: Sequence[Point],
    axis: int,
    bound: float,
    keep_greater: bool,
) -> list[Point]:
    """Clip a polygon against one axis-aligned half-plane."""
    if not polygon:
        return []

    def inside(point: Point) -> bool:
        value = point[axis]
        return value >= bound if keep_greater else value <= bound

    clipped: list[Point] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside != previous_inside:
            denominator = current[axis] - previous[axis]
            if abs(denominator) > _GEOMETRY_EPSILON:
                fraction = (bound - previous[axis]) / denominator
                intersection = (
                    previous[0] + fraction * (current[0] - previous[0]),
                    previous[1] + fraction * (current[1] - previous[1]),
                )
                clipped.append(intersection)
        if current_inside:
            clipped.append(current)
        previous = current
        previous_inside = current_inside
    return clipped


def _clean_polygon_points(polygon: Sequence[Point]) -> list[Point]:
    """Remove duplicate and redundant collinear clipping vertices."""
    cleaned: list[Point] = []
    for point in polygon:
        candidate = (float(point[0]), float(point[1]))
        if cleaned and math.dist(candidate, cleaned[-1]) <= _GEOMETRY_EPSILON:
            continue
        cleaned.append(candidate)
    if (
        len(cleaned) >= 2
        and math.dist(cleaned[0], cleaned[-1]) <= _GEOMETRY_EPSILON
    ):
        cleaned.pop()

    changed = True
    while changed and len(cleaned) >= 3:
        changed = False
        simplified: list[Point] = []
        for index, current in enumerate(cleaned):
            previous = cleaned[index - 1]
            following = cleaned[(index + 1) % len(cleaned)]
            if abs(_cross_product(previous, current, following)) <= (
                _GEOMETRY_EPSILON
            ) and _point_on_segment(current, previous, following):
                changed = True
                continue
            simplified.append(current)
        cleaned = simplified

    if len(cleaned) < 3:
        return []
    signed_area = 0.5 * sum(
        cleaned[index][0] * cleaned[(index + 1) % len(cleaned)][1]
        - cleaned[(index + 1) % len(cleaned)][0] * cleaned[index][1]
        for index in range(len(cleaned))
    )
    if abs(signed_area) <= _GEOMETRY_EPSILON:
        return []
    if signed_area < 0.0:
        cleaned.reverse()
    return cleaned


def attach_obstacles_to_workspace_border(
    workspace: WorkspaceConfig,
    obstacles: Sequence[PolygonPoints],
) -> tuple[list[list[Point]], int]:
    """Prepare visible obstacles that touch or cross the workspace border.

    DRVG inserts the raw border and obstacle edges as non-intersecting CGAL
    constraints, so an observed obstacle cannot literally cross that border.
    Border obstacles are clipped a microscopic distance inside the workspace.
    DRVG's subsequent robot-footprint expansion then joins them to the outside
    border in configuration space, producing the intended boundary indentation
    without creating an unsafe navigable gap.
    """
    width = float(workspace.width)
    height = float(workspace.height)
    if width <= 0.0 or height <= 0.0:
        raise ValueError("workspace dimensions must be positive")
    inset = min(
        _BORDER_ATTACHMENT_EPSILON_CM,
        0.001 * width,
        0.001 * height,
    )
    minimum_x, maximum_x = inset, width - inset
    minimum_y, maximum_y = inset, height - inset
    if minimum_x >= maximum_x or minimum_y >= maximum_y:
        raise ValueError("workspace is too small to attach border obstacles")

    prepared: list[list[Point]] = []
    attached_count = 0
    for polygon_index, obstacle in enumerate(obstacles):
        if len(obstacle) < 3:
            continue
        points: list[Point] = []
        for point_index, (x, y) in enumerate(obstacle):
            point = (float(x), float(y))
            if not math.isfinite(point[0]) or not math.isfinite(point[1]):
                raise ValueError(
                    f"obstacle {polygon_index} point {point_index} is not finite"
                )
            points.append(point)

        intersects_border = any(
            x <= 0.0 or x >= width or y <= 0.0 or y >= height
            for x, y in points
        )
        if not intersects_border:
            prepared.append(points)
            continue

        clipped: list[Point] = points
        clipped = _clip_polygon_to_axis(clipped, 0, minimum_x, True)
        clipped = _clip_polygon_to_axis(clipped, 0, maximum_x, False)
        clipped = _clip_polygon_to_axis(clipped, 1, minimum_y, True)
        clipped = _clip_polygon_to_axis(clipped, 1, maximum_y, False)
        clipped = _clean_polygon_points(clipped)
        if len(clipped) < 3:
            # A registered obstacle wholly outside the usable workspace does
            # not remove any free space and is safe to ignore.
            continue
        prepared.append(clipped)
        attached_count += 1
    return prepared, attached_count


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
    robot_footprint: str = "polygon"
    optimal: bool = False


@dataclass(slots=True)
class DynamicRVGPlan:
    """One planner segment with SE(2) configurations and its XY display polyline."""

    status: Any
    configurations: list[Any]
    controller_path: list[Point]
    temporary_goal: Point | None
    final_segment: bool

    @property
    def has_path(self) -> bool:
        return len(self.configurations) > 1


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

        self._circle_radius = None
        if settings.robot_footprint == "circle":
            body = robot_geometry or self.physical_robot_geometry(workspace)
            self._circle_radius = max(math.hypot(x, y) for x, y in body)
            geometry = self.circular_robot_geometry(body)
        elif settings.robot_footprint == "polygon":
            geometry = list(robot_geometry or self.default_robot_geometry(workspace))
        else:
            raise ValueError("robot_footprint must be 'polygon' or 'circle'")
        geometry = [(float(x), float(y)) for x, y in geometry]
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
        detected_obstacle_points = [
            [(float(x), float(y)) for x, y in obstacle]
            for obstacle in obstacles
            if len(obstacle) >= 3
        ]
        obstacle_points, border_attached_count = (
            attach_obstacles_to_workspace_border(
                workspace,
                detected_obstacle_points,
            )
        )
        self._obstacle_points = obstacle_points
        if border_attached_count:
            print(
                f"[DynamicRVG] Merged {border_attached_count} visible "
                "border-intersecting obstacle polygon(s) into the workspace "
                "boundary"
            )
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

        # Retain the exact native inputs for solver-geometry inspection.
        self._native_robot = robot
        self._native_border = border
        self._native_obstacles = world_obstacles

        self._planner = self._rvg.PyDynamicRVG(
            robot=robot,
            border=border,
            worldObstacles=world_obstacles,
            resolution=settings.resolution,
            numThreads=settings.num_threads,
        )
        if settings.optimal:
            setter = getattr(self._planner, "setOptimalRvgConstruction", None)
            if setter is None or not setter(True):
                raise RuntimeError(
                    "DRVG optimal mode requires updated Python bindings; rebuild ~/drvg/code/build-python"
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
        self._current_pose: Pose | None = None
        self._goal_pose: Pose | None = None

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
    def physical_robot_geometry(workspace: WorkspaceConfig) -> list[Point]:
        left = workspace.offset_w
        right = workspace.car_width - workspace.offset_w
        rear = workspace.offset_h
        front = workspace.car_height - workspace.offset_h
        return [(-rear, left), (-rear, -right), (front, -right), (front, left)]

    @staticmethod
    def circular_robot_geometry(body: PolygonPoints, sides: int = 72) -> list[Point]:
        """Circumscribe the body's full rotation disk, never inscribe it."""
        radius = max(math.hypot(x, y) for x, y in body)
        if not math.isfinite(radius) or radius <= 0 or sides < 8:
            raise ValueError("Invalid circular footprint")
        vertex_radius = radius / math.cos(math.pi / sides)
        return [(vertex_radius * math.cos(2 * math.pi * i / sides),
                 vertex_radius * math.sin(2 * math.pi * i / sides))
                for i in range(sides)]

    @property
    def planning_circle_radius(self) -> float | None:
        if self._circle_radius is None:
            return None
        return self._circle_radius * self._settings.robot_geometry_scale

    def _circle_sweep_is_valid(self, start: Point, end: Point, radius: float) -> bool:
        """Exact disk/capsule check against polygon edges and workspace bounds."""
        if any(x - radius < -_GEOMETRY_EPSILON
               or x + radius > self._workspace.width + _GEOMETRY_EPSILON
               or y - radius < -_GEOMETRY_EPSILON
               or y + radius > self._workspace.height + _GEOMETRY_EPSILON
               for x, y in (start, end)):
            return False

        def point_segment_distance(point, a, b):
            dx, dy = b[0] - a[0], b[1] - a[1]
            length_squared = dx * dx + dy * dy
            if length_squared == 0:
                return math.dist(point, a)
            t = max(0.0, min(1.0, ((point[0]-a[0])*dx + (point[1]-a[1])*dy) / length_squared))
            return math.hypot(point[0] - a[0] - t*dx, point[1] - a[1] - t*dy)

        for obstacle in self._obstacle_points:
            if _point_in_polygon(start, obstacle) or _point_in_polygon(end, obstacle):
                return False
            for index, a in enumerate(obstacle):
                b = obstacle[(index + 1) % len(obstacle)]
                if _segments_intersect(start, end, a, b):
                    return False
                distance = min(
                    point_segment_distance(start, a, b),
                    point_segment_distance(end, a, b),
                    point_segment_distance(a, start, end),
                    point_segment_distance(b, start, end),
                )
                if distance < radius - _GEOMETRY_EPSILON:
                    return False
        return True

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
        self._current_pose = start
        self._goal_pose = goal
        self._planner.initialize(self._vertex(start), self._vertex(goal))

    def update_pose(self, pose: Pose | RobotObservation) -> None:
        measured_pose = pose.pose if isinstance(pose, RobotObservation) else pose
        self._current_pose = measured_pose
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

        if self._settings.robot_footprint == "circle":
            return self._circle_sweep_is_valid((x, y), (x, y), self._circle_radius * scale)

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

    def motion_is_valid(self, start: Pose, end: Pose) -> bool:
        """Conservatively cover translation plus shortest rotation with sweeps.

        Each <=5 degree interval uses the convex hull of its endpoint
        footprints, expanded by R*(1-cos(angle/2)) to cover circular arcs.
        The supplied obstacles already include deployment padding.
        """
        if not all(math.isfinite(value) for value in (*start, *end)):
            return False
        scale = self._settings.robot_geometry_scale
        if self._settings.robot_footprint == "circle":
            return self._circle_sweep_is_valid(start[:2], end[:2], self._circle_radius * scale)
        delta = (end[2] - start[2] + 180.0) % 360.0 - 180.0
        steps = max(1, math.ceil(abs(delta) / 5.0))
        radius = max(math.hypot(x, y) for x, y in self._base_robot_geometry) * scale
        arc_padding = radius * (1.0 - math.cos(math.radians(abs(delta) / steps) / 2.0))

        def footprint(fraction):
            x = start[0] + fraction * (end[0] - start[0])
            y = start[1] + fraction * (end[1] - start[1])
            theta = math.radians(start[2] + fraction * delta)
            c, sn = math.cos(theta), math.sin(theta)
            return [(x + scale * (px * c - py * sn),
                     y + scale * (px * sn + py * c))
                    for px, py in self._base_robot_geometry]

        previous = footprint(0)
        for step in range(1, steps + 1):
            current = footprint(step / steps)
            points = sorted(set(previous + current))
            lower = []
            for point in points:
                while len(lower) >= 2 and _cross_product(lower[-2], lower[-1], point) <= 0:
                    lower.pop()
                lower.append(point)
            upper = []
            for point in reversed(points):
                while len(upper) >= 2 and _cross_product(upper[-2], upper[-1], point) <= 0:
                    upper.pop()
                upper.append(point)
            swept = lower[:-1] + upper[:-1]
            if arc_padding > 0:
                swept = pad_obstacle_polygon(swept, arc_padding)
            if any(x <= 0 or x >= self._workspace.width or y <= 0 or y >= self._workspace.height
                   for x, y in swept):
                return False
            if any(_polygons_intersect(swept, obstacle) for obstacle in self._obstacle_points):
                return False
            previous = current
        return True

    def _rotation_is_valid(
        self,
        position: Point,
        start_heading: float,
        end_heading: float,
    ) -> bool:
        """Check the full shortest rotation sweep with the inflated footprint."""
        return self.motion_is_valid(
            (*position, start_heading), (*position, end_heading)
        )

    def _direct_goal_path_is_valid(self) -> bool:
        """Return whether rotate/translate/rotate can reach the goal directly."""
        if self._current_pose is None or self._goal_pose is None:
            return False

        start_x, start_y, start_heading = self._current_pose
        goal_x, goal_y, goal_heading = self._goal_pose
        delta_x = goal_x - start_x
        delta_y = goal_y - start_y
        distance = math.hypot(delta_x, delta_y)
        if distance <= _GEOMETRY_EPSILON:
            return self._rotation_is_valid(
                (start_x, start_y), start_heading, goal_heading
            )

        travel_heading = math.degrees(math.atan2(delta_y, delta_x))
        if not self._rotation_is_valid(
            (start_x, start_y), start_heading, travel_heading
        ):
            return False
        if not self._rotation_is_valid(
            (goal_x, goal_y), travel_heading, goal_heading
        ):
            return False

        # Closely sample the translated planning footprint. The registered
        # obstacles are already padded by the deployment safety margin, and
        # pose_is_valid applies the configured robot-geometry scale as well.
        sample_spacing = max(
            0.1,
            0.1 * min(self._workspace.car_width, self._workspace.car_height),
        )
        steps = max(1, math.ceil(distance / sample_spacing))
        return all(
            self.pose_is_valid(
                (
                    start_x + delta_x * index / steps,
                    start_y + delta_y * index / steps,
                    travel_heading,
                ),
                robot_geometry_scale=self._settings.robot_geometry_scale,
            )
            for index in range(steps + 1)
        )

    def scan(self) -> Any:
        return self._planner.scan(self._scan_mode)

    def step(self) -> DynamicRVGPlan:
        result = self._planner.step(self._scan_mode)
        configurations = list(result.path)
        status = result.status

        # Prefer the original goal whenever the full scaled footprint has a
        # clear direct corridor. This also handles DRVG's empty-world result,
        # which is NoTemporaryGoal with an empty path. Preserve the scan above
        # for visualization, but skip exploratory temporary goals and detours.
        direct_goal_path = (
            status
            in {
                self._rvg.PyDynamicRVGStatus.GoalPathAvailable,
                self._rvg.PyDynamicRVGStatus.TemporaryGoalPathAvailable,
                self._rvg.PyDynamicRVGStatus.NoTemporaryGoal,
            }
            and self._direct_goal_path_is_valid()
        )
        if direct_goal_path:
            assert self._current_pose is not None
            assert self._goal_pose is not None
            configurations = [
                self._vertex(self._current_pose),
                self._vertex(self._goal_pose),
            ]
            status = self._rvg.PyDynamicRVGStatus.GoalPathAvailable
        else:
            configurations = self._collapse_redundant_full_turns(configurations)

        temporary_goal = None
        if not direct_goal_path and result.temporaryGoal is not None:
            temporary_goal = (
                result.temporaryGoal.getX(),
                result.temporaryGoal.getY(),
            )
        return DynamicRVGPlan(
            status=status,
            configurations=configurations,
            controller_path=self._to_controller_path(configurations),
            temporary_goal=temporary_goal,
            final_segment=(
                status == self._rvg.PyDynamicRVGStatus.GoalPathAvailable
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

    def _collapse_redundant_full_turns(
        self, configurations: Sequence[Any]
    ) -> list[Any]:
        """Coalesce same-position turns only when the full shortest sweep is clear."""
        collapsed = []
        index = 0
        while index < len(configurations):
            end = index
            start = configurations[index]
            while end + 1 < len(configurations):
                candidate = configurations[end + 1]
                if math.hypot(candidate.getX() - start.getX(),
                              candidate.getY() - start.getY()) > 1e-6:
                    break
                end += 1
            run = list(configurations[index:end + 1])
            if len(run) >= 3 and self._rotation_is_valid(
                (start.getX(), start.getY()),
                math.degrees(run[0].getTheta()),
                math.degrees(run[-1].getTheta()),
            ):
                run = [run[0], run[-1]]
            collapsed.extend(run)
            index = end + 1
        return collapsed

    def _to_controller_path(self, configurations: Sequence[Any]) -> list[Point]:
        """Return the actual XY polyline, without synthetic turn waypoints."""
        path = []
        for configuration in configurations:
            xy = (configuration.getX(), configuration.getY())
            if not path or math.dist(path[-1], xy) > 1e-6:
                path.append(xy)
        return path


class PhysicalVisibility:
    """Display visibility from original obstacles, without graph construction.

    Reuse the binding's scan-only API. Reinitializing before each scan clears
    pending observations so this display-only sensor retains no planner state.
    """

    def __init__(self, workspace, obstacles, scan_mode="center"):
        physical_robot = DynamicRVGSession.physical_robot_geometry(workspace)
        self._session = DynamicRVGSession(
            workspace, obstacles,
            DynamicRVGSettings(scan_mode=scan_mode, robot_geometry_scale=1.0),
            robot_geometry=physical_robot,
        )

    def scan(self, pose):
        self._session.initialize(pose, pose)
        observation = self._session.scan()
        if not observation.success:
            return [], []
        return (
            self._session._polygon_points(observation.outerBoundary),
            [self._session._polygon_points(hole) for hole in observation.holes],
        )
