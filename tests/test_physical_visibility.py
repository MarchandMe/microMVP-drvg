"""The display sees physical geometry independently of planning margins."""
import pytest
from micromvp.core.models import WorkspaceConfig
from micromvp.planner.dynamic_rvg import (
    PhysicalVisibility, DynamicRVGSession, DynamicRVGSettings,
    pad_obstacles, _point_in_polygon,
)

pytestmark = pytest.mark.rvg


def test_visible_area_reaches_original_wall_without_changing_planner(monkeypatch):
    ws = WorkspaceConfig(100,60,4.2,4.8,2.1,4.5,4.2,10,30,[3])
    raw = [[(40,20),(44,20),(44,40),(40,40)]]
    padded = pad_obstacles(raw,1)
    planner = DynamicRVGSession(ws,padded,DynamicRVGSettings(robot_geometry_scale=1.3))
    planner.initialize((20,30,0),(80,30,0))
    plan_scan = planner.scan()
    plan_boundary = planner._polygon_points(plan_scan.outerBoundary)

    display = PhysicalVisibility(ws,raw)
    monkeypatch.setattr(DynamicRVGSession,"step",
                        lambda self: pytest.fail("Display scan must not build a planning graph"))
    outer, holes = display.scan((20,30,0))
    point = (39.5,30)
    assert _point_in_polygon(point,outer)
    assert not any(_point_in_polygon(point,hole) for hole in holes)
    assert not _point_in_polygon(point,plan_boundary)
    assert (40,20) in outer and (40,40) in outer
    assert planner._obstacle_points == padded
    assert planner._settings.robot_geometry_scale == 1.3
    display.scan((70,30,180))
    assert planner._current_pose == (20,30,0)
    assert planner._goal_pose == (80,30,0)
