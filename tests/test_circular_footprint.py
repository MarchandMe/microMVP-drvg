"""Axle-centred circular envelopes and full-turn heading semantics."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from micromvp.core.models import WorkspaceConfig
from micromvp.planner.dynamic_rvg import (
    DynamicRVGSession, DynamicRVGSettings, PhysicalVisibility, _point_in_polygon,
)
from micromvp.controller.navigation_controller.pose_path_executor import differential_drive_motions


def workspace():
    return WorkspaceConfig(100,60,4.2,4.8,2.1,4.5,4.2,10,30,[3])


def test_circle_encloses_off_centre_body_through_all_rotations():
    body = DynamicRVGSession.physical_robot_geometry(workspace())
    polygon = DynamicRVGSession.circular_robot_geometry(body)
    radius = math.hypot(4.5,2.1)
    assert radius == pytest.approx(4.965883607)
    assert len(polygon) == 72
    # Polygon sides must be tangent outside the circle, not cut into it.
    for a,b in zip(polygon,polygon[1:]+polygon[:1]):
        distance = abs(a[0]*b[1]-a[1]*b[0]) / math.dist(a,b)
        assert distance == pytest.approx(radius)
    for degrees in np.linspace(0,360,721):
        c,s = math.cos(math.radians(degrees)),math.sin(math.radians(degrees))
        for x,y in body:
            assert _point_in_polygon((x*c-y*s,x*s+y*c),polygon)


def test_circle_collision_checks_do_not_depend_on_heading():
    session = DynamicRVGSession.__new__(DynamicRVGSession)
    session._workspace = workspace()
    session._settings = DynamicRVGSettings(robot_footprint="circle",robot_geometry_scale=1.3)
    session._circle_radius = math.hypot(4.5,2.1)
    session._obstacle_points = [[(20,20),(22,20),(22,40),(20,40)]]
    assert session.planning_circle_radius == pytest.approx(6.455648689)
    for heading in range(0,360,5):
        assert session.pose_is_valid((10,30,heading),robot_geometry_scale=1.3)
        assert not session.pose_is_valid((14,30,heading),robot_geometry_scale=1.3)
    assert session.motion_is_valid((10,30,0),(10,30,180))
    assert not session.motion_is_valid((10,30,0),(40,30,180))


def test_circle_omits_internal_headings_but_preserves_endpoint_heading():
    poses = [(20,20,0),(20,20,90),(30,20,90),(30,20,270),(40,20,0)]
    motions = differential_drive_motions(poses,orientation_invariant=True)
    assert all(m.kind == "drive" for m in motions)
    assert all(not m.reverse for m in motions)
    turn = differential_drive_motions([(20,20,0),(20,20,180)],orientation_invariant=True)
    assert len(turn) == 1 and turn[0].kind == "rotate"
    assert turn[0].end[2] == 180


@pytest.mark.rvg
def test_rvg_rotates_about_explicit_axle_not_body_centroid():
    import rvg
    body = DynamicRVGSession.physical_robot_geometry(workspace())
    robot = rvg.polygon([rvg.vertex(x,y) for x,y in body],rvg.vertex(0,0),False)
    rotated = robot.rotateCopy(math.pi)
    assert robot.getCentroid().getX() == pytest.approx(0)
    assert robot.getCentroid().getY() == pytest.approx(0)
    assert min(v.getX() for v in robot.getVertices()) == pytest.approx(-4.5)
    assert max(v.getX() for v in robot.getVertices()) == pytest.approx(.3)
    assert min(v.getX() for v in rotated.getVertices()) == pytest.approx(-.3)
    assert max(v.getX() for v in rotated.getVertices()) == pytest.approx(4.5)


@pytest.mark.rvg
@pytest.mark.parametrize("start,goal,expected",[(0,180,180),(1,359,2),(0,360,0),(10,190,180)])
def test_search_heading_distance_wraps_360_not_180(start,goal,expected):
    import rvg
    a = rvg.vertex(0,0,0,2*math.pi,math.radians(start))
    b = rvg.vertex(0,0,0,2*math.pi,math.radians(goal))
    assert a.rotationalDist(b) == pytest.approx(math.radians(expected))


@pytest.mark.rvg
@pytest.mark.parametrize("heading",[0,90,180,270,359])
def test_circular_planner_accepts_the_full_heading_range(heading):
    session = DynamicRVGSession(
        workspace(),[],DynamicRVGSettings(
            robot_footprint="circle",robot_geometry_scale=1.3,
            optimal=True,rotational_weight=5,
        ),
    )
    session.initialize((20,30,heading),(80,30,heading))
    plan = session.step()
    assert plan.status.name == "GoalPathAvailable"
    assert plan.configurations[0].getTheta() == pytest.approx(math.radians(heading))
    assert plan.configurations[-1].getTheta() == pytest.approx(math.radians(heading))


@pytest.mark.rvg
def test_circular_goal_keeps_opposite_heading_distinct():
    session = DynamicRVGSession(
        workspace(),[],DynamicRVGSettings(robot_footprint="circle",robot_geometry_scale=1.3,optimal=True),
    )
    session.initialize((20,30,0),(20,30,180))
    plan = session.step()
    assert plan.status.name == "GoalPathAvailable"
    assert len(plan.configurations) == 2
    assert plan.configurations[-1].getTheta() == pytest.approx(math.pi)
    display = PhysicalVisibility(workspace(),[])
    assert display._session._settings.robot_footprint == "polygon"
    assert len(display._session._base_robot_geometry) == 4
