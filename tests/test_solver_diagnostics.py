"""Native solver diagnostics and workspace-margin regression."""
from dataclasses import asdict
import copy

import matplotlib
matplotlib.use("Agg")
import pytest

from micromvp.core.models import WorkspaceConfig
from micromvp.planner.dynamic_rvg import DynamicRVGSettings
from micromvp.planner.diagnostics import SolverInspector

pytestmark = pytest.mark.rvg


def scene():
    return {
        "workspace": asdict(WorkspaceConfig(100,60,4.2,4.8,2.1,4.5,4.2,10,30,[3])),
        "pose": [10,58,90],
        "obstacles": [],
        "obstacle_padding_cm": 0,
        "settings": asdict(DynamicRVGSettings(robot_geometry_scale=1.3)),
    }


def test_removing_workspace_inset_can_make_same_physical_pose_legal():
    original = scene()
    before = SolverInspector(original)
    first = before.snapshot(90)
    assert not first["native_legal"]
    expanded = copy.deepcopy(original)
    expanded["workspace"]["width"] += 6
    expanded["workspace"]["height"] += 6
    expanded["pose"][0] += 3
    expanded["pose"][1] += 3
    after = SolverInspector(expanded).snapshot(90)
    assert after["native_legal"]
    assert after["robot_bbox"] == pytest.approx(first["robot_bbox"])
    assert after["theta_lower_deg"] == first["theta_lower_deg"]


def test_inspector_exports_native_layer_shapes_and_plot(tmp_path):
    import matplotlib.pyplot as plt
    model = SolverInspector(scene())
    data = model.snapshot(90)
    layer = model.layer(data["layer_index"])
    assert data["robot_bbox"] == model.points(layer.getRobotBBox())
    assert data["shrunken_border"] == model.points(layer.getShrinkedBorder())
    assert data["native_legal"] == layer.legalConfig(model.session._vertex(data["pose"]))
    fig, exported = model.plot(90)
    path = tmp_path / "solver.png"
    fig.savefig(path)
    plt.close(fig)
    assert path.stat().st_size > 1000
    assert exported["native_legal"] is False


def test_inspector_selects_last_full_heading_layer():
    model = SolverInspector(scene())
    data = model.snapshot(359)
    assert data["layer_index"] == 35
    assert data["theta_lower_deg"] == pytest.approx(350)
    assert data["theta_upper_deg"] == pytest.approx(360)
