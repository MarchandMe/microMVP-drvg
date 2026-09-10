"""Discovery rendering reveals only accumulated free space and observed edges."""
import os
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication
from micromvp.core.models import WorkspaceConfig
from micromvp.gui.camera_overlay import CameraOverlayCanvas
from micromvp.gui.discovery import discovered_obstacle_edges

OBSTACLE=[(4,4),(6,4),(6,6),(4,6)]
LEFT=([(0,0),(4,0),(4,10),(0,10)],[])

def test_partial_boundary_does_not_reveal_hidden_edges():
    edges,complete=discovered_obstacle_edges([OBSTACLE],[LEFT])
    assert complete==[]
    assert len(edges)==1
    assert set(edges[0])=={(4,4),(4,6)}

def test_accumulation_can_complete_an_obstacle_boundary():
    all_seen=([(0,0),(10,0),(10,10),(0,10)],[OBSTACLE])
    edges,complete=discovered_obstacle_edges([OBSTACLE],[LEFT,all_seen])
    assert len(edges)==4
    assert complete==[0]

def test_partial_collinear_overlap_reveals_only_observed_portion():
    region=([(0,0),(4,0),(4,5),(0,5)],[])
    edges,complete=discovered_obstacle_edges([OBSTACLE],[region])
    assert complete==[]
    assert len(edges)==1
    assert set(edges[0])=={(4,4),(4,5)}

def test_blackout_covers_unknown_camera_and_overlays_but_not_seen_space(monkeypatch):
    app=QApplication.instance() or QApplication([])
    clock=[0.]
    monkeypatch.setattr("micromvp.gui.camera_overlay.time.monotonic",lambda:clock[0])
    ws=WorkspaceConfig(10,10,1,1,.5,.5,1,1,30,[])
    canvas=CameraOverlayCanvas(ws,lambda x,y,h:(x*20,(10-y)*20),
                               lambda x,y:(x/20,10-y/20),
                               discovery_view=True,setup_seconds=2,blackout_seconds=0,unknown_opacity=1)
    canvas.resize(500,500);canvas.show();app.processEvents()
    frame=np.full((200,200,3),180,dtype=np.uint8)
    canvas.set_camera_frame(frame)
    edges,_=discovered_obstacle_edges([OBSTACLE],[LEFT])
    scan={"uuid":"scanned_area","type":"region","regions":[{"outer":LEFT[0],"holes":[]}],
          "projection_height_cm":0,"clip_to_workspace":True,
          "color":"transparent","fill":"transparent","width":0,
          "discovered_edges":edges,"discovered_obstacles":[]}
    canvas.update_drawings([
        scan,
        {"uuid":"fixed_obstacle_0","type":"path","points":OBSTACLE+[OBSTACLE[0]],"color":"#ff6600","width":3},
        {"uuid":"hidden_path","type":"path","points":[(7,3),(7,7)],"color":"#00ff00","width":5},
    ])
    canvas.begin_discovery_run()
    before=canvas.recording_image()
    assert before.pixelColor(150,100).red()>100
    clock[0]=3
    canvas.update_discovery()
    image=canvas.recording_image()
    assert image.pixelColor(30,100).red()==180
    assert image.pixelColor(150,100).red()==0
    assert image.pixelColor(140,100).green()==0
    assert image.pixelColor(100,100).red()==0
    assert image.pixelColor(120,100).red()==0
    assert image.pixelColor(80,100).red()>220
    # Known obstacle interior is revealed only after every edge was observed.
    scan["discovered_obstacles"]=[OBSTACLE]
    canvas.update_drawings([scan])
    assert canvas.recording_image().pixelColor(100,100).red()>100
    canvas.close()

def test_blackout_waits_for_first_scan():
    app=QApplication.instance() or QApplication([])
    canvas=CameraOverlayCanvas(
        WorkspaceConfig(10,10,1,1,.5,.5,1,1,30,[]),
        lambda x,y,h:(x*20,(10-y)*20),lambda x,y:(x/20,10-y/20),
        discovery_view=True,setup_seconds=0,blackout_seconds=0,unknown_opacity=1)
    canvas.resize(400,400);canvas.show();app.processEvents()
    canvas.set_camera_frame(np.full((200,200,3),180,dtype=np.uint8))
    canvas.begin_discovery_run()
    assert canvas.recording_image().pixelColor(100,100).red()==0
    canvas.close()


@pytest.mark.rvg
def test_native_visibility_reveals_only_exposed_obstacle_edges():
    from micromvp.planner.dynamic_rvg import PhysicalVisibility
    ws=WorkspaceConfig(100,60,4.2,4.8,2.1,4.5,4.2,10,30,[3])
    obstacle=[(40,20),(44,20),(44,40),(40,40)]
    sensor=PhysicalVisibility(ws,[obstacle])
    scans=[sensor.scan((20,30,0))]
    edges,complete=discovered_obstacle_edges([obstacle],scans)
    assert len(edges)==1 and complete==[]
    assert all(abs(point[0]-40)<1e-6 for point in edges[0])
    scans += [sensor.scan(pose) for pose in [(70,30,180),(42,50,270),(42,10,90)]]
    edges,complete=discovered_obstacle_edges([obstacle],scans)
    assert len(edges)==4 and complete==[0]



@pytest.mark.parametrize("birdseye", [False, True])
@pytest.mark.parametrize("projection_height", [0.0, 4.0])
@pytest.mark.parametrize("tilted", [False, True])
def test_saved_overlapping_scans_keep_discovered_pixels_visible(
    birdseye, projection_height, tilted
):
    """Compare recorded pixels against each scan separately, without path unions."""
    import json
    from pathlib import Path
    import cv2

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/discovery_overlapping_scans.json").read_text()
    )
    width, height = fixture["width"], fixture["height"]
    app = QApplication.instance() or QApplication([])
    ws = WorkspaceConfig(width, height, 4.2, 4.8, 2.1, 4.5, 4.2, 10, 30, [3])

    def project(x, y, z):
        # Elevated geometry expands beyond the floor boundary, as with a camera.
        if tilted:
            # This projection exposed an inversion from clipping the scan twice.
            factor = 1 + z / 100
            return (
                8.109572223282196
                + ((x - width / 2) * factor + width / 2) * 14.98589991333467
                + .007065874039408233 * y,
                14.404751466103775
                + ((height / 2 - y) * factor + height / 2) * 12.16568000100554
                + .007065874039408233 * x,
            )
        px, py = 1279 * x / width, 717 * (1 - y / height)
        if z == 0:
            return px, py
        factor = 1 + z / 100
        return 639.5 + (px - 639.5) * factor, 358.5 + (py - 358.5) * factor

    canvas = CameraOverlayCanvas(
        ws, project, lambda x, y: None, birdseye=birdseye,
        discovery_view=True, setup_seconds=0, blackout_seconds=0, unknown_opacity=1,
    )
    canvas.resize(1280, 718)
    canvas.show()
    app.processEvents()
    # Include the whole projected workspace in the synthetic camera frame.
    input_width = 1600 if tilted else 1280
    canvas.set_camera_frame(np.full((718, input_width, 3), 180, dtype=np.uint8))
    canvas.begin_discovery_run()

    def contour(points, z):
        return np.float32([canvas._workspace_to_image(x, y, z) for x, y in points])

    boundary = contour([(0, 0), (width, 0), (width, height), (0, height)], 0)
    try:
        for count in (1, 2):
            scans = fixture["regions"][:count]
            canvas.update_drawings([{
                "uuid": "scanned_area", "type": "region", "regions": scans,
                "projection_height_cm": projection_height, "clip_to_workspace": True,
                "color": "transparent", "fill": "transparent", "width": 0,
            }])
            for size in [(1280, 718), (800, 600), (600, 800)]:
                canvas.resize(*size)
                app.processEvents()
                image = canvas.recording_image()
                rings = [
                    [contour(scan["outer"], projection_height)]
                    + [contour(hole, projection_height) for hole in scan["holes"]]
                    for scan in scans
                ]
                mismatches = []
                checked = 0
                for y in range(20, image.height() - 20, 35):
                    for x in range(20, image.width() - 20, 35):
                        point = (float(x), float(y))
                        border_distance = cv2.pointPolygonTest(boundary, point, True)
                        distances = [
                            [cv2.pointPolygonTest(ring, point, True) for ring in region]
                            for region in rings
                        ]
                        # The workspace outline sits above the mask and has a
                        # scene-pixel pen; exclude it and antialiased scan edges.
                        if abs(border_distance) < max(3, 4 / canvas._camera_scale):
                            continue
                        if min(abs(d) for region in distances for d in region) < 3:
                            continue
                        expected_visible = border_distance > 0 and any(
                            region[0] > 0 and all(d < 0 for d in region[1:])
                            for region in distances
                        )
                        actual_visible = image.pixelColor(x, y).red() > 100
                        checked += 1
                        if actual_visible != expected_visible:
                            mismatches.append((x, y, expected_visible))
                assert checked > 300
                assert not mismatches, (count, size, mismatches[:5])
    finally:
        canvas.close()



@pytest.mark.parametrize("height", [0.0, 4.0])
def test_overlapping_scans_preserve_holes_and_reveal_complete_obstacles(height):
    app = QApplication.instance() or QApplication([])
    canvas = CameraOverlayCanvas(
        WorkspaceConfig(10, 10, 1, 1, .5, .5, 1, 1, 30, []),
        lambda x, y, h: (x * 20, (10 - y) * 20),
        lambda x, y: (x / 20, 10 - y / 20),
        discovery_view=True, setup_seconds=0, blackout_seconds=0, unknown_opacity=1,
    )
    canvas.resize(500, 500)
    canvas.show()
    app.processEvents()
    canvas.set_camera_frame(np.full((200, 200, 3), 180, dtype=np.uint8))
    # Opposite input winding must not cancel overlapping scans. The unseen
    # obstacle interior remains a hole even when it belongs to several scans.
    outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
    scan = {
        "uuid": "scanned_area", "type": "region", "clip_to_workspace": True,
        "projection_height_cm": height, "color": "transparent",
        "fill": "transparent", "width": 0,
        "regions": [
            {"outer": outer, "holes": [OBSTACLE]},
            {"outer": outer[::-1], "holes": [OBSTACLE[::-1]]},
        ],
    }
    canvas.update_drawings([scan])
    canvas.begin_discovery_run()
    try:
        image = canvas.recording_image()
        assert image.pixelColor(30, 100).red() == 180
        assert image.pixelColor(150, 100).red() == 180
        assert image.pixelColor(100, 100).red() == 0
        # A later scan can reveal part of an earlier hole without revealing all.
        scan["regions"].append({
            "outer": [(4, 4), (5, 4), (5, 6), (4, 6)], "holes": [],
        })
        canvas.update_drawings([scan])
        image = canvas.recording_image()
        assert image.pixelColor(90, 100).red() == 180
        assert image.pixelColor(110, 100).red() == 0
        scan["discovered_obstacles"] = [OBSTACLE]
        canvas.update_drawings([scan])
        assert canvas.recording_image().pixelColor(110, 100).red() > 100
    finally:
        canvas.close()



def test_setup_then_full_blackout_then_translucent_unknown_space(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("micromvp.gui.camera_overlay.time.monotonic", lambda: clock[0])
    app = QApplication.instance() or QApplication([])
    canvas = CameraOverlayCanvas(
        WorkspaceConfig(10, 10, 1, 1, .5, .5, 1, 1, 30, []),
        lambda x, y, h: (x * 20, (10 - y) * 20),
        lambda x, y: (x / 20, 10 - y / 20), discovery_view=True,
    )
    canvas.resize(500, 500)
    canvas.show()
    app.processEvents()
    canvas.set_camera_frame(np.full((200, 200, 3), 180, dtype=np.uint8))
    scan = {
        "uuid": "scanned_area", "type": "region",
        "regions": [{"outer": LEFT[0], "holes": []}],
        "color": "transparent", "fill": "transparent", "width": 0,
    }
    canvas.update_drawings([scan])
    canvas.begin_discovery_run()
    try:
        assert canvas.recording_image().pixelColor(30, 100).red() == 180
        assert canvas.recording_image().pixelColor(150, 100).red() == 180
        for now in (2.0, 2.4):
            clock[0] = now
            canvas.update_discovery()
            image = canvas.recording_image()
            assert image.pixelColor(30, 100).red() == 0
            assert image.pixelColor(150, 100).red() == 0
        clock[0] = 2.6
        canvas.update_discovery()
        image = canvas.recording_image()
        assert image.pixelColor(30, 100).red() == 180
        assert image.pixelColor(150, 100).red() == pytest.approx(27, abs=1)
        # Restarting a goal repeats the transition; a late scan stays black.
        canvas.begin_discovery_run()
        canvas.update_drawings([])
        assert canvas.recording_image().pixelColor(30, 100).red() == 180
        clock[0] = 5.0
        canvas.update_discovery()
        clock[0] = 6.0
        canvas.update_discovery()
        assert canvas.recording_image().pixelColor(30, 100).red() == 0
        canvas.update_drawings([scan])
        assert canvas.recording_image().pixelColor(30, 100).red() == 180
    finally:
        canvas.close()
