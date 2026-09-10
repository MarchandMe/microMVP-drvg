"""Capture and plot the geometry DRVG uses for an angular layer; no motor connection."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import time


def capture_scene(args):
    from micromvp.config import load_config
    from micromvp.core.models import WorkspaceConfig
    from micromvp.env.real_env.observer import ArucoObserver, ObserverConfig
    from micromvp.planner.dynamic_rvg import DynamicRVGSettings

    cfg = load_config(args.config)
    observer = ArucoObserver(replace(ObserverConfig.from_config(cfg), no_preview=True))
    try:
        if not observer.start():
            raise RuntimeError("Camera failed to start")
        deadline = time.monotonic() + args.timeout
        while not observer.is_workspace_ready():
            if time.monotonic() >= deadline:
                raise RuntimeError("Workspace did not lock; use align_camera.py to inspect the view")
            time.sleep(.05)
        observer.reset_obstacles()
        time.sleep(max(1.0, cfg.require("obstacle.detection_interval_sec", float)) + .1)
        estimate = observer.get_workspace_estimate()
        observations = observer.get_observations()
        if args.robot_id not in observations:
            raise RuntimeError(f"Car {args.robot_id} is not visible")
        car = observations[args.robot_id]
        axle = cfg.require_pair("car.axle_offset_cm")
        workspace = WorkspaceConfig(
            estimate.width_cm, estimate.height_cm,
            cfg.require("car.body_width_cm", float), cfg.require("car.body_height_cm", float),
            axle[0], axle[1], cfg.require("car.wheel_base_cm", float),
            cfg.require("car.max_wheel_speed_cm_s", float),
            cfg.require("runtime.frequency", float), list(observations),
            marker_to_axle_offset=cfg.require_pair("car.marker_to_axle_offset_cm"),
        )
        settings = DynamicRVGSettings(
            resolution=cfg.require("planner.rvg.resolution", int),
            euclidean_weight=cfg.require("planner.rvg.euclidean_weight", float),
            rotational_weight=cfg.require("planner.rvg.rotational_weight", float),
            optimal=cfg.require("planner.rvg.optimal", bool),
            robot_geometry_scale=cfg.require("navigation.robot_geometry_scale", float),
            robot_footprint=cfg.require("navigation.robot_footprint", str),
            scan_mode=args.scan_mode,
        )
        return {
            "captured_at": datetime.now().isoformat(),
            "config": str(args.config.resolve()),
            "workspace": asdict(workspace),
            "pose": [car.x_cm, car.y_cm, car.yaw_deg],
            "obstacles": observer.get_obstacles(),
            "obstacle_padding_cm": cfg.require("navigation.obstacle_padding_cm", float),
            "settings": asdict(settings),
        }
    finally:
        observer.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("config/car_v4.yaml"))
    parser.add_argument("--robot-id",type=int,default=3)
    parser.add_argument("--timeout",type=float,default=30)
    parser.add_argument("--scan-mode",choices=("center","footprint"),default="center")
    parser.add_argument("--scene",type=Path,help="Replay a saved scene.json without opening the camera")
    parser.add_argument("--heading",type=float,help="Inspect another heading at the saved axle position")
    parser.add_argument("--output-dir",type=Path,default=Path("diagnostics/drvg"))
    parser.add_argument("--show",action="store_true",help="Open a zoomable Matplotlib window after export")
    args=parser.parse_args()
    scene=json.loads(args.scene.read_text()) if args.scene else capture_scene(args)

    import matplotlib
    matplotlib.use("QtAgg" if args.show else "Agg")
    from micromvp.planner.diagnostics import SolverInspector
    inspector=SolverInspector(scene)
    fig,geometry=inspector.plot(args.heading)
    out=args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out.mkdir(parents=True)
    (out/"scene.json").write_text(json.dumps(scene,indent=2))
    (out/"geometry.json").write_text(json.dumps(geometry,indent=2))
    fig.savefig(out/"solver.png",dpi=160)
    fig.savefig(out/"solver.svg")
    fig.savefig(out/"preview.jpg",dpi=100)
    print(f"[Inspector] Pose: {geometry['pose']}",flush=True)
    print(f"[Inspector] Layer: {geometry['theta_lower_deg']:.1f}–{geometry['theta_upper_deg']:.1f} deg",flush=True)
    print(f"[Inspector] Native legalConfig: {geometry['native_legal']}",flush=True)
    print(f"[Inspector] Cause: {geometry['cause']}",flush=True)
    print(f"[Inspector] Unscaled pose clear: {geometry['unscaled_pose_clear']}; scaled pose clear: {geometry['scaled_pose_clear']}",flush=True)
    print(f"[Inspector] Exported: {out.resolve()}",flush=True)
    if args.show:
        import matplotlib.pyplot as plt
        fig.canvas.manager.set_window_title("DRVG Solver Inspector — captured snapshot")
        plt.show()


if __name__=="__main__":
    main()
