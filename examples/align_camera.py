"""Guide physical camera tilt with a live green alignment signal; no motors."""
import argparse
from pathlib import Path


def positive_float(raw):
    import math
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/car_v4.yaml"))
    parser.add_argument("--tilt-tolerance", type=positive_float, default=3.0,
                        help="Maximum floor tilt in degrees (default: 3)")
    parser.add_argument("--corner-tolerance", type=positive_float, default=3.0,
                        help="Maximum deviation from a 90-degree image corner (default: 3)")
    parser.add_argument("--edge-tolerance", type=positive_float, default=5.0,
                        help="Maximum opposite-edge length difference in percent (default: 5)")
    parser.add_argument("--hold-seconds", type=positive_float, default=1.5,
                        help="Continuous acceptable measurement time before green (default: 1.5)")
    args = parser.parse_args()

    from dataclasses import replace
    from PyQt6.QtWidgets import QApplication
    from micromvp.config import load_config
    from micromvp.env.real_env.observer import ObserverConfig
    from micromvp.env.real_env.camera_alignment import CameraAlignmentObserver, AlignmentMonitor
    from micromvp.gui.camera_alignment import CameraAlignmentWindow

    app = QApplication.instance() or QApplication([])
    config = replace(ObserverConfig.from_config(load_config(args.config)), no_preview=True)
    observer = CameraAlignmentObserver(config)
    monitor = AlignmentMonitor(
        args.tilt_tolerance, args.corner_tolerance, args.edge_tolerance / 100,
        args.hold_seconds,
    )
    window = CameraAlignmentWindow(observer, monitor)
    window.show()
    app.processEvents()
    try:
        if not observer.start():
            print("[Alignment] Camera failed to start. Check device access and calibration.")
            return 1
        print("[Alignment] Camera only: no AP or motor connection. Esc closes.", flush=True)
        return app.exec()
    finally:
        window.close()
        observer.stop()


if __name__ == "__main__":
    raise SystemExit(main())
