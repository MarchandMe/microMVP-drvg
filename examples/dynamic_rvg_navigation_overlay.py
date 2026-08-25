"""Run real DynamicRVG navigation in one perspective camera-overlay GUI.

This is an alternate entry point for ``dynamic_rvg_navigation.py``. It keeps
the same planner, controller, command-line arguments, and safety behavior, but
draws scan coverage and planning graphics directly over the live camera image
instead of opening separate camera and workspace windows.
"""
from __future__ import annotations

from typing import Any

from dynamic_rvg_navigation import main as navigation_main


def _create_overlay_window(
    gui_config: dict[str, Any], workspace: Any, environment: Any
) -> Any:
    from micromvp.gui.camera_overlay import CameraOverlayWindow

    return CameraOverlayWindow(gui_config, workspace, environment)


if __name__ == "__main__":
    navigation_main(
        window_factory=_create_overlay_window,
        render_environment=False,
    )
