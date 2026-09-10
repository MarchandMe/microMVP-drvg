"""Startup must remain visible and refuse navigation until workspace lock."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from unittest.mock import Mock

from PyQt6.QtWidgets import QApplication, QDialog
from micromvp.gui.startup import WorkspaceStartupDialog


def status(**changes):
    result = dict(locked=False, progress=(30, 30), cars=[3],
                  candidate_ready=True, width_cm=100.0, height_cm=60.0)
    result.update(changes)
    return result


def test_full_candidate_window_does_not_start_navigation():
    app = QApplication.instance() or QApplication([])
    env = Mock()
    env.startup_status.return_value = status()
    dialog = WorkspaceStartupDialog(env, 10)
    dialog.update_preview()
    env.observe.assert_not_called()
    assert dialog.result() != QDialog.DialogCode.Accepted
    dialog.detach()
    dialog.close()


def test_timeout_keeps_preview_open_and_requires_retry():
    app = QApplication.instance() or QApplication([])
    env = Mock()
    env.startup_status.return_value = status()
    dialog = WorkspaceStartupDialog(env, 10)
    dialog.show()
    dialog.deadline = 0
    dialog.update_preview()
    assert dialog.isVisible()
    assert dialog.timed_out
    assert not dialog.retry.isHidden()
    env.startup_status.return_value = status(locked=True)
    dialog.update_preview()
    env.observe.assert_not_called()
    dialog.restart_wait()
    dialog.update_preview()
    env.observe.assert_called_once()
    assert dialog.result() == QDialog.DialogCode.Accepted
    dialog.detach()
    dialog.close()


def test_locked_workspace_without_car_does_not_start():
    app = QApplication.instance() or QApplication([])
    env = Mock()
    env.startup_status.return_value = status(locked=True, cars=[])
    dialog = WorkspaceStartupDialog(env, 10)
    dialog.update_preview()
    env.observe.assert_not_called()
    dialog.detach()
    dialog.close()
