"""Visible, retryable hardware startup before the navigation window exists."""
import threading
import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QDialog, QLabel, QPushButton, QVBoxLayout


class WorkspaceStartupDialog(QDialog):
    def __init__(self, environment, timeout: float):
        super().__init__()
        self.environment = environment
        self.timeout = max(1.0, timeout)
        self.deadline = time.monotonic() + self.timeout
        self.timed_out = False
        self._lock = threading.Lock()
        self._frame = None
        self.setWindowTitle("MicroMVP — camera setup")
        self.resize(960, 650)
        layout = QVBoxLayout(self)
        self.preview = QLabel("Opening camera…")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(320, 180)
        layout.addWidget(self.preview, 1)
        self.status = QLabel("Keep the camera and markers still. Robots remain stopped.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.retry = QPushButton("Retry workspace lock")
        self.retry.hide()
        self.retry.clicked.connect(self.restart_wait)
        layout.addWidget(self.retry)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        layout.addWidget(cancel)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_preview)
        self.environment.set_frame_callback(self.receive_frame)

    def receive_frame(self, frame):
        with self._lock:
            self._frame = frame.copy()

    def restart_wait(self):
        self.deadline = time.monotonic() + self.timeout
        self.timed_out = False
        self.retry.hide()

    def update_preview(self):
        with self._lock:
            frame = self._frame
            self._frame = None
        if frame is not None:
            height, width = frame.shape[:2]
            image = QImage(
                frame.data, width, height, int(frame.strides[0]),
                QImage.Format.Format_BGR888,
            ).copy()
            self.preview.setPixmap(QPixmap.fromImage(image).scaled(
                self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
        status = self.environment.startup_status()
        if status["locked"] and status["cars"] and not self.timed_out:
            self.environment.observe()
            self.accept()
            return
        count, required = status["progress"]
        corner_count, corner_required, motion = status.get("corner_progress", (0, 0, False))
        if motion:
            hint = "Image motion detected; restarting marker averaging."
        elif corner_required and corner_count < corner_required:
            hint = "Collecting steady marker corners. Keep the camera and markers still."
        elif not status["candidate_ready"]:
            hint = "No usable workspace estimate. Check marker visibility, focus and calibration."
        elif not status["cars"]:
            hint = "No car detected. Put a car marker in view."
        else:
            hint = (
                f"Workspace estimate: {status['width_cm']:.1f} × "
                f"{status['height_cm']:.1f} cm. "
                "Waiting for a stable view; keep the camera and markers still."
            )
        if time.monotonic() >= self.deadline:
            self.timed_out = True
            self.retry.show()
        prefix = (
            "Workspace has not locked. Adjust the setup, then click Retry."
            if self.timed_out else
            f"Waiting for workspace ({max(0, self.deadline - time.monotonic()):.0f}s remaining)."
        )
        self.status.setText(
            f"{prefix}\nSamples: {count}/{required}; cars: {status['cars']}.\n"
            f"{hint}\nRobots remain stopped."
        )

    def detach(self):
        self.timer.stop()
        self.environment.set_frame_callback(None)


def wait_for_camera_workspace(environment, timeout: float) -> bool:
    app = QApplication.instance() or QApplication([])
    previous_quit = app.quitOnLastWindowClosed()
    app.setQuitOnLastWindowClosed(False)
    dialog = WorkspaceStartupDialog(environment, timeout)
    dialog.show()
    app.processEvents()
    try:
        if not environment.start(wait_for_ready=False):
            return False
        environment.stop_all()
        dialog.restart_wait()
        dialog.timer.start(33)
        return dialog.exec() == QDialog.DialogCode.Accepted
    except Exception:
        environment.stop_all()
        environment.close()
        raise
    finally:
        dialog.detach()
        dialog.close()
        app.setQuitOnLastWindowClosed(previous_quit)
