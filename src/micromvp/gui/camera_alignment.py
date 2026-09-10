"""Live visual camera alignment guidance."""
import math
import threading
import time

from PyQt6.QtCore import QPointF, Qt, QTimer
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QLabel, QMainWindow, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)


class CameraAlignmentWindow(QMainWindow):
    def __init__(self, observer, monitor):
        super().__init__()
        self.observer = observer
        self.monitor = monitor
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._frame = None
        self._received_at = 0.0
        self._last_state = None
        self.setWindowTitle("MicroMVP — Camera alignment (camera only)")
        self.resize(1120, 820)
        self.setStyleSheet("background: #162027; color: #edf4f6;")
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.preview = QLabel("Opening camera…")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(320, 180)
        self.preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.preview, 1)
        self.status = QLabel("WAITING FOR CAMERA")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setStyleSheet("font-size: 28px; font-weight: bold; padding: 12px;")
        layout.addWidget(self.status)
        self.instructions = QLabel("Keep flat car/obstacle markers in view.")
        self.instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.instructions.setWordWrap(True)
        self.instructions.setStyleSheet("font-size: 20px; padding: 5px;")
        layout.addWidget(self.instructions)
        self.metrics = QLabel("")
        self.metrics.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.metrics.setStyleSheet("font-size: 15px; padding: 5px;")
        layout.addWidget(self.metrics)
        note = QLabel(
            "Directions refer to the live image. Make a small tilt adjustment, then pause.\n"
            "Height and centering are manual. Use flat markers and matching camera calibration."
        )
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note.setStyleSheet("color: #b7c8d1; font-size: 13px;")
        layout.addWidget(note)
        close = QPushButton("Done / close (Esc)")
        close.setStyleSheet("background: #334753; padding: 10px; font-size: 16px;")
        close.clicked.connect(self.close)
        layout.addWidget(close)
        observer.set_frame_callback(self.receive_frame)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_view)
        self.timer.start(33)

    def receive_frame(self, frame):
        with self._frame_lock:
            self._latest_frame = frame.copy()
            self._received_at = time.monotonic()

    def update_view(self):
        with self._frame_lock:
            if self._latest_frame is not None:
                self._frame = self._latest_frame
                self._latest_frame = None
            fresh_frame = time.monotonic() - self._received_at <= 0.5
        measurement, timestamp = self.observer.alignment_measurement()
        if not fresh_frame:
            measurement = None
        state = self.monitor.update(measurement, timestamp, time.time())
        color = QColor("#16c56b" if state == "aligned" else "#ffbd42")
        if state == "waiting":
            count, required, motion = self.observer.get_workspace_corner_progress()
            title = "PAUSE — MEASURING" if count else "WAITING FOR MARKERS"
            if not fresh_frame:
                title = "WAITING FOR CAMERA"
            guidance = (
                "Camera moved. Hold it still for a fresh measurement."
                if motion else f"Keep markers visible and hold still. Samples: {count}/{required}."
            )
        elif state == "aligned":
            title = "STOP — ALIGNED"
            guidance = "Secure the mount here. The camera is close to perpendicular to the floor."
        elif state == "hold":
            title = "LOOKS GOOD — HOLD STILL"
            guidance = f"Green in {max(0.0, self.monitor.hold_seconds - self.monitor.elapsed):.1f} s."
        else:
            title = "ADJUST CAMERA TILT"
            guidance = self.monitor.guidance(measurement)
        self.status.setText(title)
        self.status.setStyleSheet(
            f"font-size: 28px; font-weight: bold; padding: 12px; "
            f"background: {'#087943' if state == 'aligned' else '#293841'}; "
            f"color: {'white' if state == 'aligned' else color.name()};"
        )
        self.instructions.setText(guidance)
        if measurement is not None:
            self.metrics.setText(
                f"Tilt {measurement.tilt_deg:.1f}° / {self.monitor.tilt_tolerance:g}°    |    "
                f"Corner error {measurement.corner_error_deg:.1f}° / {self.monitor.corner_tolerance:g}°    |    "
                f"Opposite sides differ {measurement.opposite_edge_error:.1%} / {self.monitor.edge_tolerance:.0%}"
            )
        else:
            self.metrics.setText("Waiting for a fresh, steady floor estimate")
        if state != self._last_state:
            print(f"[Alignment] {title}: {guidance}", flush=True)
            self._last_state = state
        if self._frame is not None:
            self._paint_frame(measurement, state, color)

    def _paint_frame(self, measurement, state, color):
        frame = self._frame
        height, width = frame.shape[:2]
        image = QImage(frame.data, width, height, int(frame.strides[0]),
                       QImage.Format.Format_BGR888).copy()
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if state == "aligned":
            painter.fillRect(image.rect(), QColor(0, 220, 100, 65))
        painter.setPen(QPen(color, 12))
        painter.drawRect(6, 6, width - 12, height - 12)
        if measurement is not None:
            painter.setPen(QPen(color, 4))
            painter.drawPolygon(QPolygonF([
                QPointF(float(x), float(y)) for x, y in measurement.polygon
            ]))
        center = QPointF(width / 2, height / 2)
        painter.setPen(QPen(QColor("white"), 2))
        painter.drawLine(center + QPointF(-15, 0), center + QPointF(15, 0))
        painter.drawLine(center + QPointF(0, -15), center + QPointF(0, 15))
        if state == "adjust" and measurement is not None:
            dx, dy = measurement.horizontal_deg, measurement.vertical_deg
            norm = math.hypot(dx, dy)
            if norm > self.monitor.tilt_tolerance:
                direction = QPointF(dx / norm, dy / norm)
                end = center + direction * min(width, height) * 0.22
                painter.setPen(QPen(color, 8))
                painter.drawLine(center, end)
                side = QPointF(-direction.y(), direction.x())
                painter.setBrush(color)
                painter.drawPolygon(QPolygonF([
                    end, end - direction * 25 + side * 12,
                    end - direction * 25 - side * 12,
                ]))
        painter.end()
        self.preview.setPixmap(QPixmap.fromImage(image).scaled(
            self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.timer.stop()
        self.observer.set_frame_callback(None)
        self.observer.stop()
        super().closeEvent(event)
