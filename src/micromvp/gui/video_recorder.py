"""Silent video encoding off the GUI and robot-control threads."""

from pathlib import Path
import threading
import time

import cv2
import numpy as np
from PyQt6.QtGui import QImage


class OverlayVideoRecorder:
    """Encode the latest rendered frame at 30 fps, repeating it during stalls.

    Only one pending frame is retained, so encoding cannot build an unbounded
    queue or delay robot control. A normal close finalizes the MP4 container.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=False)
        self.error: Exception | None = None
        self._condition = threading.Condition()
        self._frame: np.ndarray | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._encode, name="overlay-video-recorder", daemon=True
        )
        self._thread.start()

    def submit(self, image: QImage) -> None:
        image = image.convertToFormat(QImage.Format.Format_BGR888)
        bits = image.constBits()
        bits.setsize(image.sizeInBytes())
        frame = np.frombuffer(bits, dtype=np.uint8).reshape(
            image.height(), image.bytesPerLine()
        )[:, :image.width() * 3].reshape(image.height(), image.width(), 3).copy()
        with self._condition:
            self._frame = frame
            self._condition.notify()

    def _encode(self) -> None:
        writer = None
        try:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._frame is not None)
                if self._frame is None:
                    return
                height, width = self._frame.shape[:2]
            writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open MP4 encoder for {self.path}")
            print(f"[Recording] Saving {width} x {height} at 30 fps to {self.path}", flush=True)
            deadline = time.monotonic()
            while True:
                with self._condition:
                    while not self._closed and time.monotonic() < deadline:
                        self._condition.wait(timeout=max(0.0, deadline - time.monotonic()))
                    closing = self._closed
                    frame = self._frame
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
                if closing:
                    break
                deadline += 1.0 / 30.0
        except Exception as error:
            self.error = error
        finally:
            if writer is not None:
                writer.release()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify()
        self._thread.join()
        if self.error is None:
            print(f"[Recording] Closed {self.path}", flush=True)
