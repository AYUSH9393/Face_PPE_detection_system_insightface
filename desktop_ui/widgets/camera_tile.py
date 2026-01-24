import cv2
import requests
import threading
import numpy as np

from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap


class MJPEGThread(threading.Thread):
    def __init__(self, url, on_frame):
        super().__init__(daemon=True)
        self.url = url
        self.on_frame = on_frame
        self.running = True

    def run(self):
        try:
            response = requests.get(self.url, stream=True, timeout=10)
            buffer = b""

            for chunk in response.iter_content(chunk_size=1024):
                if not self.running:
                    break

                buffer += chunk
                a = buffer.find(b"\xff\xd8")
                b = buffer.find(b"\xff\xd9")

                if a != -1 and b != -1:
                    jpg = buffer[a:b + 2]
                    buffer = buffer[b + 2:]

                    img = cv2.imdecode(
                        np.frombuffer(jpg, np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    if img is not None:
                        self.on_frame(img)
        except Exception as e:
            print("[CameraTile] Stream error:", e)

    def stop(self):
        self.running = False


class CameraTile(QWidget):
    """
    Single camera tile for CCTV grid.
    """

    def __init__(self, cam_id, cam_name, backend_url):
        super().__init__()

        self.cam_id = cam_id
        self.cam_name = cam_name
        self.backend_url = backend_url

        self.video = QLabel()
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setStyleSheet("background:black;")

        self.title = QLabel(cam_name)
        self.title.setStyleSheet(
            "color:white; background:rgba(0,0,0,160); padding:6px; font-weight:600;"
        )
        self.title.setAlignment(Qt.AlignmentFlag.AlignLeft)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.video, 1)
        layout.addWidget(self.title, 0)

        self._start_stream()

    # -------------------------------------------------
    def _start_stream(self):
        url = f"{self.backend_url}/mjpeg/{self.cam_id}?annotated=false"
        self.thread = MJPEGThread(url, self._update_frame)
        self.thread.start()

    # -------------------------------------------------
    def _update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape

        img = QImage(
            rgb.data,
            w,
            h,
            w * ch,
            QImage.Format.Format_RGB888
        )

        pix = QPixmap.fromImage(img)
        self.video.setPixmap(
            pix.scaled(
                self.video.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.FastTransformation
            )
        )

    # -------------------------------------------------
    def closeEvent(self, event):
        if hasattr(self, "thread"):
            self.thread.stop()
        super().closeEvent(event)
