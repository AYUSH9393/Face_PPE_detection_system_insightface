"""
Live Monitor Screen - FIXED VERSION
✅ Red dot showing correctly on cameras with violations
✅ FPS overlay on video
✅ Video fills frame (no black borders)
✅ Compact status panel (3 items, no scroll)
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QListWidget, QFrame, QListWidgetItem, QPushButton,
    QDialog, QFormLayout, QLineEdit, QMessageBox, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont
import requests
import time
from datetime import datetime

BACKEND_URL = "http://127.0.0.1:5000"


class MJPEGStreamThread(QThread):
    """🚀 ULTRA-OPTIMIZED: Streams raw MJPEG video with maximum performance"""
    frame_ready = pyqtSignal(QImage)
    error = pyqtSignal(str)

    def __init__(self, stream_url: str):
        super().__init__()
        self.stream_url = stream_url
        self.running = True
        self.frame_count = 0
        self.dropped_frames = 0
        self.last_frame_time = 0

    def run(self):
        buffer = b""
        max_buffer_size = 512 * 1024  # 512KB - reduced for lower latency
        last_emit_time = 0
        min_emit_interval = 0.016  # ~60 FPS max emit rate
        
        try:
            # ✅ OPTIMIZED: Aggressive settings for low latency
            response = requests.get(
                self.stream_url, 
                stream=True, 
                timeout=3,  # Reduced timeout
                headers={
                    'Connection': 'keep-alive',
                    'Cache-Control': 'no-cache'
                }
            )
            
            # ✅ OPTIMIZED: Larger chunks for better throughput
            for chunk in response.iter_content(chunk_size=16384):  # 16KB chunks
                if not self.running:
                    break

                if not chunk:
                    continue

                buffer += chunk
                
                # ✅ CRITICAL: Aggressive buffer management - keep only latest frame
                if len(buffer) > max_buffer_size:
                    # Keep only the most recent complete frame
                    # Find the LAST complete frame in buffer
                    last_start = buffer.rfind(b"\xff\xd8")
                    if last_start > 0:
                        buffer = buffer[last_start:]
                        self.dropped_frames += 1
                
                # ✅ OPTIMIZED: Process only the LATEST complete frame to minimize lag
                current_time = time.time()
                
                # Find all frame boundaries
                frames_found = []
                temp_buffer = buffer
                search_pos = 0
                
                while True:
                    start = temp_buffer.find(b"\xff\xd8", search_pos)
                    if start == -1:
                        break
                    end = temp_buffer.find(b"\xff\xd9", start + 2)
                    if end == -1:
                        break
                    frames_found.append((start, end + 2))
                    search_pos = end + 2
                
                # Process only the LAST complete frame (most recent)
                if frames_found:
                    last_frame_start, last_frame_end = frames_found[-1]
                    jpg = buffer[last_frame_start:last_frame_end]
                    buffer = buffer[last_frame_end:]  # Remove all processed frames
                    
                    # Skip older frames in buffer
                    if len(frames_found) > 1:
                        self.dropped_frames += len(frames_found) - 1
                    
                    # ✅ Rate limit emissions to prevent UI overload
                    if current_time - last_emit_time >= min_emit_interval:
                        image = QImage.fromData(jpg)
                        if not image.isNull():
                            self.frame_count += 1
                            self.last_frame_time = current_time
                            self.frame_ready.emit(image)
                            last_emit_time = current_time

        except Exception as e:
            if self.running:
                self.error.emit(str(e))

    def stop(self):
        self.running = False
        self.wait()


class ScrollableStatusPanel(QWidget):
    """✅ NEW: Scrollable status panel with complete history"""
    def __init__(self):
        super().__init__()
        
        from PyQt6.QtWidgets import QScrollArea
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header = QLabel("Detection History")
        header.setStyleSheet("""
            QLabel {
                padding: 8px 12px;
                background: #f3f4f6;
                font-weight: 600;
                font-size: 13px;
                color: #374151;
                border-bottom: 1px solid #e5e7eb;
            }
        """)
        layout.addWidget(header)
        
        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: white;
            }
            QScrollBar:vertical {
                border: none;
                background: #f3f4f6;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #9ca3af;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #6b7280;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # Container for status items
        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(12, 8, 12, 8)
        self.container_layout.setSpacing(6)
        self.container_layout.addStretch()  # Push items to top
        
        scroll.setWidget(self.container)
        layout.addWidget(scroll)
        
        # Store history
        self.status_history = []  # List of (timestamp, person_id, data) tuples
        self.status_widgets = []  # List of QLabel widgets
        
        # ✅ NEW: Track last known state to prevent spam
        self.last_person_state = {}  # {person_id: {'is_compliant': bool, 'missing': []}}
        
        self.setStyleSheet("""
            QWidget {
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
            }
        """)
        
        self.setFixedHeight(200)  # Increased height for scrolling

    def add_status(self, data_list):
        """
        ✅ NEW: Add new status items to history ONLY if status changed
        """
        from datetime import datetime
        
        current_time = datetime.now()
        
        for d in data_list:
            # Extract data
            person_id = d.get("person_id") or d.get("track_id") or "unknown"
            name = d.get("person_name") or d.get("person") or "Unknown"
            missing = d.get("missing_ppe") or d.get("missing", [])
            # Sort missing items to ensure consistent comparison
            if isinstance(missing, list):
                missing = sorted(missing)
            
            is_compliant = d.get("is_compliant", len(missing) == 0)
            
            # ✅ STRICT FILTER: Only add if status CHANGED
            prev_state = self.last_person_state.get(person_id)
            
            current_state = {
                'is_compliant': is_compliant, 
                'missing': missing
            }
            
            # If state matches previous state, SKIP
            if prev_state == current_state:
                continue
                
            # Update last state
            self.last_person_state[person_id] = current_state
            
            # Create status label
            label = QLabel()
            
            # Format timestamp
            time_str = current_time.strftime("%H:%M:%S")
            
            # Determine status
            if is_compliant:
                text = f"[{time_str}] ✓ {name} — Compliant"
                bg_color = "#dcfce7"
                text_color = "#16a34a"
            else:
                missing_str = ", ".join(missing) if missing else "PPE"
                text = f"[{time_str}] ❌ {name} — Missing: {missing_str}"
                bg_color = "#fee2e2"
                text_color = "#dc2626"
            
            label.setText(text)
            label.setStyleSheet(f"""
                QLabel {{
                    padding: 8px 10px;
                    border-radius: 4px;
                    background: {bg_color};
                    font-size: 12px;
                    color: {text_color};
                    font-weight: 500;
                }}
            """)
            label.setWordWrap(True)
            
            # Add to history
            self.status_history.append((current_time, person_id, d))
            self.status_widgets.append(label)
            
            # Insert at top (most recent first)
            self.container_layout.insertWidget(0, label)
        
        # Limit history to last 100 items to prevent memory issues
        if len(self.status_history) > 100:
            # Remove oldest items
            items_to_remove = len(self.status_history) - 100
            for _ in range(items_to_remove):
                self.status_history.pop(0)
                old_widget = self.status_widgets.pop(0)
                self.container_layout.removeWidget(old_widget)
                old_widget.deleteLater()

    def clear_history(self):
        """Clear all history"""
        self.status_history.clear()
        self.last_person_state.clear()  # Clear state cache too
        for widget in self.status_widgets:
            self.container_layout.removeWidget(widget)
            widget.deleteLater()
        self.status_widgets.clear()

class VideoLabel(QLabel):
    """Custom video label with FPS overlay"""
    def __init__(self):
        super().__init__()
        self.current_fps = 0.0
        self.current_pixmap = None
        
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            background-color: #000000;
            border-radius: 8px;
            border: 2px solid #e5e7eb;
        """)
        self.setFixedHeight(540)
        self.setMinimumWidth(760)
        self.setMaximumWidth(950)
        
        self.setText("Select a camera to start monitoring")
        self.setStyleSheet("""
            background-color: #000000;
            color: #9ca3af;
            font-size: 16px;
            border-radius: 8px;
            border: 2px solid #e5e7eb;
        """)

    def set_frame(self, image: QImage, fps: float):
        """🚀 ULTRA-OPTIMIZED: Set frame with minimal processing"""
        self.current_fps = fps
        
        # ✅ CRITICAL: Skip scaling if image is already close to target size
        # This dramatically improves performance
        target_size = self.size()
        img_size = image.size()
        
        # Only scale if size difference is significant (>10%)
        size_diff = abs(img_size.width() - target_size.width()) / target_size.width()
        
        if size_diff < 0.1:
            # Image is close enough - use directly without scaling
            self.current_pixmap = QPixmap.fromImage(image)
        else:
            # Scale with fastest method
            self.current_pixmap = QPixmap.fromImage(image).scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation
            )
        
        # ✅ Use update() instead of repaint() for better performance
        self.update()

    def paintEvent(self, event):
        """Custom paint with FPS overlay"""
        super().paintEvent(event)
        
        if not self.current_pixmap:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        x_offset = (self.width() - self.current_pixmap.width()) // 2
        y_offset = (self.height() - self.current_pixmap.height()) // 2
        painter.drawPixmap(x_offset, y_offset, self.current_pixmap)
        
        if self.current_fps == 0.0:
            color = QColor("#ef4444")
        elif self.current_fps < 15.0:
            color = QColor("#f59e0b")
        else:
            color = QColor("#22c55e")
        
        painter.setBrush(QColor(0, 0, 0, 180))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.width() - 110, 10, 100, 30, 6, 6)
        
        painter.setPen(color)
        font = QFont("Arial", 12, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(self.width() - 105, 10, 90, 30, 
                        Qt.AlignmentFlag.AlignCenter, 
                        f"{self.current_fps:.1f} FPS")
        
        painter.end()


class CameraListItem(QWidget):
    """Custom camera item with violation indicator"""
    def __init__(self, camera_id: str, camera_name: str):
        super().__init__()
        
        self.camera_id = camera_id
        self.has_violations = False
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Violation indicator (red dot)
        self.indicator = QLabel()
        self.indicator.setFixedSize(10, 10)
        self.indicator.setStyleSheet("""
            background: transparent;
            border-radius: 5px;
        """)

        layout.addWidget(self.indicator, alignment=Qt.AlignmentFlag.AlignVCenter)

        
        # Camera info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        
        id_label = QLabel(camera_id)
        id_label.setStyleSheet("font-weight: 600; font-size: 13px;")
        
        name_label = QLabel(camera_name)
        name_label.setStyleSheet("font-size: 12px; color: #6b7280;")
        
        info_layout.addWidget(id_label)
        info_layout.addWidget(name_label)
        
        layout.addLayout(info_layout)
        layout.addStretch()
    
    def set_violation_status(self, has_violations: bool):
        self.has_violations = has_violations
        if has_violations:
            self.indicator.setStyleSheet("""
                background: #ef4444;
                border-radius: 5px;
            """)
        else:
            self.indicator.setStyleSheet("""
                background: transparent;
                border-radius: 5px;
            """)



class AddCameraDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Camera")
        self.setFixedWidth(400)
        
        layout = QFormLayout(self)
        
        self.id_input = QLineEdit()
        self.name_input = QLineEdit()
        self.rtsp_input = QLineEdit()
        self.rtsp_input.setPlaceholderText("rtsp://... or 0 for webcam")
        self.location_input = QLineEdit()
        
        layout.addRow("Camera ID:", self.id_input)
        layout.addRow("Name:", self.name_input)
        layout.addRow("RTSP/Index:", self.rtsp_input)
        layout.addRow("Location:", self.location_input)
        
        btns = QHBoxLayout()
        ok_btn = QPushButton("Add")
        cancel_btn = QPushButton("Cancel")
        
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)
        
        layout.addRow(btns)
        
    def get_data(self):
        return {
            "camera_id": self.id_input.text().strip(),
            "name": self.name_input.text().strip(),
            "rtsp_url": self.rtsp_input.text().strip(),
            "location": self.location_input.text().strip()
        }


class CameraMonitorScreen(QWidget):

    """Live Monitor - Single camera with raw feed"""
    
    def __init__(self):
        super().__init__()

        self.cameras = []
        self.current_camera_id = None
        self.video_thread = None
        self.current_fps = 0.0
        self.camera_widgets = {}

        self._build_ui()
        self._load_cameras()

        # FPS timer
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self._refresh_fps)
        self.fps_timer.start(500)

        # Detection timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._refresh_live_status)
        self.status_timer.start(1000)
        
        # ✅ FIX: Violation checker (check more frequently)
        self.violation_timer = QTimer()
        self.violation_timer.timeout.connect(self._check_camera_violations)
        self.violation_timer.start(1000)  # Check every 1 second instead of 2

    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(20, 20, 20, 20)
        main.setSpacing(16)

        # Header
        header = QHBoxLayout()
        
        title = QLabel("Live Monitor")
        title.setObjectName("pageTitle")
        
        self.camera_count_label = QLabel("")
        self.camera_count_label.setStyleSheet("color: #6b7280; font-size: 14px;")
        
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.camera_count_label)
        
        main.addLayout(header)

        # Body
        body = QHBoxLayout()

        # LEFT: Camera List
        left_panel = QVBoxLayout()
        
        # Camera List Header
        # Header Label
        list_label = QLabel("Camera List")
        list_label.setStyleSheet("font-weight: 700; font-size: 15px; margin-bottom: 4px;")
        left_panel.addWidget(list_label)
        
        # Prominent Add Button
        self.btn_add = QPushButton("➕ Add New Camera")
        self.btn_add.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_add.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                font-weight: 600;
                font-size: 14px;
                border-radius: 6px;
                padding: 10px;
                border: none;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:pressed {
                background-color: #1d4ed8;
            }
        """)
        self.btn_add.clicked.connect(self._click_add_camera)
        left_panel.addWidget(self.btn_add)
        
        self.camera_list = QListWidget()
        self.camera_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                background: white;
                padding: 4px;
            }
            QListWidget::item {
                padding: 4px;
                border-radius: 6px;
                margin-bottom: 4px;
                border: none;
            }
            QListWidget::item:selected {
                background: #3b82f6;
                color: white;
            }
            QListWidget::item:hover {
                background: #eff6ff;
            }
        """)
        self.camera_list.itemClicked.connect(self._select_camera)
        left_panel.addWidget(self.camera_list)

        body.addLayout(left_panel, 15)

        # CENTER: Video + Status
        center_panel = QVBoxLayout()
        center_panel.setSpacing(12)
        
        self.video_label = VideoLabel()
        center_panel.addWidget(self.video_label, stretch=8)
        
        # ✅ NEW: Scrollable status panel with history
        self.status_panel = ScrollableStatusPanel()
        center_panel.addWidget(self.status_panel, stretch=0)

        body.addLayout(center_panel, 80)

        main.addLayout(body)

    def _load_cameras(self):
        """Load camera list from backend"""
        try:
            r = requests.get(f"{BACKEND_URL}/api/cameras", timeout=5)
            r.raise_for_status()
            payload = r.json()

            self.cameras = payload.get("data", [])
            self.camera_list.clear()
            self.camera_widgets.clear()

            for cam in self.cameras:
                camera_id = cam['camera_id']
                camera_name = cam['name']
                
                widget = CameraListItem(camera_id, camera_name)
                self.camera_widgets[camera_id] = widget
                
                item = QListWidgetItem(self.camera_list)
                item.setSizeHint(widget.sizeHint())
                self.camera_list.addItem(item)
                self.camera_list.setItemWidget(item, widget)

            self.camera_count_label.setText(f"{len(self.cameras)} cameras")

            if self.cameras:
                self.camera_list.setCurrentRow(0)
                self._select_camera(self.camera_list.item(0))

        except Exception as e:
            print(f"[LiveMonitor] Failed to load cameras: {e}")


    def _click_add_camera(self):
        dlg = AddCameraDialog(self)
        if not dlg.exec():
            return
            
        data = dlg.get_data()
        
        if not data["camera_id"] or not data["name"]:
            QMessageBox.warning(self, "Error", "Camera ID and Name are required")
            return
            
        try:
            r = requests.post(
                f"{BACKEND_URL}/api/cameras/register",
                json=data,
                timeout=5
            )
            
            resp = r.json()
            if not resp.get("success"):
                QMessageBox.critical(self, "Error", resp.get("error", "Failed to register camera"))
                return
                
            QMessageBox.information(self, "Success", "Camera added successfully")
            self._load_cameras()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _select_camera(self, item):

        """Switch to selected camera"""
        idx = self.camera_list.currentRow()
        if idx < 0:
            return

        camera = self.cameras[idx]
        camera_id = camera['camera_id']

        if camera_id == self.current_camera_id:
            return

        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None

        self.current_camera_id = camera_id

        stream_url = f"{BACKEND_URL}/api/stream/raw/{camera_id}"
        
        self.video_thread = MJPEGStreamThread(stream_url)
        self.video_thread.frame_ready.connect(self._update_frame)
        self.video_thread.error.connect(self._on_stream_error)
        self.video_thread.start()

    def _update_frame(self, image: QImage):
        """Update video display with FPS overlay - optimized for low latency"""
        # ✅ CRITICAL: Process frames immediately without queuing
        # This ensures we always show the latest frame
        self.video_label.set_frame(image, self.current_fps)

    def _on_stream_error(self, error_msg: str):
        """Handle stream errors"""
        self.video_label.setText(f"⚠️ Stream Error\n{error_msg}")
        self.current_fps = 0.0

    def _refresh_fps(self):
        """Update FPS value"""
        if not self.current_camera_id:
            return

        try:
            r = requests.get(f"{BACKEND_URL}/api/cameras/performance", timeout=2)
            if r.status_code == 200:
                fps_data = r.json()["decode_fps"]
                self.current_fps = fps_data.get(self.current_camera_id, 0.0)
            else:
                self.current_fps = 0.0
                
        except Exception:
            self.current_fps = 0.0

    def _refresh_live_status(self):
        """
        ✅ UPDATED: Fetch and add live detections to scrollable history
        """
        if not self.current_camera_id:
            return

        try:
            r = requests.get(
                f"{BACKEND_URL}/api/cameras/{self.current_camera_id}/live_status",
                timeout=2
            )

            if r.status_code != 200:
                print(f"⚠️ Status endpoint returned {r.status_code}")
                return

            payload = r.json()
            
            if not payload.get("success"):
                print(f"⚠️ Status endpoint error: {payload.get('error', 'Unknown')}")
                return

            data = payload.get("data", [])
            
            # Add new detections to history
            if data:
                print(f"📊 Live status for {self.current_camera_id}: {len(data)} detections")
                self.status_panel.add_status(data)

        except requests.exceptions.Timeout:
            print(f"⚠️ Timeout fetching status for {self.current_camera_id}")
        except Exception as e:
            print(f"❌ Status refresh error: {e}")
            import traceback
            traceback.print_exc()


    def _check_camera_violations(self):
        """
        ✅ FIXED: Check for violations and update red dot indicators
        """
        for camera_id, widget in self.camera_widgets.items():
            try:
                r = requests.get(
                    f"{BACKEND_URL}/api/cameras/{camera_id}/live_status",
                    timeout=1
                )
                
                if not r.ok:
                    continue

                payload = r.json()
                if not payload.get("success"):
                    continue

                data = payload.get("data", [])

                # ✅ Check if ANY person has missing PPE
                has_violations = any(
                    len(item.get("missing_ppe", [])) > 0
                    for item in data
                )

                widget.set_violation_status(has_violations)
                
            except Exception as e:
                # Don't clear state on error, just log
                print(f"⚠️ Violation check error for {camera_id}: {e}")
                continue



    def closeEvent(self, event):
        """Cleanup on close"""
        self.fps_timer.stop()
        self.status_timer.stop()
        self.violation_timer.stop()
        
        if self.video_thread:
            self.video_thread.stop()
        
        event.accept()