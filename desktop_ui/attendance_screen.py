# ui/screens/attendance_screen.py (FIXED - Proper Date Filtering)
from datetime import datetime, timezone, date, timedelta
from math import ceil
import os
import requests
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QFrame,
    QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
import pytz

LOCAL_TZ = pytz.timezone("Asia/Kolkata")

def utc_to_local(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", ""))
        dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts


BACKEND_URL = "http://127.0.0.1:5000"


class AttendanceScreen(QWidget):
    def __init__(self, camera_manager=None, parent=None):
        super().__init__(parent)
        self.cm = camera_manager

        self.selected_worker = None
        self.current_month = date.today().month
        self.current_year = date.today().year

        self._build_ui()
        self._load_workers()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(25, 25, 25, 25)
        root.setSpacing(25)

        # LEFT — Worker list panel
        left = QFrame()
        left.setStyleSheet("""
            QFrame { background: white; border-radius: 12px; border:1px solid #dadada; }
        """)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(15)

        title = QLabel("Registered Workers")
        title.setStyleSheet("font-size:20px; font-weight:600; color:#1d1d1f;")
        left_layout.addWidget(title)

        # Worker List
        self.worker_list = QListWidget()
        self.worker_list.itemClicked.connect(self._select_worker)
        left_layout.addWidget(self.worker_list, 1)

        root.addWidget(left, 30)

        # RIGHT — Worker details + attendance
        right = QFrame()
        right.setStyleSheet("""
            QFrame { background:white; border-radius:12px; border:1px solid #dadada; }
        """)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(20)

        # Worker info area
        self.worker_img = QLabel()
        self.worker_img.setFixedSize(180, 220)
        self.worker_img.setStyleSheet("border:1px solid #ccc; border-radius:8px;")
        self.worker_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.worker_img.setText("No Image")

        self.worker_name = QLabel("Select a worker")
        self.worker_name.setStyleSheet("font-size:22px; font-weight:600;")

        self.worker_role = QLabel("")
        self.person_id = QLabel("")

        top = QHBoxLayout()
        top.addWidget(self.worker_img)

        info = QVBoxLayout()
        info.addWidget(self.worker_name)
        info.addWidget(self.worker_role)
        info.addWidget(self.person_id)

        # Delete worker button
        delete_worker = QPushButton("🗑 Remove Worker")
        delete_worker.clicked.connect(self._delete_worker)
        delete_worker.setStyleSheet("""
            QPushButton {
                background:#ffe5e5;
                padding:6px 12px;
                border-radius:8px;
            }
            QPushButton:hover { background:#ffcccc; }
        """)
        info.addWidget(delete_worker)

        top.addLayout(info)
        right_layout.addLayout(top)

        # Month navigation
        nav = QHBoxLayout()
        self.prev_btn = QPushButton("◀")
        self.next_btn = QPushButton("▶")
        self.month_label = QLabel("")
        self.month_label.setStyleSheet("font-size:18px; font-weight:600;")

        self.prev_btn.clicked.connect(self._prev_month)
        self.next_btn.clicked.connect(self._next_month)

        nav.addWidget(self.prev_btn)
        nav.addWidget(self.month_label)
        nav.addWidget(self.next_btn)
        nav.addStretch()

        right_layout.addLayout(nav)

        # Attendance Table
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["No.", "Date"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        right_layout.addWidget(self.table)

        root.addWidget(right, 70)

        self._update_month_label()

    def _update_month_label(self):
        d = date(self.current_year, self.current_month, 1)
        self.month_label.setText(d.strftime("%B %Y"))

    def _prev_month(self):
        self.current_month -= 1
        if self.current_month < 1:
            self.current_month = 12
            self.current_year -= 1
        self._update_month_label()
        self._load_attendance()

    def _next_month(self):
        self.current_month += 1
        if self.current_month > 12:
            self.current_month = 1
            self.current_year += 1
        self._update_month_label()
        self._load_attendance()

    def _load_workers(self):
        self.worker_list.clear()

        try:
            r = requests.get(f"{BACKEND_URL}/api/persons", timeout=5)
            resp = r.json()

            if not resp.get("success"):
                raise RuntimeError(resp.get("error"))

            for p in resp["data"]:
                item = QListWidgetItem(
                    f"{p['person_id']} | {p['name']} | {p.get('role','')}"
                )
                item.setData(Qt.ItemDataRole.UserRole, p)
                self.worker_list.addItem(item)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _select_worker(self, item):
        w = item.data(Qt.ItemDataRole.UserRole)
        self.selected_worker = w

        # Update visual details
        self.worker_name.setText(w.get("name"))
        self.worker_role.setText(f"Role: {w.get('role', 'N/A')}")
        self.person_id.setText(f"ID: {w.get('person_id')}")

        # Try to load worker image from embeddings
        self._load_worker_image(w.get('person_id'))
        
        # Load attendance
        self._load_attendance()

    def _load_worker_image(self, person_id):
        """Try to load worker's face image from backend"""
        try:
            # Get person details with embeddings
            r = requests.get(f"{BACKEND_URL}/api/persons/{person_id}", timeout=5)
            resp = r.json()
            
            if resp.get("success") and resp.get("data"):
                person = resp["data"]
                embeddings = person.get("embeddings", [])
                
                # Find primary or first embedding with image
                for emb in embeddings:
                    if emb.get("image_id"):
                        img_id = emb["image_id"]
                        
                        # Fetch image from backend
                        img_resp = requests.get(
                            f"{BACKEND_URL}/api/logs/image/{img_id}", 
                            timeout=5
                        )
                        
                        if img_resp.status_code == 200:
                            img_data = img_resp.json()
                            if img_data.get("success"):
                                # Decode base64 image
                                import base64
                                base64_str = img_data["image"].split(",")[1]
                                img_bytes = base64.b64decode(base64_str)
                                
                                # Create pixmap
                                pixmap = QPixmap()
                                pixmap.loadFromData(img_bytes)
                                
                                # Scale and display
                                scaled = pixmap.scaled(
                                    180, 220,
                                    Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation
                                )
                                self.worker_img.setPixmap(scaled)
                                return  # Success
                
                # No image found
                self.worker_img.setText("No Image")
        
        except Exception as e:
            print(f"Failed to load worker image: {e}")
            self.worker_img.setText("No Image")

    def _load_attendance(self):
        """
        ✅ FIXED: Load attendance records ONLY for selected month
        """
        if not self.selected_worker:
            return

        person_id = self.selected_worker["person_id"]

        try:
            # ✅ Calculate EXACT date range for the selected month
            # First day of selected month at 00:00:00
            start_date = datetime(self.current_year, self.current_month, 1, 0, 0, 0)
            
            # First day of NEXT month at 00:00:00
            if self.current_month == 12:
                end_date = datetime(self.current_year + 1, 1, 1, 0, 0, 0)
            else:
                end_date = datetime(self.current_year, self.current_month + 1, 1, 0, 0, 0)
            
            # ✅ CRITICAL: Request with days parameter instead of start/end dates
            # This ensures proper filtering on the backend
            # Calculate number of days in the month
            days_in_month = (end_date - start_date).days
            
            params = {
                "days": days_in_month
            }

            print(f"📅 Loading attendance for {person_id}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

            r = requests.get(
                f"{BACKEND_URL}/api/persons/{person_id}/attendance",
                params=params,
                timeout=5
            )

            resp = r.json()
            if not resp.get("success"):
                raise RuntimeError(resp.get("error"))

            # ✅ Filter records to ONLY show records from selected month
            all_records = resp.get("data", [])
            filtered_records = []
            
            for record in all_records:
                date_obj = record.get("date")
                
                # Parse date if it's a string
                if isinstance(date_obj, str):
                    try:
                        date_obj = datetime.fromisoformat(date_obj.replace('Z', '+00:00'))
                    except:
                        continue
                
                # Check if date falls within selected month
                if isinstance(date_obj, datetime):
                    if (date_obj.year == self.current_year and 
                        date_obj.month == self.current_month):
                        filtered_records.append(record)

            print(f"✅ Found {len(filtered_records)} records for {self.current_year}-{self.current_month:02d}")

            self._populate_table(filtered_records)

        except Exception as e:
            print(f"❌ Error loading attendance: {e}")
            QMessageBox.critical(self, "Error Loading Attendance", str(e))
            self.table.setRowCount(0)

    def _populate_table(self, attendance_records):
        """Populate table with attendance records"""
        self.table.setRowCount(0) 
        
        if not attendance_records:
            self.table.setRowCount(1)
            no_data = QTableWidgetItem("No attendance records for this month")
            no_data.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(0, 0, no_data)
            self.table.setSpan(0, 0, 1, 2)
            return

        self.table.setRowCount(len(attendance_records))

        for row, record in enumerate(attendance_records):
            # Column 0: Number
            num_item = QTableWidgetItem(str(row + 1))
            num_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 0, num_item)

            # Column 1: Date
            date_obj = record.get("date")
            if isinstance(date_obj, str):
                # Parse ISO date string
                try:
                    date_obj = datetime.fromisoformat(date_obj.replace('Z', '+00:00'))
                    date_str = date_obj.strftime("%Y-%m-%d")
                except:
                    date_str = date_obj
            else:
                date_str = str(date_obj)
            
            date_item = QTableWidgetItem(date_str)
            date_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 1, date_item)

    def _delete_worker(self):
        if not self.selected_worker:
            return

        person_id = self.selected_worker["person_id"]

        if QMessageBox.question(
            self,
            "Confirm",
            f"Delete person {person_id}?"
        ) != QMessageBox.StandardButton.Yes:
            return

        try:
            r = requests.delete(
                f"{BACKEND_URL}/api/persons/{person_id}",
                timeout=5
            )
            
            if r.status_code == 200:
                QMessageBox.information(self, "Success", "Worker deleted")
                self.selected_worker = None
                self._load_workers()
            else:
                raise RuntimeError("Failed to delete")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))