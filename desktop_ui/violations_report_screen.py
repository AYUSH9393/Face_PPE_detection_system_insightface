# # ui/screens/violations_report_screen.py - FIXED VERSION
# # Shows track IDs for unknown persons instead of just "UNKNOWN"

# import csv, base64, os, requests
# from math import ceil
# from datetime import datetime, timezone
# import pytz
# from PyQt6.QtWidgets import (
#     QWidget, QVBoxLayout, QLabel, QHBoxLayout, QLineEdit, QPushButton, 
#     QDialog, QTableWidget, QTableWidgetItem, QHeaderView, QFrame, 
#     QFileDialog, QMessageBox
# )
# from PyQt6.QtCore import Qt
# from PyQt6.QtGui import QPixmap

# BACKEND_URL = "http://127.0.0.1:5000"
# PAGE_SIZE = 40
# LOCAL_TZ = pytz.timezone("Asia/Kolkata")

# def utc_to_local(ts: str) -> str:
#     try:
#         dt = datetime.fromisoformat(ts.replace("Z", ""))
#         dt = dt.replace(tzinfo=timezone.utc)
#         return dt.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")
#     except Exception:
#         return ts


# class ViolationsReportScreen(QWidget):
#     def __init__(self, camera_manager=None, parent=None):
#         super().__init__(parent)
#         self.cm = camera_manager

#         self.current_page = 1
#         self.total_rows = 0
#         self.search_text = ""
#         self._pagination_buttons = []

#         self._build_ui()
#         self._load_data()

#     def _build_ui(self):
#         root = QVBoxLayout(self)
#         root.setContentsMargins(30, 30, 30, 30)
#         root.setSpacing(20)

#         # PAGE TITLE
#         title = QLabel("Safety Violation Reports")
#         title.setStyleSheet("""
#             font-size: 26px;
#             font-weight: 700;
#             color: #1d1d1f;
#         """)
#         root.addWidget(title)

#         # SEARCH BAR + BUTTONS
#         top_bar = QHBoxLayout()
#         top_bar.setSpacing(10)

#         self.search_input = QLineEdit()
#         self.search_input.setPlaceholderText("Search by worker name, ID, track ID, or missing PPE...")
#         self.search_input.textChanged.connect(self._on_search)
#         self.search_input.setStyleSheet("""
#             QLineEdit {
#                 padding: 8px;
#                 border-radius: 8px;
#                 border: 1px solid #d1d1d6;
#                 background: white;
#                 min-width: 300px;
#             }
#         """)

#         btn_refresh = QPushButton("🔄 Refresh")
#         btn_refresh.clicked.connect(self._refresh_data)
        
#         btn_export = QPushButton("⬇ Export CSV")
#         btn_export.clicked.connect(self._export_csv)

#         for b in [btn_refresh, btn_export]:
#             b.setStyleSheet("""
#                 QPushButton {
#                     background: #e5e5ea;
#                     padding: 8px 16px;
#                     border-radius: 8px;
#                     font-size: 14px;
#                 }
#                 QPushButton:hover { background: #d1d1d6; }
#             """)

#         top_bar.addWidget(self.search_input)
#         top_bar.addWidget(btn_refresh)
#         top_bar.addWidget(btn_export)
#         top_bar.addStretch()

#         root.addLayout(top_bar)

#         # TABLE
#         card = QFrame()
#         card.setStyleSheet("""
#             QFrame {
#                 background: white;
#                 border-radius: 12px;
#                 border: 1px solid #d1d1d6;
#             }
#         """)
#         card_layout = QVBoxLayout(card)
#         card_layout.setContentsMargins(20, 20, 20, 20)

#         self.table = QTableWidget(0, 6)
#         self.table.setHorizontalHeaderLabels([
#             "",
#             "Worker (ID / Track ID)",  # ✅ Updated header
#             "Role",
#             "Missing PPE",
#             "Time",
#             "Camera",
#         ])

#         header = self.table.horizontalHeader()
#         header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
#         self.table.setColumnWidth(0, 40)
#         self.table.setColumnWidth(1, 220)
#         self.table.setColumnWidth(2, 100)
#         self.table.setColumnWidth(3, 280)
#         self.table.setColumnWidth(4, 150)
#         self.table.setColumnWidth(5, 120)

#         card_layout.addWidget(self.table)
#         root.addWidget(card)

#         # PAGINATION
#         pagination_container = QFrame()
#         pagination_container.setStyleSheet("background: transparent;")
#         pagination_layout = QHBoxLayout(pagination_container)
#         pagination_layout.setContentsMargins(0, 0, 0, 0)
        
#         self.pagination = QHBoxLayout()
#         self.pagination.setSpacing(6)
#         pagination_layout.addStretch()
#         pagination_layout.addLayout(self.pagination)
#         pagination_layout.addStretch()
        
#         root.addWidget(pagination_container)
        
#         # STATUS BAR
#         self.status_label = QLabel("")
#         self.status_label.setStyleSheet("color: #666; font-size: 12px;")
#         root.addWidget(self.status_label)

#     def _load_data(self, force=False):
#         try:
#             params = {
#                 "page": self.current_page,
#                 "page_size": PAGE_SIZE
#             }

#             if self.search_text:
#                 params["search"] = self.search_text.strip()

#             r = requests.get(
#                 f"{BACKEND_URL}/api/ppe/violations",
#                 params=params,
#                 timeout=5
#             )

#             resp = r.json()

#             if not resp.get("success"):
#                 raise RuntimeError(resp.get("error"))

#             rows = resp.get("data", [])
#             self.total_rows = resp.get("total", len(rows))

#             self._populate_table(rows)
#             self._update_status()
#             self._update_pagination()

#         except Exception as e:
#             self.status_label.setText(f"Error loading data: {e}")
    
#     def _refresh_data(self):
#         self.current_page = 1
#         self._load_data(force=True)

#     def _update_status(self):
#         total_pages = ceil(self.total_rows / PAGE_SIZE) if self.total_rows > 0 else 1
#         self.status_label.setText(
#             f"Showing page {self.current_page} of {total_pages} "
#             f"({self.total_rows} total violations)"
#         )

#     def _populate_table(self, logs):
#         """
#         ✅ FIXED: Now properly displays track IDs for unknown persons
#         """
#         self.table.setRowCount(0)
#         self.table.setRowCount(len(logs))

#         for row, log in enumerate(logs):

#             # Column 0 – checkbox (disabled)
#             chk = QTableWidgetItem()
#             chk.setFlags(Qt.ItemFlag.ItemIsEnabled)
#             self.table.setItem(row, 0, chk)

#             # ✅ Column 1 – Person (with track ID support)
#             person_id = log.get("person_id")
#             track_id = log.get("track_id")
#             is_unknown = log.get("is_unknown_person", False)

#             if is_unknown and track_id:
#                 # Show track ID for unknown persons
#                 display = f"Unknown ({track_id})"
#                 display_style = "color: #FF8C00; font-weight: bold;"  # Orange
#             elif person_id and person_id != "UNKNOWN":
#                 # Known person
#                 display = person_id
#                 display_style = "color: #000;"
#             else:
#                 # Fallback for old logs without track_id
#                 display = "Unknown Visitor"
#                 display_style = "color: #999;"

#             item = QTableWidgetItem(display)
#             self.table.setItem(row, 1, item)

#             # Column 2 – Role
#             role = log.get("role", "visitor").capitalize()
#             self.table.setItem(row, 2, QTableWidgetItem(role))

#             # Column 3 – Missing PPE
#             missing = log.get("missing_ppe", [])
#             text = ", ".join(missing) if missing else "—"
#             self.table.setItem(row, 3, QTableWidgetItem(text))

#             # Column 4 – Time
#             ts = log.get("timestamp")
#             time_text = utc_to_local(ts) if ts else "—"
#             self.table.setItem(row, 4, QTableWidgetItem(time_text))

#             # Column 5 – Camera
#             cam = log.get("camera_id", "—")
#             self.table.setItem(row, 5, QTableWidgetItem(cam))

#     def _on_search(self):
#         self.search_text = self.search_input.text()
#         self.current_page = 1
#         self._load_data(force=True)

#     def _update_pagination(self):
#         """Update pagination buttons"""
#         for button in self._pagination_buttons:
#             if button.parent():
#                 button.setParent(None)
#             button.deleteLater()
#         self._pagination_buttons.clear()
        
#         while self.pagination.count():
#             item = self.pagination.takeAt(0)
#             if item.widget():
#                 item.widget().deleteLater()

#         total_pages = ceil(self.total_rows / PAGE_SIZE) if self.total_rows > 0 else 1

#         if total_pages <= 1:
#             return

#         # Previous button
#         btn_prev = QPushButton("◀ Prev")
#         btn_prev.setEnabled(self.current_page > 1)
#         if self.current_page > 1:
#             btn_prev.clicked.connect(lambda: self._go_page(self.current_page - 1))
#         self._style_pagination_button(btn_prev)
#         self.pagination.addWidget(btn_prev)
#         self._pagination_buttons.append(btn_prev)

#         # Page numbers
#         pages_to_show = []
#         if total_pages <= 7:
#             pages_to_show = list(range(1, total_pages + 1))
#         else:
#             pages_to_show.append(1)
            
#             if self.current_page > 3:
#                 pages_to_show.append("...")
            
#             start = max(2, self.current_page - 1)
#             end = min(total_pages - 1, self.current_page + 1)
            
#             for p in range(start, end + 1):
#                 if p not in pages_to_show:
#                     pages_to_show.append(p)
            
#             if self.current_page < total_pages - 2:
#                 if end < total_pages - 2:
#                     pages_to_show.append("...")
            
#             if total_pages not in pages_to_show:
#                 pages_to_show.append(total_pages)

#         for p in pages_to_show:
#             if p == "...":
#                 lbl = QLabel("...")
#                 lbl.setStyleSheet("padding: 4px 8px; color: #555;")
#                 self.pagination.addWidget(lbl)
#             else:
#                 btn = QPushButton(str(p))
#                 if p == self.current_page:
#                     btn.setStyleSheet("""
#                         QPushButton {
#                             background: #0b5cff;
#                             color: white;
#                             border-radius: 6px;
#                             padding: 4px 10px;
#                             font-weight: bold;
#                         }
#                     """)
#                 else:
#                     self._style_pagination_button(btn)
#                     btn.clicked.connect(lambda checked, page=p: self._go_page(page))
                
#                 self.pagination.addWidget(btn)
#                 self._pagination_buttons.append(btn)

#         # Next button
#         btn_next = QPushButton("Next ▶")
#         btn_next.setEnabled(self.current_page < total_pages)
#         if self.current_page < total_pages:
#             btn_next.clicked.connect(lambda: self._go_page(self.current_page + 1))
#         self._style_pagination_button(btn_next)
#         self.pagination.addWidget(btn_next)
#         self._pagination_buttons.append(btn_next)

#     def _style_pagination_button(self, btn):
#         btn.setStyleSheet("""
#             QPushButton {
#                 background: #e5e5ea;
#                 padding: 4px 10px;
#                 border-radius: 6px;
#                 color: #333;
#             }
#             QPushButton:hover {
#                 background: #d1d1d6;
#             }
#             QPushButton:disabled {
#                 background: #f0f0f0;
#                 color: #999;
#             }
#         """)

#     def _go_page(self, page):
#         total_pages = ceil(self.total_rows / PAGE_SIZE) if self.total_rows > 0 else 1
#         if 1 <= page <= total_pages and page != self.current_page:
#             self.current_page = page
#             self._load_data(force=True)

#     def _export_csv(self):
#         """✅ FIXED: CSV export includes track IDs"""
#         file_path, _ = QFileDialog.getSaveFileName(
#             self,
#             "Export Violations",
#             f"violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#             "CSV Files (*.csv)"
#         )

#         if not file_path:
#             return

#         try:
#             r = requests.get(
#                 f"{BACKEND_URL}/api/ppe/violations",
#                 params={"page": 1, "page_size": 10000},
#                 timeout=30
#             )

#             resp = r.json()
#             if not resp.get("success"):
#                 raise RuntimeError(resp.get("error"))

#             rows = resp.get("data", [])

#             with open(file_path, "w", newline="", encoding="utf-8") as f:
#                 writer = csv.writer(f)
#                 writer.writerow([
#                     "Person ID / Track ID",
#                     "Person Name",
#                     "Role",
#                     "Missing PPE",
#                     "Camera",
#                     "Timestamp",
#                     "Is Unknown"
#                 ])

#                 for log in rows:
#                     person_id = log.get("person_id", "")
#                     track_id = log.get("track_id")
#                     is_unknown = log.get("is_unknown_person", False)
                    
#                     # Format person identifier
#                     if is_unknown and track_id:
#                         person_identifier = f"Unknown ({track_id})"
#                     elif person_id:
#                         person_identifier = person_id
#                     else:
#                         person_identifier = "Unknown"

#                     writer.writerow([
#                         person_identifier,
#                         log.get("person_name", ""),
#                         log.get("role", "visitor"),
#                         ", ".join(log.get("missing_ppe", [])),
#                         log.get("camera_id", ""),
#                         log.get("timestamp", ""),
#                         "Yes" if is_unknown else "No"
#                     ])

#             QMessageBox.information(
#                 self,
#                 "Export Complete",
#                 f"Exported {len(rows)} records to {file_path}"
#             )

#         except Exception as e:
#             QMessageBox.critical(self, "Export Error", str(e))



# ui/screens/violations_report_screen.py - ENHANCED VERSION
# ✅ Proper row numbering visible
# ✅ Better spacing and readability
# ✅ Track IDs for unknown persons

import csv, base64, os, requests
from math import ceil
from datetime import datetime, timezone
import pytz
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QLineEdit, QPushButton, 
    QDialog, QTableWidget, QTableWidgetItem, QHeaderView, QFrame, 
    QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

BACKEND_URL = "http://127.0.0.1:5000"
PAGE_SIZE = 40
LOCAL_TZ = pytz.timezone("Asia/Kolkata")

def utc_to_local(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", ""))
        dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts


class ViolationsReportScreen(QWidget):
    def __init__(self, camera_manager=None, parent=None):
        super().__init__(parent)
        self.cm = camera_manager

        self.current_page = 1
        self.total_rows = 0
        self.search_text = ""
        self._pagination_buttons = []

        self._build_ui()
        self._load_data()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(30, 30, 30, 30)
        root.setSpacing(20)

        # PAGE TITLE
        title = QLabel("Safety Violation Reports")
        title.setStyleSheet("""
            font-size: 26px;
            font-weight: 700;
            color: #1d1d1f;
        """)
        root.addWidget(title)

        # SEARCH BAR + BUTTONS
        top_bar = QHBoxLayout()
        top_bar.setSpacing(10)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search by worker name, ID, track ID, or missing PPE...")
        self.search_input.textChanged.connect(self._on_search)
        self.search_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border-radius: 8px;
                border: 1px solid #d1d1d6;
                background: white;
                min-width: 300px;
            }
        """)

        btn_refresh = QPushButton("🔄 Refresh")
        btn_refresh.clicked.connect(self._refresh_data)
        
        btn_export = QPushButton("⬇ Export CSV")
        btn_export.clicked.connect(self._export_csv)

        for b in [btn_refresh, btn_export]:
            b.setStyleSheet("""
                QPushButton {
                    background: #e5e5ea;
                    padding: 8px 16px;
                    border-radius: 8px;
                    font-size: 14px;
                }
                QPushButton:hover { background: #d1d1d6; }
            """)

        top_bar.addWidget(self.search_input)
        top_bar.addWidget(btn_refresh)
        top_bar.addWidget(btn_export)
        top_bar.addStretch()

        root.addLayout(top_bar)

        # TABLE CARD
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background: white;
                border-radius: 12px;
                border: 1px solid #d1d1d6;
            }
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 20, 20, 20)

        # ✅ TABLE WITH BETTER STYLING
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            "#",  # Row number
            "Worker (ID / Track ID)",
            "Role",
            "Missing PPE",
            "Time",
            "Camera",
        ])

        # ✅ Better table styling
        self.table.setStyleSheet("""
            QTableWidget {
                border: none;
                gridline-color: #e5e5ea;
                font-size: 13px;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #f0f0f0;
            }
            QHeaderView::section {
                background: #f9fafb;
                padding: 10px;
                border: none;
                border-bottom: 2px solid #e5e5ea;
                font-weight: 600;
                font-size: 13px;
            }
        """)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        
        # ✅ Better column widths
        self.table.setColumnWidth(0, 60)   # Row number - wider
        self.table.setColumnWidth(1, 240)  # Worker
        self.table.setColumnWidth(2, 100)  # Role
        self.table.setColumnWidth(3, 300)  # Missing PPE
        self.table.setColumnWidth(4, 160)  # Time
        self.table.setColumnWidth(5, 120)  # Camera

        # ✅ Set minimum row height for better readability
        self.table.verticalHeader().setDefaultSectionSize(40)
        self.table.verticalHeader().setVisible(False)  # Hide default row numbers

        card_layout.addWidget(self.table)
        root.addWidget(card)

        # PAGINATION
        pagination_container = QFrame()
        pagination_container.setStyleSheet("background: transparent;")
        pagination_layout = QHBoxLayout(pagination_container)
        pagination_layout.setContentsMargins(0, 0, 0, 0)
        
        self.pagination = QHBoxLayout()
        self.pagination.setSpacing(6)
        pagination_layout.addStretch()
        pagination_layout.addLayout(self.pagination)
        pagination_layout.addStretch()
        
        root.addWidget(pagination_container)
        
        # STATUS BAR
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666; font-size: 12px;")
        root.addWidget(self.status_label)

    def _load_data(self, force=False):
        try:
            params = {
                "page": self.current_page,
                "page_size": PAGE_SIZE
            }

            if self.search_text:
                params["search"] = self.search_text.strip()

            r = requests.get(
                f"{BACKEND_URL}/api/ppe/violations",
                params=params,
                timeout=5
            )

            resp = r.json()

            if not resp.get("success"):
                raise RuntimeError(resp.get("error"))

            rows = resp.get("data", [])
            self.total_rows = resp.get("total", len(rows))

            self._populate_table(rows)
            self._update_status()
            self._update_pagination()

        except Exception as e:
            self.status_label.setText(f"Error loading data: {e}")
    
    def _refresh_data(self):
        self.current_page = 1
        self._load_data(force=True)

    def _update_status(self):
        total_pages = ceil(self.total_rows / PAGE_SIZE) if self.total_rows > 0 else 1
        self.status_label.setText(
            f"Showing page {self.current_page} of {total_pages} "
            f"({self.total_rows} total violations)"
        )

    def _populate_table(self, logs):
        """
        ✅ FIXED: Proper row numbering and track ID display
        """
        self.table.setRowCount(0)
        self.table.setRowCount(len(logs))

        for row, log in enumerate(logs):
            # ✅ Column 0 – Row Number (proper absolute numbering)
            row_number = (self.current_page - 1) * PAGE_SIZE + row + 1
            row_item = QTableWidgetItem(str(row_number))
            row_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            row_item.setFlags(Qt.ItemFlag.ItemIsEnabled)  # Make read-only
            self.table.setItem(row, 0, row_item)

            # ✅ Column 1 – Person (with track ID support)
            person_id = log.get("person_id")
            track_id = log.get("track_id")
            is_unknown = log.get("is_unknown_person", False)

            if is_unknown and track_id:
                display = f"Unknown ({track_id})"
            elif person_id and person_id != "UNKNOWN":
                display = person_id
            else:
                display = "Unknown Visitor"

            person_item = QTableWidgetItem(display)
            self.table.setItem(row, 1, person_item)

            # Column 2 – Role
            role = log.get("role", "visitor").capitalize()
            role_item = QTableWidgetItem(role)
            self.table.setItem(row, 2, role_item)

            # Column 3 – Missing PPE
            missing = log.get("missing_ppe", [])
            text = ", ".join(missing) if missing else "—"
            ppe_item = QTableWidgetItem(text)
            self.table.setItem(row, 3, ppe_item)

            # Column 4 – Time
            ts = log.get("timestamp")
            time_text = utc_to_local(ts) if ts else "—"
            time_item = QTableWidgetItem(time_text)
            self.table.setItem(row, 4, time_item)

            # Column 5 – Camera
            cam = log.get("camera_id", "—")
            cam_item = QTableWidgetItem(cam)
            self.table.setItem(row, 5, cam_item)

    def _on_search(self):
        self.search_text = self.search_input.text()
        self.current_page = 1
        self._load_data(force=True)

    def _update_pagination(self):
        """Update pagination buttons"""
        for button in self._pagination_buttons:
            if button.parent():
                button.setParent(None)
            button.deleteLater()
        self._pagination_buttons.clear()
        
        while self.pagination.count():
            item = self.pagination.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        total_pages = ceil(self.total_rows / PAGE_SIZE) if self.total_rows > 0 else 1

        if total_pages <= 1:
            return

        # Previous button
        btn_prev = QPushButton("◀ Prev")
        btn_prev.setEnabled(self.current_page > 1)
        if self.current_page > 1:
            btn_prev.clicked.connect(lambda: self._go_page(self.current_page - 1))
        self._style_pagination_button(btn_prev)
        self.pagination.addWidget(btn_prev)
        self._pagination_buttons.append(btn_prev)

        # Page numbers
        pages_to_show = []
        if total_pages <= 7:
            pages_to_show = list(range(1, total_pages + 1))
        else:
            pages_to_show.append(1)
            
            if self.current_page > 3:
                pages_to_show.append("...")
            
            start = max(2, self.current_page - 1)
            end = min(total_pages - 1, self.current_page + 1)
            
            for p in range(start, end + 1):
                if p not in pages_to_show:
                    pages_to_show.append(p)
            
            if self.current_page < total_pages - 2:
                if end < total_pages - 2:
                    pages_to_show.append("...")
            
            if total_pages not in pages_to_show:
                pages_to_show.append(total_pages)

        for p in pages_to_show:
            if p == "...":
                lbl = QLabel("...")
                lbl.setStyleSheet("padding: 4px 8px; color: #555;")
                self.pagination.addWidget(lbl)
            else:
                btn = QPushButton(str(p))
                if p == self.current_page:
                    btn.setStyleSheet("""
                        QPushButton {
                            background: #0b5cff;
                            color: white;
                            border-radius: 6px;
                            padding: 4px 10px;
                            font-weight: bold;
                        }
                    """)
                else:
                    self._style_pagination_button(btn)
                    btn.clicked.connect(lambda checked, page=p: self._go_page(page))
                
                self.pagination.addWidget(btn)
                self._pagination_buttons.append(btn)

        # Next button
        btn_next = QPushButton("Next ▶")
        btn_next.setEnabled(self.current_page < total_pages)
        if self.current_page < total_pages:
            btn_next.clicked.connect(lambda: self._go_page(self.current_page + 1))
        self._style_pagination_button(btn_next)
        self.pagination.addWidget(btn_next)
        self._pagination_buttons.append(btn_next)

    def _style_pagination_button(self, btn):
        btn.setStyleSheet("""
            QPushButton {
                background: #e5e5ea;
                padding: 4px 10px;
                border-radius: 6px;
                color: #333;
            }
            QPushButton:hover {
                background: #d1d1d6;
            }
            QPushButton:disabled {
                background: #f0f0f0;
                color: #999;
            }
        """)

    def _go_page(self, page):
        total_pages = ceil(self.total_rows / PAGE_SIZE) if self.total_rows > 0 else 1
        if 1 <= page <= total_pages and page != self.current_page:
            self.current_page = page
            self._load_data(force=True)

    def _export_csv(self):
        """✅ FIXED: CSV export includes track IDs"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Violations",
            f"violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv)"
        )

        if not file_path:
            return

        try:
            r = requests.get(
                f"{BACKEND_URL}/api/ppe/violations",
                params={"page": 1, "page_size": 10000},
                timeout=30
            )

            resp = r.json()
            if not resp.get("success"):
                raise RuntimeError(resp.get("error"))

            rows = resp.get("data", [])

            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Row #",
                    "Person ID / Track ID",
                    "Person Name",
                    "Role",
                    "Missing PPE",
                    "Camera",
                    "Timestamp",
                    "Is Unknown"
                ])

                for idx, log in enumerate(rows, 1):
                    person_id = log.get("person_id", "")
                    track_id = log.get("track_id")
                    is_unknown = log.get("is_unknown_person", False)
                    
                    if is_unknown and track_id:
                        person_identifier = f"Unknown ({track_id})"
                    elif person_id:
                        person_identifier = person_id
                    else:
                        person_identifier = "Unknown"

                    writer.writerow([
                        idx,
                        person_identifier,
                        log.get("person_name", ""),
                        log.get("role", "visitor"),
                        ", ".join(log.get("missing_ppe", [])),
                        log.get("camera_id", ""),
                        log.get("timestamp", ""),
                        "Yes" if is_unknown else "No"
                    ])

            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported {len(rows)} records to {file_path}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))