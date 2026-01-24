
import sys
from pathlib import Path

# Add parent directory to path so imports work from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QStackedWidget
)

# Import the modern sidebar (place this file as: desktop_ui/widgets/modern_sidebar.py)
from desktop_ui.widgets.sidebar import ModernSidebar

# Import your screens
# from desktop_ui.camera_grid_screen import CameraGridScreen
from desktop_ui.all_cameras import CameraMonitorScreen
from desktop_ui.workers_screen import WorkersScreen
from desktop_ui.attendance_screen import AttendanceScreen
from desktop_ui.violations_report_screen import ViolationsReportScreen
from desktop_ui.settings_screen import SettingsScreen


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("SiteSecureVision")
        self.resize(1400, 900)

        # Root widget
        root = QWidget(self)
        self.setCentralWidget(root)

        # Main horizontal layout
        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ==========================================
        # SIDEBAR (Modern)
        # ==========================================
        self.sidebar = ModernSidebar()
        main_layout.addWidget(self.sidebar)

        # ==========================================
        # MAIN CONTENT AREA
        # ==========================================
        content_container = QWidget()
        content_container.setObjectName("MainContent")
        
        content_layout = QHBoxLayout(content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # Stacked widget for pages
        self.stack = QStackedWidget()
        content_layout.addWidget(self.stack)

        main_layout.addWidget(content_container)

        # ==========================================
        # CREATE PAGES
        # ==========================================
        self.pages = {
            # "cameras": CameraGridScreen(),
            "monitor": CameraMonitorScreen(),
            "workers": WorkersScreen(),
            "attendance": AttendanceScreen(),
            "violations": ViolationsReportScreen(),
            "settings": SettingsScreen(),
        }

        # Add pages to stack
        for page in self.pages.values():
            self.stack.addWidget(page)

        # ==========================================
        # CONNECT SIDEBAR SIGNALS
        # ==========================================
        # self.sidebar.cameras_clicked.connect(lambda: self._switch_page("cameras"))
        self.sidebar.monitor_clicked.connect(lambda: self._switch_page("monitor"))
        self.sidebar.workers_clicked.connect(lambda: self._switch_page("workers"))
        self.sidebar.attendance_clicked.connect(lambda: self._switch_page("attendance"))
        self.sidebar.violations_clicked.connect(lambda: self._switch_page("violations"))
        self.sidebar.settings_clicked.connect(lambda: self._switch_page("settings"))

        # Show default page
        self._switch_page("monitor")

    def _switch_page(self, page_key):
        if page_key in self.pages:
            self.stack.setCurrentWidget(self.pages[page_key])

            # Sync sidebar active state
            sidebar_map = {
                # "cameras": self.sidebar.btn_cameras,
                "monitor": self.sidebar.btn_monitor,
                "workers": self.sidebar.btn_workers,
                "attendance": self.sidebar.btn_attendance,
                "violations": self.sidebar.btn_violations,
                "settings": self.sidebar.btn_settings,
            }

            btn = sidebar_map.get(page_key)
            if btn:
                self.sidebar._set_active(btn)

            print(f"[MainWindow] Switched to: {page_key}")


def main():
    app = QApplication(sys.argv)

    # Load fonts
    QFontDatabase.addApplicationFont("desktop_ui/assets/fonts/Inter-Regular.ttf")
    QFontDatabase.addApplicationFont("desktop_ui/assets/fonts/Inter-SemiBold.ttf")
    QFontDatabase.addApplicationFont("desktop_ui/assets/fonts/Inter-Bold.ttf")

    # Load modern theme
    try:
        with open("desktop_ui/assets/theme.qss", "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
            print("[Theme] Modern theme loaded successfully")
    except FileNotFoundError:
        
        print("[Theme] Warning theme.qss not found, using default styling")

    # Create and show main window
    window = MainWindow()
    window.show()

    print("""
    ==========================================
    🚀 SiteSecureVision Started
    ==========================================
    Modern UI with responsive sidebar
    ==========================================
    """)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()