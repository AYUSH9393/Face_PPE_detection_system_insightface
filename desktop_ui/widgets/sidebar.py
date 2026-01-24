"""
Modern Sidebar with Clean Design
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap


class ModernSidebar(QWidget):
    """Modern sidebar with navigation buttons"""
    
    # Signals for navigation
    # cameras_clicked = pyqtSignal()
    monitor_clicked = pyqtSignal()
    workers_clicked = pyqtSignal()
    attendance_clicked = pyqtSignal()
    violations_clicked = pyqtSignal()
    settings_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set fixed width
        self.setFixedWidth(220)
        
        # CRITICAL: Set object name for styling
        self.setObjectName("ModernSidebar")
        
        # Apply sidebar styling directly and forcefully
        self.setStyleSheet("""
            #ModernSidebar {
                background-color: #f8f9fa;
                border-right: 1px solid #e5e7eb;
            }
        """)
        
        # Force background using palette as backup
        from PyQt6.QtGui import QPalette, QColor
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#f8f9fa"))
        self.setAutoFillBackground(True)
        self.setPalette(palette)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 24, 16, 24)
        layout.setSpacing(8)
        
        # App logo/title
        logo = QLabel()
        logo_path = r"D:\Face_detection_A - Copy\desktop_ui\image.png"
        pixmap = QPixmap(logo_path)
        
        if not pixmap.isNull():
            # Scale to fit sidebar width (220px - margins)
            scaled = pixmap.scaledToWidth(180, Qt.TransformationMode.SmoothTransformation)
            logo.setPixmap(scaled)
        else:
            logo.setText("SiteSecureVision")
            logo.setStyleSheet("""
                QLabel {
                    color: #111827;
                    font-size: 22px;
                    font-weight: 800;
                    padding: 12px 8px;
                }
            """)
        
        logo.setContentsMargins(0, 0, 0, 16)
        layout.addWidget(logo)
        
        # Navigation buttons
        # self.btn_cameras = self._create_nav_button("Cameras", True)
        self.btn_monitor = self._create_nav_button("Live Monitor")
        self.btn_workers = self._create_nav_button("Workers")
        self.btn_attendance = self._create_nav_button("Attendance")
        self.btn_violations = self._create_nav_button("Violations")
        self.btn_settings = self._create_nav_button("Settings")
        
        # Store buttons for active state management
        self.nav_buttons = [
            # self.btn_cameras,
            self.btn_monitor,
            self.btn_workers,
            self.btn_attendance,
            self.btn_violations,
            self.btn_settings
        ]
        
        # Add buttons to layout
        # layout.addWidget(self.btn_cameras)
        layout.addWidget(self.btn_monitor)
        layout.addWidget(self.btn_workers)
        layout.addWidget(self.btn_attendance)
        layout.addWidget(self.btn_violations)
        layout.addWidget(self.btn_settings)
        
        # Add stretch to push buttons to top
        layout.addStretch()
        
        # Connect signals
        # self.btn_cameras.clicked.connect(lambda: self._handle_click(self.btn_cameras, self.cameras_clicked))
        self.btn_monitor.clicked.connect(lambda: self._handle_click(self.btn_monitor, self.monitor_clicked))
        self.btn_workers.clicked.connect(lambda: self._handle_click(self.btn_workers, self.workers_clicked))
        self.btn_attendance.clicked.connect(lambda: self._handle_click(self.btn_attendance, self.attendance_clicked))
        self.btn_violations.clicked.connect(lambda: self._handle_click(self.btn_violations, self.violations_clicked))
        self.btn_settings.clicked.connect(lambda: self._handle_click(self.btn_settings, self.settings_clicked))
        
        # Set first button as active
        self._set_active(self.btn_monitor)
    
    def _create_nav_button(self, text, active=False):
        """Create a navigation button"""
        btn = QPushButton(text)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setFixedHeight(48)
        
        # Set font
        font = QFont()
        font.setPixelSize(17)
        font.setWeight(QFont.Weight.Medium)
        btn.setFont(font)
        
        if active:
            btn.setStyleSheet(self._get_active_style())
        else:
            btn.setStyleSheet(self._get_inactive_style())
        
        return btn
    
    def _get_inactive_style(self):
        """Style for inactive buttons"""
        return """
            QPushButton {
                background-color: transparent;
                color: #000000;
                border: none;
                border-radius: 8px;
                text-align: left;
                padding-left: 16px;
                font-size: 15px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #f3f4f6;
            }
        """
    
    def _get_active_style(self):
        """Style for active button"""
        return """
            QPushButton {
                background-color: #dbeafe;
                color: #000000;
                border: none;
                border-radius: 8px;
                text-align: left;
                padding-left: 16px;
                font-size: 15px;
                font-weight: 800;
            }
            QPushButton:hover {
                background-color: #bfdbfe;
            }
        """
    
    def _handle_click(self, clicked_btn, signal):
        """Handle button click"""
        self._set_active(clicked_btn)
        signal.emit()
    
    def _set_active(self, active_btn):
        """Set active button state"""
        for btn in self.nav_buttons:
            if btn == active_btn:
                btn.setStyleSheet(self._get_active_style())
            else:
                btn.setStyleSheet(self._get_inactive_style())