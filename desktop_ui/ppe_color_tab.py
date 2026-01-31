"""
PPE Color Configuration Tab
Allows admins to configure role-based color requirements for helmets and vests
"""

import requests
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QCheckBox, QGroupBox, QGridLayout,
    QMessageBox, QSpinBox
)
from PyQt6.QtCore import Qt

BACKEND_URL = "http://127.0.0.1:5000"
HEADERS_ADMIN = {"X-User": "ui", "X-User-Role": "admin"}


class PPEColorTab(QWidget):
    """PPE Color Configuration - Role-based color selection for helmets and vests"""
    
    def __init__(self, is_admin: bool):
        super().__init__()
        self.is_admin = is_admin
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # ===== SECTION 1: COLOR CHECKING TOGGLE =====
        toggle_section = QGroupBox("Color Detection Settings")
        toggle_section.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #d1d1d6;
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
            }
        """)
        toggle_layout = QVBoxLayout(toggle_section)
        
        self.enable_color_checking = QCheckBox("Enable Color-Based PPE Validation")
        self.enable_color_checking.setStyleSheet("font-size: 14px; font-weight: bold;")
        toggle_layout.addWidget(self.enable_color_checking)
        
        # Threshold setting
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Color Match Threshold:"))
        self.color_threshold = QSpinBox()
        self.color_threshold.setRange(10, 100)
        self.color_threshold.setSuffix("%")
        self.color_threshold.setMinimumWidth(100)
        threshold_layout.addWidget(self.color_threshold)
        threshold_layout.addStretch()
        
        help_label = QLabel("⚙️ Percentage of PPE that must match expected color (30% recommended)")
        help_label.setStyleSheet("color: #666; font-size: 11px;")
        threshold_layout.addWidget(help_label)
        
        toggle_layout.addLayout(threshold_layout)
        layout.addWidget(toggle_section)
        
        # ===== SECTION 2: ROLE SELECTION =====
        role_section = QGroupBox("Select Role to Configure Colors")
        role_section.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #d1d1d6;
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
            }
        """)
        role_layout = QHBoxLayout(role_section)
        
        role_layout.addWidget(QLabel("Role:"))
        self.role_selector = QComboBox()
        self.role_selector.setMinimumWidth(200)
        self.role_selector.currentTextChanged.connect(self.load_role_colors)
        role_layout.addWidget(self.role_selector)
        role_layout.addStretch()
        
        layout.addWidget(role_section)
        
        # ===== SECTION 3: HELMET COLORS =====
        helmet_section = QGroupBox("🪖 Safety Helmet - Allowed Colors")
        helmet_section.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #FFD700;
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
                background: #FFFEF0;
            }
        """)
        helmet_layout = QVBoxLayout(helmet_section)
        
        helmet_info = QLabel("Select which helmet colors are allowed for this role:")
        helmet_info.setStyleSheet("color: #666; font-size: 12px; margin-bottom: 10px;")
        helmet_layout.addWidget(helmet_info)
        
        self.helmet_checkboxes = {}
        helmet_grid = QGridLayout()
        helmet_grid.setSpacing(10)
        
        # Will be populated when data loads
        helmet_layout.addLayout(helmet_grid)
        self.helmet_grid = helmet_grid
        
        layout.addWidget(helmet_section)
        
        # ===== SECTION 4: VEST COLORS =====
        vest_section = QGroupBox("🦺 Reflective Vest - Allowed Colors")
        vest_section.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #FF8800;
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
                background: #FFF5E6;
            }
        """)
        vest_layout = QVBoxLayout(vest_section)
        
        vest_info = QLabel("Select which vest colors are allowed for this role:")
        vest_info.setStyleSheet("color: #666; font-size: 12px; margin-bottom: 10px;")
        vest_layout.addWidget(vest_info)
        
        self.vest_checkboxes = {}
        vest_grid = QGridLayout()
        vest_grid.setSpacing(10)
        
        # Will be populated when data loads
        vest_layout.addLayout(vest_grid)
        self.vest_grid = vest_grid
        
        layout.addWidget(vest_section)
        
        # ===== SECTION 5: SAVE BUTTON =====
        button_layout = QHBoxLayout()
        
        
        button_layout.addStretch()
        
        self.save_btn = QPushButton("💾 Save Color Configuration")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background: #34C759;
                color: white;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background: #28A745; }
        """)
        self.save_btn.clicked.connect(self.save_color_config)
        button_layout.addWidget(self.save_btn)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        if not is_admin:
            self.setEnabled(False)
        
        self.available_colors = []
        self.role_color_requirements = {}
        
        # Load data - don't fail if API is unavailable
        try:
            self.load_data()
        except Exception as e:
            print(f"[PPEColorTab] Warning: Could not load initial data: {e}")
            print("[PPEColorTab] Tab will be available but data needs to be loaded manually")
    
    def load_data(self):
        """Load color configuration from backend"""
        print("[PPEColorTab] Loading color configuration...")
        try:
            # Get color configuration
            r = requests.get(f"{BACKEND_URL}/api/settings/ppe/colors", timeout=5)
            print(f"[PPEColorTab] API response: {r.status_code}")
            if r.status_code == 404:
                QMessageBox.warning(
                    self,
                    "Not Configured",
                    "PPE color rules not configured.\n\nPlease run: python 003_add_ppe_color_config.py"
                )
                return
            elif r.status_code != 200:
                QMessageBox.critical(self, "Error", f"Failed to load color config: {r.text}")
                return
            
            data = r.json()["data"]
            
            # Set global settings
            self.enable_color_checking.setChecked(data.get("enable_color_checking", True))
            self.color_threshold.setValue(data.get("color_match_threshold", 30))
            
            # Get available colors
            r_colors = requests.get(f"{BACKEND_URL}/api/settings/ppe/colors/available")
            if r_colors.status_code == 200:
                self.available_colors = r_colors.json()["data"]
            else:
                self.available_colors = []
            
            # Get role color requirements
            self.role_color_requirements = data.get("role_color_requirements", {})
            
            # Populate role selector
            self.role_selector.blockSignals(True)
            self.role_selector.clear()
            roles = sorted(self.role_color_requirements.keys())
            self.role_selector.addItems(roles)
            self.role_selector.blockSignals(False)
            
            # Create color checkboxes
            print(f"[PPEColorTab] Creating checkboxes for {len(self.available_colors)} colors...")
            self.create_color_checkboxes()
            
            # Load first role
            if roles:
                print(f"[PPEColorTab] Loading role: {roles[0]}")
                self.load_role_colors(roles[0])
            
            print("[PPEColorTab] Successfully loaded!")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
    
    def create_color_checkboxes(self):
        """Create checkboxes for each available color"""
        # Clear existing checkboxes
        for i in reversed(range(self.helmet_grid.count())):
            self.helmet_grid.itemAt(i).widget().deleteLater()
        for i in reversed(range(self.vest_grid.count())):
            self.vest_grid.itemAt(i).widget().deleteLater()
        
        self.helmet_checkboxes.clear()
        self.vest_checkboxes.clear()
        
        # Create checkboxes for each color
        for i, color_info in enumerate(self.available_colors):
            color_id = color_info["id"]
            color_name = color_info["name"]
            color_hex = color_info["display_color"]
            
            # Helmet checkbox with color badge
            helmet_container = QWidget()
            helmet_layout = QHBoxLayout(helmet_container)
            helmet_layout.setContentsMargins(0, 0, 0, 0)
            helmet_layout.setSpacing(8)
            
            # Color badge (small colored square)
            helmet_badge = QLabel()
            helmet_badge.setFixedSize(24, 24)
            helmet_badge.setStyleSheet(f"""
                background-color: {color_hex};
                border: 2px solid #333;
                border-radius: 4px;
            """)
            helmet_layout.addWidget(helmet_badge)
            
            # Normal checkbox with label
            helmet_cb = QCheckBox(color_name)
            helmet_cb.setStyleSheet("""
                QCheckBox {
                    font-size: 13px;
                    font-weight: bold;
                }
            """)
            helmet_layout.addWidget(helmet_cb)
            helmet_layout.addStretch()
            
            self.helmet_checkboxes[color_id] = helmet_cb
            row = i // 4
            col = i % 4
            self.helmet_grid.addWidget(helmet_container, row, col)
            
            # Vest checkbox with color badge
            vest_container = QWidget()
            vest_layout = QHBoxLayout(vest_container)
            vest_layout.setContentsMargins(0, 0, 0, 0)
            vest_layout.setSpacing(8)
            
            # Color badge (small colored square)
            vest_badge = QLabel()
            vest_badge.setFixedSize(24, 24)
            vest_badge.setStyleSheet(f"""
                background-color: {color_hex};
                border: 2px solid #333;
                border-radius: 4px;
            """)
            vest_layout.addWidget(vest_badge)
            
            # Normal checkbox with label
            vest_cb = QCheckBox(color_name)
            vest_cb.setStyleSheet("""
                QCheckBox {
                    font-size: 13px;
                    font-weight: bold;
                }
            """)
            vest_layout.addWidget(vest_cb)
            vest_layout.addStretch()
            
            self.vest_checkboxes[color_id] = vest_cb
            self.vest_grid.addWidget(vest_container, row, col)
    
    def load_role_colors(self, role):
        """Load color requirements for selected role"""
        if not role:
            return
        
        requirements = self.role_color_requirements.get(role, {})
        
        # Load helmet colors
        helmet_colors = requirements.get("safety_helmet", [])
        for color_id, cb in self.helmet_checkboxes.items():
            cb.setChecked(color_id in helmet_colors)
        
        # Load vest colors
        vest_colors = requirements.get("reflective_vest", [])
        for color_id, cb in self.vest_checkboxes.items():
            cb.setChecked(color_id in vest_colors)
    
    def save_color_config(self):
        """Save color configuration"""
        role = self.role_selector.currentText()
        
        if not role:
            QMessageBox.warning(self, "Error", "Please select a role")
            return
        
        # Get selected colors
        helmet_colors = [color_id for color_id, cb in self.helmet_checkboxes.items() if cb.isChecked()]
        vest_colors = [color_id for color_id, cb in self.vest_checkboxes.items() if cb.isChecked()]
        
        # Save global settings
        global_payload = {
            "enable_color_checking": self.enable_color_checking.isChecked(),
            "color_match_threshold": self.color_threshold.value()
        }
        
        r_global = requests.put(
            f"{BACKEND_URL}/api/settings/ppe/colors",
            json=global_payload,
            headers=HEADERS_ADMIN
        )
        
        if r_global.status_code != 200:
            QMessageBox.critical(self, "Error", f"Failed to save global settings: {r_global.text}")
            return
        
        # Save role color requirements
        role_payload = {
            "color_requirements": {
                "safety_helmet": helmet_colors,
                "reflective_vest": vest_colors
            }
        }
        
        r_role = requests.put(
            f"{BACKEND_URL}/api/settings/ppe/colors/role/{role}",
            json=role_payload,
            headers=HEADERS_ADMIN
        )
        
        if r_role.status_code == 200:
            QMessageBox.information(
                self,
                "Saved",
                f"Color configuration for '{role}' updated successfully!\n\n"
                f"Helmet colors: {', '.join(helmet_colors) if helmet_colors else 'None'}\n"
                f"Vest colors: {', '.join(vest_colors) if vest_colors else 'None'}"
            )
            self.load_data()
        else:
            QMessageBox.critical(self, "Error", f"Failed to save role colors: {r_role.text}")
