# # PyQt Settings Screen for PPE Rules & Alerts (FINAL)
# # --------------------------------------------------
# # Thin UI layer – all logic handled by backend APIs

# import requests
# from PyQt6.QtWidgets import (
#     QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
#     QComboBox, QCheckBox, QGroupBox, QGridLayout,
#     QMessageBox, QSpinBox, QTabWidget, QTableWidget,
#     QTableWidgetItem, QHeaderView
# )
# from PyQt6.QtCore import Qt

# BACKEND_URL = "http://127.0.0.1:5000"
# HEADERS_ADMIN = {"X-User": "ui", "X-User-Role": "admin"}


# class SettingsScreen(QWidget):
#     """
#     Main Settings Screen
#     Tabs:
#     - PPE Rules (Admin only)
#     - Alerts (Admin only)
#     - Audit Logs (Read-only)
#     """
#     def __init__(self, parent=None, is_admin=True):
#         super().__init__(parent)
#         self.is_admin = is_admin

#         layout = QVBoxLayout(self)

#         title = QLabel("⚙️ System Settings")
#         title.setStyleSheet("font-size:18px;font-weight:bold;")
#         layout.addWidget(title)

#         self.tabs = QTabWidget()
#         layout.addWidget(self.tabs)

#         self.ppe_tab = PPETab(is_admin)
#         self.alert_tab = AlertsTab(is_admin)
#         self.audit_tab = AuditTab()

#         self.tabs.addTab(self.ppe_tab, "🦺 PPE Rules")
#         self.tabs.addTab(self.alert_tab, "🚨 Alerts")
#         self.tabs.addTab(self.audit_tab, "📜 Audit Logs")


# # ==================================================
# # PPE RULES TAB
# # ==================================================
# class PPETab(QWidget):
#     def __init__(self, is_admin: bool):
#         super().__init__()
#         self.is_admin = is_admin

#         layout = QVBoxLayout(self)
#         layout.setContentsMargins(8, 8, 8, 8)  
#         layout.setSpacing(6)

#         role_row = QHBoxLayout()
#         role_row.addWidget(QLabel("Select Role:"))
#         self.role_selector = QComboBox()
#         role_row.addWidget(self.role_selector)
#         layout.addLayout(role_row)

#         # --- Add role creation UI ---
#         role_add_layout = QHBoxLayout()

#         self.new_role_input = QComboBox()
#         self.new_role_input.setEditable(True)
#         self.new_role_input.setPlaceholderText("Enter new role name")

#         self.add_role_btn = QPushButton("➕ Add Role")
#         self.add_role_btn.clicked.connect(self.add_role)

#         role_add_layout.addWidget(self.new_role_input)
#         role_add_layout.addWidget(self.add_role_btn)
#         layout.addLayout(role_add_layout)

#         self.ppe_group = QGroupBox("Required PPE")
#         self.ppe_layout = QGridLayout(self.ppe_group)
#         layout.addWidget(self.ppe_group)

#         self.save_btn = QPushButton("💾 Save PPE Rules")
#         self.save_btn.clicked.connect(self.save_rules)
#         layout.addWidget(self.save_btn)

#         if not is_admin:
#             self.ppe_group.setEnabled(False)
#             self.save_btn.setEnabled(False)

#         self.checkboxes = {}
#         self.rules = {}
#         self.available = []

#         self.load_data()

#     def load_data(self):
#         r = requests.get(f"{BACKEND_URL}/api/settings/ppe")
#         if r.status_code != 200:
#             QMessageBox.critical(self, "Error", r.text)
#             return

#         data = r.json()["data"]
#         self.available = data["available_ppe_classes"]
#         self.rules = data["role_rules"]

#         self.role_selector.clear()
#         self.new_role_input.clear()

#         roles = sorted(self.rules.keys())
#         self.role_selector.addItems(roles)
#         self.new_role_input.addItems(roles)

#         if not self.checkboxes:
#             for i, ppe in enumerate(self.available):
#                 cb = QCheckBox(ppe)
#                 self.checkboxes[ppe] = cb
#                 self.ppe_layout.addWidget(cb, i // 3, i % 3)

#         if roles:
#             self.load_role(roles[0])


#     def add_role(self):
#         role = self.new_role_input.currentText().strip().lower().replace(" ", "_")

#         if not role:
#             QMessageBox.warning(self, "Invalid", "Role name cannot be empty")
#             return

#         if role in self.rules:
#             QMessageBox.information(self, "Exists", "Role already exists")
#             self.role_selector.setCurrentText(role)
#             return

#         payload = {"role_rules": {role: []}}

#         r = requests.put(
#             f"{BACKEND_URL}/api/settings/ppe",
#             json=payload,
#             headers=HEADERS_ADMIN
#         )

#         if r.status_code == 200:
#             QMessageBox.information(self, "Added", f"Role '{role}' created")
#             self.load_data()
#             self.role_selector.setCurrentText(role)
#         else:
#             QMessageBox.critical(self, "Error", r.text)

#     def load_role(self, role):
#         required = self.rules.get(role, [])
#         for ppe, cb in self.checkboxes.items():
#             cb.setChecked(ppe in required)
    

#     def save_rules(self):
#         role = self.role_selector.currentText()
#         selected = [p for p, cb in self.checkboxes.items() if cb.isChecked()]

#         payload = {"role_rules": {role: selected}}
#         r = requests.put(
#             f"{BACKEND_URL}/api/settings/ppe",
#             json=payload,
#             headers=HEADERS_ADMIN
#         )

#         if r.status_code == 200:
#             QMessageBox.information(self, "Saved", "PPE rules updated successfully")
#             self.load_data()
#         else:
#             QMessageBox.critical(self, "Error", r.text)


# # ==================================================
# # ALERTS TAB
# # ==================================================
# class AlertsTab(QWidget):
#     def __init__(self, is_admin: bool):
#         super().__init__()
#         layout = QVBoxLayout(self)

#         self.enable_alerts = QCheckBox("Enable Alerts")
#         self.buzzer_cb = QCheckBox("🔊 Buzzer / Speaker")
#         self.whatsapp_cb = QCheckBox("📲 WhatsApp")

#         layout.addWidget(self.enable_alerts)
#         layout.addWidget(self.buzzer_cb)
#         layout.addWidget(self.whatsapp_cb)

#         cooldown_layout = QHBoxLayout()
#         cooldown_layout.addWidget(QLabel("Cooldown (seconds):"))
#         self.cooldown = QSpinBox()
#         self.cooldown.setRange(0, 3600)
#         cooldown_layout.addWidget(self.cooldown)
#         layout.addLayout(cooldown_layout)

#         self.save_btn = QPushButton("💾 Save Alert Settings")
#         self.save_btn.clicked.connect(self.save_alerts)
#         layout.addWidget(self.save_btn)

#         if not is_admin:
#             self.setEnabled(False)

#         self.load_alerts()

#     def load_alerts(self):
#         r = requests.get(f"{BACKEND_URL}/api/settings/alerts")
#         if r.status_code != 200:
#             return

#         cfg = r.json()["data"]
#         self.enable_alerts.setChecked(cfg.get("enable_alerts", False))
#         channels = cfg.get("alert_channels", [])
#         self.buzzer_cb.setChecked("buzzer" in channels)
#         self.whatsapp_cb.setChecked("whatsapp" in channels)
#         self.cooldown.setValue(cfg.get("cooldown_seconds", 30))

#     def save_alerts(self):
#         payload = {
#             "enable_alerts": self.enable_alerts.isChecked(),
#             "alert_channels": [
#                 c for c, cb in {
#                     "buzzer": self.buzzer_cb,
#                     "whatsapp": self.whatsapp_cb
#                 }.items() if cb.isChecked()
#             ],
#             "cooldown_seconds": self.cooldown.value()
#         }

#         r = requests.put(
#             f"{BACKEND_URL}/api/settings/alerts",
#             json=payload,
#             headers=HEADERS_ADMIN
#         )

#         if r.status_code == 200:
#             QMessageBox.information(self, "Saved", "Alert settings updated")
#         else:
#             QMessageBox.critical(self, "Error", r.text)


# # ==================================================
# # AUDIT LOG TAB
# # ==================================================
# class AuditTab(QWidget):
#     def __init__(self):
#         super().__init__()
#         layout = QVBoxLayout(self)

#         self.table = QTableWidget(0, 4)
#         self.table.setHorizontalHeaderLabels([
#             "Time", "User", "Action", "Entity"
#         ])
#         self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

#         layout.addWidget(self.table)
#         self.load_logs()

#     def load_logs(self):
#         r = requests.get(
#             f"{BACKEND_URL}/api/settings/audit",
#             headers=HEADERS_ADMIN
#         )
#         if r.status_code != 200:
#             return

#         logs = r.json()["data"]
#         for log in logs:
#             row = self.table.rowCount()
#             self.table.insertRow(row)
#             self.table.setItem(row, 0, QTableWidgetItem(str(log.get("performed_at", ""))))
#             self.table.setItem(row, 1, QTableWidgetItem(log.get("performed_by", "")))
#             self.table.setItem(row, 2, QTableWidgetItem(log.get("action", "")))
#             self.table.setItem(row, 3, QTableWidgetItem(log.get("entity", "")))




# PyQt Settings Screen - FIXED VERSION
# --------------------------------------------------
# Fixed Issues:
# 1. PPE rules now merge instead of replace
# 2. Better UI layout for PPE selection
# 3. Role management improved

import requests
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QCheckBox, QGroupBox, QGridLayout,
    QMessageBox, QSpinBox, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QLineEdit, QScrollArea,  QFileDialog
)
from PyQt6.QtCore import Qt
from ppe_color_tab import PPEColorTab

BACKEND_URL = "http://127.0.0.1:5000"
HEADERS_ADMIN = {"X-User": "ui", "X-User-Role": "admin"}


class SettingsScreen(QWidget):
    """Main Settings Screen"""
    def __init__(self, parent=None, is_admin=True):
        super().__init__(parent)
        self.is_admin = is_admin

        layout = QVBoxLayout(self)

        title = QLabel("⚙️ System Settings")
        title.setStyleSheet("font-size:18px;font-weight:bold;")
        layout.addWidget(title)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.ppe_tab = PPETab(is_admin)
        self.ppe_color_tab = PPEColorTab(is_admin)
        self.alert_tab = AlertsTab(is_admin)
        self.audit_tab = AuditTab()

        self.tabs.addTab(self.ppe_tab, "🦺 PPE Rules")
        self.tabs.addTab(self.ppe_color_tab, "🎨 PPE Colors")
        self.tabs.addTab(self.alert_tab, "🚨 Alerts")
        self.tabs.addTab(self.audit_tab, "📜 Audit Logs")


# ==================================================
# FIXED PPE RULES TAB
# ==================================================
class PPETab(QWidget):
    def __init__(self, is_admin: bool):
        super().__init__()
        self.is_admin = is_admin

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # ===== SECTION 1: ROLE MANAGEMENT =====
        role_section = QGroupBox("Role Management")
        role_section.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #d1d1d6;
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        role_layout = QVBoxLayout(role_section)

        # Current roles dropdown
        select_layout = QHBoxLayout()
        select_layout.addWidget(QLabel("Select Existing Role:"))
        self.role_selector = QComboBox()
        self.role_selector.setMinimumWidth(200)
        self.role_selector.currentTextChanged.connect(self.load_role)
        select_layout.addWidget(self.role_selector)
        select_layout.addStretch()
        role_layout.addLayout(select_layout)

        # Add new role
        add_layout = QHBoxLayout()
        add_layout.addWidget(QLabel("Create New Role:"))
        self.new_role_input = QLineEdit()
        self.new_role_input.setPlaceholderText("e.g., supervisor, contractor, manager")
        self.new_role_input.setMinimumWidth(200)
        add_layout.addWidget(self.new_role_input)
        
        self.add_role_btn = QPushButton("➕ Add Role")
        self.add_role_btn.setStyleSheet("""
            QPushButton {
                background: #007AFF;
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover { background: #005BBB; }
        """)
        self.add_role_btn.clicked.connect(self.add_role)
        add_layout.addWidget(self.add_role_btn)

        self.delete_role_btn = QPushButton("🗑️ Delete Role")
        self.delete_role_btn.setStyleSheet("""
            QPushButton {
                background: #FF3B30;
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover { background: #CC0000; }
        """)
        self.delete_role_btn.clicked.connect(self.delete_role)
        add_layout.addWidget(self.delete_role_btn)

        add_layout.addStretch()
        role_layout.addLayout(add_layout)

        layout.addWidget(role_section)

        # ===== SECTION 2: PPE REQUIREMENTS =====
        ppe_section = QGroupBox("Required PPE for Selected Role")
        ppe_section.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #d1d1d6;
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
                background: #f9f9f9;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        ppe_layout = QVBoxLayout(ppe_section)

        # Scroll area for PPE checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(300)
        
        scroll_widget = QWidget()
        self.ppe_grid = QGridLayout(scroll_widget)
        self.ppe_grid.setSpacing(10)
        scroll.setWidget(scroll_widget)
        
        ppe_layout.addWidget(scroll)
        layout.addWidget(ppe_section)

        # ===== SECTION 3: ACTIONS =====
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.save_btn = QPushButton("💾 Save PPE Rules")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background: #34C759;
                color: white;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover { background: #28A745; }
        """)
        self.save_btn.clicked.connect(self.save_rules)
        button_layout.addWidget(self.save_btn)

        layout.addLayout(button_layout)

        if not is_admin:
            role_section.setEnabled(False)
            ppe_section.setEnabled(False)
            self.save_btn.setEnabled(False)

        self.checkboxes = {}
        self.rules = {}
        self.available = []

        self.load_data()

    def load_data(self):
        """Load PPE rules from backend"""
        r = requests.get(f"{BACKEND_URL}/api/settings/ppe")
        if r.status_code != 200:
            QMessageBox.critical(self, "Error", r.text)
            return

        data = r.json()["data"]
        self.available = sorted(data["available_ppe_classes"])
        self.rules = data["role_rules"]

        # Populate role selector
        self.role_selector.blockSignals(True)
        self.role_selector.clear()
        roles = sorted(self.rules.keys())
        self.role_selector.addItems(roles)
        self.role_selector.blockSignals(False)

        # Create PPE checkboxes (organized in 3 columns)
        for i in range(self.ppe_grid.count()):
            self.ppe_grid.itemAt(i).widget().deleteLater()
        self.checkboxes.clear()

        for i, ppe in enumerate(self.available):
            cb = QCheckBox(ppe.replace('_', ' ').title())
            cb.setStyleSheet("padding: 5px; font-size: 13px;")
            self.checkboxes[ppe] = cb
            row = i // 3
            col = i % 3
            self.ppe_grid.addWidget(cb, row, col)

        # Load first role
        if roles:
            self.load_role(roles[0])

    def add_role(self):
        """Add new role"""
        role = self.new_role_input.text().strip().lower().replace(" ", "_")

        if not role:
            QMessageBox.warning(self, "Invalid", "Role name cannot be empty")
            return

        if role in self.rules:
            QMessageBox.information(self, "Exists", f"Role '{role}' already exists")
            self.role_selector.setCurrentText(role)
            return

        # Create role with empty PPE requirements
        payload = {"role_rules": {role: []}}

        r = requests.put(
            f"{BACKEND_URL}/api/settings/ppe",
            json=payload,
            headers=HEADERS_ADMIN
        )

        if r.status_code == 200:
            QMessageBox.information(self, "Success", f"Role '{role}' created")
            self.new_role_input.clear()
            self.load_data()
            self.role_selector.setCurrentText(role)
        else:
            QMessageBox.critical(self, "Error", r.text)

    def delete_role(self):
        """Delete selected role"""
        role = self.role_selector.currentText()
        
        if not role:
            return

        # Confirm deletion
        reply = QMessageBox.question(
            self, 
            "Confirm Delete",
            f"Are you sure you want to delete role '{role}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Send deletion request
        r = requests.delete(
            f"{BACKEND_URL}/api/settings/ppe/{role}",
            headers=HEADERS_ADMIN
        )

        if r.status_code == 200:
            QMessageBox.information(self, "Deleted", f"Role '{role}' deleted")
            self.load_data()
        else:
            QMessageBox.critical(self, "Error", r.text)

    def load_role(self, role):
        """Load PPE requirements for selected role"""
        if not role:
            return

        required = self.rules.get(role, [])
        
        for ppe, cb in self.checkboxes.items():
            cb.setChecked(ppe in required)

    def save_rules(self):
        """Save PPE rules for selected role"""
        role = self.role_selector.currentText()
        
        if not role:
            QMessageBox.warning(self, "Error", "Please select a role")
            return

        selected = [p for p, cb in self.checkboxes.items() if cb.isChecked()]

        payload = {"role_rules": {role: selected}}
        
        r = requests.put(
            f"{BACKEND_URL}/api/settings/ppe",
            json=payload,
            headers=HEADERS_ADMIN
        )

        if r.status_code == 200:
            QMessageBox.information(
                self, 
                "Saved", 
                f"PPE rules for '{role}' updated successfully"
            )
            self.load_data()
        else:
            QMessageBox.critical(self, "Error", r.text)


# ==================================================
# FIXED ALERTS TAB
# ==================================================
# Enhanced Alerts Tab with WhatsApp Configuration UI
# Replace the AlertsTab class in settings_screen.py

class AlertsTab(QWidget):
    def __init__(self, is_admin: bool):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # ===== ALERT STATUS =====
        status_group = QGroupBox("Alert Configuration")
        status_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #d1d1d6;
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
            }
        """)
        status_layout = QVBoxLayout(status_group)

        self.enable_alerts = QCheckBox("Enable Alerts")
        self.enable_alerts.setStyleSheet("font-size: 14px; font-weight: bold;")
        status_layout.addWidget(self.enable_alerts)

        layout.addWidget(status_group)

        # ===== ALERT CHANNELS =====
        channels_group = QGroupBox("Alert Channels")
        channels_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #d1d1d6;
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
            }
        """)
        channels_layout = QVBoxLayout(channels_group)

        self.buzzer_cb = QCheckBox("🔊 Buzzer / Speaker Alert")
        self.whatsapp_cb = QCheckBox("📲 WhatsApp Notification")
        
        for cb in [self.buzzer_cb, self.whatsapp_cb]:
            cb.setStyleSheet("padding: 5px; font-size: 13px;")
            channels_layout.addWidget(cb)

        layout.addWidget(channels_group)

        # ===== WHATSAPP CONFIGURATION =====
        whatsapp_group = QGroupBox("WhatsApp Settings (Twilio)")
        whatsapp_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #34C759;
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
                background: #f0fff4;
            }
        """)
        whatsapp_layout = QGridLayout(whatsapp_group)

        # Account SID
        whatsapp_layout.addWidget(QLabel("Twilio Account SID:"), 0, 0)
        self.whatsapp_sid = QLineEdit()
        self.whatsapp_sid.setPlaceholderText("AC1234567890abcdef...")
        whatsapp_layout.addWidget(self.whatsapp_sid, 0, 1)

        # Auth Token
        whatsapp_layout.addWidget(QLabel("Twilio Auth Token:"), 1, 0)
        self.whatsapp_token = QLineEdit()
        self.whatsapp_token.setPlaceholderText("Your auth token")
        self.whatsapp_token.setEchoMode(QLineEdit.EchoMode.Password)
        whatsapp_layout.addWidget(self.whatsapp_token, 1, 1)

        # From Number
        whatsapp_layout.addWidget(QLabel("From (Twilio WhatsApp):"), 2, 0)
        self.whatsapp_from = QLineEdit()
        self.whatsapp_from.setPlaceholderText("+14155238886")
        whatsapp_layout.addWidget(self.whatsapp_from, 2, 1)

        # To Number
        whatsapp_layout.addWidget(QLabel("To (Your Phone):"), 3, 0)
        self.whatsapp_to = QLineEdit()
        self.whatsapp_to.setPlaceholderText("+919876543210")
        self.whatsapp_to.setStyleSheet("font-weight: bold;")
        whatsapp_layout.addWidget(self.whatsapp_to, 3, 1)

        # Help text
        help_text = QLabel(
            "ℹ️ Get credentials from: <a href='https://console.twilio.com'>console.twilio.com</a><br>"
            "Include country code (e.g., +91 for India, +1 for USA)"
        )
        help_text.setOpenExternalLinks(True)
        help_text.setStyleSheet("color: #666; font-size: 11px; padding: 5px;")
        whatsapp_layout.addWidget(help_text, 4, 0, 1, 2)

        # Test button
        test_btn = QPushButton("📤 Send Test WhatsApp")
        test_btn.setStyleSheet("""
            QPushButton {
                background: #25D366;
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover { background: #1DA851; }
        """)
        test_btn.clicked.connect(self.test_whatsapp)
        whatsapp_layout.addWidget(test_btn, 5, 1, Qt.AlignmentFlag.AlignRight)

        layout.addWidget(whatsapp_group)

        # ===== BUZZER CONFIGURATION =====
        buzzer_group = QGroupBox("Buzzer Settings")
        buzzer_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #d1d1d6;
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
            }
        """)
        buzzer_layout = QHBoxLayout(buzzer_group)
        
        buzzer_layout.addWidget(QLabel("Sound File:"))
        self.buzzer_sound = QLineEdit()
        self.buzzer_sound.setText("alert.wav")
        buzzer_layout.addWidget(self.buzzer_sound)
        
        browse_btn = QPushButton("📁 Browse")
        browse_btn.clicked.connect(self.browse_sound)
        buzzer_layout.addWidget(browse_btn)

        layout.addWidget(buzzer_group)

        # ===== COOLDOWN =====
        cooldown_group = QGroupBox("Cooldown Settings")
        cooldown_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #d1d1d6;
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
            }
        """)
        cooldown_layout = QHBoxLayout(cooldown_group)
        
        cooldown_layout.addWidget(QLabel("Cooldown Period:"))
        self.cooldown = QSpinBox()
        self.cooldown.setRange(5, 3600)
        self.cooldown.setSuffix(" seconds")
        self.cooldown.setMinimumWidth(120)
        cooldown_layout.addWidget(self.cooldown)
        cooldown_layout.addStretch()
        
        help_label = QLabel("⏱️ Minimum time between alerts for same person")
        help_label.setStyleSheet("color: #666; font-size: 11px;")
        cooldown_layout.addWidget(help_label)

        layout.addWidget(cooldown_group)

        # ===== SAVE BUTTON =====
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.save_btn = QPushButton("💾 Save Alert Settings")
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
        self.save_btn.clicked.connect(self.save_alerts)
        button_layout.addWidget(self.save_btn)
        
        layout.addLayout(button_layout)
        layout.addStretch()

        if not is_admin:
            self.setEnabled(False)

        self.load_alerts()

    def load_alerts(self):
        """Load alert configuration from backend"""
        r = requests.get(f"{BACKEND_URL}/api/settings/alerts")
        if r.status_code != 200:
            return

        cfg = r.json()["data"]
        
        # Basic settings
        self.enable_alerts.setChecked(cfg.get("enable_alerts", False))
        
        channels = cfg.get("alert_channels", [])
        self.buzzer_cb.setChecked("buzzer" in channels)
        self.whatsapp_cb.setChecked("whatsapp" in channels)
        
        self.cooldown.setValue(cfg.get("cooldown_seconds", 30))
        
        # WhatsApp settings
        whatsapp = cfg.get("whatsapp", {})
        self.whatsapp_sid.setText(whatsapp.get("sid", ""))
        self.whatsapp_token.setText(whatsapp.get("token", ""))
        self.whatsapp_from.setText(whatsapp.get("from", ""))
        self.whatsapp_to.setText(whatsapp.get("to", ""))
        
        # Buzzer settings
        buzzer = cfg.get("buzzer", {})
        self.buzzer_sound.setText(buzzer.get("sound", "alert.wav"))

    def save_alerts(self):
        """Save alert configuration to backend"""
        payload = {
            "enable_alerts": self.enable_alerts.isChecked(),
            "alert_channels": [
                c for c, cb in {
                    "buzzer": self.buzzer_cb,
                    "whatsapp": self.whatsapp_cb
                }.items() if cb.isChecked()
            ],
            "cooldown_seconds": self.cooldown.value(),
            "whatsapp": {
                "sid": self.whatsapp_sid.text().strip(),
                "token": self.whatsapp_token.text().strip(),
                "from": self.whatsapp_from.text().strip(),
                "to": self.whatsapp_to.text().strip()
            },
            "buzzer": {
                "sound": self.buzzer_sound.text().strip()
            }
        }

        # Validate WhatsApp settings if enabled
        if self.whatsapp_cb.isChecked():
            if not all([payload["whatsapp"]["sid"], 
                       payload["whatsapp"]["token"],
                       payload["whatsapp"]["from"],
                       payload["whatsapp"]["to"]]):
                QMessageBox.warning(
                    self,
                    "Incomplete Configuration",
                    "Please fill in all WhatsApp fields or disable WhatsApp alerts"
                )
                return

        r = requests.put(
            f"{BACKEND_URL}/api/settings/alerts",
            json=payload,
            headers=HEADERS_ADMIN
        )

        if r.status_code == 200:
            QMessageBox.information(
                self, 
                "Saved", 
                "Alert settings updated successfully!\n\n"
                "The alert engine has been reloaded with new settings."
            )
        else:
            QMessageBox.critical(self, "Error", r.text)

    def test_whatsapp(self):
        """Send test WhatsApp message"""
        if not all([
            self.whatsapp_sid.text().strip(),
            self.whatsapp_token.text().strip(),
            self.whatsapp_from.text().strip(),
            self.whatsapp_to.text().strip()
        ]):
            QMessageBox.warning(
                self,
                "Missing Information",
                "Please fill in all WhatsApp fields before testing"
            )
            return

        # Test message
        test_payload = {
            "sid": self.whatsapp_sid.text().strip(),
            "token": self.whatsapp_token.text().strip(),
            "from": self.whatsapp_from.text().strip(),
            "to": self.whatsapp_to.text().strip(),
            "message": "🔔 Test Alert from PPE Detection System\n\nIf you received this, WhatsApp alerts are working!"
        }

        try:
            # You need to add this test endpoint to your backend
            r = requests.post(
                f"{BACKEND_URL}/api/settings/alerts/test",
                json=test_payload,
                headers=HEADERS_ADMIN,
                timeout=10
            )

            if r.status_code == 200:
                QMessageBox.information(
                    self,
                    "Test Sent",
                    "Test WhatsApp message sent!\n\n"
                    f"Check your phone: {test_payload['to']}"
                )
            else:
                QMessageBox.critical(
                    self,
                    "Test Failed",
                    f"Failed to send test message:\n{r.text}"
                )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error sending test:\n{str(e)}"
            )

    def browse_sound(self):
        """Browse for sound file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Alert Sound",
            "",
            "Audio Files (*.wav *.mp3 *.ogg);;All Files (*.*)"
        )
        
        if file_path:
            self.buzzer_sound.setText(file_path)


# ==================================================
# AUDIT LOG TAB (unchanged)
# ==================================================
class AuditTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels([
            "Time", "User", "Action", "Entity"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        layout.addWidget(self.table)
        self.load_logs()

    def load_logs(self):
        r = requests.get(
            f"{BACKEND_URL}/api/settings/audit",
            headers=HEADERS_ADMIN
        )
        if r.status_code != 200:
            return

        logs = r.json()["data"]
        for log in logs:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(str(log.get("performed_at", ""))))
            self.table.setItem(row, 1, QTableWidgetItem(log.get("performed_by", "")))
            self.table.setItem(row, 2, QTableWidgetItem(log.get("action", "")))
            self.table.setItem(row, 3, QTableWidgetItem(log.get("entity", "")))
