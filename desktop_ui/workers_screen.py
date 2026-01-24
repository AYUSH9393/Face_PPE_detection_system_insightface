import os
import requests
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QPushButton, QHBoxLayout, QFileDialog, QMessageBox, QDialog,
    QFormLayout, QLineEdit, QFrame, QComboBox
)
from PyQt6.QtCore import Qt

BACKEND_URL = "http://127.0.0.1:5000"


# -------------------------------------------------
# Add Worker Dialog
# -------------------------------------------------
class AddWorkerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Register Worker")
        self.resize(400, 250)

        layout = QFormLayout(self)

        self.id_input = QLineEdit()
        self.name_input = QLineEdit()
        self.role_input = QComboBox()
        self.role_input.addItems([
            "engineer", "worker", "supervisor", "contractor", 
            "electrician", "welder", "visitor", "manager", "employee"
        ])
        self.images = []

        btn_imgs = QPushButton("Select Face Images (3–5)")
        btn_imgs.clicked.connect(self._select_images)

        layout.addRow("Worker ID:", self.id_input)
        layout.addRow("Name:", self.name_input)
        layout.addRow("Role:", self.role_input)
        layout.addWidget(btn_imgs)

        btns = QHBoxLayout()
        ok = QPushButton("Register")
        cancel = QPushButton("Cancel")
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        btns.addWidget(ok)
        btns.addWidget(cancel)

        layout.addRow(btns)

    def _select_images(self):
        self.images, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "", 
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp *.heic *.heif *.avif *.hicv);;All Files (*)"
        )

# -------------------------------------------------
# Workers Screen
# -------------------------------------------------
class WorkersScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._load_workers()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(25, 25, 25, 25)

        title = QLabel("Worker Management")
        title.setStyleSheet("font-size:26px; font-weight:700;")
        root.addWidget(title)

        toolbar = QHBoxLayout()
        self.btn_add = QPushButton("➕ Register Worker")
        self.btn_delete = QPushButton("🗑 Delete Worker")
        self.btn_refresh = QPushButton("🔄 Refresh")

        self.btn_add.clicked.connect(self._add_worker)
        self.btn_delete.clicked.connect(self._delete_worker)
        self.btn_refresh.clicked.connect(self._load_workers)

        toolbar.addWidget(self.btn_add)
        toolbar.addWidget(self.btn_delete)
        toolbar.addWidget(self.btn_refresh)
        toolbar.addStretch()
        root.addLayout(toolbar)

        card = QFrame()
        card.setStyleSheet("background:white; border-radius:12px;")
        card_layout = QVBoxLayout(card)

        self.table = QTableWidget(0, 4)
        self.table.setWordWrap(False)
        self.table.setTextElideMode(Qt.TextElideMode.ElideNone)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, header.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, header.ResizeMode.Stretch)
        header.setSectionResizeMode(2, header.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, header.ResizeMode.ResizeToContents)

        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setColumnWidth(0, 220)  # Worker ID

        self.table.setHorizontalHeaderLabels(
            ["Worker ID", "Name", "Role", "Status"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        card_layout.addWidget(self.table)

        root.addWidget(card)

    # -------------------------------------------------
    def _load_workers(self):
        self.table.setRowCount(0)

        try:
            r = requests.get(f"{BACKEND_URL}/api/persons", timeout=10)
            resp = r.json()

            if not resp.get("success"):
                raise RuntimeError(resp.get("error", "Unknown error"))

            persons = resp.get("data", [])

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load workers:\n{e}")
            return

        for p in persons:
            row = self.table.rowCount()
            self.table.insertRow(row)

            person_id = p.get("person_id", "")
            name = p.get("name", "")
            role = p.get("role", "")
            status = p.get("status", "active")

            self.table.setItem(row, 0, QTableWidgetItem(person_id))
            self.table.setItem(row, 1, QTableWidgetItem(name))
            self.table.setItem(row, 2, QTableWidgetItem(role))
            self.table.setItem(row, 3, QTableWidgetItem(status))

            self.table.item(row, 0).setData(Qt.ItemDataRole.UserRole, person_id)

    def _add_worker(self):
        dlg = AddWorkerDialog(self)
        if not dlg.exec():
            return

        if len(dlg.images) < 3:
            QMessageBox.warning(self, "Error", "Select at least 3 images")
            return

        file_handles = []  # Track file handles for proper cleanup
        try:
            import mimetypes
            files = []
            for p in dlg.images:
                mime_type, _ = mimetypes.guess_type(p)
                if not mime_type:
                    mime_type = "application/octet-stream"
                fh = open(p, "rb")
                file_handles.append(fh)  # Track for cleanup
                files.append(("images", (os.path.basename(p), fh, mime_type)))

            # ✅ Increased timeout for ArcFace processing (can take 2-3 minutes)
            r = requests.post(
                f"{BACKEND_URL}/api/persons/register-folder",
                data={
                    "person_id": dlg.id_input.text().strip(),
                    "name": dlg.name_input.text().strip(),
                    "role": dlg.role_input.currentText().strip().lower(),
                    "department": "General"
                },
                files=files,
                timeout=180  # Increased from 60 to 180 seconds
            )

            resp = r.json()

            if not resp.get("success"):
                QMessageBox.critical(self, "Registration Failed", resp.get("error"))
                return

            QMessageBox.information(
                self,
                "Registration Complete",
                f"{resp['message']}\nImages used: {resp['images_used']}"
            )

            self._load_workers()

        except requests.exceptions.Timeout:
            QMessageBox.critical(
                self, 
                "Timeout Error", 
                "Registration took too long (>3 minutes).\nThe worker may have been registered - please refresh to check."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Registration failed:\n{str(e)}")
        finally:
            # ✅ Properly close all file handles
            for fh in file_handles:
                try:
                    fh.close()
                except:
                    pass



    # -------------------------------------------------
    def _delete_worker(self):
        row = self.table.currentRow()
        if row < 0:
            return

        person_id = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)

        if QMessageBox.question(
            self, "Confirm", f"Delete person {person_id}?"
        ) != QMessageBox.StandardButton.Yes:
            return

        try:
            requests.delete(
                f"{BACKEND_URL}/api/persons/{person_id}",
                timeout=10
            )
            self._load_workers()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
