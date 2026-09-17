from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QButtonGroup
from PySide6.QtCore import Qt, Signal


class CameraProjectionToggleWidget(QFrame):
    """
    Floating segmented toggle pill widget placed directly beneath the 3D orientation gizmo.
    Allows switching between Perspective and Orthographic camera projections in the 3D Viewport.
    """
    projection_changed = Signal(str)  # Emits "perspective" or "orthographic"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CameraProjectionToggle")
        self.setFixedSize(100, 26)

        # Style definition for the floating pill container and buttons
        self.setStyleSheet("""
            QFrame#CameraProjectionToggle {
                background-color: rgba(20, 20, 20, 220);
                border: 1px solid #3A3A3A;
                border-radius: 13px;
            }
            QPushButton.proj-btn {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 10px;
                color: #8E8E8E;
                font-size: 10px;
                font-weight: 600;
                font-family: 'Outfit', 'Inter', -apple-system, sans-serif;
                padding: 2px 4px;
                margin: 0px;
            }
            QPushButton.proj-btn:hover {
                background-color: rgba(255, 255, 255, 0.08);
                color: #E0E0E0;
            }
            QPushButton.proj-btn:checked {
                background-color: #1B382B;
                border: 1px solid #00E676;
                color: #00E676;
                font-weight: bold;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.btn_persp = QPushButton("Persp", self)
        self.btn_persp.setProperty("class", "proj-btn")
        self.btn_persp.setCheckable(True)
        self.btn_persp.setChecked(True)
        self.btn_persp.setToolTip("Perspective Projection (Persp)")
        self.btn_persp.setCursor(Qt.PointingHandCursor)

        self.btn_ortho = QPushButton("Ortho", self)
        self.btn_ortho.setProperty("class", "proj-btn")
        self.btn_ortho.setCheckable(True)
        self.btn_ortho.setChecked(False)
        self.btn_ortho.setToolTip("Orthographic Projection (Ortho)")
        self.btn_ortho.setCursor(Qt.PointingHandCursor)

        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)
        self.button_group.addButton(self.btn_persp, 0)
        self.button_group.addButton(self.btn_ortho, 1)

        layout.addWidget(self.btn_persp)
        layout.addWidget(self.btn_ortho)

        self.btn_persp.clicked.connect(lambda: self._on_btn_clicked("perspective"))
        self.btn_ortho.clicked.connect(lambda: self._on_btn_clicked("orthographic"))

    def _on_btn_clicked(self, mode: str):
        self.projection_changed.emit(mode)

    def set_projection(self, mode: str, block_signals: bool = False):
        """
        Updates the toggle button UI state.
        mode: 'perspective' or 'orthographic'
        """
        mode_lower = mode.lower()
        if block_signals:
            self.button_group.blockSignals(True)
            self.btn_persp.blockSignals(True)
            self.btn_ortho.blockSignals(True)

        if "ortho" in mode_lower:
            self.btn_ortho.setChecked(True)
        else:
            self.btn_persp.setChecked(True)

        if block_signals:
            self.button_group.blockSignals(False)
            self.btn_persp.blockSignals(False)
            self.btn_ortho.blockSignals(False)

    def get_projection(self) -> str:
        """Returns currently active projection mode: 'perspective' or 'orthographic'."""
        return "orthographic" if self.btn_ortho.isChecked() else "perspective"

    def toggle(self):
        """Toggles between Perspective and Orthographic modes and emits projection_changed."""
        if self.btn_ortho.isChecked():
            self.set_projection("perspective")
            self.projection_changed.emit("perspective")
        else:
            self.set_projection("orthographic")
            self.projection_changed.emit("orthographic")
