"""Model import settings and material assignments, prepared asynchronously."""
from pathlib import Path
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QCheckBox, QTableWidget, QTableWidgetItem, QFileDialog, QDialogButtonBox,
    QHeaderView, QWidget,
)
from mesh_editor.scene import load_mesh_file
from mesh_editor.import_options import match_textures


class ImportPreviewWorker(QThread):
    loaded = Signal(object, str)

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path

    def run(self):
        try:
            self.loaded.emit(load_mesh_file(self.path), '')
        except Exception as error:
            self.loaded.emit(None, str(error))


class ModelImportDialog(QDialog):
    def __init__(self, file_path, parent=None, texture_paths=()):
        super().__init__(parent)
        self.file_path = file_path
        self.meshes = None
        self.assignments = {}
        self._cancel_requested = False
        self.texture_paths = list(texture_paths)
        self.setWindowTitle('Import Model')
        self.resize(680, 490)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)
        title = QLabel(Path(file_path).name)
        title.setStyleSheet('font-size: 16px; font-weight: bold;')
        layout.addWidget(title)
        path_label = QLabel(str(file_path))
        path_label.setWordWrap(True)
        path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(path_label)
        self.status = QLabel('Reading objects and materials…')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        texture_row = QHBoxLayout()
        texture_row.addWidget(QLabel('Texture'))
        self.texture_mode = QComboBox()
        self.texture_mode.addItems(['Use Embedded', 'External'])
        self.texture_mode.setToolTip('Use Embedded keeps model textures and falls back to material colour when none are available.')
        texture_row.addWidget(self.texture_mode, 1)
        layout.addLayout(texture_row)
        self.texture_hint = QLabel('Embedded textures are used when available. Otherwise the model’s material colour or a default material is used.')
        self.texture_hint.setWordWrap(True)
        layout.addWidget(self.texture_hint)
        self.external = QWidget()
        external_layout = QVBoxLayout(self.external)
        external_layout.setContentsMargins(0, 0, 0, 0)
        self.browse = QPushButton('Select Texture / Material Files…')
        self.browse.clicked.connect(self._browse)
        external_layout.addWidget(self.browse)
        self.file_summary = QLabel('No external files selected')
        self.file_summary.setWordWrap(True)
        external_layout.addWidget(self.file_summary)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(['Material', 'Base-colour texture'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.setMinimumHeight(140)
        external_layout.addWidget(self.table)
        external_layout.addWidget(QLabel('Unmatched materials use their material colour. Choose an assignment to override.'))
        layout.addWidget(self.external)
        self.group = QCheckBox('Group under file name')
        self.group.setChecked(True)
        self.group.setToolTip('Group separate objects under a transformable parent at their combined center. Existing source groups are retained.')
        layout.addWidget(self.group)
        layout.addStretch()
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        self.import_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.import_button.setText('Import Model')
        self.import_button.setEnabled(False)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.texture_mode.currentIndexChanged.connect(self._update_external)
        self._update_external()
        self.worker = ImportPreviewWorker(file_path, self)
        self.worker.loaded.connect(self._loaded)
        self.worker.finished.connect(self._finished)
        self.worker.start()

    def _loaded(self, meshes, error):
        if self._cancel_requested:
            return
        if error:
            self.status.setText(f'Could not read this model: {error}')
            return
        self.meshes = meshes
        materials = {mesh.material_name for _, mesh in meshes}
        self.status.setText(f'{len(meshes)} objects · {len(materials)} materials')
        if self.texture_paths:
            self.texture_mode.setCurrentIndex(1)
        self._populate_assignments()

    def _finished(self):
        if self._cancel_requested:
            super().reject()
        else:
            self.import_button.setEnabled(bool(self.meshes))

    def _browse(self):
        paths, _ = QFileDialog.getOpenFileNames(self, 'Select Texture / Material Files', str(Path(self.file_path).parent),
            'Textures and materials (*.png *.jpg *.jpeg *.bmp *.tga *.webp *.tif *.tiff *.mtl)')
        if paths:
            self.texture_paths = list(dict.fromkeys(self.texture_paths + paths))
            self._populate_assignments()

    def _populate_assignments(self):
        if self.meshes is None:
            return
        images, automatic = match_textures(self.meshes, self.texture_paths)
        self.file_summary.setText(', '.join(Path(path).name for path in self.texture_paths) or 'No external files selected')
        materials = list(dict.fromkeys(mesh.material_name for _, mesh in self.meshes))
        self.table.setRowCount(len(materials))
        self.assignments = {}
        for row, material in enumerate(materials):
            item = QTableWidgetItem(material)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, item)
            picker = QComboBox()
            picker.addItem('Material colour', None)
            for path in images:
                picker.addItem(Path(path).name, path)
                picker.setItemData(picker.count()-1, path, Qt.ItemDataRole.ToolTipRole)
            match = automatic.get(material)
            if match:
                picker.setCurrentIndex(picker.findData(match))
                self.assignments[material] = match
            picker.currentIndexChanged.connect(lambda _, key=material, combo=picker: self.assignments.update({key: combo.currentData()}))
            self.table.setCellWidget(row, 1, picker)

    def _update_external(self):
        external = self.texture_mode.currentIndex() == 1
        self.external.setVisible(external)
        self.texture_hint.setText('Select base-colour images or an MTL file. Automatic matches can be changed below.' if external else
            'Embedded textures are used when available. Otherwise the model’s material colour or a default material is used.')

    def accept(self):
        if self.meshes and not self.worker.isRunning():
            super().accept()

    def reject(self):
        if self.worker.isRunning():
            self._cancel_requested = True
            self.status.setText('Cancelling import…')
            self.import_button.setEnabled(False)
            self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setEnabled(False)
        else:
            super().reject()

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)
