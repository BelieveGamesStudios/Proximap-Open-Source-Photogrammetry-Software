import numpy as np
import os
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFrame, QLabel, QScrollArea,
    QListWidget, QGridLayout, QDoubleSpinBox, QCheckBox, QPushButton,
    QColorDialog, QComboBox, QFileDialog, QMessageBox, QDialog, QProgressBar,
    QMenuBar, QMenu, QWidgetAction, QSizePolicy, QInputDialog
)
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QColor, QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent, QIcon
from mesh_editor.viewport import MeshEditorViewport
from mesh_editor.scene import Object, load_mesh_file, apply_texture_to_meshes
from mesh_editor.undo import (
    UndoStack, BaseCommand, TransformCommand, SpinboxTransformCommand,
    DeleteCommand, AddObjectCommand, SceneState, SceneGraphCommand
)
from addon_manager import AddonManager
from preferences_dialog import PreferencesDialog


class MeshImportWorker(QThread):
    # Signals: finished (success, meshes, error_message)
    finished = Signal(bool, list, str)
    
    def __init__(self, file_path: str, texture_path: str = None, meshes=None, assignments=None):
        super().__init__()
        self.file_path = file_path
        self.texture_path = texture_path
        self.meshes = meshes
        self.assignments = assignments
        
    def run(self):
        try:
            # Load mesh in the worker thread (heavy CPU task)
            meshes = self.meshes if self.meshes is not None else load_mesh_file(self.file_path)
            if self.assignments is not None:
                from mesh_editor.import_options import apply_assignments
                apply_assignments(meshes, self.assignments)
            elif self.texture_path:
                apply_texture_to_meshes(meshes, self.texture_path)
            self.finished.emit(True, meshes, "")
        except Exception as e:
            import traceback
            err_details = traceback.format_exc()
            self.finished.emit(False, None, f"{str(e)}\n\n{err_details}")

class MeshExportWorker(QThread):
    # Signals: finished (success, error_message)
    finished = Signal(bool, str)
    
    def __init__(self, scene, file_path: str):
        super().__init__()
        self.scene = scene
        self.file_path = file_path
        
    def run(self):
        try:
            from mesh_editor.scene import export_scene_to_file
            export_scene_to_file(self.scene, self.file_path)
            self.finished.emit(True, "")
        except Exception as e:
            import traceback
            err_details = traceback.format_exc()
            self.finished.emit(False, f"{str(e)}\n\n{err_details}")

class MeshImportProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowTitleHint)
        self.setWindowTitle("Importing Model")
        self.setFixedSize(320, 110)
        self.setStyleSheet("""
            QDialog {
                background-color: #1A1A1A;
                border: 1px solid #333333;
                border-radius: 8px;
            }
            QLabel {
                color: #ffffff;
                font-size: 12px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(10)
        
        self.label = QLabel("Parsing geometry & processing mesh...", self)
        self.label.setStyleSheet("color: #00E676; font-weight: bold;")
        
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 0)  # Indeterminate animation
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3A3A3A;
                border-radius: 4px;
                background-color: #222222;
                height: 12px;
            }
            QProgressBar::chunk {
                background-color: #00E676;
                border-radius: 4px;
            }
        """)
        
        layout.addWidget(self.label)
        layout.addWidget(self.progress)

class CollapsibleAddonPanel(QFrame):
    """Collapsible card container for an individual add-on in the sidebar."""
    def __init__(self, addon_name: str, content_widget: QWidget, parent=None, is_expanded: bool = True, on_toggle_callback=None):
        super().__init__(parent)
        self.addon_name = addon_name
        self.content_widget = content_widget
        self.is_expanded = is_expanded
        self.on_toggle_callback = on_toggle_callback

        self.setObjectName("CollapsibleAddonPanel")
        self.setStyleSheet("""
            QFrame#CollapsibleAddonPanel {
                background-color: #121212;
                border: 1px solid #333333;
                border-radius: 6px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header button displaying toggle arrow and add-on name
        self.btn_toggle = QPushButton(self)
        self.btn_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle.clicked.connect(self._toggle_collapse)
        layout.addWidget(self.btn_toggle)

        # Body container holding the inner content widget
        self.body_container = QWidget(self)
        body_layout = QVBoxLayout(self.body_container)
        body_layout.setContentsMargins(6, 0, 6, 6)
        body_layout.setSpacing(0)
        if self.content_widget:
            body_layout.addWidget(self.content_widget)
        layout.addWidget(self.body_container)

        self._update_ui_state()

    def _toggle_collapse(self):
        self.is_expanded = not self.is_expanded
        self._update_ui_state()
        if self.on_toggle_callback:
            self.on_toggle_callback(self.is_expanded)

    def _update_ui_state(self):
        icon = "▼" if self.is_expanded else "▶"
        self.btn_toggle.setText(f"{icon}  {self.addon_name}")
        self.body_container.setVisible(self.is_expanded)

        if self.is_expanded:
            self.btn_toggle.setStyleSheet("""
                QPushButton {
                    font-weight: bold;
                    font-size: 11px;
                    color: #cccccc;
                    text-align: left;
                    background-color: #1A1A1A;
                    border: none;
                    border-top-left-radius: 5px;
                    border-top-right-radius: 5px;
                    border-bottom-left-radius: 0px;
                    border-bottom-right-radius: 0px;
                    padding: 5px 8px;
                }
                QPushButton:hover {
                    color: #eeeeee;
                    background-color: #242424;
                }
            """)
        else:
            self.btn_toggle.setStyleSheet("""
                QPushButton {
                    font-weight: bold;
                    font-size: 11px;
                    color: #cccccc;
                    text-align: left;
                    background-color: #1A1A1A;
                    border: none;
                    border-radius: 5px;
                    padding: 5px 8px;
                }
                QPushButton:hover {
                    color: #eeeeee;
                    background-color: #242424;
                }
            """)


class MeshEditorWidget(QWidget):
    """The main wrapper widget for the Mesh Editor tab."""
    selection_changed = Signal(object)  # Unified selection signal: Emits active Object (or None) whenever selection changes anywhere

    @property
    def active_object(self) -> Object | None:
        if hasattr(self, "viewport") and self.viewport and hasattr(self.viewport, "scene"):
            return self.viewport.scene.active_object
        return None

    @property
    def selected_objects(self) -> list:
        if hasattr(self, "viewport") and self.viewport and hasattr(self.viewport, "scene"):
            return self.viewport.scene.selected_objects
        return []

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("MeshEditorWidget")
        self.setAcceptDrops(True)
        
        # Initialize Undo / Redo Stack Engine (Limit 30 items)
        self.undo_stack = UndoStack(max_size=30)
        self._last_spinbox_pos = None
        self._last_spinbox_rot = None
        self._last_spinbox_scale = None
        self._addon_collapse_states = {}
        
        # Build UI layout
        self._init_ui()
        self._wire_events()
        self._populate_outliner()
        
        # Initialize Add-on Manager for Mesh Editor
        self.addon_manager = AddonManager(self)
        self.addon_manager.initialize_addons(self)
        self.addon_manager.addon_state_changed.connect(lambda *_: self.update_addon_panels())
        self.update_addon_panels()


    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.outliner_panel = QWidget(self)
        outliner_layout = QVBoxLayout(self.outliner_panel)
        outliner_layout.setContentsMargins(6, 6, 6, 6)
        outliner_layout.setSpacing(6)
        from mesh_editor.outliner import SceneOutliner
        self.outliner_list = SceneOutliner()
        self.outliner_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.outliner_list.setMinimumHeight(80)
        outliner_layout.addWidget(self.outliner_list)
        self.btn_delete_mesh = QPushButton("Delete Selected")
        self.btn_delete_mesh.setEnabled(False)
        outliner_layout.addWidget(self.btn_delete_mesh)

        self.properties_box = QFrame(self)
        properties_layout = QVBoxLayout(self.properties_box)
        properties_layout.setContentsMargins(10, 10, 10, 10)
        properties_layout.setSpacing(8)
        self.selection_hint = QLabel("Select an object to edit its transform")
        self.selection_hint.setWordWrap(True)
        properties_layout.addWidget(self.selection_hint)
        properties_layout.addWidget(QLabel("Transform"))

        def create_spinbox(min_val=-1000.0, max_val=1000.0, step=0.1):
            sb = QDoubleSpinBox()
            sb.setRange(min_val, max_val)
            sb.setSingleStep(step)
            sb.setDecimals(3)
            sb.setMinimumWidth(60)
            sb.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
            return sb

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(6)
        for column, axis in enumerate("XYZ", 1):
            label = QLabel(axis)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(label, 0, column)
            grid.setColumnStretch(column, 1)
        for row, (name, prefix, limits, default) in enumerate([
            ("Location", "pos", (-1000.0, 1000.0, 0.1), 0.0),
            ("Rotation", "rot", (-360.0, 360.0, 5.0), 0.0),
            ("Scale", "scale", (0.01, 100.0, 0.1), 1.0),
        ], 1):
            grid.addWidget(QLabel(name), row, 0)
            for column, axis in enumerate("xyz", 1):
                spinbox = create_spinbox(*limits)
                spinbox.setValue(default)
                spinbox.setToolTip(f"{name} {axis.upper()}" + (" (degrees)" if prefix == "rot" else ""))
                spinbox.setAccessibleName(f"{name} {axis.upper()}")
                setattr(self, f"sp_{prefix}_{axis}", spinbox)
                grid.addWidget(spinbox, row, column)
        properties_layout.addLayout(grid)
        properties_layout.addStretch()

        self._is_addon_section_expanded = True
        self.addon_box = QFrame(self)
        addon_layout = QVBoxLayout(self.addon_box)
        addon_layout.setContentsMargins(6, 6, 6, 6)
        self.btn_addon_toggle = QPushButton("▼  Add-on Tools")
        self.btn_addon_toggle.clicked.connect(self._toggle_addon_section)
        addon_layout.addWidget(self.btn_addon_toggle)
        self.addon_container = QWidget(self.addon_box)
        self.addon_container_layout = QVBoxLayout(self.addon_container)
        self.addon_container_layout.setContentsMargins(0, 4, 0, 0)
        self.addon_container_layout.setSpacing(6)
        addon_layout.addWidget(self.addon_container)

        # Create Settings widgets for the View dropdown menu
        self.btn_projection = QPushButton("Camera: Perspective")
        
        self.cb_snap = QCheckBox("Enable Snapping")
        
        self.cb_invert_y = QCheckBox("Invert Y Mouse")
        
        # Snapping options
        self.sp_snap_t = create_spinbox(0.01, 10.0, 0.1)
        self.sp_snap_t.setValue(0.5)
        
        self.sp_snap_r = create_spinbox(1.0, 90.0, 5.0)
        self.sp_snap_r.setValue(15.0)
        
        # Grid line controls
        self.sp_subdivs = create_spinbox(1.0, 100.0, 1.0)
        self.sp_subdivs.setValue(10.0)
        
        # Color pickers
        self.btn_color_bg = QPushButton("Background Color")
        self.btn_color_grid = QPushButton("Grid Major Color")
        
        # --- RIGHT PANEL: Viewport Container with Top Toolbar ---
        self.viewport_container = QWidget(self)
        self.viewport_container.setObjectName("ViewportContainer")
        viewport_layout = QVBoxLayout(self.viewport_container)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.setSpacing(0)
        
        # Create Menu Bar / Toolbar
        self.menu_bar = QMenuBar(self.viewport_container)
        self.menu_bar.setObjectName("ViewportMenuBar")
        self.menu_bar.setNativeMenuBar(False)
        
        # 1. File Menu
        file_menu = QMenu("File", self.menu_bar)
        
        action_import = file_menu.addAction("Import Mesh (.obj, .glb)")
        action_import.triggered.connect(self._on_import_model_clicked)
        
        action_load_recon = file_menu.addAction("Load Reconstructed Model")
        action_load_recon.triggered.connect(self._on_load_reconstructed_model_clicked)
        
        action_import_pxm = file_menu.addAction("Import .PXM")
        action_import_pxm.triggered.connect(self._on_import_pxm_clicked)
        
        file_menu.addSeparator()
        
        # Unified sub-menu for exporting consolidated scene (format-specific sub-actions)
        export_scene_menu = QMenu("Export Scene", file_menu)
        export_scene_menu.setStyleSheet(file_menu.styleSheet())
        
        self.action_export_glb = export_scene_menu.addAction("GLB (.glb)")
        self.action_export_obj = export_scene_menu.addAction("OBJ (.obj)")
        self.action_export_usdz = export_scene_menu.addAction("USDZ (.usdz)")
        
        self.action_export_glb.triggered.connect(lambda: self._on_export_scene_clicked(".glb"))
        self.action_export_obj.triggered.connect(lambda: self._on_export_scene_clicked(".obj"))
        self.action_export_usdz.triggered.connect(lambda: self._on_export_scene_clicked(".usdz"))
        
        file_menu.addMenu(export_scene_menu)
        
        file_menu.addSeparator()
        self.action_upload_proximap = file_menu.addAction("Upload to Proximap")
        
        self.menu_bar.addMenu(file_menu)
        
        # 2. Edit Menu
        edit_menu = QMenu("Edit", self.menu_bar)
        
        self.action_undo = edit_menu.addAction("Undo")
        self.action_undo.setShortcut("Ctrl+Z")
        self.action_undo.setEnabled(False)
        self.action_undo.triggered.connect(self._perform_undo)
        
        self.action_redo = edit_menu.addAction("Redo")
        self.action_redo.setShortcut("Ctrl+Y")
        self.action_redo.setEnabled(False)
        self.action_redo.triggered.connect(self._perform_redo)
        
        edit_menu.addSeparator()
        
        action_delete = edit_menu.addAction("Delete Selected")
        action_delete.triggered.connect(self._delete_selected_object)
        
        edit_menu.addSeparator()
        self.action_preferences = edit_menu.addAction("Preferences")
        self.action_preferences.triggered.connect(self._open_preferences)
        
        self.menu_bar.addMenu(edit_menu)

        
        # 3. View Menu with embedded Settings panel (dropdown style)
        view_menu = QMenu("View", self.menu_bar)
        
        settings_widget = QWidget(view_menu)
        settings_layout = QVBoxLayout(settings_widget)
        settings_layout.setContentsMargins(15, 12, 15, 12)
        settings_layout.setSpacing(10)
        
        lbl_title = QLabel("Viewport Settings", settings_widget)
        settings_layout.addWidget(lbl_title)
        
        settings_layout.addWidget(self.btn_projection)
        settings_layout.addWidget(self.cb_snap)
        settings_layout.addWidget(self.cb_invert_y)
        
        # Grid for Snap Inputs
        snap_grid = QGridLayout()
        snap_grid.setSpacing(6)
        
        lbl_tsnap = QLabel("T-Snap (m):", settings_widget)
        snap_grid.addWidget(lbl_tsnap, 0, 0)
        snap_grid.addWidget(self.sp_snap_t, 0, 1)
        
        lbl_rsnap = QLabel("R-Snap (deg):", settings_widget)
        snap_grid.addWidget(lbl_rsnap, 1, 0)
        snap_grid.addWidget(self.sp_snap_r, 1, 1)
        
        lbl_subdiv = QLabel("Grid Subdivs:", settings_widget)
        snap_grid.addWidget(lbl_subdiv, 2, 0)
        snap_grid.addWidget(self.sp_subdivs, 2, 1)
        
        settings_layout.addLayout(snap_grid)
        
        # Color pickers
        settings_layout.addWidget(self.btn_color_bg)
        settings_layout.addWidget(self.btn_color_grid)
        
        # Wrap the settings widget in a QWidgetAction
        view_settings_action = QWidgetAction(view_menu)
        view_settings_action.setDefaultWidget(settings_widget)
        view_menu.addAction(view_settings_action)
        
        self.menu_bar.addMenu(view_menu)
        
        
        # Horizontal layout for the toolbar and the viewport
        viewport_hbox = QHBoxLayout()
        viewport_hbox.setContentsMargins(0, 0, 0, 0)
        viewport_hbox.setSpacing(0)
        
        # Horizontal tools centered in the workspace header
        self.trans_toolbar = QFrame(self.viewport_container)
        self.trans_toolbar.setObjectName("ViewportToolbar")
        
        toolbar_layout = QHBoxLayout(self.trans_toolbar)
        toolbar_layout.setContentsMargins(0, 2, 0, 2)
        toolbar_layout.setSpacing(4)
        
        from PySide6.QtWidgets import QButtonGroup
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        
        # Translate Tool Button
        self.btn_tool_translate = QPushButton(self.trans_toolbar)
        self.btn_tool_translate.setCheckable(True)
        self.btn_tool_translate.setChecked(True)
        self.btn_tool_translate.setToolTip("Translate (G)")
        self.tool_group.addButton(self.btn_tool_translate)
        toolbar_layout.addWidget(self.btn_tool_translate)
        
        # Rotate Tool Button
        self.btn_tool_rotate = QPushButton(self.trans_toolbar)
        self.btn_tool_rotate.setCheckable(True)
        self.btn_tool_rotate.setToolTip("Rotate (R)")
        self.tool_group.addButton(self.btn_tool_rotate)
        toolbar_layout.addWidget(self.btn_tool_rotate)
        
        # Scale Tool Button
        self.btn_tool_scale = QPushButton(self.trans_toolbar)
        self.btn_tool_scale.setCheckable(True)
        self.btn_tool_scale.setToolTip("Scale (S)")
        self.tool_group.addButton(self.btn_tool_scale)
        toolbar_layout.addWidget(self.btn_tool_scale)

        public_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'public')
        for button, filename, name in (
            (self.btn_tool_translate, 'position.svg', 'Translate'),
            (self.btn_tool_rotate, 'rotation.svg', 'Rotate'),
            (self.btn_tool_scale, 'scale.svg', 'Scale'),
        ):
            button.setIcon(QIcon(os.path.join(public_dir, filename)))
            button.setIconSize(QSize(20, 20))
            button.setAccessibleName(name)
        
        # Separator line
        line = QFrame(self.trans_toolbar)
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #333333; margin: 4px;")
        toolbar_layout.addWidget(line)
        
        # Space Toggle Button (Defaults to Global)
        self.btn_tool_space = QPushButton("Global", self.trans_toolbar)
        self.btn_tool_space.setToolTip("Transform Space: Global (Click to toggle)")
        toolbar_layout.addWidget(self.btn_tool_space)
        
        # Separator line
        line2 = QFrame(self.trans_toolbar)
        line2.setFrameShape(QFrame.Shape.VLine)
        line2.setFrameShadow(QFrame.Shadow.Sunken)
        line2.setStyleSheet("background-color: #333333; margin: 4px;")
        toolbar_layout.addWidget(line2)

        # Projection button
        self.btn_projection_toolbar = QPushButton("Persp", self.trans_toolbar)
        self.btn_projection_toolbar.setCheckable(True)
        self.btn_projection_toolbar.setChecked(True)
        self.btn_projection_toolbar.setToolTip("Toggle Perspective/Orthographic (Numpad 5)")
        toolbar_layout.addWidget(self.btn_projection_toolbar)

        
        
        self.viewport = MeshEditorViewport(self)
        viewport_hbox.addWidget(self.viewport, stretch=1)
        viewport_layout.addLayout(viewport_hbox)
        
        from mesh_editor.workspace import EditorHeader, MeshEditorWorkspace
        self.workspace = MeshEditorWorkspace(self)
        self.editor_header = EditorHeader(self.menu_bar, self.trans_toolbar)
        self.workspace.setMenuWidget(self.editor_header)
        self.workspace.setCentralWidget(self.viewport_container)
        self.workspace.add_panels(self.outliner_panel, self.properties_box,
                                  self.addon_box)
        self.workspace.populate_view_menu(view_menu)
        layout.addWidget(self.workspace)
        # The editor has its own scoped theme, independent of reconstruction UI.
        from mesh_editor.workspace import EDITOR_STYLE
        self.setStyleSheet(EDITOR_STYLE)
        self.trans_toolbar.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        
        # Disable properties inputs initially until an object is selected
        self._set_properties_enabled(False)
        self.properties_box.setVisible(True)

    def get_outliner_widget(self):
        return self.outliner_list

    def get_properties_widget(self):
        return self.properties_box

    def get_addon_widget(self):
        return getattr(self, "addon_container", None)


    def _wire_events(self):
        # 1. Viewport Selection -> Sync Sidebar
        self.viewport.selection_changed.connect(self._on_viewport_selection_changed)
        # 2. Viewport Gizmo Edit -> Sync Spinboxes & Commit Undo
        self.viewport.transform_changed.connect(self._on_viewport_transform_changed)
        self.viewport.transform_committed.connect(self._on_viewport_transform_committed)
        self.viewport.undo_requested.connect(self._perform_undo)
        self.viewport.redo_requested.connect(self._perform_redo)
        
        # 3. Sidebar Selection -> Sync Viewport Selection
        self.outliner_list.itemSelectionChanged.connect(self._on_outliner_selection_changed)
        self.outliner_list.customContextMenuRequested.connect(self._outliner_context_menu)
        self.outliner_list.move_requested.connect(self._on_outliner_drop)
        self.viewport.context_menu_requested.connect(self._show_object_context_menu)
        
        # 4. Spinbox value changes -> Sync Viewport Object Transform & Commit Undo
        for sb in [self.sp_pos_x, self.sp_pos_y, self.sp_pos_z]:
            sb.valueChanged.connect(self._on_translation_spinbox_changed)
            sb.editingFinished.connect(self._on_translation_spinbox_committed)
        for sb in [self.sp_rot_x, self.sp_rot_y, self.sp_rot_z]:
            sb.valueChanged.connect(self._on_rotation_spinbox_changed)
            sb.editingFinished.connect(self._on_rotation_spinbox_committed)
        for sb in [self.sp_scale_x, self.sp_scale_y, self.sp_scale_z]:
            sb.valueChanged.connect(self._on_scale_spinbox_changed)
            sb.editingFinished.connect(self._on_scale_spinbox_committed)
            
        # 5. Projection Toggles
        self.btn_projection.clicked.connect(self._on_projection_clicked)
        self.btn_projection_toolbar.clicked.connect(self._on_projection_clicked)
        self.viewport.camera_changed.connect(self._on_camera_changed)
        
        # 6. Snapping Toggles and values
        self.cb_snap.stateChanged.connect(self._on_snap_toggled)
        self.cb_invert_y.stateChanged.connect(self._on_invert_y_toggled)
        self.sp_snap_t.valueChanged.connect(self._on_snap_values_changed)
        self.sp_snap_r.valueChanged.connect(self._on_snap_values_changed)
        
        # 7. Subdivisions changed
        self.sp_subdivs.valueChanged.connect(self._on_subdivs_changed)
        
        # 8. Colors changed
        self.btn_color_bg.clicked.connect(self._on_color_bg_clicked)
        self.btn_color_grid.clicked.connect(self._on_color_grid_clicked)
        
        # 9. Delete model clicked
        self.btn_delete_mesh.clicked.connect(self._delete_selected_object)
        
        # 10. Viewport key shortcuts
        self.viewport.delete_pressed.connect(self._delete_selected_object)
        
        # 11. Toolbar events
        self.btn_tool_translate.clicked.connect(self._change_tool_to_translate)
        self.btn_tool_rotate.clicked.connect(self._change_tool_to_rotate)
        self.btn_tool_scale.clicked.connect(self._change_tool_to_scale)
        self.btn_tool_space.clicked.connect(self._toggle_transform_space)
        self.viewport.tool_changed.connect(self._on_viewport_tool_changed)

    def _populate_outliner(self):
        self.outliner_list.rebuild(self.viewport.scene)

    def _set_properties_enabled(self, enabled: bool):
        obj = self.active_object
        self.selection_hint.setText(obj.name if enabled and obj else "Select an object to edit its transform")
        for sb in [
            self.sp_pos_x, self.sp_pos_y, self.sp_pos_z,
            self.sp_rot_x, self.sp_rot_y, self.sp_rot_z,
            self.sp_scale_x, self.sp_scale_y, self.sp_scale_z
        ]:
            sb.setEnabled(enabled)

    def _block_properties_signals(self, block: bool):
        for sb in [
            self.sp_pos_x, self.sp_pos_y, self.sp_pos_z,
            self.sp_rot_x, self.sp_rot_y, self.sp_rot_z,
            self.sp_scale_x, self.sp_scale_y, self.sp_scale_z
        ]:
            sb.blockSignals(block)

    # --- SLOT HANDLERS ---
    def _update_outliner_highlights(self):
        self.outliner_list.blockSignals(True)
        for obj, item in self.outliner_list.items_by_object.items():
            item.setSelected(obj in self.selected_objects)
            item.setForeground(0, QColor('#a8ebc5' if obj is self.active_object else '#cccccc'))
        self.outliner_list.blockSignals(False)

    # --- SLOT HANDLERS ---
    def _on_viewport_selection_changed(self, selected_obj: Object):
        selected_objs = self.viewport.scene.selected_objects
        self.btn_delete_mesh.setEnabled(len(selected_objs) > 0)
        
        if not selected_objs:
            self.outliner_list.blockSignals(True)
            self.outliner_list.clearSelection()
            self.outliner_list.blockSignals(False)
            self._set_properties_enabled(False)
            self.properties_box.setVisible(True)
            self._block_properties_signals(True)
            for sb in [self.sp_pos_x, self.sp_pos_y, self.sp_pos_z, self.sp_rot_x, self.sp_rot_y, self.sp_rot_z]:
                sb.setValue(0.0)
            for sb in [self.sp_scale_x, self.sp_scale_y, self.sp_scale_z]:
                sb.setValue(1.0)
            self._block_properties_signals(False)
        else:
            self._update_outliner_highlights()
            
            # Populate fields based on active selection (selected_obj)
            if selected_obj is not None:
                self._set_properties_enabled(True)
                self.properties_box.setVisible(True)
                self._sync_properties_from_object(selected_obj)
            else:
                self._set_properties_enabled(False)

        # Emit unified selection signal for top-level subscribers and add-ons
        self.selection_changed.emit(selected_obj)

    def _on_viewport_transform_changed(self, obj: Object):
        # Update inputs dynamically from gizmo dragging without loops
        if obj:
            self._sync_properties_from_object(obj)

    def _on_outliner_selection_changed(self):
        selected = [self.outliner_list.object_for(item) for item in self.outliner_list.selectedItems()]
        active = self.outliner_list.object_for(self.outliner_list.currentItem())
        self.viewport.scene.selected_objects = selected
        self.viewport.scene.active_object = active if active in selected else (selected[-1] if selected else None)
        self.viewport.selection_changed.emit(self.active_object)
        self.viewport.update()

    def _outliner_context_menu(self, point):
        item = self.outliner_list.itemAt(point)
        if item is None:
            self.outliner_list.clearSelection()
        elif not item.isSelected():
            self.outliner_list.clearSelection()
            self.outliner_list.setCurrentItem(item)
            item.setSelected(True)
        self._show_object_context_menu(self.outliner_list.viewport().mapToGlobal(point))

    def _on_outliner_drop(self, payload):
        objects, parent, before_sibling = payload
        before = SceneState(self.viewport.scene)
        try:
            changed = self.viewport.scene.move_objects(objects, parent, before_sibling)
        except (ValueError, np.linalg.LinAlgError):
            before.restore(self.viewport.scene)
            self._populate_outliner()
            return
        if changed:
            self._commit_scene_edit('Move Objects', before)
            if parent in self.outliner_list.items_by_object:
                self.outliner_list.items_by_object[parent].setExpanded(True)

    def _build_object_context_menu(self):
        menu = QMenu(self)
        selection = list(self.selected_objects)
        delete = menu.addAction('Delete')
        delete.setEnabled(bool(selection))
        delete.triggered.connect(self._delete_selected_object)
        rename = menu.addAction('Rename')
        rename.setEnabled(len(selection) == 1)
        rename.triggered.connect(self._rename_selected_object)
        menu.addSeparator()
        parent = menu.addAction('Create Parent Object')
        parent.setEnabled(bool(selection))
        parent.triggered.connect(self._create_parent_object)
        menu.addAction('Create Empty Object', self._create_empty_object)
        return menu

    def _show_object_context_menu(self, global_point):
        menu = self._build_object_context_menu()
        menu.exec(global_point)
        menu.deleteLater()

    def _unique_object_name(self, name):
        names = {obj.name for obj in self.viewport.scene.objects}
        candidate, index = name, 1
        while candidate in names:
            candidate = f'{name} ({index})'
            index += 1
        return candidate

    def _commit_scene_edit(self, label, before):
        self.push_command(SceneGraphCommand(label, before, self.viewport, self._populate_outliner))
        self._populate_outliner()
        self.viewport.selection_changed.emit(self.active_object)
        self.viewport.update()

    def _create_parent_object(self):
        if not self.selected_objects:
            return
        from mesh_editor.import_options import group_objects
        before = SceneState(self.viewport.scene)
        parent = group_objects(self.selected_objects, self._unique_object_name('Parent'))
        self.viewport.scene.add_object(parent)
        self.viewport.scene.selected_objects = [parent]
        self.viewport.scene.active_object = parent
        self._commit_scene_edit('Create Parent', before)

    def _create_empty_object(self):
        before = SceneState(self.viewport.scene)
        obj = Object(self._unique_object_name('Empty'), None)
        self.viewport.scene.add_object(obj)
        self.viewport.scene.selected_objects = [obj]
        self.viewport.scene.active_object = obj
        self._commit_scene_edit('Create Empty', before)

    def _rename_selected_object(self):
        if len(self.selected_objects) != 1:
            return
        obj = self.selected_objects[0]
        name, accepted = QInputDialog.getText(self, 'Rename Object', 'Name', text=obj.name)
        if accepted and name.strip() and name.strip() != obj.name:
            before = SceneState(self.viewport.scene)
            obj.name = self._unique_object_name(name.strip())
            self._commit_scene_edit('Rename', before)

    def push_command(self, cmd: BaseCommand):
        """Public API for registering custom commands into the Undo/Redo stack (supports addons)."""
        self.undo_stack.push(cmd)
        self._update_undo_redo_actions()

    def _perform_undo(self):
        label = self.undo_stack.undo()
        if label:
            self._populate_outliner()
            active = self.viewport.scene.active_object
            self.viewport.selection_changed.emit(active)
            self._update_undo_redo_actions()
            self.viewport.update()

    def _perform_redo(self):
        label = self.undo_stack.redo()
        if label:
            self._populate_outliner()
            active = self.viewport.scene.active_object
            self.viewport.selection_changed.emit(active)
            self._update_undo_redo_actions()
            self.viewport.update()

    def _update_undo_redo_actions(self):
        if hasattr(self, 'action_undo') and hasattr(self, 'action_redo'):
            self.action_undo.setEnabled(self.undo_stack.can_undo())
            self.action_redo.setEnabled(self.undo_stack.can_redo())
            self.action_undo.setText(f"Undo {self.undo_stack.undo_label()}" if self.undo_stack.can_undo() else "Undo")
            self.action_redo.setText(f"Redo {self.undo_stack.redo_label()}" if self.undo_stack.can_redo() else "Redo")

    def _on_viewport_transform_committed(self, payload):
        states, operation = payload
        cmd = TransformCommand(states, viewport=self.viewport, operation=operation)
        self.push_command(cmd)

    def _sync_properties_from_object(self, obj: Object):
        self._block_properties_signals(True)
        self.sp_pos_x.setValue(obj.position[0])
        self.sp_pos_y.setValue(obj.position[1])
        self.sp_pos_z.setValue(obj.position[2])
        
        self.sp_rot_x.setValue(obj.rotation[0])
        self.sp_rot_y.setValue(obj.rotation[1])
        self.sp_rot_z.setValue(obj.rotation[2])
        
        self.sp_scale_x.setValue(obj.scale[0])
        self.sp_scale_y.setValue(obj.scale[1])
        self.sp_scale_z.setValue(obj.scale[2])
        self._block_properties_signals(False)
        self._last_spinbox_pos = np.copy(obj.position)
        self._last_spinbox_rot = np.copy(obj.rotation)
        self._last_spinbox_scale = np.copy(obj.scale)

    def _on_translation_spinbox_changed(self):
        obj = self.viewport.scene.selected_object
        if obj:
            obj.position = np.array([
                self.sp_pos_x.value(),
                self.sp_pos_y.value(),
                self.sp_pos_z.value()
            ], dtype=np.float32)
            self.viewport.update()

    def _on_translation_spinbox_committed(self):
        obj = self.viewport.scene.selected_object
        if obj and self._last_spinbox_pos is not None:
            new_pos = np.array([self.sp_pos_x.value(), self.sp_pos_y.value(), self.sp_pos_z.value()], dtype=np.float32)
            if not np.array_equal(self._last_spinbox_pos, new_pos):
                cmd = SpinboxTransformCommand(obj, 'position', self._last_spinbox_pos, new_pos, self.viewport)
                self.push_command(cmd)
                self._last_spinbox_pos = np.copy(new_pos)

    def _on_rotation_spinbox_changed(self):
        obj = self.viewport.scene.selected_object
        if obj:
            obj.rotation = np.array([
                self.sp_rot_x.value(),
                self.sp_rot_y.value(),
                self.sp_rot_z.value()
            ], dtype=np.float32)
            self.viewport.update()

    def _on_rotation_spinbox_committed(self):
        obj = self.viewport.scene.selected_object
        if obj and self._last_spinbox_rot is not None:
            new_rot = np.array([self.sp_rot_x.value(), self.sp_rot_y.value(), self.sp_rot_z.value()], dtype=np.float32)
            if not np.array_equal(self._last_spinbox_rot, new_rot):
                cmd = SpinboxTransformCommand(obj, 'rotation', self._last_spinbox_rot, new_rot, self.viewport)
                self.push_command(cmd)
                self._last_spinbox_rot = np.copy(new_rot)

    def _on_scale_spinbox_changed(self):
        obj = self.viewport.scene.selected_object
        if obj:
            obj.scale = np.array([
                self.sp_scale_x.value(),
                self.sp_scale_y.value(),
                self.sp_scale_z.value()
            ], dtype=np.float32)
            self.viewport.update()

    def _on_scale_spinbox_committed(self):
        obj = self.viewport.scene.selected_object
        if obj and self._last_spinbox_scale is not None:
            new_scale = np.array([self.sp_scale_x.value(), self.sp_scale_y.value(), self.sp_scale_z.value()], dtype=np.float32)
            if not np.array_equal(self._last_spinbox_scale, new_scale):
                cmd = SpinboxTransformCommand(obj, 'scale', self._last_spinbox_scale, new_scale, self.viewport)
                self.push_command(cmd)
                self._last_spinbox_scale = np.copy(new_scale)

    def _on_projection_clicked(self):
        cam = self.viewport.camera
        cam.is_perspective = not cam.is_perspective
        self.viewport.notify_camera_changed()
        self.viewport.update()

    def _on_camera_changed(self, camera):
        # Update sidebar projection setting toggle button
        if camera.is_perspective:
            self.btn_projection.setText("Camera: Perspective")
            self.btn_projection_toolbar.setText("Persp")
            self.btn_projection_toolbar.setChecked(True)
        else:
            self.btn_projection.setText("Camera: Orthographic")
            self.btn_projection_toolbar.setText("Ortho")
            self.btn_projection_toolbar.setChecked(False)

    def _on_snap_toggled(self, state):
        self.viewport.gizmo.use_snapping = (state == Qt.CheckState.Checked.value)
        self.viewport.update()
        
    def _on_invert_y_toggled(self, state):
        self.viewport.invert_y = (state == Qt.CheckState.Checked.value)
        self.viewport.update()

    def _on_snap_values_changed(self):
        gizmo = self.viewport.gizmo
        gizmo.snap_translation = self.sp_snap_t.value()
        gizmo.snap_rotation = self.sp_snap_r.value()
        # Keep scale snapping proportional to 10% of value
        gizmo.snap_scale = 0.1
        self.viewport.update()

    def _on_subdivs_changed(self):
        self.viewport.grid_subdivisions = self.sp_subdivs.value()
        self.viewport.update()

    def _on_color_bg_clicked(self):
        from PySide6.QtGui import QColor
        c = self.viewport.bg_color
        initial = QColor.fromRgbF(c[0], c[1], c[2])
        cur_color = QColorDialog.getColor(
            initial, self, "Select Viewport Background Color"
        )
        if cur_color.isValid():
            r, g, b, _ = cur_color.getRgbF()
            self.viewport.bg_color = np.array([r, g, b], dtype=np.float32)
            self.viewport.update()

    def _on_color_grid_clicked(self):
        from PySide6.QtGui import QColor
        c = self.viewport.grid_color_1
        initial = QColor.fromRgbF(c[0], c[1], c[2], c[3])
        cur_color = QColorDialog.getColor(
            initial, self, "Select Major Grid Line Color"
        )
        if cur_color.isValid():
            r, g, b, a = cur_color.getRgbF()
            self.viewport.grid_color_1 = np.array([r, g, b, a], dtype=np.float32)
            self.viewport.update()

    def _import_mesh_from_path(self, file_path: str, texture_path=None):
        if not file_path or not os.path.exists(file_path):
            QMessageBox.critical(self, 'Import Error', f'Model file not found:\n{file_path}')
            return
        from mesh_editor.import_dialog import ModelImportDialog
        paths = ([texture_path] if isinstance(texture_path, str) else texture_path) or []
        options = ModelImportDialog(file_path, self, paths)
        if options.exec() != QDialog.DialogCode.Accepted:
            options.deleteLater()
            return
        self._import_group = options.group.isChecked()
        assignments = dict(options.assignments) if options.texture_mode.currentIndex() == 1 else None
        self.import_dialog = MeshImportProgressDialog(self)
        self.import_worker = MeshImportWorker(file_path, meshes=options.meshes, assignments=assignments)
        self.import_worker.finished.connect(self._on_import_finished)
        options.deleteLater()
        self.import_worker.start()
        self.import_dialog.exec()

    def _on_import_model_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Import 3D Model', '',
            '3D Mesh Files (*.obj *.glb *.gltf *.ply)')
        if file_path:
            self._import_mesh_from_path(file_path)

    def _on_load_reconstructed_model_clicked(self):
        # Traverse up parent to find active mvs_dir
        mvs_dir = None
        curr = self.parent()
        while curr is not None:
            if hasattr(curr, '_get_active_mvs_dir'):
                mvs_dir = curr._get_active_mvs_dir()
                break
            curr = curr.parent()
            
        if not mvs_dir:
            try:
                from main_window import get_reconstruction_out_dir
                mvs_dir = os.path.join(get_reconstruction_out_dir(), "mvs")
            except Exception:
                local_appdata = os.environ.get("LOCALAPPDATA")
                if local_appdata:
                    mvs_dir = os.path.join(local_appdata, "Proximap", "reconstruction_out", "mvs")
                else:
                    mvs_dir = os.path.join(os.path.expanduser("~"), ".proximap", "reconstruction_out", "mvs")
                    
        # Check candidates
        found_path = None
        for candidate in [
            "scene_dense_mesh_texture.obj",
            "scene_dense_mesh_texture.glb",
            "scene_dense_mesh_texture.ply",
            "scene_dense_mesh_refine.ply",
            "scene_dense_mesh.ply",
            "scene_mesh.ply"
        ]:
            path = os.path.join(mvs_dir, candidate)
            if os.path.exists(path):
                found_path = path
                break
                
        if found_path:
            self._import_mesh_from_path(found_path)
        else:
            QMessageBox.warning(
                self, 
                "Reconstruction Model Not Found", 
                f"No reconstructed model could be found in the current session folder:\n{mvs_dir}\n\nPlease run the dense reconstruction/texturing step first."
            )

    def _on_import_pxm_clicked(self, file_path: str = None):
        if not isinstance(file_path, str):
            file_path = None
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Import Proximap Project", "", "Proximap Project Files (*.pxm)"
            )
            if not file_path:
                return
            
        import zipfile
        import tempfile
        import uuid
        
        try:
            # Create a temporary extraction directory inside temp root
            temp_dir = os.path.join(tempfile.gettempdir(), f"proximap_pxm_import_{uuid.uuid4().hex}")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Extract zip contents
            with zipfile.ZipFile(file_path, 'r') as zipf:
                zipf.extractall(temp_dir)
                
            # Scan for candidate 3D mesh files
            found_mesh_path = None
            standard_obj = os.path.join(temp_dir, "scene_dense_mesh_texture.obj")
            if os.path.exists(standard_obj):
                found_mesh_path = standard_obj
            else:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if file.lower().endswith(('.obj', '.glb', '.gltf', '.ply')):
                            found_mesh_path = os.path.join(root, file)
                            break
                    if found_mesh_path:
                        break
                        
            if found_mesh_path:
                self._import_mesh_from_path(found_mesh_path)
            else:
                QMessageBox.warning(
                    self,
                    "Import PXM Error",
                    "The selected .pxm file was successfully unpacked, but no 3D model files (.obj, .glb, etc.) were found inside."
                )
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "Import PXM Error",
                f"Failed to extract and import .pxm file:\n{str(e)}"
            )

    def _on_import_finished(self, success: bool, meshes: list, error_msg: str):
        # 1. Close modal dialog
        if hasattr(self, 'import_dialog') and self.import_dialog:
            self.import_dialog.accept()
            self.import_dialog = None
            
        if not success:
            QMessageBox.critical(
                self, "Import Error", f"Failed to load 3D model:\n{error_msg}"
            )
            self.import_worker = None
            return
            
        try:
            # 2. Setup OpenGL buffers on the main thread with context active
            self.viewport.makeCurrent()
            file_path = self.import_worker.file_path
            from mesh_editor.import_options import build_import_objects
            created_objects, active = build_import_objects(meshes, file_path,
                group=getattr(self, '_import_group', True),
                existing_names=[obj.name for obj in self.viewport.scene.objects])
            try:
                for obj in created_objects:
                    if obj.mesh is not None:
                        obj.mesh.setup_buffers()
            finally:
                self.viewport.doneCurrent()
            cmd = AddObjectCommand(created_objects, self.viewport, on_scene_changed_cb=self._populate_outliner,
                                   selection=[active] if active is not None else [])
            cmd.execute()
            if active is not None:
                self.viewport.scene.selected_objects = [active]
                self.viewport.scene.active_object = active
            self._populate_outliner()
            self.viewport.selection_changed.emit(active)
            self.push_command(cmd)

            # 5. Update rendering
            self.viewport.update()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self, "Import Error", f"Failed to configure GL buffers:\n{str(e)}"
            )
            
        self.import_worker = None

    def _delete_selected_object(self):
        selected_objs = list(self.viewport.scene.selected_objects)
        if selected_objs:
            cmd = DeleteCommand(selected_objs, self.viewport, on_scene_changed_cb=self._populate_outliner)
            cmd.execute()
            self.push_command(cmd)

    def _on_export_scene_clicked(self, fmt=None):
        if not self.viewport.scene.objects:
            QMessageBox.warning(
                self, "Export Warning", "There are no objects in the scene to export."
            )
            return
            
        filter_str = "3D Mesh Files (*.obj *.glb *.usdz)"
        default_suffix = ""
        if fmt == ".glb":
            filter_str = "GLB Files (*.glb)"
            default_suffix = ".glb"
        elif fmt == ".obj":
            filter_str = "OBJ Files (*.obj)"
            default_suffix = ".obj"
        elif fmt == ".usdz":
            filter_str = "USDZ Files (*.usdz)"
            default_suffix = ".usdz"
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Scene", "", filter_str
        )
        if not file_path:
            return
            
        if default_suffix and not file_path.lower().endswith(default_suffix):
            file_path += default_suffix
            
        # Create and configure the waiting dialog
        self.export_dialog = QDialog(self, Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowTitleHint)
        self.export_dialog.setWindowTitle("Exporting Scene")
        self.export_dialog.setMinimumSize(280, 100)
        self.export_dialog.setModal(True)
        
        dialog_layout = QVBoxLayout(self.export_dialog)
        dialog_layout.setContentsMargins(20, 20, 20, 20)
        dialog_layout.setSpacing(10)
        
        lbl_msg = QLabel("Merging and exporting scene objects...\nPlease wait.", self.export_dialog)
        lbl_msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_msg.setStyleSheet("color: #ffffff; font-size: 11px;")
        dialog_layout.addWidget(lbl_msg)
        
        progress_bar = QProgressBar(self.export_dialog)
        progress_bar.setRange(0, 0)  # Infinite progress bar
        dialog_layout.addWidget(progress_bar)
        
        # Start export thread
        self.export_worker = MeshExportWorker(self.viewport.scene, file_path)
        self.export_worker.finished.connect(self._on_export_finished)
        self.export_worker.start()
        
        self.export_dialog.exec()

    def _on_export_finished(self, success: bool, error_msg: str):
        if hasattr(self, "export_dialog") and self.export_dialog:
            self.export_dialog.accept()
            self.export_dialog = None
            
        if success:
            QMessageBox.information(
                self, "Export Success", "Successfully consolidated and exported the scene!"
            )
        else:
            QMessageBox.critical(
                self, "Export Error", f"Consolidation/export failed:\n{error_msg}"
            )
            
        self.export_worker = None

    def _change_tool_to_translate(self):
        self.viewport.gizmo.operation = "translate"
        self.btn_tool_translate.setChecked(True)
        self.viewport.update()

    def _change_tool_to_rotate(self):
        self.viewport.gizmo.operation = "rotate"
        self.btn_tool_rotate.setChecked(True)
        self.viewport.update()

    def _change_tool_to_scale(self):
        self.viewport.gizmo.operation = "scale"
        self.btn_tool_scale.setChecked(True)
        self.viewport.update()

    def _on_viewport_tool_changed(self, operation):
        if operation == "translate":
            self.btn_tool_translate.setChecked(True)
        elif operation == "rotate":
            self.btn_tool_rotate.setChecked(True)
        elif operation == "scale":
            self.btn_tool_scale.setChecked(True)

    def _toggle_transform_space(self):
        gizmo = self.viewport.gizmo
        if gizmo.space == "local":
            gizmo.space = "global"
            self.btn_tool_space.setText("Global")
            self.btn_tool_space.setToolTip("Transform Space: Global (Click to toggle)")
        else:
            gizmo.space = "local"
            self.btn_tool_space.setText("Local")
            self.btn_tool_space.setToolTip("Transform Space: Local (Click to toggle)")
        self.viewport.update()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            # Check if there's at least one supported file
            has_supported = False
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                ext = os.path.splitext(path)[1].lower()
                if ext in ('.obj', '.glb', '.gltf', '.ply', '.pxm'):
                    has_supported = True
                    break
            if has_supported:
                self.viewport_container.setStyleSheet(
                    "QWidget#ViewportContainer { border: 2px dashed #00E676; background-color: #1A2820; }"
                )
                event.acceptProposedAction()
                return
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent):
        self.viewport_container.setStyleSheet("")

    def dropEvent(self, event: QDropEvent):
        self.viewport_container.setStyleSheet("")
        
        mesh_files = []
        pxm_files = []
        texture_files = []
        
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if not path or not os.path.exists(path):
                continue
            ext = os.path.splitext(path)[1].lower()
            if ext in ('.obj', '.glb', '.gltf', '.ply'):
                mesh_files.append(os.path.normpath(path))
            elif ext == '.pxm':
                pxm_files.append(os.path.normpath(path))
            elif ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tga', '.mtl'):
                texture_files.append(os.path.normpath(path))
                
        # Handle project file drop
        if pxm_files:
            # We process the first pxm file dropped
            self._on_import_pxm_clicked(pxm_files[0])
            event.acceptProposedAction()
            return
            
        # Handle mesh files drop
        if mesh_files:
            for mesh_file in mesh_files:
                self._import_mesh_from_path(mesh_file, texture_files)
            event.acceptProposedAction()
        else:
            event.ignore()

    def _open_preferences(self):
        main_win = getattr(self, "main_window", self)
        dlg = PreferencesDialog(self.addon_manager, main_win, self)
        dlg.exec()

    def _toggle_addon_section(self):
        self._is_addon_section_expanded = not getattr(self, "_is_addon_section_expanded", True)
        self.addon_container.setVisible(self._is_addon_section_expanded)
        if self._is_addon_section_expanded:
            self.btn_addon_toggle.setText("▼  Add-on Tools")
        else:
            self.btn_addon_toggle.setText("▶  Add-on Tools")

    def update_addon_panels(self):
        if not hasattr(self, "addon_container_layout"):
            return

        # Clear container layout
        while self.addon_container_layout.count():
            item = self.addon_container_layout.takeAt(0)
            if item and item.widget():
                item.widget().setParent(None)

        if not hasattr(self, "addon_manager"):
            return

        if not hasattr(self, "_addon_collapse_states"):
            self._addon_collapse_states = {}

        # Get active add-on panel widgets
        active_instances = self.addon_manager.get_enabled_instances()
        panel_added = False
        for addon_inst in active_instances:
            addon_id = getattr(addon_inst, "addon_id", "unknown")
            addon_name = getattr(addon_inst, "addon_name", "Add-on")
            is_expanded = self._addon_collapse_states.get(addon_id, True)

            w = addon_inst.get_panel_widget(self.addon_container)
            if w is None:
                # Add-on is active but doesn't supply a custom widget -> show active badge
                w = QFrame(self.addon_container)
                w.setStyleSheet("""
                    QFrame {
                        background-color: transparent;
                        border: none;
                        padding: 4px;
                    }
                    QLabel {
                        color: #e0e0e0;
                        font-size: 11px;
                    }
                """)
                b_layout = QHBoxLayout(w)
                b_layout.setContentsMargins(4, 4, 4, 4)
                lbl_name = QLabel(f"✓ {addon_name} <span style='color: #00E676;'>(Active)</span>", w)
                lbl_name.setTextFormat(Qt.TextFormat.RichText)
                b_layout.addWidget(lbl_name)

            def make_cb(aid):
                return lambda exp: self._addon_collapse_states.update({aid: exp})

            collapsible_card = CollapsibleAddonPanel(
                addon_name=addon_name,
                content_widget=w,
                parent=self.addon_container,
                is_expanded=is_expanded,
                on_toggle_callback=make_cb(addon_id)
            )
            self.addon_container_layout.addWidget(collapsible_card)
            panel_added = True

        if hasattr(self, "addon_box"):
            self.addon_box.setVisible(panel_added)

