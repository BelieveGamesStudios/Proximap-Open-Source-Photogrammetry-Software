"""Native, persistent docking workspace for the mesh editor."""

from PySide6.QtCore import QByteArray, QSettings, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QDockWidget, QLabel, QMainWindow, QScrollArea,
    QGridLayout, QHBoxLayout, QSizePolicy, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)


EDITOR_STYLE = """
QWidget { background-color: #282828; color: #cccccc; font-size: 11px; }
QLabel { color: #cccccc; background: transparent; border: none; }
QMainWindow { background-color: #202020; }
QMainWindow::separator { background: #191919; width: 5px; height: 5px; }
QMainWindow::separator:hover { background: #5b7568; }
QDockWidget { color: #cccccc; font-weight: normal; border: none; }
QDockWidget::title { background: #303030; padding: 6px 8px; text-align: left; }
QWidget#DockTitleBar { background: #303030; }
QWidget#DockTitleBar QToolButton { background: transparent; border: none; padding: 0 2px; }
QWidget#DockTitleBar QToolButton:hover { background: #484848; }
QMenuBar { background: #242424; color: #cccccc; padding: 2px; border-bottom: 1px solid #191919; }
QMenuBar::item { padding: 4px 10px; background: transparent; }
QMenuBar::item:selected { background: #414141; }
QMenu { background: #303030; color: #cccccc; border: 1px solid #161616; padding: 4px; }
QMenu::item { padding: 5px 22px; }
QMenu::item:selected { background: #46554d; }
QMenu::separator { height: 1px; background: #444444; margin: 4px; }
QPushButton, QToolButton { background: #383838; color: #cccccc; border: 1px solid #202020;
    border-radius: 3px; padding: 4px 7px; font-size: 11px; font-weight: normal; }
QPushButton:hover, QToolButton:hover { background: #484848; border-color: #666666; }
QPushButton:pressed { background: #252525; }
QPushButton:checked, QToolButton:checked { background: #375c48; color: #a8ebc5; border-color: #508b68; }
QPushButton:disabled { color: #777777; background: #303030; border-color: #252525; }
QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox { background: #3b3b3b; color: #cccccc;
    border: 1px solid #202020; border-radius: 3px; padding: 4px 5px;
    selection-background-color: #456c54; selection-color: #eeeeee; }
QDoubleSpinBox:focus, QLineEdit:focus { border-color: #70a584; }
QDoubleSpinBox:disabled { color: #888888; background: #333333; }
QTreeWidget { background: #252525; color: #cccccc; border: none; outline: none; padding: 2px; }
QTreeWidget::item { padding: 5px 6px; border: none; }
QTreeWidget::item:selected { background: #365644; color: #d5efdf; }
QTreeWidget::item:hover:!selected { background: #343434; }
QCheckBox { color: #cccccc; spacing: 6px; }
QCheckBox::indicator { width: 12px; height: 12px; border: 1px solid #707070; border-radius: 2px; background: #383838; }
QCheckBox::indicator:checked { background: #609a76; border-color: #8cb69c; }
QScrollArea { border: none; background: #282828; }
QScrollBar:vertical { background: #252525; width: 10px; margin: 0; }
QScrollBar:horizontal { background: #252525; height: 10px; margin: 0; }
QScrollBar::handle { background: #505050; border-radius: 4px; min-height: 20px; min-width: 20px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QTabBar::tab { background: #292929; color: #bdbdbd; padding: 5px 12px; border: 1px solid #202020; }
QTabBar::tab:selected { background: #3b3b3b; color: #dddddd; border-bottom: 2px solid #68977a; }
QTableWidget { background: #252525; color: #cccccc; gridline-color: #414141; border: 1px solid #414141; }
QHeaderView::section { background: #333333; color: #cccccc; border: none; padding: 5px; }
QToolTip { background: #202020; color: #dddddd; border: 1px solid #555555; padding: 4px; }
"""


class EditorHeader(QWidget):
    """Menus on the left with tools centered across the whole workspace."""

    def __init__(self, menu_bar, tools, parent=None):
        super().__init__(parent)
        self.setObjectName("EditorHeader")
        self.menu_bar = menu_bar
        self.tools = tools
        layout = QGridLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(6)
        # Equal side columns keep the tools centered independently of menu width.
        menu_width = menu_bar.sizeHint().width()
        for column in (0, 2):
            layout.setColumnMinimumWidth(column, menu_width)
            layout.setColumnStretch(column, 1)
        menu_bar.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        layout.addWidget(menu_bar, 0, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(tools, 0, 1, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(QWidget(), 0, 2)


class DockTitleBar(QWidget):
    """Use compositor-managed movement for floating docks on Wayland.

    QDockWidget's QWidget mouse-grab/move path is retained for docking and
    other platforms. Floating Wayland windows must use startSystemMove from
    the original press event, while its input serial is still valid.
    """

    def __init__(self, dock, compact=False):
        super().__init__(dock)
        self.dock = dock
        self.setObjectName("DockTitleBar")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4 if compact else 8, 3, 2, 3)
        layout.setSpacing(3)
        self.label = QLabel(dock.windowTitle())
        self.label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(self.label, 1)
        self.dock_button = QToolButton(self)
        self.dock_button.setAutoRaise(True)
        self.dock_button.clicked.connect(self._toggle_floating)
        layout.addWidget(self.dock_button)
        self.close_button = QToolButton(self)
        self.close_button.setText("×")
        self.close_button.setToolTip("Hide panel (reopen from View → Panels)")
        self.close_button.clicked.connect(dock.close)
        layout.addWidget(self.close_button)
        self._compact = compact
        dock.topLevelChanged.connect(self._update_state)
        dock.windowTitleChanged.connect(self.label.setText)
        self._update_state(dock.isFloating())

    def _update_state(self, floating):
        self.dock_button.setText("↙" if floating else "↗")
        self.dock_button.setToolTip("Dock panel" if floating else "Float panel")
        self.dock_button.setVisible(floating or not self._compact)
        self.setToolTip("Drag to move; double-click to dock" if floating
                        else "Drag to dock; double-click to float")

    def _toggle_floating(self):
        self.dock.setFloating(not self.dock.isFloating())

    def _uses_system_move(self):
        return self.dock.isFloating() and QApplication.platformName().startswith("wayland")

    def mousePressEvent(self, event):
        if self._uses_system_move() and event.button() == Qt.MouseButton.LeftButton:
            handle = self.dock.windowHandle()
            if handle is not None:
                handle.startSystemMove()
            # Do not fall back to QDockWidget's unsupported mouse grab even
            # if the compositor rejects the move (e.g. a synthetic input).
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if self._uses_system_move() and event.buttons() & Qt.MouseButton.LeftButton:
            event.accept()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event):
        if self._uses_system_move() and event.button() == Qt.MouseButton.LeftButton:
            event.accept()
        else:
            event.ignore()

    def mouseDoubleClickEvent(self, event):
        if self._uses_system_move() and event.button() == Qt.MouseButton.LeftButton:
            self.dock.setFloating(False)
            event.accept()
        else:
            event.ignore()


class MeshEditorWorkspace(QMainWindow):
    """Keep the GL viewport central while ordinary panels can float and tab."""

    LAYOUT_VERSION = 2
    STATE_KEY = "meshEditor/workspaceState"

    def __init__(self, parent=None, settings=None):
        super().__init__(parent, Qt.WindowType.Widget)
        # QMainWindow's constructor adds Window even when Widget is passed.
        # Explicitly embed it in the Mesh Editor tab after construction.
        self.setWindowFlags(Qt.WindowType.Widget)
        self.setObjectName("MeshEditorWorkspace")
        self.settings = settings if settings is not None else QSettings("ProximaXR", "Proximap")
        self.docks = {}
        self.setDockOptions(
            QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
            | QMainWindow.DockOption.AnimatedDocks
        )
        self.setTabPosition(Qt.DockWidgetArea.AllDockWidgetAreas, QTabWidget.TabPosition.South)
        self._ready = False
        self._sized = False
        QApplication.instance().aboutToQuit.connect(self.save_layout)

    def _dock(self, name, title, content, minimum_width=180):
        dock = QDockWidget(title, self)
        dock.setObjectName(f"meshEditor.{name}")
        dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        dock.setMinimumWidth(minimum_width)
        dock.setWidget(content)
        dock.setTitleBarWidget(DockTitleBar(dock, compact=name == "tools"))
        self.docks[name] = dock
        return dock

    @staticmethod
    def _scroll(content):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(content)
        return scroll

    def add_panels(self, outliner, properties, addons):
        self._dock("outliner", "Outliner", outliner, 300)
        self._dock("properties", "Properties", self._scroll(properties), 300)
        addon_content = QWidget()
        addon_layout = QVBoxLayout(addon_content)
        addon_layout.setContentsMargins(4, 4, 4, 4)
        hint = QLabel("Manage mesh editor add-ons in Edit → Preferences.")
        hint.setWordWrap(True)
        addon_layout.addWidget(hint)
        addon_layout.addWidget(addons)
        addon_layout.addStretch()
        self._dock("addons", "Add-ons", self._scroll(addon_content), 260)
        self.reset_layout(save=False)
        self._restored = self.restore_layout()
        self._ready = True

    def populate_view_menu(self, menu):
        # Put workspace commands before the existing viewport settings widget.
        first = menu.actions()[0] if menu.actions() else None
        panels = menu.addMenu("Panels")
        menu.insertMenu(first, panels)
        for dock in self.docks.values():
            panels.addAction(dock.toggleViewAction())
        reset = menu.addAction("Reset Layout")
        reset.triggered.connect(lambda: self.reset_layout())
        menu.insertAction(first, reset)
        menu.insertSeparator(first)

    def reset_layout(self, save=True):
        """Return every panel, including floating/hidden panels, to the default."""
        for dock in self.docks.values():
            dock.setFloating(False)
            self.removeDockWidget(dock)
        outliner, properties, addons = (self.docks[key] for key in (
            "outliner", "properties", "addons"))
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, outliner)
        self.splitDockWidget(outliner, properties, Qt.Orientation.Vertical)
        self.tabifyDockWidget(properties, addons)
        for dock in self.docks.values():
            dock.show()
        properties.raise_()
        self._size_default_panels()
        if save:
            QTimer.singleShot(0, self.save_layout)

    def _size_default_panels(self):
        self.resizeDocks([self.docks['outliner']], [330], Qt.Orientation.Horizontal)
        self.resizeDocks([self.docks['outliner'], self.docks['properties']],
                         [240, 460], Qt.Orientation.Vertical)

    def save_layout(self):
        if self._ready:
            self.settings.setValue(self.STATE_KEY, self.saveState(self.LAYOUT_VERSION))
            self.settings.sync()

    def restore_layout(self):
        state = self.settings.value(self.STATE_KEY)
        if not isinstance(state, QByteArray) or state.isEmpty():
            return False
        return self.restoreState(state, self.LAYOUT_VERSION)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._sized:
            self._sized = True
            if not self._restored:
                QTimer.singleShot(0, self._size_default_panels)

    def hideEvent(self, event):
        # Tab switches persist the workspace as well as application shutdown.
        self.save_layout()
        super().hideEvent(event)
