"""Dock layout persistence and existing editor behavior after panel migration."""
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QByteArray, QEvent, QPointF, QSettings, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QMenu, QMenuBar, QPushButton, QWidget

from mesh_editor.workspace import EditorHeader, MeshEditorWorkspace

app = QApplication.instance() or QApplication([])


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = QSettings(os.path.join(self.temp_dir.name, "layout.ini"), QSettings.Format.IniFormat)
        self.windows = []

    def tearDown(self):
        for window in self.windows:
            window.close()
            window.deleteLater()
        app.processEvents()
        self.temp_dir.cleanup()

    def workspace(self):
        window = MeshEditorWorkspace(settings=self.settings)
        window.setCentralWidget(QLabel("Viewport"))
        window.resize(1280, 800)
        window.add_panels(*(QWidget() for _ in range(3)))
        window.show()
        app.processEvents()
        self.windows.append(window)
        return window

    def test_default_layout_and_panel_menu(self):
        window = self.workspace()
        self.assertNotIn("tools", window.docks)
        self.assertEqual(window.dockWidgetArea(window.docks['outliner']), Qt.DockWidgetArea.RightDockWidgetArea)
        self.assertIn(window.docks['addons'], window.tabifiedDockWidgets(window.docks['properties']))
        menu = QMenu()
        menu.addAction("Viewport settings")
        window.populate_view_menu(menu)
        panels = menu.actions()[0].menu()
        self.assertEqual(len(panels.actions()), 3)
        panels.actions()[0].trigger()
        self.assertTrue(window.docks['outliner'].isHidden())
        panels.actions()[0].trigger()
        self.assertFalse(window.docks['outliner'].isHidden())
        self.assertGreater(window.centralWidget().width(), 600)

    def test_layout_roundtrip_includes_floating_hidden_and_resized_panels(self):
        window = self.workspace()
        window.docks['addons'].hide()
        window.docks['outliner'].setFloating(True)
        window.docks['outliner'].resize(410, 330)
        window.resizeDocks([window.docks['properties']], [390], Qt.Orientation.Horizontal)
        app.processEvents()
        properties_width = window.docks['properties'].width()
        window.save_layout()
        restored = self.workspace()
        self.assertTrue(restored.docks['addons'].isHidden())
        self.assertTrue(restored.docks['outliner'].isFloating())
        self.assertEqual(restored.docks['outliner'].size(), window.docks['outliner'].size())
        self.assertAlmostEqual(restored.docks['properties'].width(), properties_width, delta=5)

    def test_layout_roundtrip_preserves_tabbed_panels(self):
        window = self.workspace()
        window.save_layout()
        restored = self.workspace()
        self.assertIn(restored.docks['addons'], restored.tabifiedDockWidgets(restored.docks['properties']))

    def test_reset_recovers_hidden_and_floating_panels(self):
        window = self.workspace()
        window.docks['properties'].setFloating(True)
        window.docks['outliner'].hide()
        window.reset_layout()
        app.processEvents()
        for dock in window.docks.values():
            self.assertFalse(dock.isFloating())
            self.assertFalse(dock.isHidden())
        self.assertEqual(window.dockWidgetArea(window.docks['properties']), Qt.DockWidgetArea.RightDockWidgetArea)
        restored = self.workspace()
        self.assertFalse(restored.docks['properties'].isFloating())
        self.assertFalse(restored.docks['outliner'].isHidden())

    @staticmethod
    def mouse_event(kind, button=Qt.MouseButton.LeftButton):
        return QMouseEvent(kind, QPointF(15, 10), QPointF(115, 110),
                           button, button, Qt.KeyboardModifier.NoModifier)

    def test_floating_wayland_headers_use_system_move_and_consume_drag_events(self):
        window = self.workspace()
        for name, dock in window.docks.items():
            with self.subTest(panel=name):
                dock.setFloating(True)
                app.processEvents()
                bar = dock.titleBarWidget()
                handle = Mock()
                handle.startSystemMove.return_value = True
                press = self.mouse_event(QEvent.Type.MouseButtonPress)
                with patch('mesh_editor.workspace.QApplication.platformName', return_value='wayland'), \
                     patch.object(dock, 'windowHandle', return_value=handle):
                    QApplication.sendEvent(bar, press)
                    handle.startSystemMove.assert_called_once_with()
                    self.assertTrue(press.isAccepted())
                    for kind in (QEvent.Type.MouseMove, QEvent.Type.MouseButtonRelease):
                        event = self.mouse_event(kind)
                        QApplication.sendEvent(bar, event)
                        self.assertTrue(event.isAccepted())
                    handle.startSystemMove.assert_called_once_with()
                dock.setFloating(False)

    def test_wayland_rejected_move_does_not_fall_back_to_mouse_grab(self):
        window = self.workspace()
        dock = window.docks['properties']
        dock.setFloating(True)
        handle = Mock()
        handle.startSystemMove.return_value = False
        event = self.mouse_event(QEvent.Type.MouseButtonPress)
        with patch('mesh_editor.workspace.QApplication.platformName', return_value='wayland-egl'), \
             patch.object(dock, 'windowHandle', return_value=handle):
            dock.titleBarWidget().mousePressEvent(event)
        self.assertTrue(event.isAccepted())
        handle.startSystemMove.assert_called_once_with()

    def test_normal_docking_and_other_platforms_keep_qt_drag_handling(self):
        window = self.workspace()
        dock = window.docks['properties']
        for floating, platform in ((False, 'wayland'), (True, 'xcb'), (True, 'windows')):
            dock.setFloating(floating)
            bar = dock.titleBarWidget()
            with patch('mesh_editor.workspace.QApplication.platformName', return_value=platform), \
                 patch.object(dock, 'windowHandle') as handle:
                for kind, handler in ((QEvent.Type.MouseButtonPress, bar.mousePressEvent),
                                      (QEvent.Type.MouseMove, bar.mouseMoveEvent),
                                      (QEvent.Type.MouseButtonRelease, bar.mouseReleaseEvent),
                                      (QEvent.Type.MouseButtonDblClick, bar.mouseDoubleClickEvent)):
                    event = self.mouse_event(kind)
                    handler(event)
                    self.assertFalse(event.isAccepted())
                handle.assert_not_called()

    def test_wayland_float_dock_close_buttons_and_double_click(self):
        window = self.workspace()
        dock = window.docks['properties']
        bar = dock.titleBarWidget()
        bar.dock_button.click()
        self.assertTrue(dock.isFloating())
        self.assertEqual(bar.dock_button.toolTip(), 'Dock panel')
        with patch('mesh_editor.workspace.QApplication.platformName', return_value='wayland'):
            event = self.mouse_event(QEvent.Type.MouseButtonDblClick)
            bar.mouseDoubleClickEvent(event)
            self.assertTrue(event.isAccepted())
        self.assertFalse(dock.isFloating())
        bar.dock_button.click()
        bar.dock_button.click()
        self.assertFalse(dock.isFloating())
        bar.close_button.click()
        self.assertTrue(dock.isHidden())
        dock.toggleViewAction().trigger()
        self.assertFalse(dock.isHidden())

    def test_header_tools_stay_centered_when_resized(self):
        menu = QMenuBar()
        for title in ('File', 'Edit', 'View'):
            menu.addMenu(title)
        tools = QWidget()
        layout = QHBoxLayout(tools)
        for title in ('Move', 'Rotate', 'Scale', 'Global', 'Persp'):
            layout.addWidget(QPushButton(title))
        header = EditorHeader(menu, tools)
        for width in (800, 1280, 1920):
            header.resize(width, 40)
            header.show()
            app.processEvents()
            self.assertAlmostEqual(tools.geometry().center().x(), header.rect().center().x(), delta=1)
            self.assertLess(menu.geometry().right(), tools.geometry().left())
        header.close()
        header.deleteLater()

    def test_old_sidebar_layout_is_replaced_without_a_tools_dock(self):
        old = self.workspace()
        self.settings.setValue(MeshEditorWorkspace.STATE_KEY, old.saveState(1))
        window = self.workspace()
        self.assertFalse(window._restored)
        self.assertNotIn('tools', window.docks)
        self.assertEqual(window.dockWidgetArea(window.docks['properties']), Qt.DockWidgetArea.RightDockWidgetArea)

    def test_invalid_saved_layout_falls_back_to_default(self):
        self.settings.setValue(MeshEditorWorkspace.STATE_KEY, QByteArray(b"invalid layout"))
        window = self.workspace()
        self.assertFalse(window._restored)
        self.assertFalse(window.docks['properties'].isHidden())


class EditorBehaviorTests(unittest.TestCase):
    def test_selection_transform_undo_and_toolbar_still_work(self):
        from mesh_editor import MeshEditorWidget
        from mesh_editor.scene import Object
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = QSettings(os.path.join(temp_dir, "editor.ini"), QSettings.Format.IniFormat)
            with patch('mesh_editor.workspace.QSettings', return_value=settings), \
                 patch('addon_manager.AddonManager._load_prefs'), \
                 patch('addon_manager.AddonManager.initialize_addons'):
                editor = MeshEditorWidget()
            self.assertFalse(editor.workspace.isWindow())
            self.assertIs(editor.workspace.menuWidget(), editor.editor_header)
            self.assertIs(editor.trans_toolbar.parentWidget(), editor.editor_header)
            self.assertNotIn('tools', editor.workspace.docks)
            self.assertIs(editor.workspace.parentWidget(), editor)
            self.assertIs(editor.layout().itemAt(0).widget(), editor.workspace)
            obj = Object("Test mesh", None)
            editor.viewport.scene.objects.append(obj)
            editor.viewport.scene.selected_object = obj
            editor._populate_outliner()
            editor._on_viewport_selection_changed(obj)
            self.assertTrue(editor.sp_pos_x.isEnabled())
            editor.sp_pos_x.setValue(2.5)
            editor.sp_pos_x.editingFinished.emit()
            np.testing.assert_allclose(obj.position, [2.5, 0, 0])
            editor._perform_undo()
            np.testing.assert_allclose(obj.position, [0, 0, 0])
            editor._perform_redo()
            np.testing.assert_allclose(obj.position, [2.5, 0, 0])
            editor.btn_tool_rotate.click()
            self.assertEqual(editor.viewport.gizmo.operation, "rotate")
            editor.btn_projection_toolbar.click()
            self.assertFalse(editor.viewport.camera.is_perspective)
            editor.btn_tool_space.click()
            self.assertEqual(editor.viewport.gizmo.space, "local")
            editor.btn_tool_space.click()
            self.assertEqual(editor.viewport.gizmo.space, "global")
            editor.viewport.scene.selected_object = None
            editor._on_viewport_selection_changed(None)
            self.assertFalse(editor.sp_pos_x.isEnabled())
            self.assertFalse(editor.properties_box.isHidden())
            editor.deleteLater()
            app.processEvents()


if __name__ == '__main__':
    unittest.main()
