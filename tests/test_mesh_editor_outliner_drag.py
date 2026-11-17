"""Scene moves and real Qt drop-position routing (without a native desktop drag)."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import QApplication, QWidget

from mesh_editor import MeshEditorWidget
from mesh_editor.scene import Object, Scene, create_cube_mesh
from mesh_editor.outliner import SceneOutliner
from mesh_editor.undo import SceneState

app = QApplication.instance() or QApplication([])


class MoveTests(unittest.TestCase):
    def setUp(self):
        self.scene = Scene()
        self.a, self.b, self.c = [Object(name, create_cube_mesh()) for name in 'ABC']
        self.scene.objects = [self.a, self.b, self.c]
        self.a.position[:] = [2,3,4]

    def test_reparent_and_unparent_preserve_world_transform(self):
        self.b.rotation[:] = [30,20,50]
        self.b.scale[:] = [2,3,4]
        before = self.a.get_model_matrix().copy()
        self.scene.move_objects([self.a], self.b)
        self.assertIs(self.a.parent, self.b)
        np.testing.assert_allclose(self.a.get_model_matrix(), before, atol=2e-6)
        self.scene.move_objects([self.a])
        self.assertIsNone(self.a.parent)
        self.assertEqual(self.scene.objects, [self.b,self.c,self.a])
        np.testing.assert_allclose(self.a.get_model_matrix(), before, atol=3e-6)

    def test_root_and_sibling_reorder(self):
        self.scene.move_objects([self.c], before=self.a)
        self.assertEqual(self.scene.objects, [self.c,self.a,self.b])
        self.scene.move_objects([self.a,self.c], self.b)
        self.assertEqual(self.b.children, [self.a,self.c])
        self.scene.move_objects([self.c], self.b, self.a)
        self.assertEqual(self.b.children, [self.c,self.a])
        self.assertEqual(self.scene.objects, [self.b,self.c,self.a])

    def test_parent_and_selected_descendant_move_once(self):
        self.a.set_parent(self.b)
        world = self.a.get_model_matrix().copy()
        self.scene.move_objects([self.b,self.a], self.c)
        self.assertIs(self.b.parent, self.c)
        self.assertIs(self.a.parent, self.b)
        np.testing.assert_allclose(self.a.get_model_matrix(), world, atol=2e-6)

    def test_invalid_moves_leave_scene_unchanged(self):
        self.a.set_parent(self.b)
        for objects, parent in (([self.b], self.a), ([self.a], self.a), ([self.a], Object('External',None))):
            with self.assertRaises(ValueError):
                self.scene.move_objects(objects, parent)
            self.assertEqual(self.scene.objects, [self.a,self.b,self.c])
            self.assertIs(self.a.parent, self.b)
        self.c.scale[:] = 0
        with self.assertRaises(ValueError):
            self.scene.move_objects([self.a], self.c)
        self.assertIs(self.a.parent, self.b)

    def test_noop_and_snapshot_keep_explicit_child_order(self):
        self.a.set_parent(self.b)
        self.c.set_parent(self.b)
        self.b.children[:] = [self.c,self.a]
        before = SceneState(self.scene)
        self.scene.move_objects([self.a], self.b, self.c)
        self.assertEqual(self.b.children, [self.a,self.c])
        before.restore(self.scene)
        self.assertEqual(self.b.children, [self.c,self.a])
        self.assertFalse(self.scene.move_objects([self.a], self.b))

    def test_mesh_parent_bounds_include_children(self):
        self.a.position[:] = [10,0,0]
        self.a.set_parent(self.b)
        low,high = self.b.get_world_aabb()
        np.testing.assert_allclose(low, [-0.5,-0.5,-0.5])
        np.testing.assert_allclose(high, [10.5,0.5,0.5])


class Enter(QDragEnterEvent):
    def source(self):
        return self.test_source


class Move(QDragMoveEvent):
    def source(self):
        return self.test_source


class Drop(QDropEvent):
    def source(self):
        return self.test_source


class OutlinerDropTests(unittest.TestCase):
    def setUp(self):
        self.scene = Scene()
        self.a,self.b,self.c = [Object(name, None) for name in 'ABC']
        self.scene.objects = [self.a,self.b,self.c]
        self.scene.selected_object = self.a
        self.tree = SceneOutliner()
        self.tree.resize(400,450)
        self.tree.rebuild(self.scene)
        self.tree.move_requested.connect(self.apply_move)
        self.tree.show()
        app.processEvents()

    def tearDown(self):
        self.tree.close()
        self.tree.deleteLater()
        app.processEvents()

    def apply_move(self, payload):
        self.scene.move_objects(*payload)
        self.tree.rebuild(self.scene)

    def drag_to(self, point, source=None):
        source = self.tree if source is None else source
        data = self.tree.mimeData(self.tree.selectedItems())
        for cls in (Enter, Move, Drop):
            event = cls(QPointF(point) if cls is Drop else point,
                        Qt.DropAction.MoveAction, data,
                        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
            event.test_source = source
            QApplication.sendEvent(self.tree.viewport(), event)
        return event

    def test_drop_on_object_parents_and_blank_area_unparents(self):
        target = self.tree.visualItemRect(self.tree.items_by_object[self.b]).center()
        event = self.drag_to(target)
        self.assertTrue(event.isAccepted())
        self.assertIs(self.a.parent, self.b)
        self.assertIs(self.tree.items_by_object[self.a].parent(), self.tree.items_by_object[self.b])
        event = self.drag_to(QPoint(80,400))
        self.assertTrue(event.isAccepted())
        self.assertIsNone(self.a.parent)
        self.assertEqual(self.scene.objects, [self.b,self.c,self.a])

    def test_above_and_below_rows_reorder_without_parenting(self):
        rect = self.tree.visualItemRect(self.tree.items_by_object[self.c])
        self.drag_to(QPoint(rect.center().x(), rect.top()+1))
        self.assertEqual(self.scene.objects, [self.b,self.a,self.c])
        self.assertIsNone(self.a.parent)
        rect = self.tree.visualItemRect(self.tree.items_by_object[self.c])
        self.drag_to(QPoint(rect.center().x(), rect.bottom()-1))
        self.assertEqual(self.scene.objects, [self.b,self.c,self.a])

    def test_cycle_and_external_drags_are_rejected(self):
        self.b.set_parent(self.a)
        self.tree.rebuild(self.scene)
        target = self.tree.visualItemRect(self.tree.items_by_object[self.b]).center()
        event = self.drag_to(target)
        self.assertFalse(event.isAccepted())
        self.assertIsNone(self.a.parent)
        external = QWidget()
        self.assertFalse(self.drag_to(QPoint(80,400), external).isAccepted())

    def test_drag_mime_payload_does_not_serialize_mesh_objects(self):
        data = self.tree.mimeData(self.tree.selectedItems())
        self.assertEqual(data.formats(), [SceneOutliner.MIME_TYPE])
        self.assertLess(len(data.data(SceneOutliner.MIME_TYPE)), 32)


class EditorMoveUndoTests(unittest.TestCase):
    def test_drag_move_undo_redo_rebuilds_hierarchy_and_retains_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = QSettings(os.path.join(folder,'settings.ini'), QSettings.Format.IniFormat)
            with patch('mesh_editor.workspace.QSettings',return_value=settings), patch('addon_manager.AddonManager._load_prefs'), patch('addon_manager.AddonManager.initialize_addons'):
                editor = MeshEditorWidget()
            a,b,c = [Object(name,create_cube_mesh()) for name in 'ABC']
            editor.viewport.scene.objects = [a,b,c]
            editor.viewport.scene.selected_objects = [a,c]
            editor.viewport.scene.active_object = c
            editor._populate_outliner()
            editor._on_outliner_drop(([a,c],b,None))
            self.assertEqual(b.children,[a,c])
            self.assertEqual(editor.outliner_list.items_by_object[b].childCount(),2)
            self.assertEqual(editor.selected_objects,[a,c])
            editor._perform_undo()
            self.assertEqual(editor.viewport.scene.objects,[a,b,c])
            self.assertEqual(b.children,[])
            editor._perform_redo()
            self.assertEqual(b.children,[a,c])
            self.assertEqual(editor.selected_objects,[a,c])
            editor._on_outliner_drop(([c],b,a))
            self.assertEqual(b.children,[c,a])
            editor._perform_undo()
            self.assertEqual(b.children,[a,c])
            editor._perform_redo()
            self.assertEqual(b.children,[c,a])
            editor.deleteLater()
            app.processEvents()


if __name__ == '__main__':
    unittest.main()
