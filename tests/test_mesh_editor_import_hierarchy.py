"""Import, hierarchy transforms, context actions, and graph undo regressions."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import trimesh
from PIL import Image
from PySide6.QtCore import QSettings, QTimer, QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QDialog
from PySide6.QtTest import QTest

from mesh_editor import MeshEditorWidget
from mesh_editor.scene import Object, Scene, create_cube_mesh, load_mesh_file, export_scene_to_file
from mesh_editor.import_options import group_objects, build_import_objects, match_textures, apply_assignments
from mesh_editor.import_dialog import ModelImportDialog
from mesh_editor.undo import SceneState, SceneGraphCommand, DeleteCommand, AddObjectCommand, TransformCommand
from mesh_editor.viewport import GizmoPivot

app = QApplication.instance() or QApplication([])


class HierarchyTests(unittest.TestCase):
    def object(self, name, position=(0, 0, 0)):
        obj = Object(name, create_cube_mesh())
        obj.position[:] = position
        return obj

    def test_parent_center_world_preservation_and_parent_trs(self):
        a, b = self.object('A', (-3, 0, 0)), self.object('B', (3, 0, 0))
        a.rotation[:] = [25, 70, 0]
        before = {obj: obj.get_model_matrix().copy() for obj in (a,b)}
        bounds = [obj.get_world_aabb() for obj in (a,b)]
        expected = (np.min([b[0] for b in bounds], axis=0) + np.max([b[1] for b in bounds], axis=0)) / 2
        parent = group_objects([a,b], 'Parent')
        np.testing.assert_allclose(parent.position, expected, atol=1e-6)
        initial_parent = parent.get_model_matrix().T.copy()
        for child in (a,b):
            np.testing.assert_allclose(child.get_model_matrix(), before[child], atol=1e-6)
        parent.position[:] = [2, 4, 6]
        parent.rotation[:] = [0, 0, 90]
        parent.scale[:] = [2, 3, 4]
        for child in (a,b):
            np.testing.assert_allclose(child.get_model_matrix().T, parent.get_model_matrix().T @ np.linalg.inv(initial_parent) @ before[child].T, atol=1e-5)

    def test_reparent_preserves_world_under_nonuniform_rotated_parent(self):
        parent = self.object('P')
        parent.rotation[:] = [35, 20, 60]
        parent.scale[:] = [2, 3, 4]
        child = self.object('C', (5, 1, -2))
        child.rotation[:] = [10, 40, 25]
        world = child.get_model_matrix().copy()
        child.set_parent(parent)
        np.testing.assert_allclose(child.get_model_matrix(), world, atol=2e-6)
        child.set_parent(None)
        np.testing.assert_allclose(child.get_model_matrix(), world, atol=3e-6)

    def test_reject_cycles_and_selected_descendants_are_not_parented_twice(self):
        child = self.object('C', (3,0,0))
        parent = group_objects([child], 'P')
        group = group_objects([parent, child], 'Group')
        self.assertIs(child.parent, parent)
        self.assertIs(parent.parent, group)
        with self.assertRaises(ValueError):
            group.set_parent(child)

    def test_import_grouping_names_and_existing_parent(self):
        meshes = [('A', create_cube_mesh()), ('B', create_cube_mesh())]
        objects, active = build_import_objects(meshes, '/tmp/Lung.glb', True)
        self.assertEqual(active.name, 'Lung')
        self.assertEqual(len(active.children), 2)
        flat, _ = build_import_objects(meshes, '/tmp/Lung.glb', False)
        self.assertEqual(len(flat), 2)
        self.assertTrue(all(obj.parent is None for obj in flat))
        for _, mesh in meshes:
            mesh.source_hierarchy = ('Existing Root', 'Branch')
        objects, active = build_import_objects(meshes, '/tmp/Lung.glb', True)
        self.assertEqual(active.name, 'Existing Root')
        self.assertEqual(len(objects), 4)
        self.assertEqual(active.children[0].name, 'Branch')

    def test_graph_delete_and_undo_restore_nested_relationships(self):
        child = self.object('C', (3,0,0))
        parent = group_objects([child], 'P')
        scene = Scene()
        scene.objects = [child, parent]
        scene.selected_object = parent
        viewport = SimpleNamespace(scene=scene, update=lambda: None, selection_changed=SimpleNamespace(emit=lambda *_: None))
        cmd = DeleteCommand([parent], viewport)
        cmd.execute()
        self.assertEqual(scene.objects, [])
        cmd.undo()
        self.assertEqual(scene.objects, [child, parent])
        self.assertIs(child.parent, parent)
        self.assertEqual(parent.children, [child])
        cmd.redo()
        cmd.undo()
        child_delete = DeleteCommand([child], viewport)
        child_delete.execute()
        self.assertEqual(parent.children, [])
        child_delete.undo()
        self.assertEqual(parent.children, [child])

    def test_export_bakes_parent_transform_and_skips_empty_nodes(self):
        child = self.object('C')
        parent = group_objects([child], 'P')
        parent.position += [3,2,1]
        scene = Scene()
        scene.objects = [parent, child, Object('Empty', None)]
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder)/'result.glb')
            export_scene_to_file(scene, path, pipeline_native_coords=False)
            exported = trimesh.load(path, force='scene')
            np.testing.assert_allclose(exported.bounds, child.get_world_aabb(), atol=1e-5)


class TextureImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def image(self, name, color):
        path = self.folder/name
        Image.new('RGB', (2,2), color).save(path)
        return str(path)

    def meshes(self):
        meshes = []
        for name in ('Body', 'Leaf'):
            mesh = create_cube_mesh()
            mesh.texcoords = np.zeros(len(mesh.vertices)//3 * 2, dtype=np.float32)
            mesh.material_name = name
            meshes.append((name, mesh))
        return meshes

    def test_external_matches_each_material_and_falls_back_for_unmatched(self):
        meshes = self.meshes()
        body = self.image('Body_baseColor.png', 'red')
        normal = self.image('Body_normal.png', 'blue')
        images, assignments = match_textures(meshes, [body, normal])
        self.assertEqual(assignments, {'Body': body})
        apply_assignments(meshes, assignments)
        self.assertIsNotNone(meshes[0][1].texture_data)
        self.assertIsNone(meshes[1][1].texture_data)

    def test_mtl_material_links_and_ambiguous_images(self):
        body = self.image('custom name.png', 'red')
        mtl = self.folder/'materials.mtl'
        mtl.write_text('newmtl Body\nmap_Kd -s 1 1 1 "custom name.png"\n')
        _, assignments = match_textures(self.meshes(), [str(mtl)])
        self.assertEqual(assignments, {'Body': body})
        first = self.image('Body_diffuse.png', 'red')
        second = self.image('Body_albedo.png', 'green')
        _, assignments = match_textures(self.meshes(), [first, second])
        self.assertNotIn('Body', assignments)

    def test_real_glb_load_embedded_texture_and_material_colour(self):
        scene = trimesh.Scene()
        for name, image, color in [('Textured', Image.new('RGB',(2,2),'red'), [255,255,255,255]), ('Plain', None, [0,128,0,255])]:
            mesh = trimesh.creation.box()
            mesh.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(mesh.vertices),2)),
                material=trimesh.visual.material.PBRMaterial(name=name, baseColorTexture=image, baseColorFactor=color))
            scene.add_geometry(mesh, node_name=name)
        path = self.folder/'model.glb'
        scene.export(str(path))
        meshes = load_mesh_file(str(path))
        by_material = {mesh.material_name: mesh for _,mesh in meshes}
        self.assertIsNotNone(by_material['Textured'].texture_data)
        self.assertIsNone(by_material['Plain'].texture_data)
        np.testing.assert_allclose(by_material['Plain'].base_color, [0,128/255,0], atol=1e-5)

    def test_import_dialog_loads_async_and_allows_manual_override(self):
        meshes = self.meshes()
        image = self.image('Body_baseColor.png', 'red')
        with patch('mesh_editor.import_dialog.load_mesh_file', return_value=meshes):
            dialog = ModelImportDialog(str(self.folder/'model.glb'), texture_paths=[image])
            for _ in range(200):
                app.processEvents()
                if dialog.meshes is not None and not dialog.worker.isRunning():
                    break
                QTest.qWait(5)
            self.assertEqual(dialog.table.rowCount(), 2)
            self.assertEqual(dialog.assignments['Body'], image)
            self.assertEqual(dialog.texture_mode.currentText(), 'External')
            dialog.table.cellWidget(0,1).setCurrentIndex(0)
            self.assertIsNone(dialog.assignments['Body'])
            dialog.accept()
            self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
            dialog.deleteLater()


class EditorHierarchyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        settings = QSettings(str(Path(self.temp.name)/'settings.ini'), QSettings.Format.IniFormat)
        with patch('mesh_editor.workspace.QSettings', return_value=settings), patch('addon_manager.AddonManager._load_prefs'), patch('addon_manager.AddonManager.initialize_addons'):
            self.editor = MeshEditorWidget()

    def tearDown(self):
        self.editor.deleteLater()
        app.processEvents()
        self.temp.cleanup()

    def test_context_menu_parent_empty_rename_and_undo(self):
        editor = self.editor
        a = Object('Duplicate', create_cube_mesh())
        b = Object('Duplicate', create_cube_mesh())
        b.position[:] = [4,0,0]
        editor.viewport.scene.objects = [a,b]
        editor.viewport.scene.selected_objects = [a,b]
        editor.viewport.scene.active_object = b
        editor._populate_outliner()
        self.assertEqual(len(editor.outliner_list.items_by_object), 2)
        menu = editor._build_object_context_menu()
        self.assertFalse(menu.actions()[1].isEnabled())
        editor._create_parent_object()
        parent = editor.active_object
        self.assertIs(a.parent, parent)
        np.testing.assert_allclose(parent.position, [2,0,0])
        self.assertEqual(editor.outliner_list.items_by_object[parent].childCount(), 2)
        editor._perform_undo()
        self.assertIsNone(a.parent)
        self.assertEqual(len(editor.viewport.scene.objects), 2)
        editor._perform_redo()
        self.assertIs(a.parent, parent)
        with patch('mesh_editor.QInputDialog.getText', return_value=('Renamed', True)):
            editor._rename_selected_object()
        self.assertEqual(parent.name, 'Renamed')
        editor._perform_undo()
        self.assertEqual(parent.name, 'Parent')
        editor._create_empty_object()
        self.assertIsNone(editor.active_object.mesh)
        self.assertIsNone(editor.active_object.parent)
        editor._perform_undo()
        self.assertEqual(len(editor.viewport.scene.objects), 3)

    def test_import_flow_uses_one_preview_load_and_never_missing_texture_popup(self):
        editor = self.editor
        path = Path(self.temp.name)/'model.glb'
        path.write_bytes(b'preview fixture')
        meshes = [('A', create_cube_mesh()), ('B', create_cube_mesh())]
        original_exec = ModelImportDialog.exec
        def auto_accept(dialog):
            def ready():
                if dialog.meshes is not None and not dialog.worker.isRunning():
                    dialog.accept()
                else:
                    QTimer.singleShot(5, ready)
            QTimer.singleShot(5, ready)
            return original_exec(dialog)
        with patch('mesh_editor.import_dialog.load_mesh_file', return_value=meshes) as loader, \
             patch.object(ModelImportDialog, 'exec', auto_accept), \
             patch.object(editor.viewport, 'makeCurrent'), patch.object(editor.viewport, 'doneCurrent'), \
             patch('mesh_editor.scene.Mesh.setup_buffers'), patch('mesh_editor.QMessageBox.question') as popup, \
             patch('mesh_editor.QMessageBox.critical') as error:
            editor._import_mesh_from_path(str(path))
        loader.assert_called_once()
        popup.assert_not_called()
        error.assert_not_called()
        self.assertEqual(len(editor.viewport.scene.objects), 3)
        parent = editor.active_object
        self.assertEqual(parent.name, 'model')
        self.assertEqual(len(parent.children), 2)
        editor._perform_undo()
        self.assertEqual(editor.viewport.scene.objects, [])
        editor._perform_redo()
        self.assertIs(editor.active_object, parent)
        self.assertEqual(editor.selected_objects, [parent])

    def test_right_click_opens_context_menu_but_right_drag_retains_fly_navigation(self):
        viewport = self.editor.viewport
        viewport.context_menu_requested.disconnect()
        menus = []
        viewport.context_menu_requested.connect(menus.append)
        def event(kind, x):
            return QMouseEvent(kind, QPointF(x,10), QPointF(x+100,110),
                Qt.MouseButton.RightButton, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier)
        with patch.object(viewport, '_pick_object_at', return_value=None):
            viewport.mousePressEvent(event(QEvent.Type.MouseButtonPress, 10))
            viewport.mouseReleaseEvent(event(QEvent.Type.MouseButtonRelease, 10))
            self.assertEqual(len(menus), 1)
            viewport.mousePressEvent(event(QEvent.Type.MouseButtonPress, 10))
            viewport.mouseMoveEvent(event(QEvent.Type.MouseMove, 50))
            viewport.mouseReleaseEvent(event(QEvent.Type.MouseButtonRelease, 50))
            self.assertEqual(len(menus), 1)
            self.assertFalse(viewport.is_fly_navigating)

    def test_baked_off_origin_mesh_keeps_gizmo_at_visible_geometry(self):
        editor = self.editor
        mesh = create_cube_mesh()
        mesh.vertices.reshape(-1,3)[:] += [10,3,4]
        mesh.local_aabb_min += [10,3,4]
        mesh.local_aabb_max += [10,3,4]
        obj = Object('Offset mesh', mesh)
        editor.viewport.scene.objects = [obj]
        editor.viewport.scene.selected_object = obj
        np.testing.assert_allclose(editor.viewport._get_gizmo_pivot().position, [10,3,4])

    def test_parent_rotation_and_scaling_move_children_about_parent_pivot(self):
        editor = self.editor
        a, b = Object('A', create_cube_mesh()), Object('B', create_cube_mesh())
        a.position[:] = [-2,0,0]
        b.position[:] = [2,0,0]
        parent = group_objects([a,b], 'P')
        viewport = editor.viewport
        viewport.scene.objects = [a,b,parent]
        viewport.scene.selected_object = parent
        pivot = viewport._get_gizmo_pivot()
        viewport._drag_start_world = {parent: parent.get_model_matrix().T.copy()}
        viewport._drag_start_pivot_pos = pivot.position.copy()
        viewport._drag_start_pivot_scale = pivot.scale.copy()
        viewport._drag_start_pivot_r_mat = np.eye(4)
        viewport._drag_pivot = GizmoPivot(pivot.position, np.array([0,0,90]), pivot.scale)
        viewport.gizmo.operation = 'rotate'
        viewport._apply_gizmo_transform()
        np.testing.assert_allclose(a.get_model_matrix()[3,:3], [0,2,0], atol=1e-5)
        viewport._drag_pivot = GizmoPivot(pivot.position, pivot.rotation, np.array([2,2,2]))
        viewport.gizmo.operation = 'scale'
        viewport._apply_gizmo_transform()
        np.testing.assert_allclose(a.get_model_matrix()[3,:3], [-4,0,0], atol=1e-5)

    def test_gizmo_world_transform_applies_once_to_selected_parent_and_child(self):
        editor = self.editor
        child = Object('C', create_cube_mesh())
        child.position[:] = [4,0,0]
        parent = group_objects([child], 'P')
        editor.viewport.scene.objects = [parent, child]
        editor.viewport.scene.selected_objects = [parent, child]
        editor.viewport.scene.active_object = parent
        viewport = editor.viewport
        pivot = viewport._get_gizmo_pivot()
        viewport._drag_start_world = {parent: parent.get_model_matrix().T.copy()}
        viewport._drag_start_pivot_pos = pivot.position.copy()
        viewport._drag_pivot = GizmoPivot(pivot.position + [2,3,4], pivot.rotation, pivot.scale)
        viewport.gizmo.operation = 'translate'
        viewport._apply_gizmo_transform()
        np.testing.assert_allclose(child.get_model_matrix()[3,:3], [6,3,4], atol=1e-5)


if __name__ == '__main__':
    unittest.main()
