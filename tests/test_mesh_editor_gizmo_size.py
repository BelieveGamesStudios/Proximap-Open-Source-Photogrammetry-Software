"""Projection and picking regressions for zoom and camera fly movement."""
import unittest

import numpy as np
import pyrr

from mesh_editor.gizmo import Gizmo
from mesh_editor.scene import Camera, Object


class GizmoSizeTests(unittest.TestCase):
    def setUp(self):
        self.camera = Camera()
        self.camera.yaw = 0.0
        self.camera.pitch = 0.0
        self.obj = Object('Selection', None)
        self.gizmo = Gizmo()

    def project(self, point, height=800):
        width = height * 1.5
        vp = pyrr.matrix44.multiply(self.camera.get_view_matrix(),
                                   self.camera.get_projection_matrix(width / height))
        _, _, transform = self.gizmo.get_gizmo_transforms(self.obj)
        scale = self.gizmo.get_scale_factor(self.camera, self.obj, height)
        return self.gizmo.project_local_to_screen(point, transform, scale, vp, width, height)

    def assert_handle_size_and_pick(self, height=800):
        center = self.project([0, 0, 0], height)
        tip = self.project([1, 0, 0], height)
        self.assertAlmostEqual(float(np.linalg.norm(tip - center)), 90.0, places=3)
        pick = self.project([0.7, 0, 0], height)
        for operation in ('translate', 'scale'):
            self.gizmo.operation = operation
            self.assertEqual(self.gizmo.check_hover(*pick, self.obj, self.camera, height * 1.5, height), 'axis_x')

    def test_perspective_zoom_keeps_screen_size_and_matching_hit_targets(self):
        for distance in (50, 8, 1, 0.2):
            self.camera.distance = distance
            self.assert_handle_size_and_pick()

    def test_flying_toward_selection_shrinks_world_size_without_orbit_zoom(self):
        before = self.gizmo.get_scale_factor(self.camera, self.obj, 800)
        self.camera.move(self.camera.get_forward_vector() * 7)
        self.assertEqual(self.camera.distance, 8.0)
        after = self.gizmo.get_scale_factor(self.camera, self.obj, 800)
        self.assertAlmostEqual(after, before / 8, places=6)
        self.assert_handle_size_and_pick()

    def test_off_center_pivot_uses_forward_depth_not_distance_from_camera(self):
        self.obj.position[:] = [5, 0, 2]
        self.assert_handle_size_and_pick()

    def test_orthographic_zoom_and_viewport_resize_keep_pixel_size(self):
        self.camera.is_perspective = False
        for scale in (0.5, 6, 100):
            self.camera.ortho_scale = scale
            for height in (400, 800, 1600):
                self.assert_handle_size_and_pick(height)

    def test_fov_and_perspective_viewport_resize_keep_pixel_size(self):
        for fov in (30, 45, 90):
            self.camera.fov = fov
            for height in (400, 800, 1600):
                self.assert_handle_size_and_pick(height)

    def test_pivot_behind_camera_is_not_drawn_or_pickable(self):
        self.obj.position[:] = self.camera.get_position() - self.camera.get_forward_vector()
        self.assertEqual(self.gizmo.get_scale_factor(self.camera, self.obj, 800), 0)
        self.assertIsNone(self.gizmo.check_hover(600, 400, self.obj, self.camera, 1200, 800))


if __name__ == '__main__':
    unittest.main()
