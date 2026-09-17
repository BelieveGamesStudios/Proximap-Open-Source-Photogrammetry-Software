import sys
import unittest
from unittest.mock import Mock
from PySide6.QtWidgets import QApplication, QWidget

app = QApplication.instance() or QApplication(sys.argv)

from camera_projection_toggle import CameraProjectionToggleWidget


class CameraProjectionToggleWidgetTests(unittest.TestCase):
    def setUp(self):
        self.parent = QWidget()
        self.toggle = CameraProjectionToggleWidget(self.parent)

    def tearDown(self):
        self.parent.deleteLater()

    def test_initial_state(self):
        """Initial state should be perspective projection."""
        self.assertEqual(self.toggle.get_projection(), "perspective")
        self.assertTrue(self.toggle.btn_persp.isChecked())
        self.assertFalse(self.toggle.btn_ortho.isChecked())

    def test_button_clicks_emit_signals(self):
        """Clicking buttons should emit projection_changed signal with correct mode."""
        received = []
        self.toggle.projection_changed.connect(lambda mode: received.append(mode))

        # Click Ortho
        self.toggle.btn_ortho.click()
        self.assertEqual(received, ["orthographic"])
        self.assertEqual(self.toggle.get_projection(), "orthographic")
        self.assertTrue(self.toggle.btn_ortho.isChecked())
        self.assertFalse(self.toggle.btn_persp.isChecked())

        # Click Persp
        self.toggle.btn_persp.click()
        self.assertEqual(received, ["orthographic", "perspective"])
        self.assertEqual(self.toggle.get_projection(), "perspective")
        self.assertTrue(self.toggle.btn_persp.isChecked())
        self.assertFalse(self.toggle.btn_ortho.isChecked())

    def test_set_projection(self):
        """set_projection should update internal button states."""
        self.toggle.set_projection("orthographic")
        self.assertEqual(self.toggle.get_projection(), "orthographic")
        self.assertTrue(self.toggle.btn_ortho.isChecked())

        self.toggle.set_projection("perspective")
        self.assertEqual(self.toggle.get_projection(), "perspective")
        self.assertTrue(self.toggle.btn_persp.isChecked())

    def test_toggle_method(self):
        """toggle() should switch modes and emit signal."""
        emitted = []
        self.toggle.projection_changed.connect(lambda m: emitted.append(m))

        self.toggle.toggle()
        self.assertEqual(self.toggle.get_projection(), "orthographic")
        self.assertEqual(emitted, ["orthographic"])

        self.toggle.toggle()
        self.assertEqual(self.toggle.get_projection(), "perspective")
        self.assertEqual(emitted, ["orthographic", "perspective"])

    def test_positioning_relative_to_gizmo(self):
        """Projection toggle should be positioned directly under the navigation gizmo."""
        from mesh_editor.nav_gizmo import NavGizmoWidget
        gizmo = NavGizmoWidget(self.parent)
        container_w = 800
        margin = 12

        gx = container_w - gizmo.width() - margin
        gy = margin
        gizmo.move(max(margin, gx), gy)

        px = gx + (gizmo.width() - self.toggle.width()) // 2
        py = gy + gizmo.height() + 4
        self.toggle.move(max(margin, px), py)

        self.assertEqual(self.toggle.y(), gy + gizmo.height() + 4)
        self.assertEqual(self.toggle.x(), gx + (gizmo.width() - self.toggle.width()) // 2)


if __name__ == "__main__":
    unittest.main()
