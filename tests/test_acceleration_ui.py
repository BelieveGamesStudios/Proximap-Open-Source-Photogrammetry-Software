"""Exercise the actual small UI handlers without importing the renderer stack."""
import ast
import builtins
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QWidget

from acceleration import AccelerationProfile, Device, StageCapability, normalize_selection


ROOT = Path(__file__).resolve().parents[1]


def ui_handler(name):
    tree = ast.parse((ROOT / "main_window.py").read_text())
    window = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
    method = next(node for node in window.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"normalize_selection": normalize_selection}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ROOT / "main_window.py"), "exec"), namespace)
    return namespace[name]


class SelectionHost(QWidget):
    _update_acceleration_selection = ui_handler("_update_acceleration_selection")
    _selected_acceleration_badge = ui_handler("_selected_acceleration_badge")
    _restore_acceleration_selection = ui_handler("_restore_acceleration_selection")

    def __init__(self):
        super().__init__()
        self.acceleration_profile = AccelerationProfile(
            (Device("apple", "Apple GPU", "Apple", True, {"metal": 0}),),
            (StageCapability("DensifyPointCloud", "metal", device_flag="--gpu-device"),),
        )
        self.gpu_combo = QComboBox(self)
        # Reorder options intentionally: handlers must use stable data, never positions.
        for label, value in reversed(self.acceleration_profile.options()):
            self.gpu_combo.addItem(label, value)
        self.acceleration_details = QLabel(self)
        self.status_label = QLabel(self)
        self.console_text = []
        self.gpu_combo.currentIndexChanged.connect(self._update_acceleration_selection)
        self._update_acceleration_selection()


class AccelerationUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.host = SelectionHost()
        self.addCleanup(self.host.deleteLater)

    def test_partial_gpu_selection_explains_cpu_stages_and_planned_device(self):
        self.host._restore_acceleration_selection("gpu:apple")
        text = self.host.acceleration_details.text()
        self.assertIn("Apple GPU", text)
        self.assertIn("depth maps (METAL)", text)
        self.assertIn("CPU: feature extraction, feature matching", text)
        self.assertIn("texturing", text)
        self.assertEqual(self.host._selected_acceleration_badge(), "Planned GPU: Apple GPU")

    def test_legacy_cpu_selection_uses_data_after_options_reordered(self):
        self.host._restore_acceleration_selection("Force CPU Fallback")
        self.assertEqual(self.host.gpu_combo.currentData(), "cpu")
        self.assertEqual(self.host.acceleration_details.text(), "CPU: all reconstruction stages.")
        self.assertEqual(self.host._selected_acceleration_badge(), "Planned acceleration: CPU")

    def test_missing_gpu_restores_cpu_and_explains_without_using_available_gpu(self):
        self.host._restore_acceleration_selection("gpu:apple")
        self.host._restore_acceleration_selection("gpu:removed")
        self.assertEqual(self.host.gpu_combo.currentData(), "cpu")
        self.assertIn("unavailable", self.host.status_label.text())
        self.assertIn("CPU selected", self.host.console_text[-1])
        self.assertEqual(self.host.acceleration_details.text(), "CPU: all reconstruction stages.")

    def test_auto_choice_is_visible_before_processing(self):
        self.host._restore_acceleration_selection("auto")
        self.assertEqual(self.host.gpu_combo.currentData(), "auto")
        self.assertIn("Apple GPU", self.host.acceleration_details.text())
        self.assertEqual(self.host._selected_acceleration_badge(), "Planned GPU: Apple GPU")


class InventoryEntrypointTests(unittest.TestCase):
    def test_inventory_child_exits_before_heavy_imports_in_both_entrypoints(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in ("PySide6", "numpy", "vispy", "pipeline_manager", "hardware_profiler"):
                self.fail(f"Inventory helper imported {name}")
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "inventory.json")
            for entrypoint in ("main_window.py", "launcher.py"):
                with self.subTest(entrypoint=entrypoint), \
                        patch.object(sys, "argv", ["Proximap", "--acceleration-inventory", output]), \
                        patch("acceleration._print_inventory") as inventory, \
                        patch("builtins.__import__", guarded_import):
                    with self.assertRaises(SystemExit) as raised:
                        runpy.run_path(str(ROOT / entrypoint), run_name="__main__")
                    self.assertEqual(raised.exception.code, 0)
                    inventory.assert_called_once_with(output_path=output)


if __name__ == "__main__":
    unittest.main()
