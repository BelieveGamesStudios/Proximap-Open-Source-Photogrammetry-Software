import builtins
import importlib.util
from pathlib import Path
import runpy
import sys
from subprocess import _args_from_interpreter_flags
import tempfile
import unittest
from unittest.mock import patch

from model_assets import SILUETA_SIZE, validate_silueta
from pipeline_manager import BackgroundRemovalWorker


ROOT = Path(__file__).resolve().parents[1]


class ModelAssetTests(unittest.TestCase):
    def test_missing_and_lfs_models_have_actionable_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "silueta.onnx"
            with self.assertRaisesRegex(ValueError, "Cannot read.*model"):
                validate_silueta(model)
            model.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
            with self.assertRaisesRegex(ValueError, "Git LFS pointer"):
                validate_silueta(model)

    def test_truncated_and_same_size_corrupt_models_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "silueta.onnx"
            model.write_bytes(b"invalid protobuf")
            with self.assertRaisesRegex(ValueError, "incomplete or invalid"):
                validate_silueta(model)
            with model.open("wb") as stream:
                stream.truncate(SILUETA_SIZE)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                validate_silueta(model)

    def test_worker_keeps_sources_and_does_not_import_rembg_for_invalid_model(self):
        with tempfile.TemporaryDirectory() as directory:
            models = Path(directory) / "models"
            models.mkdir()
            (models / "silueta.onnx").write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
            source = Path(directory) / "photo.jpg"
            source.write_bytes(b"original image")
            output = Path(directory) / "output"
            worker = BackgroundRemovalWorker([str(source)], str(output))
            results = []
            worker.finished.connect(lambda *args: results.append(args))
            original_import = builtins.__import__

            def guarded_import(name, *args, **kwargs):
                if name == "rembg":
                    self.fail("Invalid model must be detected before importing rembg")
                return original_import(name, *args, **kwargs)

            with patch("pipeline_manager.get_base_dir", return_value=directory), patch("builtins.__import__", guarded_import):
                worker.run()
            self.assertEqual(len(results), 1)
            self.assertFalse(results[0][0])
            self.assertEqual(results[0][1], [str(source)])
            self.assertIn("Git LFS pointer", results[0][2])
            self.assertEqual(source.read_bytes(), b"original image")
            self.assertFalse(output.exists())


class FrozenStartupTests(unittest.TestCase):
    def test_resource_tracker_exits_before_gui_imports_in_both_entrypoints(self):
        import multiprocessing
        import multiprocessing.spawn
        import multiprocessing.resource_tracker

        spec = importlib.util.find_spec("PyInstaller")
        if spec is None:
            self.skipTest("PyInstaller not installed")
        hook = Path(spec.origin).parent / "hooks" / "rthooks" / "pyi_rth_multiprocessing.py"
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in ("PySide6", "numpy", "vispy"):
                self.fail("A multiprocessing helper attempted to load the GUI")
            return original_import(name, *args, **kwargs)

        for entrypoint in ("main_window.py", "launcher.py"):
            with self.subTest(entrypoint=entrypoint), \
                    patch.object(multiprocessing, "freeze_support"), \
                    patch.object(multiprocessing.spawn, "freeze_support"), \
                    patch.object(multiprocessing.process, "ORIGINAL_DIR"), \
                    patch.object(multiprocessing.resource_tracker, "main") as tracker, \
                    patch.object(sys, "argv", ["Proximap", *_args_from_interpreter_flags(), "-c", "from multiprocessing.resource_tracker import main;main(7)"]), \
                    patch("builtins.__import__", guarded_import):
                runpy.run_path(str(hook))
                with self.assertRaises(SystemExit):
                    runpy.run_path(str(ROOT / entrypoint), run_name="__main__")
                tracker.assert_called_once_with(7)


if __name__ == "__main__":
    unittest.main()
