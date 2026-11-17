"""Exercise stage commands and outcomes at the pipeline integration boundary."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PySide6.QtCore import QCoreApplication

from acceleration import AccelerationProfile, Device, StageCapability
from pipeline_manager import PipelineWorker


class AccelerationPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def worker(self, selection="auto"):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        devices = (
            Device("nvidia", "NVIDIA", "NVIDIA", indices={"cuda": 3}),
            Device("amd", "AMD", "AMD", True, {"hip": 1}),
        )
        capabilities = (
            StageCapability("feature_extractor", "hip", use_gpu_flag="--FeatureExtraction.use_gpu", device_flag="--FeatureExtraction.gpu_index", cpu_device="-1"),
            StageCapability("feature_matching", "hip", use_gpu_flag="--FeatureMatching.use_gpu", device_flag="--FeatureMatching.gpu_index", cpu_device="-1"),
            StageCapability("DensifyPointCloud", "cuda", device_flag="--cuda-device"),
        )
        return PipelineWorker(None, self.directory.name, gpu_mode=selection,
                              acceleration_profile=AccelerationProfile(devices, capabilities))

    def test_explicit_device_is_pinned_and_cpu_handles_unsupported_stage(self):
        worker = self.worker("gpu:nvidia")
        with patch.object(worker, "_run_process_realtime", return_value=True) as run:
            self.assertTrue(worker._run_accelerated_stage("DensifyPointCloud", ["densify", "scene.mvs"], 10))
        command = run.call_args.args[0]
        self.assertEqual(command[-2:], ["--cuda-device", "3"])
        self.assertEqual(worker.stage_backends["DensifyPointCloud"], "NVIDIA (CUDA)")
        with patch.object(worker, "_run_process_realtime", return_value=True) as run:
            self.assertTrue(worker._run_feature_stage(
                ["colmap", "feature_extractor", "--SiftExtraction.use_gpu", "1"],
                ["colmap", "feature_extractor", "--SiftExtraction.use_gpu", "0"], 10))
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--SiftExtraction.use_gpu") + 1], "0")
        self.assertEqual(worker.stage_backends["feature_extractor"], "CPU")

    def test_auto_uses_one_device_and_does_not_substitute_second_gpu(self):
        worker = self.worker()
        self.assertEqual(worker.execution_plan.device.id, "amd")
        with patch.object(worker, "_run_process_realtime", return_value=True) as run:
            worker._run_accelerated_stage("DensifyPointCloud", ["densify"], 10)
        self.assertEqual(run.call_args.args[0][-2:], ["--cuda-device", "-2"])

    def test_gpu_backend_failure_retries_same_stage_explicitly_on_cpu(self):
        worker = self.worker("gpu:nvidia")
        calls = []

        def execute(command, **kwargs):
            calls.append((command, kwargs["env"]))
            worker.last_process_outcome = "process_error" if len(calls) == 1 else "success"
            worker.last_process_output = ["CUDA error: driver initialization failed"] if len(calls) == 1 else []
            return len(calls) == 2

        with patch.object(worker, "_run_process_realtime", side_effect=execute):
            self.assertTrue(worker._run_accelerated_stage(
                "DensifyPointCloud", ["densify"], 10,
                env={"CUDA_VISIBLE_DEVICES": "GPU-allowed", "OMP_NUM_THREADS": "2"}))
        self.assertEqual([command[-1] for command, _ in calls], ["3", "-2"])
        self.assertTrue(all(env["CUDA_VISIBLE_DEVICES"] == "GPU-allowed" for _, env in calls))
        self.assertEqual(worker.stage_backends["DensifyPointCloud"], "CPU (GPU fallback)")

    def test_internal_tool_cpu_fallback_is_reported_as_cpu(self):
        worker = self.worker("gpu:nvidia")

        def execute(command, **kwargs):
            worker.last_process_output = ["GPU unavailable; falling back to CPU"]
            worker.last_process_outcome = "success"
            return True

        with patch.object(worker, "_run_process_realtime", side_effect=execute) as run:
            self.assertTrue(worker._run_accelerated_stage("DensifyPointCloud", ["densify"], 10))
        self.assertEqual(run.call_count, 1)
        self.assertEqual(worker.stage_backends["DensifyPointCloud"], "CPU (GPU fallback)")

    def test_cpu_retry_failure_is_terminal_and_keeps_checkpoint(self):
        worker = self.worker("gpu:nvidia")
        checkpoint = Path(self.directory.name) / "checkpoint.mvs"
        checkpoint.write_bytes(b"previous completed stage")
        calls = []

        def execute(command, **kwargs):
            calls.append(command)
            worker.last_process_outcome = "process_error"
            worker.last_process_output = ["CUDA error"] if len(calls) == 1 else ["CPU reconstruction failed"]
            worker.failure_reason = worker.last_process_output[-1]
            return False

        finished = []
        worker.finished.connect(lambda ok, message: finished.append((ok, message)))
        with patch.object(worker, "_verify_binaries", return_value=True), patch.object(worker, "_run_real_pipeline", side_effect=lambda: worker._run_accelerated_stage("DensifyPointCloud", ["densify"], 10)), patch.object(worker, "_run_process_realtime", side_effect=execute), patch("pipeline_manager.time.sleep"):
            worker.run()
        self.assertEqual(len(calls), 2)
        self.assertEqual(finished, [(False, "CPU reconstruction failed")])
        self.assertEqual(checkpoint.read_bytes(), b"previous completed stage")

    def test_missing_toolchain_reports_failure_instead_of_simulated_success(self):
        worker = self.worker("cpu")
        finished = []
        worker.finished.connect(lambda ok, message: finished.append((ok, message)))
        with patch.object(worker, "_verify_binaries", return_value=False), patch.object(worker, "_run_real_pipeline") as real, patch("pipeline_manager.time.sleep"):
            worker.run()
        real.assert_not_called()
        self.assertFalse(finished[0][0])
        self.assertIn("tools are missing", finished[0][1])

    def test_matcher_fallback_removes_values_with_unsupported_options(self):
        command = ["colmap", "vocab_tree_matcher", "--database_path", "database.db",
                   "--VocabTreeMatching.vocab_tree_path", "tree.bin",
                   "--VocabTreeMatching.num_images=50", "--SiftMatching.use_gpu", "0"]
        self.assertEqual(PipelineWorker._without_options(command, "--VocabTreeMatching"),
                         ["colmap", "vocab_tree_matcher", "--database_path", "database.db",
                          "--SiftMatching.use_gpu", "0"])


if __name__ == "__main__":
    unittest.main()
