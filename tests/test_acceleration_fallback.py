"""Fallback must never restart cancelled or resource-terminated processing."""
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from PySide6.QtCore import QCoreApplication
from pipeline_manager import PipelineWorker


class AccelerationFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        self.worker = PipelineWorker(None, tempfile.gettempdir())
        self.messages = []
        self.worker.log_message.connect(self.messages.append)
        self.gpu = ["colmap", "gpu"]
        self.cpu = ["colmap", "cpu"]

    def run_attempts(self, outcomes):
        calls = []
        def attempt(cmd, **kwargs):
            calls.append(cmd)
            outcome, output = outcomes[len(calls) - 1]
            self.worker.last_process_outcome = outcome
            self.worker.last_process_output = output
            return outcome == "success"
        with patch.object(self.worker, "_run_process_realtime", side_effect=attempt):
            result = self.worker._run_with_gpu_fallback(self.gpu, self.cpu, 10)
        return result, calls

    def test_backend_failure_retries_on_cpu_even_with_force_gpu_preference(self):
        self.worker.gpu_mode = "force_gpu"
        ok, calls = self.run_attempts([
            ("process_error", ["CUDA error: driver initialization failed"]),
            ("success", []),
        ])
        self.assertTrue(ok)
        self.assertEqual(calls, [self.gpu, self.cpu])
        self.assertFalse(self.worker._using_gpu_sift)
        self.assertTrue(any("driver initialization failed" in message for message in self.messages))

    def test_terminal_outcomes_and_unrelated_errors_never_retry(self):
        for outcome, output in [
            ("cancelled", ["CUDA error"]),
            ("timeout", ["CUDA error"]),
            ("memory_guard", ["CUDA error"]),
            ("spawn_error", ["CUDA error"]),
            ("process_error", ["Cannot open database"]),
            ("process_error", ["CUDA driver version: 13", "Cannot open database"]),
            ("process_error", ["No images with matches found"]),
            ("process_error", ["std::bad_alloc: out of memory"]),
        ]:
            with self.subTest(outcome=outcome, output=output):
                ok, calls = self.run_attempts([(outcome, output)])
                self.assertFalse(ok)
                self.assertEqual(calls, [self.gpu])

    def test_cpu_retry_failure_is_terminal_and_explained(self):
        ok, calls = self.run_attempts([
            ("process_error", ["Failed to create SiftGPU"]),
            ("process_error", ["CPU also failed"]),
        ])
        self.assertFalse(ok)
        self.assertEqual(calls, [self.gpu, self.cpu])
        self.assertTrue(any("CPU retry failed" in message for message in self.messages))

    def test_old_gpu_error_cannot_trigger_retry_for_new_command(self):
        self.worker.last_process_output = ["CUDA error"]
        ok, calls = self.run_attempts([("process_error", ["Invalid image path"])])
        self.assertFalse(ok)
        self.assertEqual(calls, [self.gpu])

    def test_cpu_only_mode_never_attempts_gpu(self):
        self.worker.gpu_mode = "force_cpu"
        ok, calls = self.run_attempts([("success", [])])
        self.assertTrue(ok)
        self.assertEqual(calls, [self.cpu])
        self.assertFalse(self.worker._using_gpu_sift)

    def test_already_cancelled_runner_never_spawns_process(self):
        self.worker.is_running = False
        with patch("subprocess.Popen") as popen, patch.object(self.worker, "_adapt_colmap_cmd") as adapt:
            self.assertFalse(self.worker._run_process_realtime([sys.executable, "-c", "pass"], 10))
        popen.assert_not_called()
        adapt.assert_not_called()
        self.assertEqual(self.worker.last_process_outcome, "cancelled")

    def test_cancellation_between_gpu_exit_and_retry_never_spawns_cpu(self):
        calls = []
        def cancel(cmd, **kwargs):
            calls.append(cmd)
            self.worker.last_process_outcome = "process_error"
            self.worker.last_process_output = ["CUDA error"]
            self.worker.is_running = False
            return False
        with patch.object(self.worker, "_run_process_realtime", side_effect=cancel):
            self.assertFalse(self.worker._run_with_gpu_fallback(self.gpu, self.cpu, 10))
        self.assertEqual(calls, [self.gpu])
        self.assertEqual(self.worker.last_process_outcome, "cancelled")

    def test_missing_executable_is_a_terminal_launch_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            ok = self.worker._run_process_realtime([directory + "/missing-tool"], timeout=10)
        self.assertFalse(ok)
        self.assertEqual(self.worker.last_process_outcome, "spawn_error")
        self.assertIn("missing-tool", self.worker.failure_reason)
        self.assertIsNone(self.worker._retryable_gpu_failure())

    def test_real_memory_guard_records_terminal_outcome(self):
        exhausted = SimpleNamespace(percent=98.0, available=100 * 1024 * 1024)
        with patch("pipeline_manager.time.time", side_effect=[0, 0, 0, 5]), patch("pipeline_manager.psutil.virtual_memory", return_value=exhausted):
            ok = self.worker._run_process_realtime(
                [sys.executable, "-c", "import time; time.sleep(5)"], timeout=100
            )
        self.assertFalse(ok)
        self.assertEqual(self.worker.last_process_outcome, "memory_guard")
        self.assertIn("protect system memory", self.worker.failure_reason)
        self.assertIsNone(self.worker._retryable_gpu_failure())

    def test_real_runner_records_success_failure_and_timeout(self):
        for code, timeout, expected in [
            ("print('ok')", 5, "success"),
            ("raise SystemExit(7)", 5, "process_error"),
            ("import time; time.sleep(5)", 0.01, "timeout"),
        ]:
            with self.subTest(expected=expected):
                ok = self.worker._run_process_realtime([sys.executable, "-c", code], timeout)
                self.assertEqual(ok, expected == "success")
                self.assertEqual(self.worker.last_process_outcome, expected)
                if ok:
                    self.assertEqual(self.worker.failure_reason, "")
                else:
                    self.assertTrue(self.worker.failure_reason)


if __name__ == "__main__":
    unittest.main()
