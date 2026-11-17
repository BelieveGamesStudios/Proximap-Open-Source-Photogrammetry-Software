"""
Hardware Profiler Utility Module
This module handles hardware diagnostics, memory constraint checks, COLMAP version validation,
and reconstruction acceleration capabilities.
"""

import os
import sys
import atexit
import subprocess
import psutil
from collections import namedtuple


# Invariant check: Under no circumstances should the system query total static RAM
# via platform or os libraries. Only psutil is permitted for dynamic memory checks.

MemoryBudget = namedtuple('MemoryBudget', ['available_gb', 'ram_gb', 'swap_used_gb', 'swap_total_gb', 'safe_thread_count', 'pressure_level'])

# Track active subprocesses to clean them up on exit to prevent leaks
_active_subprocesses = set()


def _cleanup_subprocesses():
    """Fallback handler to terminate any running subprocesses on interpreter exit."""
    for proc in list(_active_subprocesses):
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass

atexit.register(_cleanup_subprocesses)


class InitializationBlockError(Exception):
    """Exception raised when an initialization constraint or dependency check fails."""
    pass


def get_available_memory() -> int:
    """
    Returns the dynamic available system memory in bytes.
    CRITICAL: Adheres to workspace rule by exclusively using psutil.
    """
    return psutil.virtual_memory().available


def get_total_memory() -> int:
    """
    Returns the total physical system memory in bytes.
    CRITICAL: Adheres to workspace rule by exclusively using psutil.
    """
    return psutil.virtual_memory().total


def get_memory_budget() -> MemoryBudget:
    """
    Returns a MemoryBudget namedtuple containing dynamic memory details and calculated safe thread limits.
    """
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    
    avail_gb = vm.available / (1024**3)
    total_ram_gb = vm.total / (1024**3)
    swap_used_gb = sw.used / (1024**3)
    swap_total_gb = sw.total / (1024**3)
    
    cpu_cnt = os.cpu_count() or 4
    safe_threads = max(1, min(cpu_cnt, int(avail_gb / 1.5)))
    
    if avail_gb < 3.0:
        pressure = "critical"
    elif avail_gb < 6.0:
        pressure = "warn"
    else:
        pressure = "ok"
        
    return MemoryBudget(
        available_gb=avail_gb,
        ram_gb=total_ram_gb,
        swap_used_gb=swap_used_gb,
        swap_total_gb=swap_total_gb,
        safe_thread_count=safe_threads,
        pressure_level=pressure
    )


def check_memory_pressure(threshold_gb: float = 3.0) -> bool:
    """Returns True if dynamic available memory is below threshold_gb."""
    budget = get_memory_budget()
    return budget.available_gb < threshold_gb


def get_recommended_matching_mode(n_images: int, available_gb: float) -> str:
    """
    Determines optimal feature matcher strategy based on image count & available memory.
    Returns: 'exhaustive', 'exhaustive_blocked', or 'sequential'
    """
    pair_count = (n_images * (n_images - 1)) // 2
    
    if n_images > 80 and available_gb < 6.0:
        return "sequential"
    elif n_images > 120 and available_gb < 10.0:
        return "sequential"
    elif n_images > 50 and available_gb < 6.0:
        return "exhaustive_blocked"
    elif pair_count > 3000 and available_gb < 4.0:
        return "sequential"
    return "exhaustive"


def run_safe_subprocess(cmd: list, timeout: float = 30.0, **kwargs) -> tuple:
    """
    Runs a subprocess safely with a fallback termination signal handler
    to prevent dangling zombie processes or memory leaks.
    
    Returns:
        tuple: (returncode, stdout, stderr)
    """
    creationflags = 0
    if sys.platform == 'win32':
        creationflags = subprocess.CREATE_NO_WINDOW

    try:
        # Avoid console window flashing on Windows if needed, but standard defaults are fine.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=creationflags,
            **kwargs
        )
        _active_subprocesses.add(proc)
        
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return proc.returncode, stdout, stderr
        except subprocess.TimeoutExpired:
            # Terminate first, then kill if it doesn't respond
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            raise TimeoutError(f"Subprocess timed out after {timeout} seconds: {' '.join(cmd)}")
        finally:
            _active_subprocesses.discard(proc)
    except Exception as e:
        raise e


def validate_colmap_version(colmap_binary_path: str) -> bool:
    """
    Validates that the COLMAP binary is executable and supports CLI help.
    Throws InitializationBlockError if execution fails.
    """
    if not os.path.exists(colmap_binary_path):
        raise FileNotFoundError(f"COLMAP binary does not exist at: {colmap_binary_path}")
        
    try:
        ret, stdout, stderr = run_safe_subprocess([colmap_binary_path, "help"], timeout=5.0)
    except Exception as e:
        raise InitializationBlockError(f"Failed to execute COLMAP binary: {e}")
        
    if ret != 0:
        raise InitializationBlockError(
            f"COLMAP binary execution failed with exit code {ret}.\n"
            f"Output: {(stdout or '') + (stderr or '')}"
        )
        
    return True


# Populated on the startup worker; importing this module never probes hardware.
from acceleration import AccelerationProfile, discover_profile

acceleration_profile = AccelerationProfile()


def initialize():
    """Inspect devices and the resolved reconstruction tools off the UI thread."""
    global acceleration_profile
    base_dir = (os.path.dirname(sys.executable) if getattr(sys, "frozen", False)
                else os.path.dirname(os.path.abspath(__file__)))
    acceleration_profile = discover_profile(base_dir)


if __name__ == "__main__":
    initialize()
    print(acceleration_profile.describe("auto"))
    for diagnostic in acceleration_profile.diagnostics:
        print(diagnostic)
