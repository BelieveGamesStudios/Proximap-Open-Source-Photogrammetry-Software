"""Capability based reconstruction acceleration, independent of Qt and UI state.

A driver device and an explicit backend build marker are both required. Generic
``use_gpu``/``gpu-device`` switches alone never prove a compiled backend.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field, replace
from typing import Mapping

STAGES = ('feature_extractor', 'feature_matching', 'DensifyPointCloud', 'RefineMesh')
STAGE_LABELS = dict(zip(STAGES, ('feature extraction', 'feature matching', 'depth maps', 'mesh refinement')))


def normalize_selection(value):
    value = str(value or 'auto').strip()
    if value.startswith('gpu:'):
        return value
    if value.lower() in ('cpu', 'force_cpu', 'force cpu fallback', 'cpu fallback'):
        return 'cpu'
    return 'auto'


@dataclass(frozen=True)
class Device:
    id: str
    name: str
    vendor: str
    integrated: bool = False
    indices: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class StageCapability:
    stage: str
    backend: str
    tool: str = ''
    use_gpu_flag: str = ''
    device_flag: str = ''
    cpu_device: str = '-2'


@dataclass(frozen=True)
class StageExecution:
    stage: str
    backend: str = 'cpu'
    index: int | None = None
    use_gpu_flag: str = ''
    device_flag: str = ''
    cpu_device: str = '-2'

    @property
    def accelerated(self):
        return self.backend != 'cpu'

    def arguments(self):
        args = []
        if self.use_gpu_flag:
            args.extend((self.use_gpu_flag, '1' if self.accelerated else '0'))
        if self.device_flag:
            args.extend((self.device_flag, str(self.index) if self.accelerated else self.cpu_device))
        return args

    def cpu(self):
        return replace(self, backend='cpu', index=None)

    def environment(self):
        # Explicit flags enforce CPU; inherited visibility masks remain untouched.
        return {}


@dataclass(frozen=True)
class ExecutionPlan:
    selection: str
    device: Device | None
    stages: Mapping[str, StageExecution]
    diagnostics: tuple[str, ...] = ()

    def describe(self):
        if self.device is None:
            return 'CPU: all reconstruction stages.'
        gpu = ', '.join(f'{STAGE_LABELS[k]} ({v.backend.upper()})' for k, v in self.stages.items() if v.accelerated)
        cpu = [STAGE_LABELS[k] for k, v in self.stages.items() if not v.accelerated]
        cpu.extend(('camera reconstruction', 'mesh reconstruction', 'texturing'))
        return f'{self.device.name}: {gpu}. CPU: {", ".join(cpu)}.'


@dataclass(frozen=True)
class AccelerationProfile:
    devices: tuple[Device, ...] = ()
    capabilities: tuple[StageCapability, ...] = ()
    diagnostics: tuple[str, ...] = ()

    def _coverage(self, device):
        return sum(any(c.stage == stage and c.backend in device.indices for c in self.capabilities) for stage in STAGES)

    def options(self):
        options = [('Auto-Detect', 'auto'), ('CPU', 'cpu')]
        for device in sorted(self.devices, key=lambda d: (d.name, d.id)):
            if self._coverage(device):
                backends = sorted({c.backend.upper() for c in self.capabilities if c.backend in device.indices})
                options.append((f'{device.name} ({" / ".join(backends)})', f'gpu:{device.id}'))
        return options

    def resolve(self, selection='auto'):
        selection = normalize_selection(selection)
        candidates = [d for d in self.devices if self._coverage(d)]
        diagnostics = list(self.diagnostics)
        if selection == 'auto':
            device = min(candidates, key=lambda d: (-self._coverage(d), d.id), default=None)
        elif selection == 'cpu':
            device = None
        else:
            device = next((d for d in candidates if f'gpu:{d.id}' == selection), None)
            if device is None:
                diagnostics.append('The saved GPU is unavailable or unsupported; using CPU.')
        stages = {}
        for stage in STAGES:
            caps = [c for c in self.capabilities if c.stage == stage]
            supported = next((c for c in caps if device and c.backend in device.indices), None)
            # Keep advertised CPU switches even when a particular GPU cannot run it.
            control = supported or next(iter(caps), None)
            stages[stage] = StageExecution(
                stage, supported.backend if supported else 'cpu',
                device.indices[supported.backend] if supported else None,
                control.use_gpu_flag if control else '', control.device_flag if control else '',
                control.cpu_device if control else '-2')
        return ExecutionPlan(selection, device, stages, tuple(diagnostics))

    def describe(self, selection='auto'):
        return self.resolve(selection).describe()


def resolve_toolchain(base_dir, mapping=None):
    """Return the same nested map used by PipelineWorker, with absolute paths."""
    base = Path(base_dir)
    if mapping is None:
        try:
            mapping = json.loads((base / 'toolchain_map.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            mapping = {}
    result = {}
    for group, binaries in mapping.items():
        if not isinstance(binaries, dict):
            continue
        result[group] = {}
        for name, configured in binaries.items():
            path = str(configured)
            if sys.platform != 'win32' and path.lower().endswith('.exe'):
                path = path[:-4]
            absolute = str(base / path)
            result[group][name] = absolute if os.path.isfile(absolute) else (shutil.which(name) or shutil.which(os.path.basename(path)) or absolute)
    return result


def _probe(path, arguments):
    env = os.environ.copy()
    # cv2/Qt inject plugin paths incompatible with standalone COLMAP.
    for key in ('QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_FONTDIR', 'QT_PLUGIN_PATH'):
        env.pop(key, None)
    env['QT_QPA_PLATFORM'] = 'offscreen'
    try:
        # OpenMVS opens a log even for --help; do not litter the workspace.
        with tempfile.TemporaryDirectory(prefix='proximap-capability-') as cwd:
            proc = subprocess.run([path, *arguments], cwd=cwd, env=env,
                                  stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, timeout=5, text=True,
                                  errors='replace', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return proc.stdout[:200000] if proc.returncode in (0, 1) else ''
    except (OSError, subprocess.TimeoutExpired):
        return ''


def _backend_markers(text, declared=()):
    found = set()
    # Negative build markers must win over matching text such as 'without CUDA'.
    for backend, tokens in (('cuda', 'CUDA'), ('hip', 'HIP|ROCm'), ('metal', 'Metal')):
        token = f'(?:{tokens})'
        positive = rf'(?:with\s+{token}\b|{token}[ _-](?:ENABLED|SUPPORT)\s*[:=]\s*(?:ON|TRUE|1)|{token}\s+(?:enabled|backend|support enabled))'
        negative = rf'(?:without\s+{token}\b|{token}[ _-](?:ENABLED|SUPPORT)\s*[:=]\s*(?:OFF|FALSE|0)|{token}\s+(?:disabled|not supported))'
        if (backend in declared or re.search(positive, text, re.I)) and not re.search(negative, text, re.I):
            found.add(backend)
    return found


def _verified_manifest(base_dir, toolchain):
    """Build metadata is trusted only when tied to the exact executed binary.

    Optional toolchain_capabilities.json:
    {"tools": {"DensifyPointCloud": {"sha256": "...", "backends": ["metal"],
                                     "device_order": "MTLCopyAllDevices"}}}
    Packagers must generate this from the build, never from detected hardware.
    """
    try:
        data = json.loads((Path(base_dir) / 'toolchain_capabilities.json').read_text())['tools']
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    verified = {}
    for binaries in toolchain.values():
        for name, path in binaries.items():
            entry = data.get(name)
            if (not isinstance(entry, dict) or not isinstance(entry.get('backends'), list)
                    or any(not isinstance(backend, str) or backend not in ('cuda', 'hip', 'metal') for backend in entry['backends'])):
                continue
            try:
                digest = hashlib.sha256()
                with open(path, 'rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(chunk)
                if digest.hexdigest() == entry.get('sha256'):
                    verified[name] = entry
            except OSError:
                pass
    return verified


def _capabilities(toolchain, manifest=None):
    caps, diagnostics = [], []
    manifest = manifest or {}
    colmap = toolchain.get('colmap', {}).get('colmap')
    if colmap:
        banner = _probe(colmap, ['help'])
        if not banner:
            diagnostics.append(f'COLMAP could not be inspected at {colmap}; verify its executable and runtime dependencies.')
        backends = _backend_markers(banner, manifest.get('colmap', {}).get('backends', [])) & {'cuda', 'hip'}
        for stage, command, prefix in (('feature_extractor', 'feature_extractor', 'SiftExtraction'), ('feature_matching', 'exhaustive_matcher', 'SiftMatching')):
            help_text = _probe(colmap, [command, '--help'])
            # Current COLMAP also uses FeatureExtraction/FeatureMatching prefixes.
            prefixes = (prefix, 'FeatureExtraction' if stage == 'feature_extractor' else 'FeatureMatching')
            chosen = next((p for p in prefixes if f'--{p}.use_gpu' in help_text and f'--{p}.gpu_index' in help_text), None)
            if chosen:
                for backend in sorted(backends or {'cpu'}):
                    caps.append(StageCapability(stage, backend, colmap, f'--{chosen}.use_gpu', f'--{chosen}.gpu_index', '-1'))
            else:
                diagnostics.append(f'COLMAP {command} has no verified GPU enable/index flags; using CPU.')
        if not backends:
            diagnostics.append('COLMAP has no verified CUDA/HIP build marker; its feature stages use CPU. OpenGL is not exposed because physical device selection cannot be guaranteed.')
    else:
        diagnostics.append('COLMAP is missing from the toolchain mapping; no feature acceleration is available.')
    for stage in ('DensifyPointCloud', 'RefineMesh'):
        path = toolchain.get('openMVS', {}).get(stage)
        if not path:
            diagnostics.append(f'{stage} is missing from the toolchain mapping.')
            continue
        text = _probe(path, ['--help'])
        if not text:
            diagnostics.append(f'{stage} could not be inspected at {path}; verify its executable and runtime dependencies.')
        # https://github.com/cdcseacave/openMVS/blob/develop/apps/DensifyPointCloud/DensifyPointCloud.cpp
        # Old --cuda-device is compiled only with CUDA; generic --gpu-device is ambiguous.
        if '--cuda-device' in text:
            caps.append(StageCapability(stage, 'cuda', path, device_flag='--cuda-device'))
        elif '--gpu-device' in text:
            backends = _backend_markers(text, manifest.get(stage, {}).get('backends', [])) & {'cuda', 'metal'}
            if stage == 'RefineMesh':
                backends &= {'cuda'}  # upstream Metal only implements depth-map estimation
            for backend in sorted(backends or {'cpu'}):
                caps.append(StageCapability(stage, backend, path, device_flag='--gpu-device'))
            if not backends:
                diagnostics.append(f'{stage} exposes a generic GPU switch without a verified backend marker; using CPU.')
        elif text:
            diagnostics.append(f'{stage} has no GPU device selector; this installed build uses CPU.')
    return tuple(caps), diagnostics


def _load_library(names):
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    return None


def _compute_devices(backend):
    """Use runtime-visible ordinals; never overwrite administrator visibility masks."""
    if backend == 'cuda':
        lib = _load_library(('nvcuda.dll', 'libcuda.so.1', 'libcuda.dylib'))
        prefix, vendor = 'cu', 'NVIDIA'
    else:
        lib = _load_library(('amdhip64.dll', 'libamdhip64.so', 'libamdhip64.so.6'))
        prefix, vendor = 'hip', 'AMD'
    if lib is None:
        return []
    def call(name, *args):
        return getattr(lib, name)(*args) == 0
    try:
        if not call(prefix + 'Init', 0):
            return []
        count = ctypes.c_int()
        count_name = 'cuDeviceGetCount' if backend == 'cuda' else 'hipGetDeviceCount'
        if not call(count_name, ctypes.byref(count)):
            return []
        devices = []
        for index in range(min(count.value, 64)):
            handle = ctypes.c_int(index)
            if backend == 'cuda' and not call('cuDeviceGet', ctypes.byref(handle), index):
                continue
            name = ctypes.create_string_buffer(256)
            if not call(prefix + 'DeviceGetName', name, len(name), handle):
                continue
            pci = ctypes.create_string_buffer(64)
            uuid = ctypes.create_string_buffer(16)
            # v2 preserves MIG compute-instance identity (CUDA 11.4+).
            # https://docs.nvidia.com/cuda/archive/12.3.0/cuda-driver-api/group__CUDA__DEVICE.html
            uuid_function = 'cuDeviceGetUuid_v2' if backend == 'cuda' and hasattr(lib, 'cuDeviceGetUuid_v2') else prefix + 'DeviceGetUuid'
            uuid_ok = call(uuid_function, uuid, handle) if hasattr(lib, uuid_function) else False
            pci_ok = call(prefix + 'DeviceGetPCIBusId', pci, len(pci), handle) if hasattr(lib, prefix + 'DeviceGetPCIBusId') else False
            # MIG instances may share a PCI address; UUID preserves their identity.
            if uuid_ok and any(bytes(uuid)):
                identity = backend + ':' + bytes(uuid).hex()
            elif pci_ok and pci.value:
                identity = 'pci:' + pci.value.decode().lower()
            else:
                continue  # Ordinals and display names are not persistent identities.
            integrated = ctypes.c_int()
            attribute = 'cuDeviceGetAttribute' if backend == 'cuda' else 'hipDeviceGetAttribute'
            # CUDA CU_DEVICE_ATTRIBUTE_INTEGRATED=18; HIP hipDeviceAttributeIntegrated=16.
            # https://github.com/ROCm/HIP/blob/develop/include/hip/hip_runtime_api.h
            call(attribute, ctypes.byref(integrated), 18 if backend == 'cuda' else 16, handle)
            devices.append(Device(identity, name.value.decode(errors='replace'), vendor, bool(integrated.value), {backend: index}))
        return devices
    except (AttributeError, OSError, ValueError):
        return []


def _metal_devices():
    if sys.platform != 'darwin':
        return []
    # Native enumeration order is exposed only when the tool build confirms it.
    # A single-device topology is unambiguous without build metadata.
    try:
        metal = ctypes.CDLL('/System/Library/Frameworks/Metal.framework/Metal')
        objc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        send_ptr = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(('objc_msgSend', objc))
        send_int = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p)(('objc_msgSend', objc))
        at_index = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)(('objc_msgSend', objc))
        sel = objc.sel_registerName
        metal.MTLCopyAllDevices.restype = ctypes.c_void_p
        array = metal.MTLCopyAllDevices()
        count = send_int(array, sel(b'count'))
        devices = []
        for index in range(min(count, 64)):
            device = at_index(array, sel(b'objectAtIndex:'), index)
            registry_id = send_int(device, sel(b'registryID'))
            string = send_ptr(device, sel(b'name'))
            name_ptr = send_ptr(string, sel(b'UTF8String'))
            name = ctypes.string_at(name_ptr).decode()
            devices.append(Device(f'metal:{registry_id:x}', name, 'Apple' if 'Apple' in name else 'Metal', bool(send_int(device, sel(b'hasUnifiedMemory'))), {'metal': index}))
        send_ptr(array, sel(b'release'))
        return devices
    except (AttributeError, OSError, ValueError):
        return []


def _inventory():
    devices = {}
    ambiguous = set()
    for device in (*_compute_devices('cuda'), *_compute_devices('hip'), *_metal_devices()):
        if device.id in ambiguous:
            continue
        prior = devices.get(device.id)
        if prior:
            if any(backend in prior.indices and prior.indices[backend] != index for backend, index in device.indices.items()):
                # A legacy UUID/PCI collision cannot safely identify one device.
                ambiguous.add(device.id)
                devices.pop(device.id)
                continue
            device = Device(device.id, device.name, device.vendor, device.integrated, {**prior.indices, **device.indices})
        devices[device.id] = device
    return tuple(devices.values())


def _print_inventory(output_path=None):
    data = json.dumps([asdict(device) for device in _inventory()])
    if output_path:
        Path(output_path).write_text(data, encoding='utf-8')
    else:
        print(data)


def _bounded_inventory():
    try:
        with tempfile.TemporaryDirectory(prefix='proximap-devices-') as directory:
            output_path = str(Path(directory) / 'devices.json')
            command = ([sys.executable, '--acceleration-inventory', output_path] if getattr(sys, 'frozen', False)
                       else [sys.executable, str(Path(__file__).resolve()), '--acceleration-inventory', output_path])
            proc = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, timeout=8,
                                  creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if proc.returncode == 0:
                values = json.loads(Path(output_path).read_text(encoding='utf-8'))
                if not isinstance(values, list) or len(values) > 128:
                    return ()
                devices = []
                for value in values:
                    if not isinstance(value, dict):
                        return ()
                    if any(not isinstance(value.get(key), str) or not value[key] for key in ('id', 'name', 'vendor')):
                        return ()
                    indices = value.get('indices')
                    if (not isinstance(indices, dict)
                            or any(backend not in ('cuda', 'hip', 'metal')
                                   or type(index) is not int or index < 0 or index > 63
                                   for backend, index in indices.items())
                            or type(value.get('integrated', False)) is not bool):
                        return ()
                    devices.append(Device(**value))
                return tuple(devices)
    except (OSError, subprocess.TimeoutExpired, ValueError, TypeError):
        pass
    return ()


def discover_profile(base_dir, mapping=None):
    toolchain = resolve_toolchain(base_dir, mapping)
    manifest = _verified_manifest(base_dir, toolchain)
    caps, diagnostics = _capabilities(toolchain, manifest)
    devices = _bounded_inventory()
    metal_devices = [d for d in devices if 'metal' in d.indices]
    if len(metal_devices) > 1:
        metadata = manifest.get('DensifyPointCloud', {})
        if metadata.get('device_order') != 'MTLCopyAllDevices':
            devices = tuple(replace(d, indices={b: i for b, i in d.indices.items() if b != 'metal'}) for d in devices)
            diagnostics.append('Multiple Metal GPUs detected; the build must confirm MTLCopyAllDevices index mapping before individual devices can be selected.')
    if not devices:
        diagnostics.append('No usable CUDA/HIP runtime device or unambiguous Metal device was detected.')
    for device in devices:
        if not any(cap.backend in device.indices and cap.backend != 'cpu' for cap in caps):
            diagnostics.append(f'{device.name} was detected, but no installed reconstruction stage has verified support for its available backends.')
    diagnostics.append('Build and runtime discovery cannot prove every GPU kernel is compatible; GPU execution failures are reported and retried on CPU.')
    return AccelerationProfile(devices, caps, tuple(diagnostics))


if __name__ == "__main__" and "--acceleration-inventory" in sys.argv:
    _print_inventory(sys.argv[sys.argv.index('--acceleration-inventory') + 1] if len(sys.argv) > sys.argv.index('--acceleration-inventory') + 1 else None)
