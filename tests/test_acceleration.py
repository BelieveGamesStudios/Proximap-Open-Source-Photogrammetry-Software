import hashlib
import json
from pathlib import Path
import tempfile
import os
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from acceleration import (AccelerationProfile, Device, StageCapability,
                          StageExecution, _backend_markers, _capabilities,
                          _verified_manifest, _compute_devices, _bounded_inventory, _inventory,
                          discover_profile, normalize_selection,
                          resolve_toolchain)


class SelectionTests(unittest.TestCase):
    def test_cpu_and_legacy_selections(self):
        for value in ('CPU', 'force_cpu', 'Force CPU Fallback'):
            self.assertEqual(normalize_selection(value), 'cpu')
        for value in ('gpu', 'force_gpu', 'Force GPU (CUDA)', None):
            self.assertEqual(normalize_selection(value), 'auto')
        self.assertEqual(normalize_selection('gpu:pci:01:00.0'), 'gpu:pci:01:00.0')

    def test_no_driver_is_always_cpu(self):
        profile = AccelerationProfile()
        self.assertEqual(profile.options(), [('Auto-Detect', 'auto'), ('CPU', 'cpu')])
        self.assertIsNone(profile.resolve().device)

    def test_single_device_used_for_every_supported_stage(self):
        nvidia = Device('nvidia', 'NVIDIA', 'NVIDIA', indices={'cuda': 2})
        amd = Device('amd', 'AMD APU', 'AMD', True, {'hip': 1})
        caps = (StageCapability('feature_extractor', 'hip'), StageCapability('feature_matching', 'hip'),
                StageCapability('DensifyPointCloud', 'cuda', device_flag='--cuda-device'))
        profile = AccelerationProfile((nvidia, amd), caps)
        plan = profile.resolve('gpu:amd')
        self.assertEqual(plan.device, amd)
        self.assertEqual(plan.stages['feature_extractor'].index, 1)
        self.assertEqual(plan.stages['DensifyPointCloud'].backend, 'cpu')
        self.assertEqual(plan.stages['DensifyPointCloud'].arguments(), ['--cuda-device', '-2'])
        self.assertEqual(profile.resolve().device, amd)  # coverage outranks packaging
        self.assertIn('CPU:', plan.describe())
        self.assertEqual(len(profile.options()), 4)

    def test_auto_tie_breaker_is_stable(self):
        caps = (StageCapability('DensifyPointCloud', 'cuda'),)
        a = Device('a', 'A', 'NVIDIA', indices={'cuda': 1})
        b = Device('b', 'B', 'NVIDIA', indices={'cuda': 0})
        self.assertEqual(AccelerationProfile((b, a), caps).resolve().device, a)

    def test_missing_selected_device_does_not_switch_to_another_gpu(self):
        device = Device('a', 'A', 'NVIDIA', indices={'cuda': 0})
        profile = AccelerationProfile((device,), (StageCapability('DensifyPointCloud', 'cuda'),))
        plan = profile.resolve('gpu:removed')
        self.assertIsNone(plan.device)
        self.assertTrue(any('saved GPU' in message for message in plan.diagnostics))

    def test_cpu_retry_preserves_explicit_cpu_selector_and_masks(self):
        stage = StageExecution('DensifyPointCloud', 'metal', 0, device_flag='--gpu-device')
        self.assertEqual(stage.arguments(), ['--gpu-device', '0'])
        self.assertEqual(stage.cpu().arguments(), ['--gpu-device', '-2'])
        self.assertEqual(stage.environment(), {})

    def test_unrelated_gpu_not_exposed(self):
        profile = AccelerationProfile((Device('amd', 'AMD', 'AMD', indices={'hip': 0}),),
                                      (StageCapability('DensifyPointCloud', 'cuda'),))
        self.assertEqual(len(profile.options()), 2)


class ProbeTests(unittest.TestCase):
    def test_negative_cuda_marker(self):
        self.assertEqual(_backend_markers('COLMAP (without CUDA)'), set())
        self.assertEqual(_backend_markers('COLMAP (with CUDA)'), {'cuda'})
        self.assertEqual(_backend_markers('HIP_ENABLED=ON'), {'hip'})
        self.assertEqual(_backend_markers('Metal support enabled'), {'metal'})

    def test_generic_gpu_switch_never_implies_cuda_or_metal(self):
        with patch('acceleration._probe', return_value='--gpu-device arg (=-1) GPU device(s)'):
            caps, messages = _capabilities({'openMVS': {'DensifyPointCloud': 'densify'}})
        self.assertEqual(caps[0].backend, 'cpu')
        profile = AccelerationProfile(capabilities=caps)
        self.assertEqual(profile.resolve().stages['DensifyPointCloud'].arguments(), ['--gpu-device', '-2'])
        self.assertTrue(messages)

    def test_old_cuda_specific_switch_is_build_evidence(self):
        with patch('acceleration._probe', return_value='--cuda-device arg (=-1)'):
            caps, _ = _capabilities({'openMVS': {'DensifyPointCloud': 'densify'}})
        self.assertEqual(caps[0].backend, 'cuda')

    def test_build_manifest_requires_binary_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / 'densify'
            binary.write_bytes(b'build-one')
            path = Path(directory) / 'toolchain_capabilities.json'
            metadata = {'tools': {'DensifyPointCloud': {'sha256': hashlib.sha256(binary.read_bytes()).hexdigest(), 'backends': ['metal']}}}
            path.write_text(json.dumps(metadata))
            tools = {'openMVS': {'DensifyPointCloud': str(binary)}}
            self.assertIn('DensifyPointCloud', _verified_manifest(directory, tools))
            binary.write_bytes(b'build-two')
            self.assertEqual(_verified_manifest(directory, tools), {})

    def test_manifest_and_flags_both_required(self):
        metadata = {'DensifyPointCloud': {'backends': ['metal']}}
        tools = {'openMVS': {'DensifyPointCloud': 'densify'}}
        with patch('acceleration._probe', return_value='CPU-only build'):
            self.assertEqual(_capabilities(tools, metadata)[0], ())
        with patch('acceleration._probe', return_value='--gpu-device arg (=-1)'):
            self.assertEqual(_capabilities(tools, metadata)[0][0].backend, 'metal')

    def test_non_native_exe_resolves_to_same_system_tool(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('acceleration.sys.platform', 'linux'), patch('acceleration.shutil.which', return_value='/usr/bin/colmap'):
                tools = resolve_toolchain(directory, {'colmap': {'colmap': 'backend_bin/colmap.exe'}})
            self.assertEqual(tools['colmap']['colmap'], '/usr/bin/colmap')

    def test_ambiguous_multi_metal_topology_is_hidden(self):
        devices = (Device('m1', 'Metal A', 'Apple', indices={'metal': 0}),
                   Device('m2', 'Metal B', 'AMD', indices={'metal': 1}))
        caps = (StageCapability('DensifyPointCloud', 'metal', device_flag='--gpu-device'),)
        with patch('acceleration._bounded_inventory', return_value=devices), patch('acceleration._capabilities', return_value=(caps, [])), patch('acceleration._verified_manifest', return_value={}):
            profile = discover_profile('.')
        self.assertEqual(len(profile.options()), 2)


class NativeInventoryTests(unittest.TestCase):
    def _library(self, backend, *, uuid_available=True):
        prefix = 'cu' if backend == 'cuda' else 'hip'
        def count(pointer):
            pointer._obj.value = 2
            return 0
        def handle(pointer, index):
            pointer._obj.value = index
            return 0
        def name(buffer, size, device):
            buffer.value = b'Identically named GPU'
            return 0
        def uuid(buffer, device):
            buffer.raw = bytes([device.value + 1]) * 16
            return 0 if uuid_available else 1
        def pci(buffer, size, device):
            buffer.value = b'0000:01:00.0'  # MIG partitions may share this
            return 0
        def attribute(pointer, attribute_id, device):
            self.assertEqual(attribute_id, 18 if backend == 'cuda' else 16)
            pointer._obj.value = int(backend == 'hip')
            return 0
        functions = {prefix + 'Init': lambda flags: 0,
                     'cuDeviceGetCount' if backend == 'cuda' else 'hipGetDeviceCount': count,
                     prefix + 'DeviceGetName': name, prefix + 'DeviceGetUuid': uuid,
                     prefix + 'DeviceGetPCIBusId': pci,
                     prefix + 'DeviceGetAttribute': attribute}
        if backend == 'cuda':
            functions['cuDeviceGet'] = handle
        return SimpleNamespace(**functions)

    def test_cuda_uuid_keeps_mig_instances_separate_and_visible_ordinals(self):
        with patch('acceleration._load_library', return_value=self._library('cuda')):
            devices = _compute_devices('cuda')
        self.assertEqual(len(devices), 2)
        self.assertNotEqual(devices[0].id, devices[1].id)
        self.assertEqual([d.indices['cuda'] for d in devices], [0, 1])

    def test_cuda_prefers_mig_aware_uuid_v2(self):
        lib = self._library('cuda')
        lib.cuDeviceGetUuid_v2 = lib.cuDeviceGetUuid
        def legacy(*args):
            self.fail('legacy CUDA UUID API must not be used when v2 exists')
        lib.cuDeviceGetUuid = legacy
        with patch('acceleration._load_library', return_value=lib):
            self.assertEqual(len(_compute_devices('cuda')), 2)

    def test_colliding_identity_is_not_silently_assigned_to_one_device(self):
        devices = [Device('same-pci', 'GPU', 'NVIDIA', indices={'cuda': 0}),
                   Device('same-pci', 'GPU', 'NVIDIA', indices={'cuda': 1})]
        with patch('acceleration._compute_devices', side_effect=(devices, [])), patch('acceleration._metal_devices', return_value=[]):
            self.assertEqual(_inventory(), ())

    def test_hip_uses_native_integrated_attribute_and_visibility_is_preserved(self):
        with patch.dict(os.environ, {'HIP_VISIBLE_DEVICES': '4,2', 'ROCR_VISIBLE_DEVICES': '7'}):
            with patch('acceleration._load_library', return_value=self._library('hip')):
                devices = _compute_devices('hip')
            self.assertTrue(all(device.integrated for device in devices))
            self.assertEqual([d.indices['hip'] for d in devices], [0, 1])
            self.assertEqual(os.environ['HIP_VISIBLE_DEVICES'], '4,2')
            self.assertEqual(os.environ['ROCR_VISIBLE_DEVICES'], '7')

    def test_missing_native_runtime_is_empty(self):
        with patch('acceleration._load_library', return_value=None):
            self.assertEqual(_compute_devices('cuda'), [])
            self.assertEqual(_compute_devices('hip'), [])

    def test_pci_identity_fallback_when_uuid_is_unavailable(self):
        with patch('acceleration._load_library', return_value=self._library('hip', uuid_available=False)):
            devices = _compute_devices('hip')
        self.assertEqual(devices[0].id, 'pci:0000:01:00.0')

    def test_bounded_child_timeout_does_not_escape(self):
        with patch('acceleration.subprocess.run', side_effect=subprocess.TimeoutExpired(['helper'], 8)) as run:
            self.assertEqual(_bounded_inventory(), ())
        self.assertEqual(run.call_args.kwargs['timeout'], 8)

    def test_child_start_failure_does_not_escape(self):
        with patch('acceleration.subprocess.run', side_effect=OSError('missing interpreter')):
            self.assertEqual(_bounded_inventory(), ())

    def test_frozen_child_uses_output_file_and_preserves_visibility(self):
        def run(command, **kwargs):
            self.assertEqual(command[1], '--acceleration-inventory')
            self.assertNotIn('env', kwargs)
            Path(command[2]).write_text(json.dumps([{
                'id': 'uuid:1', 'name': 'GPU', 'vendor': 'NVIDIA',
                'indices': {'cuda': 0}, 'integrated': False}]))
            return SimpleNamespace(returncode=0)
        with patch('acceleration.sys.frozen', True, create=True), patch('acceleration.subprocess.run', side_effect=run):
            devices = _bounded_inventory()
        self.assertEqual(devices[0].indices, {'cuda': 0})

    def test_malformed_child_output_never_creates_device(self):
        for malformed in ('null', '{}', '[null]', '[{"indices": {"cuda": -1}}]',
                          '[{"id":"x","name":"GPU","vendor":"AMD","indices":{"hip":"0"}}]'):
            def run(command, **kwargs):
                Path(command[-1]).write_text(malformed)
                return SimpleNamespace(returncode=0)
            with self.subTest(output=malformed), patch('acceleration.subprocess.run', side_effect=run):
                self.assertEqual(_bounded_inventory(), ())

    def test_malformed_manifest_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = {'openMVS': {'DensifyPointCloud': str(Path(directory) / 'densify')}}
            for data in ({'tools': []}, {'tools': None}, {'tools': {'DensifyPointCloud': {'backends': [{}]}}},
                         {'tools': {'DensifyPointCloud': {'backends': ['vulkan']}}}):
                (Path(directory) / 'toolchain_capabilities.json').write_text(json.dumps(data))
                self.assertEqual(_verified_manifest(directory, tools), {})

    def test_negative_build_marker_overrides_manifest(self):
        tools = {'openMVS': {'DensifyPointCloud': 'densify'}}
        metadata = {'DensifyPointCloud': {'backends': ['metal']}}
        with patch('acceleration._probe', return_value='without Metal --gpu-device arg'):
            caps, _ = _capabilities(tools, metadata)
        self.assertEqual(caps[0].backend, 'cpu')

    def test_missing_and_unprobeable_tools_explain_failure(self):
        with patch('acceleration._probe', return_value=''):
            _, messages = _capabilities({'openMVS': {'DensifyPointCloud': 'broken'}})
        self.assertTrue(any('COLMAP is missing' in m for m in messages))
        self.assertTrue(any('DensifyPointCloud could not be inspected' in m for m in messages))
        self.assertTrue(any('RefineMesh is missing' in m for m in messages))


if __name__ == '__main__':
    unittest.main()
