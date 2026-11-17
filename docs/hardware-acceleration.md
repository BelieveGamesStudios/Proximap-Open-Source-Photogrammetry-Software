# Reconstruction hardware acceleration

The selector always offers Auto-Detect and CPU. Individual GPUs appear when a
native runtime detects the device and an installed reconstruction tool has a
verified compatible backend and explicit device-selection switches. Integrated
and unified-memory GPUs qualify under the same rules as dedicated GPUs.

Auto chooses one GPU, prioritizing the number of supported stages and using the
stable device identifier to break ties. An explicit selection uses that same
physical GPU for every supported stage. Other stages use CPU; the application
does not substitute a second GPU. The description under the selector identifies
GPU stages and CPU stages before processing starts. Runtime logs record completed
stage backends separately from the requested plan.

## Supported adapters

| Runtime/backend | Discovery | Reconstruction stages |
| --- | --- | --- |
| NVIDIA CUDA | CUDA driver API | COLMAP features/matching, OpenMVS depth maps/refinement, where compiled in |
| AMD HIP/ROCm | HIP runtime API | COLMAP features/matching, where compiled in |
| Metal | Native Metal device enumeration | OpenMVS depth maps, where compiled in |
| CPU | Always available | All implemented reconstruction stages |

OpenGL is not offered as an individually selectable GPU backend because its
current tool integration does not establish which physical GPU will execute the
stage. This does not affect viewport rendering. DirectML, Vulkan and OpenCL are
not implemented reconstruction adapters. Detecting a GPU does not install its
drivers or add acceleration to a CPU-only tool binary.

Native driver discovery runs in a child process with an eight-second timeout.
Tool help probes have five-second timeouts, run off the UI thread during startup,
and resolve binaries using the same rules as pipeline execution. Existing GPU
visibility masks are preserved. A matrix benchmark is not used to infer support.

## Failures and sessions

An eligible GPU backend failure triggers one CPU retry of that stage, with the
reason shown in the log. If the CPU retry fails, processing stops with an error
and preserves existing checkpoints. Cancellation, timeouts, memory-protection
stops, launch errors and unrelated processing failures do not trigger GPU-to-CPU
retries. Failed densification/refinement no longer silently lowers output quality.
Missing tools produce an error rather than simulated reconstruction success.

Saved selections use stable device IDs rather than dropdown positions or labels.
Old CPU selections migrate to CPU; old Auto/Force GPU preferences migrate to
Auto because they did not identify a physical device. A saved GPU that is no
longer usable resolves to CPU with an explanation.

## Bundling backend capabilities

Some tools advertise a specific compiled backend in their version/help output.
Older OpenMVS `--cuda-device` switches also establish a CUDA build. A generic
`--gpu-device` switch alone cannot distinguish Metal from CUDA. Such ambiguous
builds default to CPU until the packager supplies build metadata.

For these builds, place an optional `toolchain_capabilities.json` next to
`toolchain_map.json` (next to the application executable in packaged builds):

```json
{
  "tools": {
    "DensifyPointCloud": {
      "sha256": "<SHA-256 of the actual resolved DensifyPointCloud binary>",
      "backends": ["metal"],
      "device_order": "MTLCopyAllDevices"
    },
    "colmap": {
      "sha256": "<SHA-256 of the actual resolved colmap binary>",
      "backends": ["hip"]
    }
  }
}
```

Generate metadata from the actual build configuration and validate it on the
target platform. Do not infer it from the GPU installed on the packaging machine.
Supported backend names are `cuda`, `hip`, and `metal`; adapters limit each tool
to its implemented backends. A matching hash and advertised command switches
are both required. Replacing a binary invalidates its metadata until the hash is
updated. The packaging scripts copy this optional metadata automatically.

For multiple Metal devices, the verified depth-map tool metadata must additionally
confirm that its device indices follow `MTLCopyAllDevices`. If that mapping is
unknown, the devices remain unavailable for individual selection. A single Metal
device has an unambiguous index, but still requires proof of a Metal tool build.

Build/driver discovery cannot prove every GPU kernel will execute successfully.
CUDA, HIP and Metal builds need real hardware testing on their target platforms;
unit tests verify planning, command construction, persistence migration, and
fallback rules. No reconstruction binaries are rebuilt or downloaded by this
change.
