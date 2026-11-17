# Hardware acceleration review

Date: 2026-11-17

Scope: hardware discovery, reconstruction backend selection, status reporting,
session persistence, and GPU-to-CPU fallback. This is a review and proposed
refactoring sequence; it does not change runtime behavior. It is not a full
application audit.

Implementation follow-up: the changes and setup requirements are documented in
[hardware-acceleration.md](hardware-acceleration.md). Findings below describe the
code at the time of the initial review; their line numbers are historical.

## Findings

1. **High: dedicated GPU detection is incorrectly treated as CUDA support.**
   `hardware_profiler.py:456` enables the GPU path whenever the dedicated GPU
   list is nonempty, including AMD cards. `pipeline_manager.py:719` uses that
   boolean to report CUDA execution without confirming the tool's build,
   driver compatibility, or usable devices.

2. **High: CPU fallback does not distinguish backend errors from cancellation,
   timeout, or memory pressure.** `pipeline_manager.py:1701` retries on every
   false process result. The runner returns the same result for cancellation
   at line 578, timeout at line 583, and the memory guard at line 591. This can
   attempt another process after a user cancellation or protective shutdown.
   Return a structured outcome and restrict automatic retry to eligible
   backend failures.

3. **High: discovery misses platforms and misclassifies devices.**
   `hardware_profiler.py:271` has no macOS inventory source or generic Linux
   inventory source independent of NVIDIA/ROCm utilities. At line 297 every
   Intel device is classified as integrated; Apple devices are also assigned
   to the integrated category, which does not qualify for acceleration under
   the current policy. Device packaging is not a compute capability check.

4. **Medium: dynamic dropdowns would break mode selection.**
   `main_window.py:5209` maps fixed item positions to modes. Use Qt item data
   with stable backend/device identifiers; labels and order must be independent
   of execution configuration.

5. **Medium: session persistence mixes labels and identifiers.**
   `main_window.py:4880` saves display text, while `pipeline_manager.py:345`
   saves internal mode values. `main_window.py:5780` capitalizes either value
   and searches the dropdown for an exact label, which does not reliably
   restore the selection. Add a versioned representation and migrate legacy
   values; resolve missing devices on the current machine.

6. **Medium: status reporting claims unverified acceleration.**
   `main_window.py:3603` reports an active iGPU whenever no dedicated GPU is
   detected. `pipeline_manager.py:1708` describes a generic GPU attempt as
   OpenGL, while line 2254 reports its result as CUDA. Track detection,
   selection, and actual execution separately, including execution per stage.

7. **Medium: backend policy is duplicated and tied to the UI parent.**
   `pipeline_manager.py:426` and line 709 independently construct hardware
   policy using the parent window's GPU boolean. Workers without that parent
   can make different choices. Pass an explicit execution plan to workers and
   share backend adapters across reconstruction, refinement, and retexturing.

8. **Medium: CPU enforcement only changes CUDA visibility for OpenMVS.**
   `pipeline_manager.py:435` and line 715 clear `CUDA_VISIBLE_DEVICES`; the
   densification command at line 1328 has no explicit device selector. That
   environment setting cannot enforce CPU use for a future Metal backend.
   Use supported, tool-specific CPU/device flags after checking capabilities.

## Proposed design

- Discover device identities and vendor/platform details in background workers
  with bounded probe times. Include supported integrated and unified-memory
  GPUs without requiring a dedicated card.
- Separately inspect the actual resolved tool binaries, supported backends,
  runtime availability, and usable devices. A generic matrix benchmark or GPU
  name is not evidence that a reconstruction tool can use a device.
- Build capabilities per pipeline stage. A device may accelerate one stage
  while other stages use CPU. Metal is an API/backend; Apple GPU is hardware.
- Populate selection from verified usable device/backend combinations. Keep
  CPU available; keep Auto as a policy if desired, resolving to CPU when no
  supported GPU is available. Put unavailable-device explanations in a
  diagnostic view, outside the selectable list.
- Store requested selection separately from the effective plan and actual
  execution outcome. Save stable identifiers, and revalidate on resume.
- Centralize execution policy and backend-specific command construction.
  Preserve cancellation and memory protections across every retry.

## Refactoring sequence and validation

1. Introduce explicit process outcomes; test cancellation, timeout, memory
   protection, backend failure, and eligible CPU retry.
2. Introduce typed device/capability/selection records and legacy selection
   migration; replace positional UI mapping and duplicated policy.
3. Add bounded platform and binary capability probes; exercise simulated CPU,
   NVIDIA, AMD, Apple, integrated, multiple-device, and missing-runtime cases.
4. Populate the UI and stage execution plan from those capabilities; report
   the effective backend accurately.
5. Validate with actual supported tool builds on each target platform. Mocked
   inventories verify selection logic, not real hardware compatibility.

## Product decisions

Confirmed: automatically continue on CPU after an eligible GPU backend failure,
including when the GPU was explicitly selected, and explain the fallback.
Cancellation, memory-guard termination, and unrelated failures must not trigger
automatic GPU-to-CPU retries.

Confirmed: a GPU is selectable when it accelerates only some reconstruction
stages, with a clear indication that the remaining stages use CPU.

Confirmed: on systems with multiple usable GPUs, users select one physical GPU.
Use that device for every stage it can accelerate, using the appropriate
supported backend for each stage. Stages unsupported by that device use CPU;
do not substitute another GPU. Simultaneous use of multiple GPUs is outside
this initial refactor.

Confirmed: Auto chooses one verified usable GPU for the entire run, prioritizing
supported stage coverage with a deterministic tie-breaker. Show the chosen
device before processing starts. Stages that device cannot accelerate use CPU;
do not switch to another GPU. If no GPU qualifies, Auto uses CPU.

Confirmed: if GPU processing fails and its CPU retry also fails, stop
reconstruction, explain the failure, and preserve available checkpoints for
retry. Do not silently proceed with sparse or otherwise reduced-quality output.
