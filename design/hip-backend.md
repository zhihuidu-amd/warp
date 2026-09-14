# HIP Backend for AMD GPUs

**Status**: Proposed

**Issue**: [GH-XXXX](https://github.com/NVIDIA/warp/issues/XXXX)

## Motivation

Warp targets CUDA devices only. A growing share of physical-AI and robotics research
runs on AMD Instinct hardware, and several groups have asked for Warp on ROCm because
their simulation stacks (notably MuJoCo Warp) depend on it. Today those users maintain
private forks, which diverge quickly and duplicate effort.

This document proposes adding HIP as a second GPU backend, so that a single upstream
tree supports both CUDA and ROCm devices.

The intent is explicitly *not* to make Warp vendor-neutral in its public API. Warp's
CUDA vocabulary (`wp.get_cuda_device()`, `cuda:0`, `is_cuda`) stays as it is. HIP is
implemented as an alternative code-generation and runtime target underneath that
vocabulary, in the same way `WP_ENABLE_CUDA` already gates the CUDA layer.

## Requirements

| ID  | Requirement | Priority | Notes |
| --- | --- | --- | --- |
| R1  | `WP_ENABLE_HIP=0` builds are byte-identical to builds without this change | Must | Verifiable by hash; the core review guarantee |
| R2  | No new required dependency for CUDA-only users | Must | ROCm toolchain is opt-in |
| R3  | Translation concentrated in one header, not scattered at call sites | Must | Bounds long-term maintenance cost |
| R4  | Capability differences expressed at runtime where possible | Must | So NVIDIA CI compiles and exercises the code paths |
| R5  | Fewer than 30 preprocessor guards added to existing files | Should | Measurable proxy for R3. **Met: 21 guards across 9 existing files** (plus 11 in 6 new HIP-only files, which carry no merge risk) |
| R6  | AMD-hosted CI runners for the HIP path | Should | NVIDIA CI cannot test AMD hardware |
| R7  | Public API and type signatures unchanged where possible | Should | One known exception, see "Compile architecture" |

**Non-goals**:

- Vendor-neutral renaming of the public API.
- Feature parity on day one. Tile/MMA paths, graph conditional nodes, and some
  interop surfaces are staged separately (see "Staging").
- Support for AMD consumer (RDNA) parts in the first increment; CDNA (`gfx942`)
  is the initial target.
- Deterministic reduction on ROCm (see Key Implementation Details).
- Any change to CUDA behavior, performance, or codegen.

## Design

### Approach

Two mechanisms, applied to different classes of difference.

**1. A backend gate, mirroring `WP_ENABLE_CUDA`.**
`warp/native/cuda_util.h` already wraps the entire CUDA layer in
`#if WP_ENABLE_CUDA` ... `#endif`. HIP follows the same shape with `WP_ENABLE_HIP`,
selected at build time. This is the existing convention in this codebase, not a new
pattern.

**2. A translation header, `warp/native/hip_util.h`.**
The large majority of CUDA/HIP differences are vocabulary: `cudaMalloc` ->
`hipMalloc`, `CUDA_SUCCESS` -> `hipSuccess`, `nvrtcGetPTX` -> `hiprtcGetCode`. These
are resolved by aliases in a single new header (~150 `#define`s plus a few `static
inline` shims where the signatures differ). Because the file is new, it carries no
merge risk against upstream refactors.

Differences are then routed by kind:

| Kind of difference | Where it is handled |
| --- | --- |
| Name, enum, or signature aliasing | `hip_util.h` |
| Genuinely different semantics | Named inline helper, called unconditionally |
| Capability present on one platform only | Runtime query, not `#if` |
| Backend selection, toolkit version gates | `#if` (the preprocessor must decide) |

The third row matters most for maintenance. For example, HIP has no equivalent of CUDA
graph conditional nodes. Expressing that as `device.supports_conditional_graphs()`
rather than `#ifdef` means the branch is compiled, type-checked, and exercised by
NVIDIA's existing CI, even though the false case only occurs on AMD hardware.

### Alternatives Considered

**Call-site `#ifdef` guards.** The natural first implementation, and the one our
working fork used: guard each divergence where it occurs. Measured on that fork, this
produced **275 HIP preprocessor guards across 77 native files**, 83 of them bare
`#if defined(__HIP_PLATFORM_AMD__)` in `cuda_util.cpp` alone.

Rejected for two measured reasons:

1. *Merge cost.* Over a recent four-month window, upstream rewrote 693 of 1867 lines
   in `warp/native/tile.h` alone. Every guard planted in an actively edited file is a
   recurring conflict and a line of unrelated code the maintainer must reason about.

2. *Silent rot.* NVIDIA CI cannot compile the HIP branches, so defects in them are
   invisible upstream. This is not hypothetical: in our fork, `hip_util.h` defined
   `cudaStreamSetCaptureDependencies` as a literal `0` while the correct alias sat
   below it, unreachable. The SET spelling silently resolved to the ADD value. It
   compiled cleanly and ran with wrong dependency semantics. Concentrating this class
   of risk in one auditable file is preferable to distributing it across 275 sites
   that cannot be tested.

**A full backend abstraction layer (virtual dispatch over device APIs).** Cleanest in
principle, but it would restructure code that is currently direct CUDA calls, touching
far more upstream lines than the feature justifies and imposing an indirection cost on
the CUDA path, which must stay unchanged (R1). Rejected as disproportionate.

**Separate fork, not upstreamed.** The status quo. Rejected because it diverges: our
fork accumulated 476 commits of drift in four months, and roughly 75-85% of its diff
turned out to be drift rather than AMD-specific work.

### Key Implementation Details

**Build system.** `build_dll.py` gains ROCm discovery (`ROCM_PATH`, then `hipconfig`,
then `hipcc` on `PATH`, then `/opt/rocm`) and invokes `hipcc` with `--offload-arch`
instead of `nvcc` with `-gencode`. One caution learned in practice: resolving the ROCm
prefix from `which hipcc` is wrong on distributions that ship `/usr/bin/hipcc`, since
the headers are not under `/usr/include`. The prefix must be validated by probing for
`include/hip/hip_runtime.h` before use.

**Compile architecture is a string on HIP.** Warp types the compile architecture as an
`int` (`sm_90` -> `90`). AMD architectures are strings (`gfx942`). This is the one
place the proposal changes an existing signature: the annotation widens to
`int | str`. Measured blast radius is **12 annotations, 11 of them in
`warp/_src/context.py`**. It is called out explicitly because it touches
`get_cuda_compile_arch()`, and because mismatches here fail confusingly: passing an
`int` where HIP expects `"gfx942"` produces an arch-not-found error far from its cause.

**Empty arrays.** On ROCm, a zero-size allocation may return a non-NULL pointer, so
`ptr is None` is not a reliable emptiness test. Warp already prefers `size == 0` for
this after [GH-1702]. No further change is needed here; it is noted because it is a
recurring source of platform-specific bugs.

**Deterministic reduction is not built on ROCm.** `deterministic.cu` needs
`__CUDACC__` for Warp's own device code (`wp::half`'s operators, `wp::min`/`max`
in `builtin.h`), but rocThrust keys on the same macro in
`thrust/detail/config/compiler.h` and then selects `THRUST_DEVICE_COMPILER_NVCC`,
falling back to a CPU tag whose iterator operators are host-only. Every use of a
transform iterator in device code then fails to compile.

The macro cannot be worked around from the build: `THRUST_DEVICE_COMPILER` has no
`#ifndef` guard, and the tag is baked into the iterator type when it is formed, so
setting `THRUST_DEVICE_SYSTEM` afterwards does not help. The file is therefore
excluded from HIP builds and the feature reports as unavailable, as it already
does on the CPU. This looks like a rocThrust issue worth reporting upstream to
AMD rather than something to work around in Warp.

**Warp size is 32 on CUDA and 64 on AMD.** This is the one difference that is not
a naming or API question, and it reaches into algorithm code.
`warp/native/tile_reduce.h` hardcodes `#define WP_TILE_WARP_SIZE 32`, and the
warp-level primitives assume a lane mask fits in 32 bits, e.g. in `sparse.cu`:

```c
constexpr unsigned int full_warp_mask = 0xffffffffu;
const unsigned int keep_mask = __ballot_sync(full_warp_mask, run_start);
```

On ROCm a wavefront is 64 lanes, so `__ballot_sync` returns a 64-bit mask and HIP
rejects a 32-bit one outright:

    static assertion failed: The mask must be a 64-bit integer. Implicitly
    promoting a smaller integer is almost always an error.

Making this portable means WP_TILE_WARP_SIZE becoming architecture-dependent and
the mask type widening with it, across 44 uses in four files
(`tile_reduce.h`, `tile_scan.h`, `tile_radix_sort.h`, `sparse.cu`). Because it
changes shared-memory sizing and lane arithmetic in tuned code, it was staged
separately rather than folded into the enablement work.

**This is now done and upstream.** [GH-1865] parameterized the warp size and
widened the lane masks, and it merged on its own merits with no HIP in the diff
— it is a portability fix, not an AMD feature. Correctness on `gfx942` is
verified for tile sort across 8 of 9 sizes. One subtlety worth recording,
because it is the kind of thing that looks like a bug on review: the bitonic
shuffle is bounded by *stride*, not by lane, so a block-wide caller reaches the
"subwarp" shuffle path, and it is `stride <= 16` — not the wavefront width —
that makes a 32-wide shuffle safe there.

**Graph capture.** ROCm supports stream capture but not conditional graph nodes.
Warp's conditional-node paths are therefore gated on a runtime capability query, and
`capture_save()` is unavailable on HIP because the `.wrp` format does not encode `gfx`
architectures.

`wp.capture_while()` is lowered as a **predicated static unroll**: the body graph is
cloned into the parent up to 32 times, with a device-side condition refresh between
copies. The clones must self-skip, or a loop whose condition is false on entry still
runs its body 32 times — measured, that returned an array scaled by 2^32 with no
warning. Warp-generated kernels therefore take a hidden trailing
`const unsigned int* _wp_cond_guard` and return immediately when it reads zero. The
load is **relaxed, not acquire**: a per-thread acquire load measured a 73x dispatch
penalty on CDNA, and the graph edge from the refresh node already supplies the
ordering. The null test comes first, so a kernel launched outside any region pays a
register compare rather than a memory load.

Three categories of body content cannot be predicated this way, and are refused or
warned about rather than left to run silently: a body passed as an already-captured
`Graph` (its kernel nodes were recorded outside the region and cannot be rebound),
non-kernel nodes such as `memcpy`/`memset`, and native library kernels, which are
real kernel nodes carrying no guard prologue and so are detectable only by *counting*
kernel nodes against the number of bound conditions.

`wp.capture_if()` remains refused on HIP. No correct lowering exists under an
unconditional unroll — it would execute the branch the condition selected against.
The guard mechanism this work adds is the prerequisite; wiring `hipGraphCondTypeIf`
plus an inverting seed kernel is follow-on work.

CUDA is unaffected by all of the above: the codegen format slot is empty there and the
argument append is gated on the HIP runtime, so generated CUDA source is
byte-identical by measurement.

### Staging

The work is proposed as a sequence of independently reviewable and revertable changes,
rather than a single large one:

0. **Portability prerequisites** — *already merged, no HIP in the diff.*
   [GH-1702] (empty-array and typing fixes) and [GH-1865] (warp-size
   parameterization and lane-mask widening across the tile headers and
   `sparse.cu`). Both landed on their own merits. Nothing in the remaining
   stages re-litigates them.
1. **Enablement** — build system, `hip_util.h`, device init. Produces a Warp that
   builds and runs basic kernels on `gfx942`.
2. **Graph capture** — stream capture, mempool interaction, and the predicated
   unroll for `wp.capture_while()` described above.
3. **Tile / MMA** — the rocWMMA-backed paths.
4. **CI** — AMD-hosted runners.
5. **Examples and documentation.**

Stage 0 is listed to make a point about method rather than to claim credit: the
portability work was separable from the backend, and separating it is what let it
land upstream without anyone having to decide about AMD. We would expect to keep
splitting work that way — anything defensible on CUDA alone goes up as its own PR
as we find it, not batched behind the backend.

## Testing Strategy

**Unchanged-by-default is the primary guarantee.** With `WP_ENABLE_HIP=0` the produced
library must hash identically to a build of the same tree without this feature. This is
checkable in CI without any AMD hardware and is the strongest evidence that CUDA users
are unaffected.

**Existing suite, second backend.** The HIP path runs `warp/tests` unmodified. Tests do
not branch on backend; where a capability is genuinely absent, the runtime query is the
skip condition, so the same test source covers both platforms.

**Hardware coverage.** Initial target `gfx942` (MI300X / MI325X). MI355X (`gfx950`) has
also been exercised by AMD Research. Because NVIDIA CI has no AMD hardware, AMD can
provide hosted runners for the HIP jobs (R6); without them, the HIP path can only be
validated out-of-tree, which reintroduces the silent-rot failure mode described above.

**Known gaps to close before "Implemented".** Numerical agreement between CUDA and HIP
for the tile paths, behavior on RDNA parts, and multi-GPU/IPC surfaces are not yet
characterized.

**There is no clean full-suite number yet, and this document will not round one up.**
`test_conditional_captures.py` passes on `gfx942` and tile sort is verified across 8 of
9 sizes, but the following are open and must be resolved or explicitly waived before any
claim of suite-level health:

- `test_cas_2d_float32` hangs on `gfx942` — zero CPU, not slow.
- `hipGraphAddMemFreeNode` fails when the graph is capturing, though it succeeds on a
  hand-built graph. This looks like a ROCm defect and is the blocker behind the
  `test_graph` failures.
- A single invalidated capture poisons the process, so one genuine error can present as
  thousands. Error *counts* from a HIP suite run are therefore not a meaningful measure
  until the originating event is isolated.

The last point is a measurement hazard as much as a defect: a run reporting 2087 errors
and a run reporting one may be the same event. Any suite number quoted for the HIP path
should state whether cascade suppression was in effect.
