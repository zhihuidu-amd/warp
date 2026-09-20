# Vendor-neutral upstream candidates

Running list of fixes found during the HIP port that are defensible to
NVIDIA/warp **on CUDA merits alone**, with no HIP in the diff. This is the
channel that already works: GH-1702 and GH-1865 both merged this way.

Rule for this list: a candidate qualifies only if the bug exists in
`upstream/main` and the fix can be explained without mentioning AMD. HIP may
appear in the PR body as *how we found it*, never as *why it matters*.

Status values: `CONFIRMED` (verified present upstream), `TO VERIFY`,
`REJECTED` (checked, not upstreamable).

---

## CONFIRMED

### 1. `end_capture` leaks capture state when `cudaStreamEndCapture` fails

`warp/native/warp.cu`, upstream lines 3625-3626:

```c
    if (!check_cuda(cudaStreamEndCapture(cuda_stream, &graph)))
        return false;
```

All three other failure returns in this function call `clean_up()`; this one
does not. So a failed `cudaStreamEndCapture` permanently leaks the capture
bookkeeping — the `g_captures` entry, the graph alloc table, and the terminating
EndCapture — and the device stays marked as capturing for the life of the
process.

**Why it's upstreamable:** it is an internal inconsistency in Warp's own error
handling, independent of backend. The fix is one line.

**Honest framing for the PR:** on CUDA this path is hard to reach, because a
failed copy inside a capture leaves the capture recoverable — `test_async`'s own
comment says so ("capture can succeed despite some errors during capture"). We
should not claim a CUDA user is hitting this today. The argument is consistency
and blast radius: when it does fire, the process is unusable, and every sibling
return already guards against exactly that.

**Demonstration (job 67846834, HIP):** reproduced in isolation — an h2d copy
inside a capture with the mempool disabled. The allocation inside the capture is
illegal without memory pools and correctly fails; that part is not the bug. The
bug is that the failed `EndCapture` then leaves the capture bookkeeping and the
outstanding graph allocations unreferenced forever. Mempool enabled survives.

**Do NOT cite job 67846710 for this.** I originally attached the `test_async`
cascade (259 passes, then ~2100 tests failing) to this fix as its demonstration.
Job 67852953 traced that window with `WP_TRACE_CAPTURE=1` and found
`EndCapture calls: 0` — the streams were already invalidated before Warp's
capture entry points ran, so this fix cannot be what fixes that cascade. The
leak is real and reproduces on its own; the suite-wide cascade is a separate,
still-open problem. Claiming the second as evidence for the first is exactly the
overclaim a maintainer would catch.

Fix is already in our tree at `warp/native/warp.cu:3806`, commit `7293165ea`
("Unwind capture state when EndCapture fails").

---

### 2. `test_atomic_cas.py` requires independent thread scheduling, but is registered for every device

Verified 2026-09-14. `warp/tests/test_atomic_cas.py` exists in `upstream/main`
byte-identical to ours (`git diff upstream/main HEAD --` is empty), and all four
registration loops pass `devices=devices` with **no architecture guard**
(upstream lines 285, 289, 292, 295).

Every one of these tests runs a **GPU spinlock**: `n = 100` threads contend for a
single lock with

```python
while wp.atomic_cas(lock, 0, 0, warp_type(0), warp_type(1)) == 1:
    pass
```

The thread that wins the CAS must exit the loop and reach
`spinlock_release_2d()` while its siblings are still spinning. That requires
**independent forward progress between threads of the same warp**. CUDA
guarantees that only from Volta (sm_70) onward; under the older reconvergence
model the winning lane cannot leave the loop until every lane leaves, so it can
never release the lock. The result is a deadlock — the process hangs at zero CPU
rather than failing an assertion.

Warp still emits pre-Volta targets: `warp/_src/build_dll.py:526` lists
`sm_52`, `sm_60`, `sm_61` alongside `sm_70`. On any of those devices this test
module hangs the suite.

**Why it's upstreamable:** the fix is to register these tests only for devices
that provide independent thread scheduling, which is a statement about CUDA
compute capability and needs no mention of AMD. It is also test-only — no
runtime or codegen change.

**Honest limits, both of which belong in the PR body:**

- I have **not** run this on a pre-Volta CUDA device. The claim rests on CUDA's
  documented scheduling semantics plus the fact that Warp builds those targets,
  not on a measurement. State it that way.
- Only `test_cas_2d_float32` was *observed* to hang on gfx942. That it is the
  alphabetically first test in the module (`'2' < '3' < '4' < 'f'`, and
  `'f' < 'i' < 'u'` among the 2d dtypes) makes "all of them share this
  dependency, and the runner simply stops at the first" the natural reading —
  but that is an inference from ordering, not a measured result for the other
  fifteen.

### 3. `tile_matmul.h` entry points have no execution-space annotation

**OPEN as NVIDIA/warp#1973** — https://github.com/NVIDIA/warp/pull/1973
Branch `zhihuidu/tile-matmul-cuda-callable`, commit `f736c6af8`, cut from
`upstream/main` `015e5a178`. Opened 2026-09-19. GitHub confirms +4/−4 in one
file. PR body: `C:\tmp\PR-C1-body.md`. Worktree: `C:\tmp\uppr2`.

`tile_matmul`, `tile_matmul_acc`, `adj_tile_matmul_acc`, `adj_tile_matmul` are
called only from generated device code and call `WP_TILE_SYNC()`, but carry no
annotation. They compile only because NVRTC gets
`--device-as-default-execution-space` (upstream `warp/native/warp.cu:4800`).
`wp::index`, `wp::matmul`, `wp::scalar_matmul` in the same file are already
`CUDA_CALLABLE`, so this is an internal-consistency argument that stands without
mentioning any other vendor.

Diff is +4/−4, one token per line, **no comments** — the measured arm carried
three-line comments naming hipcc; those were dropped for the PR. The code change
is byte-identical to what job 67963009 measured.

Left out deliberately: `partition_size`, `partition_coord`, `partition_load` in
the same file are also unannotated and rely on the same option. The PR body says
so and offers to extend.

### 4. `rebuild_find_lowest_on` passes `uint64_t` to `__ffsll` unconverted

**OPEN as NVIDIA/warp#1974** — https://github.com/NVIDIA/warp/pull/1974
Branch `zhihuidu/volume-builder-ffsll-cast`, commit `0e100406c`, cut from
`upstream/main` `015e5a178`. Opened 2026-09-19. GitHub confirms +3/−1 in one
file (1 code line replaced, 2 comment lines added).
PR body: `C:\tmp\PR-C2-body.md`.

`warp/native/volume_builder.h:250`. Explicit `static_cast<unsigned long long>`;
no-op on every 64-bit target. Two-line comment, vendor-neutral wording.

Neither PR carries a changelog fragment. `changelog/README.md` needs one only
when a change is observable to users in API, runtime, packaging, documentation
or a supported workflow; both are no-ops on every configuration upstream
supports. Each body states the reasoning and offers to add one.

## Prepared branches (not pushed)

**STALE as of 2026-09-19.** The worktree at `C:\tmp\up-pr` that held
`zhihuidu/end-capture-unwind` (`df3bd97b1`) and `zhihuidu/skip-cas-without-its`
(`cfc6e658d`) has been deleted; those SHAs no longer resolve. Both branches
would have to be cut again from the current `upstream/main` (`015e5a178`).

The anchors below were re-verified against `upstream/main` at `6931005e1` and
should be re-checked before re-cutting — line numbers will have moved.

Pushing and opening PRs is outward-facing and needs explicit approval first.

Exact upstream anchors, re-verified against `upstream/main` at `6931005e1`:

- `wp_cuda_graph_end_capture` begins at `warp/native/warp.cu:3499`; the
  `clean_up` lambda is defined at `:3532`. Of the four failure returns *after*
  that definition, three call `clean_up()` and one — the `cudaStreamEndCapture`
  check at `:3624` — does not. The two returns at `:3507` and `:3514` precede
  the lambda and touch no bookkeeping, so they are correctly unguarded; the PR
  body should say so, because a reviewer will count six returns and ask.
- The CAS fix follows an idiom upstream already uses. `warp/tests/test_atomic.py:337`
  reads `[d for d in devices if not d.is_cuda or d.arch >= 80]` for bfloat16.
  Ours is the same shape with `>= 70`, which makes it a convention match rather
  than a new pattern to argue for.
- No changelog fragment for the CAS branch: `changelog/README.md` names
  "test-only changes" as internal maintenance that does not need one.

The upstream CAS variant **must not** contain the `device.is_hip` clause our
tree carries. `is_hip` does not exist upstream at all (`grep -c is_hip
warp/_src/context.py` → 0 on `upstream/main`), so that line would be a
`AttributeError` there, not merely off-topic.

## TO VERIFY

**C1 and C2 moved to CONFIRMED 3 and 4 above; branches cut 2026-09-19. Only C3
remains here.**

**MEASURED 2026-09-19, job 67963009** (H100 80GB HBM3, CUDA 13.1, upstream/main
`015e5a178`, `C:\tmp\nvcap\nvcand.sbatch`). Two arms from one tarball, **per-arm
`WARP_CACHE_PATH`** so each JIT-compiled its own 64 kernels; `warp.so` md5s differ
(`6ba2280b…` vs `ec3c4698…`), so both arms genuinely rebuilt.

| | base | fix |
|---|---|---|
| AOT nvcc distinct warning/error lines | 3 (all packman) | 3, **identical** |
| JIT NVRTC distinct warning/error lines | 5 | 5, **identical** |
| kernels JIT-compiled | 64 | 64 |
| 8 tile/volume modules | Ran 273, errors=3, skipped=53, 109.4 s | Ran 273, errors=3, skipped=53, 107.1 s |

**C1 and C2 pass and are proposable.** The three errors are `*_cpu` volume tests
failing on `'NoneType' object has no attribute 'wp_compile_cpp'` — `build_lib.py`
exited 1 in **both** arms because `tools/packman/packman` is absent on that node,
so `warp-clang.so` was never built. That is a CPU-path harness gap identical
across arms, not a result.

**C3 is not decidable here.** The run only shows its assertion does not misfire on
a CUDA build. Its *benefit* is unmeasurable on CUDA by construction. Hold it.

Original framing, kept for the reasoning:

| | File | Change | Risk being measured |
|---|---|---|---|
| C1 | `warp/native/tile_matmul.h` | `CUDA_CALLABLE` on 4 functions that call `WP_TILE_SYNC()` | `CUDA_CALLABLE` expands to `__host__ __device__` under nvcc (`builtin.h:16-22`), so this instantiates a `__syncthreads()` caller on the host. If nvcc emits a new diagnostic, C1 must not be proposed in this form — the fallback is `CUDA_CALLABLE_DEVICE`. |
| C2 | `warp/native/volume_builder.h` | `static_cast<unsigned long long>` on the `__ffsll` argument | Should be codegen-neutral; CUDA declares one overload, HIP two, so only HIP sees the ambiguity. |
| C3 | `warp/_src/build_dll.py` | raise if a compiled `.cu` never reaches the linker | **Weak.** The failure mode it guards is one our HIP branch introduces. `_obj_tag` exists upstream (`:599`) so it ports cleanly, but the rationale is HIP-specific and the odds are low. |

**Pass criterion is the compiler-diagnostic diff, not the test count.** Identical
`warp.so` md5s would be a legitimate outcome for C1/C2 (both can be codegen-neutral),
which is exactly why the tests cannot decide this — the normalised per-arm
`warning|error:` diff can. And the diff must cover **both** compile stages:
`builtin.h:2916` (upstream `:2862`) includes `tile_matmul.h` so nvcc *parses* it AOT, but the
templates are only *instantiated* by NVRTC at JIT time, which is where a
`__host__ __device__` function calling `__syncthreads()` would actually be
diagnosed. Run 67962966 measured only the AOT half and served the fix arm the
base arm's cubins (213 s vs 17 s) — see
[[warp-module-hash-ignores-native-headers]].

**Not in the patch, deliberately:** our tree is missing a `WP_TILE_SYNC()` that
upstream added inside `adj_tile_matmul_acc` (upstream `tile_matmul.h:514`). The
merge-base `ee46f991` lacks it and so does our HEAD, so that is **staleness, not
a change of ours** — it must never appear in a PR diff, and we owe it back at the
next upstream merge.

---

## REJECTED

### Invalidated-capture three-state tracking

**Rejected on a measurement, 2026-09-19 (job 67962785, H100 80GB HBM3, CUDA 13.1,
upstream/main `015e5a178`).** Supersedes an earlier 2026-09-14 entry here that
claimed "there is no such code" — that check grepped for `CAPTURE_NONE` /
`CaptureState`, names that never existed in either tree, and so found nothing.
The code is real; it is spelled with the CUDA runtime's own enum.

`cudaStreamCaptureStatus` has three values: `None`, `Active`, `Invalidated`.
Upstream handles two:

- `warp/native/warp.cu:3351`, `wp_cuda_stream_is_capturing`:
  `return int(status != cudaStreamCaptureStatusNone);` — an **Invalidated**
  capture reports as capturing.
- `warp/native/warp.cu:2925`, `wp_cuda_context_check`:
  `if (status == cudaStreamCaptureStatusNone) { cudaDeviceSynchronize(); }` —
  silently **skips** the synchronize on an invalidated capture.

Our fork fixes both (`:3465`, `:3021`) by testing `== / != cudaStreamCaptureStatusActive`.
On ROCm this matters a great deal: one aborted capture poisoned the process for
3615 subsequent errors (job 67841213).

**On CUDA the two arms differ in exactly one observable cell:**

| point | base | fix |
|---|---|---|
| inside the invalidated window, raw / Warp | Invalidated / **1** | Invalidated / **0** |
| after the failed `capture_end`, raw | None | None |
| `synchronize_device` / 50 allocs / re-capture | ok / 0 fail / ok | ok / 0 fail / ok |
| 4 CUDA test modules, one process | Ran 1917, OK, 0 ERROR | Ran 1917, OK, 0 ERROR |

`cudaStreamEndCapture` **terminates** the invalidated capture and returns the
stream to `None`, and Warp's `capture_end` has already cleared `device.captures`
by then. The unpatched build therefore recovers identically. The Invalidated
state is reachable only between the aborting call and `capture_end` — a window
no user code observes.

Two reasons not to propose it:

1. Benefit measured as **zero** on the backend upstream cares about. "The code
   reads wrong" is not a bug report.
2. The `context_check` half would *change* CUDA behaviour in the risky
   direction: it starts calling `cudaDeviceSynchronize()` inside an invalidated
   window where upstream deliberately skips it. Proposing a behaviour change
   with no demonstrated benefit is how a PR gets closed.

Keep both as internal HIP fixes. The A/B harness is `C:\tmp\nvcap\nvcap.sbatch`
plus `capprobe.py`, which records the raw `cudaStreamIsCapturing` status
alongside Warp's answer at four points — that is what made the verdict analytic
rather than a judgement call.

### `array.h` — `byte_offset_helper` missing `CUDA_CALLABLE`

**Already fixed upstream.** Checked 2026-09-19. I listed this as a live
candidate off a three-dot diff (`upstream/main...HEAD`), which shows only what
*we* changed since the merge-base and says nothing about what upstream has done
since. The two-dot diff settles it:

    git show upstream/main:warp/native/array.h | grep -n 'byte_offset_helper'
    831:inline CUDA_CALLABLE size_t byte_offset_helper(...)

Our only remaining delta in that file is a comment. Nothing to send.

Same sweep found two more of our fixes already landed upstream independently:

- the tile-reduce lane mask (`1 << lane` → 64-bit) — upstream `0a155515f`, GH-1936
- the `__shared__` raw-storage fix
- upstream `tile.h:6-18` already carries our lane-mask block verbatim, merged as
  PR #1865, which shrinks the remaining seam ask to four `#ifndef` guards.

**Rule this cost us:** size a fork delta with a three-dot diff; **classify** a PR
candidate with a two-dot diff plus a grep of upstream's own copy for the symbol
you intend to add.

### `hipGraphAddMemFreeNode` during capture

Checked 2026-09-14. This is a ROCm defect, not a Warp bug, and the probe
(job 67878631) isolates it properly — the call was run four ways and the only
variable that changed the outcome was whether the target graph came from an
active stream capture:

    hand-built graph, free depends on alloc  -> 0
    hand-built graph, zero dependencies      -> 0
    capture graph, capture-frontier deps     -> 1 (hipErrorInvalidValue)
    capture graph, zero dependencies         -> 1

Not a dependency-list problem and not a missing binding. Warp is adding the free
node at a point CUDA explicitly permits — it is how the CUDA path works today and
passes. Nothing to fix upstream. Worth filing against ROCm.

**Weak sub-candidate, also rejected.** On failure Warp prints a warning and then
`g_graph_allocs.erase(alloc_iter)`, dropping its record of an allocation the
graph still owns — the allocation leaks and the only signal is stderr. Upstream
has the identical code. But unlike candidate 1, this is **not** an internal
inconsistency: all three failure returns in that block erase, so the pattern
reads as deliberate rather than as one path that forgot. Without a maintainer's
intent to point to, there is no argument here that does not reduce to "we would
have written it differently."

### Segmented-sort offset iterator (`sort.cu`)

Checked 2026-09-14. Upstream already uses `thrust::make_counting_iterator` /
`thrust::make_transform_iterator`, which is the CCCL-sanctioned spelling and is
**not** broken by CCCL 3.x / CUDA 13 — only the `cub::`-namespaced iterators
were removed there. Upstream also already has the `ValidatedSegmentOffset`
bounds check.

Our change collapses the `bool IsBegin` template parameter into a member so both
offset iterators share one type, which hipCUB's single `OffsetIteratorT`
requires. That is a pure ROCm accommodation with no CUDA-side justification.

I initially believed this one fixed a real CUDA 13 break. It does not.

### Synchronous temp allocator in `sort.cu`

Behind `WP_ENABLE_HIP`; CUDA still takes `mempool_supported`. No CUDA-visible
change by construction.

### Conditional-region predication, capture-resume re-key

Both measured no-ops on CUDA — generated CUDA source is byte-identical
(job 67932933), and the two capture ids never diverge on CUDA. Nothing to
justify upstream on its own.
