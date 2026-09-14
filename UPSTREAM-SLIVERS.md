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

## Prepared branches (not pushed)

Both CONFIRMED candidates are committed on branches cut from `upstream/main`
in a separate worktree at `C:\tmp\up-pr`, signed off, and verified to contain
**zero** occurrences of hip/amd/rocm in the diff:

| Branch | Commit | Diff |
|---|---|---|
| `zhihuidu/end-capture-unwind` | `df3bd97b1` | `warp.cu`, +8 |
| `zhihuidu/skip-cas-without-its` | `cfc6e658d` | `test_atomic_cas.py`, +8 −1 |

Neither is pushed. Pushing and opening PRs is outward-facing and needs
explicit approval first.

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

*Empty as of 2026-09-14 — every candidate raised so far has been resolved into
CONFIRMED or REJECTED. New candidates land here first.*

---

## REJECTED

### Invalidated-capture three-state tracking

Checked 2026-09-14. There is no such code. `grep` for
`CAPTURE_NONE` / `CAPTURE_INVALIDATED` / `CaptureState` / `capture_state` across
`warp/_src` and `warp/native` returns nothing in our tree — the only hits
anywhere are an unrelated local variable in `warp/tests/test_linear_solvers.py`.

The three-state distinction was a *diagnosis* written down while chasing the
`test_async` poison, not a change that ever landed. Nothing to upstream, and
nothing to compare against upstream. Removed from TO VERIFY rather than left
sitting there implying work exists.

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
