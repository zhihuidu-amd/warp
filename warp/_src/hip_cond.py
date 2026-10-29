# SPDX-FileCopyrightText: Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HIP-side support for conditional graph regions.

ROCm has no conditional graph node. ``wp.capture_if`` and ``wp.capture_while``
are lowered instead as a *predicated splice*: the body is cloned into the parent
graph and each clone's kernels return early when the region's guard word reads
zero (``warp/native/hipgraph_cond.cu``). Everything in this module exists to make
that lowering safe -- declaring which region encloses which, and detecting
kernel nodes that do not honour a guard.

None of it runs on CUDA, where a conditional node schedules the body for real.

This module is AMD-added and has no upstream counterpart. It is kept separate
from ``warp/_src/context.py`` so that tracking upstream costs a one-line conflict
at a call site rather than a merge through the middle of ``capture_if``. The
corresponding rule is: logic lives here, and ``context.py`` holds only the call.
"""

import ctypes

from warp._src.logger import log_warning

_context = None


def _rt():
    """The live :class:`warp._src.context.Runtime`.

    Resolved on first use rather than imported at module scope: ``context``
    imports this module, and ``runtime`` is rebound by ``context.init()`` well
    after import, so a name bound at import time would be ``None`` forever.
    """
    global _context
    if _context is None:
        from warp._src import context

        _context = context
    return _context.runtime


def _set_enclosing_guard(stream) -> None:
    """Tell the backend which conditional region encloses the next one to open.

    Call immediately before ``wp_cuda_graph_insert_if_else`` or
    ``wp_cuda_graph_insert_while``. The top of :attr:`Runtime._cond_guards` is
    exactly the enclosing region's guard, because each region pushes and pops
    within its own callback; an empty stack is the top level and passes ``None``.

    HIP needs this to keep nested regions exclusive. The lowering there is a
    predicated static unroll: body kernels self-skip on their guard word, but the
    *seed* kernels that write those guards cannot themselves be predicated --
    something has to make the first write. When a region nests, the enclosing body
    graph is cloned into the parent along with the inner region's seeds, and those
    clones run unconditionally, re-arming the inner guard from the user's condition
    array even inside a branch that was not taken. Passing the enclosing guard makes
    each seed write ``(*condition != 0) and (*enclosing != 0)`` instead, which
    composes to any depth.

    A no-op on CUDA, where a conditional node's body is genuinely not scheduled and
    there is no cloned seed to suppress.

    Not raising on failure would be the wrong trade: the consequence is a silent
    wrong answer in an unrelated branch, not a slowdown.
    """
    guards = _rt()._cond_guards
    enclosing = ctypes.c_void_p(guards[-1] if guards else None)
    if not _rt().core.wp_cuda_graph_set_enclosing_guard(stream.cuda_stream, enclosing):
        raise RuntimeError(_rt().get_error_string())


def _warn_unpredicated_library_dispatch(op: str) -> None:
    """Warn when a :mod:`warp._src.utils` operation is called inside a HIP conditional region.

    Every operation that reaches this function allocates a temporary buffer for
    its hipCUB/rocPRIM dispatch, and a conditional body graph cannot allocate. So
    on HIP none of them run inside a region, in one of two ways:

    * ``wp.utils.radix_sort_pairs`` / ``wp.utils.segmented_sort_pairs``
      (``sort.cu``, ``cached_side_alloc``) log ``CUDA error 900: operation not
      permitted when stream is capturing`` and **return normally**, leaving the
      destination unsorted. Nothing raises. This is the dangerous one.
    * ``wp.utils.array_scan`` (``scan.cu``), ``wp.utils.array_sum`` /
      ``wp.utils.array_inner`` (``reduce.cu``) and ``wp.utils.runlength_encode``
      (``runlength_encode.cu``) call ``wp_alloc_device``, which becomes a graph
      allocation node, and ``wp_cuda_graph_check_conditional_body`` raises
      ``Conditional body graph contains an unsupported operation (memory
      allocation)`` when the region closes.

    Measured on gfx942, job 67932870: ``radix_sort_pairs`` error 900 with rc=0,
    ``array_scan`` and ``array_inner`` both raising. ``segmented_sort_pairs`` and
    ``runlength_encode`` were not run, and are grouped by the allocator their
    source shares, not by measurement.

    This corrects what this function used to say. It claimed these were ordinary
    kernel nodes that "run on every unrolled copy regardless of the condition" --
    a reasonable reading of the lowering, but wrong for these operations, because
    the allocation stops them before a single library kernel node is emitted. The
    claim is right in general for a native kernel that does *not* allocate, and
    that case is real: an ``array.fill_()`` in a body puts an unguarded
    ``memtile_value_kernel`` node in the body graph, where it does run on all
    ``max_iters`` copies (same job, ``fill`` arm). :func:`_warn_unguarded_body_kernels`
    is what catches that, by counting, and it needs no list of entry points.

    The docstring also used to say the error-900 defect was "fixed in the native
    library" because ``resume_capture`` re-keys the capture registry. That fix is
    real but addresses plain pause/resume. It does not cover this path: a
    conditional body is captured as a *child* graph, so ``capture->current_id``
    differs from ``capture_id`` by design (``sort.cu:174``), ``acquire_temp_buffer``
    takes ``cached_side_alloc`` anyway, and the 900 still happens.

    Warn rather than refuse: the same body is correct on CUDA, and a refusal would
    reject code that works there.

    The guard stack is never pushed on CUDA and is empty on HIP outside a
    conditional region, so the fast path is one boolean test.
    """
    if _rt()._cond_guards:
        log_warning(
            f"{op}() was called inside a conditional capture region. On HIP this operation allocates a temporary "
            "buffer and a conditional body graph cannot allocate, so it does not run: wp.utils.radix_sort_pairs() "
            "and wp.utils.segmented_sort_pairs() log CUDA error 900 and return with the destination unchanged, "
            "while the scan, reduction and run-length operations make the enclosing region raise when it closes. "
            "Move the call outside the region, or use wp.capture_if()/wp.capture_while() outside of a graph "
            "capture.",
            once=True,
        )


def _warn_unguarded_body_kernels(body_graph: ctypes.c_void_p, launches: int, nested: bool) -> None:
    """Report kernel nodes in a HIP conditional body that Warp did not generate.

    :func:`_warn_unpredicated_library_dispatch` catches the library entry points
    Warp knows about (:mod:`warp._src.utils`). This catches the general case, and
    needs no list: every kernel Warp generates is dispatched through a path that
    binds the guard pointer, so a body graph holding more kernel nodes than the
    region bound guards contains at least that many kernels from somewhere else --
    a native library, a raw driver launch, a third-party module. Those have no
    guard parameter and run on all ``max_iters`` unrolled copies.

    Skipped when a nested region opened inside the body: an inner
    :func:`capture_while` splices its own unroll into this body graph, so the node
    count is a multiple of the inner body and the comparison is meaningless.

    Warn rather than refuse, for the same reason as the memcpy/memset case: the
    answer is still right if the operation is idempotent or the loop runs its full
    bound, and the same body is correct on CUDA.
    """
    unguarded = _count_unguarded_body_kernels(body_graph, launches, nested)
    if unguarded is not None and unguarded > 0:
        log_warning(
            f"The body of this conditional region contains {launches + unguarded} kernel node(s) but Warp launched "
            f"only {launches} guarded kernel(s), so at least {unguarded} kernel(s) came from elsewhere (a native "
            "library, a raw driver launch, or another module). On HIP the region is a predicated static unroll and "
            "only Warp-generated kernels honor the region's guard word, so those kernels run on every unrolled copy "
            "regardless of the condition. The result is correct only if they are idempotent or the loop runs the "
            "full iteration bound.",
            once=True,
        )


def _count_unguarded_body_kernels(body_graph: ctypes.c_void_p, launches: int, nested: bool) -> int | None:
    """Kernel nodes in a HIP conditional body beyond the ones Warp bound a guard to.

    ``None`` when the question cannot be answered -- a nested region, or a failed
    node count -- and the caller must not treat that as zero.

    Shared by :func:`_warn_unguarded_body_kernels` and
    :func:`_check_unguarded_branch_kernels`, which differ only in what they do
    with the number.
    """
    if nested:
        return None

    node_count = ctypes.c_uint64(0)
    if not _rt().core.wp_cuda_graph_count_kernel_nodes(body_graph, ctypes.byref(node_count)):
        # Non-fatal: this is a diagnostic, and failing it must not fail the capture.
        log_warning(f"Could not count the conditional body's kernel nodes: {_rt().get_error_string()}", once=True)
        return None

    return int(node_count.value) - launches


def _check_unguarded_branch_kernels(branch_graph: ctypes.c_void_p, launches: int, nested: bool, branch: str) -> None:
    """Refuse an if/else branch body holding a kernel node Warp did not generate.

    The same measurement as :func:`_warn_unguarded_body_kernels`, raised rather
    than warned, for the reason the two shapes differ. An unguarded kernel in a
    ``capture_while`` body runs on copies the loop would not have executed --
    wasted work, and the right answer if the kernel is idempotent. The same
    kernel in a branch body runs when the condition selected the *other* branch,
    which is a side effect of code the program said not to run. Nothing
    downstream can detect that, so it cannot be a warning.

    Raises :class:`RuntimeError`. A branch that opens a nested region, or a body
    whose nodes could not be counted, is not refused -- see
    :func:`_count_unguarded_body_kernels` for why the number is unavailable
    there.
    """
    unguarded = _count_unguarded_body_kernels(branch_graph, launches, nested)
    if unguarded is not None and unguarded > 0:
        raise RuntimeError(
            f"capture_if(): the {branch} branch contains {launches + unguarded} kernel node(s) but Warp launched only "
            f"{launches} guarded kernel(s), so at least {unguarded} kernel(s) came from elsewhere (a native library, "
            "a raw driver launch, or another module). On HIP a branch is predicated by a guard word its kernels read, "
            "and only Warp-generated kernels read it, so those kernels would run even when the condition selects the "
            "other branch. Move them out of the branch, or express them as Warp kernels."
        )
