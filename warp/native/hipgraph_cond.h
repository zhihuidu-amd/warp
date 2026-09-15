/*
 * hipgraph_cond.h — GPU-side conditional execution for HIP graphs.
 *
 * Provides the API surface of CUDA 12.4's conditional graph nodes
 * (cudaGraphConditionalHandle / cudaGraphSetConditional / CU_GRAPH_COND_TYPE_WHILE)
 * for AMD ROCm, where no such API exists in any released version.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS DOES, AND WHAT IT DOES NOT DO
 * ---------------------------------------------------------------------------
 * CUDA's conditional node makes the GPU command processor *skip* or *re-dispatch*
 * a child graph. No AMD GPU can do that today: the required CP microcode support
 * does not exist (AMD's own proof-of-concept for conditional AQL packets,
 * ROCm/rocm-systems PR #7152, reports HSA_STATUS_ERROR_INVALID_PACKET_FORMAT
 * because "vendor packet types 5 and 6 are not yet recognised by CP microcode").
 *
 * So this library does NOT skip nodes. It lowers a conditional loop to a
 * *predicated unroll*: the body is embedded max_iters times, and every body
 * kernel early-returns once the condition goes false. The condition is evaluated
 * and propagated entirely on the GPU -- there is no host round trip, which is the
 * property that matters for graph replay.
 *
 *   CUDA:  condition false -> body is not dispatched at all.
 *   here:  condition false -> body is dispatched but every kernel returns
 *                             immediately (a few microseconds of dispatch).
 *
 * These are observationally equivalent whenever the body is idempotent-once-done,
 * which is the normal shape of a convergence-checked iterative solver. They are
 * NOT equivalent if your body has side effects that must not run after
 * convergence -- see hipGraphCondSetGuard() for how to make a body safe.
 *
 * The API is deliberately CUDA-shaped so that call sites need not change if AMD
 * later ships real conditional nodes. When that happens the lowering changes
 * underneath; hipGraphCondGetLowering() reports which tier is in use.
 *
 * ---------------------------------------------------------------------------
 * VERIFIED PLATFORM FACTS (MI325X gfx942, ROCm 7.2, hipRuntime 70226015)
 * ---------------------------------------------------------------------------
 *   hipGraphConditionalHandleCreate ....... absent from libamdhip64.so
 *   hipGraphInstantiateFlagDeviceLaunch ... rejected (hipErrorInvalidValue)
 *   hipStreamWaitValue32 during capture ... SILENTLY ESCAPES THE GRAPH,
 *                                           executes eagerly, then deadlocks
 *   hipStreamBatchMemOp / AddBatchMemOpNode  hipErrorInvalidValue
 *   hipGraphAddChildGraphNode ............. works; 8x embed replays correctly
 *   hipStreamBeginCaptureToGraph .......... works
 *
 * Because hipStreamWaitValue32 reports hipSuccess inside capture while quietly
 * running outside the graph, any implementation built on it corrupts silently.
 * This library never calls it.
 */

#ifndef HIPGRAPH_COND_H
#define HIPGRAPH_COND_H

/* Guarded for Warp: this header pulls in HIP headers, and Warp's CPU-only
 * build compiles every entry in build_lib.py cpp_sources. Including it from
 * a non-HIP translation unit must be inert rather than a compile error.
 * Added for vendoring; keep other edits out so the file stays diffable
 * against upstream hipgraph_cond. */
#if defined(WP_ENABLE_HIP) && WP_ENABLE_HIP

#include <hip/hip_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Types
 * ------------------------------------------------------------------------ */

/** Opaque handle to a condition slot owned by a graph. Mirrors
 *  cudaGraphConditionalHandle. Valid until hipGraphCondHandleDestroy(). */
typedef struct hipGraphCondHandle_st* hipGraphCondHandle;

/** Which lowering strategy is actually in effect. Query with
 *  hipGraphCondGetLowering() when you need to reason about cost or semantics. */
typedef enum hipGraphCondLowering {
    /** Body embedded max_iters times; body kernels predicated off after the
     *  condition goes false. Pure GPU, no host round trip. Iterations after
     *  convergence still cost a dispatch. This is what runs today. */
    hipGraphCondLoweringPredicatedUnroll = 0,

    /** Reserved: native conditional nodes, once ROCm gains CP support.
     *  Semantics would become identical to CUDA (true skip). */
    hipGraphCondLoweringNative = 1,

    /** Opt-in: graph split into chunks, host inspects the condition between
     *  chunk replays. Fewer wasted dispatches, but reintroduces a host round
     *  trip -- only worth it when max_iters is large and convergence is early. */
    hipGraphCondLoweringChunkedReplay = 2,
} hipGraphCondLowering;

/** Loop form. Mirrors CU_GRAPH_COND_TYPE_*. */
typedef enum hipGraphCondType {
    hipGraphCondTypeIf = 0, /**< run body at most once, if condition != 0 */
    hipGraphCondTypeWhile = 1, /**< run body while condition != 0, up to max_iters */
} hipGraphCondType;

/** Default value written into the condition slot at the start of each replay. */
typedef enum hipGraphCondAssign {
    hipGraphCondAssignDefault = 0, /**< default 1 == "enter the loop" */
    hipGraphCondAssignZero = 1, /**< default 0 == "skip the loop" */
} hipGraphCondAssign;

/* ---------------------------------------------------------------------------
 * Handle lifetime
 * ------------------------------------------------------------------------ */

/**
 * Create a condition handle bound to @p graph.
 *
 * Counterpart of cudaGraphConditionalHandleCreate. The handle owns a small
 * device-resident condition slot. @p defaultValue selects what the slot is reset
 * to at the start of every replay -- without a reset, a graph that converged on
 * replay N would still be "done" on replay N+1.
 *
 * @param handle_out   receives the handle
 * @param graph        graph the handle belongs to; must outlive the handle
 * @param defaultValue value the slot resets to each replay
 * @param flags        reserved, pass 0
 */
hipError_t hipGraphCondHandleCreate(
    hipGraphCondHandle* handle_out, hipGraph_t graph, hipGraphCondAssign defaultValue, unsigned int flags
);

/**
 * Release a handle. Safe on NULL.
 *
 * The handle's condition slot is NOT freed here. The slot has never been an
 * independent allocation: it is an interior pointer into the pool
 * (`g_pool.base + index`), so calling hipFree on it frees the whole slab out
 * from under every other live handle. That is exactly what this function used
 * to do, and it is the root cause of the `cond`-config SIGSEGV and of the HIP
 * context being unable to allocate 4 bytes afterwards.
 *
 * Whether the slot becomes reusable depends on what the handle was used for:
 *
 *  - NEVER spliced into a graph (a support probe, an aborted capture, an error
 *    unwind): nothing references the slot, so it returns to the free list and
 *    the next handle may reuse it.
 *  - SPLICED into a graph: the slot's device address is baked into kernel nodes
 *    that outlive this call -- an instantiated hipGraphExec_t keeps replaying
 *    them, and HIP provides no way to learn when it is destroyed. The slot is
 *    retired permanently. Reissuing it would let two graphs share one condition
 *    word, producing wrong iteration counts with no error at all.
 *
 * So size the pool for the number of conditional CAPTURES the process will
 * perform, not for its peak concurrent handle count. Destroying handles is
 * still worthwhile -- it frees the host record and lifts the region-reference
 * restriction below -- but it does not reclaim a spliced slot.
 *
 * @return hipErrorIllegalState if a conditional region on some stream still
 *         references this handle (close it with hipGraphCondEnd,
 *         hipGraphCondEndWithGraph or hipGraphCondAbortRegion first);
 *         hipErrorInvalidValue on a double destroy or a garbage pointer.
 */
hipError_t hipGraphCondHandleDestroy(hipGraphCondHandle handle);

/**
 * Free the condition-slot slab.
 *
 * The correct way to give the device memory back, and the only one. Returns
 * hipErrorIllegalState while any handle is live, any region is open, or any
 * slot has ever been spliced into a graph -- in each case something may still
 * write through an address in the slab.
 *
 * That last condition makes this a one-way latch in practice: once the process
 * has performed a single conditional capture, the slab stays reserved for the
 * life of the process. HIP cannot report when a hipGraphExec_t is destroyed, so
 * the alternative to being conservative here is a memory fault. The slab is a
 * few hundred bytes; most callers should never call this.
 *
 * Must not be called during a capture (it frees device memory).
 */
hipError_t hipGraphCondPoolRelease(void);

/**
 * Discard the conditional region open on @p stream without splicing it.
 *
 * The recovery path for a capture that failed between hipGraphCondBegin and
 * hipGraphCondEnd. Without it the region stayed in the table forever, holding a
 * body stream stuck in capture, and every subsequent hipGraphCondBegin on that
 * stream failed with hipErrorIllegalState -- turning one transient error into a
 * permanently wedged stream across a repeated capture loop.
 *
 * Idempotent: returns hipSuccess when no region is open.
 */
hipError_t hipGraphCondAbortRegion(hipStream_t stream);

/**
 * Pre-reserve condition slots.
 *
 * Condition slots live in a slab because device allocation is illegal while a
 * stream capture is active -- calling hipMalloc mid-capture returns
 * hipErrorStreamCaptureIsolation (906) and poisons the entire capture. Since
 * handles are naturally created inside a capture (that is where the loop is,
 * and it is how CUDA's equivalent API is used), the slab must be warmed first.
 *
 * A default slab is created on the first hipGraphCondHandleCreate that happens
 * outside a capture. Call this explicitly if you will create more than the
 * default 64 handles, or if your very first handle is created during a capture.
 *
 * The slab is never resized once live, so reserve for the peak. Calling this
 * with a larger count after a slab already exists returns hipErrorOutOfMemory
 * rather than silently succeeding with the smaller slab -- outstanding handles
 * hold interior pointers, so the slab cannot be moved.
 *
 * SIZE THIS FOR THE NUMBER OF CONDITIONAL CAPTURES, not the peak concurrent
 * handle count. A slot spliced into a graph is retired for the life of the
 * process (see hipGraphCondHandleDestroy for why); only never-spliced handles
 * return their slots. A training loop that captures once and replays many times
 * needs one slot; one that re-captures every epoch needs one per epoch.
 */
hipError_t hipGraphCondPoolReserve(unsigned int slots);

/**
 * How many condition slots are currently reserved, and how many are free.
 *
 * Lets a caller check that the pool exists *before* opening a capture. This
 * matters because the pool is grown lazily on first
 * hipGraphCondHandleCreate(), and if that first call happens inside an active
 * capture the allocation is rejected (hipError 906 / capture isolation) and
 * takes the whole capture with it. Either argument may be NULL.
 *
 * @p used_out is the number of slots that are UNAVAILABLE: slots held by live
 * handles plus slots retired to graphs. It is not the live-handle count and not
 * a high-water mark. `used == capacity` is exactly the condition under which
 * hipGraphCondHandleCreate returns hipErrorOutOfMemory, which is what makes it
 * usable as a pre-capture check.
 */
hipError_t hipGraphCondPoolStatus(unsigned int* capacity_out, unsigned int* used_out);

/**
 * Device address of the condition slot.
 *
 * This is the escape hatch for callers who want to write the condition from
 * their own kernel rather than from an int32 array, i.e. the direct analogue of
 * calling cudaGraphSetConditional() inside device code. Write 0 to stop the
 * loop, non-zero to continue. Use a release-ordered store.
 */
hipError_t hipGraphCondHandleGetDevicePtr(hipGraphCondHandle handle, unsigned int** ptr_out);

/* ---------------------------------------------------------------------------
 * Building a conditional region during stream capture
 * ------------------------------------------------------------------------ */

/**
 * Open a conditional region on a capturing stream.
 *
 * Call sequence:
 *
 *     hipGraphCondBegin(stream, handle, hipGraphCondTypeWhile, cond_array, 10);
 *     hipGraphCondGetBodyStream(stream, &body);
 *     ... launch the loop body on `body` exactly once ...
 *     hipGraphCondEnd(stream);
 *
 * The body is captured once on a private stream (so the caller's capture is
 * never interrupted), then embedded @p max_iters times into the parent graph,
 * each embedding preceded by a condition-update kernel that copies @p condition
 * into the handle's slot. The body must therefore be safe to run repeatedly --
 * which it already is if its kernels honour the guard (see
 * hipGraphCondSetGuard).
 *
 * @param stream     a stream with an ACTIVE capture
 * @param handle     handle from hipGraphCondHandleCreate on the capturing graph
 * @param type       If or While
 * @param condition  device int32; != 0 means "keep going". The body is
 *                   responsible for driving this to 0.
 * @param max_iters  hard bound on body embeddings. Required: without real
 *                   conditional nodes there is no unbounded loop. For a solver
 *                   this is the iteration budget you already have.
 *
 * @return hipErrorIllegalState if @p stream is not capturing,
 *         hipErrorInvalidValue on bad arguments.
 */
hipError_t hipGraphCondBegin(
    hipStream_t stream, hipGraphCondHandle handle, hipGraphCondType type, const int* condition, unsigned int max_iters
);

/**
 * Stream the loop body must be launched on, between Begin and End.
 *
 * The body is captured separately from the parent so that the parent's capture
 * is never interrupted -- ending the parent capture to make room for the body
 * would finalize the parent graph. Launch the body here, exactly once.
 *
 * @return hipErrorIllegalState if no region is open on @p stream.
 */
hipError_t hipGraphCondGetBodyStream(hipStream_t stream, hipStream_t* body_out);

/** Close the region opened by hipGraphCondBegin and splice it into the parent. */
hipError_t hipGraphCondEnd(hipStream_t stream);

/**
 * Close the region using a body graph the caller captured itself.
 *
 * Same as hipGraphCondEnd, except the body comes from @p body_graph instead of
 * the private stream handed out by hipGraphCondGetBodyStream. Use this when the
 * caller's framework already owns the body-capture dance and cannot be made to
 * launch onto a stream we chose.
 *
 * This is the path Warp needs. Warp's capture_while does not launch the body on
 * a stream it is given -- it redirects its OWN stream into a body graph via
 * capture_pause/capture_resume, then hands the finished graph back. Forcing it
 * onto our private stream would mean rewriting that logic in Python, so instead
 * we accept the graph it already produced.
 *
 * The caller retains ownership of @p body_graph: it is cloned, not consumed, so
 * destroying it afterwards is safe. The private body stream created by
 * hipGraphCondBegin is discarded without being captured into.
 *
 * @param stream      the stream with the open region (capture still ACTIVE)
 * @param body_graph  a finished graph holding the loop body exactly once
 */
hipError_t hipGraphCondEndWithGraph(hipStream_t stream, hipGraph_t body_graph);

/**
 * Override the iteration bound of the region currently open on @p stream.
 *
 * hipGraphCondBegin takes max_iters up front, but a framework may only learn
 * the real budget while capturing the body. Call this any time before End.
 */
hipError_t hipGraphCondSetMaxIters(hipStream_t stream, unsigned int max_iters);

/**
 * Emit a condition-update kernel on @p stream.
 *
 * The analogue of calling cudaGraphSetConditional() from device code. Normally
 * hipGraphCondBegin/End insert these for you; call it directly only when you
 * want the condition refreshed at a specific point inside the body.
 */
hipError_t hipGraphCondSetCondition(hipStream_t stream, hipGraphCondHandle handle, const int* condition);

/**
 * Emit a kernel on @p stream that writes the logical complement of
 * @p condition into @p handle's slot.
 *
 * This exists for the if/else shape. An else-branch region has to test
 * `!condition`, and the complement must be evaluated as a graph node so that it
 * is recomputed on every replay -- reading the condition on the host would bake
 * one capture-time value into the graph forever.
 *
 * It must also be evaluated ONCE, up front, before the if-branch region is
 * spliced: the two regions are spliced sequentially into the parent, so an
 * if-body that writes @p condition would otherwise change what the else region
 * sees. Emitting this node ahead of both regions is what makes the pair behave
 * like CUDA's single conditional node with two bodies.
 *
 * @p handle is used purely as storage here -- it is never passed to
 * hipGraphCondBegin. Pass its device pointer
 * (hipGraphCondHandleGetDevicePtr) as the `condition` argument of the
 * else region's Begin. Like Begin, this bakes the slot address into the
 * caller's graph, so the slot is retired and will not be recycled.
 */
hipError_t hipGraphCondWriteComplement(hipStream_t stream, hipGraphCondHandle handle, const int* condition);

/* ---------------------------------------------------------------------------
 * Predication support
 * ------------------------------------------------------------------------ */

/**
 * Bind a guard pointer that body kernels read to decide whether to no-op.
 *
 * This is what makes predicated lowering correct rather than merely cheap.
 * Every kernel in the body must begin with the HIPGRAPH_COND_GUARD prologue
 * below. Frameworks that generate their own kernels (Warp, Triton, JAX) should
 * inject it automatically when lowering a conditional region; hand-written
 * kernels add it themselves.
 *
 * A body whose kernels do not honour the guard still produces correct results
 * *if* it is naturally idempotent after convergence (mujoco_warp's solver is:
 * every kernel returns early on a per-world `done` flag), but it saves no work.
 *
 * @param guard_out receives the device pointer to pass to body kernels
 */
hipError_t hipGraphCondSetGuard(hipGraphCondHandle handle, unsigned int** guard_out);

/**
 * Declare which region lexically encloses this one, so nesting stays exclusive.
 *
 * Call this BEFORE hipGraphCondBegin. Passing the guard of the enclosing region
 * (obtained from hipGraphCondSetGuard on the enclosing handle) makes every seed
 * that writes this handle's slot compute
 *
 *     slot = (*condition != 0) && (*enclosing != 0)
 *
 * instead of `(*condition != 0)`. Pass NULL, or simply do not call this, for a
 * top-level region; the behaviour is then exactly as before.
 *
 * WHY THIS IS REQUIRED FOR NESTING. Body kernels are predicated, but the seed
 * kernels that WRITE the guards cannot be -- something has to make the first
 * write. When a region nests, the enclosing body graph is cloned into the parent
 * and the inner region's seed nodes are cloned with it. Those clones execute
 * unconditionally, so without this conjunction they re-arm the inner guard from
 * the user's condition array even when the enclosing guard is 0, and the inner
 * body runs inside a branch that was not taken. On gfx942 that produced a silent
 * wrong answer (5005 where 385 was correct) rather than any error.
 *
 * The conjunction composes to arbitrary depth: each enclosing guard was itself
 * written by a seed that ANDed with its own enclosing guard.
 *
 * The value is stored on the handle, not on the open region, because
 * hipGraphCondEnd's unroll emits further seeds between body copies and they need
 * the same treatment.
 *
 * @param enclosing guard word of the enclosing region, or NULL for top level
 */
hipError_t hipGraphCondSetEnclosingGuard(hipGraphCondHandle handle, const unsigned int* enclosing);

/**
 * The guard prologue. Put this as the first statement of every body kernel.
 *
 *     __global__ void my_solver_step(const unsigned int* guard, ...) {
 *         HIPGRAPH_COND_GUARD(guard);
 *         ... real work ...
 *     }
 *
 * ---------------------------------------------------------------------------
 * WHY A RELAXED LOAD, AND WHY THAT IS NOT A CORRECTNESS BUG
 * ---------------------------------------------------------------------------
 * The obvious spelling of this check is an acquire load. Do not use one. On
 * CDNA an acquire load lowers to an L1 invalidate per wave, and since every
 * thread in the body evaluates the guard, the invalidates serialize the entire
 * dispatch. Measured on MI210 gfx90a, 524288 threads, 1M-element body:
 *
 *     no guard (floor) ....... 0.0092 ms    1.00x
 *     ACQUIRE atomic load .... 0.6701 ms   72.98x     <-- catastrophic
 *     RELAXED atomic load .... 0.0127 ms    1.38x
 *     plain load ............. 0.0095 ms    1.03x
 *
 * The acquire is unnecessary. The guard is written by hgc_set_condition in one
 * kernel and read by body kernels in a *different* kernel, and the two are
 * always separated by a graph edge. An AQL kernel-dispatch boundary already
 * carries system-scope release on completion and acquire on launch, including
 * an L2 writeback-invalidate -- that is what makes ordinary kernel-to-kernel
 * data flow work at all. Re-acquiring inside the kernel re-establishes an
 * ordering the dispatch barrier has already established.
 *
 * A relaxed atomic load keeps the access indivisible (so no torn read) while
 * dropping the redundant fence. That is exactly the guarantee needed here.
 */
#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC__)
#define HIPGRAPH_COND_GUARD(guard_ptr)                                        \
    do {                                                                      \
        if ((guard_ptr) &&                                                    \
            __atomic_load_n((guard_ptr), __ATOMIC_RELAXED) == 0u) return;      \
    } while (0)
#else
#define HIPGRAPH_COND_GUARD(guard_ptr) ((void)0)
#endif

/**
 * Block-wide guard, for bodies with large blocks.
 *
 * Reads the guard once per block into LDS instead of once per thread. Costs a
 * __syncthreads(), so it only pays off when the relaxed per-thread load shows up
 * in a profile -- with a relaxed load the per-thread version is already within
 * ~1.4x of the unguarded floor, so prefer HIPGRAPH_COND_GUARD unless measured
 * otherwise. Must be reached by all threads in the block (it barriers).
 */
#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC__)
#define HIPGRAPH_COND_GUARD_BLOCK(guard_ptr)                                  \
    do {                                                                      \
        __shared__ unsigned int hgc_live_;                                    \
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)         \
            hgc_live_ = (guard_ptr)                                           \
                ? __atomic_load_n((guard_ptr), __ATOMIC_RELAXED) : 1u;        \
        __syncthreads();                                                      \
        if (hgc_live_ == 0u) return;                                          \
    } while (0)
#else
#define HIPGRAPH_COND_GUARD_BLOCK(guard_ptr) ((void)0)
#endif

/* ---------------------------------------------------------------------------
 * Introspection
 * ------------------------------------------------------------------------ */

/** Which lowering the current platform selected. */
hipError_t hipGraphCondGetLowering(hipGraphCondLowering* lowering_out);

/** True if the platform has real conditional nodes (i.e. lowering == Native).
 *  Today this returns false on every shipping ROCm. */
int hipGraphCondIsNativeSupported(void);

/** Human-readable description of the active lowering and why it was chosen. */
const char* hipGraphCondGetLoweringDescription(void);

/** Number of body embeddings the last region emitted -- for tests and for
 *  reporting how much dispatch overhead the unroll costs. */
hipError_t hipGraphCondGetLastUnrollCount(unsigned int* count_out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* WP_ENABLE_HIP */

#endif /* HIPGRAPH_COND_H */
