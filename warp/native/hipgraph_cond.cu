/*
 * hipgraph_cond.cpp — predicated-unroll lowering of conditional graph regions.
 *
 * See hipgraph_cond.h for the API contract and for the platform facts that
 * force this design. The short version: AMD has no conditional graph nodes and
 * no device-side graph launch, and hipStreamWaitValue32 silently escapes stream
 * capture, so the only mechanism that both stays on the GPU and survives replay
 * is to embed the body N times and predicate it off.
 *
 * The capture-splicing dance below (pause parent / capture body / add child
 * nodes / resume parent) mirrors what CUDA's own conditional nodes require of
 * their callers, and what Warp does in wp_cuda_graph_insert_while(). Keeping the
 * same shape means a future native lowering is a local change here, not an API
 * break for callers.
 */

// VENDORED from the hipgraph_cond library, unmodified except for the
// WP_ENABLE_HIP guard added around the whole file. Warp's CPU-only build
// (build_lib.py --no-cuda) compiles every entry in cpp_sources, and this file
// includes HIP headers unconditionally, so without the guard stage [C] fails.
// Keep edits to this file limited to the guard so it stays diffable against
// upstream hipgraph_cond.
#if defined(WP_ENABLE_HIP) && WP_ENABLE_HIP

#include "hipgraph_cond.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/* ---------------------------------------------------------------------------
 * Diagnostics
 *
 * Every failure in this library used to be swallowed by a (void) cast, which is
 * how a slab free-after-free and a stale region entry both reached the point of
 * producing a bare SIGSEGV with nothing in the log. Errors now go through
 * HGC_TRY (propagate) or HGC_BEST_EFFORT (teardown, nothing to do but say so),
 * both of which print under HIPGRAPH_COND_VERBOSE=1. Off by default this costs
 * one predictable branch; on, it makes a failure legible BEFORE the crash
 * rather than only after it.
 * ------------------------------------------------------------------------ */
namespace {
bool hgcVerbose()
{
    static int v = -1;
    if (v < 0) {
        const char* s = getenv("HIPGRAPH_COND_VERBOSE");
        v = (s && *s && *s != '0') ? 1 : 0;
    }
    return v != 0;
}
}  // namespace

#define HGC_LOG(...)                                                          \
    do {                                                                      \
        if (hgcVerbose()) {                                                   \
            fprintf(stderr, "[hipgraph_cond] " __VA_ARGS__);                  \
            fputc('\n', stderr);                                              \
        }                                                                     \
    } while (0)

/* Evaluate `expr`; on failure log it (with site) and return the error. */
#define HGC_TRY(expr)                                                         \
    do {                                                                      \
        const hipError_t hgc_e_ = (expr);                                     \
        if (hgc_e_ != hipSuccess) {                                           \
            HGC_LOG("%s:%d: %s -> %d (%s)", __FILE__, __LINE__, #expr,        \
                    (int)hgc_e_, hipGetErrorString(hgc_e_));                  \
            return hgc_e_;                                                    \
        }                                                                     \
    } while (0)

/* Evaluate `expr` on a teardown path where there is nothing useful to do with
 * a failure, but it must still be visible. Replaces the bare (void) casts. */
#define HGC_BEST_EFFORT(expr)                                                 \
    do {                                                                      \
        const hipError_t hgc_e_ = (expr);                                     \
        if (hgc_e_ != hipSuccess)                                             \
            HGC_LOG("%s:%d: %s -> %d (%s) [ignored]", __FILE__, __LINE__,     \
                    #expr, (int)hgc_e_, hipGetErrorString(hgc_e_));           \
    } while (0)

/* ---------------------------------------------------------------------------
 * Device kernels
 * ------------------------------------------------------------------------ */

/* Copy the user's int32 condition into the handle's slot, conjoined with the
 * guard of the enclosing region (if any).
 *
 * Deliberately a kernel and not hipMemcpyAsync: a kernel node is unambiguously
 * capturable and ordered, whereas the stream memory-op path is the one that
 * escapes capture (see header). Single thread; the cost is dispatch, not work.
 *
 * These are the only two places that need real memory ordering, and they cost
 * nothing because exactly one thread executes them. The GUARD READ in body
 * kernels must NOT use acquire -- see HIPGRAPH_COND_GUARD below.
 *
 * WHY `enclosing` EXISTS. A region is lowered as a predicated static unroll:
 * the body graph is cloned into the parent and the body's *Warp* kernels
 * self-skip on the guard word. The seed nodes -- this kernel, hgc_reset_condition,
 * hgc_write_complement -- are NOT predicated, because something has to write the
 * guard in the first place. That is fine at the top level and wrong as soon as
 * regions nest: cloning an outer body graph clones the inner region's seed nodes
 * too, and those clones execute even when the outer guard is 0, re-arming the
 * inner guard straight from the user's untouched condition array. The inner body
 * then runs inside a branch that was not taken.
 *
 * Measured on gfx942 before this conjunction, test_complex_capture with
 * cond1=0, cond2=0 returned 5005 where 385 is correct -- exactly 385 x 13, the
 * 13x kernel being the inner else body of the branch that was suppressed.
 *
 * ANDing here composes to any depth, because each enclosing guard was itself
 * written by a seed that ANDs with *its* enclosing. A null `enclosing` means
 * "top level" and reads as 1.
 *
 * The enclosing load is RELAXED, like the body guard read: the node that wrote
 * it is an ancestor of this one in the graph, so the edge already orders them.
 * See HIPGRAPH_COND_GUARD for the 73x measurement that makes acquire wrong. */
__global__ void hgc_set_condition(unsigned int* slot, const int* condition, const unsigned int* enclosing)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const int v = __atomic_load_n(condition, __ATOMIC_RELAXED);
        unsigned int result = (v != 0) ? 1u : 0u;
        if (enclosing && __atomic_load_n(enclosing, __ATOMIC_RELAXED) == 0u)
            result = 0u;
        __atomic_store_n(slot, result, __ATOMIC_RELEASE);
    }
}

/* Write the logical complement of *condition into a slot. The else-branch half
 * of an if/else pair: same ordering requirements as hgc_set_condition, just the
 * opposite sense.
 *
 * Deliberately does NOT take an `enclosing` guard. The word this writes is
 * consumed as the *condition* of the else region's hipGraphCondBegin, and that
 * region's seed (hgc_set_condition above) performs the conjunction. ANDing here
 * as well would be redundant, and ANDing here INSTEAD would be wrong: the
 * complement must stay a faithful !cond so the two branches remain exclusive. */
__global__ void hgc_write_complement(unsigned int* slot, const int* condition)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const int v = __atomic_load_n(condition, __ATOMIC_RELAXED);
        __atomic_store_n(slot, (v != 0) ? 0u : 1u, __ATOMIC_RELEASE);
    }
}

/* Reset the slot to its default at the start of a replay. Without this a graph
 * that converged on one replay would start the next replay already "done". */
__global__ void hgc_reset_condition(unsigned int* slot, unsigned int value)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __atomic_store_n(slot, value, __ATOMIC_RELEASE);
    }
}

/* ---------------------------------------------------------------------------
 * Handle state
 * ------------------------------------------------------------------------ */

struct hipGraphCondHandle_st {
    hipGraph_t graph = nullptr;
    unsigned int* slot = nullptr; /* device: 0 = stop, 1 = continue */
    unsigned int slotIndex = 0u; /* index into the slab, for release */

    /* Which device's slab `slot` points into. Recorded at creation because
     * HandleDestroy may well run with a different device current -- Warp tears
     * handles down on error paths that have already switched context -- and
     * returning the slot to the wrong device's free list would hand a device-0
     * address to a device-1 handle, which is exactly the fault this field exists
     * to prevent. */
    int device = 0;
    unsigned int defaultValue = 1u;
    unsigned int flags = 0u;

    /* Guard word of the region lexically enclosing this one, or null at top
     * level. Every seed that writes `slot` ANDs with it -- see hgc_set_condition
     * for why, and hipGraphCondSetEnclosingGuard for how it is supplied.
     *
     * Lives on the HANDLE rather than only on the open CondRegion because
     * spliceBody emits refresh seeds between unroll copies, long after the
     * region's Begin has been consumed, and those need the same conjunction. */
    const unsigned int* enclosingGuard = nullptr;

    /* How many times this handle's slot has been baked into a graph.
     *
     * THIS IS THE FIELD THAT MAKES SLOT REUSE SAFE. spliceBody writes
     * `slot` into kernel nodes as a raw device address, and those nodes
     * outlive this handle -- an instantiated hipGraphExec_t keeps replaying
     * them, and there is no HIP callback telling us when it dies. So a slot
     * that has ever been spliced can NEVER be handed to another handle: the
     * two graphs would share one condition word and silently corrupt each
     * other's convergence.
     *
     * A handle that was created and destroyed WITHOUT being spliced (a probe,
     * an aborted capture, an error unwind) references no graph, so its slot
     * is safe to recycle immediately. That is the distinction this counter
     * exists to draw, and drawing it in the type is why callers no longer
     * have to know the rule. */
    unsigned int splices = 0u;
};

/* --- handle validation ---------------------------------------------------
 *
 * Every entry point that dereferences a handle checks it against a registry of
 * live handles first. A magic word inside the object would be cheaper, but
 * reading it from an already-deleted handle is itself undefined behaviour --
 * precisely the use-after-free this is meant to catch. A side registry is
 * checked WITHOUT touching the candidate pointer, so a destroyed, zeroed or
 * garbage handle produces hipErrorInvalidValue instead of a segfault.
 *
 * This matters because handles cross a ctypes/uint64 boundary in the Warp
 * integration, where a stale id used to be turned into a pointer and
 * dereferenced blind.
 */
namespace {

/* One lock for the whole library. It guards the slot pool, the live-handle
 * registry, the region table and the pending-bound table together, because a
 * region holds a handle which holds a pool slot and the four must never be
 * observed out of step with each other. */
std::mutex g_lock;

std::unordered_set<hipGraphCondHandle> g_liveHandleSet;

/* Caller must hold g_lock. */
bool handleValidLocked(hipGraphCondHandle h) { return h != nullptr && g_liveHandleSet.count(h) != 0; }

/* Caller must hold g_lock. Derived rather than counted so it cannot drift from
 * the registry it describes. */
unsigned int liveHandleCountLocked() { return static_cast<unsigned int>(g_liveHandleSet.size()); }

/* Validate and read the slot in ONE critical section.
 *
 * Doing it in two (validate, unlock, dereference) leaves a window in which
 * another thread destroys the handle between the check and the read, which is
 * the same use-after-free with a smaller window. Returns nullptr if the handle
 * is not live. Callers must NOT already hold g_lock.
 */
unsigned int* handleSlotChecked(hipGraphCondHandle h)
{
    std::lock_guard<std::mutex> lk(g_lock);
    return handleValidLocked(h) ? h->slot : nullptr;
}

/* Same contract, for the callers that launch a seed kernel: they need the slot
 * and the enclosing guard together, and reading them in two critical sections
 * would let a SetEnclosingGuard land between the two and seed a node with a
 * slot from one region and a guard from another. Returns nullptr on an invalid
 * handle; *enclosing_out is only written on success. */
unsigned int* handleSeedArgsChecked(hipGraphCondHandle h, const unsigned int** enclosing_out)
{
    std::lock_guard<std::mutex> lk(g_lock);
    if (!handleValidLocked(h))
        return nullptr;
    *enclosing_out = h->enclosingGuard;
    return h->slot;
}

}  // namespace

/* --- condition slot pool -------------------------------------------------
 *
 * Device allocation is ILLEGAL while a stream capture is active -- hipMalloc
 * during capture returns hipErrorStreamCaptureIsolation (906). But the natural
 * place to create a condition handle is *inside* the capture, right where the
 * loop appears, which is also how CUDA's cudaGraphConditionalHandleCreate is
 * used. Verified on gfx90a/ROCm 7.1: allocating in HandleCreate during capture
 * fails and poisons the whole capture.
 *
 * So slots come from a slab reserved ahead of time. Callers who create handles
 * during capture must warm the pool first (hipGraphCondPoolReserve, or simply
 * by creating one handle outside capture); if the pool is exhausted mid-capture
 * we surface hipErrorStreamCaptureIsolation rather than corrupting the capture.
 */
namespace {
constexpr unsigned int kDefaultPoolSlots = 64;

struct SlotPool {
    unsigned int* base = nullptr;
    unsigned int capacity = 0;
    unsigned int used = 0; /* high-water mark of never-yet-handed-out slots */

    /* Slots whose handle was destroyed WITHOUT ever being spliced into a graph.
     *
     * Only never-spliced slots land here -- see hipGraphCondHandle_st::splices.
     * Recycling a spliced slot would hand a live graph's condition word to a new
     * handle, which is a silent wrong-answer bug and strictly worse than running
     * out of slots.
     *
     * Without any recycling, a long-running process that captures repeatedly
     * burns one slot per capture and eventually exhausts the slab. It used to
     * "work" because HandleDestroy freed the slab out from under the pool, which
     * is the bug this replaces. */
    std::vector<unsigned int> freeSlots;

    /* Slots that have been spliced into a graph and can never be reissued, even
     * after their handle is destroyed. The slab cannot be freed while this is
     * non-zero: an instantiated exec may still write through those addresses,
     * and nothing in HIP tells us when it stops. */
    unsigned int retiredSlots = 0;
};

/* ONE SLAB PER DEVICE, keyed by HIP device ordinal.
 *
 * This was a single process-global SlotPool, and that was a fault rather than an
 * untidiness. The slab is allocated on whichever device happened to be current at
 * the first reserve, and a plain hipMalloc'd address is not dereferenceable from
 * another device unless peer access has been enabled explicitly. So a handle
 * created on device 1 received a device-0 interior pointer, and the first store a
 * seed kernel made through it faulted -- "Memory access fault by GPU node-3 ...
 * Reason: Unknown", measured on a two-GPU MI300X node with test_complex_capture
 * on cuda:1 while cuda:0 merely returned a wrong answer.
 *
 * Keying by device also puts the LAZY reserve on the right device. Warp calls
 * hipGraphCondPoolReserve from wp_cuda_graph_begin_capture under a ContextGuard,
 * so the first capture on each device warms that device's slab at the last point
 * in the flow that is still outside a capture -- which is the only point where
 * allocating is legal at all (see the header comment above). Without per-device
 * keying, the second device found a non-null base, skipped the reserve entirely,
 * and never got a slab of its own. */
std::unordered_map<int, SlotPool> g_pools;

/* Caller must hold g_lock.
 *
 * Every pool operation is relative to the calling thread's current device, which
 * is what HIP itself uses to decide where an allocation lands.
 *
 * On failure the error is propagated -- every caller returns on it. *device_out is
 * still set to 0 so a caller that chooses to ignore the error cannot go on to index
 * g_pools with an uninitialised value; it is not a fallback that makes the
 * operation succeed. hipGetDevice failing at all means the runtime is unusable. */
hipError_t currentDeviceLocked(int* device_out)
{
    hipError_t err = hipGetDevice(device_out);
    if (err != hipSuccess) {
        HGC_LOG("hipGetDevice failed: %d (%s); assuming device 0", (int)err, hipGetErrorString(err));
        *device_out = 0;
    }
    return err;
}

/* Caller must hold g_lock. */
hipError_t poolReserveLocked(SlotPool& pool, int device, unsigned int slots)
{
    if (pool.base && pool.capacity >= slots)
        return hipSuccess;
    if (pool.base) {
        /* Never move a live slab: outstanding handles hold interior pointers
         * into it. Report the shortfall rather than silently under-serving --
         * the old code returned hipSuccess here, so a caller asking for 256
         * slots after a default 64-slot slab existed was told "fine" and then
         * ran out mid-capture. */
        HGC_LOG("device %d pool already live with %u slots; request for %u not honoured", device, pool.capacity, slots);
        return hipErrorOutOfMemory;
    }

    unsigned int* base = nullptr;
    const size_t bytes = sizeof(unsigned int) * slots;
    /* Uncached so a store from one kernel is promptly visible to the next
     * without depending on L2 writeback timing. Plain device memory is also
     * correct -- every reader/writer pair is separated by a graph edge -- so
     * fall back rather than fail. */
    hipError_t err = hipExtMallocWithFlags(reinterpret_cast<void**>(&base), bytes, hipDeviceMallocUncached);
    if (err != hipSuccess) {
        HGC_LOG("device %d hipExtMallocWithFlags(%zu) -> %d, falling back to hipMalloc", device, bytes, (int)err);
        err = hipMalloc(&base, bytes);
        /* Drain the error the tolerated attempt latched on this thread.
         *
         * hipGetLastError is per-thread and sticky, so a failure we chose to
         * ignore would otherwise surface later as somebody else's -- including
         * in hipGraphCondBegin's launch checks, making the first conditional
         * capture fail on a perfectly healthy device. Draining HERE is safe
         * precisely because the error is provably ours; draining in Begin would
         * have swallowed the caller's errors too. */
        (void)hipGetLastError();
    }
    if (err != hipSuccess) {
        HGC_LOG("device %d pool reserve of %u slots FAILED: %d (%s)", device, slots, (int)err, hipGetErrorString(err));
        return err;
    }

    err = hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(base), 1u, slots);
    if (err != hipSuccess) {
        HGC_BEST_EFFORT(hipFree(base));
        return err;
    }

    pool.base = base;
    pool.capacity = slots;
    pool.used = 0;
    pool.freeSlots.clear();
    HGC_LOG("device %d pool reserved: %u slots at %p", device, slots, (void*)base);
    return hipSuccess;
}
}  // namespace

/* Per-stream state for an open hipGraphCondBegin region. */
struct CondRegion {
    hipGraphCondHandle handle = nullptr;
    hipGraphCondType type = hipGraphCondTypeWhile;
    const int* condition = nullptr;
    unsigned int maxIters = 1;

    /* The parent capture is never interrupted; we only remember where its
     * frontier was when the region opened, so the unrolled chain can be
     * anchored there. */
    hipGraph_t parentGraph = nullptr;
    std::vector<hipGraphNode_t> parentDeps;

    /* Scratch stream the body is captured on, so the parent capture on the
     * caller's stream stays open throughout. */
    hipStream_t bodyStream = nullptr;
};

namespace {

/* g_lock is declared with the slot pool above -- it guards the pool, the region
 * table and the pending-bound table together, because a region holds a handle
 * that holds a pool slot and the three must not be observed out of step. */
std::unordered_map<hipStream_t, CondRegion> g_regions;
std::atomic<unsigned int> g_lastUnrollCount { 0 };

/* Unroll bounds set before their region exists. hipGraphCondSetMaxIters is
 * naturally called BEFORE hipGraphCondBegin -- the caller knows its iteration
 * budget up front -- so the bound is parked here and consumed when the region
 * opens. */
std::unordered_map<hipStream_t, unsigned int> g_pendingMaxIters;

/* Cached probe of whether this runtime has native conditional nodes.
 * Checked once; the answer cannot change within a process. */
int probeNativeSupport()
{
    static int cached = -1;
    if (cached >= 0)
        return cached;
    /* A native implementation would expose hipGraphAddNode with a conditional
     * node type. Rather than dlsym-probing a symbol that has never existed in
     * any ROCm release, we treat native as unavailable and let a future port
     * flip this deliberately. Verified absent on ROCm 7.1, 7.2, and clr
     * develop (HIP 7.13). */
    cached = 0;
    return cached;
}

/* Clone every node of `src` into `dst`, preserving src's internal edges.
 *
 * Nodes of src with no predecessor are attached to `entryDeps` (the parent's
 * current frontier); nodes with no successor are returned in `exitNodes` so the
 * caller can chain the next copy onto them. This is what lets the body be
 * flattened into the parent instead of embedded as a child-graph node, which
 * costs 2.8x at replay (see hipGraphCondEnd).
 *
 * Only node types a captured body can actually contain are handled. Anything
 * else -- notably a nested child graph -- is rejected rather than silently
 * dropped, because silently dropping a node would produce a graph that runs but
 * computes the wrong thing.
 */
hipError_t cloneGraphInto(
    hipGraph_t dst, hipGraph_t src, const std::vector<hipGraphNode_t>& entryDeps, std::vector<hipGraphNode_t>* exitNodes
)
{
    size_t numNodes = 0;
    hipError_t err = hipGraphGetNodes(src, nullptr, &numNodes);
    if (err != hipSuccess)
        return err;
    if (numNodes == 0) {
        if (exitNodes)
            *exitNodes = entryDeps;
        return hipSuccess;
    }

    std::vector<hipGraphNode_t> nodes(numNodes);
    err = hipGraphGetNodes(src, nodes.data(), &numNodes);
    if (err != hipSuccess)
        return err;

    size_t numEdges = 0;
    err = hipGraphGetEdges(src, nullptr, nullptr, &numEdges);
    if (err != hipSuccess)
        return err;
    std::vector<hipGraphNode_t> from(numEdges), to(numEdges);
    if (numEdges) {
        err = hipGraphGetEdges(src, from.data(), to.data(), &numEdges);
        if (err != hipSuccess)
            return err;
    }

    /* Predecessors and successors within src, by index. */
    std::unordered_map<hipGraphNode_t, size_t> index;
    for (size_t i = 0; i < numNodes; ++i)
        index[nodes[i]] = i;

    std::vector<std::vector<size_t>> preds(numNodes);
    std::vector<bool> hasSucc(numNodes, false);
    for (size_t e = 0; e < numEdges; ++e) {
        auto f = index.find(from[e]);
        auto t = index.find(to[e]);
        if (f == index.end() || t == index.end())
            continue;
        preds[t->second].push_back(f->second);
        hasSucc[f->second] = true;
    }

    /* Topological order, so a node's clone is created after its predecessors'.
     * Captured graphs are DAGs by construction; a cycle would mean the runtime
     * handed us something malformed, so bail rather than loop forever. */
    std::vector<size_t> indegree(numNodes, 0);
    for (size_t i = 0; i < numNodes; ++i)
        indegree[i] = preds[i].size();
    std::vector<size_t> order;
    order.reserve(numNodes);
    for (size_t i = 0; i < numNodes; ++i)
        if (indegree[i] == 0)
            order.push_back(i);
    for (size_t k = 0; k < order.size(); ++k) {
        const size_t n = order[k];
        for (size_t e = 0; e < numEdges; ++e) {
            auto f = index.find(from[e]);
            auto t = index.find(to[e]);
            if (f == index.end() || t == index.end() || f->second != n)
                continue;
            if (--indegree[t->second] == 0)
                order.push_back(t->second);
        }
    }
    if (order.size() != numNodes)
        return hipErrorInvalidValue; /* cycle */

    std::vector<hipGraphNode_t> clone(numNodes, nullptr);

    for (size_t oi = 0; oi < order.size(); ++oi) {
        const size_t i = order[oi];

        /* A node with no predecessor inside src hangs off the parent frontier. */
        std::vector<hipGraphNode_t> deps;
        if (preds[i].empty()) {
            deps = entryDeps;
        } else {
            deps.reserve(preds[i].size());
            for (size_t p : preds[i])
                deps.push_back(clone[p]);
        }
        const hipGraphNode_t* depPtr = deps.empty() ? nullptr : deps.data();
        const size_t depCount = deps.size();

        hipGraphNodeType type;
        err = hipGraphNodeGetType(nodes[i], &type);
        if (err != hipSuccess)
            return err;

        switch (type) {
        case hipGraphNodeTypeKernel: {
            hipKernelNodeParams kp;
            std::memset(&kp, 0, sizeof(kp));
            err = hipGraphKernelNodeGetParams(nodes[i], &kp);
            if (err != hipSuccess)
                return err;
            err = hipGraphAddKernelNode(&clone[i], dst, depPtr, depCount, &kp);
            break;
        }
        case hipGraphNodeTypeMemcpy: {
            hipMemcpy3DParms mp;
            std::memset(&mp, 0, sizeof(mp));
            err = hipGraphMemcpyNodeGetParams(nodes[i], &mp);
            if (err != hipSuccess)
                return err;
            err = hipGraphAddMemcpyNode(&clone[i], dst, depPtr, depCount, &mp);
            break;
        }
        case hipGraphNodeTypeMemset: {
            hipMemsetParams ms;
            std::memset(&ms, 0, sizeof(ms));
            err = hipGraphMemsetNodeGetParams(nodes[i], &ms);
            if (err != hipSuccess)
                return err;
            err = hipGraphAddMemsetNode(&clone[i], dst, depPtr, depCount, &ms);
            break;
        }
        case hipGraphNodeTypeHost: {
            hipHostNodeParams hp;
            std::memset(&hp, 0, sizeof(hp));
            err = hipGraphHostNodeGetParams(nodes[i], &hp);
            if (err != hipSuccess)
                return err;
            err = hipGraphAddHostNode(&clone[i], dst, depPtr, depCount, &hp);
            break;
        }
        case hipGraphNodeTypeEmpty:
            err = hipGraphAddEmptyNode(&clone[i], dst, depPtr, depCount);
            break;
        default:
            /* Event record/wait, child graph, mem alloc/free, ext semaphore.
             * Refuse rather than drop: a body containing one of these needs
             * deliberate handling, and a wrong graph is worse than an error. */
            return hipErrorNotSupported;
        }
        if (err != hipSuccess)
            return err;
    }

    if (exitNodes) {
        exitNodes->clear();
        for (size_t i = 0; i < numNodes; ++i)
            if (!hasSucc[i])
                exitNodes->push_back(clone[i]);
        /* A body that is one long chain has a single exit; a fan-out body has
         * several, and the next copy must wait for all of them. */
        if (exitNodes->empty())
            *exitNodes = entryDeps;
    }
    return hipSuccess;
}

/* Snapshot the capturing graph and the current capture frontier of `stream`.
 * Both are needed to re-attach the parent capture after the body detour. */
hipError_t captureInfo(hipStream_t stream, hipGraph_t* graph_out, std::vector<hipGraphNode_t>* deps_out)
{
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    hipGraph_t graph = nullptr;
    const hipGraphNode_t* deps = nullptr;
    size_t numDeps = 0;

    hipError_t err = hipStreamGetCaptureInfo_v2(stream, &status, nullptr, &graph, &deps, &numDeps);
    if (err != hipSuccess)
        return err;
    if (status != hipStreamCaptureStatusActive || graph == nullptr)
        return hipErrorIllegalState;

    if (graph_out)
        *graph_out = graph;
    if (deps_out)
        deps_out->assign(deps, deps + numDeps);
    return hipSuccess;
}

}  // namespace

/* ---------------------------------------------------------------------------
 * Handle lifetime
 * ------------------------------------------------------------------------ */

extern "C" hipError_t hipGraphCondHandleCreate(
    hipGraphCondHandle* handle_out, hipGraph_t graph, hipGraphCondAssign defaultValue, unsigned int flags
)
{
    if (!handle_out || !graph)
        return hipErrorInvalidValue;

    std::lock_guard<std::mutex> lk(g_lock);

    /* The slot must live on the device whose kernels will dereference it, and
     * that device is the current one -- Warp opens the region from a stream it
     * has already made current via ContextGuard. */
    int device = 0;
    if (hipError_t err = currentDeviceLocked(&device); err != hipSuccess)
        return err;
    SlotPool& pool = g_pools[device];

    /* Grow the pool only when it is safe to allocate, i.e. not mid-capture. */
    if (!pool.base) {
        hipError_t err = poolReserveLocked(pool, device, kDefaultPoolSlots);
        if (err != hipSuccess)
            return err;
    }
    /* Prefer a recycled slot; only then extend the high-water mark.
     *
     * Everything in pool.freeSlots is a NEVER-SPLICED slot (HandleDestroy only
     * pushes those), so reissuing one cannot alias a live graph. Slots that
     * were spliced are counted in pool.retiredSlots and never come back. */
    unsigned int index = 0;
    if (!pool.freeSlots.empty()) {
        index = pool.freeSlots.back();
        pool.freeSlots.pop_back();
    } else if (pool.used < pool.capacity) {
        index = pool.used++;
    } else {
        /* Exhausted, and we cannot grow during a capture. Tell the caller to
         * reserve more up front instead of silently breaking their capture.
         *
         * Name the retired count explicitly: a caller who sees the pool full
         * with zero free slots needs to know the slots are held by live graphs
         * (size the pool for TOTAL captures) rather than by leaked handles
         * (a bug to go fix). */
        HGC_LOG(
            "device %d condition-slot pool exhausted (%u/%u used, 0 free, %u "
            "retired to live graphs); call hipGraphCondPoolReserve() with a "
            "larger bound before capture",
            device, pool.used, pool.capacity, pool.retiredSlots
        );
        return hipErrorOutOfMemory;
    }

    auto* h = new (std::nothrow) hipGraphCondHandle_st();
    if (!h) {
        pool.freeSlots.push_back(index);
        return hipErrorOutOfMemory;
    }

    h->graph = graph;
    h->slotIndex = index;
    h->device = device;
    h->slot = pool.base + index;
    h->defaultValue = (defaultValue == hipGraphCondAssignZero) ? 0u : 1u;
    h->flags = flags;
    g_liveHandleSet.insert(h);

    /* Deliberately NO device-side initialization here.
     *
     * Handles are normally created inside an active capture (that is where the
     * loop is). Any host-side device op issued during capture -- hipMemsetD32
     * included -- is rejected with hipErrorStreamCaptureIsolation AND poisons
     * the entire capture, so every later call on that stream fails with
     * hipErrorIllegalState. Swallowing the error is not enough; the capture is
     * already dead. Verified on gfx90a/ROCm 7.1.
     *
     * The slab is pre-initialized in poolReserveLocked, and the value that
     * actually governs execution is written by the reset kernel that
     * hipGraphCondBegin emits into the graph on every replay. */

    *handle_out = h;
    return hipSuccess;
}

extern "C" hipError_t hipGraphCondPoolReserve(unsigned int slots)
{
    if (slots == 0)
        return hipErrorInvalidValue;
    std::lock_guard<std::mutex> lk(g_lock);
    /* Reserves for the CURRENT device only -- slabs are per-device, so a
     * multi-GPU caller must call this once per device, each time with that
     * device current. Warp gets this for free: it reserves from
     * wp_cuda_graph_begin_capture, under a ContextGuard, so the first capture on
     * each device warms that device's slab. */
    int device = 0;
    if (hipError_t err = currentDeviceLocked(&device); err != hipSuccess)
        return err;
    return poolReserveLocked(g_pools[device], device, slots);
}

extern "C" hipError_t hipGraphCondPoolStatus(unsigned int* capacity_out, unsigned int* used_out)
{
    std::lock_guard<std::mutex> lk(g_lock);
    /* Reports the CURRENT device's slab, matching hipGraphCondPoolReserve. */
    int device = 0;
    if (hipError_t err = currentDeviceLocked(&device); err != hipSuccess)
        return err;
    const SlotPool& pool = g_pools[device];
    if (capacity_out)
        *capacity_out = pool.base ? pool.capacity : 0u;
    /* UNAVAILABLE slots, not the high-water mark and not the live-handle count.
     *
     * pool.used only ever grows; slots handed back by HandleDestroy sit in
     * pool.freeSlots and are genuinely available. Reporting the high-water mark
     * made a pool with free slots look exhausted, and the Warp shim then
     * refused a capture that would have succeeded.
     *
     * But live handles alone UNDER-reports: a slot retired to a live graph is
     * neither held by a handle nor free. `used - free` counts both, so this
     * stays exactly equivalent to "HandleCreate would return
     * hipErrorOutOfMemory", which is what the shim's pre-check needs, and it
     * agrees with the exhaustion log in HandleCreate. */
    if (used_out)
        *used_out = pool.base ? pool.used - static_cast<unsigned int>(pool.freeSlots.size()) : 0u;
    return hipSuccess;
}

extern "C" hipError_t hipGraphCondHandleDestroy(hipGraphCondHandle handle)
{
    if (!handle)
        return hipSuccess;

    /* DO NOT hipFree(handle->slot).
     *
     * This is the bug that produced both the `cond` SIGSEGV and the wedged HIP
     * context that followed it. The slot is NOT an allocation -- it is an
     * interior pointer into the slab reserved by poolReserveLocked:
     *
     *     h->slot = pool.base + pool.used++;
     *
     * so the first handle's slot aliases pool.base exactly. Freeing it
     * returns the ENTIRE 64/256-slot slab to the allocator while pool.base
     * still points at it and pool.capacity still says it is live. Every
     * subsequent hipGraphCondHandleCreate hands out a pointer into freed device
     * memory, poolReserveLocked refuses to re-reserve ("never move a live
     * slab"), and the condition kernels then write through a dangling device
     * pointer -- a fault inside the HIP runtime, i.e. rc=139, and a context that
     * cannot satisfy even a 4-byte allocation afterwards.
     *
     * Freeing any handle OTHER than the first is worse still: hipFree on an
     * interior pointer that is not an allocation base is undefined behaviour
     * and is what corrupts the allocator's bookkeeping directly.
     *
     * Slots are owned by the pool for the life of the process. Handle teardown
     * releases only the host-side record, and returns the slot to the free list
     * ONLY if it was never spliced into a graph -- see `splices` on the handle.
     * hipGraphCondPoolRelease() exists for a caller that genuinely wants the
     * slab back, and it refuses while any slot is still reachable from a graph.
     */
    {
        std::lock_guard<std::mutex> lk(g_lock);

        /* Checked against the registry, never by reading through `handle` --
         * a double destroy would make that read a use-after-free itself. */
        if (!handleValidLocked(handle)) {
            HGC_LOG("HandleDestroy on unknown or already-destroyed handle %p", (void*)handle);
            return hipErrorInvalidValue;
        }

        /* Refuse to retire a handle that a region still points at: spliceBody
         * dereferences region.handle->slot after End. Destroying it first left
         * the region holding a freed host object -- a second, independent
         * SIGSEGV route on the error paths in the Warp shim, which destroy the
         * handle while the region is still open. */
        for (const auto& kv : g_regions) {
            if (kv.second.handle == handle) {
                HGC_LOG(
                    "HandleDestroy refused: handle %p is in use by an open "
                    "region on stream %p",
                    (void*)handle, (void*)kv.first
                );
                return hipErrorIllegalState;
            }
        }

        /* Recycle ONLY a slot no graph can reach.
         *
         * If this handle was ever spliced, its slot address is baked into
         * kernel nodes in a graph that may still be instantiated and replaying.
         * HIP gives us no notification when that exec dies, so the only safe
         * answer is to retire the slot permanently. Handing it out again would
         * make two graphs share one condition word: no crash, no error, just a
         * loop that stops early or never stops -- the worst possible failure
         * mode for a solver.
         *
         * A never-spliced handle (a probe, an aborted capture, an error unwind)
         * is referenced by nothing, so its slot goes straight back. That is what
         * keeps a repeated capture/abort loop from exhausting the slab. */
        /* Back to the slab it actually came from, not the current device's --
         * see hipGraphCondHandle_st::device. */
        auto pool_it = g_pools.find(handle->device);
        if (pool_it != g_pools.end() && pool_it->second.base && handle->slotIndex < pool_it->second.capacity) {
            SlotPool& pool = pool_it->second;
            if (handle->splices == 0u) {
                pool.freeSlots.push_back(handle->slotIndex);
            } else {
                ++pool.retiredSlots;
                HGC_LOG(
                    "device %d slot %u retired: spliced into %u graph(s), "
                    "cannot be reissued",
                    handle->device, handle->slotIndex, handle->splices
                );
            }
        }
        g_liveHandleSet.erase(handle);
    }

    delete handle;
    return hipSuccess;
}

/* Release the condition-slot slab.
 *
 * Exists so a caller that really wants the device memory back has a correct way
 * to get it -- previously the only "way" was hipGraphCondHandleDestroy freeing
 * an interior pointer, which corrupted the allocator.
 *
 * REFUSES WHILE ANY SLOT IS STILL REACHABLE FROM A GRAPH. Checking only for
 * live handles is NOT sufficient and was itself a defect: the natural teardown
 * order is to destroy the handles and keep the instantiated hipGraphExec_t for
 * replay, which drives the handle count to zero while spliced kernel nodes
 * still hold slot addresses. Freeing the slab there reproduces exactly the
 * dangling-device-pointer fault this whole change exists to remove.
 *
 * Since HIP offers no way to learn that a hipGraphExec_t has been destroyed,
 * `retiredSlots != 0` on any device is a one-way latch: once a slot has been spliced,
 * the slab is pinned for the life of the process. That is deliberately
 * conservative -- the alternative is a silent memory fault -- and it is why
 * the error message tells the caller what would have to be true instead.
 *
 * Must not be called during a capture: it frees device memory.
 */
extern "C" hipError_t hipGraphCondPoolRelease(void)
{
    std::lock_guard<std::mutex> lk(g_lock);

    /* Releases EVERY device's slab, unlike Reserve/Status which act on the
     * current one. Those two are per-device because the caller is asking about
     * the device it is about to capture on; this is teardown, and a release that
     * silently left another device's slab allocated would be a leak nobody has a
     * second call to fix -- the caller would have to know which devices were
     * ever touched, which is precisely what this file knows and they do not.
     *
     * All-or-nothing: the refusal conditions below are checked across all
     * devices before anything is freed, so a partial release cannot leave the
     * bookkeeping describing memory that is already gone. */
    const unsigned int live = liveHandleCountLocked();
    if (live != 0 || !g_regions.empty()) {
        HGC_LOG("PoolRelease refused: %u live handle(s), %zu open region(s)", live, g_regions.size());
        return hipErrorIllegalState;
    }
    for (const auto& kv : g_pools) {
        if (kv.second.retiredSlots != 0) {
            HGC_LOG(
                "PoolRelease refused: %u slot(s) on device %d are spliced into "
                "graphs that may still be instantiated. HIP cannot report when "
                "a hipGraphExec_t is destroyed, so the slab stays reserved for "
                "the life of the process once any capture has used it.",
                kv.second.retiredSlots, kv.first
            );
            return hipErrorIllegalState;
        }
    }

    /* hipFree is device-agnostic for an address the runtime owns, so no device
     * switch is needed here -- and switching would be worse, because it would
     * leave the caller's current device changed on an error return. */
    for (auto& kv : g_pools) {
        if (kv.second.base)
            HGC_TRY(hipFree(kv.second.base));
    }
    g_pools.clear();
    return hipSuccess;
}

extern "C" hipError_t hipGraphCondHandleGetDevicePtr(hipGraphCondHandle handle, unsigned int** ptr_out)
{
    if (!ptr_out)
        return hipErrorInvalidValue;
    unsigned int* slot = handleSlotChecked(handle);
    if (!slot) {
        HGC_LOG("GetDevicePtr on invalid handle %p", (void*)handle);
        return hipErrorInvalidValue;
    }
    *ptr_out = slot;
    return hipSuccess;
}

extern "C" hipError_t hipGraphCondSetGuard(hipGraphCondHandle handle, unsigned int** guard_out)
{
    if (!guard_out)
        return hipErrorInvalidValue;
    unsigned int* slot = handleSlotChecked(handle);
    if (!slot) {
        HGC_LOG("SetGuard on invalid handle %p", (void*)handle);
        return hipErrorInvalidValue;
    }
    *guard_out = slot; /* guard and condition slot are the same word */
    return hipSuccess;
}

/* Record the guard of the lexically enclosing region on this handle.
 *
 * Must be called BEFORE hipGraphCondBegin, because Begin emits the seed that
 * consumes it. `enclosing` may be null, which means "this region is at the top
 * level" and disables the conjunction. See hgc_set_condition for the failure
 * this prevents.
 *
 * Deliberately a property of the HANDLE and not a parameter of Begin: the same
 * value is needed again by spliceBody's inter-copy refresh seeds, which run
 * after the region has been consumed, and Begin's signature is shared with the
 * while-loop path. */
extern "C" hipError_t hipGraphCondSetEnclosingGuard(hipGraphCondHandle handle, const unsigned int* enclosing)
{
    /* Validate and assign in ONE critical section, for the reason spelled out
     * above handleSlotChecked. */
    std::lock_guard<std::mutex> lk(g_lock);
    if (!handleValidLocked(handle)) {
        HGC_LOG("SetEnclosingGuard on invalid handle %p", (void*)handle);
        return hipErrorInvalidValue;
    }
    handle->enclosingGuard = enclosing;
    return hipSuccess;
}

/* ---------------------------------------------------------------------------
 * Condition update
 * ------------------------------------------------------------------------ */

extern "C" hipError_t hipGraphCondSetCondition(hipStream_t stream, hipGraphCondHandle handle, const int* condition)
{
    if (!condition)
        return hipErrorInvalidValue;
    /* This is the launch that used to run with a dangling slot pointer once
     * HandleDestroy had freed the slab. A destroyed handle is now caught here
     * instead of faulting inside the kernel. */
    const unsigned int* enclosing = nullptr;
    unsigned int* slot = handleSeedArgsChecked(handle, &enclosing);
    if (!slot) {
        HGC_LOG("SetCondition on invalid handle %p", (void*)handle);
        return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(hgc_set_condition, dim3(1), dim3(1), 0, stream, slot, condition, enclosing);
    return hipGetLastError();
}

extern "C" hipError_t hipGraphCondWriteComplement(hipStream_t stream, hipGraphCondHandle handle, const int* condition)
{
    if (!condition)
        return hipErrorInvalidValue;
    unsigned int* slot = handleSlotChecked(handle);
    if (!slot) {
        HGC_LOG("WriteComplement on invalid handle %p", (void*)handle);
        return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(hgc_write_complement, dim3(1), dim3(1), 0, stream, slot, condition);
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
        return err;

    /* Retire the slot, for the same reason hipGraphCondBegin does: the launch
     * above became a kernel node in the caller's graph holding this address as
     * a raw device pointer, and that node outlives the handle. Unlike
     * SetCondition -- which is only ever called on a handle already inside a
     * region, and so already retired -- this is the ONLY point at which a
     * complement handle's address escapes into a graph. Without the bump the
     * slot goes back on the free list at HandleDestroy and a later region
     * aliases a word this graph still writes on every replay. */
    ++handle->splices;
    return hipSuccess;
}

/* ---------------------------------------------------------------------------
 * Conditional region
 * ------------------------------------------------------------------------ */

extern "C" hipError_t hipGraphCondBegin(
    hipStream_t stream, hipGraphCondHandle handle, hipGraphCondType type, const int* condition, unsigned int max_iters
)
{
    if (!condition || max_iters == 0)
        return hipErrorInvalidValue;
    if (type == hipGraphCondTypeIf)
        max_iters = 1;

    std::lock_guard<std::mutex> guard(g_lock);

    /* Validated under the lock, so the handle cannot be destroyed between the
     * check and the dereferences below. */
    if (!handleValidLocked(handle)) {
        HGC_LOG("Begin with invalid handle %p", (void*)handle);
        return hipErrorInvalidValue;
    }
    if (g_regions.count(stream)) {
        /* A region left open by an earlier failed capture would otherwise make
         * every later Begin on this stream fail forever. Say so loudly; the
         * caller must close it (End / EndWithGraph) or call
         * hipGraphCondAbortRegion. */
        HGC_LOG(
            "Begin: a region is ALREADY open on stream %p -- the previous "
            "capture did not close it",
            (void*)stream
        );
        return hipErrorIllegalState; /* no nesting yet */
    }
    for (const auto& kv : g_regions) {
        if (kv.second.handle == handle) {
            /* Same handle, different stream. Both regions would splice the SAME
             * condition slot into two independent graphs, which then reset and
             * read one word -- the exact aliasing `splices` exists to prevent,
             * except it happens before any destroy, so retirement cannot help.
             * One handle, one region: create a second handle instead. */
            HGC_LOG(
                "Begin: handle %p is already driving a region on stream %p; "
                "a handle cannot back two regions at once",
                (void*)handle, (void*)kv.first
            );
            return hipErrorIllegalState;
        }
    }

    /* Adopt a bound parked by an earlier hipGraphCondSetMaxIters on this stream.
     * A `while` region takes it; an `if` is a single embedding by definition. */
    {
        auto p = g_pendingMaxIters.find(stream);
        if (p != g_pendingMaxIters.end()) {
            if (type != hipGraphCondTypeIf)
                max_iters = p->second;
            g_pendingMaxIters.erase(p);
        }
    }

    CondRegion region;
    region.handle = handle;
    region.type = type;
    region.condition = condition;
    region.maxIters = max_iters;

    /* Drain the thread's sticky error before launching, and REPORT what we
     * drained rather than swallowing it.
     *
     * hipGetLastError returns and clears the last error recorded on this
     * thread, not the result of the preceding call. Without a drain, the checks
     * after the launches report whatever was already latched: measured on
     * MI300X/ROCm 7.2, a stale 401 present at entry made hipGraphCondBegin fail
     * on a healthy device with both launches succeeding. So the drain is
     * load-bearing, not defensive.
     *
     * But draining silently would hide a real failure the CALLER latched -- a
     * user kernel that failed to launch into this same capture -- and their
     * error handling would then see hipSuccess. Logging it keeps the launch
     * checks below attributable while leaving a trace of anything inherited. */
    {
        const hipError_t stale = hipGetLastError();
        if (stale != hipSuccess)
            HGC_LOG(
                "Begin: cleared an error already latched on this thread "
                "before our launches: %d (%s). It did NOT come from this "
                "call; if you were expecting to retrieve it, do so before "
                "entering a conditional region.",
                (int)stale, hipGetErrorString(stale)
            );
    }

    /* Reset the slot inside the parent graph so each replay starts fresh, then
     * seed it from the caller's condition. Both are ordinary kernel nodes and
     * land in the parent capture. Without the reset, a graph that converged on
     * replay N would begin replay N+1 already "done". */
    hipLaunchKernelGGL(hgc_reset_condition, dim3(1), dim3(1), 0, stream, handle->slot, handle->defaultValue);
    /* Check the launch. A launch failure here used to be discarded, so a poisoned
     * capture (the 906 case) sailed on and only surfaced as a crash later.
     * Safe to attribute to this launch only because of the drain above. */
    HGC_TRY(hipGetLastError());
    /* handle->enclosingGuard is read under g_lock, which this function already
     * holds. It is null unless the caller parked one with
     * hipGraphCondSetEnclosingGuard, so a top-level region is unchanged. */
    hipLaunchKernelGGL(hgc_set_condition, dim3(1), dim3(1), 0, stream, handle->slot, condition, handle->enclosingGuard);
    HGC_TRY(hipGetLastError());

    /* The slot is NOW baked into the caller's graph, so retire it here.
     *
     * The two launches above became kernel nodes in the parent capture holding
     * handle->slot as a raw device address. That is true regardless of whether
     * a body is ever spliced: an empty body returns early from spliceBody, and
     * an aborted region leaves these nodes behind if the caller instantiates
     * the partial graph anyway. Counting only spliced body copies missed both,
     * and the recycled slot would then alias a graph that still writes it.
     *
     * Begin is the earliest point at which the address escapes into a graph,
     * so it is the correct place to mark it. spliceBody increments again per
     * splice; the count is only ever tested against zero. */
    ++handle->splices;

    /* Read the parent's graph and current frontier without disturbing it. */
    hipError_t err = captureInfo(stream, &region.parentGraph, &region.parentDeps);
    if (err != hipSuccess) {
        HGC_LOG("Begin: captureInfo(stream=%p) -> %d (%s)", (void*)stream, (int)err, hipGetErrorString(err));
        return err;
    }

    /* Capture the body on a private stream so the caller's capture is never
     * interrupted. Ending the caller's capture to make room for the body would
     * finalize the parent graph -- verified: hipStreamEndCapture returns the
     * parent and closes it, and re-attaching afterwards fights the runtime's
     * capture bookkeeping. ThreadLocal keeps this capture from being pulled
     * into the parent's capture set. */
    err = hipStreamCreateWithFlags(&region.bodyStream, hipStreamNonBlocking);
    if (err != hipSuccess) {
        HGC_LOG("Begin: hipStreamCreateWithFlags -> %d (%s)", (int)err, hipGetErrorString(err));
        return err;
    }

    err = hipStreamBeginCapture(region.bodyStream, hipStreamCaptureModeThreadLocal);
    if (err != hipSuccess) {
        HGC_LOG("Begin: hipStreamBeginCapture(body) -> %d (%s)", (int)err, hipGetErrorString(err));
        HGC_BEST_EFFORT(hipStreamDestroy(region.bodyStream));
        region.bodyStream = nullptr;
        return err;
    }

    g_regions[stream] = std::move(region);
    return hipSuccess;
}

extern "C" hipError_t hipGraphCondGetBodyStream(hipStream_t stream, hipStream_t* body_out)
{
    if (!body_out)
        return hipErrorInvalidValue;
    std::lock_guard<std::mutex> lk(g_lock);
    auto it = g_regions.find(stream);
    if (it == g_regions.end())
        return hipErrorIllegalState;
    *body_out = it->second.bodyStream;
    return hipSuccess;
}

namespace {

/* Unroll `bodyGraph` into the parent and re-anchor the capture frontier.
 *
 * Shared by hipGraphCondEnd (body captured on our private stream) and
 * hipGraphCondEndWithGraph (body captured by the caller). `ownsBody` says
 * whether we destroy the template when finished; a caller-supplied graph is
 * cloned, not consumed, so the caller may still destroy it afterwards.
 *
 * Caller must hold g_lock and must already have erased the region.
 */
hipError_t spliceBody(hipStream_t stream, CondRegion& region, hipGraph_t bodyGraph, bool ownsBody)
{
    hipError_t err = hipSuccess;
    if (!bodyGraph)
        return hipErrorInvalidValue;

    /* The region carries the handle across several caller-visible calls. If the
     * caller destroyed it in between (the Warp shim's error paths did exactly
     * that), region.handle->slot below is a read of freed host memory feeding a
     * device pointer into a kernel node. Refuse instead. */
    /* ...Locked, not the locking wrapper: every caller of spliceBody already
     * holds g_lock, and g_lock is not recursive. */
    if (!handleValidLocked(region.handle)) {
        HGC_LOG("splice: region on stream %p holds an invalid handle %p", (void*)stream, (void*)region.handle);
        if (ownsBody)
            HGC_BEST_EFFORT(hipGraphDestroy(bodyGraph));
        return hipErrorInvalidValue;
    }
    if (!region.parentGraph) {
        HGC_LOG("splice: region on stream %p has no parent graph", (void*)stream);
        if (ownsBody)
            HGC_BEST_EFFORT(hipGraphDestroy(bodyGraph));
        return hipErrorIllegalState;
    }

    size_t bodyNodes = 0;
    err = hipGraphGetNodes(bodyGraph, nullptr, &bodyNodes);
    if (err != hipSuccess) {
        /* Was discarded. A bad graph handle -- e.g. one Warp already destroyed
         * -- reported the failure here and then got cloned anyway. */
        HGC_LOG("splice: hipGraphGetNodes(body=%p) -> %d (%s)", (void*)bodyGraph, (int)err, hipGetErrorString(err));
        if (ownsBody)
            HGC_BEST_EFFORT(hipGraphDestroy(bodyGraph));
        return err;
    }
    if (bodyNodes == 0) {
        /* Empty body: nothing to unroll, and the parent frontier is unchanged. */
        if (ownsBody)
            HGC_BEST_EFFORT(hipGraphDestroy(bodyGraph));
        g_lastUnrollCount.store(0);
        return hipSuccess;
    }

    /* Emit the body max_iters times into the parent, chaining each copy to the
     * previous one and interleaving a condition refresh. The refresh is what
     * lets the body drive itself to a stop: iteration k writes the user's
     * condition array, and the following hgc_set_condition propagates it into the
     * slot that iteration k+1's kernels read as their guard.
     *
     * The body is FLATTENED into the parent rather than embedded as a child
     * graph node. hipGraphAddChildGraphNode would be the obvious tool, and it
     * works, but HIP does not flatten child graphs at instantiate time -- each
     * one is dispatched as a nested graph with its own barrier. Measured on
     * MI210, 10 tiny kernels:
     *
     *     10 kernel nodes, flat ................. 2.07 us/kernel
     *     10 child-graph nodes (1 kernel each) ... 5.84 us/kernel   2.81x
     *
     * That 2.8x is pure replay overhead paid on every iteration, converged or
     * not, so it must not be in the design. Flattening costs a little build-time
     * work here and nothing at replay.
     *
     * Every copy after convergence is a dispatch of already-predicated kernels
     * -- cheap, but not free. That residual is the honest cost of not having
     * real conditional nodes. */
    std::vector<hipGraphNode_t> deps = region.parentDeps;
    unsigned int embedded = 0;

    for (unsigned int i = 0; i < region.maxIters; ++i) {
        std::vector<hipGraphNode_t> tail;
        err = cloneGraphInto(region.parentGraph, bodyGraph, deps, &tail);
        if (err != hipSuccess) {
            HGC_LOG(
                "splice: cloneGraphInto copy %u/%u -> %d (%s)", i, region.maxIters, (int)err, hipGetErrorString(err)
            );
            break;
        }
        deps = std::move(tail);
        ++embedded;

        /* No refresh after the final body: nothing would consume it. */
        if (i + 1 == region.maxIters)
            break;

        hipKernelNodeParams kp;
        std::memset(&kp, 0, sizeof(kp));
        /* Same conjunction as Begin's seed. Without it, a while-region nested
         * inside a suppressed if-branch re-arms its guard here on every copy,
         * straight from the user's untouched condition array, and runs the full
         * unroll inside a branch that was not taken. hipGraphAddKernelNode
         * copies the pointed-to values immediately, so these addresses need only
         * outlive this call. */
        void* args[3] = { &region.handle->slot, &region.condition, &region.handle->enclosingGuard };
        kp.func = reinterpret_cast<void*>(hgc_set_condition);
        kp.gridDim = dim3(1);
        kp.blockDim = dim3(1);
        kp.sharedMemBytes = 0;
        kp.kernelParams = args;
        kp.extra = nullptr;

        hipGraphNode_t condNode = nullptr;
        err = hipGraphAddKernelNode(
            &condNode, region.parentGraph, deps.empty() ? nullptr : deps.data(), deps.size(), &kp
        );
        if (err != hipSuccess) {
            HGC_LOG("splice: condition-refresh node after copy %u -> %d (%s)", i, (int)err, hipGetErrorString(err));
            break;
        }
        deps.assign(1, condNode);
    }

    /* The parent owns copies of the body now. Destroy the template only if it
     * was ours; a caller-supplied graph stays alive for the caller to manage. */
    if (ownsBody)
        HGC_BEST_EFFORT(hipGraphDestroy(bodyGraph));
    g_lastUnrollCount.store(embedded);

    /* Mark the slot as reachable from a graph, so HandleDestroy retires it
     * instead of recycling it.
     *
     * This must happen even when `err` is set: a partial unroll still left
     * `embedded` copies wired into the parent graph, each holding this slot as
     * a raw device address. A failed splice is exactly the case where a caller
     * is most likely to destroy the handle and try again, so it is the case
     * where recycling would do the most damage. Condition on nodes actually
     * emitted, not on success.
     *
     * Caller holds g_lock (documented above), so this is safe unlocked. */
    if (embedded > 0)
        ++region.handle->splices;

    if (err != hipSuccess)
        return err;

    /* Move the parent's capture frontier to the tail of the unrolled chain, so
     * whatever the caller launches next is ordered after the loop rather than
     * racing it. This is the same mechanism CUDA requires around conditional
     * nodes (cuStreamUpdateCaptureDependencies), and it is why the parent
     * capture never had to be interrupted.
     *
     * Only meaningful while the stream is capturing into THE PARENT. "Is the
     * stream capturing at all" is not the same question, and the difference is
     * not hypothetical: on the EndWithGraph path the caller captured the body
     * itself, and it need not have finished. Warp is exactly that caller --
     * capture_while pauses the PARENT capture, redirects the stream into the
     * body graph, runs the body, and only calls set_condition (which lands
     * here) while the stream is still capturing into the BODY. An
     * "is it capturing" test says Active, and the update below is then handed
     * parent-graph nodes as dependencies of a body-graph capture.
     *
     * Measured, job 67923127 on gfx942: test_while_capture failed with
     * hipErrorInvalidValue (1) at precisely this call. Every other failure in
     * spliceBody logs, and this one did not, which is what identified it -- the
     * splice itself had already succeeded (the handle retired reporting
     * splices>0, so cloneGraphInto ran to completion).
     *
     * So compare graph identity, not capture state. When the stream is
     * capturing into something else, or not capturing, there is no parent
     * frontier to move: the caller's own resume re-anchors on the parent
     * graph's leaves, which after the splice above are precisely `deps`. */
    hipGraph_t capturing = nullptr;
    if (captureInfo(stream, &capturing, nullptr) != hipSuccess || capturing != region.parentGraph) {
        HGC_LOG(
            "splice: not re-anchoring stream %p -- captures into %p, parent is %p", (void*)stream, (void*)capturing,
            (void*)region.parentGraph
        );
        return hipSuccess;
    }

    err = hipStreamUpdateCaptureDependencies(
        stream, deps.empty() ? nullptr : deps.data(), deps.size(), hipStreamSetCaptureDependencies
    );
    /* Never let this return unlogged again: it was the one silent non-success
     * path in the whole splice, and that silence cost three jobs. */
    if (err != hipSuccess)
        HGC_LOG(
            "splice: hipStreamUpdateCaptureDependencies(stream=%p, n=%zu) -> %d (%s)", (void*)stream, deps.size(),
            (int)err, hipGetErrorString(err)
        );
    return err;
}

/* Detach the region for `stream`, or report that none is open.
 * Caller must hold g_lock. */
hipError_t takeRegion(hipStream_t stream, CondRegion* out)
{
    auto it = g_regions.find(stream);
    if (it == g_regions.end())
        return hipErrorIllegalState;
    *out = std::move(it->second);
    g_regions.erase(it);
    return hipSuccess;
}

}  // namespace

extern "C" hipError_t hipGraphCondEnd(hipStream_t stream)
{
    std::lock_guard<std::mutex> guard(g_lock);
    CondRegion region;
    hipError_t err = takeRegion(stream, &region);
    if (err != hipSuccess)
        return err;

    /* Close the body capture on the private stream. The caller's capture has
     * been open and untouched this whole time. */
    hipGraph_t bodyGraph = nullptr;
    err = hipStreamEndCapture(region.bodyStream, &bodyGraph);
    HGC_BEST_EFFORT(hipStreamDestroy(region.bodyStream));
    region.bodyStream = nullptr;
    if (err != hipSuccess) {
        HGC_LOG("End: hipStreamEndCapture(body) -> %d (%s)", (int)err, hipGetErrorString(err));
        /* The runtime may still have handed back a graph on the error path.
         * Not destroying it leaked one graph per failed capture across the
         * repeated load/capture cycle. */
        if (bodyGraph)
            HGC_BEST_EFFORT(hipGraphDestroy(bodyGraph));
        return err;
    }

    return spliceBody(stream, region, bodyGraph, /*ownsBody=*/true);
}

/* Discard the region open on `stream` without splicing anything.
 *
 * A caller whose capture failed between Begin and End had no way to drop the
 * region: it stayed in g_regions forever, its body stream stayed in capture,
 * and the NEXT Begin on the same stream returned hipErrorIllegalState. In a
 * repeated load/capture/free loop that turns one transient failure into a
 * permanent one -- part of the "works once, wedges after" signature. */
extern "C" hipError_t hipGraphCondAbortRegion(hipStream_t stream)
{
    std::lock_guard<std::mutex> guard(g_lock);

    CondRegion region;
    if (takeRegion(stream, &region) != hipSuccess) {
        /* No region open. Do NOT touch g_pendingMaxIters here.
         *
         * A bound is parked BEFORE the region opens -- there is no point in
         * between where a caller could reach in -- so on the normal path the
         * sequence is SetMaxIters, Abort (defensive, clears stale state), then
         * Begin. Erasing the bound in this branch would throw away the one the
         * caller just set microseconds earlier, and Begin would silently fall
         * back to the default unroll. That reintroduces the already-diagnosed
         * "cond cost is flat across budgets" defect.
         *
         * The stale-bound case this used to guard against is instead handled
         * where it can be distinguished: Begin erases the entry it consumes,
         * and the branch below clears the bound of a region that really was
         * open and is now being discarded. */
        return hipSuccess;
    }

    /* A region really was open and is being discarded, so any bound that
     * belonged to it is stale too. Safe to clear here because the parked-bound
     * ordering (set, then Begin) means an entry surviving alongside an OPEN
     * region cannot be one a caller is still waiting to have consumed. */
    g_pendingMaxIters.erase(stream);

    if (region.bodyStream) {
        hipGraph_t junk = nullptr;
        if (hipStreamEndCapture(region.bodyStream, &junk) == hipSuccess && junk)
            HGC_BEST_EFFORT(hipGraphDestroy(junk));
        HGC_BEST_EFFORT(hipStreamDestroy(region.bodyStream));
    }
    HGC_LOG("region on stream %p aborted", (void*)stream);
    return hipSuccess;
}

extern "C" hipError_t hipGraphCondEndWithGraph(hipStream_t stream, hipGraph_t body_graph)
{
    if (!body_graph)
        return hipErrorInvalidValue;

    std::lock_guard<std::mutex> guard(g_lock);
    CondRegion region;
    hipError_t err = takeRegion(stream, &region);
    if (err != hipSuccess)
        return err;

    /* The private body stream was never captured into on this path. End its
     * capture so the runtime's bookkeeping stays balanced, and discard whatever
     * empty graph comes back, before destroying the stream. */
    if (region.bodyStream) {
        hipGraph_t unused = nullptr;
        if (hipStreamEndCapture(region.bodyStream, &unused) == hipSuccess && unused)
            HGC_BEST_EFFORT(hipGraphDestroy(unused));
        HGC_BEST_EFFORT(hipStreamDestroy(region.bodyStream));
        region.bodyStream = nullptr;
    }

    return spliceBody(stream, region, body_graph, /*ownsBody=*/false);
}

extern "C" hipError_t hipGraphCondSetMaxIters(hipStream_t stream, unsigned int max_iters)
{
    if (max_iters == 0)
        return hipErrorInvalidValue;
    std::lock_guard<std::mutex> guard(g_lock);

    auto it = g_regions.find(stream);
    if (it != g_regions.end()) {
        /* An `if` region is by definition a single embedding. */
        if (it->second.type != hipGraphCondTypeIf)
            it->second.maxIters = max_iters;
        return hipSuccess;
    }

    /* No region open yet -- record the bound for the NEXT hipGraphCondBegin on
     * this stream.
     *
     * This is the case that matters in practice. A caller knows its iteration
     * budget before the loop starts, and Warp's capture_while must set the bound
     * before wp_cuda_graph_insert_while opens the region -- there is no point in
     * between where it could reach in. Requiring an already-open region made
     * this call return hipErrorIllegalState every time, silently leaving the
     * unroll at the env default. Measured symptom: `cond` sat flat at 8.89 ms
     * across budgets 2/4/10/20 while `full` scaled 2.58 -> 6.31. */
    g_pendingMaxIters[stream] = max_iters;
    return hipSuccess;
}

/* ---------------------------------------------------------------------------
 * Introspection
 * ------------------------------------------------------------------------ */

extern "C" hipError_t hipGraphCondGetLowering(hipGraphCondLowering* out)
{
    if (!out)
        return hipErrorInvalidValue;
    *out = probeNativeSupport() ? hipGraphCondLoweringNative : hipGraphCondLoweringPredicatedUnroll;
    return hipSuccess;
}

extern "C" int hipGraphCondIsNativeSupported(void) { return probeNativeSupport(); }

extern "C" const char* hipGraphCondGetLoweringDescription(void)
{
    if (probeNativeSupport())
        return "native conditional graph nodes (true skip)";
    return "predicated unroll: body embedded max_iters times, kernels guarded off "
           "after the condition clears. Pure GPU, no host round trip; post-"
           "convergence iterations still cost a kernel dispatch. Native "
           "conditional nodes and device-side graph launch are unavailable on "
           "this ROCm runtime.";
}

extern "C" hipError_t hipGraphCondGetLastUnrollCount(unsigned int* count_out)
{
    if (!count_out)
        return hipErrorInvalidValue;
    *count_out = g_lastUnrollCount.load();
    return hipSuccess;
}

#endif  // WP_ENABLE_HIP
