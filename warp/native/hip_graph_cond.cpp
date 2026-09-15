// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Adapt Warp's conditional-graph-node contract onto hipgraph_cond.
//
// The impedance mismatch is small but real. Warp's flow is
//
//     insert_while(...)          -> *body_graph_ret, *handle_ret
//     ... caller captures the body into that graph ...
//     set_condition(..., handle)
//
// so the body arrives as a GRAPH the caller already owns. hipgraph_cond's
// native idiom hands out a body STREAM instead, which is why it also provides
// hipGraphCondEndWithGraph -- the entry point meant for exactly this case.
//
// Warp calls insert_while and then closes the body later, so the region must
// stay open across the boundary. The open region is parked per-stream here and
// closed when Warp reports the body graph back.

#include "warp.h"

#if defined(WP_ENABLE_HIP) && WP_ENABLE_HIP

#include "error.h"
#include "hip_graph_cond.h"
#include "hipgraph_cond.h"

#include <map>
#include <mutex>
#include <vector>

namespace {

// A region opened by insert_while and not yet closed.
struct PendingRegion {
    hipGraphCondHandle handle;
    hipGraph_t parent_graph;
    hipGraph_t body_graph;  // handed to Warp; Warp captures the loop body into it
};

// An if/else pair opened by insert_if_else and not yet spliced.
//
// No region is open while this is parked -- see the header. Only the graphs,
// the handles and the condition pointer are held; the Begin/EndWithGraph pairs
// all happen inside splice_if_else.
struct PendingIfElse {
    int* condition;
    hipGraph_t parent_graph;
    hipGraphCondHandle if_handle;  // null if the caller asked for no if branch
    hipGraphCondHandle else_handle;  // null if the caller asked for no else branch
    hipGraphCondHandle inv_handle;  // scratch word holding !*condition; null with no else
    hipGraph_t if_graph;  // null if no if branch
    hipGraph_t else_graph;  // null if no else branch
};

std::mutex g_cond_mutex;
std::map<hipStream_t, PendingRegion> g_pending;

// A stack, not a single entry, because capture_if nests: the body of one branch
// may itself call capture_if, and Warp completes the inner one -- insert, both
// branches, splice -- before the outer branch closes. That is strict LIFO, and
// each level records the parent graph it was opened against, which splice_if_else
// re-checks. The innermost level is back().
std::map<hipStream_t, std::vector<PendingIfElse>> g_pending_if;

// Unroll bound requested before a region exists. hipGraphCondSetMaxIters needs
// an open region, but the bound has to be known when the region OPENS, and
// Warp has no point in between where it could set it. Park it per stream and
// apply it at Begin.
std::map<hipStream_t, unsigned int> g_pending_max_iters;

// Guard word of the region that lexically encloses the NEXT one to be opened on
// this stream, or absent for a top-level region. Parked for the same reason as
// g_pending_max_iters: the value has to be known when the region opens, and the
// only caller who knows it -- context.py, which owns the guard stack -- has no
// point inside insert_while / insert_if_else where it could supply it.
//
// It exists at all because the HIP lowering predicates BODY kernels but cannot
// predicate the SEED kernels that write the guards: something has to make the
// first write. Cloning an enclosing body graph clones the inner region's seeds
// with it, and those clones run unconditionally, re-arming the inner guard from
// the user's untouched condition array even when the enclosing guard is 0. On
// gfx942 that returned 5005 where 385 was correct -- silently, with no error.
// hipGraphCondSetEnclosingGuard makes every seed compute
// `(*condition != 0) && (*enclosing != 0)` instead. See hipgraph_cond.h.
//
// Consumed and erased when a region opens, so a value parked for a region that
// was never opened cannot leak into an unrelated one later.
std::map<hipStream_t, const unsigned int*> g_pending_enclosing;

// Body graphs whose region has been closed but which the caller may still be
// holding. See wp_hip_graph_release_body_graphs in the header for why they
// cannot be destroyed at close time.
std::vector<hipGraph_t> g_retired_body_graphs;

// Default when the caller never sets one. 32 is what the library uses, and it
// is a poor default for a solver needing one iteration -- it emits 32
// predicated bodies and costs roughly 2x the unconditional graph. Warp should
// set the real budget; this only keeps an unset case from being unbounded.
constexpr unsigned int kDefaultMaxIters = 32;

bool report(hipError_t err, const char* what)
{
    if (err == hipSuccess)
        return true;
    wp::set_error_string("Warp error: %s failed: %s (%d)", what, hipGetErrorString(err), (int)err);
    return false;
}

// Take the guard parked for this stream, removing it. Caller must hold
// g_cond_mutex. A null return means "no enclosing region", which is what a
// top-level capture_if / capture_while gets and which leaves the seed kernels
// behaving exactly as they did before enclosing guards existed.
const unsigned int* take_enclosing_locked(hipStream_t stream)
{
    auto it = g_pending_enclosing.find(stream);
    if (it == g_pending_enclosing.end())
        return nullptr;
    const unsigned int* guard = it->second;
    g_pending_enclosing.erase(it);
    return guard;
}

// Attach an enclosing guard to a handle. Must run BEFORE that handle's
// hipGraphCondBegin, which is what emits the seed that reads it. A null guard
// is still worth setting: handles are recycled from a pool, and leaving a
// previous occupant's guard in place would predicate this region on a branch it
// has nothing to do with.
bool set_enclosing(hipGraphCondHandle handle, const unsigned int* enclosing)
{
    if (!handle)
        return true;
    hipError_t err = hipGraphCondSetEnclosingGuard(handle, enclosing);
    return err == hipSuccess ? true : report(err, "hipGraphCondSetEnclosingGuard");
}

}  // anonymous namespace

bool wp_hip_graph_set_enclosing_guard(void* stream, void* guard)
{
    std::lock_guard<std::mutex> lock(g_cond_mutex);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    if (guard)
        g_pending_enclosing[hip_stream] = static_cast<const unsigned int*>(guard);
    else
        g_pending_enclosing.erase(hip_stream);
    return true;
}

bool wp_hip_graph_reserve_cond_pool(unsigned int slots)
{
    // Idempotent per device: the slab is keyed by the calling thread's current
    // device, and hipGraphCondPoolReserve returns early once that device's slab
    // is big enough, so repeated calls are a lock and a map lookup.
    //
    // Deliberately NOT memoised behind a process-global flag. It was, and a
    // capture on a second GPU then never reached the reserve at all: the first
    // device set the flag, device 1 fell through to the lazy reserve inside
    // hipGraphCondHandleCreate, and that runs with the parent stream already
    // capturing. hipMalloc is tolerated there under the Relaxed capture mode,
    // but the slab's hipMemsetD32 is not -- job 67941557 failed every device-1
    // conditional test with "900 (operation not permitted when stream is
    // capturing)". A per-device memo would work; it would also be a second
    // device-keyed map shadowing the one the library already maintains.
    hipError_t err = hipGraphCondPoolReserve(slots);
    if (err != hipSuccess)
        return report(err, "hipGraphCondPoolReserve");
    return true;
}

bool wp_hip_graph_query_truncations(unsigned int* count_out)
{
    if (!count_out)
        return false;
    hipError_t err = hipGraphCondQueryTruncations(count_out);
    if (err != hipSuccess)
        return report(err, "hipGraphCondQueryTruncations");
    return true;
}

bool wp_hip_graph_insert_while(void* stream, int* condition, void** body_graph_ret, uint64_t* handle_ret)
{
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);

    // The pool is reserved in wp_cuda_graph_begin_capture, before any capture
    // is active. It deliberately is NOT reserved here: by this point the parent
    // stream is already capturing, and the pool's allocation is illegal there
    // (906, "operation would make the legacy stream depend on a capturing
    // blocking stream"). Jobs 67853883 and 67854737 hit that on
    // hipGraphCondHandleCreate and hipGraphCondPoolReserve respectively --
    // moving the call earlier WITHIN this function cannot help, because the
    // whole function runs inside the capture.

    // The parent must already be capturing; the region splices into its graph.
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    hipGraph_t parent_graph = nullptr;
    const hipGraphNode_t* deps = nullptr;
    size_t dep_count = 0;
    hipError_t err = hipStreamGetCaptureInfo_v2(hip_stream, &status, nullptr, &parent_graph, &deps, &dep_count);
    if (err != hipSuccess)
        return report(err, "hipStreamGetCaptureInfo_v2");
    if (!parent_graph || status != hipStreamCaptureStatusActive) {
        wp::set_error_string("Stream is not capturing");
        return false;
    }

    std::lock_guard<std::mutex> lock(g_cond_mutex);

    if (g_pending.find(hip_stream) != g_pending.end()) {
        // Nested while on one stream is not supported: the parked region would
        // be overwritten and the outer one would never close. Fail loudly
        // rather than silently drop it.
        wp::set_error_string("Warp error: a conditional region is already open on this stream");
        return false;
    }

    // Reset to 1 (keep looping) on every replay. Without a per-replay reset a
    // graph that converged on replay N would still read "done" on replay N+1.
    hipGraphCondHandle handle = nullptr;
    err = hipGraphCondHandleCreate(&handle, parent_graph, hipGraphCondAssignDefault, 0);
    if (err != hipSuccess)
        return report(err, "hipGraphCondHandleCreate");

    unsigned int max_iters = kDefaultMaxIters;
    auto it = g_pending_max_iters.find(hip_stream);
    if (it != g_pending_max_iters.end())
        max_iters = it->second;

    // Strictly before Begin: Begin emits the seed kernel that reads this, and
    // the unroll's between-copy refresh seeds read it again later off the
    // handle. A while loop nested inside a suppressed if-branch re-arms its own
    // guard on every copy without it, and runs the whole unroll inside a branch
    // that was not taken.
    if (!set_enclosing(handle, take_enclosing_locked(hip_stream))) {
        hipGraphCondHandleDestroy(handle);
        return false;
    }

    err = hipGraphCondBegin(hip_stream, handle, hipGraphCondTypeWhile, condition, max_iters);
    if (err != hipSuccess) {
        hipGraphCondHandleDestroy(handle);
        return report(err, "hipGraphCondBegin");
    }

    // Hand Warp a FRESH, EMPTY graph -- never the parent.
    //
    // context.py does `main_graph.graph = body_graph` and redirects its own
    // capture into whatever pointer we return here. Returning the parent meant
    // the loop body was captured straight back into the parent graph, where it
    // ran once inline instead of becoming a loop body. That is exactly what job
    // 67855315 measured: the body executed once at every convergence point
    // (stop_at 3, 5 and 8 all ran 1).
    hipGraph_t body_graph = nullptr;
    err = hipGraphCreate(&body_graph, 0);
    if (err != hipSuccess) {
        hipGraphCondAbortRegion(hip_stream);
        hipGraphCondHandleDestroy(handle);
        return report(err, "hipGraphCreate");
    }

    g_pending[hip_stream] = PendingRegion { handle, parent_graph, body_graph };

    *body_graph_ret = body_graph;
    *handle_ret = reinterpret_cast<uint64_t>(handle);
    return true;
}

bool wp_hip_graph_get_guard(uint64_t handle_bits, void** guard_ret)
{
    if (!guard_ret) {
        wp::set_error_string("Warp error: guard: null out pointer");
        return false;
    }
    *guard_ret = nullptr;

    hipGraphCondHandle handle = reinterpret_cast<hipGraphCondHandle>(handle_bits);

    {
        // Only hand out a guard for a region this shim currently has open.
        // hipGraphCondSetGuard validates the handle against the library's own
        // live set, but a handle that is live there and unknown here means the
        // caller is binding a guard from someone else's region -- the body
        // kernels would then test a slot no one re-arms. Refuse instead.
        std::lock_guard<std::mutex> lock(g_cond_mutex);
        bool known = false;
        for (const auto& entry : g_pending) {
            if (entry.second.handle == handle) {
                known = true;
                break;
            }
        }
        if (!known) {
            wp::set_error_string("Warp error: guard: no open region for this conditional handle");
            return false;
        }
    }

    unsigned int* guard = nullptr;
    hipError_t err = hipGraphCondSetGuard(handle, &guard);
    if (err != hipSuccess)
        return report(err, "hipGraphCondSetGuard");

    *guard_ret = static_cast<void*>(guard);
    return true;
}

bool wp_hip_graph_set_condition(void* stream, int* condition, uint64_t handle_bits)
{
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    (void)condition;  // the condition pointer was bound at Begin

    std::lock_guard<std::mutex> lock(g_cond_mutex);

    auto it = g_pending.find(hip_stream);
    if (it == g_pending.end()) {
        wp::set_error_string("Warp error: no conditional region is open on this stream");
        return false;
    }

    if (reinterpret_cast<uint64_t>(it->second.handle) != handle_bits) {
        // A mismatch means the caller is closing a different region than the
        // one open here. Closing the wrong region would corrupt the parent's
        // capture frontier, so refuse.
        wp::set_error_string("Warp error: conditional handle does not match the open region");
        return false;
    }

    hipGraphCondHandle handle = it->second.handle;
    hipGraph_t body_graph = it->second.body_graph;
    g_pending.erase(it);

    // EndWithGraph, NOT End. hipGraphCondEnd splices the library's own private
    // body stream, which Warp never launches onto -- so it is empty and the
    // region does nothing. The header says so directly: EndWithGraph "is the
    // path Warp needs ... capture_while does not launch the body on a stream it
    // is given -- it redirects its OWN stream into a body graph via
    // capture_pause/capture_resume, then hands the finished graph back."
    //
    // The graph is cloned, not consumed, so our copy can be freed -- but NOT
    // here. The caller's stream is still capturing into this graph at this
    // point (context.py::capture_while pauses the body capture only after
    // set_condition returns), and it then hands the same pointer to
    // wp_cuda_graph_check_conditional_body. Destroying it here made both of
    // those reads use-after-free; on the error path the following
    // hipStreamEndCapture dumped core. Park it instead.
    hipError_t err = hipGraphCondEndWithGraph(hip_stream, body_graph);
    hipGraphCondHandleDestroy(handle);
    if (body_graph)
        g_retired_body_graphs.push_back(body_graph);

    if (err != hipSuccess) {
        // Say WHICH operation the body could not carry. The cloner replicates
        // node params one type at a time and refuses anything it cannot
        // reproduce -- a memory allocation, an event, a child graph -- with a
        // bare hipErrorNotSupported that names nothing. Warp's contract
        // promises the operation by name, and on CUDA that message comes from
        // wp_cuda_graph_check_conditional_body one step later in
        // context.py::capture_while; running the same checker here makes HIP
        // report the same thing.
        //
        // Only on the failure path, and only after EndWithGraph, so the checker
        // can never perturb a splice that was going to succeed. (That ordering
        // was first tried as a FIX for test_while_capture's invalid argument
        // (1), on the theory that walking the body while the stream still
        // captures into it was what broke the splice. Job 67922602 refuted it:
        // moving the checker here changed nothing, byte for byte. The real
        // cause was in hipgraph_cond's frontier re-anchor -- see spliceBody's
        // closing comment -- and is fixed there. Set HIPGRAPH_COND_VERBOSE=1
        // to have the library name its own failure path; every one of them
        // logs now, which is what made the last one findable.
        bool named = !wp_cuda_graph_check_conditional_body(body_graph);
        if (!named)
            report(err, "hipGraphCondEndWithGraph");  // checker found nothing; report the raw error

        // Abandon the region. EndWithGraph failed, so it did not consume it,
        // and an open region leaves the parent capture forked -- see
        // wp_hip_graph_abort_open_region. Ignore the abort's own status: it
        // must not overwrite the error string we just built.
        hipGraphCondAbortRegion(hip_stream);
        return false;
    }

    return true;
}

namespace {

// Resolve the graph the stream is capturing into, or fail with Warp's error
// string set. Shared by the if/else entry points, which have the same
// precondition as insert_while: the parent must be capturing, because the
// regions splice into its graph.
bool capturing_parent(hipStream_t hip_stream, hipGraph_t* parent_graph_ret)
{
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    hipGraph_t parent_graph = nullptr;
    const hipGraphNode_t* deps = nullptr;
    size_t dep_count = 0;
    hipError_t err = hipStreamGetCaptureInfo_v2(hip_stream, &status, nullptr, &parent_graph, &deps, &dep_count);
    if (err != hipSuccess)
        return report(err, "hipStreamGetCaptureInfo_v2");
    if (!parent_graph || status != hipStreamCaptureStatusActive) {
        wp::set_error_string("Stream is not capturing");
        return false;
    }
    *parent_graph_ret = parent_graph;
    return true;
}

// Destroy the handles of a pending pair and park its graphs for later release.
// Graphs are parked rather than destroyed for the same reason set_condition
// parks them: the caller may still be holding the pointers.
void retire_if_else(PendingIfElse& p)
{
    if (p.if_handle)
        hipGraphCondHandleDestroy(p.if_handle);
    if (p.else_handle)
        hipGraphCondHandleDestroy(p.else_handle);
    if (p.inv_handle)
        hipGraphCondHandleDestroy(p.inv_handle);
    if (p.if_graph)
        g_retired_body_graphs.push_back(p.if_graph);
    if (p.else_graph)
        g_retired_body_graphs.push_back(p.else_graph);
}

}  // anonymous namespace

bool wp_hip_graph_insert_if_else(void* stream, int* condition, void** if_graph_ret, void** else_graph_ret)
{
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);

    if (!if_graph_ret && !else_graph_ret) {
        wp::set_error_string("Warp error: insert_if_else: neither branch was requested");
        return false;
    }
    if (!condition) {
        wp::set_error_string("Warp error: insert_if_else: null condition pointer");
        return false;
    }

    hipGraph_t parent_graph = nullptr;
    if (!capturing_parent(hip_stream, &parent_graph))
        return false;

    std::lock_guard<std::mutex> lock(g_cond_mutex);

    // A nested capture_if pushes onto the stack rather than replacing the outer
    // entry; splice_if_else pops the innermost. What must not happen is two
    // levels sharing a parent graph, which would mean the inner one was opened
    // after the outer had already been spliced or before its branch capture had
    // been repointed -- in either case the inner region would land in the wrong
    // graph.
    auto& stack = g_pending_if[hip_stream];
    for (const PendingIfElse& open : stack) {
        if (open.parent_graph == parent_graph) {
            wp::set_error_string("Warp error: an if/else pair is already open on this stream against the same graph");
            return false;
        }
    }

    // The slot pool was reserved in wp_cuda_graph_begin_capture, outside any
    // capture. Creating handles here is safe only because of that; a lazy
    // reservation at this point returns 906 and kills the capture.
    PendingIfElse pending {};
    pending.condition = condition;
    pending.parent_graph = parent_graph;

    // hipGraphCondAssignDefault (reset to 1 each replay) matches insert_while.
    // The value barely matters here -- Begin re-seeds the slot from the
    // condition before the single body copy -- but a default of "run" keeps the
    // failure mode "ran a branch" rather than "silently ran nothing", which is
    // the easier one to notice.
    auto create = [&](hipGraphCondHandle* out) -> bool {
        hipError_t err = hipGraphCondHandleCreate(out, parent_graph, hipGraphCondAssignDefault, 0);
        return err == hipSuccess ? true : report(err, "hipGraphCondHandleCreate");
    };

    bool ok = true;
    if (if_graph_ret)
        ok = create(&pending.if_handle);
    if (ok && else_graph_ret) {
        // Two handles for the else branch: one is the region, the other is
        // storage for the complement of the condition. The complement has to
        // live in device memory that outlives the capture, and a condition slot
        // is exactly that -- already allocated, already never recycled once
        // baked into a graph. Allocating our own here would return 906.
        ok = create(&pending.else_handle) && create(&pending.inv_handle);
    }
    if (ok) {
        // Attach the enclosing guard now, not at splice time. It is a property of
        // the HANDLE and Begin is what consumes it, so setting it here covers both
        // Begins that splice_if_else performs -- and one fewer thing has to be
        // remembered on the far side of two branch captures.
        //
        // Both branch handles get it; inv_handle deliberately does not. The
        // complement must stay a faithful !cond, because the else region's own
        // seed is what ANDs it with the enclosing guard. Conjoining it twice would
        // be redundant, and conjoining it there INSTEAD would make both branches
        // run when the enclosing guard is 0.
        const unsigned int* enclosing = take_enclosing_locked(hip_stream);
        ok = set_enclosing(pending.if_handle, enclosing) && set_enclosing(pending.else_handle, enclosing);
    }
    if (ok) {
        // Fresh, EMPTY graphs -- never the parent. context.py repoints its own
        // capture at whatever pointer we return, so returning the parent would
        // capture the branch body straight back into the parent graph, where it
        // would run unconditionally.
        if (if_graph_ret)
            ok = report(hipGraphCreate(&pending.if_graph, 0), "hipGraphCreate");
        if (ok && else_graph_ret)
            ok = report(hipGraphCreate(&pending.else_graph, 0), "hipGraphCreate");
    }

    if (!ok) {
        retire_if_else(pending);
        return false;
    }

    stack.push_back(pending);

    if (if_graph_ret)
        *if_graph_ret = pending.if_graph;
    if (else_graph_ret)
        *else_graph_ret = pending.else_graph;
    return true;
}

bool wp_hip_graph_get_if_else_guards(void* stream, void** if_guard_ret, void** else_guard_ret)
{
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);

    if (if_guard_ret)
        *if_guard_ret = nullptr;
    if (else_guard_ret)
        *else_guard_ret = nullptr;

    std::lock_guard<std::mutex> lock(g_cond_mutex);

    auto it = g_pending_if.find(hip_stream);
    if (it == g_pending_if.end() || it->second.empty()) {
        wp::set_error_string("Warp error: no if/else pair is open on this stream");
        return false;
    }
    // The innermost pair -- the one whose branches are about to be captured.
    const PendingIfElse& pending = it->second.back();

    auto fetch = [&](hipGraphCondHandle handle, void** out) -> bool {
        if (!handle || !out)
            return true;  // branch not requested, or caller does not want it
        unsigned int* guard = nullptr;
        hipError_t err = hipGraphCondSetGuard(handle, &guard);
        if (err != hipSuccess)
            return report(err, "hipGraphCondSetGuard");
        *out = static_cast<void*>(guard);
        return true;
    };

    return fetch(pending.if_handle, if_guard_ret) && fetch(pending.else_handle, else_guard_ret);
}

bool wp_hip_graph_splice_if_else(void* stream)
{
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);

    // The parent must be capturing again. capture_if pauses it to fill each
    // branch graph and resumes it at the end; this call belongs strictly after
    // that resume, and checking is how a future reordering gets caught here
    // instead of as a corrupted graph.
    hipGraph_t parent_graph = nullptr;
    if (!capturing_parent(hip_stream, &parent_graph))
        return false;

    std::lock_guard<std::mutex> lock(g_cond_mutex);

    auto it = g_pending_if.find(hip_stream);
    if (it == g_pending_if.end() || it->second.empty()) {
        wp::set_error_string("Warp error: no if/else pair is open on this stream");
        return false;
    }
    // Pop the innermost pair. Strict LIFO: a nested capture_if splices inside the
    // enclosing branch body, so it always finishes first.
    PendingIfElse pending = it->second.back();
    it->second.pop_back();
    if (it->second.empty())
        g_pending_if.erase(it);

    if (pending.parent_graph != parent_graph) {
        // The stream is capturing into a different graph than the one the
        // handles were created against. Splicing would put the regions in a
        // graph whose handles are foreign to it.
        retire_if_else(pending);
        wp::set_error_string("Warp error: the capture moved to a different graph between insert_if_else and splice");
        return false;
    }

    // Splice one region, from an already-captured body graph. Begin seeds the
    // handle's slot from `cond` as ordinary kernel nodes in the parent, then
    // EndWithGraph clones the body in behind them. max_iters is 1: an If region
    // runs its body at most once.
    auto splice_one = [&](hipGraphCondHandle handle, const int* cond, hipGraph_t body) -> bool {
        hipError_t err = hipGraphCondBegin(hip_stream, handle, hipGraphCondTypeIf, cond, 1);
        if (err != hipSuccess)
            return report(err, "hipGraphCondBegin");
        err = hipGraphCondEndWithGraph(hip_stream, body);
        if (err != hipSuccess) {
            // Name the operation the body could not carry, the same way
            // set_condition does -- EndWithGraph reports a bare
            // hipErrorNotSupported that identifies nothing. The checker returns
            // false when it found and named one.
            if (wp_cuda_graph_check_conditional_if_body(body))
                report(err, "hipGraphCondEndWithGraph");  // checker found nothing; report the raw error

            // Abandon the region either way. EndWithGraph failed, so it did not
            // consume it, and an open region leaves the parent capture forked.
            // Its status must not overwrite the error string just built.
            hipGraphCondAbortRegion(hip_stream);
            return false;
        }
        return true;
    };

    bool ok = true;

    // 1. The complement, BEFORE either region. Both branches must see the same
    //    condition value; the if-body is spliced between this node and the else
    //    region's own seeding kernel, so an if-body that writes the condition
    //    would otherwise flip the else branch on.
    if (pending.else_handle) {
        hipError_t err = hipGraphCondWriteComplement(hip_stream, pending.inv_handle, pending.condition);
        if (err != hipSuccess)
            ok = report(err, "hipGraphCondWriteComplement");
    }

    // 2. The if region.
    if (ok && pending.if_handle)
        ok = splice_one(pending.if_handle, pending.condition, pending.if_graph);

    // 3. The else region, predicated on the complement written in step 1.
    if (ok && pending.else_handle) {
        unsigned int* inv_slot = nullptr;
        hipError_t err = hipGraphCondHandleGetDevicePtr(pending.inv_handle, &inv_slot);
        if (err != hipSuccess)
            ok = report(err, "hipGraphCondHandleGetDevicePtr");
        else
            ok = splice_one(pending.else_handle, reinterpret_cast<const int*>(inv_slot), pending.else_graph);
    }

    retire_if_else(pending);
    return ok;
}

void wp_hip_graph_release_body_graphs()
{
    std::lock_guard<std::mutex> lock(g_cond_mutex);
    for (hipGraph_t graph : g_retired_body_graphs)
        hipGraphDestroy(graph);
    g_retired_body_graphs.clear();
}

bool wp_hip_graph_abort_open_region(void* stream)
{
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);

    std::lock_guard<std::mutex> lock(g_cond_mutex);

    // An abandoned if/else pair is a leak rather than a forked capture -- no
    // region was ever opened for it (see insert_if_else) -- but it still holds
    // condition slots and two graphs, and leaving the entry behind would make
    // the NEXT capture_if on this stream refuse with "already open". Drain the
    // whole stack -- an exception inside a nested branch abandons every level
    // above it too. Independent of the while-region case below.
    auto if_it = g_pending_if.find(hip_stream);
    if (if_it != g_pending_if.end()) {
        std::vector<PendingIfElse> stack;
        stack.swap(if_it->second);
        g_pending_if.erase(if_it);
        for (PendingIfElse& pending : stack)
            retire_if_else(pending);
    }

    auto it = g_pending.find(hip_stream);
    if (it == g_pending.end())
        return true;  // nothing open -- the normal path, where set_condition closed it

    // Getting here means capture_while raised between insert_while and
    // set_condition: the body callback threw, or the body was a graph that
    // wp_cuda_graph_insert_child_graph refused. Python's except restores and
    // resumes the parent capture but cannot know a region is still open, and an
    // open region leaves the parent's capture frontier forked, so the
    // hipStreamEndCapture in end_capture fails and HIP keeps the stream
    // capturing. That is a whole-process failure -- every later launch and
    // synchronize on that stream returns 900 -- from one recoverable error.
    hipGraphCondHandle handle = it->second.handle;
    hipGraph_t body_graph = it->second.body_graph;
    g_pending.erase(it);

    hipError_t err = hipGraphCondAbortRegion(hip_stream);
    hipGraphCondHandleDestroy(handle);
    if (body_graph)
        g_retired_body_graphs.push_back(body_graph);
    return report(err, "hipGraphCondAbortRegion");
}

bool wp_hip_graph_set_max_iters(void* stream, unsigned int max_iters)
{
    if (max_iters == 0) {
        wp::set_error_string("Warp error: max_iters must be at least 1");
        return false;
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    std::lock_guard<std::mutex> lock(g_cond_mutex);

    // If a region is already open the library owns the bound; otherwise park it
    // for the next Begin. Setting it on an open region returns
    // hipErrorIllegalState, which is why the parked path exists at all.
    if (g_pending.find(hip_stream) != g_pending.end())
        return report(hipGraphCondSetMaxIters(hip_stream, max_iters), "hipGraphCondSetMaxIters");

    g_pending_max_iters[hip_stream] = max_iters;
    return true;
}

#endif  // WP_ENABLE_HIP
