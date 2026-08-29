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

namespace {

// A region opened by insert_while and not yet closed.
struct PendingRegion {
    hipGraphCondHandle handle;
    hipGraph_t parent_graph;
    hipGraph_t body_graph;  // handed to Warp; Warp captures the loop body into it
};

std::mutex g_cond_mutex;
std::map<hipStream_t, PendingRegion> g_pending;

// Unroll bound requested before a region exists. hipGraphCondSetMaxIters needs
// an open region, but the bound has to be known when the region OPENS, and
// Warp has no point in between where it could set it. Park it per stream and
// apply it at Begin.
std::map<hipStream_t, unsigned int> g_pending_max_iters;

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

}  // anonymous namespace

bool wp_hip_graph_reserve_cond_pool(unsigned int slots)
{
    // Idempotent: the library returns early once the slab exists, so repeated
    // calls are cheap and only the first one allocates.
    static bool reserved = false;
    if (reserved)
        return true;
    hipError_t err = hipGraphCondPoolReserve(slots);
    if (err != hipSuccess)
        return report(err, "hipGraphCondPoolReserve");
    reserved = true;
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
    // The graph is cloned, not consumed, so destroying our copy afterwards is
    // safe and avoids leaking one graph per conditional region.
    hipError_t err = hipGraphCondEndWithGraph(hip_stream, body_graph);
    hipGraphCondHandleDestroy(handle);
    if (body_graph)
        hipGraphDestroy(body_graph);
    return report(err, "hipGraphCondEndWithGraph");
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
