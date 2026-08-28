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

bool wp_hip_graph_insert_while(void* stream, int* condition, void** body_graph_ret, uint64_t* handle_ret)
{
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);

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

    // Warp captures the body itself and hands the graph back at set_condition
    // time, so the body graph pointer it receives here is the parent's -- the
    // region is what actually carries the body. Park the state until then.
    g_pending[hip_stream] = PendingRegion { handle, parent_graph };

    *body_graph_ret = parent_graph;
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
    g_pending.erase(it);

    hipError_t err = hipGraphCondEnd(hip_stream);
    hipGraphCondHandleDestroy(handle);
    return report(err, "hipGraphCondEnd");
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
