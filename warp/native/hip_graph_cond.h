// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Conditional graph regions for HIP/ROCm.
//
// ROCm has no equivalent of CUDA's conditional graph node
// (CU_GRAPH_NODE_TYPE_CONDITIONAL), so Warp's capture_while / capture_if cannot
// be expressed the way warp.cu builds them: cudaGraphConditionalHandleCreate is
// bound for HIP only to return hipErrorNotSupported, and cudaGraphAddNode and
// cudaGraphNodeParams are not bound at all.
//
// This header adapts Warp's conditional-node contract onto hipgraph_cond, which
// emulates the same semantics with a predicated static unroll: the loop body is
// emitted max_iters times and each copy is guarded by a device-side flag, so
// iterations after the condition goes false retire without doing work and
// without any host round trip (a host poll is illegal inside a capture).
//
// The adaptation is only in the shape of the call. Warp expects
//
//     insert_while(...)  ->  *body_graph_ret, *handle_ret
//     ...caller populates the body graph...
//     set_condition(..., handle)
//
// i.e. the body comes back as a GRAPH, not a stream. hipgraph_cond's native
// idiom is a body STREAM, which is why it grew hipGraphCondEndWithGraph.
//
// Measured on MI325X (gfx942, ROCm 7.2): up to 7.43x over dispatching every
// iteration, at 64M elements converging at iteration 1. The predication is
// verified rather than assumed -- effective iterations track the convergence
// point rather than the unroll bound, and replaying the same exec reproduces
// the counts, which is what catches a missing per-replay condition reset.
//
// IMPORTANT for callers: the slot pool must be reserved BEFORE a capture
// begins. A lazy allocation inside a live capture returns error 906 and kills
// the capture.

#if defined(WP_ENABLE_HIP) && WP_ENABLE_HIP

#include "hip_util.h"

// Reserve the condition-slot pool. Idempotent and cheap after the first call.
//
// Call this during setup, outside any capture. hipGraphCondHandleCreate would
// otherwise reserve lazily on first use, and that reservation allocates --
// which is illegal inside an active capture and returns 906, poisoning it.
// insert_while calls this defensively, but by then the parent stream is
// already capturing, so an explicit early call is the clean path.
bool wp_hip_graph_reserve_cond_pool(unsigned int slots);

// Open a conditional (while) region on `stream` and hand Warp a graph to fill.
//
// `condition` is a device pointer to an int that the body updates; the region
// stops doing work once it reads zero. `handle_ret` receives an opaque value
// that wp_cuda_graph_set_condition later uses to re-arm the condition, matching
// the CUDA path's handle semantics.
//
// Returns false and sets the Warp error string on failure.
bool wp_hip_graph_insert_while(void* stream, int* condition, void** body_graph_ret, uint64_t* handle_ret);

// Re-arm the condition for a region opened above. Mirrors
// wp_cuda_graph_set_condition; separated so warp.cu's dispatch stays symmetric.
bool wp_hip_graph_set_condition(void* stream, int* condition, uint64_t handle);

// Bound the static unroll. Must be called before the region opens: the bound
// cannot be changed once hipGraphCondBegin has run, and Warp has no point
// between the two where it could reach in, so a bound set early is parked
// per-stream and adopted by the next region.
//
// The default (WARP_HIP_COND_MAX_ITERS, 32) is a poor fit for a solver that
// needs one iteration: it emits 32 predicated bodies and costs roughly 2x the
// unconditional graph. Callers that know their budget should set it.
bool wp_hip_graph_set_max_iters(void* stream, unsigned int max_iters);

#endif  // WP_ENABLE_HIP
