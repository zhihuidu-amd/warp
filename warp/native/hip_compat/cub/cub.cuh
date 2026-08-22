// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// HIP compatibility shim: satisfies `#include <cub/cub.cuh>` on ROCm.
//
// Warp uses CUB's device-wide primitives (radix sort, scan, reduce, select,
// run-length encode). ROCm ships hipCUB, which mirrors that API on top of
// rocPRIM, so the sources compile unchanged once cub:: resolves to hipcub::.
//
// The vendored cuBQL is already HIP-aware and declares `namespace cub` itself
// when __HIPCC__ is set, so the alias here is guarded to avoid redeclaring it
// as a different kind of entity.

#pragma once

#include <hipcub/hipcub.hpp>

#ifndef WP_HIP_CUB_NAMESPACE_ALIASED
#define WP_HIP_CUB_NAMESPACE_ALIASED
namespace cub {
using namespace hipcub;
}
#endif  // WP_HIP_CUB_NAMESPACE_ALIASED
