// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// HIP compatibility shim: satisfies `#include <cub/cub.cuh>` on ROCm.
//
// Warp uses CUB's device-wide primitives (radix sort, scan, reduce, select,
// run-length encode). ROCm ships hipCUB, which mirrors that API on top of
// rocPRIM, so the sources compile unchanged once cub:: resolves to hipcub::.
//
// The include lives here rather than in hip_util.h so that it is pulled in only
// by the translation units that actually use CUB, and always at file scope.

#pragma once

#include <hipcub/hipcub.hpp>

namespace cub = hipcub;
