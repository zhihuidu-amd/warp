// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// HIP compatibility shim: satisfies `#include <cub/device/device_scan.cuh>` on ROCm.
//
// hipCUB provides the same primitives; see cub/cub.cuh in this directory for
// how the cub namespace is mapped.

#pragma once

#include "../cub.cuh"
