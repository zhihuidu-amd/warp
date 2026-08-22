// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// HIP compatibility shim: satisfies `#include <cub/cub.cuh>` on ROCm.
//
// hipCUB mirrors CUB's device-wide primitives on top of rocPRIM, and
// hip_util.h aliases the cub namespace onto hipcub.

#pragma once

#include "../hip_util.h"
