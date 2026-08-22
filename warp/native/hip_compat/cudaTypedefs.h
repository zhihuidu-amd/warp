// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// HIP compatibility shim: satisfies `#include <cudaTypedefs.h>` on ROCm.
//
// The CUDA toolkit header declares the versioned driver-API function pointer
// types (PFN_cuFoo_vNNNN). hip_util.h synthesises the same names from the HIP
// entry points, so forwarding there is sufficient.

#pragma once

#include "hip_util.h"
