// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// HIP compatibility shim: satisfies `#include <math_constants.h>` on ROCm.
//
// CUDA ships this header with the toolkit. ROCm has no equivalent, so the few
// constants Warp's vendored cuBQL uses are defined here with the same names
// and values.

#pragma once

#include <cfloat>

#ifndef CUDART_INF_F
#define CUDART_INF_F __builtin_huge_valf()
#endif
#ifndef CUDART_NAN_F
#define CUDART_NAN_F __builtin_nanf("")
#endif
#ifndef CUDART_INF
#define CUDART_INF __builtin_huge_val()
#endif
#ifndef CUDART_NAN
#define CUDART_NAN __builtin_nan("")
#endif
// cuBQL version-gates on this. hip_util.h maps CUDA_VERSION to HIP_VERSION;
// mirror it so the same gates evaluate consistently.
#ifndef CUDART_VERSION
#define CUDART_VERSION CUDA_VERSION
#endif
#ifndef CUDART_PI_F
#define CUDART_PI_F 3.141592654f
#endif
#ifndef CUDART_PI
#define CUDART_PI 3.1415926535897931e+0
#endif
