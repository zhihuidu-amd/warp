// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Device-side builtins that CUDA provides and HIP does not.
//
// Included by hip_util.h, so any translation unit that reaches the shim gets
// these before the device headers use them.

#pragma once

#if defined(WP_ENABLE_HIP) && WP_ENABLE_HIP

// CUDA's __brkpt() traps on the device. HIP has no equivalent builtin;
// __builtin_trap() lowers to the same s_trap instruction on AMDGCN.
#if !defined(__brkpt)
#define __brkpt() __builtin_trap()
#endif  // __brkpt

#endif  // WP_ENABLE_HIP
