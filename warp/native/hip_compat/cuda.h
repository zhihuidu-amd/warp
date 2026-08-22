// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// HIP compatibility shim: satisfies `#include <cuda.h>` on ROCm.
//
// Placed on the include path only for HIP builds (see build_dll.py), so the
// device sources compile unmodified. The real translation lives in
// warp/native/hip_util.h.

#pragma once

#include "hip_util.h"
