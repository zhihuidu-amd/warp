// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// HIP compatibility shim: satisfies `#include <nvrtc.h>` on ROCm.
//
// hip_util.h already maps the NVRTC entry points Warp uses onto their hiprtc
// equivalents, so this only has to make the include resolve.

#pragma once

#include "hip_util.h"
