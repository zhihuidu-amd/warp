// SPDX-FileCopyrightText: Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Lane-participation macros for HIP, pre-defined as a set.
//
// tile.h guards each of WP_TILE_WARP_SIZE and WP_TILE_LANE_MASK_{ALL,POPC,FFS}, and
// tile_radix_sort.h guards WP_TILE_BITONIC_GROUP_MASK, with #ifndef so that a build targeting a different lane width can define the whole set
// without patching that file. This is that definition for HIP.
//
// A CUDA warp is 32 lanes; an AMD wavefront is 64. Anything that indexes lanes or
// builds a lane mask has to follow, so the width and the mask type are selected
// together. HIP enforces the pairing: __ballot_sync static_asserts that the mask is
// 64-bit and returns unsigned long long, so a 32-bit mask is a compile error rather
// than a silent truncation.
//
// Included from tile.h before the guarded defaults, and only when WP_ENABLE_HIP is set.

#define WP_TILE_WARP_SIZE 64

#define WP_TILE_LANE_MASK_ALL 0xffffffffffffffffull
#define WP_TILE_LANE_MASK_POPC(mask) __popcll(mask)
// __ffs must widen with the mask for the same reason __popc does. HIP overloads
// __ffs(int) and __ffsll(long long), so passing a 64-bit mask to __ffs is
// ambiguous rather than truncating -- "call to '__ffs' is ambiguous".
#define WP_TILE_LANE_MASK_FFS(mask) __ffsll(mask)

// The bitonic group size stays 32 on every backend -- it is a property of the
// algorithm, not of the hardware, so it is not defined here. Only the mask has to
// widen: HIP's __shfl_xor_sync static_asserts that the mask is 64-bit, so a 32-bit
// literal is a compile error rather than a silent truncation, exactly as for
// __ballot_sync above.
#define WP_TILE_BITONIC_GROUP_MASK 0xffffffffffffffffull
