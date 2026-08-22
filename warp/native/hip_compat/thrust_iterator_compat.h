// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// rocPRIM offsets iterators with an unsigned index:
//
//   const KeyIterator block_keys = keys_input + block_offset;   // const unsigned int
//
// rocThrust's transform_iterator declares operator+ only for its signed
// difference_type, so an unsigned operand is ambiguous and
// rocprim::reduce_by_key fails to compile. CUB with CUDA Thrust accepts both.
//
// Adding the unsigned overload keeps warp/native/deterministic.cu unchanged.
// It lives in namespace thrust so argument-dependent lookup finds it from
// rocPRIM's call site, and is constrained to Thrust's transform_iterator so no
// other type is affected.

#pragma once

#include <thrust/iterator/transform_iterator.h>

#include <type_traits>

namespace thrust {

template <typename F, typename I, typename U,
          typename = std::enable_if_t<std::is_unsigned_v<std::remove_cv_t<U>>>>
__host__ __device__ inline transform_iterator<F, I> operator+(
    const transform_iterator<F, I>& it, U n)
{
    using diff_t = typename transform_iterator<F, I>::difference_type;
    return it + static_cast<diff_t>(n);
}

}  // namespace thrust
