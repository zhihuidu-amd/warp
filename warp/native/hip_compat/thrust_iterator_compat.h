// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// rocPRIM advances and subscripts iterators with an unsigned index:
//
//   const KeyIterator block_keys = keys_input + block_offset;   // const unsigned int
//   value = block_keys[unsigned_index];
//
// rocThrust's transform_iterator declares operator+ and operator[] only for its
// signed difference_type, so both are ambiguous with an unsigned operand and
// rocprim::reduce_by_key fails to compile. CUB with CUDA Thrust accepts either.
//
// wp_make_transform_iterator returns an iterator that also accepts unsigned
// offsets. On CUDA it is exactly thrust::make_transform_iterator.

#pragma once

#include <thrust/iterator/transform_iterator.h>

#include <type_traits>

namespace wp {

#if defined(WP_ENABLE_HIP) && WP_ENABLE_HIP

template <typename Iterator>
struct unsigned_offsetable_iterator : Iterator {
    using difference_type = typename Iterator::difference_type;
    using reference = typename Iterator::reference;

    unsigned_offsetable_iterator() = default;

    // NOLINTNEXTLINE(google-explicit-constructor)
    __host__ __device__ unsigned_offsetable_iterator(const Iterator& base) : Iterator(base) {}

    __host__ __device__ unsigned_offsetable_iterator operator+(difference_type n) const
    {
        return unsigned_offsetable_iterator(static_cast<const Iterator&>(*this) + n);
    }

    template <typename U, typename = std::enable_if_t<std::is_unsigned_v<std::remove_cv_t<U>>>>
    __host__ __device__ unsigned_offsetable_iterator operator+(U n) const
    {
        return *this + static_cast<difference_type>(n);
    }

    __host__ __device__ reference operator[](difference_type n) const
    {
        return static_cast<const Iterator&>(*this)[n];
    }

    template <typename U, typename = std::enable_if_t<std::is_unsigned_v<std::remove_cv_t<U>>>>
    __host__ __device__ reference operator[](U n) const
    {
        return (*this)[static_cast<difference_type>(n)];
    }
};

template <typename Function, typename Iterator>
__host__ __device__ auto wp_make_transform_iterator(Iterator it, Function fun)
{
    return unsigned_offsetable_iterator<thrust::transform_iterator<Function, Iterator>>(
        thrust::make_transform_iterator(it, fun));
}

#else  // CUDA

template <typename Function, typename Iterator>
__host__ __device__ auto wp_make_transform_iterator(Iterator it, Function fun)
{
    return thrust::make_transform_iterator(it, fun);
}

#endif  // WP_ENABLE_HIP

}  // namespace wp
