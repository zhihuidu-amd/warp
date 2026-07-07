# Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.
# AMD rocWMMA tile_matmul unit test
# File: warp/tests/cuda/test_tile_rocwmma.py

"""
Unit tests for AMD rocWMMA tile_matmul fast path.
Uses plain pytest — no warp-internal test decorators required.
Tests run only on HIP devices (gfx942+, ROCm 7.x+).
"""

import pytest
import numpy as np
import warp as wp


def get_hip_device():
    """Return first HIP device, or None if no HIP device available."""
    wp.init()
    for d in wp.get_cuda_devices():
        if d.is_hip:
            return str(d)
    return None


HIP_DEVICE = get_hip_device()
pytestmark = pytest.mark.skipif(HIP_DEVICE is None, reason="No HIP device available")


@wp.kernel
def tile_matmul_16x16_kernel(
    A: wp.array2d(dtype=float),
    B: wp.array2d(dtype=float),
    C: wp.array2d(dtype=float),
):
    """16x16 tile matmul: C = A @ B"""
    wp.tile_matmul(wp.tile_load(A, shape=(16, 16), storage="shared"),
                   wp.tile_load(B, shape=(16, 16), storage="shared"),
                   wp.tile_load(C, shape=(16, 16), storage="shared"))


@wp.kernel
def tile_matmul_16x16_k32_kernel(
    A: wp.array2d(dtype=float),
    B: wp.array2d(dtype=float),
    C: wp.array2d(dtype=float),
):
    """16x16 output, K=32"""
    wp.tile_matmul(wp.tile_load(A, shape=(16, 32), storage="shared"),
                   wp.tile_load(B, shape=(32, 16), storage="shared"),
                   wp.tile_load(C, shape=(16, 16), storage="shared"))


@wp.kernel
def tile_matmul_alpha_beta_kernel(
    A: wp.array2d(dtype=float),
    B: wp.array2d(dtype=float),
    C_in: wp.array2d(dtype=float),
    C_out: wp.array2d(dtype=float),
    alpha: float,
    beta: float,
):
    """C_out = alpha * A @ B + beta * C_in"""
    a_tile = wp.tile_load(A, shape=(16, 16), storage="shared")
    b_tile = wp.tile_load(B, shape=(16, 16), storage="shared")
    c_tile = wp.tile_load(C_in, shape=(16, 16), storage="shared")
    wp.tile_matmul(a_tile, b_tile, c_tile, alpha=alpha, beta=beta)
    wp.tile_store(C_out, c_tile)


def _run_matmul(M, N, K, alpha=1.0, beta=0.0):
    """Helper: run tile_matmul and return result + numpy reference."""
    device = HIP_DEVICE
    rng = np.random.default_rng(42)
    A_np = rng.standard_normal((M, K)).astype(np.float32)
    B_np = rng.standard_normal((K, N)).astype(np.float32)
    C_np = rng.standard_normal((M, N)).astype(np.float32) if beta != 0.0 \
           else np.zeros((M, N), dtype=np.float32)

    ref = alpha * (A_np @ B_np) + beta * C_np

    A_wp = wp.array(A_np, dtype=float, device=device)
    B_wp = wp.array(B_np, dtype=float, device=device)
    C_wp = wp.array(C_np, dtype=float, device=device)
    C_out_wp = wp.zeros((M, N), dtype=float, device=device)

    if alpha == 1.0 and beta == 0.0 and K == 16:
        wp.launch_tiled(tile_matmul_16x16_kernel, dim=(1,),
                       inputs=[A_wp, B_wp, C_wp], outputs=[C_out_wp],
                       block_dim=64, device=device)
    elif alpha == 1.0 and beta == 0.0 and K == 32:
        wp.launch_tiled(tile_matmul_16x16_k32_kernel, dim=(1,),
                       inputs=[A_wp, B_wp, C_wp], outputs=[C_out_wp],
                       block_dim=64, device=device)
    else:
        wp.launch_tiled(tile_matmul_alpha_beta_kernel, dim=(1,),
                       inputs=[A_wp, B_wp, C_wp, float(alpha), float(beta)],
                       outputs=[C_out_wp],
                       block_dim=64, device=device)

    return C_out_wp.numpy(), ref


def test_tile_matmul_16x16_identity():
    """16x16 matmul with identity B → result equals A."""
    device = HIP_DEVICE
    A_np = np.random.randn(16, 16).astype(np.float32)
    B_np = np.eye(16, dtype=np.float32)
    A_wp = wp.array(A_np, dtype=float, device=device)
    B_wp = wp.array(B_np, dtype=float, device=device)
    C_wp = wp.zeros((16, 16), dtype=float, device=device)
    C_out_wp = wp.zeros((16, 16), dtype=float, device=device)
    wp.launch_tiled(tile_matmul_16x16_kernel, dim=(1,),
                   inputs=[A_wp, B_wp, C_wp], outputs=[C_out_wp],
                   block_dim=64, device=device)
    np.testing.assert_allclose(C_out_wp.numpy(), A_np, atol=1e-4)


def test_tile_matmul_16x16_zeros():
    """16x16 matmul with zero B → result is zero."""
    device = HIP_DEVICE
    A_wp = wp.ones((16, 16), dtype=float, device=device)
    B_wp = wp.zeros((16, 16), dtype=float, device=device)
    C_wp = wp.zeros((16, 16), dtype=float, device=device)
    C_out_wp = wp.ones((16, 16), dtype=float, device=device)
    wp.launch_tiled(tile_matmul_16x16_kernel, dim=(1,),
                   inputs=[A_wp, B_wp, C_wp], outputs=[C_out_wp],
                   block_dim=64, device=device)
    np.testing.assert_allclose(C_out_wp.numpy(), np.zeros((16, 16)), atol=1e-6)


def test_tile_matmul_16x16_random():
    """16x16 FP32 matmul vs numpy reference."""
    result, ref = _run_matmul(16, 16, 16)
    np.testing.assert_allclose(result, ref, atol=1e-4, rtol=1e-4)


def test_tile_matmul_16x16_k32():
    """K=32 (two MFMA iterations of K=4)."""
    result, ref = _run_matmul(16, 16, 32)
    np.testing.assert_allclose(result, ref, atol=1e-4, rtol=1e-4)


def test_tile_matmul_alpha_beta():
    """C = 2.0*A@B + 0.5*C_in (non-trivial alpha/beta)."""
    result, ref = _run_matmul(16, 16, 16, alpha=2.0, beta=0.5)
    np.testing.assert_allclose(result, ref, atol=1e-4, rtol=1e-4)


def test_tile_matmul_beta_one():
    """C = A@B + C_in (beta=1.0, accumulate)."""
    result, ref = _run_matmul(16, 16, 16, alpha=1.0, beta=1.0)
    np.testing.assert_allclose(result, ref, atol=1e-4, rtol=1e-4)


def test_tile_matmul_accumulate_false_ignores_beta():
    """Accumulate=false: C must equal alpha*A@B regardless of initial C content.
    Bug fixed: Accumulate template param was ignored in rocWMMA path."""
    device = HIP_DEVICE
    A_np = np.random.randn(16, 16).astype(np.float32)
    B_np = np.random.randn(16, 16).astype(np.float32)
    C_np = np.ones((16, 16), dtype=np.float32) * 999.0  # large to detect leakage
    expected = A_np @ B_np  # overwrite mode: ignore C_np

    A_wp = wp.array(A_np, dtype=float, device=device)
    B_wp = wp.array(B_np, dtype=float, device=device)
    C_wp = wp.array(C_np, dtype=float, device=device)
    C_out_wp = wp.zeros((16, 16), dtype=float, device=device)

    # tile_matmul_16x16_kernel uses Accumulate=false (overwrite) path
    wp.launch_tiled(tile_matmul_16x16_kernel, dim=(1,),
                   inputs=[A_wp, B_wp, C_wp], outputs=[C_out_wp],
                   block_dim=64, device=device)

    np.testing.assert_allclose(C_out_wp.numpy(), expected, atol=1e-4, rtol=1e-4,
        err_msg="Accumulate=false: initial C content (999) leaked into result")


def test_tile_matmul_alpha2_beta1():
    """C = 2.0*A@B + 1.0*C_in — catches alpha-applied-to-wrong-operand bug.
    Bug fixed: original code computed alpha*(A@B + beta*C) instead of alpha*A@B + beta*C."""
    result, ref = _run_matmul(16, 16, 16, alpha=2.0, beta=1.0)
    np.testing.assert_allclose(result, ref, atol=1e-4, rtol=1e-4,
        err_msg="alpha=2, beta=1: bug gives 2*(A@B+C) instead of 2*A@B+C")


def test_tile_matmul_alpha0_beta1():
    """C = 0*A@B + 1.0*C_in = C_in exactly.
    Bug fixed: original code gave 0*(A@B+C_in)=0 instead of C_in."""
    device = HIP_DEVICE
    A_np = np.random.randn(16, 16).astype(np.float32)
    B_np = np.random.randn(16, 16).astype(np.float32)
    C_np = np.random.randn(16, 16).astype(np.float32)
    expected = C_np.copy()  # 0*A@B + 1*C = C

    A_wp = wp.array(A_np, dtype=float, device=device)
    B_wp = wp.array(B_np, dtype=float, device=device)
    C_in_wp = wp.array(C_np, dtype=float, device=device)
    C_out_wp = wp.zeros((16, 16), dtype=float, device=device)

    wp.launch_tiled(tile_matmul_alpha_beta_kernel, dim=(1,),
                   inputs=[A_wp, B_wp, C_in_wp, float(0.0), float(1.0)],
                   outputs=[C_out_wp],
                   block_dim=64, device=device)

    np.testing.assert_allclose(C_out_wp.numpy(), expected, atol=1e-5,
        err_msg="alpha=0, beta=1: result should be C_in, not 0")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
