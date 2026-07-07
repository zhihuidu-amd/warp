# Copyright (c) 2026 NVIDIA CORPORATION.  All rights reserved.
# AMD rocWMMA tile_matmul unit test
# File: warp/tests/cuda/test_tile_rocwmma.py

"""
Unit tests for AMD rocWMMA tile_matmul fast path.
Tests activate only when WP_AMD_ROCWMMA_AVAILABLE is defined
(AOT-compiled warp.so on AMD gfx942+ with ROCm 7.x+).
"""

import unittest
import numpy as np
import warp as wp
from warp.tests.unittest_utils import *


def get_test_devices():
    """Return HIP devices only."""
    return [d for d in wp.get_cuda_devices() if d.is_hip]


@wp.kernel
def tile_matmul_16x16_kernel(
    A: wp.array2d(dtype=float),
    B: wp.array2d(dtype=float),
    C: wp.array2d(dtype=float),
):
    """16x16 tile matmul: C = A @ B using wp.tile_matmul."""
    i, j = wp.tid()
    a_tile = wp.tile_load(A, shape=(16, 16), storage="shared")
    b_tile = wp.tile_load(B, shape=(16, 16), storage="shared")
    c_tile = wp.tile_zeros(shape=(16, 16), dtype=float, storage="shared")
    wp.tile_matmul(a_tile, b_tile, c_tile)
    wp.tile_store(C, c_tile)


@wp.kernel
def tile_matmul_16x16_k32_kernel(
    A: wp.array2d(dtype=float),  # [16, 32]
    B: wp.array2d(dtype=float),  # [32, 16]
    C: wp.array2d(dtype=float),  # [16, 16]
):
    """16x16 output tile with K=32 (exercises multiple MFMA iterations)."""
    i, j = wp.tid()
    a_tile = wp.tile_load(A, shape=(16, 32), storage="shared")
    b_tile = wp.tile_load(B, shape=(32, 16), storage="shared")
    c_tile = wp.tile_zeros(shape=(16, 16), dtype=float, storage="shared")
    wp.tile_matmul(a_tile, b_tile, c_tile)
    wp.tile_store(C, c_tile)


@wp.kernel
def tile_matmul_alpha_beta_kernel(
    A: wp.array2d(dtype=float),
    B: wp.array2d(dtype=float),
    C_in: wp.array2d(dtype=float),
    C_out: wp.array2d(dtype=float),
    alpha: float,
    beta: float,
):
    """Tests alpha/beta scaling: C_out = alpha * A @ B + beta * C_in."""
    i, j = wp.tid()
    a_tile = wp.tile_load(A, shape=(16, 16), storage="shared")
    b_tile = wp.tile_load(B, shape=(16, 16), storage="shared")
    c_tile = wp.tile_load(C_in, shape=(16, 16), storage="shared")
    wp.tile_matmul(a_tile, b_tile, c_tile, alpha=alpha, beta=beta)
    wp.tile_store(C_out, c_tile)


class TestTileMatmulRocWMMA(unittest.TestCase):

    def _run_matmul_test(self, M, N, K, device, alpha=1.0, beta=0.0,
                          atol=1e-4, rtol=1e-4):
        """Run tile_matmul and compare against numpy reference."""
        rng = np.random.default_rng(42)
        A_np = rng.standard_normal((M, K)).astype(np.float32)
        B_np = rng.standard_normal((K, N)).astype(np.float32)
        C_np = rng.standard_normal((M, N)).astype(np.float32) if beta != 0.0 \
               else np.zeros((M, N), dtype=np.float32)

        ref = alpha * (A_np @ B_np) + beta * C_np

        A_wp = wp.array(A_np, dtype=float, device=device)
        B_wp = wp.array(B_np, dtype=float, device=device)
        C_in_wp = wp.array(C_np, dtype=float, device=device)
        C_out_wp = wp.zeros((M, N), dtype=float, device=device)

        if beta == 0.0 and alpha == 1.0 and K == 16:
            wp.launch_tiled(tile_matmul_16x16_kernel,
                           dim=(1,), inputs=[A_wp, B_wp], outputs=[C_out_wp],
                           block_dim=64, device=device)
        elif beta == 0.0 and alpha == 1.0 and K == 32:
            wp.launch_tiled(tile_matmul_16x16_k32_kernel,
                           dim=(1,), inputs=[A_wp, B_wp], outputs=[C_out_wp],
                           block_dim=64, device=device)
        else:
            wp.launch_tiled(tile_matmul_alpha_beta_kernel,
                           dim=(1,),
                           inputs=[A_wp, B_wp, C_in_wp, alpha, beta],
                           outputs=[C_out_wp],
                           block_dim=64, device=device)

        result = C_out_wp.numpy()
        np.testing.assert_allclose(result, ref, atol=atol, rtol=rtol,
            err_msg=f"tile_matmul({M}x{K}x{N}) failed on {device}")

    @parameterize_with_devices(get_test_devices())
    def test_tile_matmul_16x16_identity(self, device):
        """16x16 matmul with identity B → result equals A."""
        A_np = np.random.randn(16, 16).astype(np.float32)
        B_np = np.eye(16, dtype=np.float32)
        A_wp = wp.array(A_np, dtype=float, device=device)
        B_wp = wp.array(B_np, dtype=float, device=device)
        C_wp = wp.zeros((16, 16), dtype=float, device=device)
        wp.launch_tiled(tile_matmul_16x16_kernel, dim=(1,),
                       inputs=[A_wp, B_wp], outputs=[C_wp],
                       block_dim=64, device=device)
        np.testing.assert_allclose(C_wp.numpy(), A_np, atol=1e-4,
            err_msg="16x16 identity matmul failed")

    @parameterize_with_devices(get_test_devices())
    def test_tile_matmul_16x16_zeros(self, device):
        """16x16 matmul with zero B → result is zero."""
        A_wp = wp.ones((16, 16), dtype=float, device=device)
        B_wp = wp.zeros((16, 16), dtype=float, device=device)
        C_wp = wp.ones((16, 16), dtype=float, device=device)
        wp.launch_tiled(tile_matmul_16x16_kernel, dim=(1,),
                       inputs=[A_wp, B_wp], outputs=[C_wp],
                       block_dim=64, device=device)
        np.testing.assert_allclose(C_wp.numpy(),
                                   np.zeros((16, 16), dtype=np.float32),
                                   atol=1e-6, err_msg="Zero matmul failed")

    @parameterize_with_devices(get_test_devices())
    def test_tile_matmul_16x16_random(self, device):
        """16x16 FP32 matmul vs numpy reference."""
        self._run_matmul_test(16, 16, 16, device)

    @parameterize_with_devices(get_test_devices())
    def test_tile_matmul_16x16_k32(self, device):
        """16x16 output, K=32 (two MFMA iterations of K=4)."""
        self._run_matmul_test(16, 16, 32, device)

    @parameterize_with_devices(get_test_devices())
    def test_tile_matmul_alpha_beta(self, device):
        """C = 2.0 * A @ B + 0.5 * C_in (non-trivial alpha/beta)."""
        self._run_matmul_test(16, 16, 16, device, alpha=2.0, beta=0.5)

    @parameterize_with_devices(get_test_devices())
    def test_tile_matmul_beta_one(self, device):
        """C = A @ B + C_in (beta=1.0, accumulate)."""
        self._run_matmul_test(16, 16, 16, device, alpha=1.0, beta=1.0)



    @parameterize_with_devices(get_test_devices())
    def test_tile_matmul_accumulate_false_ignores_beta(self, device):
        """Accumulate=false (overwrite mode): C must equal alpha*A@B, regardless of beta or initial C.
        
        This tests the Accumulate template parameter is respected in the rocWMMA path.
        Bug: original code used T(beta) unconditionally; Accumulate=false should force beta=0.
        When tile_matmul is called in non-accumulate mode, initial C content must be ignored.
        """
        A_np = np.random.randn(16, 16).astype(np.float32)
        B_np = np.random.randn(16, 16).astype(np.float32)
        # Initialize C with large non-zero values that would corrupt output if beta!=0 leaked
        C_np = np.ones((16, 16), dtype=np.float32) * 999.0

        expected = A_np @ B_np  # alpha=1, beta must be 0 (overwrite mode)

        A_wp = wp.array(A_np, dtype=float, device=device)
        B_wp = wp.array(B_np, dtype=float, device=device)
        C_wp = wp.array(C_np, dtype=float, device=device)

        # tile_matmul_16x16_kernel uses tile_matmul without accumulate (Accumulate=false path)
        # Result should be A@B, completely ignoring initial C_wp contents
        wp.launch_tiled(tile_matmul_16x16_kernel,
                       dim=(1,), inputs=[A_wp, B_wp], outputs=[C_wp],
                       block_dim=64, device=device)

        result = C_wp.numpy()
        np.testing.assert_allclose(result, expected, atol=1e-4, rtol=1e-4,
            err_msg="Accumulate=false ignored beta: initial C content leaked into result")


    @parameterize_with_devices(get_test_devices())
    def test_tile_matmul_alpha2_beta1(self, device):
        """C = 2.0*A@B + 1.0*C_in — catches alpha-applied-to-wrong-operand bug.
        
        Bug: original code computed alpha*(A@B + beta*C_in) instead of alpha*A@B + beta*C_in.
        With alpha=2, beta=1: buggy gives 2*(A@B + C_in), correct gives 2*A@B + C_in.
        """
        self._run_matmul_test(16, 16, 16, device, alpha=2.0, beta=1.0)

    @parameterize_with_devices(get_test_devices())
    def test_tile_matmul_alpha0_beta1(self, device):
        """C = 0*A@B + 1.0*C_in = C_in — alpha=0 with beta=1 must preserve C_in exactly.
        
        Bug: original code computed alpha*(A@B + beta*C_in) = 0*(A@B + C_in) = 0.
        Correct: 0*A@B + 1*C_in = C_in.
        """
        A_np = np.random.randn(16, 16).astype(np.float32)
        B_np = np.random.randn(16, 16).astype(np.float32)
        C_np = np.random.randn(16, 16).astype(np.float32)

        expected = C_np.copy()  # 0*A@B + 1*C_in = C_in

        A_wp = wp.array(A_np, dtype=float, device=device)
        B_wp = wp.array(B_np, dtype=float, device=device)
        C_in_wp = wp.array(C_np, dtype=float, device=device)
        C_out_wp = wp.zeros((16, 16), dtype=float, device=device)

        wp.launch_tiled(tile_matmul_alpha_beta_kernel,
                       dim=(1,),
                       inputs=[A_wp, B_wp, C_in_wp, float(0.0), float(1.0)],
                       outputs=[C_out_wp],
                       block_dim=64, device=device)

        np.testing.assert_allclose(C_out_wp.numpy(), expected, atol=1e-5,
            err_msg="alpha=0, beta=1: result should be C_in, not 0")

if __name__ == "__main__":
    wp.init()
    unittest.main(verbosity=2)
