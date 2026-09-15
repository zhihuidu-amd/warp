Fix tile kernels failing to compile on HIP devices with `CUDA kernel build failed with
error code 6`. Most tile builtins were affected, including `wp.tile_arange()`,
`wp.tile_sum()`, `wp.tile_min()`, `wp.tile_max()`, `wp.tile_argmin()`, `wp.tile_argmax()`,
`wp.tile_extract()` and `wp.tile_cholesky_solve()`. The underlying HIPRTC diagnostic was

```text
error: no matching function for call to 'tile_argmin'
note: candidate function not viable: call to __host__ function from __global__ function
```

Warp passes `--device-as-default-execution-space` on every runtime compile, which makes
NVRTC treat an unannotated function as `__device__`. HIPRTC has no equivalent spelling, so
Warp's HIP backend drops the option and clang applies its own default of `__host__`. The
thin unannotated wrappers in the tile headers then become uncallable from a `__global__`
function. The tile headers now restore the option's semantics with the equivalent clang
pragma, scoped to their `namespace wp` bodies and gated on HIPRTC, so CUDA, `nvcc` and the
CPU build are unaffected.
