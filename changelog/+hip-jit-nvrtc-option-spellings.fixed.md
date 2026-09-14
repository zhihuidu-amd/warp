Fix kernel compilation failing on HIP devices whenever a module requests fast math, or
`wp.config.lineinfo` / `wp.config.mode = "debug"` is set. Each failed with
`CUDA kernel build failed with error code 6` and compiled no kernels at all.

Warp builds its runtime-compilation option list in NVRTC spellings. Warp's HIP backend
translates that list for HIPRTC, which rejects an unrecognized option by failing the entire
compile rather than ignoring it — so a single untranslated option takes down every kernel in
the module. Three were still untranslated:

| NVRTC | HIPRTC | Reached by |
| --- | --- | --- |
| `--use_fast_math` | `-ffast-math` | `wp.set_module_options({"fast_math": True})` |
| `--generate-line-info` | `-gline-tables-only` | `wp.config.lineinfo = True` |
| `--device-debug` | `-g` | `wp.config.mode = "debug"` |

`warp/examples/benchmarks/benchmark_bvh.py` and `benchmark_mesh.py` set `fast_math` at module
scope and so could not run on HIP.

Option order is preserved, so `--fmad=false` after `--use_fast_math` still disables FMA
contraction, matching NVRTC.
