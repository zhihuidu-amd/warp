Fix `wp.tile_sum()`, `wp.tile_min()`, `wp.tile_max()`, `wp.tile_argmin()` and
`wp.tile_argmax()` returning silently incorrect results on HIP devices whenever the warp
is not fully active. `warp_reduce()` and `warp_reduce_tracked()` took the active-lane mask
as `unsigned int` while every caller passes the 64-bit `wp_tile_warp_mask_t`, so the
partial-mask lane test `mask & (1 << lane)` shifted an `int` by up to 63. Both now use
`wp_tile_warp_mask_t`, matching the 13 `warp_shuffle_down()` overloads and the six call
sites that already did. CUDA is unaffected: `wp_tile_warp_mask_t` is `unsigned int` there,
so the generated code is identical.
