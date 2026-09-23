Fix `wp.tile_sum()` and `wp.tile_reduce()` along an axis of 33 to 256 elements returning
silently incorrect results when the kernel is launched with a `block_dim` smaller than the
warp size. `tile_reduce_axis_impl()` indexed the reduction axis by `lane_index` but strided
it by `WP_TILE_WARP_SIZE`, so the lanes above `block_dim`, which never run, owned axis
elements that were then never read: a 200-element axis reduced to 56 at `block_dim=8` and
104 at `block_dim=16` on a 32-lane warp, with no error or warning. The axis is now walked in
chunks of `min(WP_TILE_BLOCK_DIM, WP_TILE_WARP_SIZE)`. Launches at or above the warp size are
unaffected, since the two values are equal there and the generated code is identical. The
shorter and longer axis tiers already strided by `WP_TILE_BLOCK_DIM` and were never affected.
