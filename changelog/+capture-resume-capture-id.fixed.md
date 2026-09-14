Fix operations issued after `wp.capture_resume()` failing to find their capture, which on HIP made
`warp.utils` library operations such as `radix_sort_pairs()` return without doing anything and leave
their destination array holding its input.

`wp_cuda_graph_resume_capture()` calls `cudaStreamBeginCaptureToGraph()`, which is a fresh
`BeginCapture`. CUDA hands back the same capture id, so the entry registered at begin time still
resolved; HIP mints a new one (measured on gfx942: `1` before the pause, `2` after the resume). Any
code asking `find_capture_info()` about the current capture therefore missed it. `acquire_temp_buffer()`
in `warp/native/sort.cu` reads that miss as "a child conditional body graph is being captured" and
falls back to allocating on a non-capturing side stream under
`cudaThreadExchangeStreamCaptureMode(Relaxed)` — CUDA's escape hatch for allocating during a capture,
which ROCm does not honor. HIP refused the allocation with error 900, `acquire_temp_buffer()` returned
false, and the sort returned unsorted, reporting only on stderr.

`CaptureInfo` now carries `current_id` alongside the canonical `id`. `id` still tags graph allocations
and identifies the capture across a pause; `current_id` is the registry key and the "is this capture
live on its own stream" comparison, and `wp_cuda_graph_resume_capture()` re-keys the registry when it
changes.

The re-key is gated on the resumed graph being the capture's own top-level graph, recorded at begin
time. `wp.capture_if()` and `wp.capture_while()` reach the same function to *enter* a conditional body
graph rather than to resume the parent, and re-keying there would register the parent capture under
the body's id — defeating the child-body detection that `acquire_temp_buffer()`, `wp_alloc_device_async()`
and `wp_free_device_async()` rely on to keep graph allocations out of a conditional body. A body graph
is never the top-level graph at any nesting depth.

On CUDA the two ids never diverge and the gate always holds, so this is a no-op there.
