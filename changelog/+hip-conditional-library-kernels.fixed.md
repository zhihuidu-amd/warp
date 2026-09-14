Warn when a `warp.utils` operation implemented with library kernels (`radix_sort_pairs`,
`segmented_sort_pairs`, `array_scan`, `runlength_encode`, `array_sum`, `array_inner`) is called
inside a `wp.capture_while()` body on a HIP device, which was previously silent.

Each of these allocates a temporary buffer for its hipCUB/rocPRIM dispatch, and a conditional body
graph cannot allocate, so on HIP none of them run inside a region. The two sort entry points log
`CUDA error 900: operation not permitted when stream is capturing` and return normally with the
destination unchanged — nothing raises, so the failure is silent. The scan, reduction and
run-length entry points emit a graph allocation node, which makes the enclosing region raise
`Conditional body graph contains an unsupported operation (memory allocation)` when it closes.

Separately, catch unpredicated kernels in general. ROCm has no conditional graph node, so the body
is lowered as a predicated static unroll and each copy skips itself by testing the region's guard
word — but only Warp-generated kernels have the guard prologue. A native kernel that does *not*
allocate gets into the body graph, is cloned like everything else, and runs on every copy
regardless of the condition; an `array.fill_()` in a body does exactly this. The existing node-type
check cannot detect it: those *are* kernel nodes, and nothing at the graph level tells them apart
from Warp's own. Catch it by counting rather than by name: every kernel Warp generates is
dispatched through a path that binds the condition pointer, so a body graph holding more kernel
nodes than the region bound conditions contains at least that many unpredicated kernels from
elsewhere. Warp now reports that difference once per process, which covers any native library or
third-party module and not only the `warp.utils` entry points above. The count is skipped for a
nested conditional region, whose unrolled copies are spliced into the outer body graph and would
make the comparison meaningless.

The user guide's description of the HIP lowering previously said re-running such kernels was
"wasted dispatch rather than a wrong answer", which is now corrected.

Gate two `test_utils.py` graph tests on the feature each one uses rather than on
`wp.is_conditional_graph_supported()`. That function is a version probe — on CUDA it implies 12.4+,
which also brings capture pause/resume and `wp.capture_if()` — and it now returns `True` on HIP,
where neither of those holds. `test_radix_sort_pairs_capture_if_forked_stream` was reporting Warp's
own deliberate HIP refusal as a test error.
