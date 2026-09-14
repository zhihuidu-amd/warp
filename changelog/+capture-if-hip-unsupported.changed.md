Raise a descriptive error when `wp.capture_if()` is called inside a graph capture on a HIP device,
instead of failing deep in the conditional-node path with a CUDA-specific message. ROCm has no
conditional graph node, and the HIP fallback lowers a conditional region by unrolling its body into
the parent graph unconditionally, which for an if/else would execute the branch the condition
selected against. Outside of a graph capture, `wp.capture_if()` is unaffected: the condition is read
on the host and only the selected branch runs.

`wp.is_conditional_graph_supported()` now returns `True` on HIP devices. It previously compared the
ROCm version against a CUDA toolkit number and so reported `False` on every HIP build, even though
`wp.capture_while()` works there; code gating on it took the uncaptured slow path on AMD GPUs. The
HIP lowering of `wp.capture_while()` is also now documented in the user guide, including which
parts of a loop body the lowering cannot predicate.
