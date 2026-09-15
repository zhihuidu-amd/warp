Support `wp.capture_if()` inside a graph capture on HIP devices. ROCm has no conditional graph node,
so each branch is spliced into the parent graph and predicated on a guard word that Warp-generated
kernels read. `hipGraphCondBegin()` opens only one region per stream, so Warp hands back two empty
graphs up front, captures each branch body into one of them, and splices both sequentially once both
are complete. The else branch is guarded on a device-side complement of the condition that is
computed before either branch runs, so an if body that writes the condition cannot change what the
else branch sees.

Three cases cannot be predicated and now raise on HIP, rather than executing the branch the condition
selected against: a branch passed as an already-captured `Graph` instead of a callable, a branch body
containing a `memcpy`, `memset` or child-graph node, and a branch body containing kernels Warp did not
generate (a native library call or a raw driver launch). The equivalent cases only warn for
`wp.capture_while()`, where they cost extra iterations instead of a wrong answer. Replaying a recorded
`wp.capture_if()` through APIC is refused on HIP for the same reason -- APIC records only a kernel's
declared Warp arguments, leaving no slot to bind the guard into -- but recording is unaffected, since
the live capture is correct.

Nested conditional regions compose. A nested region's seed kernels are cloned into the parent along
with the enclosing body and carry no guard of their own, so without this they would re-arm the inner
guard inside an enclosing branch the outer condition suppressed. Each seed now sets its slot to
`(*condition != 0) && (*enclosing != 0)`, and since every enclosing guard is already the conjunction
of its own chain, this holds to any depth for either conditional form nested in the other. The
condition-slot pool is also keyed by device, so a capture on a second GPU no longer hands its kernels
an interior pointer into the first GPU's slab.

`wp.is_conditional_graph_supported()` now returns `True` on HIP devices. It previously compared the
ROCm version against a CUDA toolkit number and so reported `False` on every HIP build, even though
`wp.capture_while()` works there; code gating on it took the uncaptured slow path on AMD GPUs. The
HIP lowering of both conditional forms is documented in the user guide, including which parts of a
body the lowering cannot predicate.
