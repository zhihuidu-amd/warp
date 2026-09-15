Bound `wp.capture_while()` iterations explicitly on HIP, and report the loops that exceed the bound.
ROCm has no conditional graph node, so Warp lowers a conditional region to a static unroll of
predicated body copies. That unroll has an iteration bound a real conditional node does not, and the
bound was fixed at 32 and unreachable from Python: a loop needing a 33rd iteration simply stopped,
returning an under-iterated result with nothing set to say so. A solver asking for a 100-iteration
budget -- the MuJoCo default -- silently got 32.

The bound is now `warp.config.hip_conditional_max_iters` (environment variable
`WARP_HIP_CONDITIONAL_MAX_ITERS`, still 32 by default), and callers that know their iteration budget
should raise it to at least that budget. Raising it costs graph size and one dispatch per added copy,
not GPU work: copies past convergence are predicated off and their kernels return immediately.

Every truncation is now counted on device and printed to `stderr` once per device per process, so a
bound that is too low is loud rather than silent. `wp.conditional_graph_truncations()` returns the
count, for asserting its absence in a test or benchmark. Both are no-ops on CUDA, where a conditional
node re-evaluates the condition until it goes false and has no bound to exceed; the query returns `0`
there rather than raising, so a portable test needs no skip.
