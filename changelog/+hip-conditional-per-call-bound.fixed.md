Add a `max_iters` argument to `wp.capture_while()`, so a loop's unroll bound on HIP comes from the
caller that knows its iteration budget rather than from a process-global default. `warp.optim.linear`
and `warp.geometry`'s Delaunay edge-flip pass now supply theirs.

ROCm has no conditional graph node, so Warp lowers a conditional region to a static unroll of
predicated body copies, and that unroll count is a hard cap. Until now the only channel for it was
`warp.config.hip_conditional_max_iters`, a single number for the whole process -- and no single number
fits every loop. At the default of 32, `wp.optim.linear.cg()`, `cr()` and `bicgstab()` with
`maxiter=1000` stopped after 32 iterations and returned an unconverged result; on a 64x64 Laplacian
the residual was `5.1e+03` against a tolerance of `2.2e-07`, where the same solve run eagerly
converged in 110 iterations. Raising the global bound instead is not a fix: it makes every short loop
pay a dispatch per unused copy, and a GMRES restart cycle wanting two iterations was being handed
thousands of clones of its body.

`max_iters` is the largest number of times the body can be required to run. Callers that pass nothing
keep the previous behavior and now get a one-time warning naming the default that was used. The
argument is ignored on CUDA, where a conditional node re-evaluates until the condition goes false and
has no bound, and ignored outside of graph capture, where the condition is read back to the host and
the loop runs the right number of times regardless -- so a caller passing its own budget is portable.
