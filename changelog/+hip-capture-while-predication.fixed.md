Fix `wp.capture_while()` producing wrong results under graph capture on HIP devices. ROCm has no
conditional graph node, so a while-region is lowered by cloning the loop body into the parent graph
a fixed number of times. Nothing predicated those copies, so every copy ran: a loop whose condition
was already false on entry still executed its body 32 times, and the result was silently wrong
rather than merely slow. Warp-generated kernels now take a hidden trailing condition pointer and
return immediately when the region's condition reads false, so the loop stops at the right iteration
and a false condition runs zero iterations. Generated CUDA source is unchanged. A body passed as an
already-captured `Graph` cannot be bound to the condition and now raises `NotImplementedError` on
HIP instead of running unpredicated; pass a callable body instead. Non-kernel nodes in a body
(`memcpy`, `memset`, nested child graphs) are still cloned verbatim, and Warp now warns when it
finds one.
