Fix a false truncation report from `wp.capture_while()` on HIP when a loop uses exactly its budget.
The unroll emits a condition-refresh kernel between body copies, and the truncation check reads the
value that refresh writes. No refresh was emitted after the *final* copy, on the reasoning that no
body copy was left to consume it -- but the check consumes it too, so the check saw the condition as
of before the last copy rather than after it. A loop that converged on its final copy therefore
returned the correct answer and was still counted and printed as truncated; with `max_iters=1` no
refresh had ever run and the check read `hipGraphCondBegin`'s seed.

Measured on gfx942: a 7-iteration loop answered 7 at `max_iters=7` and reported one truncation, while
`max_iters=6` (genuinely truncated, answer 6) and `max_iters=8` (answer 7, silent) both behaved
correctly. Only the diagnostic was wrong -- no result was ever under-iterated by this -- but a
diagnostic that fires on correct code is worse than none, and it made the truncation counter unusable
as a test assertion for callers that size their budget exactly.

The final copy now gets the same refresh as every other, gated on the same condition as the check
itself so the if-region path emits nothing new. It ANDs with the enclosing guard exactly as the
between-copy refresh does, which keeps a suppressed nested region from reporting.
