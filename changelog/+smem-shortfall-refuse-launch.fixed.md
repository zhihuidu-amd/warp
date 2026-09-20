Refuse a kernel launch whose dynamic shared memory could not be configured
instead of dispatching it. Such a launch can only fail, so the error message is
unchanged; on HIP, where an over-budget request aborts the queue asynchronously
rather than returning an error, this is what makes the message reachable at all.
