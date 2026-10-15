# Cached frozen metadata projector

`prepare_cached.py` runs the exact frozen server51_campaign_port_v1 projector (manifest5107eb…) and substitutes only the stdlib metadata `pin` function. Each immutable file is byte-hashed once; reuse requires identical device/inode/size/mtime_ns/ctime_ns. Before/after hashing and final snapshot checks reject mutations. Every returned receipt retains the exact original path/bytes/SHA shape. No source, scientific setting, admission, descriptor binding, or no-overwrite guard is changed.

The invocation requires `--wrapper-sha256 SHA --cache-receipt FRESH_JSON`, followed by all original projector arguments. The cache receipt must stay inside this wrapper directory. Its actual outer producer and unchanged inner projector are recorded separately. Run a fresh output attempt; existing partial output remains preserved. No resume or overwrite semantics are added.

One stdlib source test passes: unchanged receipts, defensive copies, mutation invalidation and final snapshot rejection. No remote/native execution was performed by this agent.
