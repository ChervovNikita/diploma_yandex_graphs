# Fixed native branch diagnosis — disabled source

This packet prepares one bounded engineering diagnosis of the existing synthetic native directional derivative mismatch. Preparation used only static source/AST/JSON/hash inspection. No prepared source was imported or executed, and no SSH, dataset access, fit, or additional agent was used.

`diagnose.py` returns `DISABLED` before numerical imports or output creation unless `--execute-authorized` is supplied. Parent owns independent review and any later run. The runtime and synthetic state are fixed in `PLAN.json`: the same 15 nodes, native width 512, 10 local plus 1 global layers, four members, seeds, direction and unchanged operator/qualifier. The only scalar points are base and both signs of 1e-5, 1e-6 and 1e-7, under a fixed 180-second alarm.

Temporary process-local functional ReLU/LeakyReLU wrappers return the original functions' results with unchanged arguments. They capture ordinary before/probe-adapted/query callbacks and record explicit skips for transformed or unsupported tensors. Restoration runs through `finally`, including partial installation and errors. Exact sign/zero masks, counts, source/module/layer identities and all changed coordinates are retained; no branch cutoff is used. Masks are saved only after all seven scalar points complete; the result retains completed points incrementally.

The one required fresh base shared gradient supplies the analytic projection and full shared gradient norm. The existing live-Q chain gradient norm is reused from the hash-bound prior same-state root result; it is not recomputed. Planned work is eight outer evaluations including that gradient: 160 member callbacks, 64 private gradient constructions, one shared gradient, zero fits and zero persistent updates.

Right, left and central finite differences are reported without a new tolerance or pass decision. Observed branch transitions can identify boundaries, but alone cannot establish the cause of the unstable left difference. The diagnostic does not qualify the native worker or provide predictive evidence. All predecessor sources and their false release guards are preserved.

`SOURCE_BINDINGS.json` binds the immutable qualifier, native/operator/port/boundary sources and authorized prior diagnostic source/result. `STATIC_CHECK.json` records static checks only. `MANIFEST.json` and `SEAL.json` bind this disabled packet.
