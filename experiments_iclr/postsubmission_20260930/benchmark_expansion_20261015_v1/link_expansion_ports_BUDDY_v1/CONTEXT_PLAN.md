# BUDDY native context admission

The retained `buddy_newdata_root_20261012_v1/cache/manifest.json` describes
year0, all-TRAIN topology and a fixed negative seed20261002. Its fixed-family
configuration also changes batch ordering. These materialized inputs are not
the native source year2007 context required by the current frozen BUDDY recipe.
They must not be injected into the native source session for qualification.

Official GRAPH/TRAIN/VALID provider custody receipts exist. No TEST split,
sealed endpoint queries, `query.pt` or final targets are needed or opened.
`prepare_context_cpu.py` will construct the unchanged native BUDDY source
session with the exact smoke recipe, seed0 and CPU device. It will preserve
year2007 filtering, weighted graph coalescing, SIGN-k0 features, full native
hash tables, degrees, negative generation and1024-row sampler.

The preparation materializes two original TRAIN batches and all60,084 positive
and100,000 negative VALID inputs through the source hydration function, with
exact endpoint/feature/hash source receipts. It constructs no predictions,
computes no score and performs zero optimizer transitions. Checkpoint binaries
stay on the allocation; local evidence is compact JSON only.

Materialization must wait for the sequential NCNC CPU checks to finish. Its
own guard/resource preflight must inspect current cgroup anon/file/cache/events,
with two CPU threads and a64GiB address-space limit. No cache clearing or other
job interruption is permitted. A failed context preparation is preserved and
is not retried automatically. BUDDY qualification may start only after a
successful materialized context receipt and separate current resource preflight.

BUDDY C6 and responsibility training must use its original joint BCE batch mean.
No NCN positive-mean plus negative-mean reduction is transferable here. Existing
portable input v2 already maps both structural and endpoint-feature branches;
that source remains unchanged. No BUDDY port/readiness is claimed by this plan.
