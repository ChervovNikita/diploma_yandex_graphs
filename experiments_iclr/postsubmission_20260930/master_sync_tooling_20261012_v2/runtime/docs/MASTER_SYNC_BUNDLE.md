# Plan-bound master synchronization

Status: tooling only until root freezes the consolidated finite master.
Preparation and checks run on the authorized allocation. No server77/MacLink
access, scientific fit, CUDA pilot, GO, credentials or running-result access
is performed by this workflow. Original five paper datasets are hard excluded.

## Artifact contract

`prepare_sync_bundle.py prepare` constructs the deduplicated union of exact
path/bytes/SHA256 receipts from the master PLAN, every master spec, source and
data bindings, runtime entrypoints, explicit role inputs, author raw/source
files, frozen synchronization tooling/environment record and phase AGENTS.md.
It verifies opaque bytes only; no target, graph or tensor is deserialized.
Historical absolute provider receipts inside the allocation phase are mirrored
at their original phase-relative path, with explicit aliases recorded and no
spec rewrite. Absolute provenance outside this phase is rejected without access.
Execution/source/config file bindings still must be relative; the allocation
author's declared absolute environment remains the only execution exception.
Conflicting receipts, traversal, symlinks, credentials, Git history, scientific
execution directories, quarantine and checkpoints fail before a bundle is
built. No whole-repository recursion or unbound file is copied. Initial author
basis/processed directories are preserved as declared empty directories.

PLAN/spec identities and phase-relative artifact paths are unchanged. A fresh
immutable output contains MANIFEST.json and SUMMARY.json. Optional ARTIFACTS.tar
contains only union members plus required empty directories; every tar member
is rehashed, and extra/link/duplicate/missing members are rejected. The archive
may remain server-side. Mirror the compact manifest/summary and statistics.
MANIFEST/SUMMARY travel as sidecars; pin the canonical manifest hash from the
allocation proof before installation/preflight. `install` reuses byte-identical
destination files and rejects conflicts before any write, then installs new
files atomically without clobbering and rechecks all hashes.

## Git and host portability

Overlay the bundle onto an authorized existing or later cloned Git checkout.
No .git content or private URL/credential is bundled. The phase sidecar needs
the existing configured remote whose URL SHA256 matches the frozen plan. Git
checkout and remote are preflighted without fetch/push. The destination phase
may have a different absolute root while retaining every internal relative
artifact path and its AGENTS marker. The ordinary `start_massive.sh` works
inside this checkout and remains the sole startup path; no controller edit or
alternative launcher is introduced.

The launcher validates source bindings of **all** studies on **all** hosts.
Thus server77 eventually needs the declared author source/raw/private Python
dependency files for hash validation even though author groups execute only
on allocation. Their absolute Python/PYTHONPATH/LD_LIBRARY_PATH bindings are
recorded, never rewritten or executed on server77. Environment executables,
site-packages binaries, CUDA libraries and the whole virtualenv are not bundled.

Server77 stays unqualified and unauthorized during this preparation. After
later user reauthorization, root must freeze its real hostname/GPU UUIDs in
the actual plan and qualify the environment before scientific GO. A pending
binding is rejected. Preflight checks the literal physical route before
reading project artifacts; server77 requires explicit expected hostname/UUID
arguments, which must then match the frozen plan. Allocation always requires
anogena-2-0 and sole UUID GPU-44039938-fd82-41d2-fefd-de71514e2fac.

## Exact environment and command sequence

The allocation metadata record pins CPython version, Linux/machine/libc,
SQLite/OpenSSL stdlib versions, Torch/PyG/scatter/sparse compiled build versions,
NumPy/SciPy/scikit-learn/OGB distributions and required import modules. Matching
version strings alone do not prove binary usability. Preflight can import the
native stack in a child with CUDA_VISIBLE_DEVICES empty and asserts CUDA was
not initialized. GPU execution/resource qualification remains separate; no
environment replication or numerical/score equivalence is claimed for server77.
Author runtime requirements are separately retained from its exact contract.

Allocation tooling commands, with actual root-owned paths supplied:

```sh
python -B prepare_sync_bundle.py environment --output ENVIRONMENT.json
python -B prepare_sync_bundle.py prepare --project-root PHASE --plan MASTER/PLAN.json --environment TOOLING/ENVIRONMENT.json --output FRESH_SYNC --tar
```

After later authorization and transport, use `install` with the pinned manifest
hash and physical host arguments, then `preflight --binary-import-check` with
the same arguments. Installation does not start workers. Preflight does not
register jobs, write phase proofs or create approval. Once root has completed
environment/hash/hardware/resource checks and the user has issued explicit GO,
the user has one startup command per host:

```sh
GNNM_PYTHON=/existing/repository/.venv/bin/python bash FROZEN_RUNTIME/start_massive.sh HOST MASTER/PLAN.json EXISTING_APPROVAL.json
```

This starts the unchanged host launcher/phase sidecar once per host. Existing
approval and runtime ownership prevent preparation from authorizing fits.

## Timing limitation

Many current per-fit timing values are provisional placeholders, including
one-second values. They are scheduling metadata and are **not an ETA** or a
reliable experiment window. Hash/environment/source checks do not measure
training throughput, serializer cost, external headroom or two-slot contention.
Only later concrete resource qualification can support an accurate window.
