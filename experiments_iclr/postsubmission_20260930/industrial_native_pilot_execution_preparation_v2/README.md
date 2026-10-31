# Complete first-stage native Tolokers2 source packet v2

This packet supplies the public preprocessor, a restricted native GraphPFN GAT source closure, disposable full-graph qualification, a concrete closed-worker launcher, and the frozen nine-fit baseline runner. It is source-only preparation. This author performed source retrieval, text/AST inspection, hashes and metadata checks; no archive/data/model/GPU execution, scientific import, installation, compilation or SSH was performed.

## Fixed scientific scope

`CONFIG_MATRIX.json` and `ROLE_POLICY.json` are byte-identical to sealed v1. v1 and the R17 source packet remain unchanged. The grid is c0/c1/c2 at seeds 17/29/43: exactly nine native fits, at most 9000 useful updates, max 1000 per fit and AP patience 100. c0 is lr 3e-4/fraction none; c1 is lr 1e-3/fraction none; c2 is lr 3e-4/fraction quantile-normal. All use numeric quantile-normal, dense categorical one-hot, preprocessing seed 0 and dropout .1.

Selection is maximum arithmetic mean of each seed's earliest strict validation-AP best; exact ties follow c0,c1,c2. The selected seeds must each exceed the train-derived constant predictor's validation AP, with mean uplift at least .05 and finite nonconstant predictions. The selected three native fits are reused as one three-member mean-probability validation ensemble. Fixed-last warm50 state/RNG is retained for provenance. There are no extension fits, initializer arms, test-label decoding or final metrics in this runner.

The model is the pinned custom width-512, three-block, four-head, pre-LayerNorm residual GAT with exact `[N,128,4]` channel packing and two-logit CE. The graph pipeline removes loops, makes the graph simple, then bidirected; it never readds loops. The saved tuning TOML's `add_self_loops=true` remains a disclosed source/config discrepancy. The source-defined feature transform order is numeric/fraction/category; model concatenation is numeric/category/fraction.

## Declared v2 source adaptations

- Native classes/functions are extracted mechanically from commit `3b9b115490249cc777227c846babfb55f35bd8c4`. The whole `Model` class, including RNG-consuming construction of unused linear alternatives, is unchanged. Imports/lookup tables admit only the frozen GAT, LayerNorm and backbone; embeddings and PEARL are rejected. This is a restricted source closure, rather than a full GraphPFN package import.
- The independent public-only oracle preserves the pinned loader AST after removing target loading, all-label missingness masking and returned labels, and replacing label-derived node count with public feature-row count. Exact source transforms are shared by candidate/oracle and statically checked; independent runtime comparisons cover loader assembly, typed arrays, effective source masks and graph eid order.
- v1's missing-token vocabulary is explicitly a draft in its preserved role policy. Before any outcome, v2 resolves that gate using native `pandas.read_csv(index_col=0).astype(numpy.float32)` on buffers containing official train/val rows only. Compact packs and nonmissing source masks are constructed directly from those native parsed values. Parser versions and hashes are recorded. Test/unassigned rows are skipped before semantic target conversion. The source-role and statistical selection definitions are unchanged. This construction receipt is provenance, not an independent security certificate.
- Candidate runtime is Python 3.12.9, Torch/DGL 2.4.0+cu118, and torchdata .8.0, with other selected top versions in `CANDIDATE_RUNTIME.json`. The published lock uses Torch's PyPI artifact and DGL cu124, with no torchdata pin. The compatible cu118 choice is explicit for the available driver, and is not exact published-environment reproduction or a claim of intrinsic version requirements.

## Execution boundary and current evidence

Root separately provisioned the candidate environment and passed package-import, CUDA launch and NumPy bridge capability on its authorized A100 route. `SOURCE_BINDINGS.json` binds those existing metadata receipts separately from this author's work. That capability does not establish model/preprocessing parity, resource feasibility or isolation.

`launch_isolated.py` uses unshare user/mount/PID/network/IPC/UTS namespaces, a dedicated repo-owned library image, chroot, closed inherited descriptors, cleared environment, capability removal and no-new-privileges. Source/public/compact inputs are readonly; only a dedicated output volume and temporary tmpfs are writable. One approved physical GPU node plus required control/UVM nodes is mounted for CUDA. Host root, repo, home, archive, raw targets and test packs are absent. A fresh proc filesystem provides PID namespace and readonly device-identity observations; it is not a host-process bind.

A **public-only isolation probe must pass before source-label export**. It records actual mounts, deny-path open failures, zero capabilities, network isolation, image/source hashes and mounted GPU UUID without importing scientific packages. Every later worker launch independently repeats these OS observations. Receipts cannot replace the observed access boundary. Root remains responsible for the archive-side custodian process, dedicated Python/environment source-prefix custody and output-volume provenance.

`build_runtime_image.py` copies only reviewed dedicated CPython/environment prefixes and explicit system/driver library paths. It does not install or compile. The worker hashes every declared library file/symlink, rejects unexpected library payloads and binds actual image identity along with resolved package RECORD/METADATA hashes. Exact observed runtime, input, source and GPU scope must match between qualification and the nine fits.

## Outstanding runtime gates

All v2 operational/numerical gates remain unexecuted by this author:

1. Actual image relocation/shared-library closure, namespace/mount/chroot/capability removal and deny-path/GPU-minor binding.
2. Custodian public freeze, pre-label worker probe, native source-role parsing and compact-pack custody.
3. All three configurations on the complete graph: public/native preprocessing and graph parity; native DGL/Torch outputs and named gradients; populated-history AdamW group/state/parameter/logit parity; K1/K4 identity/private-bias transport with R/S frozen; cold K1 JVP/VJP/three-direction central finite differences at the two predeclared epsilons; closed validation-AP rank parity.
4. Full-graph CUDA determinism/unsupported primitives, K4/AD memory and latency feasibility, then exactly nine native fits and the fixed competence screen.

There is no smaller graph, backbone substitution, loop correction, AMP, epsilon search, hidden seed/config search or unisolated fallback. A failed phase preserves error/cost/progress evidence and stops dependent work. Disposable qualification schedules seven optimizer steps per configuration (21 maximum); these are separately costed and retain no useful checkpoints. No selected-donor warm/extension qualification is implied by this cold disposable certificate.

## Files and source verification

`HANDOFF.md` gives the ordered future CLI steps. `SOURCES.md` attributes source/runtime evidence. `EXTRACTION_LINEAGE.json` binds source segments. `check_sources.py` uses only standard-library text/AST/hash/JSON checks and never imports scientific sources. Its sealed read-only invocation is:

```sh
python3 check_sources.py
```

`STATIC_SOURCE_CHECKS.json`, `MANIFEST.json` and `SEAL.json` bind this preparation. Source-check success verifies source/metadata consistency only. No novelty, competitiveness, generalization, reproduction or low-cost result is claimed.
