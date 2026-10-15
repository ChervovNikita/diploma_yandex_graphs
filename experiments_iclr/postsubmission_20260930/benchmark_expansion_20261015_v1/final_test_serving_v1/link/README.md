# Prepared final selected-state link serving

This directory prepares NCN, NCNC and BUDDY inference for the frozen Collab
cells. It leaves the native integration, policy, roster, source snapshots and
original paper scores unchanged. No model, dataset, optimizer map, VALID pass,
TEST payload or TEST metric was executed during preparation. Three small
author-source files were acquired at the existing pinned BUDDY commit and
verified against its retained Git tree.

Status: source and metadata/gating checks pass. Actual selected checkpoints,
official query/provider receipts, native numerical parity, resources and an
explicit final TEST GO are still required. There is no runnable TEST request
or GO artifact in this package.

## What the backend prepares

`backend.py` exposes:

```python
validate_request(request, repository_root)  # source/metadata checks only
qualify_VALID(torch, request, repository_root, admit_runtime=actual_allocation_gate)
serve_TEST(torch, request, approval, repository_root, admit_runtime=actual_allocation_gate)
```

The execution APIs require the caller's actual allocation admission function
before data/model work. This package contains no server command, launcher,
automatic source fetch, fit loop or tuning operation. Importing it loads no
tensor library or native source definitions.

`qualify_VALID` accepts a VALID request whose data role map contains GRAPH,
TRAIN and VALID only. It reconstructs isolated selected native sessions,
restores their own state/RNG, compares prepared inference against the frozen
selected session's complete VALID outputs using explicitly declared tolerances,
and returns a qualification report. The caller must persist that report with
its byte/SHA receipt. It does not open TEST. BUDDY qualification compares the
original combined-query inference against the frozen port's two class-specific
calls; that call-order difference is recorded rather than treated as exact
source operation parity.

`serve_TEST` requires all of the following before a TEST payload can be read:

- A source-bound request pinning all inputs, selected states, runtime bindings,
  raw-logit serving rule and exact backend sources.
- A matching successful **actual** VALID qualification report for those same
  selected checkpoints, GRAPH/TRAIN/VALID inputs and comparison tolerances.
- Explicit human authorization for the SHA256 of that exact canonical request,
  with `action="score_final_selected_TEST"`, `authorized=true`,
  `approved_by="human"`, `selection_frozen=true`, `request_sha256` and a
  nonempty `authorization_reference` identifying the user's GO.

An approval envelope is a binding check, not proof of human identity. The
calling orchestration must obtain and preserve the real user's authorization;
this package never creates or infers it. Editing a selected checkpoint, query
file or request invalidates the pinned input or GO. No private gate,
temperature, posthoc fit, alternate test population or fallback checkpoint is
supported.

## Own selected native, shared-bank and I4 states

The selected-state interface matches runtime_v2's
`native-full-epoch-recovery-v1` payload: `job_hash`, direct LinkEpochAdapter
state, selector state, global RNG, exact runtime binding and complete VALID
summary. A serving descriptor pins `JOB.json` and `selected.pt` separately.
The backend requires a positive completed winning epoch with
`selected_epoch == completed_epoch`, official VALID160084 evidence, matching
source/data/controller binding and original Parameter identities. A current
state, `recovery.pt`, final epoch or no-winner BUDDY outcome cannot substitute
for an own selected checkpoint.

One native or shared boundary fit uses `own_selected_raw_logits`. A shared
bank retains its existing route parameters, task-only NCNC output scope and
raw-logit mean. Genuine I4 uses
`mean_four_individually_selected_native_logits`: four native_single jobs,
bodies0..3, the same cohort/data/fixed recipe, declared frozen seed offsets,
disjoint Parameter storage and four own selected checkpoints/epochs. It averages
the four selected raw logits. A pooled bank epoch is not selected here.

## Native NCN/NCNC TEST graph and calls

The source `loaddataset` discards graph weights. For Collab with the author
command's `--use_valedges_as_input`, TEST uses the original processed TRAIN
`edge_index` concatenated with raw positive VALID edges, then
`SparseTensor.coalesce()` followed by `to_symmetric()`. This exact ordering is
implemented; VALID continues to use the original TRAIN adjacency. TEST query
edges are never added to the graph.

The selected encoder runs on the role's native adjacency for each existing
route. The unchanged predictor evaluates positive queries then negative queries
through the source `PermIterator(..., training=False)` with batch131072.
NCNC recursive completion remains native and its corrected output factors
apply only at the top-level task. The output is the native raw score, without
sigmoid, calibration or a new private module.

## Native BUDDY TEST graph, hashes and cache

The source first filters TRAIN to year>=2007, preserving its weights and the
source `to_undirected(..., reduce="add")` law. TEST appends separately
undirected positive VALID edges with the original VALID weights. The unchanged
source `HashDataset` then coalesces the combined weighted graph, constructs its
weighted CSR degrees, applies original SIGN-k0 feature propagation, builds
TEST-graph MinHash/HLL context and hydrates every official query's structural
features. It retains H1024, max_hash_hops2, hll_p8, minhash128, original
normalization, floor_sf=false, use_zero_one=false, use_RA=false and all learned
native/boundary state.

The backend executes unchanged `HashDataset` and `get_buddy_preds` AST
definitions from verified author bytes. Source branches excluded by the fixed
Collab recipe stay disabled. There is no dummy data, dummy label, substitute
feature generator or TRAIN-cache fallback. The selected model is restored
without any TRAIN update.

Every request must declare `buddy_cache_mode="fresh_original_HashDataset"`
and a **new**, project-contained `cache_directory`. Existing paths are rejected.
Original cache construction/writing happens only during the authorized role
operation. A context sidecar records the role, graph/query input receipts,
backend/source pins and generated native cache receipts. TEST never reuses
TRAIN or VALID hashes/features/cache. Missing VALID weights or incompatible
graph/query/cache schemas fail; no default weight of one is supplied.

The newly acquired source resolves a prior audit ambiguity: although the
outer source Collab VALID/TEST loaders are configured to shuffle,
`get_buddy_preds` reads `loader.dataset` and creates its own sequential loader
over **combined positive-first/negative-second links**, batch1,000,000. The
outer shuffled loaders are not iterated in this BUDDY path. The backend uses
that original function, not the frozen port's separate positive/negative model
calls. Its output is raw logits. Each internal DataLoader can consume an
iterator base seed; omitted whole-main TRAIN/TEST evaluation and construction
order remain RNG-fidelity limitations. The earlier frozen policy's statement
about outer loader configuration must be read with this source-level correction.

## Queries and evaluator

Role query files must be the official Collab dictionary schema `edge` and
`edge_neg`, each int64 Nx2 with node IDs in0..235867. The backend visits every
record in provider order, retaining positive-first labels, endpoint pairs and
stable row IDs. VALID counts are fixed60084/100000. No TEST count is guessed:
the request must pin the official TEST file and explicit complete counts from
an authenticated provider audit artifact.

The provider artifact must have schema `official-collab-query-population-v1`,
dataset, role, `complete_official_split=true`, `provider_release`, exact
`query_file` receipt and `positive_records`/`negative_records`. Its receipt is
pinned in `query_population.provider_evidence`. A matching tensor shape alone
does not certify an official population.

Evaluation uses `ogb.linkproppred.Evaluator(name="ogbl-collab")`, K50 and the
complete official positive and negative prediction vectors. This is the
source NCN evaluator operation and BUDDY's saved `evaluate_hits` operation.
Output metrics use raw `hits@50`. Native/shared/I4 comparisons share the same
official graph role, query files and evaluator. Source auxiliary TRAIN metrics
are not fabricated for final serving.

## Request construction and remaining work

`REQUEST_CONTRACT.json` describes the required fields. `backend_sources` comes
from `backend_sources()` and includes this exact backend and source-binding
file hashes. Every input descriptor has `path`, `bytes`, `sha256`; paths and
symlinks must resolve within the provided repository root. TEST request
`VALID_qualification` pins the actual report's JSON file. `request_sha256` for
GO is `digest(request)`, using sorted compact JSON with nonfinite values
forbidden. Explicit rtol/atol must be frozen before VALID qualification.

Still required: actual link selected fits with eligible winners, official
provider/TEST receipts and counts, BUDDY VALID-weight schema confirmation,
actual source context construction and VALID parity, I4 own-state qualification,
resource and failure receipts, external runtime/export wiring and the final
user GO. Source main-program TRAIN/VALID/TEST sequencing and RNG trajectory are
not reproduced by a standalone selected-state evaluator. No TEST access or
score is authorized by this source package or its static checks.

Run source-only checks from the phase directory:

```sh
python3 -B benchmark_expansion_20261015_v1/final_test_serving_v1/link/check_static.py
```

`SOURCE_AUDIT.md`, `primary/ACQUISITION.json`, `LOCAL_CHECKS.json` and
`MANIFEST.json` retain the evidence, exact source hashes and pending status.
