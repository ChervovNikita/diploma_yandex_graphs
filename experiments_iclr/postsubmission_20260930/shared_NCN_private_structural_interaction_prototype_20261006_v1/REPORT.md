# Native NCN structural interaction prototype

This package contains actual model code, a TRAIN/VALID driver, a first-fixture
qualifier, and disabled per-job argv/environment bindings. No scientific run,
server action, input payload read, checkpoint read, or saved-score recalculation
was performed while authoring it. Numerical qualification and costs are pending.

## Fixed proposed method

For current target-masked support A, C=N(u)∩N(v), and all unordered internal
edges F={{w,z} in A:w,z in C,w<z}:

```
Psi = MLP768→64→32([h_w+h_z; h_w*h_z; h_u*h_v])
xi  = [1, log1p(|C|), log1p(deg_C(w))+log1p(deg_C(z)), |deg_C(w)-deg_C(z)|]
g_m = sigmoid(r_m dot xi)
f_m = f_base_m + sum_{e in F} g_m(e) * (Psi(e,q) dot a_m)
```

The feature construction is symmetric in both endpoint pairs. The function
family can express nonseparable edge interactions; learning is not guaranteed
to use them. Psi has one shared dense MLP for the main ensemble. The native GCN
runs once per shared forward. Native F4 dense matrices, bias placement, private
factor rows, LayerNorm affine, and beta are preserved. Existing routed dropout
training calls still each pay their own encoder computation.

Only the **new** private branch parameters are gate r and readout a; the native
base private parameters remain present. r and a start at zero. This includes the
base function at initialization, while Psi and r gradients are initially zero.
a generally receives a gradient; Psi/r can learn once a moves. When there are
no witnesses, an exact-zero graph anchor connects every branch parameter so
ordinary Adam still advances its declared moments with genuine zero gradients.
No witness work is fabricated by that anchor.

At width256 the shared branch adds 51,296 dense parameters and 144 private
parameters (four times 4+32). The tied branch adds only 36 private parameters.
That tie has lower effective capacity. Shared F4 totals are 1,460,322 shared-role
and 25,752 private-role parameters at CiteSeer's input dimension3703. Role labels
describe the existing partition; ordinary joint Adam updates all parameters.
Independent control matrices are separately owned even where partition role
names say `shared`.

## Builders and fits

| Builder | Added information and sharing | Fresh fits |
|---|---|---:|
| `shared_f4` | Original E_joint builder, dynamic input dimension | 0 |
| `ordinary_native4` | Original independent native4 builder | 0 |
| `structural_private` | One encoder/Psi; four private r/a rows | 3 |
| `structural_tied` | One encoder/Psi; one r/a row broadcast to four heads | 3 |
| `structure_blind` | All unordered CN pairs; no induced-adjacency/degree gate input | 3 |
| `count_only` | CN/LCL/CAR/CRA/CAA; no neighbor-pair embedding interaction | 3 |
| `informed_single` | Capable full native nonlinear head with same internal-edge branch | 3 |
| `informed_independent4` | Four independent native encoders/heads/Psi networks/r/a | 3 |

The blind and count branches match the main branch's nominal parameter count.
Blind enumeration has different compute. Count-only activates only five raw
statistics, padded into the same Psi input width; its effective input/capacity
differs. It also permits a learned count intercept. These differences are
declared controls, not exact compute or effective-capacity matches.

## Prospective protocol

`FIXED_RECIPE.json` fixes ordinary original BCE, Adam0.001, inner/outer/repeated
inner passes, outer64/inner256, endpoint geometry, seeds0/1/2, original factor
seeds, structural seed600000+seed, width32/hidden64, and chunk4096. The outer
objective is half BCE of mean raw logits plus half member BCE. Independent4
keeps the existing inner sum and outer times4 scaling. No scaled-BPR change is
combined with this architecture test.

All new fits pay exactly60 complete TRAIN cycles, including the tail. Complete
VALID with fixed500 negatives is evaluated every5 cycles. Selection is the
first maximum rounded4 VALID MRR; there is no result-based early stop. TEST
remains closed until a separately frozen confirmation.

The first stage fixes private/tied/informed-single/informed-independent4 across
all three seeds (12 fits). The second stage fixes blind/count across all three
seeds (6 fits), required before adjacency/learned-interaction attribution.
Order is fixed before outcomes. Saved complete39 E_joint and J4_joint scores
and banks are historical references where recipe, inputs, seeds, and selection
align. Six archived FREEZE/job locators are retained; no original E/J4 fit or
score recomputation is in default commands. Historical bitwise replay is not
claimed.

## Root adoption and execution

Choose **one** alternative target plan and review the actual source and recipe:

1. Bind existing runtime, feature/negative-pool authority, the exact acquisition
   manifest, and resource limits into the corresponding qualification job(s).
   Use the existing owned process helper; each child sees one exact GPU UUID.
2. Run `qualify_prototype.py` on each GPU assigned to fits. It uses only TRAIN,
   tests startup/empty-branch gradients on a tiny synthetic fixture, compares
   zero-residual live scores to its native base, and compares one actual first
   TRAIN episode against independent direct model forwards plus native Adam.
   Committed parameters, moments, logits, three updates, dropout streams, and
   global RNG restoration are checked. All six architecture gates are required
   on that physical GPU. Old original-head gates cannot substitute.
3. Bind that new receipt into six seed0 cost jobs. Run one full discarded TRAIN
   cycle per architecture on its assigned GPU; VALID values are not opened.
   If assignments change, measure the additional architecture/GPU combinations.
4. Before any scores, root adopts the target plan after those costs, fixes job
   and phase hard/soft budgets, concurrency, comparator budget description,
   owner telemetry limits, and exact evidence hashes. Rebind each fit job to
   that adopted plan hash and mark only its authorized scope executable.
5. Feed the exact argv/env arrays to the already owned helper/owner. Preserve
   the fixed stages/order. No new process or guard framework is included.

All templates have source/runtime/input/gate/resource approvals unresolved,
`fits_authorized=false`, `VALID_values_access=false`, and null caps. Changing
those flags is not evidence. Root must bind the actual receipts and grants.
The runner checks both exact repository/host/inventory/runtime bindings:
allocation (`anogena-2-0`, one GPU) or77 (`peptide`, two GPUs), with one GPU
visible per child. Existing jobs, failed receipts, and sources are untouched.

## Accounting and limits

Topology is enumerated from `support.coo()` on CPU for every forward. Query
targets present in that support are rejected. Support symmetry, unweighted
values, and absence of loops/duplicates are checked. Every internal edge is
processed. There is no edge cap, sample, skip, fallback, or topology cache.

`drain_work()` counts copied support/query rows, CN instances, inspected
internal-neighbor entries, internal edges or blind pairs, count rows, Psi/gate/
readout rows, chunks, exact empty anchors, and actual encoders. TRAIN and VALID
work are logged separately alongside wall time and peak CUDA allocation/reserve.
CN intersection counters are explicitly upper bounds, not exact FLOPs.

Chunking bounds one chunk's temporary tensors. Saved training activations and
CPU topology still scale with **all** witnesses. GPU memory, CPU/RSS, output,
inclusive wall time, validation cost, and owner telemetry must be measured and
bounded. There is no resource-feasibility or sharing-savings claim.

The model receives `features.shape[1]` dynamically. The current authenticated
loader remains full HeaRT CiteSeer (3327×3703); a second dataset needs separately
adopted loader/data authority. No hidden portability claim is made.

## Static certificate and prior work

`static_cases.py` passes seven stdlib-only graph/algebra cases. For fixed CN
embeddings `[1,1,-1,-1]`, one internal matching has product sum+2 and another
has-2. Both have CN4, LCL2, CAR8, equal induced/global degrees, equal CRA/CAA,
and the same unary embedding multiset. A nonseparable edge product distinguishes
them. This is a conditional fixed-H interface certificate, not a full-GNN
expressiveness theorem. It does not establish novelty or explain saved common
STRICT errors causally.

CAR/LCL/CRA/CAA, NCN's CN2, and SEAL/query-subgraph GNNs are close priors. A
constant-feature regular-graph witness already distinguishable by counts is an
interface limitation example, not evidence of learned-message novelty.

`STATIC_VERIFICATION.json` records AST/topology/algebra checks and byte equality
of the original BCE training step and unchanged private Adam helper. The sealed
source manifest is an authoring receipt. No numerical qualification pass,
training improvement, held-out competence, or practical cost result exists yet.
