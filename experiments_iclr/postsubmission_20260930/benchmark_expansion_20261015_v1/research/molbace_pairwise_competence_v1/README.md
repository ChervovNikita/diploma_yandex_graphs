# MolBACE pairwise competence: prospective source package

Implements the fixed proposal in
`research/pairwise_competence_synthesis_v1/NOTE.md`. This is **source only**:
no new model, tensor/data execution, scientific fit, remote operation or GPU
qualification has been performed. Both admission templates remain closed.
The running structural study and its outcomes are untouched.

## Fixed producer, objective and controls

Shared `common`, `structured` and `permuted` arms reuse the exact existing
residual-only/private-BN producer: native MolBACE GIN, five layers, width
300, dropout .5, full categorical features, batch 32, rank-8 zero-up input
residual, private BN running state and original shared BN affine parameters.
No source operators, native learned parameters, sampler, Adam groups,
dropout law, evaluator or serving rule are changed by this package.

For one eligible native TRAIN minibatch, observed positives i and negatives
j define distinct pairs. Reuse each route's existing raw task logits:

```text
F = native mean own-route observed-label BCE
R = mean_route mean_observed_pair a_m(i,j) softplus(-(z_m(i)-z_m(j)))
J = F + 0.5 R
```

All native and private learned coordinates receive one backward of J and
one ordinary native Adam step. Shared arms still perform four route
forwards; native controls perform one. There is no extra model forward or
embedding objective. The pair denominator is the observed opposite-label
pair count, never the weight sum. Missing labels use the native mask.
For an empty pair set the loss is the **exact F object**, and the eligible
native optimizer update proceeds. There is no pair-dependent resampling,
skipping, stale-logit carryover or BCE renormalization.

`native_common` is an unwrapped native single body with its own BCE plus
the same .5 common pair auxiliary. Its descriptor retains
`procedure=base_procedure=native_single`; the explicit extension, settings
and source receipts identify its changed training objective. This keeps
the original own-selected I4 aggregator applicable. Four independently
initialized bodies per cohort use offsets 0, 100003, 200003, 300007; their
M1 is exact body 0. Each body has its own optimizer and selector. No body
is a teacher or is fitted from another body.

Descriptors are available in memory through `extension.descriptor` for
cohorts 101, 202, 303. **There is no 21-fit launcher or generated job queue.**
The prospective cost would be nine new shared fits plus twelve native
bodies. Original native-BCE M1/I4, original GNNM and residual-F references
remain separate, unchanged comparison procedures.

The retained matched controller is min 100 epochs, patience 50, cap 1000,
first strict whole-VALID ROC-AUC maximum, no checkpoint 0. Mean raw logits
serve shared banks; a native body's own raw logit serves that body. BCE at
the AUC-selected checkpoint remains a required proper-loss diagnostic.

## Frozen TRAIN roles and permutation

The native context helper constructs CPU FP32 concatenated per-field bond
category frequencies from actual source directed bond records. No edges
are coalesced or symmetrized. Native vocabulary authority is atom
[119,5,12,12,10,6,6,2,2], bond [5,6,2], checked against the actual native
encoder functions and, in a live model, every native bond encoder layer.
Only the full 1210 TRAIN inputs are indexed. Full features stay unchanged.

`roles.py` orders molecules by source graph ID, normalizes each nonzero B
in Python float64, and computes cosine with `math.fsum`. An exactly zero B
remains zero, so its similarity to any input is zero. There is no numerical
cutoff, similarity clipping or tunable distance parameter.

All **731445 unordered distinct TRAIN input pairs**, including same-label
and missing-label pairs, determine four equal-mass similarity quartiles.
No label, prediction or VALID record participates in this rule. Exact
score ties receive fractional membership: a tied sorted group [L,R) is
intersected with each quartile in integer quarter-rank units. Only the
three boundaries and their tied-group integer numerators are retained.
The resulting symmetric weights are `a_m=.5+2q_m`, in [.5,2.5], with mean
one across routes. If every input score ties, all routes receive weight one.

`structured` uses this rule on observed positive-negative TRAIN pairs.
`common` uses exactly one for every route/pair. `permuted` takes the same
structured four-vectors in ascending positive-ID then negative-ID order,
and shuffles the **whole vectors once** with isolated Python
`Random(fit_seed+7030031)`. The vector multiset, route totals and every
pair's route mean are preserved. Pair lookup is symmetric under swapping
inputs. It never redraws during updates or restore.

At most seven structured vectors are possible: four pure bins and three
boundary-tie vectors. A byte per observed opposite-label pair records the
assignment, instead of a dense four-route tensor table. The table records
all-input and opposite-label support/mass, unique scores, largest tie,
boundary ties, zero-bond support, class mass and per-molecule support/mass
distributions with an exact-vector digest. Equal all-input mass does not
imply equal opposite-label, molecule or class mass. No weights are used at
inference. Native controls need no role table.

Inspection also reports the exact class-count hypergeometric distribution
of observed pair support, including empty-pair probability and expected
pair count for batch 32 and the native tail size. This analytic calculation
uses no RNG or loader changes. It describes uniform sampling before native
skip conditions, without treating minibatches as independent or predicting
batch-dependent dropout/BN scores.

Fresh shared construction rebuilds roles from authentic TRAIN inputs.
Checkpoint binding retains compact recipe/source, TRAIN ID/label/B,
quartile and assigned-vector hashes plus summaries, without copying the
all-pair score array into every checkpoint. Restore rejects changed role,
source or permutation authority. Existing native optimizer and private-BN
restore mechanisms are retained. Pair-support histograms, empty batches,
observed class counts and route mass counters persist with the update
clock. Host preprocessing/arithmetic times are diagnostic; synchronized
CUDA latency and memory remain unmeasured.

## Bounded inspection before any model construction

`qualify.py --mode inspect` is the default. After exact allocation, original
virtualenv, CPU-only environment, manifest, source/data and root-admission
gates, it uses the unchanged native processed provider, the pinned scaffold
TRAIN role and actual native `mol_encoder` vocabulary functions. It indexes
**only all 1210 TRAIN graphs and TRAIN labels**, then constructs and reports
the role table. It creates zero models, forwards, optimizer steps or VALID
views. VALID/TEST graph records and labels are not indexed. Reading opaque
provider cache bytes for integrity is separate from indexing records.

Root can use this mode to inspect pair viability, coarse/degenerate ties,
opposite-label support and route/class/molecule mass before admitting any
new numerical model qualification. It can be called programmatically as
`qualify.inspect_train` after the same outer guards. No automatic
interpretation threshold or outcome-triggered recipe amendment is added.

After root writes an actual approved admission from
`INSPECT_ADMISSION_TEMPLATE.json`, the interface is:

```text
CUDA_VISIBLE_DEVICES='' <repository>/.venv/bin/python -B qualify.py
  --project-root <repository> --mode inspect --arm structured
  --admission <approved-admission> --admission-sha256 <actual-sha256>
  --output <fresh-repository-contained-output>
```

Root owns a finite outer timeout and resource admission. The source
template itself grants no execution. The CLI is restricted to the exact
authorized one-GPU allocation even though this mode uses no GPU model.

## Separate fresh-restore qualifier source

`--mode lifecycle` uses the separately approved lifecycle template for
one arm, mechanical seed 101 (body 0 for the native control), CPU only.
It permits two constructions and **two total optimizer transitions**:
one update, full VALID, save, fresh construction/restore, bit-exact full
VALID replay, restored second update, full VALID. Exactly three complete
VALID151 calls occur, with no selector or scientific fit. The original
CUDA descriptor is retained as planned authority; CPU is an explicit
engineering execution override.

One genuine first shuffled native batch is reused for both updates. No
label-based batch search is allowed: unsupported first-batch pair loss
is a preserved qualification failure. Actual optimizer calls and native
forward attempts/completions are charged. Expected native forwards are
68 for a shared arm and 17 for a native control; no initial-function,
BN-isolation or other diagnostic forwards repeat the existing producer's
qualification. Failures, interruptions, closure failures and unknown work
are preserved without retry. This does not qualify full epochs, scientific
selection, CUDA behavior, production timing or concurrent resource use.

## Checks and scientific limits

`python3 -B check_source.py` runs five source/JSON/synthetic scalar checks:
source closure; frozen native/shared descriptors and original runner
validation; label-independent fractional ties; symmetric whole-vector
permutation/missing-label support; and bounded objective/inspection
contracts. It imports no scientific tensor library and reads no dataset
payload. `SOURCE_CHECKS.json` is deterministic; `MANIFEST.json` pins the
package. `BASE_CASE.json` is an exact copy of the existing residual-F seed
101 descriptor, preserving its complete recipe provenance.

The NOTE's rejection and attribution criteria remain unchanged. Reject the
fixed structural pair recipe unless mean served AUC beats residual F,
common-pair training and the matched permutation; no bin/similarity/.5/
selector retuning follows. Advancement also requires genuine native-BCE
I4 and common-pair I4, with every paired seed and proper-loss harm visible.
Mean/worst route competence, pooling gain, common reversals, acquired/lost
alternatives and served repairs minus harms remain necessary later
analysis. Three exposed-VALID seeds are exploratory; dependent molecule
pairs do not add experimental repetitions. The deterministic fixed-score
uniform-batch expectation has a both-class-support factor; native dropout
and BN make scores batch-dependent. No calibration, superiority or novelty
guarantee is asserted. This target is not ported to PubMed or link tasks.
