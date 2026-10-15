# MolBACE normalization × geometry × private correction

**Prepared and unlaunched.** Nine new fits, three seeds (101/202/303), one
official scaffold split. This package is separate from the 6,460-fit campaign;
it adds zero roster entries. Root owns review, publication, resource admission
and the launch decision. Original studies and paper scores remain immutable.

## Question and reused evidence

The completed shared-BN factorial found a negative AUC interaction between
geometry and private input correction. The completed private-BN pilot then
improved mean member and pooled AUC, with an unfavorable BCE result, especially
seed 303. These observations were available before this follow-up was frozen.
The study is therefore exploratory development on selected VALID, with no
independent confirmation or TEST/generalization claim.

The earlier interaction **was conditional on shared running normalization**.
Shared BN was held fixed across its four cells, so this does not invalidate
that factorial estimate. It could be an important moderator of ingredient
effects; the earlier result cannot reject their combination under private BN.
Calling shared BN a proven cause of the negative interaction would overstate
the evidence. This study measures moderation by normalization policy.

| BN state | Neither ingredient | Geometry only | Private correction only | Both |
| --- | --- | --- | --- | --- |
| Shared | Reuse 3 | Reuse 3 | Reuse 3 | Reuse 3 |
| Private | Reuse 3 | **New 3** | **New 3** | **New 3** |

`SPEC.json` identifies and hashes all 15 reused RESULT authorities, with their
native selected-VALID scalar outcomes, selected/trained epochs, stopping reason
and historical cost. Native M1/I4 remain descriptive context from their retained
parent summary. No control is retrained. The processed-cache hash acquired here
is a prospective integrity pin; the completed RESULTs did not record that hash,
so it is not represented as retrospective proof of identical cache bytes.

## Exact implementation and selection

Use the original `runtime`, callback, complete 1,210-graph TRAIN and 151-graph
VALID, all categorical features, native graph/bond operators, GIN 5×300,
dropout .5, mean pooling, JK-last, batch32, native skip/route/sampler order,
Adam .001 with weight decay0 and original defaults. The recipe descriptor's
`_seed=0` is retained: the unchanged BaseAdapter sets actual `_seed` and `seed`
from each fit's seed. The common descriptor is byte-for-byte equivalent in all
native/controller fields to the completed private-BN jobs.

`session_extension.py` first invokes the unchanged parent molecular transform,
then the unchanged private-BN v3 transform. The parent training closure remains
the four-route mean native BCE plus the declared geometry objective, followed
by one backward/Adam step. The BN wrapper surrounds each original `run_member`.
No extra backbone forwards or learned normalization parameters are introduced.

- Geometry: parent alignment/variance objective, temperature .2, coefficient
  .05, std fraction .5, all-parameter gradient scope, native TRAIN labels and
  graph identities only. All reductions and empty-class handling remain in
  the pinned parent source.
- Correction: four rank8 residuals at the complete native AtomEncoder output,
  `encoded + U_m GELU(V_m encoded)`, isolated seed+901003 orthogonal down-frame,
  zero up matrices, existing atom-embedding Adam group.
- Private BN: shared original affine parameters; each route owns running
  means, variances and counters, updated by its native TRAIN calls only. Eval
  reads its own bank without recalibration. Native canonical buffers and
  identity guards remain intact.
- Serving and selection: arithmetic mean of the four raw logits; first strict
  best whole-VALID ROC-AUC, own checkpoint. Original matched-pilot controller:
  cap1000, minimum100, patience50, no per-epoch TRAIN eval. In this exact
  controller the stale counter is held at zero before epoch100; an early-best
  trajectory can first stop at epoch149. Do not substitute a later controller.

Only new composition metadata and a checkpoint combination/seed/source guard
are added. Native state serialization already carries registered residuals and
BN banks. The unchanged runtime also binds complete job/source hashes. These
source/mock checks do not establish numerical lifecycle parity for the newly
composed native model; no additional model construction or replay is performed
in preparation.

## Frozen hypotheses and reporting

Write `Y[n,g,c]` for one seed's own-selected whole-VALID metric; n=1 means private
BN, g=1 geometry and c=1 private correction. Use **AUC percentage points** and
unmodified BCE units. Positive AUC effects favor the left-hand treatment;
negative BCE effects favor it. Never reverse BCE signs silently.

1. Does geometry or correction help under private BN? Report each conditional
   ingredient effect at both levels of the other ingredient, plus both versus
   neither, for both normalization policies. No benefit is presumed.
2. Does private BN mitigate the earlier AUC antagonism?
   `J_n = Y[n,1,1] - Y[n,1,0] - Y[n,0,1] + Y[n,0,0]`.
   The directional exploratory hypothesis is `J_private - J_shared > 0` for
   AUC. Positive moderation with a still-negative J_private means weaker
   antagonism, not a positive within-private-BN interaction. Report negative,
   zero and positive outcomes, all three seeds, and corresponding BCE values.
3. Does changing normalization improve each of the four matched ingredient
   cells? Report all four `Y[1,g,c]-Y[0,g,c]` effects. AUC gains accompanied by
   BCE harm remain a tradeoff. No scalar composite or post-hoc winner rule.

For every cell and contrast: show all per-seed values, descriptive mean and
sample SD only when all three required seeds are available. Sign counts carry
their denominator. No p-values, significance claims, seed/molecule bootstrap,
confidence intervals, missing-cell imputation or partial favorable promotion.
Three training seeds remain three; routes and molecules are not replications.

`study.py --mode report` reads saved JSON scalars only. BCE is always the native
`RESULT.selected_VALID.loss` (source batch sums divided by observed labels),
including reused controls. Previous summaries sometimes recomputed BCE with a
different floating-point reduction; their tiny numerical differences are
preserved in those reports and are not mixed into this factorial calculation.
No prediction or hidden-state replay, member analysis, recalibration, threshold
fitting, route reweighting, coefficient tuning or checkpoint reselection occurs.

Ordinary TRAIN BN uses batch moments. Shared running EMA mixes successive route
histories, with later routes receiving greater weight; private banks retain
separate histories. Under matched parameters/input/RNG, ordinary instantaneous
TRAIN math does not depend on these running buffers. Eval and the selection
trajectory do. Different selected epochs and stopping lengths mean this study
tests methods under different normalization policies, not an isolated
same-weight causal buffer intervention.

## All outcomes, convergence and cost

The exact finite order is seed101/202/303, then geometry/correction/both within
each seed. One new child at a time. A failure stops further admission, records
all remaining jobs as not admitted and receives no automatic retry. A cap is
reported as complete but convergence-unresolved; it never triggers extension
or retuning. Only the existing patience rule establishes a controller plateau,
not an optimization or generalization theorem.

Preserve every CURVE, log, selected/recovery checkpoint, selected export,
failed/capped outcome, actual optimizer attempts/completions, wall time and peak
memory. Missing terminal receipts retain available progress and unknown cost.
Report selected and trained epochs even when far apart. No missing interaction
cell is imputed. At most9×1000 complete epochs and342,000 native Adam steps are
planned; skipped batches can reduce that bound. Actual costs govern reporting.

## Ancestry and limits

Retained primary readings are pinned in SPEC:

- [BatchEnsemble](https://arxiv.org/abs/2002.06715): shared slow weights and
  route-specific factors. Retained abstract and located BN-related paragraphs
  are not proof of this exact private-buffer implementation. The inspected
  Uncertainty Baselines source uses ordinary Keras BatchNormalization.
- [AdaBN](https://arxiv.org/abs/1603.04779): retained abstract and §3.2/Algorithm1
  prose support statistics-only/domain-specific normalization ancestry. Its
  target-domain recalibration is not used. VALID and TEST never update banks.

Private normalization and structured ensembles have antecedents. Success here
would support an improved baseline and a conditional interaction observation,
not novelty, manuscript acceptance, or broad transfer by itself.

## Review and execution interface

`python3 -B test_source.py` runs only static/fake-session and scalar-algebra
checks. Remote `study.py --spec SPEC.json --spec-sha256 <frozen hash>` defaults
to guarded source preflight; it creates no model and launches nothing. Use
absolute project paths. `MANIFEST.json` binds this immutable package.

After root's decision, launch mode additionally requires a root-owned JSON
receipt with schema `root-reviewed-nine-fit-admission-v1`, the exact
`SPEC_sha256`, approved=true, new_fits=9, maximum_concurrent_fits=1,
allocation_GPU_UUID=`GPU-44039938-fd82-41d2-fefd-de71514e2fac`, and
resource_admission_approved=true. No such receipt is created here.
Use the allocation's existing project `.venv/bin/python`, a fresh `--output`,
`--mode launch`, and `--launch-authority` only after that review/admission.
It cannot resume or overwrite a prior fit. Reporting accepts a fresh
`--report-path`. All remote access uses the parent guarded `remote_control.call`.
