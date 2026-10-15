# Scientific audit and compact task × method matrix

## Assessment

The catalogue is a useful mechanism inventory, but C1–C4/C6 are not five fully
specified, interchangeable cross-task algorithms. The existing code has sound
supervision and identity safeguards, several meaningful task adaptations, and
important differences in targets, gradient credit, placement and implementation
status. A matrix should name those differences explicitly.

**Retaining the native supervised term is a necessary design constraint, not a
competence guarantee.** None of the contrastive, variance, reweighting, adapter
or gate definitions proves that individual or pooled predictions improve, that
proper loss is protected, or that shared-weight trajectories equal the base.
The supplied result index already contains counterexamples to interpreting
additional member coverage or accuracy alone as better ensemble quality.

This audit reads the supplied catalogue, objective/adapter sources and the
30-record prior-result index. It performs no history search, paper review,
model import, score recomputation, fit or server operation. The active molecular
pilot and its frozen criteria are unchanged. Proposed definitions below are
future scientific freezes, not assertions of implemented/qualified coverage.

## Compact matrix

At every admitted task/preset retain: **separate native source reference; matched
native M1; base boundary GNNM G0; genuine independently trained/selected I4**.
Matched M1 is exactly I4 body0. GNNM selects one complete shared bank; I4 selects
four own bodies. Source-horizon references remain a separate protocol track.

| Task | C1: task geometry | C2: persistent risk assignment | C3: context information | C4: nonlinear private input | C6: conditional serving |
| --- | --- | --- | --- | --- | --- |
| Node classification | TRAIN class-compatible contrast + within-class distinct-node variance; compare alignment alone. Native node CE stays intact. Native binding/budget pending. | Within-TRAIN-class incoming-degree quartiles; positive weights; exact classwise permutation control. | Fixed native-feature versus graph-propagated-feature affinity targets; route assignment versus common mixture. | Zero-start residual after native `lin1` encoded input; retain BE for primary cell. Portable source prepared, native qualification pending. | Uniform mean versus TRAIN-learned global/conditional convex weights on class-aligned raw logits, under the same own/pool objective. |
| Binary molecular graphs | Existing graph-prehead class-balanced alignment + within-class distinct-graph variance. Missing labels excluded; no augmentation assumed label-preserving. | Within-observed-class atom-count quartiles; positive weights; classwise permutation control. Scaffold assignment is a different deferred target. | Native atom-category versus bond-category histogram affinities; route assignment versus common mixture. No new inference descriptor. | Existing zero-start rank8 residual after full AtomEncoder output, before unchanged message passing; retain BE. | Graph-level native preheads/logit descriptors; same mean/global/conditional comparison and observed-label BCE. |
| Sampled-query link ranking | Existing same-query cross-route positive with endpoint/opposite-source-label alternatives; compare support/count-matched global alternatives. **No variance term exists in this implementation.** | Corrected endpoint-correlated random field versus raw-marginal-matched query field, preserving complete native reduction groups. This tests correlation, not semantic expert ownership. | Edge-excluded TRAIN endpoint-degree versus witness-context affinities, within source label/relation groups; route assignment versus common mixture. | Choose portable input residual after the declared native first encoder branches. The older task-prehead residual is a different site hypothesis, not another alias. | Query-aligned native preheads/raw logits; source NCN separate positive/negative means or BUDDY joint batch mean; same mean/global/conditional objective. |

The table is a compact scientific structure, not an instruction to cross every
cell with all 27 recipes or to launch new fits. Critical controls travel with
their question. Uniform C2 weights reuse G0; the uniform mean half-own/half-pool
control needed by C6 reuses the already named **global** half-pool rule L1.
Implementation gaps do not exclude any of these three tasks.

## What supervision actually guarantees

Let `F = mean_m native_loss(z_m,y)` and `L = native_loss(served_logits,y)`.
Molecular code preserves FP32 observed-label BCE. Link code preserves
`mean(pos loss)+mean(neg loss)` for NCN/NCNC and the original joint mixed-batch
BCE mean for BUDDY. Auxiliary query budgets do not reduce F's population.

For all-block C1 the update criterion is `F + .05 A`. For current private link
C1, shared gradients are from F and private gradients from `F + .05 A`.
For C2, shared gradients are from F and private gradients from positive weighted
risk W. Those are distinct vector fields, not necessarily gradients of one
global scalar objective. Subsequent F gradients depend on the altered private
state, so “shared F” does not mean unchanged shared training trajectories.

Positive weights retain every native supervision record but change its
importance. A floor on **raw** weights is not a fixed floor on normalized
influence, and within-batch ratio normalization is not an unbiased original-risk
guarantee. A coefficient .05 is not a bound on auxiliary gradient magnitude
relative to task gradients. Measure both losses, gradient norms/exposure and
member/pool predictions; do not call these definitions competence-preserving.

### C1: alignment, variation and scope are different questions

The molecular positive set contains same-label graphs across all existing
routes, including cross-route copies of the same graph. The denominator averages
each observed binary class before the outer logsumexp; anchor losses are
class-balanced. With only one observed class, contrast is zero. Within-class
variance can still be active when that class contains at least two distinct
graphs. A singleton class can have cross-route same-graph positives but no
distinct-graph variance support. Report these cases separately.

The variance term is across graphs **within each route/class** after unit
normalization. It does not require routes to differ. At exact within-class
collapse, `d var(z)/dz = 0`; a positive variance penalty can therefore have zero
gradient. It does not prove collapse avoidance or preserve task-relevant
variation. Aligning all same-class examples may also discard useful within-class
structure. Retained native supervision must resolve that tradeoff empirically.

The frozen active molecular extension always instantiates `alignment_variance`;
it does not expose the catalogue's alignment-only B1 through its current boolean
auxiliary switch. Its 2×2 pilot estimates the total auxiliary effect and its
interaction with the input adapter, **not the separate value of variance**.
Do not amend that pilot or infer a variance benefit without its targeted control.

Current link C1 is conditional InfoNCE/cosine alignment, with no variance loss.
Its positive is the same authenticated query in another existing route; its
alternatives are existing opposite observed/sampled-label rows with the same
relation, optionally sharing an endpoint. Different observed links are not
declared interchangeable positives, and sampled negatives are not certified
absent edges. Global and endpoint controls match anchor support and alternative
counts, which is a useful design strength. Cosine versus InfoNCE also changes
loss scale and denominator, so it is an alignment ablation rather than a clean
single-factor causal test of endpoint structure.

Link preheads are captured before the final boundary factors. Auxiliary credit
cannot directly train those final R/S/bias factors; private credit reaches
upstream private coordinates. State that actual gradient support rather than
implying every private factor receives A. Endpoint coverage can be much smaller
in BUDDY's batch1024 than NCN's combined131072 rows. Zero coverage means zero
auxiliary, not evidence against contrast. Report exposure before allocating a
large method comparison; do not fabricate negatives or change source batches.

### C2: a compact, discriminating assignment

For nodes use incoming-degree quartiles **within each TRAIN class**. For
molecules use atom-count quartiles within each observed binary TRAIN class.
At ties spanning boundaries, fractional assignments q must satisfy
`sum_m q_m(i)=1` and `sum_{i:y=c} q_m(i)=N_c/4`. Set `w_m(i)=.5+2 q_m(i)`.
Thus every member has exactly the original total class mass and every record's
mean weight across members is one. Freeze the table; do not assign winners from
current losses. The decisive unstructured control permutes the **whole member
weight vector** among objects within each class. It preserves the actual weight
values, member/class mass and per-object total, while removing association with
degree/size. An additional unmatched exponential-IID table is not necessary for
this particular structural contrast.

Keep original source loss reductions: a molecular class-mass-balanced table
does not license replacing native mixed-batch BCE with a balanced two-class
loss. Full-TRAIN mass and within-minibatch mass are different objects.

The frozen link v1 compares endpoint averages of two Exp variables with a
single query Exp variable: for nonloops their ideal raw variances differ
(.5 versus1). That comparison cannot isolate endpoint correlation. The new
standalone v2 corrects the raw marginals using two query draws for nonloops and
one for loops, and states that normalized marginals remain unmatched. Its
provisional65536 total-row cap was incompatible with native NCN's
65536+65536 rows; the source was corrected during this audit to cap262144,
with a reported explicit131072-row/group-normalization check. That size issue is
resolved in the helper, not evidence of native integration. The frozen link
integration still imports v1; a new explicitly qualified bridge is needed for
the corrected comparison.

Endpoint hashing supplies a persistent **topology-correlated random field**:
incident queries share latent endpoint draws. It does not assign semantic
communities, hard experts or known “easy”/“hard” niches. Its attribution should
be covariance/persistence under positive task weights. Even v2 does not isolate
only covariance after group normalization; avoid a stronger claim.

### C3: replace the umbrella with one explicit information contrast

“Fixed-key, affinity-weighted or positive-exclusion” lists three objectives;
it is not one runnable C3. A compact proposed freeze is **fixed context affinity**:

1. Choose two deterministic TRAIN-only target channels A/B: node native x versus
   normalized-graph propagated x; molecule native atom-category versus
   bond-category count histograms; link endpoint log-degree/balance versus
   common-witness context, computed with the queried positive edge excluded
   from the target graph. Target computation does not change predictor inputs,
   source topology or negative sampling. Pin histogram vocabularies/propagation
   law and charge target preprocessing.
2. Use the same fixed candidate-budget/identity rule in both conditions. Exclude
   the same object. Give target mass only to other same observed-class donors
   (same source binary group and relation for link queries). Use a fixed
   normalized-context affinity, for example `exp(cos(t_i,t_j)/.2)`, normalized
   over supported donors. Do not convert sampled nonlinks into latent truth.
3. Predict a donor distribution from live same-route native-prehead cosine
   scores at fixed temperature. Minimize cross-entropy to the fixed affinity
   target in addition to F. Omit/report anchors with no compatible distinct
   donor; no fabricated class/prototype is added.
4. Stable assignment gives routes0/1 target A and routes2/3 target B. The common
   control gives every route `.5 q_A + .5 q_B`, with identical candidate rows,
   work, coefficient and declared gradient credit. Use all-block `F+.05 A`
   for this proposed primary contrast and report that choice explicitly.

This tests task/context information and stable target assignment, with no extra
inference input or auxiliary head. A uniform class-compatible target is a useful
target-information ablation when operation-matched, not automatically the same
as C1's different cross-route positive set. Handle absent bond/context signal
and donor coverage explicitly. This proposal is not implemented/qualified by
the current catalogue and is not claimed to reproduce AMCL/CGCL/MoCL.

The retained index shows that updatewise target shuffling can have COMMON's
conditional expected objective/raw gradients plus extra covariance. Do not
count that expected-mean-equivalent shuffle as a new context-signal mechanism
or claim trajectory equivalence. Denominator exclusion renormalizes retained
gradients; it needs its own exact control and is deferred from the compact C3.

### C4: choose one site hypothesis and limit attribution

The active molecular adapter is a known nonlinear adapter-style correction,
zero-start `h+U_m GELU(V_m h)`, after full AtomEncoder output. The new portable
source maps the same law to native encoded inputs: node lin1 output; NCN/NCNC
first GCN linear output; BUDDY structural/feature branch encoder outputs. Rank
is `min(8,width)` with an explicit frame fallback when four disjoint rank8
subspaces do not fit. These site/dimension differences must be reported.

Use **BE+input residual** as the primary C4 cell. The portable residual-only
four-route shared-body variant is a meaningful optional BE-ingredient ablation,
not native M1, not I4 and not a factorized single. It should not silently replace
the mandatory G0 reference. Its reconstruction/checkpoint schema differs from
the frozen final-serving entrypoint and remains unsupported there until an
explicit reconstruction map is added.

The older link task-prehead residual is a different placement hypothesis. Do
not run input and prehead placements as synonymous C4 cells. A linear residual
before a scalar affine head yields only another linear functional; generically
the existing unconstrained final input factors already span arbitrary private
scalar head weights when shared head coordinates are nonzero. Such a cell mainly
changes parameterization/regularization, not nonlinear expressive capacity.

The catalogue excludes parameter-capacity-control fits, whereas the frozen
optional link roster still lists equal-parameter shared adapters/concatenation
gates. Those are stale, unresolved control requirements, not authorized ready
cells. Keep within-GNNM ingredient controls and disclose parameters/work; then
limit claims to the tested pipeline difference, not “private structure rather
than extra capacity” or a novel LoRA principle.

### C6: distinguish training credit from conditional readout

The prepared gate starts uniform and fits on TRAIN; it uses existing unlabeled
preheads/logits at serving. Its declared criterion is all-block
`J_g=.5F+.5 loss(gated raw logits)`. The correct uniform-mean training control
is all-block global half-pool J, not own-only G0 or shared-F/private-J.
Otherwise a gain confounds aggregation with pooled training credit.

Use mean, TRAIN-learned global route weights and conditional weights as the
small distinguishing set. Evaluate the conditional model's **same selected
bank** under both its gate and a fixed mean to measure a readout effect. A
separately trained mean-J bank versus gated-J bank estimates the whole joint
training-and-serving pipeline, not a pure readout intervention. No VALID-fitted
combiner or selected-checkpoint gate refit is permitted.

For a binary scalar output the current gate's mean route-logit deviation is
delta and its RMS is approximately abs(delta); these are not independent
reliability statistics. For multiclass logits, a mean-logit deviation can be
sensitive to arbitrary class-common shifts. Interpret input ablations accordingly.
Convex binary scores stay inside the route score range per query, but ranking
can still change through query-specific weights. Per-member Hits50 coverage is
not a universal upper bound on the gated/pool ranking result. Multiclass pooling
can also repair examples where every individual argmax was wrong.

## Limited interactions and redundancies

- **Keep existing molecular I1 unchanged:** G0, C1, C4 and C1+C4 are the already
  frozen2×2 pilot. Estimate `Q11-Q10-Q01+Q00` on the predeclared task scale and
  inspect the final combined model as well as interaction. Do not reinterpret
  it as an isolated variance study or rescue a historical node combination.
- **Prepare one C2 × private pooled-credit interaction after its components are
  meaningful:** shared F in all four cells; private `(1-lambda)W+lambda L`,
  lambda0 or.5; W uniform or structured. Reuse G0, C2 and existing private
  half-pool controls. Do not cross this over every task or all contrast variants.
- **Defer C4×C6** until the chosen gate objective/site and standalone effects are
  qualified. C1×C2, all-context grids and all-placement grids are not core cells.
- **Treat C8 as a state question first:** BN-bearing OGB/BUDDY have a meaningful
  object; inspected BN-free PolyFormer/NCN/NCNC do not. Do not add BN to create
  compatibility. Same-weight readouts require prospectively recorded state;
  current frozen checkpoints cannot reconstruct missing histories.
- Merge exact aliases: G0/link-own-reference; uniform C2/G0; global-half-pool
  mean/C6 uniform-J control; existing private half-pool/L1. Cosine-only C1,
  legacy unmatched IID, linear scalar-head residual and updatewise target
  shuffle are controls/history, not independent new mechanism counts.

## Confirmation estimand

The primary efficacy estimand should be the paired expected **native held-out
task metric difference, candidate minus G0**, over prospectively fixed fresh
seed cohorts, conditional on a declared dataset/split/backbone/source preset and
matched training/selection controller. Native M1 and independently selected I4
are mandatory deployment comparisons. Source-horizon reproduction numbers are
a different estimand and do not enter paired tests as new observations.

Report the corresponding paired proper-loss difference, member/mean/worst
strength, actual pool repairs/harms, pooled cancellation/rescue, calibration
where meaningful and full training/serving costs. Molecular BCE is conditional
on observed labels; link BCE/logistic loss is a source negative-sampling
surrogate, not calibrated natural edge prevalence. Threshold correctness is not
the primary molecular AUC or link ranking metric.

Freeze substantive metric/proper-loss acceptance margins before confirmation;
do not choose them from opened outcomes or require every route to beat M1.
An accuracy gain accompanied by materially worse proper loss is a reported
tradeoff, not automatic competence protection. The prior index's shared-wrapper
accuracy/proper-loss disagreement makes this distinction necessary.

Keep per-seed deltas and failed/capped/no-winner outcomes. Do not treat graph
rows, link queries, members or several presets of one dataset as independent
replications of the method effect. Any across-dataset average needs fixed
recipe weights and dataset-level uncertainty; the many seed fits do not create
many independent datasets. Select one final protocol before unused confirmation,
preserve all development alternatives and account for tested contrasts. A
positive interaction, lower auxiliary loss, latent spread or smoke/checkpoint
pass is not confirmation of predictive benefit.

See `BN_AUDIT.md` for the requested direct source audit of private running state.
Its v1 bypass/active-checkpoint hazards were reported to root; root subsequently
reported a v3 repair and passing zero-step CPU/native checks. Those historical
findings are not attributed to v3 without a separate review. `SOURCE_READY.json`
distinguishes prepared source from missing task binding and actual qualification;
none of its pending cells is added to an execution count. `SOURCES.json` pins the
versions read. No requested source was modified by this audit.
