# One held acquisition hypothesis: neighborhood distributions

10 October 2026. **Retain one inactive, attributed alternative to the running
class-state decoder: a compact distributional correction on native hidden
neighborhoods.** This reuses the unexecuted 5 October projected-quantile proposal
and saved PNA/FSW primary scopes. It is not presented as a new discovery or a
novel aggregation principle. No implementation, fit or extra launch is admitted.

## Why the portable wrapper remains difficult

The completed studies now distinguish three requirements that a successful
shared wrapper must meet together: learn competent predictors, acquire useful
different correct decisions, and retain them in the deployed pool. A fourth
requirement is that the advantage survives capable singles and genuinely
independent ensembles receiving the same ingredients. Neither parameter savings
nor a larger disagreement statistic supplies these results.

The stronger factorized single reproduces most of the older PubMed accuracy
clue. A fixed member's fast factors can fold into native affine weights; the
factorized optimizer can still learn differently. This makes reparameterization
a serious alternative explanation to ensemble learning. On native WikiCS,
shared SAGE's accuracy clue fails to transfer to GCN/GAT, and same-information
independent banks remain stronger on important comparisons. Private masks,
rotation, exchange and bootstrap can change predictions yet weaken members,
lose useful alternatives, or introduce enough harm to cancel repairs.

The literature has yielded useful clues, but they identify operations worth
testing rather than a generally beneficial wrapper:

| Complete evidence | Useful implication | Boundary |
| --- | --- | --- |
| A correct shared member exists on a majority of independent-correct/shared-wrong cases; graph scoring improves shared SAGE/GCN | Graph context can help retain existing alternatives | Equally graph-scored I4 remains competitive or better; this is a serving clue |
| Factorized singles explain much of the old shared gain | Qualify capable factorized singles before assigning a plurality/sharing explanation | Parameterization/optimizer effects are not identified causally |
| Live retrieval adds some terminal accuracy but repairs zero native common strict false rivals across all nine live shared banks | A common-error acquisition mechanism is still needed in these conditional readouts | Retrieval is not universally incapable; selection/stopping changes across models |
| PNA and projected distribution embeddings explicitly retain more than a neighbor mean | Neighborhood distributions offer a different evidence interface | No current error has been proved to require those statistics |

For a single false class c that every route ranks above truth y, a nonnegative
class-shared mixture of their same-node probabilities still ranks c above y.
This explains one pooling limit. It does not establish that every all-wrong
case has the same rival, nor that a distributional correction will be right.

Source preparation and repeated encountered-development studies do not establish
portable accuracy. The duration of the search does not relax the capable-reference
or unused-confirmation requirements. The practical response is to use complete
error evidence to choose one distinct computation, rather than repeat a failed
loss or certify literature absence.

### Confidence is a separate measured problem

The sealed complete graph-reliability study already answers much of the SAGE NLL
question; no new scoring or temperature fitting is needed. Its global temperature
is positive and applied to each member's log probabilities **before** probability
pooling. It preserves every member's class order, though it can change the pool's
order. It is not a temperature applied after the completed pool.

| SAGE bank/readout | Mean accuracy % | Mean NLL |
| --- | ---: | ---: |
| Original shared4, raw probability mean |80.571356|1.001493|
| Same shared4, existing global-temperature rule |80.596638|0.682065|
| Ordinary genuine I4, same global-temperature rule |80.021489|0.661398|
| Factorized genuine I4, same global-temperature rule |80.091013|0.659891|

For shared4, calibration leaves any-member correct coverage exactly unchanged
at 13,740 summed seed readouts. It repairs 54 native pooling losses and introduces
50 harms, giving only four extra correct readouts. It repairs **zero of 1,955
native common strict false-rival readouts**. These are summed repetitions on the
same nodes, not independent observations. The almost unchanged accuracy and large
NLL reduction demonstrate a substantial correctable confidence/probability
component without acquisition of new correct member alternatives. A smaller NLL
disadvantage remains against equally calibrated I4. Fitted temperature values,
confidence bins and correctness-conditioned NLL are not in this scalar summary,
so it does not supply a complete overconfidence decomposition or identify its
training cause.

Graph reliability takes shared accuracy to 81.070661% but NLL remains 0.968328.
That separates recovery of already available alternatives from confidence
correction. Likewise, a new distributional head must be judged by **new correct
alternatives and their served net effect**, while NLL tracks its confidence harms.
Lower NLL alone cannot count as successful common-error acquisition. No calibration
rescue or additional temperature grid is appended to this proposal. The study's
folds remain encountered development because base checkpoint selection used all
VALID labels before fusion folds were formed.

## One concrete operation

Use a full, qualified native classifier's current preclassifier route states
`H_m[N,D]` and logits `Z_m[N,C]`. Keep the native factual graph and factor sites.
Let `N_in(v)` be all factual incoming **nonself edge records**, with duplicates
retained. This descriptor support does not alter native backbone support.

Each route learns one direction u_m and one small score map V_m. Initialize u_m
from an independent fixed normal draw, using the existing route seed lineage;
normalize it inside the forward with a declared 1e-8 norm floor. Initialize V_m
to zero. Do not search directions or initialize from outcomes.

```text
a_m(v) = dot(H_m(v), u_m / max(norm(u_m), 1e-8))
t_m(v) = [Q_.25 - mean, Q_.50 - mean, Q_.75 - mean, log(1+d_v)]
S_m(v) = Z_m(v) + t_m(v) V_m                 # V_m is 4 by C
L = mean_m [ .5 CE(Z_m, TRAIN) + .5 CE(S_m, TRAIN) ]
serve = mean_m softmax(S_m)
```

Quantiles and mean use the multiset of projected values at `N_in(v)`. Sort the
d_v scalars and interpolate at fixed position `(d_v−1)q`; positions and edge
weights are not learned. Empty neighborhoods have all four descriptor values
zero. Degree-one neighborhoods have zero centered quartiles and degree log2.
All class labels remain loss targets only. Keep gradients through H, projection,
sorting/interpolation and V live. Every route retains full TRAIN supervision.
There is one descriptor pass after each native forward, with no class-state
recurrence, teacher, literal TRAIN-label retrieval, extra native forward, mask
or disagreement penalty. At V=0, scores and the initial native-loss coefficient
match native training. Later competence is not protected by that identity.

This is an adaptation of the saved proposal: explicit zero-start residual logits
and probability pooling replace its enlarged-head/raw-logit serving sketch. The
half-native/half-corrected loss merely retains a native anchor with unit total
coefficient at initialization; it supplies no new learning principle. Each V_m
is private, while the large native body is shared. The running decoder instead
transports predicted class mass through a common class matrix; neither operation
is changed or combined here.

## Acquisition path, strongest collision and expected failure

A distribution signature can change a score margin even when every original
route prefers c: `t_m(V_m[:,y]−V_m[:,c]) > Z_m(c)−Z_m(y)`, also beating other
rivals. This is ordinary residual-classifier arithmetic, not evidence of correct
context. Shapes of neighboring native features may carry information about mixed
local populations that a particular mean interface discards. Native attention,
nonlinear message maps and a capable statistic-rich single may already recover
it elsewhere; no whole-GNN expressivity limitation is claimed.

**PNA, arXiv2004.05718v3 §2–3**, is the direct established alternative: learned
messages aggregated with mean/min/max/std and degree scalers, combined within a
single graph model. It already supplies distribution-aware graph evidence.
**Fourier Sliced-Wasserstein Embedding for Multisets and Measures,
arXiv2504.02544v1 §2.2–3 and selected Appendix A.1**, is the closer projected-
distribution collision. It projects multisets and embeds quantile functions by
cosine-transform samples. It explicitly discusses graph aggregation, cardinality
and isolates, and explains why finitely sampled direct quantiles are noninjective.
The proposed three fixed interpolated quartiles are not its Fourier embedding
and inherit none of its injectivity, bi-Lipschitz or Wasserstein guarantees.
Fixed interpolation at fixed neighborhood size is piecewise linear in the
projected values; ties and complete high-degree sorting still need source/runtime
qualification before root can admit a fit.

The sharing hypothesis is a finite-sample regularization hypothesis: a common
body can learn a representation supporting several supervised distributional
correctors without paying for four separately learned large representations.
Sharing supplies neither independent observations nor a larger function class.
Private directions can align, smoothing can erase their input variation, or a
single all-signature head can learn everything. The principal expected failure
is that the extra descriptor only changes confidence or fits TRAIN-specific
neighborhood shapes, while correct alternatives remain redundant or harms cancel
repairs. Same-operation I4 can win because its better native states contain more
useful neighborhood evidence. These are decisive falsifiers, not reasons to
weaken the references.

The saved quantile example distinguishing two equal-mean/std/extrema multisets
is a local information illustration. It does not establish that WikiCS labels
depend on quartiles, that three statistics are sufficient, or that real native
hidden neighborhoods exhibit this failure. The older cached raw-std token is a
related unexecuted simpler ingredient; it is not a completed positive result.

## One bounded representative comparison

Only after root closes the current complete decoder family, nominate a later
**full SAGE WikiCS development comparison** on the same declared TRAIN/VALID
roles, width128/depth2 and original 1000-update/300-patience opportunity. This is
chosen for the qualified native hidden interface and direct mean-aggregation
comparison, not from decoder outcomes. It remains an encountered graph. Root
must freeze seeds, stream lineage, exact source, selectors, gain/NLL gates and
cost cap before any new execution. No shortened or favorable subgraph fallback.

Five arms, three paired seeds:

1. **Shared4/private quartile corrections:** the sole candidate above.
2. Shared4 with the same private projection but the stronger matched
   `[mean,std,min,max,log_degree]` descriptor and private 5×C maps. Same graph,
   added receptive field and live training; established PNA-style statistics
   provide a decisive ingredient reference, not a claimed full PNA reproduction.
3. Shared4 quartiles with one common projection direction across routes, keeping
   each private V. Route H_m still differs; descriptors are not asserted equal.
4. Capable factorized M1 with four learned directions, all twelve centered
   quartiles plus degree, and one joint nonlinear residual readout that sees its
   native H as well. Preserve its native head and zero-start residual output.
   One direction would be an inadequate same-information single.
5. **Genuine factorized I4 with the identical quartile operation:** four separately
   initialized/fitted native bodies, each its own direction/V and decoded selector,
   followed by probability pooling. No common parameter or joint selector.

This is 15 banks and 24 independently owned optimizer/body fits; each shared bank
fits one large body. The original plain native shared/single/I4 archives remain
context, not substitute controls. The moment arm tests whether ordinary richer
statistics explain the correction; the all-signature single and genuine I4 test
whether sharing/plurality adds anything. A common-projection match removes the
private-direction explanation. A win over plain shared alone is insufficient.

Evaluate all arms before interpretation. Whole-population served accuracy/NLL,
native and corrected member quality, any-correct coverage, newly acquired and
served alternatives, lost alternatives, strict common-rival repairs, introduced
errors and all paired seeds decide utility. Cohort masks are diagnostic only,
never fit inputs or selectors. A coverage gain without net served quality fails.
Passing this development screen would prioritize complete native-backbone
replication and frozen unused whole-pipeline confirmation; no portable or
manuscript claim follows yet. Failure closes this exact operation without a
quantile-count, head-width, coefficient, norm, direction or sorting grid.

## Compute and next action

At D128/C10, the candidate adds four*(128+4*10)=672 parameters. For N11701/M4 it
adds 4ND=5,990,912 projection MACs per native bank forward, 4E scalar edge gathers
and four complete segmented sorts: `O(4 sum_v d_v log d_v)`. A materialized
four-route projected-edge float32 field at E431206 is 6,899,296 bytes; sorting
indices/workspace, existing native H/autograd, selection and backward are extra.
No runtime, GPU peak or speed benefit has been measured. I4 receives the same
descriptor information and may reuse graph segments; duplicate preparation must
not manufacture an efficiency claim. Full TRAIN/evaluation/backward and every
attempt count.

**One next action:** keep this source-only proposal in the inactive queue; if
root elects it after the decoder closes, first qualify the native hidden boundary,
complete segmented sorting/tie gradients, ownership and actual TRAIN cost for
the fixed five-arm comparison. That qualification is engineering evidence only.
The containing hypothesis and controls must be fixed before quality is opened.

RESEARCH_STATE and the complete retrieval report were read first. The state
snapshot still says the decoder is unlaunched; root's newer message reports launch
at 17:47:10 UTC. This report adopts that status correction without reading any
decoder progress or quality file. Saved rejections of cavity/hop/class-code/loss
renominations remain in force. Zero new primary identities, remote operations,
scientific model/data execution, TEST access, canonical edits or novelty claims.
