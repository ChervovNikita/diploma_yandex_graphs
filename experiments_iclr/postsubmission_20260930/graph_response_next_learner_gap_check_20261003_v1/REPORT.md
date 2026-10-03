# Structural response diversity: a limitation and one useful next learner comparison

3 October 2026. Bounded saved-literature and analytic review. No predictive experiment or original-score replay was performed by this child.

## Decision

**Do not add a new graph-response loss to rescue the current PolyFormer transfers. Move first to the authored Amazon Polynormer-r release and the already specified paired comparison of boundary GNNM versus an independent native ensemble.** This is a concrete, attributed learner change. It tests whether sharing retains useful prediction on a stronger sourced backbone before spending intervention compute. Further PolyFormer tuning is not the recommended next step.

The parent reports replayed mean validation accuracy of 42.7024% for source defaults and 44.0851% for the Roman transfer, with the Roman transfer winning all three blocks. It also reports Roman validation NLLs 2.437, 3.932 and 3.080 and fit accuracies 78.6–83.7%. These are parent-supplied context, not child-recomputed or independently verified results. Accuracy selection chooses the Roman transfer among these two recipes; it does not establish the competent base required to interpret a response-diversity intervention. There is no directly comparable published threshold for this 80%-of-official-TRAIN adaptation. The source-authored alternative supplies a fixed Amazon recipe, not a promised score.

**No novel diversity objective or predictive guarantee is established.** The native-agreement witness proves that ordinary full-feature FoRDE can miss a declared structural response. It does not show that the missing response is useful, that DICE must miss it, or that the existing squared-cosine penalty can exploit it. Two additional limitations make that distinction precise: native risk can stay exactly fixed, and the proposed normalized penalty can be completely flat on the witness.

## What was reused and actually read

Read first: `graph_structural_response_gap_literature_20261003_v1/REPORT.md`, its theory and prospective diagnostic, and `literature_memory/index_v35` structure/accounting and selected relevant records. Relevant existing context included the conditional candidate, closest-prior report, corrected DICE/FoRDE v2 source specification, native-source v4 report, pilot proposal, loss-matched ensemble-theory report, and Amazon baseline-context report/hyperparameter/scoping records. The index's 149 conclusion records are not full-paper-read totals.

**Fresh public queries: 0. Fresh primary method scopes: 0. Full-paper reads: 0. Author-code audits: 0.** A new broad search would duplicate already resolved operator ancestry. Exact retained primary excerpts were reused for GNCL, loss-matched diversity, function-space repulsion and interpolated spectral augmentation; the previously inspected Polynormer Amazon `run.sh` line was retained verbatim. Reading these bounded saved quotations/source lines is recorded as retained excerpt exposure, not a newly read paper. `READ_SCOPES.json`, `INPUT_BINDINGS.json` and `PUBLIC_SOURCE_EXCERPTS.json` preserve the limits and hashes.

## The witness does not establish learner utility

For the saved two-node native averaging operator `P0`, signed features `X=(1,-1)^T`, and logits `(z_m,0)`,

```
z_m(P,X) = s X + b_m (P²-P) X,
P_t = (1-t) P0 + t I,
z_m(P_t,X) = [s + b_m(t²-t)] X.
```

At native `t=0` and complete deletion `t=1`, every member has the same prediction and full-X Jacobian `s I`. At `t=1/2`, different `b_m` yield different probabilities; the native structural derivative is `-b_m X`. This is an observable response difference that a native feature-gradient kernel can fail to identify.

However, native predictions are identical for **every choice of b**. Native mean-logit pooling, mean-probability pooling, every member native CE and every native task score remain identical. Even away from native, mean-logit predictions depend only on mean(b); changing contrasts with that mean fixed cannot alter the served logit function. The witness therefore supplies no native-error complementarity at all. It is a discriminator of measurements, not a sufficient mechanism for better prediction.

Locally, if `F(theta)` is the native served predictor and `T(theta)` is the structural response map, a direction `v` can satisfy `J_F v=0` but `J_T v!=0`. Response changes along that direction have no first-order native prediction effect. In the witness they have no native effect at any step size. For a generic descent step on response penalty D, the first-order native loss change is `-eta <grad L_native, grad D>`; that inner product can have either sign or vanish. Existing TRAIN-control CE guards can limit deterioration on their declared cells, but do not force a positive native gain or protect heldout labels.

## The normalized squared-cosine loss can be blind too

At one nontrivial witness probe, write the four flattened binary-probability differences as `r_m=d_m v`, with `v=(1,-1,-1,1)` up to ordering. Appending the complete-deletion response appends zeros. Any fixed linear group-centering operator either annihilates v or preserves collinearity. For nonzero retained responses,

```
U_m = sign(d_m) normalize(v),
<U_m,U_n>² = 1.
```

Thus the saved squared-cosine D remains maximal and constant as b varies within this family; its gradient along b is zero away from a zero-response boundary. Normalization discards magnitude and the square identifies opposite directions. Reference norm bands may restrict b but do not make this penalty rank or reward the witnessed response differences. If centering annihilates v, the norm floor instead rejects/inactivates the response. This is an algebraic limitation, not a production grouping/coverage claim.

The statement applies to the saved native/interior/complete-deletion discriminator, not to every arbitrary concatenation of nonlinear probability-response curves. In this degree-two witness, symmetric interior points t and 1-t are exact duplicate functional probes because `t²-t` is equal at both. Multiple probes must not be counted as independent evidence merely because their edge weights differ.

Replacing the squared kernel by an RBF kernel could distinguish some opposite response vectors, but this is already an attributed functional-kernel control in the closest-prior packet. It also does not change the native-risk counterexample. No new loss is proposed here.

## Label preservation, gauges and exact probes

The normalized lazy family has an exact degree-preserving weighted realization. For native weighted adjacency A with degree D, use `A_t=(1-t)A+tD`. Row normalization gives `(1-t)P+tI`; symmetric normalization gives the same identity for the symmetrically normalized operator. For a native graph with unit loops, apply this construction to the complete native adjacency including those loops. Off-diagonal weights shrink while diagonal weights grow in proportion to degree. Ordinary fixed-loop edge attenuation with fresh normalization is generally a different family. In the two-node witness its fixed-loop edge weight is `(1-t)/(1+t)`.

This operation preserves nodes, features, degrees and the original operator's invariant eigenspaces. It **does not certify preservation of the data-generating class label** or causal meaning. Native target labels can be held fixed as a declared supervised augmentation convention; that is not an invariance theorem. On a heterophilic task, reducing propagation can remove useful evidence or change frequency weighting. Probe CE, response energy bands and TRAIN-control guards do not settle that question.

Probability responses are invariant to function-preserving hidden transformations, classifier-nullspace changes, reciprocal shared/private parameter rescalings and additive class-common logit shifts at each graph. Raw factor distances or norms do not share these invariances. If logit probes are used, subtract the class mean or use fixed class-logit contrasts. Conditional centering intentionally discards each group's mean response and may discard useful common class evidence. Squared normalized responses additionally identify positive scaling and response sign; these are properties of the chosen statistic, not intrinsic predictor equivalences.

For a frozen graph-independent H0 and cached `S_j=P^j H0`, exact token probes are

```
P_t^k H0 = sum_(j=0)^k binom(k,j) (1-t)^j t^(k-j) S_j.
```

No fresh sparse propagation is mathematically needed for this declared family. J selected probes cost a direct `O(J B H K²)` dense transform for B selected rows, plus every member/head forward, storage and backward. A row-local nonlinear PolyFormer head can consume these exact tokens; its output is not necessarily a degree-K polynomial. Only a fixed-readout linear degree-K response curve is identified by K+1 exact distinct points. Identifying coefficients additionally requires visible nonstationary modes. Interpolation conditioning, floating-point parity and arbitrary-edge sensitivity are unqualified.

All partial tokens remain in the span of the native polynomial bank. They supply no new graph information or new linear native inference capacity. Their possible benefit is changed regularization/evidence allocation. This overlaps the retained filter/order allocation and spectral-view literature.

**The exact full-predictor shortcut does not transfer to Polynormer-r.** Its local GAT scores, multiplicative/ReLU states and later global projections/reductions evolve along each member's trajectory. Intervening on graph structure requires qualifying and paying for those complete trajectories; an input-token binomial transform cannot substitute for them. Parameter sharing alone does not make graph-dependent states reusable.

## Ensemble error bookkeeping must match the actual pool

For mean-logit serving, with `p*=softmax(mean z_m)`, the retained exact identity is

```
mean CE(z_m,y) = CE(mean z_m,y) + mean KL(p* || p_m).
Delta pooled NLL = Delta mean-member NLL - Delta native ambiguity.
```

The geometric-probability centroid and its loss-matched decomposition are established prior. A structural-response cosine statistic is not the native ambiguity term. In the witness the native ambiguity is zero regardless of structural diversity. A larger native ambiguity can also be offset by worse members.

The proposed Polynormer comparison uses arithmetic **probability** pooling. Its CE Jensen gap is `log(mean p_m(y)) - mean log(p_m(y))`, which depends on the true label and differs from the reverse-KL mean-logit term. For Brier score, `Brier(mean p_m)=mean Brier(p_m)-mean ||p_m-mean p_m||²`. Report these endpoint-appropriate quantities; do not import the geometric-centroid formula into the probability pool. In the native-agreement witness both pools and both decompositions still yield identical native risk.

## Overlap and novelty boundary

| Prior | Relevant established operation | Remaining boundary |
|---|---|---|
| Corrected native FoRDE | Selected true-class raw-logit full-X response, normalized kernel and repulsive pullback | A graph input coordinate is different from X; the witness shows blindness of this chosen native-X object. It does not make generic response repulsion new. |
| DICE | Label-conditioned feature redundancy with a learned estimator; invariant to invertible representation transforms | Squared response cosine is not conditional MI. No universal DICE blindness or independence guarantee is established. |
| GNCL and loss-matched diversity theory | Trade-offs between individual and ensemble loss; loss/pool-specific decompositions | Reducing a graph-response penalty alone says nothing about native NLL. Direct native pool/member loss trade-offs are already prior. |
| Function-space repulsive ensembles | Kernels on finite function evaluations and chain-rule parameter pullback | The saved D is exactly degree-two functional energy after its response transform. CE/private adaptation and guards do not inherit posterior-sampling guarantees. |
| Spectral/filter augmentation | Existing order allocation, structural/spectral views and interpolated spectral embeddings | Lazy-walk probes keep eigenspaces fixed and exactly reuse a polynomial bank. They are a declared efficient response family, not a new spectral primitive or label-preserving augmentation theorem. |
| BatchEnsemble/GNNM and Polynormer-r | Shared native weights plus boundary factors; complete member local/global paths | The empirical quality/cost of sharing under the adapted Amazon fit protocol remains testable. No globally absent architecture, new factor primitive or superiority result is claimed. |

Unresolved GENN and AAAI counterfactual-regularization access gaps remain unresolved; failed access is not evidence of novelty. No fresh source resolves them in this packet.

## One paired next test, with no loss search

Retain **exactly the existing Amazon baseline-context pair**: M4 boundary GNNM Polynormer-r versus four independently trained native Polynormer-r models. Do not add a response, contrastive, FoRDE, DICE, GNCL, router or temperature objective to this test. It is the necessary competent-backbone/sharing screen before an attributed response-regularization study, not a test of superiority over those methods.

Use the pinned ReLU author release, fixed Amazon command and inherited defaults: 512 effective width from 256×2 heads; 10 local/1 global layers; 200 local+2,500 global updates; Adam lr .001/weight decay 0; input/local/global dropout .2/.3/.3; learned beta, shared Q/K, pre-LN false; no scheduler/early stop. Preserve raw features, native graph preprocessing, official splits 0/1/2 and existing fit/control hash roles; blocks 17/29/43 and independent member seeds `block+1009*m`. Fit-only CE, strict official-VAL accuracy selection, native local checkpoint/optimizer restore at global transition and explicit serialized stage flag are required. Use arithmetic probability pooling in both arms and the already described selector differences.

Source/runtime parity and complete stage/checkpoint/resource custody must precede any launch. The boundary layout is a saved unexecuted draft, not an admitted efficient implementation. Charge all preparation, twelve independent fits, three GNNM fits, every member local/global trajectory, peak memory and serving work. Shared parameter counts do not prove wall-time savings. This child launches nothing.

The primary recorded contrast is paired TRAIN-control NLL at the selected checkpoints, with per-block/mean values; control accuracy/Brier/member NLL and actual quality-versus-paid-cost accompany it. Official TEST remains untouched. If the independent source models remain weak, the competent-base premise remains unresolved. If the independent ensemble is competent but GNNM is materially worse without compensating measured costs, sharing—not response diversity—is the immediate failure to fix. If GNNM matches or improves predictive quality with measured cost savings, a shared source base becomes plausible; no graph-response utility follows automatically. No published-score threshold or post-outcome recipe rescue is supplied.

This one comparison changes both the backbone and sharing context relative to current PolyFormer transfers. It does not isolate which backbone feature explains a gain. It reuses an already fixed source comparison rather than adding a tuning grid. A later structural-response test would need a separate frozen pair on that competent base, complete graph-probe trajectories and an actually qualified standard comparator; it is not specified as another fit here.

## Limits

The packet preserves existing packets/indexes and supplies no launch, manuscript or verdict authority. It performs no GPU/remote execution, dataset/checkpoint/label access, original-score recomputation, numerical model import, installation, PDF compilation, scientific fit or subagent call. Source excerpts and prose/algebra were inspected; stdlib file hashing and JSON verification were used. **Disposition: retain the measurement limitation; move to the already sourced fixed backbone/sharing pair; defer response regularization and novelty claims.**
