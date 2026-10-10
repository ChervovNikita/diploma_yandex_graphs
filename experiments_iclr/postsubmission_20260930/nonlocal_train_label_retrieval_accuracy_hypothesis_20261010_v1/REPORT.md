# One accuracy hypothesis: learn nonlocal TRAIN-label retrieval

10 October 2026. Source assessment, stateless helper and thin extension of the
existing Family fit/selector. Root admitted one fixed development pilot; this
packet performs no model import, numerical execution, data access or fit.

## Why the shared wrapper remains unresolved

Complete families have not delivered a protected advantage over capable,
equally processed single and genuine independent predictors. The latest
graph-reliability closure gives shared gains of .50/.78/.03 percentage points
over native SAGE/GCN/GAT, while equally processed ordinary I4 is tied or stronger
(root-supplied complete results). Earlier histories show both weaker members and
correct member predictions lost during pooling. Geometry, gradient conflict or
sharing alone therefore does not explain all failures. A convex same-node pool
cannot repair a truth class below the same rival in every input vector; it also
does not follow that every all-wrong bank is unrepairable. No general wrapper
impossibility has been established.

**Hypothesis:** directly supervising nonlocal class retrieval through the live
native hidden representation can acquire correct class evidence on common-rival
errors and improve served predictions. Native own CE retains its competence
objective; native/member weaknesses remain visible. This changes the
prediction operation beyond reweighting the old vectors. It uses the same TRAIN
supervision, made explicitly available as prediction-time class values.

## One fixed learning recipe

Keep the native graph, backbone, final head and existing shared/private factors.
Capture each route's post-normalization hidden H immediately before its final
head through `common_routes.NativeModelAdapter.native`. From the same current
label-free full forward, normalize H rowwise and compute

`a_m(i,j) = softmax_j(cos(H_m(i),H_m(j)) / .1)`;
`r_m(c|i) = sum_{j in S: y_j=c} a_m(i,j)`.

S contains every visible TRAIN anchor, regardless of original graph distance.
Class values are literal one-hots; there is no learned label embedding, teacher,
extra projection, new parameter or changed graph. Each update draws **one common
half-TRAIN Q** for all routes/arms, approximately half per class, with the fixed
odd-count quota rule in `label_memory.py`. Every class must have at least two
TRAIN examples. All Q labels are removed from every S before retrieval.

Train `L = mean_route CE(native, all TRAIN) + mean_route CE(r, Q)` with coefficients
1 and 1. Direct retrieval CE avoids attenuation by an already confident native
TRAIN posterior. Gradients through query and support H update the existing shared
body and upstream private factors; the final native head receives native CE.
At serving S is all TRAIN: use `.5 native_probability + .5 r` per route, then
mean route probabilities. Temperature, loss weights and mixture are fixed;
there is no rank, coefficient or temperature search.

H receives only X/graph inputs. Retrieval is one terminal query-to-anchor read:
no retrieved labels enter H, support states, another route or a later message
pass; there is no cached label state. Common Q removes direct label lookup. The
native all-TRAIN CE still makes Q representations supervised in-sample; this is
transductive episodic fitting, not withheld-label generalization. Stratification
uses TRAIN truths and carries no uniform-mask or unbiasedness claim.

Sharing changes estimation: four retrieval losses jointly train one slow feature
geometry through four existing private views. Separate I4 geometries have the
same information. Neither coupling nor anchor access guarantees complementarity.

For query truth y, standard differentiation gives
`d[-log r_y]/ds_j = a_j - a_j 1[y_j=y]/r_y`: correct-class anchors are pulled closer
and others suppressed. With .5/.5 serving, retrieval overturns a native wrong
rival only when its truth-minus-rival margin exceeds the native rival-minus-truth
margin; every rival still has to be beaten. These are arithmetic, not new results.

## Closest prior and actual difference

| Primary/source scope | Established operation and comparison |
| --- | --- |
| [NCA, NeurIPS 2004 publisher abstract](https://papers.nips.cc/paper/2004/hash/42fe880812925e520249e808937738d2-Abstract.html) | Supervised metric learning for stochastic leave-one-out nearest-neighbor classification is established. The original abstract states a score objective; our log class-mass loss is the Matching Networks version in this family. No new full NCA method read is claimed. |
| [Matching Networks, 1606.04080v2 §2](https://arxiv.org/html/1606.04080v2#S2) | Exact softmax-cosine attention and weighted support one-hot classification; jointly learned embeddings and support/query episodes. This is the retrieval primitive here. |
| [TPN, 1805.10002v3 §3](https://arxiv.org/html/1805.10002v3#S3) | Shared support/query embedding, learned similarity graph, multistep label propagation and end-to-end query CE. Learned nonlocal class evidence is already established. |
| Saved UniMP author source / MPNP method / two-hop label report | Masked TRAIN contexts, joint label-aware graph fitting or conditional graph prediction already exist. These are competent label-aware reference obligations after a positive screen. |

The untested extension is utility of terminal all-anchor retrieval through the
current shared native GNN metric. The failed typed42 label recipe used graph-local
label-conditioned factors; the saved two-hop proposal propagates a fixed native
representation's label values. Neither tests this live, nonlocal metric update.
No new primitive, novelty or predictive gain is claimed.

## Decisive complete comparison and refutation

Use all three native backbones; complete every arm before interpreting outcomes.
Root fixed seeds7301/7403/7507 and four new arms: shared4 retrieval; the same
shared4 retrieval with attention detached; capable ordinary M1 retrieval;
genuine ordinary I4 retrieval. Reuse the root-bound closed unchanged shared4 and
raw native M1/I4 archives. All retrieval arms receive identical TRAIN values,
Q draws, temperature, losses, serving and fitting/selection opportunities. The
detached arm trains native CE and tests retrieval without metric gradients. Keep
the existing full schedule and complete VALID selector; no subgroup chooses fits.
`run_retrieval_family.py` preserves the existing native Family lifecycle and
selector, adds the helper call and saves/restores the separate common-query RNG.
It records process-wide forward/work counts and native/mixed endpoint arrays.

The authoritative operational gates are root's
`nonlocal_label_retrieval_pilot_root_20261010_v1/DECISION.md`, superseding this
scout's earlier proposed zero-NLL-deterioration criterion. For each backbone,
live shared4 must gain at least .2pp mean accuracy over same-information I4 and
.1pp over M1/detached shared4; all three paired changes are nonnegative with at
least two positive. NLL deterioration is bounded by .02 mean/.05 each seed versus
these controls. Unchanged shared4 and ordinary/factorized raw M1/I4 provide context.
Each mixed arm selects its checkpoint by whole-VALID mixed predictions; the
unchanged archives used native predictions. Native readouts are reported at the
**mixed-selected** state. These are complete-recipe contrasts, including selection;
they do not isolate the optimizer update from checkpoint selection. Genuine I4
keeps the existing individual-member selection policy before pooling.

Report full-population served accuracy/NLL, mean/worst native and served member
quality, any-correct coverage, common-rival repairs and introduced errors against
the unchanged anchor. A native-only selected-trajectory readout distinguishes
better native learning from terminal memory utility. Repairs outweighed by harms,
failure of the frozen quality/risk gates, or no live-over-detached benefit refute
the proposed useful acquisition operation. Weaker individual members alone do
not exclude a true pooled win. If same-information
M1/I4 explains the gain, the broad shared-wrapper objective remains unmet.

36 new banks require 63 fitted native bodies and 117 member paths per
family-wide forward, plus the reused closed references; the helper needs
**zero extra native forward calls**. For
TRAIN580/Q290/S290/D128/C10, extra similarity work is 10,764,800 MACs per route;
complete VALID5274 uses 391,541,760. No sparse edge pass is added. Serving chunks
of 512 bound each route's score tensor at 1,187,840 float32 bytes; backward retains
more than this, and native activations dominate additional unknown costs. Actual
wall time/peak memory require root qualification; no GPU-hour forecast is made.

This is encountered-development research. Original paper scores and TEST custody
remain unchanged; a positive screen still requires competent source-native label
references and unused confirmation.
