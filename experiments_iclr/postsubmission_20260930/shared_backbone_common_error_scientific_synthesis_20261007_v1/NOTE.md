# Scientific value of the four-path correction

**Run the fixed v2 comparison as a test of learned common-error repair. Its main scientific risk is redundant correction or displaced errors, not small repeated-forward differences.** Keep one competent selected native backbone, give the four private paths genuine raw-feature and neighborhood computation, and judge the final served predictor against its capable controls. The architecture has close prior; its useful quality contribution remains untested.

This note reuses the saved closest-prior P note and completed scientific summaries. It opens no partial initialization/growth results, raw prediction payloads or ongoing donor outcomes. No remote compute, source change, new reviewer, literature retrieval, novelty claim or acceptance claim is made. It adds one prospective follow-on hypothesis and does not modify a fixed study.

## What the completed evidence says

| Completed evidence | Scientific implication |
| --- | --- |
| Amazon selected VALID diagnosis: shared mean members52.24% versus independent52.59%; pools52.40% versus53.18%;93.60% of shared pooled errors were wrong in every member | Member strength and useful pooling both matter. Selecting an existing member cannot solve an all-members-wrong decision; learned corrections are a plausible direction. The selected-state comparison does not establish a causal information deficit. |
| Citeseer complete39 study: live transfer MRR0.2431 versus ordinary shared0.2970 and capable single0.2797; live has more pooling benefit but much weaker members | Disagreement or a larger pooling increment can accompany a worse predictor. Retain ordinary shared and capable-single controls. |
| Citeseer completed common-negative analysis: common slots are74.74% of ordinary shared pooled strict errors versus45.23% for jointly trained native four; live lowers overlap while expanding its union of mistakes | Common errors are a real selected-state bottleneck; lower overlap is not the target. Existing alternatives correctly rank some of those same slots, but architecture/selection differences prevent causal attribution to sharing. |
| Frozen Amazon G0 assessment: candidate41.98%, independent46.30%, single45.37%; graph-free has the same accuracy | That configuration failed and stays closed. Softer probabilities do not rescue classification failure. Different16-SGD versus2300-Adam horizons and reference competence limit any broad conclusion about graph conditioning. |

These completed node/link studies motivate a new training comparison. They do not predict WikiCS outcomes or prove that one new GAT block is the missing operation.

## The current executable proposal

Retain v2 exactly: official WikiCS split0; TRAIN580; complete VALID5274; seeds17,29,43; one selected competent native1100 backbone per seed. Capture its serving H0,z0, freeze it, and construct four independently drawn width512 private paths:

`u_m=H0+V_m x; z_m=z0+U_m B_m(u_m,A)+c_m; p_m=softmax(z_m)`.

Each B_m has full private GAT/root/h/gating/LayerNorm parameters; U_m,c_m start at zero. The served result is the fixed mean of four probabilities. Train each path100 times in order0,1,2,3 with own CE plus twice pool Brier, using deterministic detached peers and the active path's training dropout. No disagreement reward, routing target or hidden teacher is needed. The same source already specifies all paired controls and complete acquisition/continuation horizons; do not replace them with a weak copied ensemble or a single narrow head.

There are two reasons this might help: raw x and a new private neighborhood transformation can correct a bias left by the selected backbone, and later paths can respond to residual probability errors after earlier correction. There are also two reasons it may fail: all paths see the same labels/graph and can learn redundant functions, and freezing a biased trunk plus previously trained paths restricts later joint repair. Own CE protects the objective from being pool-only; it does not guarantee competent members.

The fixed mean also dilutes early correction. If the donor favors wrong class k over true class y by probability margin d, one changed path can contribute at most+1 to y-versus-k while the three unchanged copies contribute−3d. **For d>1/3, the first path cannot repair that pooled ordering regardless of its capacity.** With t changed paths, a necessary bound is `t>(4−t)d`. Thus a confident common mistake may require several paths to move together. This is an analytical constraint, not a newly measured dataset fact. It requires no stagewise VALID scoring or source change. Use existing TRAIN traces and the already prescribed final error flows.

## What the fixed seven arms actually answer

| Arm | Real question | Prior overlap and limit |
| --- | --- | --- |
| E_stage | Does the complete proposed staged correction improve useful served accuracy? | Known neural/graph boosting and shared-trunk ancestry; a positive result would support this composition, not establish a new ensemble principle. |
| E_joint | Does ordered permanent freezing outperform interleaved per-route correction with the same cache/draws/update counts? | A schedule comparison. Detached peers and a frozen backbone mean it is not fully corrective GrowNet or live end-to-end joint ensemble learning. |
| E_own | Is the pool-residual term useful beyond learning four competent private functions? | The closest Deep Sub-Ensembles baseline: learned frozen trunk, same-dataset private learners, probability mean; raw-x/GAT/z0 anchoring are the adaptations. Own-only paths are independent fits, so their order supplies no residual specialization. |
| S_continue | Would ordinary native training suffice? | Strong practical baseline with live end1100 model/Adam/RNG and400 further epochs. It does not isolate added private capacity or match correction compute exactly. |
| S_paths | Does a capable single using all four paths and a live native backbone match the gain? | Essential capacity/optimization alternative, motivated by residual-network ancestry. Its nonlinear decoder and live base differ from the bank, so a bank win is not a proof that ensembles are necessary. |
| I_native | Does conventional independently acquired four already do better? | Essential quality and paid-cost reference; four1200 native fits are genuinely distinct. Its acquisition diversity differs from one common donor. |
| U_stage | Does sharing one donor improve quality relative to the same correction on four independently acquired donors? | Tests practical value of common-backbone reuse under this design; the contrast includes donor diversity, not a pure causal estimate of tying parameters. |

Hydra overlaps the shared body/private heads and two-phase growth architecture, but trains a distillation student against separately acquired teacher-member targets. It is not equivalent to the candidate's ground-truth CE and live pool residual. Deep Sub-Ensembles is closer: it already trains a competent full model, freezes its trunk and learns further private task networks with probability pooling. TreeNets already supplies shared initial/full private later layers; GrowNet already supplies raw/prior hidden features and sequential residual learners with fully corrective refinement. AdaGCN/BGNN/B3F-GNN supply graph boosting/error specialization. The narrow untested difference is the complete raw-x/H native graph residual and probability-pool update, with the recent unread AdaGNN/GENNN/FAGEL method gaps retained. None of the seven controls can certify historical novelty.

At the fixed final endpoint, use the complete population and the existing prospective rule: at least0.5pp mean gain over the strongest required quality reference, positive paired-seed contrasts, served NLL and member-competence safeguards. Do not select a seed, stage, class or loss after failure. For donor baseline B and candidate C, report counts `B wrong→C correct`, `B correct→C wrong`, and wrong→different-wrong; net accuracy is `(repairs−introduced errors)/5274`. Also report the same flows on donor common-competitor errors, every member's accuracy/NLL/Brier, and pool gain over mean member accuracy. All four initial bank members equal the donor, so their initial all-wrong cohort is one donor-error cohort, not four observations. Calibrated unchanged mistakes, reduced overlap, or repairs canceled by new errors do not establish useful repair.

## One follow-on hypothesis: aligned new messages matter

**Hypothesis:** after a competent trunk is frozen, correct original-graph neighborhood information in the new private paths produces net common-error repair beyond equally strong raw-x/H pointwise capacity. The fixed seven arms do not answer this graph-specific question: all residual banks contain the added graph block.

Freeze a separate finite mechanistic comparison on the same representative WikiCS split and paired seeds17/29/43. Use three conditions, nine correction cells total:

1. The existing E_stage native GAT residual bank on the original graph.
2. A full width512 pointwise residual bank: retain raw-x/H input, h/root/gate/LayerNorm/output and replace only neighbor aggregation with a full512×512 local affine message transform. It is a nonlinear private learner, not a linear classifier. Report the removed attention-vector parameters and actual cost; do not pad with unused weights or pretend exact function-class equality.
3. The original full GAT residual bank on one fixed label-blind degree-preserving rewiring per seed, shared by all four private paths. Preserve the source graph convention, edge count and existing self-loops. The native donor/cache remains from the original graph. Rewire only private-path edges, with a fixed bounded edge-swap procedure before any endpoint predictions; labels and error locations never enter that procedure.

Keep the common cache, initial corresponding raw/h/root/gate/output and message-transform matrices, private seeds, fresh Adam, dropout, loss coefficient, stage order and100 updates/path fixed. Copy the reference initial tensors and route RNG streams into the new controls; initialize fresh Adam rather than carrying trained private moments. Generate rewiring with an isolated RNG stream. No rank/depth/learning-rate/rewiring-strength search. Reuse the three existing E_stage results only when their full contracts match; otherwise the study has nine freshly paired correction fits. A compatible reuse means six added cells and no new donor acquisitions. Charge actual compute; this is equal-update mechanism comparison, not equal wall time.

This follow-on is worthwhile only if the fixed v2 study shows useful aggregate quality and competence. If that study fails, stop its promotion rather than use these controls as rescue. Before execution, require original-graph correction to gain at least0.5pp mean accuracy over the stronger of pointwise/rewired controls, win both comparisons in all three paired seeds, and improve net donor-common-error repair, with no worse mean served NLL and the existing competence safeguards. If pointwise matches, attribute gain to generic private nonlinear capacity; if rewired matches, aligned extra neighborhood information is unestablished. If gain is only calibration, reject the accuracy hypothesis. A favorable same-graph result remains development, requiring a separately frozen confirmation before generalization claims.

## Saved evidence consulted

The evidence below consists of completed summaries and scoped literature notes; no partial ongoing quality artifact was opened.

- [Closest-prior P note](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/staged_private_graph_residual_closest_prior_P_note_20261007_v1/P.md>)
- [Completed Amazon error diagnosis](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/amazon_polynormer_valid_error_analysis_root_interpretation_20261005_v1/REPORT.md>)
- [Completed Amazon G0 decision](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/amazon_G0_complete_comparison_result_root_disposition_20261006_v1/CONCLUSIONS.md>)
- [Completed Citeseer39 decision](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/shared_private_transfer_complete39_result_root_disposition_20261006_v1/CONCLUSIONS.md>)
- [Completed common-negative diagnosis](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/shared_private_transfer_complete39_common_negative_owned_operation_20261006_v1/RESULT_CONCLUSION.md>)
- [Completed prediction-aggregation outcome](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/amazon_polynormer_logits_graph_moment_complete_outcome_root_20261005_v2/REPORT.md>)
- [Completed private-frame comparison](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/citeseer_frame_complete_root_adoption_20261005_v1/REPORT.md>)
- [Saved shared/private mechanism assessment](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/shared_private_common_error_mechanism_closure_20261006_v1/REPORT.md>)

This is one compact scientific proposal. It creates no new fitting job, modifies no fixed source or result, and grants no confirmation or manuscript claim.
