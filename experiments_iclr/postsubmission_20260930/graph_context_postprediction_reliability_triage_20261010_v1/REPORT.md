# One graph-context reliability scorer for frozen pooling

10 October 2026. Source review and one prospective known-method utility question. No fit, export, forward or execution admission.

## Decision and changed priority

Retain one fixed post-prediction scorer that weights already computed member probabilities using entropy, margin, consensus and label-free neighbor prediction agreement. Its purpose is to preserve useful alternatives during pooling. It is untested. Every route executes; this adds aggregation work and makes no route-compute saving claim.

Root supplied the following closed-family diagnosis through the parent message preserved in `ROOT_CONTEXT.json`: majority-common-missing fails in all18 comparisons; most I4-correct/shared-wrong nodes already have a correct shared member, with approximately65–69% recoverable fractions for SAGE,74–76% for GCN and57–69% for GAT. Wrong members on SAGE pooling losses have median top confidence .84/.99/.97. The new coherent24 bank restores competence but adds few different correct predictions. This packet has not independently audited those numbers, their denominators or seed mapping, and has not accessed current24 or full9on77 outputs.

This context supersedes the narrow pooling/acquisition priority rationale in the sealed `shared_wrapper_acquisition_priority_decision_20261010_v1`: absence of correct alternatives is no longer the leading explanation for those newly diagnosed nodes. Its failures and measured competence deficits remain intact. The mechanism targets retention where alternatives exist; it supplies no remedy for every shared error.

## The one predictor

Fix an authenticated selected-state bank of M>=2 aligned C-class probability vectors `p_vm`. For each node/member define the leave-one-out peer mean `b_vm=(sum_{j!=m}p_vj)/(M-1)`. Let P be the incoming-neighbor row mean on the exact authorized graph-visible support: distinct incoming neighbors, exclude self-loops on nonempty rows, and use `P_vv=1` for isolates. Keep graph directedness as supplied and add no reverse edges. Define `g_vm=(P p_m)_v` and neighbor peer mean `d_vm=(sum_{j!=m}g_vj)/(M-1)`.

The five scalar inputs are:

1. normalized entropy `H(p_vm)/log C`;
2. top1-minus-top2 probability margin;
3. own-node consensus `p_vm dot b_vm`;
4. own-member neighbor agreement `p_vm dot g_vm`;
5. peer-neighbor agreement `p_vm dot d_vm`.

Use exactly one shared scorer `s_vm=a^T tanh(W f_vm+b)`, with W8x5, b8 and a8:56 parameters, no output bias, member ID or class ID. Set `alpha_vm=softmax_m(s_vm)` and serve `q_v=sum_m alpha_vm p_vm`. Fit mixture NLL plus a fixed0.01 mean `KL(alpha_v || uniform_M)` penalty. This discourages unnecessary departures from uniform pooling; it is not a diversity reward. Fusion labels train q directly. No member-correctness oracle, TRAIN residual, neighbor label, true homophily, hidden state or backbone gradient enters the features.

The fixed prospective fit uses W uniform[-0.1,0.1] from declared seed11709, zero b/a, full-batch Adam learning rate0.01, default betas/epsilon, no weight decay and exactly500 updates. Serve the last update. The zero output a initializes uniform pooling without making the whole hidden layer identical. Width, shrinkage and seed have no proposed search. Root still owns finite numerical qualification and any admission in an existing interface; this memo creates no trainer/source implementation or authorized fit.

Under a common class permutation, entropy/top-order scalars and dot products are invariant, so weights stay the same and q follows the class permutation. Under a member permutation, the symmetric leave-one-out means and shared scorer permute the weights, leaving q unchanged. These statements require aligned class coordinates and do not permit a different class permutation for each member. `PROOF.json` records the symbolic argument.

This is a supervised reliability cue, not a correctness guarantee. High-confidence wrong members and wrong agreeing neighborhoods can suppress the correct outlier. A correct top1 member does not prove the cue can identify it. Conversely, every member being top1-wrong is not by itself an impossibility proof: different rivals can cancel. The exact limitation is a strict common rival: if one wrong class exceeds the true class in every member, no convex weighting can repair that node. The frozen individual predictions and their competence remain unchanged; served quality can worsen.

## Two essential baseline families

Retain native uniform probability pooling as the unfitted anchor. All fitted rules receive the same eligible fusion labels, fixed folds, base bank and assessment opportunity.

| Baseline family | Fixed controls and decisive interpretation |
| --- | --- |
| Global/per-member temperature calibration | Fit one positive shared T and, separately, M positive T_m values, followed by uniform probability pooling: `p_m(T_m)=softmax(log p_m/T_m)`. Fit NLL with the same fusion roles and fixed finite budget; no temperature grid. A positive temperature preserves each member's class ordering but can change the pooled ordering. If this suffices, confidence miscalibration explains the recoverable gain without learned neighbor weighting. |
| Non-graph learned pooling/plain stacking | First use the exact same56-parameter scorer, fit and penalty with P=I: the last two inputs become `p_vm dot p_vm` and own-node consensus. This keeps dimensions/architecture fixed and removes neighboring predictions; repeated coordinates are disclosed. Also retain one ordinary regularized multinomial linear stacker on concatenated own-node member probabilities, `softmax(B[p_v1;...;p_vM]+c)`, fitted with NLL and fixed0.01 mean-square penalty over B and c, zero initialization and the same500-update Adam schedule. It is a capable class-specific own-node decoder and may leave the convex hull. No feature, penalty or depth grid is proposed. |

For temperatures use log-T parameters initialized at0 and the same500-update Adam schedule. Ordinary positive temperature calibration has no graph cues. Candidate/scorer supervision learns mixture behavior; it does not certify per-member calibration. Baselines cannot silently receive fewer labels or a weaker selection procedure. These are two baseline families with fixed operators, not alternative candidate mechanisms.

If the P=I scorer matches or beats the graph scorer, this neighbor-prediction ingredient has no supported contribution. If the plain stacker matches, use ordinary stacking for the measured utility. Better calibration against the raw pool alone is insufficient. Before a sharing or ensemble-specific claim, capable processed single and genuine independent banks need equal fusion opportunity and all acquisition costs; a same-bank postprocessor win cannot supply that claim.

## What was already tested or proposed

**The5October graph stacker is the exact local ancestor.** It froze predictions, formed `p_m-mean_j p_j` and their neighbor summaries, and learned softmax member weights for a probability mixture. It established no predictive gain. This proposal narrows the inputs to class-invariant reliability scalars and supplies a fixed shared scorer after a new retention diagnosis. Those details do not clear methodological novelty or justify renaming the broad idea as a new family.

**Direct12 closed one regularized frozen TRAIN-classifier refit.** Extra unrestricted BE head freedom changed correct counts by0,-1,+1 and mean accuracy by zero; all four refit conditions lowered their corresponding native mean accuracy. Six finite endpoints failed the declared convergence condition. Its exact refit stays closed, with no penalty/solver or private-head extension. It altered each route classifier on TRAIN; it did not test a frozen probability mixer supervised on fusion-development labels. That distinction supports an untested question, not a predicted gain.

**Amazon99 is a genuine negative aggregation result.** All99 configurations completed. Selected shared4 accuracy52.5450487% was below processed single53.1275519% and processed independent4 53.2527628%. Relative to processed single, shared Brier worsened0.023703398 and NLL0.340900739; relative to processed independent, accuracy fell0.7077141pp and NLL worsened0.048939813. Every practical screen failed, and analytic local moments also failed their global/diagonal/stacker comparisons. Its fixed NO_GO remains intact. Calibration gains over a badly calibrated raw bank did not survive capable controls. A different Wiki bank with retained alternatives can justify a scoped known-method question; it does not reverse that result or establish generic graph stacking success.

## Closest primary ancestry and limits

The sources below reuse saved bounded primary-method scopes; no primary body, author implementation or proof was reopened here. `READ_SCOPES.json` preserves exact original locators and credits zero new reads.

| Source | Relevant collision and remaining scope |
| --- | --- |
| META-DES, arXiv1810.01270v1 §3 | Confidence, local posterior behavior/accuracy and output-profile matching estimate member competence. Reliability-conditioned combination is established; its labeled local reference features and majority-selection endpoint differ from the proposed label-free deployment cues and soft probability mixture. |
| GATS, arXiv2210.06391v1 §5/AppendixA.3; RBS, arXiv2206.01570v1 §§II-C2/IV | Already use graph/local prediction agreement to condition temperature calibration. Their positivity/label-role qualifications remain preserved. They provide close graph reliability ancestry, not proof that weighting improves accuracy. |
| MoE-NP, arXiv2412.00418v3 §3 | Graph-pattern/context features feed softmax expert weights. Its printed binary-logistic/multiclass ambiguity remains unresolved; adapting its bank/endpoint is not native reproduction. |
| Link-MoE, arXiv2402.08583v2 §4/Eqs1–3/Algorithm1 | Dense second-stage structure/feature-conditioned expert weighting. Its link-logit output and native supervision split differ; ordinary gate ancestry is direct. |
| MoSE, arXiv2509.09337v1 saved main-method blocks18–73 | Anonymous walks form structural subgraphs; topology-aware noisy top-K routing chooses trainable hidden-graph/random-walk-kernel experts. It changes representation/expert computation before prediction. It is graph expert-routing ancestry rather than this frozen dense late mixer. Printed balance reduction ambiguity remains; appendix algorithm/proofs/code/results were not qualified. |
| C&S, arXiv2010.13993v2 and saved pinned author source | Propagates permitted TRAIN-label residuals, scales correction and smooths. Saved source uses Y-p and adds correction despite the paper sign mismatch. The present features propagate predictions only and have no label seeds; adding C&S would change information/cost and is outside this one mechanism. |
| NLC, arXiv2405.17139v2 §3/AppendixA/§4.2 | Frozen feature-conditioned per-example model coefficients combine existing logits under supervised CE. Every backbone runs. Exact coefficient constraints/temperature map remain unqualified in saved prose. Dense late learned weighting is already established. |
| GEENI, DOI10.1145/3489517.3530416; GENNN, DOI10.1016/j.inffus.2024.102461 | Exact primary methods remain unresolved. GEENI abstract identifies likely-error nodes and suppresses outgoing messages;20 prior public attempts did not recover the body/code. GENNN has metadata/SSRN locator only. No retries, exclusion certificate or novelty inference follows. |

## Development scope, decisive result and costs

Use identical fixed five folds over the encountered VALID population for candidate and controls; each fold's labels are excluded from that fold's aggregator fit. The base selectors already used all VALID labels, and the current hypothesis was chosen after development diagnostics. Aggregator-only cross-fitting remains encountered development and does not cross-fit the selected whole pipeline. Transductive neighbor prediction context is allowed only on the unchanged authorized support and supplies no independence of connected observations. No claim of honest untouched assessment is available here.

Record the full paired roster before interpreting results. Compare served accuracy, NLL/Brier and repairs versus harms relative to native pooling. The diagnostic subset of pooling losses can describe whether retained correct alternatives survive, but its outcome-derived membership is never a gate input or selector. Stop this recipe if it provides no net served accuracy gain with acceptable proper-risk behavior beyond the strongest temperature/non-graph control; an NLL-only gain supports calibration utility. Stop the graph-context explanation if P=I matches or beats it. Root must freeze any numerical worthwhile-effect/uncertainty criterion before a study rather than choose one from outcomes. A promising development result still requires a fully frozen pipeline and separately authorized genuinely unused confirmation. This packet establishes no TEST custody or access.

One batched sparse P on the M probability fields costs O(|E|MC); neighbor peer means follow from sums, without another propagation. Own-node features and head add O(NM(C+48)) work plus8 tanh evaluations per node/member. There are56 trained parameters and O(NMC) probability/neighbor storage. The plain stacker has C(MC+1) coefficients including biases and O(NMC^2) forward work; charge its reference cost separately. With B frozen admitted banks, five folds and five fitted operators give25B small fits of500 updates each; native pooling has no fit and duplicate M=1 controls can be collapsed prospectively. No final refit is part of this development screen. These are operation/count estimates, not measured time or memory results. All base-route acquisition/serving remains charged.

Permitted full-node probability rows and exact graph support are necessary. VALID-only logits do not provide honest neighbor summaries. Missing retained fields make this design non-executable as written; they do not authorize a replay/export, access to sealed TEST-role predictions or silent support reduction. Any later authorized extraction, I/O/storage and replay cost must be declared. This task has read aggregate/source reports and metadata only, executed standard-library text/JSON/hash bookkeeping, and changed only this fresh packet. Current24/full9on77, canonical state, scientific sources and prior sealed packets remain untouched.
