# Shared channel coordinates for fixed-depth ensemble aggregation

Date: 2026-10-05. Scope: a theoretical and primary-method assessment of whether shared GNN weights support mean, dispersion or cross-member hidden-channel aggregation before logits. This packet builds on saved conclusions and adds two bounded primary reads: CKA and optimal-transport model fusion. It reports no training, model/data/logit/checkpoint access, numerical experiment or predictive result. Novelty remains unproved.

## Decision

**Shared weights provide a defensible coordinate convention and possible inductive bias, but no established special aggregation advantage.** In a selected architecture and checkpoint, a shared channel index refers to the same shared parameter template. This can make channelwise summaries easier to specify and reduce the need for an explicit matching procedure. Private factors and different input trajectories can still change what that channel measures, its magnitude, its sign before nonlinearities and its activation region. Neither sharing nor channelwise dispersion establishes target information, complementary errors or useful uncertainty.

A frozen independent hidden bank is usable without first solving a separate alignment problem. A learned concatenation head receives each member in its own fixed block and can learn a separate projection for each block. Fixed channel permutations can be absorbed exactly in its first linear map. The decisive comparison is therefore **one capable nonlinear concatenation/readout on exactly the same full hidden bank, scores and graph context**, applied within each family. The structured candidate can be included as a zero-residual branch to ensure exact containment. No new experiment is allocated or authorized by this assessment.

## 1. Three meanings of a shared basis

1. **Operational indexing.** The shared architecture assigns channel j to one template in W. Once a checkpoint and export interface are fixed, those indices are deterministic. This is enough to implement a channelwise mean or variance.
2. **Semantic correspondence.** Channel j would have to respond to comparable target-relevant patterns across members. Shared W alone does not establish this. The shared template acts on member-specific inputs, including private feature scales and upstream nonlinear trajectories.
3. **Functional identifiability.** Every parameterization representing the same predictor would have to induce a unique hidden coordinate system, up to declared harmless transformations. Ordinary networks have permutation and other representation freedoms. The constrained factor family removes some freedoms but does not establish unique hidden semantics or a canonical statistical scale.

Only the first meaning follows directly from sharing. A useful empirical claim can be that this convention improves sample efficiency, head size, alignment cost or deployed quality at a measured budget. It cannot be promoted to a theorem of semantic alignment or target advantage without additional assumptions and evidence.

### Private factors change effective features

For one factorized layer in column-vector notation, write

    W_m = D(s_m) W D(r_m),
    a_m = D(s_m) W D(r_m) h_m + b_m.

BatchEnsemble explicitly supplies this shared-W/private-rank-one mechanism and its input/output feature-scaling implementation [BE]. Even if the jth row of W is common, its effective incoming direction is the jth row of W D(r_m). Private output factors and biases further change magnitude, sign and activation thresholds. Common parameter templates therefore do not imply equal effective directions or equal activations.

Do not claim that arbitrary independent hidden permutations remain symmetries of the shared-W/diagonal-factor parameter family. In generic dense W, distinct member permutations cannot all be absorbed into one shared W and private diagonal factors. Architecture-compatible **common** permutations can remain symmetries when all affected weights, biases, normalization parameters, gates and adjacent channel axes are transformed together; attention head/block constraints can restrict which permutations are compatible. Special degeneracies can add freedoms, but they do not justify an unrestricted member-wise permutation claim.

### A qualified scaling non-identifiability example

In a generic network with private diagonal input/output factors at consecutive affine ReLU layers, let C_m be any positive diagonal matrix. Replace the previous layer's private output scale and bias by C_m times those quantities, and replace the next layer's private input factor by C_m inverse times that factor. ReLU(C_m a) = C_m ReLU(a), so h_m becomes C_m h_m while the next preactivation and final logits remain unchanged. Hidden means and dispersions can change although predictions do not. This is a direct algebraic example, with positive scales, compatible private biases and no intervening operation that breaks the transport.

**This example is not an established gauge of the actual Amazon Polynormer adapter.** Its private factors occur at the stem and local/global heads. Its internal GNN body is shared and includes attention, branch products, LayerNorm, learned channel gates and cumulative state. Those operations and missing intermediate private factors prevent importing the generic consecutive-layer scaling argument without an architecture-specific proof. A shared normalization can stabilize scales; it is not by itself a certificate of semantic identity. The general lesson is to state an actual export coordinate convention and avoid claiming functional identifiability from shared W alone.

## 2. What mean, dispersion and interactions retain

At a fixed node v and declared depth/interface, let h_(v,m) be a row vector in R^d and let H_v be the labeled M-member bank. Define channelwise

    mu_v = (1/M) sum_m h_(v,m),
    delta_(v,m) = h_(v,m) - mu_v,
    variance_(v,j) = (1/M) sum_m delta_(v,m,j)^2.

The mean and symmetric dispersion are invariant to a permutation of the member rows. They assume cross-member channel correspondence. Under one common channel permutation P, they transform as mu_v P and variance_v P. Under different fixed P_m for each member, their values generally change. A downstream channel-equivariant or consistently reindexed head can handle the common permutation. Separate permutations change which features are pooled together.

The full mean plus all **labeled** centered contrasts reconstructs the bank exactly: h_(v,m) = mu_v + delta_(v,m). The contrasts have a zero-sum constraint; mean plus M-1 suitable labeled contrasts is sufficient. This is an invertible, possibly redundant coordinate change, not extra information. Mean plus variance alone is generally lossy and cannot reconstruct the member states. A scalar dispersion is still more lossy.

For same-channel pair products, a symmetric aggregate can also be redundant:

    sum_(m<n) h_(v,m,j) h_(v,n,j)
      = ((sum_m h_(v,m,j))^2 - sum_m h_(v,m,j)^2) / 2.

Thus a complete symmetric sum of pair products is determined by mean and second moment. Labeled member-pair products or cross-channel interaction features can retain different structure, but all are deterministic functions of H_v. They may offer a useful feature map or regularizer; they do not create observations. A nonlinear concatenation readout can learn interactions, while an explicit candidate branch supplies exact containment if its products cannot be represented exactly by the chosen finite generic network.

### Member identity and channel identity are different

The saved round-5 graph-alignment work concerns using the **same model/member across different nodes** when computing cross-node covariance. Independently scrambling member identity at each node destroys that correspondence. The present question concerns **hidden feature coordinates across different members**. A fixed P_m applied to all nodes of member m preserves member identity, its node-to-node information and the entire bank up to an invertible relabeling. These two permutations cannot be treated as the same diagnostic.

## 3. Independent banks do not require pre-alignment for concatenation

Independent compatible networks admit function-preserving hidden permutations when the corresponding adjacent weights and all relevant channel parameters are permuted [RB]. A single frozen exported bank can also be relabeled by any fixed invertible transformation, whether or not that transformation is a symmetry inside the original backbone. This is a statement about available information at the exported interface, not about the constrained backbone parameter family.

Let x_v be the concatenated member bank and B = blockdiag(P_1,...,P_M). The relabeled bank is x_v B. If a head first uses x_v A + b, replacing A by B inverse A yields exactly the original preactivation. This establishes exact function preservation for a fixed permutation and a corresponding head reindexing. It does not establish that finite-data fitting always finds the corresponding parameters, or that regularization and optimization are equally favorable in two families.

A single linear concatenation layer already learns independent projections of every fixed member block. It need not assume that channel 17 in member 0 matches channel 17 in member 1. A capable nonlinear continuation supplies feature-dependent routing and interactions. An independent bank can therefore be a strong fusion input without a separate activation-matching or weight-merging procedure. Matching can still be useful if the objective is to average weights, compress to one model, impose a shared summary or reduce head complexity.

## 4. Primary precedents and their limits

The following are exact precedents for ingredients, with different training/deployment placements. They neither prove that the present GNN composition is published in full nor establish its novelty.

| Work and exact identity | Qualified mechanism | Consequence for this question |
|---|---|---|
| BatchEnsemble, arXiv:2002.06715v2 [BE] | Shared slow W times private rank-one factors; equivalent input/output feature scaling. Reused selected primary passages. | Shared templates and private scaling are established mechanisms; rank-one sharing does not certify identical channel meaning. |
| Git Re-Basin, arXiv:2209.04836v6 [RB] | Function-preserving channel permutations; activation linear-assignment matching and joint-layer weight-matching heuristics. Reused selected primary passages. | Independent networks need correspondence for naive weight averaging. Exact exported-bank concatenation is permutation-reindexable. No universal connectivity guarantee is adopted. |
| Model Fusion via Optimal Transport, arXiv:1910.05653v1 [OT] | Layerwise soft transport from preactivation observations or incoming-weight vectors; transports incoming edges, aligns current neurons, then averages weights. PDF pp 3-5, sections 3-4, Eqs 1-9/Algorithm 1. | Same-index hidden units need not correspond. Alignment has activation/data, transport and normalization costs. Soft transport is generally not an exact function-preserving permutation; zero-bias experiments do not qualify a complete GNN with norm/attention/gates. |
| Similarity of Neural Network Representations Revisited, ICML 2019/PMLR 97 [CKA] | CKA/HSIC and invariance arguments; PDF pp 2-3, sections 2-3, Theorem1 statement and Eqs 1-4. | High linear CKA is compatible with separate orthogonal transforms, including channel permutations, and isotropic scaling. It does not certify same-index channel identity, target information or error complementarity. |
| Feature Fusion for Online Mutual Knowledge Distillation, arXiv:1904.09058v2 [FFL] | Concatenated branch hidden maps to depthwise spatial and pointwise channel convolution plus BN/ReLU; mutual KD; shared low/private high branches. Saved qualified method conclusions. | Direct nonlinear hidden-fusion ancestry, with online training and a different serving recipe. |
| Peer Collaborative Learning, arXiv:2006.04147v2 [PCL] | Concatenated peer features to an additional FC classifier, with CE/KD/EMA collaboration. Saved qualified method conclusions. | Direct learned hidden-fusion ancestry; the simple fact of cross-member feature fusion is established. |
| Cross-stitch, arXiv:1604.03539v1; sluice, arXiv:1705.08142v1 [CS] | Learned continuing cross-stream mixing; sluice also organizes shared/private subspaces. Saved qualified conclusions. | Cross-stream mixing and shared/private subspace ingredients are prior; frozen late aggregation is a different placement. |
| Jumping Knowledge, PMLR:v80/xu18c [JK] | Graph-depth concat plus linear map, max, or BiLSTM attention. Saved qualified method conclusions. | Depth fusion is established. The saved original-state versus LSTM-state weighted-sum prose discrepancy remains unresolved; no new resolution claimed. |
| DAGNN, arXiv:2007.09296v1 [DAGNN] | Class-coordinate propagation bank and sigmoid retainment weights. Saved qualified method conclusions. | Output class coordinates have task-defined identity; they are different from private hidden axes. |
| Online Knowledge Distillation with Diverse Peers, arXiv:1912.00350v2 [DP] | Projected peer features yield asymmetric weights over peer-probability distillation targets; online collaboration and leader-only deployment. Saved qualified method conclusions. | Feature-conditioned cross-member attention is prior, with different training/deployment placement. |
| Vertical Fusion, arXiv:2607.10391v1 [VF] | Frozen intermediate CLS concatenation to nonlinear bottleneck plus LayerNorm and classifier; CE plus latent correlation penalty; separate horizontal final-feature fusion. Saved qualified method conclusions; under-review metadata. | Very close frozen nonlinear feature-fusion ancestry. Author code/results and the full graph member-by-depth composition are not certified here. |

The paper's arbitrary-invertible-invariance degeneracy theorem has full-row-rank and width/sample assumptions; it is not a universal impossibility result about representation comparison. Linear CKA's orthogonal invariance is enough to reject a high-CKA-to-same-channel inference. A permutation or orthogonal transform can preserve linear CKA while changing channelwise pooled summaries. CKA is a similarity statistic, not mutual information.

OT Fusion reinforces the operational value of a shared convention when averaging/compressing aligned units is required. It does not show that shared weights uniquely solve aggregation, that OT should be added to the current study, or that independent concatenation must pay its matching cost. Neither new paper's performance or runtime results are transferred.

## 5. One decisive same-information control

**Question:** on the same frozen fixed-depth bank, does the proposed shared-coordinate structure improve a learned readout beyond a competent concatenation/readout, or offer measured efficiency at comparable quality?

Use one predeclared regularized nonlinear concatenation/readout with:

- Exactly the same complete H_v, all member scores, member ordering and labels-free graph context as the candidate. If the candidate aggregates neighboring hidden states, its graph operator/context is also available to the control; a node-only control would be insufficient.
- The same coordinate convention and information-preserving input preparation where possible. If the candidate normalization is lossy, retain the full exact raw bank for the containing control and count any additional inputs/cost. Do not silently give one arm a different hidden depth, graph view or score subset.
- The same native probability-pool anchor when a residual predictor is used, the same fit labels, evaluation endpoint, validation/selection opportunity and comparable head/regularization budget.
- A structured candidate branch plus a linear/nonlinear residual on the full concatenation, with residual initialized to zero if exact containment is needed. Alternatively, demonstrate that the chosen generic class can represent the candidate. A tiny generic head need not contain a product, attention or graph operator just because it is called nonlinear.
- The same head design and role rules on both the shared and independent banks. The control can learn correspondence through its block projections; no separate alignment-search grid is required.

If adding the exact branch exceeds a head cap, report the cap and containment/budget tradeoff explicitly. Architectural containment alone does not guarantee equal finite-data generalization or equal optimizer success. A budget-compatible generic control must be sufficiently capable to test the stated mechanism.

A candidate tie or loss against its same-bank generic control removes a claim of superior structured aggregation on that bank. A candidate win supports a bank- and budget-specific inductive bias. It does not establish additional information, unique shared-weight necessity or novelty. Similar results in both families weaken attribution to common channel coordinates. Cross-family differences can also arise from backbone quality, hidden sufficiency, error diversity and base selection; they do not isolate alignment by themselves. Measured head/storage/alignment efficiency can remain valuable even with a predictive tie.

### Optional paired algebraic diagnostic, not a new tuning grid

Choose one fixed independent channel permutation for each member, used for every node and role. Reindex a concatenation head's input weights analytically so its predictions are identical. Compare the behavior of the structured channel summary under that relabeling, either by analytically deriving its change or under a separately authorized prospective check. A degradation with an unreindexed structured summary diagnoses coordinate dependence. It does not prove lack of target information, the impossibility of learned alignment or a unique advantage of shared weights. No per-node/member scrambling, independent hyperparameter sweep or execution is prescribed here.

## 6. Information, errors and utility

Condition on frozen training/selection context T and available scores/graph context Q. For a deterministic structured summary Z = phi(H,Q,T),

    I(Y;Z | Q,T) <= I(Y;H | Q,T).

For the relevant distribution under unrestricted optimal log-loss prediction, the Bayes risk improvement from adding the full bank H to Q is

    H(Y | Q,T) - H(Y | H,Q,T) = I(Y;H | Q,T).

These are conditional-information identities, not estimates, finite-sample guarantees or authorization to fit another head. A learned summary may improve restricted finite-data performance by imposing a useful bias, even though it creates no information. Conversely, hidden differences can be nuisance scale, redundant representation or target-irrelevant variation.

Channel variance measures spread of hidden coordinates. Member error covariance involves targets and prediction errors. Neither hidden orthogonality, centered contrasts, common W nor high CKA certifies error independence, error complementarity, calibration or recovery of a correct class. Graph interactions can expose structure that a restricted head misses, but the identical graph information must enter the decisive control.

## 7. Actual Amazon interface and acquisition caveats

Static source evidence distinguishes the boundary adapter from an all-layer BatchEnsemble network. BoundaryProjector.forward implements F.linear(x * R[member], weight) * S[member] + B[member] at source line 37. The stem and local/global heads have private R/S/B; the internal Polynormer body is shared. The raw actual active-head input h and h * R[member] are distinct exported interfaces. Any channel-summary claim must say which is used.

The four shared-family members have one selected local/global stage; independent members can have different selected stages. A strict fixed-depth comparison must choose the same actual named state/interface for every selected checkpoint. It cannot mix a local cumulative state and a globally attended head input under a single depth label. If the intended interface is the common local cumulative state after the same number of layers, that hook must be declared even for checkpoints whose active output head is global. Earlier states require an additional exact interface review. Learned/cumulative/global states are not automatically exact graph hop radii.

Saved hidden states are not presently certified caches. The existing source-only final-state export estimate is 15 full-graph physical replays, 24 complete member trajectories and 1,203,830,784 raw FP32 bytes over three paired banks. These are dimensions and source replay counts, not measured latency, physical availability or integrity. Metadata/serialization, restoration, temporary activations, copies, I/O, graph/head work and any alignment are additional. A fixed-depth or multi-depth export can alter this estimate.

The frozen study allocates no new honest fusion-fit role. TRAIN-control is not a free readout-training set, VAL has already selected base checkpoints, and TEST label/scoring authority is absent. This report does not change those roles, open payloads, export hidden states, allocate fits or amend the protocol. A future utility qualification must account for both hidden acquisition and any honest supervised readout fit.

## 8. Evidence and accounting

- Two new bounded primary identities/scopes: CKA and OT Fusion. Four rendered pages were inspected in the prior segment of this same task: CKA PDF pp 2-3 and OT PDF pp 4-5.
- Reused primary passage exposure: Git Re-Basin and BatchEnsemble from saved round-13 passages. Other table entries use saved qualified conclusions/scopes; FFL, PCL, JK and VFusion were not rescouted.
- Zero new full-paper certifications, proof audits, author-code qualification, result transfers or global literature absence/novelty clearances. The generic algebra above is derived here and is qualified separately from the actual adapter.
- Zero scientific executions, training, hidden exports, SSH, installs, model/data/logit/checkpoint payload opens or hashes, manuscript edits, index edits or canonical edits. Document extraction/rendering and source/report authentication are the only computation in this task.
- Narrow file navigation initially included an incidental broad filename listing of existing payload filenames; no payload content, stat or hash was accessed. Source and report hashes in this packet cover public documents, static code or safe metadata only.

READ_SCOPES.json states the adopted and incidental primary exposure boundaries. SOURCE_BINDINGS.json pins reused safe evidence and the two new primary PDFs. RETRIEVAL_LEDGER.json records successful exact-URL retrievals. VERIFICATION.json, MANIFEST.json and SEAL.json authenticate this packet without certifying scientific results.

## References used in this assessment

[CKA] Kornblith, Norouzi, Lee and Hinton. Similarity of Neural Network Representations Revisited. ICML 2019, PMLR 97. Primary: https://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf . Read: PDF pp 2-3 only; no appendix proof/result adoption.

[OT] Singh and Jaggi. Model Fusion via Optimal Transport. arXiv:1910.05653v1, printed 12 October 2019. Primary: https://arxiv.org/pdf/1910.05653v1 . Read: PDF pp 3-5; section4.3 continuation on p6 not adopted. The printed v1 identity, rather than the later PDF creation metadata, determines the version binding.

[RB,BE] Saved round13_neural_enkf_qualification_v1/PRIMARY_PASSAGES.md: Git Re-Basin R1/R2, lines 591-778; BatchEnsemble B1, lines 956-1075. Existing local sources reused; not new retrieved identities.

[FFL,PCL,CS] cross_member_hidden_graph_error_aggregation_prior_scout_20261005_v1/PAPER_CONCLUSIONS.json and SOURCE_BINDINGS.json; prior scoped reads, not new full-paper certifications.

[JK,DAGNN,DP,VF] member_depth_late_aggregation_primary_scout_20261005_v1/SCOPED_CONCLUSIONS.json and READ_SCOPES.json; prior scoped reads, not new primary scouting in this assessment.

[Amazon] amazon_polynormer_frozen_hidden_fusion_source_feasibility_20261005_v1/FINDINGS.json; static boundary adapter at amazon_polynormer_paired_family_source_preparation_20261003_v6/reused_models/backbone_boundary_adapter.py, lines 15-108.
