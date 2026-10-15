# C3 context targets: ancestry, actual difference and missing evidence

15 October 2026. **Disposition: retain the existing common/split experiment as a specific training-recipe question. No methodological novelty or predictive benefit is established by this reading.** No new arm, target, backbone or launch is proposed.

The important implementation difference is where the views act: C3 assigns different TRAIN auxiliary target distributions to existing native routes. It keeps their native inputs/operators and arithmetic mean-logit serving. This differs from changing the graph views actually encoded, or routing predictions through a learned mixture. Those differences are substantive implementation choices, but the broader ingredients—graph views, factual similarity supervision, shared representations and specialized objectives—have clear ancestry.

## 1. Exact current method

The authority is `context_affinity_v1/core.py`, the frozen `expanded_study_design_v1/TASK_CONTEXT_SPEC.md`, and its node/molecular/link ports. Historical README readiness statements are not a current qualification assessment. This note did not inspect running pilot outcomes.

For each admitted TRAIN anchor i, D_i contains other distinct objects in its observed class; links additionally require the same source relation and binary supervision group. Normalize each factual context c^A,c^B, then form

    q^r_ij = softmax over j in D_i [cos(c^r_i,c^r_j)/0.2],  r in {A,B}.
    p_mij  = softmax over j in D_i [cos(h_mi,h_mj)/0.2].

The targets are detached. The route preheads h are live. COMMON uses qbar=(qA+qB)/2 for every route; SPLIT uses qA for routes 0/1 and qB for 2/3. Both minimize the same native F plus .05 times mean route target cross-entropy, using the same donor opportunities and source-specific row reduction. Nodes/molecules balance supported anchor classes; NCN/NCNC sum positive/negative supported-group means; BUDDY uses a joint supported-anchor mean. F retains all original source rows and reductions.

There are no opposite-class donors in this denominator. Consequently C3 fits **relative within-class relationships**; F provides cross-class discrimination. This is not ordinary SupCon's full contrastive denominator, cross-route repulsion, a learned domain gate or a class-prototype target. All compatible donors have positive softmax mass, so COMMON and SPLIT have exactly the same donor support. The earlier hard-positive design's increased-common-support argument does not apply here.

| Family | A and B | Information limit |
| --- | --- | --- |
| Nodes | Native x versus one incoming degree-normalized propagation of x; at most 512 fixed TRAIN anchor/donor nodes | Public transductive features/topology enter B. TRAIN-only describes supervised anchor/donor identities and labels, not an assertion that all factual feature/topology inputs come from TRAIN nodes. |
| Molecules | Per-field native atom-category versus directed bond-category frequency histograms, within the original minibatch | Coarse composition summaries; they do not specify substructure arrangement or establish pharmacological relevance. Missing labels are excluded only from AUX. |
| Links | Sorted log1p endpoint degrees versus log1p CN/AA/RA, from the simple TRAIN-positive graph with the queried pair excluded | Source-sampled negatives are supervision groups, not verified nonedges. The auxiliary graph law differs deliberately from the unchanged predictor's batch mask. |

Cosine normalization discards radial scale. In particular, equal-degree endpoint vectors all have the same direction regardless of degree magnitude. Common-witness coordinates may be highly correlated; zero witness contexts give uniform targets. Different raw context names do not establish distinct or useful q distributions.

Molecular/link donor sets follow the authentic minibatch/query cohort, so normalized q rows are not one globally fixed matrix across every update. Node A/B/cohort are fixed per fit. No context is required at serving. No new parameters, native forwards, optimizer groups or output heads are introduced. Auxiliary gradients stop at the existing prehead's downstream output layer/bank; native F still trains those parameters. Pairwise target/scoring and preprocessing work remain real costs.

## 2. Closest retained ancestry

| Retained primary source | Existing idea | Actual C3 difference and limit |
| --- | --- | --- |
| **GraphDIVE** [P1], §§4.1–4.3, Eqs2–6 | Degree/statistic-based node aggregation and multiple graph readouts feed groups of classifiers; two learned gates combine experts and views. | C3 supplies target-side factual views to four unchanged native trajectories and uniformly pools logits. It does not reproduce GraphDIVE's operators, classifier hierarchy, probability mixture or gated supervision. GraphDIVE already defeats a broad novelty claim for graph-view-driven expert specialization. |
| **MVGRL** [P2], §3 | Adjacency and diffusion are correlated structural inputs to two dedicated graph encoders; cross-view node/global agreement trains representations, which are summed for downstream use. | C3 uses one shared native backbone with private boundary routes, no alternate graph input or projector, and distinct same-class relational targets. It does not maximize agreement between encoded adjacency/diffusion views. Local/propagated graph information is nevertheless established ancestry. |
| **MoCL** [P3], §3.4 | Factual molecular similarity from ECFP/Tanimoto or interaction profiles supervises graph embedding similarity, by direct least squares or neighbor/background contrast. | This is the closest retained factual-relation target precedent. C3 uses native histograms, observed-class compatibility and row-softmax CE; it adds a fixed split/common assignment around a shared bank. It is not the first use of chemical similarities as graph-learning supervision. |
| **Training Diverse Graph Experts for Ensembles** [P4], §3.4, plus retained report | Meaningful domain specialization should improve domain prediction against a full-data expert; domain labels and independent experts differ from deployable shared mixtures. | C3 trains every route on complete native F and partitions target channels rather than training examples. It supplies no explicit domain-success or marginal-utility target. Its shared backbone and mean readout do not inherit independent-expert or oracle-domain gains. |

The earlier internal **8 October graph-level context-steering assessment** already defined route-specific Q versus common Qbar around a shared bank, discussed target-linearity/shared-gradient cancellation and warned about head-unused changes. Its fingerprints, auxiliary panels, denominator and older backbone differ from current C3. That history is direct design ancestry, not another public novelty proof. Its old proposed fits are not revived here.

The retained quality/architecture report's GraphDIVE method comparison is sufficient. Its printed Eq9/Eq10 sign issue remains unresolved; this note neither rereads the whole paper nor imports its diversity loss. Its current-source addendum, rather than the older C6 row, governs any C6 discussion; C6 is outside this C3 contrast.

## 3. What the common/split contrast does and does not isolate

It holds architecture, native risk, context information available to the training procedure, donor support, assignment counts and stated coefficient fixed. It changes which target each route must fit. It is not a test of extra labels, a new inference view or parameter count.

Three analytical limits matter; full derivations are in `ANALYTIC_LIMITS.md`.

1. **Different entropy floors.** H(qbar) exceeds average H(qA),H(qB) by JS(qA,qB). Lower SPLIT auxiliary CE can therefore occur without better target fit. Compare entropy-adjusted KL using the exact same rows, as well as native prediction losses.
2. **Different target realizability.** Each route's symmetric cosine softmax is a reversible donor kernel. Averaging two individually reversible targets can break reversibility. COMMON can therefore contain an irreducible approximation burden absent from the individual targets. An explicit valid three-object cosine example is supplied. A SPLIT win alone does not uniquely isolate helpful ensemble specialization.
3. **No automatic useful gradient or prediction diversity.** Mean score cotangents match at a fixed bank; shared-parameter gradients match only under equal route Jacobians. Different private derivatives may separate routes, but exact within-class directional collapse can have zero cosine gradient even for nonuniform targets. One-donor rows have zero auxiliary gradient. Feature separation can also be unused by the prediction head.

These limits do not invalidate the declared experiment or prove C3 cannot help. They restrict attribution. In particular, positive target-distinction counters, larger embedding distance, a lower auxiliary scalar or a larger oracle gap are not evidence of improved served prediction.

## 4. Actionable guidance for the existing arms

Use the frozen COMMON/SPLIT arms and existing native M1, boundary GNNM and independent-I4 references where their protocols are compatible. Do not add a new uniform-target, shuffled-target, altered-average, extra-backbone or capacity-control fit on this note's authority. C1 is not an operation-matched substitute for C3's common target.

**Before interpreting scores:** preserve exact cohort/target/signature, assignment and method hashes. For existing TRAIN target records, tabulate support 0/1/2+, qA/qB total variation and JS, entropy, zero contexts, and common-target three-cycle residuals. Counts merely exceeding 1e-8 establish numerical distinction, not effect size. These are passive analysis requests; if target records are unavailable, report the gap instead of silently adding forwards or rerunning a fit.

**Target mechanism:** on the same saved eligible TRAIN rows and selected states, report a four-route by two-target KL matrix and the assignment-alignment contrast defined in the derivation. Compare it with the same ordered route groups in COMMON and across already-declared paired seeds. This establishes whether routes fit different target relations; it does not establish better predictions. Inspect actual auxiliary-to-native gradient norms or route-gradient alignment only if already captured or separately budgeted; zero/finite gradient qualification is insufficient for a signal-strength claim.

**Prediction:** compare SPLIT−COMMON and SPLIT−base on the exact served native VALID metric and proper native loss. Report each/mean/worst route quality, served pooling improvement, same-object paired repairs/harms and common errors. For link ranking use native per-query ranks/Hits/MRR and the original negative protocol, not an invented binary threshold. For molecules retain masks and class/scaffold limitations. An oracle best-route readout is only an upper bound. A positive convex-loss pooling gap is not itself a gain over the native baseline.

**Interpretation:** if SPLIT fits its targets but does not improve the served predictor, record representation steering without predictive utility. If it beats COMMON but not base, record an inferior/common-target burden explanation. If served quality improves without assignment-aligned prediction repairs, report a recipe gain with unresolved mechanism. Useful specialization needs credible native quality plus complementary repairs that survive the predeclared selection/convergence protocol; one development-selected endpoint cannot establish generalization or novelty.

**Cost/selection:** use each method's frozen selector and stopping rule; preserve full convergence/cost records and any baseline selection differences. Charge context construction, pairwise scoring, memory and all diagnostic work. No synchronized runtime or speedup is inferred from zero extra native forwards. Any later confirmation follows the existing bounded-pilot authorization and a frozen protocol; full-campaign GO remains separate.

## 5. Citation and reading custody

- [P1] Hu et al. **GraphDIVE: Graph Classification by Mixture of Diverse Experts.** IJCAI 2022. DOI [10.24963/ijcai.2022/289](https://doi.org/10.24963/ijcai.2022/289). Targeted reinspection of retained PDF-page 3–4 text, §§4.1–4.3; other loss/protocol cautions reused from the existing scoped report.
- [P2] **Contrastive Multi-View Representation Learning on Graphs.** [arXiv:2006.05582v1](https://arxiv.org/html/2006.05582v1). Retained §3 method excerpt, blocks15–41; no new scope extension.
- [P3] Sun et al. **MoCL: Contrastive Learning on Molecular Graphs with Multi-level Domain Knowledge.** [arXiv:2106.04509v1](https://arxiv.org/html/2106.04509v1), DOI [10.1145/3447548.3467186](https://doi.org/10.1145/3447548.3467186). Retained §3.4 passage and title metadata. The current primary title corrects a title variant in the older internal report; that report is untouched.
- [P4] **Training Diverse Graph Experts for Ensembles.** [arXiv:2510.18370v1](https://arxiv.org/html/2510.18370v1). Retained §3.4 excerpt and existing quality/architecture conclusions; no new empirical result adopted.

Zero newly retrieved papers, zero new primary method scope extensions, zero full-paper-read claims. Four retained primary identities were used only for the bounded comparisons above. No scholarly search was needed; this is not an exhaustive novelty clearance. No model/data/TEST execution, remote access, frozen source edit or Git operation occurred. `SOURCE_RECEIPTS.json` pins consulted local evidence; no raw literature copies are added.
