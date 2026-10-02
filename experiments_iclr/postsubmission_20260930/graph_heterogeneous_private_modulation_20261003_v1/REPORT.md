# Relation-dependent private modulation: a bounded quality lane

**Disposition: one prospective utility test, with known ingredients attributed.** The question is whether a compact member-by-relation interaction in private fast factors improves prediction on complete released heterogeneous graphs. No new tensor factorization, adapter, hypernetwork, ensemble principle or graph-computation reduction is claimed. The operation below is separate from the current initialization/projection lanes. It is neither source-qualified nor adopted; no driver, data, result artifact or compute was accessed.

## Evidence reused and newly scoped

Latest saved memory v24 was consulted first: 108 conclusion records and 64 normalized paper identifiers are catalog counts, not full-paper read totals. Saved BatchEnsemble, HG-Adapter, CHoE, R-GCN, HGT and SeHGNN conclusions were reused. The sealed GEM closure was retained unchanged. Earlier typed-factor reports establish diagonal-adapter equivalence and the private-transport boundary; their old no-go disposition does not decide the merit of a properly controlled quality test.

Three new scoped primary method reads were completed, with zero full-paper, author-code or retained-primary rereads:

| Exact source | Scoped conclusion |
|---|---|
| [REEF: Relation-Aware Graph Foundation Model, 2505.12027v2](https://arxiv.org/html/2505.12027v2), updated 28 September 2026 | Relation descriptions condition aggregator/classifier parameter generators; dataset descriptions condition projectors and feature bias. This directly attributes relation-conditioned parameter generation. The heterogeneous extension uses type-specific feature projections and canonical-edge relation tokens on original HGB ACM/DBLP splits. The full recipe includes text encoders, SVD, augmentation and multi-dataset pretraining. Relation-indexed generator notation makes exact weight tying across relations unclear without code. It is not a member-factor ensemble. |
| [DRSA, 2605.00731v1](https://arxiv.org/html/2605.00731v1), 1 May 2026 | Type-pair bilinear operators `M_r=A_source B_target^T`, fixed Gaussian low-rank projections, structural latent targets, semantic projections and residual features precede shared-encoder pretraining. It is relation-aware alignment, not BE. Two relation labels with the same endpoint types receive the same printed operator. Its alternating-update and convergence claims are not transferred. |
| [HGB / Simple-HGN, 2112.14936v1](https://arxiv.org/html/2112.14936v1), 30 December 2021 | Complete released benchmark graphs, standard splits, feature choices and strong native comparators matter. Simple-HGN uses typed edge embeddings in attention, node/edge residuals, multiple heads and output normalization. Properly configured GAT is a necessary challenger; HGT and Simple-HGN native settings were scoped. |

Exact reading counts: **150 paragraph/heading blocks (106 paragraphs, 44 headings), 32 display-math nodes plus 16 other equation fragments, and one complete printed algorithm**. The 48 separately inspected formula nodes/fragments are not 48 separate equations; inline terms displayed within prose/listings are outside this formula-preview count. Three abstracts and citation/version metadata were read. Public code links were identified as metadata, without visiting repositories. A REEF heterograph-method paragraph also displays source performance numbers; they are incidental exposure, supply no current result and are not used to claim competence or benefit. Full details and limitations are in `READ_SCOPES.json`.

## Exact naive equivalences

In row-vector notation, at an eligible fixed relation map,

```text
h_m D(a_mr) W_r D(b_mr).
```

If `a_mr=a_m ⊙ alpha_r` and `b_mr=b_m ⊙ beta_r`, diagonal matrices commute on each side, giving ordinary global BE on the shared typed map `W'_r=D(alpha_r)W_rD(beta_r)`. Merely splitting member and relation vectors multiplicatively adds no operator beyond that composition. Type-pair indexing has the same limitation when its factors are separable.

Every such fast map is also exactly an equally located diagonal activation-adapter path: scale the input by `a_mr`, apply the shared map, then scale the output by `b_mr`. Writing either scaling as `x+(factor-1)⊙x` changes no function. Matched coordinates, initialization, decay, optimizer and bias treatment are needed for a training equivalence; superficially similar residual adapters can have different training rules. HG-Adapter is a different complete algorithm with low-rank structural adapters, graph construction and label-informed training; this algebra does not claim it is identical.

## One proposed operation

Use a native HGT backbone with shared type/relation matrices, attention parameters, output maps, normalization and head. Maintain **four complete private member states**. At each layer, place global member input/output fast vectors only around the native relation-message transform, with the same sites in all matched factor arms. Query/key and other branches keep their native shared parameterization and consume each member's own hidden state. Replace the output vector by

```text
b_mr = b_m + c_m q_r u
message_mr(h) = h D(a_m) W_source,relation D(b_mr).
```

Here `c_m` is one member scalar, `q_r` one scalar per canonical non-self relation, and `u` one common output-channel vector, all layer-specific. This is a **rank-one CP residual in the member × relation × channel fast-factor tensor**, or equivalently a factorized conditional diagonal adapter. Train it with ordinary mean own-member TRAIN CE and the same native state rule; serve a fixed mean-logit pool. No structural router, auxiliary diversity objective, cotangent initialization or update projection is added. Native biases, heads, reverse relations and factor placement still require source binding.

One layer of width `d` adds `M+R+d` parameters to `2Md` ordinary fast-vector parameters; unconstrained relation-output vectors require `Md+MRd`. This is shared parameter storage, not shared member propagation. In recurrent HGT, different hidden states generally produce different attention and relation messages. Packing members reduces call count without removing their edge/channel work. No speed claim follows.

### The local distinction and its limits

For one fixed effective weight coordinate with these exact sites, ordinary global BE gives `E_mr=a_mi b_mj W_rij`: its member × relation matrix has rank at most one. The CP residual can give rank two. For example, `a=1`, `b=(1,1)`, `c=(1,2)`, `q=(1,2)`, `u=1`, and `W_r=1` yield `[[2,3],[3,5]]`, determinant one. The proposed family contains global BE at `c=0` and is contained in unrestricted relation-output BE at the same sites.

This is a **local effective-map distinction**, not a theorem that the complete HGT function class is strictly larger. Learned relation maps, attention, nonlinear private paths and wider global-BE channels can already create member-by-relation interactions. A linear mean pool can absorb or cancel the added effect. A changed coordinate matters only if nonzero base-map entries, features, relation operators and downstream derivatives expose it. Parallel relations with the same endpoint types can motivate canonical-relation conditioning, but that is already an established relational-GNN idea. No predictive complementarity or generalization follows from rank alone.

## Initialization and nulls

The exact gradients and null cases are in `DERIVATION.md`. They matter before attributing any member distinction to the new term.

- With `c=0` and nonzero `q,u`, the forward equals global BE. At identical routes, every `c_m` gradient can be identical; the operation then creates no member distinction. Existing ordinary-BE factors or independent dropout can break route symmetry, but that distinction is already supplied by those components.
- With all three factors zero, all product gradients vanish. With `q=0`, antithetic `c` summing to zero and identical routes, the pooled `q` gradient cancels; the other two gradients also vanish. This is an algebraic dead initialization under the stated symmetric objective/state assumptions.
- Prospectively initialize all products nonzero: `c=0.01*(-3,-1,1,3)/sqrt(5)`, `q` balanced ±1 across canonical relations, and `u` ±1 across channels; use the bound seed convention. `R>=2` and variation in `q` are required. Pair the same ordinary BE initialization, shared weights and member stochastic conventions across matched arms. This changes the initial function from global BE and must be disclosed. Nonzero parameter Jacobians do not guarantee nonzero loss gradients or useful diversity. Antithetic centering can cancel the first-order pooled contribution in a purely linear common-response case.

Only the product-dependent **additional** relation response is the new experimental variable. Report how much diversity already exists in global BE and dropout. Do not credit the CP term for all observed route differences. Fix rank one, sites and amplitude before outcomes; no initialization-search rescue is part of this proposal.

## Representative comparison

Use the **complete released HGB-DBLP and HGB-ACM graphs**, without analyst-selected subgraphs, with original transductive splits (24% TRAIN, 6% validation, 70% test). HGB itself is a benchmark release drawn from larger sources; “complete” refers to that release. Preserve ACM citations/references and released typed features, relations and node order. The paper-linked [HGB release](https://github.com/THUDM/HGB) is identified; files, availability, revision, local use history and split hashes are unqualified.

Nine prospectively fixed arms:

1. Native author HGT.
2. Properly configured native author GAT.
3. Native author Simple-HGN.
4. Native author SeHGNN, including its declared preprocessing/label inputs.
5. Global message-site BE on HGT.
6. The single CP interaction operation above.
7. Unrestricted canonical-relation output factors at the same HGT sites.
8. Four independently trained native HGT members with the same fixed pool.
9. Wider global-BE HGT: the smallest head-compatible width whose total trainable parameter count is at least that of the CP arm, determined from source before outcomes. Width 72 is a plausible next width above native 64 with eight heads, not a qualified count. Report the remaining budget gap.

The wider control tests a simple capacity explanation; it does not provide exact parameter equality. The unrestricted arm initializes its full output-factor table to the CP-generated values, so its initial function matches the CP arm before training; optimization coordinates/regularization can still differ. It tests whether tensor sharing discards useful relation freedom. Global BE cannot generally match that initial map. A utility gain therefore concerns the specified operation including its initialization, without isolating learned interaction from initial perturbation. The exact conditional diagonal-adapter path needs a source equivalence audit, not a falsely independent training arm. Recent REEF, HG-Adapter, CHoE and DRSA are complete prior comparators at the method level: their generation, pretraining, frozen experts, graph construction and objectives differ. Published scores or a stripped version cannot stand in for native baselines. A foundation/adaptation claim beyond this supervised HGT operation requires qualified complete native recipes and all acquisition costs; this packet makes no such claim.

Use five paired master seeds (prospectively 131, 137, 139, 149, 151, pending root use-history checks), author-competent validation selection and a matched, prebound validation budget for HGT factor arms. HGB native HGT uses width 64/eight heads, depth two on ACM and three on DBLP; its feature choice differs by graph. Other author baselines retain their own declared competent recipes. Bind source/feature/normalization/head/dropout/optimizer/pooling differences explicitly before outcomes. SeHGNN's sparse learned-embedding branch and label propagation are material computation, not a free cache.

Primary utility is later heldout predictive NLL; native Micro/Macro-F1 are required competence checks. Report raw and common validation-only scalar-temperature-calibrated NLL because output normalization/pooling differ across native methods; calibration has one prebound rule per arm and uses no test labels. Open test labels once after freeze. A prospective practical screen is ≥0.01 nats mean calibrated NLL gain over global BE and the best validation-chosen native challenger on both graphs, positive gain in at least four of five paired seeds, and no >0.5 percentage-point mean Micro/Macro-F1 decline. This is not a significance claim. Also report comparisons to wider BE, unrestricted factors and untied members; gains explained by wider capacity or ordinary factor freedom do not establish a relation-interaction benefit.

Retain every paired run and failed construction. Parameter/factor diversity, lower cosine similarity or heterogeneous labels alone are insufficient. Required diagnostics include effective-map interaction residuals, member logits and calibration, relation contributions with fixed TRAIN-only analyses, gradient norms, native/operator equivalence residuals, actual stored parameters, peak memory and complete preparation/training/inference work. No diagnostic here has been computed on data.

## Resource assumptions and limits

Nine arms × two graphs × five seeds give 90 downstream configurations. With joint BE variants counted as one optimizer job and each untied member trained separately, there are 120 downstream training jobs, before qualification, tuning or preprocessing. Four joint factor variants each execute four member trajectories; this does not translate to 120 equal-cost single-model fits. No homogeneous-graph timing is transferred and no device-hour estimate is justified without HGB profiling. Shared matrices save storage relative to untied members; private activations/attention remain material. `RESOURCE_ESTIMATE.json` gives formulas and explicit unmeasured costs. Current compute availability does not decide merit.

The bounded searches had two API timeouts and many off-topic metadata returns; neither supports absence of prior. An initial saved-index locator was overbroad and displayed truncated saved conclusions; no retained primary or original result artifact was reopened. No global novelty/expressive-power certificate, author-code qualification or measured quality advantage is supplied. Root owns source/use-history qualification, protocol/tolerances, baseline admission, execution and resources. This packet closes the requested literature lane; further speculative growth is not proposed.
