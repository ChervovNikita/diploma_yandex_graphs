# Recent priors for private learning response and graph specialist supervision

5 October 2026. The closest newly inspected challenge is **Gradient-Aligned Routing**, a 29 September 2026 preprint. It directly occupies gradient-informed expert assignment and distinguishes balanced usage from specialization. Its inspected update differs from the candidate finite private response and shared-core derivative. Together with the saved meta-weighting, MCL, DERTS, GNCL and shared-adapter priors, it makes the candidate an attributed composition whose useful differences require comparison.

**One concrete missing comparison survives:** replace the finite response cost with its first-order gradient-alignment approximation while preserving the candidate's supervision, solver, derivative ownership and serving. Two conditional empirical gaps are described below. No complete duplication was established in the three bounded primary method scopes; this does not clear novelty, especially against unresolved recent primary sources. No fit, code change or study amendment is proposed or authorized.

## The candidate being challenged

At fixed shared parameters, each persistent private member takes one own-CE probe step. Each class-versus-competitor cost is the resulting finite softplus-margin change, centered across members and smoothly scaled within an unordered class pair. Both label directions contribute. A fixed finite response map allocates positive responsibilities with exact row and member masses in real arithmetic, entropy and the native graph Laplacian. The main private update treats the assignment as an independent coefficient; the complete outer shared derivative retains assignment dependence through adaptation. A disjoint permitted TRAIN query supplies member and probability-pool CE. After the shared step, the response and private update are recomputed from the original private state. Serving is the fixed probability mean.

This composition is already strongly attributed. The saved nearest-prior report establishes Ren/Meta-Weight-Net gradient utility, bilevel ownership and recomputation; MCL specialization; shared/private adaptation; and the small-step alignment expansion. Index72 and supplementsV5 also retain recent graph MoE, shared adapters, task-dependent representation learning and DERTS scopes. Their conclusions were consulted before retrieval. The following three IDs were absent from index72 text and the consulted supplement reports.

## Three recent primary methods

| Primary version and inspected method | Established operation | Consequence for the candidate |
|---|---|---|
| Yuchen Li, Mingyu Du, Zongqi Fan, Nguyen H. Tran and Ken-Tye Yong, **[Routing in Gradient Space: Balanced Usage Is Not Expert Specialization](https://arxiv.org/html/2609.36724v1)**, 2026-09-29, §§3 and5, AppendixD.10 andG | Same-task groups supply gradients summed across corresponding expert-parameter coordinates. Their detached Gram affinity rewards load-normalized within-expert gradient coherence. A recomputed router branch receives the auxiliary alignment gradient; other trainable parameters receive ordinary task-loss gradients. | The broad principle of allocating expert training by gradient compatibility is direct recent ancestry. Balanced mass alone cannot demonstrate useful specialization. The retained first-order cost comparison below is now particularly material. |
| Pavel Rumiantsev and Mark Coates, **[Graph Knowledge Distillation to Mixture of Experts](https://arxiv.org/html/2406.11919v1)**, 2024-06-17, §3 and complete§4 | A pretrained GNN distills to a student with experts routed by cosine proximity to memory centroids. Moving centroid updates, commitment, embedding self-similarity and load balance promote regional specialists. Graph structure enters teacher targets, DeepWalk encodings and neighbor-target sampling. | Graph-derived local specialist processing, routing stability and utilization balance are established. This strengthens the need for a competent graph specialist reference if making a broad graph-ensemble-specialization claim. |
| Jakub Piwko and coauthors, **[Divide, Specialize, and Route: A New Approach to Efficient Ensemble Learning](https://arxiv.org/html/2506.20814v1)**, 2025-06-25, complete§3 and Algorithms1–2 | Hellsemble passes mistakes plus a portion of correctly classified items to later learners, trains a difficulty router and selects/stops using validation performance. Serving delegates each example to one selected learner. | Training specialists around common difficult examples is recent explicit ancestry as well as older boosting ancestry. Mistake concentration by itself does not distinguish the candidate's learning rule. |

These are preprint-version method scopes. Peer-reviewed publication and later-version equivalence were not established. RbM's primary submission history exposes a v2 dated2024-11-21; its method was not read. Numerical claims in abstracts and GAR's cost paragraph were textually exposed but supply no adopted predictive or runtime evidence.

## The closest overlap is gradient assignment

GAR defines row-simplex assignment P and current-model gradient observations

    g_i = sum_e grad_(expert_e) loss_i
    W_ij = <stopgrad(g_i), stopgrad(g_j)>.

Its practical auxiliary objective is

    - sum_m ||sum_i P_im stopgrad(g_i)||² / (sum_i P_im + epsilon).

The paper explains that this aggregate occupies a common parameter template and is not the actual physical expert update under a different assignment. It retains diagonal/self terms and signed cross-item alignment. Practical GAR uses zero explicit entropy coefficient and load normalization rather than exact member-mass constraints. It detaches both observations and the inputs used to recompute router probabilities, avoiding the higher-order auxiliary path. All trainable non-router parameters receive only task-loss gradients before joint clipping and an optimizer step.

The candidate instead uses member-specific finite changes of every correct-versus-competitor margin following each member's own-CE step. Its assignment objective combines a linear response cost, entropy, a nonnegative native-graph Laplacian and hard balanced masses. Q allocates additional supervised margins; it does not gate the served prediction. Its shared update differentiates through the finite response and Q-dependent private update. These inspected algebraic and update differences prevent an exact GAR identity. They do not establish that the differences help prediction.

RbM supplies the closer graph specialist architecture. Its input-space centroids and training losses occupy region assignment, representation commitment and utilization balance. It has neither the inspected finite margin-response valuation nor the candidate shared outer query/recompute map. Its neighbor distillation propagates teacher targets; a confidently wrong teacher can supply a wrong target. The displayed method provides no general guarantee of correcting common confident errors. The top-k display and operational masking should be resolved from author source before any faithful port; author code was not read here.

Hellsemble supplies the closer error-driven allocation motivation. Its subset filtering, candidate fits and validation-selected router need separate accounting. The inspected algorithm can repeatedly assign the same easy examples to different difficulty classes because they remain in later training subsets; the displayed router-set union does not resolve conflicting labels. No code qualification or guarantee is inherited. The candidate's all-member own-CE anchor and fixed probability serving are material recipe differences, and competence remains an empirical requirement.

## Conditional gap 1 Finite learning response beyond gradient utility

The saved local expansion is

    c_ikm = -eta_probe * <grad_phi b_ikm, grad_phi own_CE_m>
            + O(eta_probe²).

GAR adds a recent reason to isolate that first-order ancestry. A useful **missing comparison** would substitute the displayed first-order cost, then apply the identical member centering, pair scaling, eight-step feasible solver, graph affinity, main supervision, query loss, recomputation and serving. Preserve the declared total shared derivative of that substituted map. This comparison does not automatically become Hessian-free: differentiating a gradient-based cost can itself require higher derivatives.

Uniform and current-margin controls test generic extra supervision and current difficulty. Stop-Q tests assignment credit at the same forward assignments. None isolates the finite-response remainder from first-order utility. A literal GAR reproduction would additionally change router parameters, prediction routing, grouping and task loss; it cannot isolate this cost choice without a separately attributed matched adaptation.

The conditional gap is empirical: **does finite member learning response, with its retained shared credit, improve the complete competent probability pool beyond first-order utility and competent ordinary own/pool training?** Full-bank relabeling remains an equality check. A same-capacity model trained on served-pool or own-plus-pool CE remains a necessary learning-rule reference. An improvement over a weaker shared baseline would not settle this gap.

## Conditional gap 2 Native topology must add corrective learning

The unresolved graph question is whether adjacency supplies productive specialization of class-competitor learning responses after accounting for label-pair balance and ordinary gradient compatibility. Current pair graphs include both classes and retain cross-label edges. Smoothing their responsibilities can coordinate useful correction learning, but it can also coordinate unhelpful assignments. Balanced loads, stable Q, distinct private rows and lower conditional unanimity do not establish complete served quality.

The already specified gamma=0 and fixed within-class graph-permutation contrasts are the first attribution checks; neither gains support from the observed recurrence alone. If claiming a broader graph-specialization mechanism, GAR's gradient-affinity partitioning and RbM's structural/representation specialist routing are concrete comparisons to account for. Their literal recipes have different information, prediction and training costs. No new arm, affinity, numerical threshold or fit is admitted by this report.

The defensible conditional gap is **correction utility of the actual graph-conditioned private learning map**: transferable permitted-TRAIN class-pair response heterogeneity, preserved member competence and complete probability-pool benefit over competent references. A graph smoother on costs is already established composition. The descriptive common-error pattern supplies no causal graph defect or guarantee that topology regularization corrects it. The preserved negative fusion screen supplies no positive training evidence.

## Unresolved recent sources and read limits

Two new discovery leads remain unresolved: **A dynamic ensemble learning model for robust Graph Neural Networks**, DOI[10.1016/j.neunet.2025.107810](https://doi.org/10.1016/j.neunet.2025.107810), and **Class-aware ensemble reweighting for robust learning under label noise**, DOI[10.1016/j.neucom.2026.134910](https://doi.org/10.1016/j.neucom.2026.134910). The public Elsevier abstract routes returnedHTTP200 with an identifier only, without abstract or method. Their titles and dates are discovery metadata; no overlap judgment follows. The older unresolved GENNN gap remains preserved without retrying its routes.

The situational-meta-task ensemble paper, DOI[10.3389/fnbot.2024.1391247](https://doi.org/10.3389/fnbot.2024.1391247), was exposed only through a discovery abstract describing cooperation of adapted initial models. Liquid Ensemble Selection, GMoPE, ERMoE, GC-MoE and other retrieved leads remain discovery-only here. They supply locators for later scope, not method or absence evidence.

Ten bounded 2024–2026 OpenAlex searches retained72 ranked metadata rows, including aliases and irrelevant results; this was not a field census. Six exact arXiv requests and two public primary identifier routes succeeded. Exactly three version-specific bounded method scopes and three primary abstract/metadata scopes were read. Full proofs, result sections/tables, figure pixels, author code and complete papers were not audited. Exact primary bytes, block ranges, passages, retrieval times and hashes are saved in READ_SCOPES.json and the source receipts. Saved-prior use is separately recorded in REUSE_SCOPES.json.

The outcome is a concrete missing comparison and two conditional empirical questions. No novelty clearance, predictive gain, native-source change, manuscript edit, data/checkpoint/output access, SSH, model execution, fit or additional agent is supplied.
