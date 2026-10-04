# Shared graph views: accuracy hypothesis and close prior

Retain the persistent-view candidate as an unresolved **accuracy effect of sharing**, with a narrow claim. The new primary read supplies direct architectural prior, but does not show that tying improves predictive accuracy against exactly matched untied graph-view members. This packet adds one scoped primary paper, not a full-paper certification, a novelty clearance, or a method launch.

## New primary evidence

**AM-GCN: Adaptive Multi-channel Graph Convolutional Networks** (KDD 2020; arXiv:2007.02265v1; DOI 10.1145/3394486.3403177) is particularly close. It propagates the same node features through the native topology and a feature-kNN graph. Two specific GCNs have distinct weights. Its Common-GCN uses the same layer weight matrices on both graphs, averages the resulting common embeddings, then uses attention to combine common and specific embeddings before one CE classifier. It also uses Gram-matrix consistency and HSIC disparity losses. Exact passages and source identifiers are in `PRIMARY_PASSAGES.json`.

The critical passages are §3.2: “we share the same weight matrix ... for every layer of Common-GCN”; Eq. (6), `Z_C=(Z_CT+Z_CF)/2`; and §4.3, where AM-GCN-w/o removes the two constraints. The inspected variants change constraints while retaining the architecture. They therefore **do not isolate the accuracy effect of sharing**. This source is prior for shared graph representations with private view capacity, and supports neither the proposed member-serving claim nor an added penalty under a new name.

DGCN (WWW 2018; DOI 10.1145/3178876.3186116) remains a metadata-only lead: ACM primary endpoints returned 403 and the attempted WWW archive returned 404. No method conclusion is adopted from its title or metadata. The retained v51 conclusions for GRAND, GraphMix, graph filter banks, HGEN, shared ensemble objectives and PCGrad were consulted first; their primary papers were not reread.

## One precise remaining hypothesis

The two frozen views delete equal-class versus different-class edges **whose endpoints are both in official TRAIN**. They perturb distinct evidence: loss of same-class support versus removal of class-discordant neighbours. Persistent tied members may learn feature channels useful across both changes, while private factors retain different neighbourhood responses. The claim worth testing is that this reuse improves the **native pooled prediction** over otherwise matched untied persistent members. Treating all class-discordant edges as corrupt would overstate the hypothesis.

Sharing can help when useful channel semantics remain compatible across the two views and native-graph CE anchors them. It can hurt when class-discordant edges carry valid heterophilous or boundary evidence, when equal-class deletion removes scarce support, or when label-defined TRAIN topology creates a shortcut that does not transfer to native serving. Opposing shared gradients can also suppress a beneficial member direction; greater prediction spread alone does not resolve member quality or pooled accuracy.

The decisive comparison is **tied persistent versus untied persistent on exactly the same views and complete TRAIN label budget**, with the same native/probe passes, loss coefficients, corresponding initialization, private factors, recipe, selector and native inference pool. A positive targeted-view tie benefit compared with the tie benefit under budget-matched random deletion would support a graph-specific sharing effect. A benefit shared equally by random deletion would support more generic shared augmentation. Untied matching performance leaves a sharing-quality claim unsupported even if the ensemble beats a single predictor.

The amended shuffle is a valid **temporal persistence intervention**: two members per view at every update, balanced per-member view exposure over each even-length stage, and equal bank-level graph/label exposure. It is not a fixed global permutation. The earlier caveat about a mere slot relabeling does not apply. Persistent-versus-shuffled improvement would concern trajectory persistence; it would not by itself establish the benefit of tying.

The amendment's VALIDATION development endpoint is selection-associated evidence, not independent generalization evidence. TRAIN diagnostics cannot prove predictive gain. This report adopts no current or partial predictive values and makes no assumption about fresh TEST availability.

AM-GCN has no scoped operation establishing the separate one-step pooled-loss covariance selector. Its contribution here is attribution and a sharper accuracy question. No implementation, dataset inspection, server, PDF compilation, run launch, or external-file change was performed.
