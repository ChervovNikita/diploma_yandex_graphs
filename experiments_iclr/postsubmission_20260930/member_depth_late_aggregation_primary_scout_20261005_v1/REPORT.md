# Member and depth states for late graph-ensemble aggregation

5 October 2026. Narrow primary-literature scout and prose hypothesis only. Index72 and saved aggregation scopes were consulted first. No training, target inference, checkpoint/data/logit/label/history payload read, remote allocation contact, source import, paper edit, index integration or cumulative reading-total claim occurred.

## Decision

**A compact frozen readout of member and intermediate-depth states is a defensible hypothesis, but its ingredients have close prior art.** Jumping Knowledge and DAGNN already adapt graph depth; GAMLP already attends a precomputed graph-propagation bank; PCL/FFL already fuse shared/private member features; Diverse Peers already uses member-feature-conditioned attention across peer predictions. Recent VFusion is especially close to the frozen placement: concatenate frozen intermediate states, learn a nonlinear bottleneck and classifier, and separately apply feature fusion across frozen backbones.

The unresolved useful delta is whether the *actual graph member×depth bank* contains label-relevant information that final states/scores lose, and whether a compact attention bias learns to use it better than a competent ordinary readout of that same bank. The search does not establish historical novelty, global absence, target gain or readiness to train.

## Closest original methods

| Source and bounded scope | Verified operator | Consequence for this proposal |
|---|---|---|
| **Representation Learning on Graphs with Jumping Knowledge Networks**, official ICML2018/PMLR80 paper, §4–4.2, PDF pp5–6; rendered method pages inspected | Final layer receives intermediate graph representations. Options include concatenation followed by a linear map, coordinatewise max, and node-specific softmax attention from a bidirectional LSTM. The text first describes a weighted average of original layer states, then describes weighting the bidirectional hidden features; source correspondence was not checked. | Adaptive depth and intermediate-state fusion are established. JK's statement that concatenation is not node-adaptive applies to its linear transformation; it does not prove a nonlinear concatenation readout cannot react to node context. No theorem proof or performance claim transferred. |
| **Towards Deeper Graph Neural Networks**, arXiv:2007.09296v1, complete §4/Eqs8–9 | `Z=MLP(X)` has class width; bank is `[Z,Ahat Z,…,Ahat^k Z]`; shared projection gives per-node/per-depth **sigmoid** retainment scores; weighted sum feeds class softmax. CE trains the transformation and gate together. | A depth bank with learned local/global weighting is direct ancestry. Gates are not normalized across depths. These states share class coordinates and are propagated versions of one transformation, unlike unaligned private hidden channels in a frozen ensemble. |
| **Online Knowledge Distillation with Diverse Peers**, arXiv:1912.00350v2, complete named method section, Eqs1–11 and training/deployment paragraph | Peer features supply learned query/key projections and asymmetric softmax weights; weights mix peer probabilities into personalized distillation targets. Auxiliary peers and leader have CE; two levels of KL collaboration train concurrently. Shared lower/private upper branches or independent networks are supported; deployment keeps only the leader. | Member-feature-conditioned attention and shared/private architecture are explicit prior. A frozen served fusion head changes fitting and deployment placement. Similarity-derived attention is not an estimate of labeled residual covariance or a guarantee of useful prediction diversity. |
| **Vertical Fusion: Condensing Internal Representations for Robust ViT Classification**, arXiv:2607.10391v1, §3.1/§4, selected baseline/setup/limitations and AppendixC.2 control definition | Frozen intermediate CLS states are concatenated, mapped by a nonlinear MLP bottleneck with LayerNorm, then classified. CE plus latent off-diagonal correlation penalty trains only encoder/head. Depth subsets are predefined. Separate horizontal feature fusion concatenates final states from frozen backbones. | Direct frozen-depth and frozen-member feature-fusion ancestry. The method does not establish this graph member×depth combination, yet prevents claiming freezing, nonlinear depth compression, or latent decorrelation as a new principle. Preprint says under review; author code and numerical claims were not qualified. |

Saved scopes reused without reopening primary bodies: GAMLP2108.10097v3 §3; FFL1904.09058v2 §III and relevant figures; PCL2006.04147v2 method; AM-GCN2007.02265v1 common/specific embedding attention; GETS2410.09570v2 and pinned source; the existing nonlinear score/hidden stacker and graph-error scouts. GETS source scales one predictor's class logits and can change argmax; it is not a bank of independently competent classifier members. AM-GCN already attends common/specific graph embeddings. The separate Information Fusion paper **Graph ensemble neural network**, DOI10.1016/j.inffus.2024.102461, still has no readable primary method in saved evidence; this scout does not clear its overlap.

## One small readout hypothesis

Keep every base/member parameter fixed. Predeclare a small set of actual state interfaces and depths before outcome inspection. Shared activations that are physically identical belong in the bank once. Do not average raw private channels merely because widths match.

For state `h_v,m,l`, normalize each interface consistently and project to a small common value space `u_v,m,l`. Supply member/depth identity and a query from saved member scores plus fixed labels-free graph context `g_v` (neighbor pooled probabilities and degree/isolate context). A shallow gate produces

`alpha_v,m,l = softmax_(m,l) a(u_v,m,l, query_v, member_id, depth_id)`

`r_v = sum_(m,l) alpha_v,m,l u_v,m,l`.

A small residual classifier consumes `r_v`, the member scores and the same context. For the actual Amazon/Polynormer source, the anchor is **mean softmax probabilities**, not mean raw logits. Let

`s_v = log[(1/M) sum_m softmax(z_v,m)]`

computed as a stable log mixture. Define

`p_F(v) = softmax[s_v + U phi(r_v, z_v,1,…,z_v,M, g_v) + b]`,

with `U=b=0` initially. This preserves the mathematical native probability pool at initialization. It is a learned readout beyond a scalar convex logit pool; its output need not remain in the member-prediction convex hull. No gain guarantee follows. The actual arithmetic/order should be qualified if it is later implemented.

This is a composition of established operators, not an attributed reproduction of JK, DAGNN, GAMLP or VFusion. Separate value projections handle distinct coordinate systems; same dimensions alone do not supply alignment. Attention is a restricted interaction bias, while a nonlinear concatenation readout can also condition its response on each node. Equal parameter budgets do not prove equal function classes.

## Where useful information could come from

- A final classifier compresses features in a way that loses a signal retained by an intermediate private state.
- Relevant graph ranges differ between nodes, and earlier states preserve local information diluted later.
- Members retain different label-relevant components even when their final probabilities look similar.

These are conditions to test, not observed facts. Graph depth is not automatically an exact hop radius: learned transformations, skip/cumulative states and global attention change the interpretation. In the actual Polynormer path, local cumulative interfaces and the active global-head input must be named precisely. A final-head extraction design does not qualify arbitrary intermediate-depth exports.

The readout can fail when the bank is redundant, every state has discarded the relevant signal, private coordinate scales/semantics differ, graph neighborhoods mix incompatible labels, a low-dimensional attention bottleneck removes useful information, or limited fusion labels let extra parameters overfit. Latent orthogonality does not imply complementary prediction errors. A late head cannot recover information absent from its full input bank.

## One cheap decisive comparison

**Compare compact attention against a regularized nonlinear concatenation bottleneck on the exact same member×depth bank**, with the same normalization, native probability-pool anchor, graph context, fusion-fit labels, development endpoint, training/selection opportunity and a comparable readout parameter budget. Keep the bank/depth set fixed; do not run a depth/attention/gate grid. Both heads consume already materialized states and leave all base parameters fixed.

If attention only matches this capable readout, attention-specific utility is unsupported and the simpler readout is the practical choice. If it wins on the predeclared endpoint, that supports a useful inductive bias for this bank, not a new fusion principle or independent confirmation. This comparison alone does **not** isolate the contribution of earlier depths: that attribution requires the already contemplated capable final-state readout under the same roles and capacity. Similarly, beating a score stacker does not by itself distinguish extra hidden evidence from attention. These interpretation boundaries do not allocate additional arms here.

No honest fusion-training role is currently assigned by the inspected frozen study. TRAIN-control remains its evaluation endpoint; VAL already selected base checkpoints; TEST authorization is false. A later aggregation-only split/crossfit of already used VAL is retrospective development. Evaluation labels must be excluded from all supervised residual/propagation fields, not merely a node's own entry; fixed whole-fold masks are needed if label-derived context is ever added. The hypothesis above uses labels-free graph context.

## Cost and source feasibility

The saved source-feasibility report establishes raw logits retention but **no saved private penultimate activations**. Its final-state export estimate is about1.20GB raw FP32 tensors over the complete paired family, before overhead; it is a source-dimensional estimate, not observed availability or runtime. Intermediate-depth storage adds each retained interface. Any export requires an authorized complete selected-checkpoint forward and exact state/node/member/stage custody; it is not free because a model was previously trained. Capturing additional states during a separately admitted full replay could share that forward, but this scout changes no source or permission.

All dense member/base work remains charged. A late gate applied after member computation does not save that work. Charge extraction, state storage/I/O, graph context construction, head fitting/selection and serving. The next action is a source-bound state-bank/label-role decision by root, followed only if admitted by the single same-bank comparison above.

## Retrieval and evidence limits

`RETRIEVAL_LEDGER.json` records exact URLs, UTC receipts, byte counts, status and hashes. Queries were bounded arXiv title/keyword discovery; all routes used in this packet succeeded. The separate recent **Beyond Modality Fusion** item was screened only through metadata/abstract and is not a read method or transferred result. No journal/paywall, author implementation or native reproduction was resolved here. Search coverage cannot certify absence.

`READ_SCOPES.json` distinguishes selected substantive methods, metadata screens and incidental exposure. Public DAGNN result tables were incidentally displayed by a descendant-table query; VFusion abstract/control-result prose and JK contextual/figure text were also visible. No numerical outcome was audited, adopted or used to choose the hypothesis. Retained source bytes are broader than read scopes. Local extraction failures (`python` unavailable; `bs4` missing) led to the available bundled Python/lxml/pypdf; no dependency was installed. `SOURCE_BINDINGS.json` and `REUSED_SCOPES.json` bind safe prior metadata. The manifest and seal certify this packet's bytes only.
