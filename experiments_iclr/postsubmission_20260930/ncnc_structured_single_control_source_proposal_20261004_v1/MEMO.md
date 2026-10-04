# Structured single controls for the frozen NCNC J/F hypothesis

4 October 2026. **Source-only proposal; no execution release or scientific verdict.** This packet uses the acceptance-path memo and literature memory `index_v41`, reuses four relevant retained conclusions, and adds two bounded primary method reads. It contains no fit, project outcome, checkpoint, TEST access, canonical publication, or change to the sealed J/F source. The proposals require their own implementation, qualification and prospective comparison.

## What the controls need to answer

The frozen hypothesis compares J, which mixes four completion components after multiplying residual-slot Bernoulli terms, with F, which mixes each bit before multiplying. Both supervise the complete supplied TRAIN observation bits on the same native masked residual supports. A favorable J/F contrast would support this auxiliary in its declared setting. It would not establish that an ensemble is necessary, that joint graph reconstruction is new, or that aggregate counts and covariance cannot explain the benefit.

The nearest ingredient duplicate remains **GRAN**: its retained method and pinned author loss already supply a mixture of factorial Bernoullis and component-wide responsibilities. MaskGAE supplies masked topology and degree supervision; SGDiff supplies joint enclosing-subgraph density for link prediction; CAM supplies correlated linksets and strong set/global context. The two new sources strengthen those boundaries:

| Source | Scoped finding | Consequence here |
| --- | --- | --- |
| Tarlow et al., *Fast Exact Inference for Recursive Cardinality Models*, arXiv:1210.4899v1 | Unary terms plus count potentials define a normalized joint binary law. Nested/disjoint subset counts admit exact tree inference; balanced FFT trees have the stated faster bound. | Count-aware joint reconstruction and its inference are established methods. A proper structural single is practical to specify without four independently served predictors. |
| Vignac et al., *DiGress*, arXiv:2209.14734v3 | Per-node/edge classification and conditionally factorized reverse transitions, repeatedly conditioned on the current noisy graph, define a joint graph generator. Structural features enrich shared context. | Per-edge CE/BCE alone does not identify the full model as a marginal graph law. Joint generation and structural context are prior; dense whole-graph diffusion is not a qualified Collab comparator. |

No exact duplicate of the complete frozen NCNC composition is established in these scopes. This is not a global novelty clearance. Reading and access accounting are in `READ_SCOPES.json`; retained evidence is referenced rather than copied in `REUSED_MEMORY.json`.

## Shared context and exact masks

Both proposed controls use only the same raw 128-dimensional features, the native visible TRAIN graph, query endpoints, native common neighbors, and the exact ordered left/right residual supports available to J/F. A single may compute nonlinear functions of these inputs. It receives no extra complete-TRAIN adjacency as predictor context and no true count, observation bit, held-out edge, query class flag or J/F learned state.

For training, delete the **same positive record IDs** from the original TRAIN record multiset, preserve surviving duplicates, and then apply the same symmetry/coalescing operation. Use identical positive and sampled-negative queries and identical residual coordinates. The complete TRAIN teacher supplies labels only. Its bit Z=0 means **unobserved in TRAIN**, not a verified latent nonlink. Teacher counts are targets computed after enumeration; they never feed a head or candidate provider. Validation context is the same complete TRAIN-only graph used by J/F.

The frozen `epoch_stream` consumes global RNG for negatives and permutation and saves their hashes, not replay arrays. Different models consume different dropout randomness. A common seed consequently does not guarantee identical realized masks, and hashes cannot reconstruct the draws. Before a future exact comparison, admit a TRAIN-only replay provider containing the actual ordered record masks, full negative pairs, permutation/tail identities and ordered supports, with graph/support equality checked across all arms. Its RNG must be independent of model RNG. If the existing frozen realizations are unavailable, use separately admitted new J/F repetitions of the same frozen hypothesis with this common provider. Do not describe a mask-law-only comparison as exact realized matching. This proposal does not alter current J/F runs or claim their hashes are replayable inputs.

## Control C: one count-aware structural predictor

**Definition.** One fresh native NCNC encoder/scorer/target-decoder parameter set, width 64, augmented with a nonlinear count potential. “Single” refers to one trained target decoder; latent completion draws reuse that decoder and their work is charged.

For a query context C, let L and R be its exact native residual slot sets, with sizes n_L and n_R. The native recursive scorer supplies

\[
\eta_i=2.5(s_i-6)+\log(0.1),\qquad K_L=\sum_{i\in L}z_i,\quad K_R=\sum_{i\in R}z_i.
\]

Use the proper joint density

\[
p_C(z\mid C)=\frac{\exp\{\sum_i\eta_i z_i+g_C(K_L,K_R)\}}{\mathcal Z_C}.
\]

The count head scores **every** valid pair `(k,l)` with one shared MLP (hidden width 64, ReLU). Inputs are a visible-context summary, support sizes, absolute/log counts and normalized candidate counts. The summary uses endpoint features, native common-neighbor features, and separate sum/mean pools of a two-layer width-64 nonlinear map over every residual slot, its left/right role, counterpart features and native score. The head evaluates hypothetical counts, not teacher counts. It has no fixed maximum-degree output table and no count or slot truncation. This is a nonlinear parametrization of a full two-sided count potential; it is not claimed to represent every arbitrary count table at this finite width.

**Exact likelihood.** Define elementary-symmetric coefficients

\[
A_L(k)=\sum_{S\subseteq L:|S|=k}\exp\Big(\sum_{i\in S}\eta_i\Big),
\quad A_R(l)\text{ analogously}.
\]

Then

\[
\log\mathcal Z_C=\operatorname{logsumexp}_{k=0\ldots n_L,\,l=0\ldots n_R}
\{\log A_L(k)+\log A_R(l)+g_C(k,l)\}.
\]

A log-domain dynamic program computes all coefficients with `a[j,k]=logaddexp(a[j-1,k], eta[j]+a[j-1,k-1])`, `a[0,0]=0`, impossible entries `-inf`. The per-query auxiliary is `(log Z - sum_i eta_i Z_i - g_C(K_L(Z),K_R(Z)))/(n_L+n_R)`. Empty support has zero auxiliary and remains in the all-query average. Use lambda 1 and the same separate positive/negative query means as J/F. No clipping or epsilon changes the likelihood. Gradients train the scorer, count head and shared representation.

**Retain structure at serving.** Use four exact full-support joint draws from this density, rather than evaluating only mean completion weights. First sample `(K_L,K_R)` from probabilities proportional to `A_L(k) A_R(l) exp g_C(k,l)`; then independently sample the two unary-weighted subsets conditioned on their drawn counts using backward DP. Each draw gives weights `1.05 z_i` on all residual slots. Feed each completed neighborhood, unchanged native common contribution and endpoint features through the **same nonlinear NCNC decoder**. Train with native positive/negative mean BCE over the four draw logits and serve their mean raw logit. Completion draws and their density inputs are detached on the main target route; the exact auxiliary is the source of scorer/count-density gradients. Shared encoder features still receive ordinary main gradients.

Draw RNG is separate from the replay provider and dropout. A prospectively bound counter/key scheme uses only seed, epoch/batch/query/draw/slot identity for training; validation uses fixed query/draw keys independent of labels and candidate rank, making selected-state replay deterministic. Four draws are a declared Monte Carlo target-prediction approximation, although each structural sample is exact. Its adequacy must be qualified; it is not an exact decoder expectation or a cost advantage. No rare configuration is omitted by a degree/count cap.

**Costs and limit.** A straightforward exact implementation uses `O(n_L²+n_R²+(n_L+1)(n_R+1) H)` count/DP work per query, where H is head evaluation cost; storing backward tables can use quadratic memory. Query blocks and recomputation may bound memory while preserving every slot, count state and original batch reduction. Four decoder passes add work. Do not borrow the paper's `O(D log² D)` bound for arbitrary `g(K_L,K_R)`: a restricted additive potential on L, R and their union can use nested cardinality inference, but that is a weaker, separately specified model. Parameter count and full-batch time/memory are unmeasured.

This control changes completion weights and stochastic decoding while retaining the input information, label semantics and source/main detachment. It is a capable quality comparator, not a pure one-line J/F ablation. If it reproduces an eventual J benefit, a count-potential explanation remains viable. Its conditional distribution within each count pair is unary-weighted; it does not express arbitrary same-count spatial dependencies.

## Control S: one nonlinear predictor with full completion covariance

**Definition.** One fresh model has a shared encoder and its own four equal-prior internal generative completion heads, with the frozen native q calibration and exact J auxiliary. The heads are trained from scratch; J/F states, features and outputs are not donors. They are density components, not independently served target predictors. Their paid completion-bank work and parameters remain part of this control.

For its own density, compute all residual-slot moments, including left/right cross pairs:

\[
\mu_i=\tfrac14\sum_m q_{mi},\quad
\Sigma_{ii}=\mu_i(1-\mu_i),\quad
\Sigma_{ij}=\tfrac14\sum_mq_{mi}q_{mj}-\mu_i\mu_j\quad(i\ne j).
\]

An exact implicit representation is

\[
V_{im}=\tfrac12(q_{mi}-\mu_i),\quad
d_i=\tfrac14\sum_mq_{mi}(1-q_{mi}),\quad
\Sigma=\operatorname{diag}(d)+VV^T.
\]

This includes conditional Bernoulli variance; using only between-component spread would omit part of the covariance. The four-column factor is an exact representation for this density, not an eigen truncation. It is used internally to materialize pair blocks. The target head does not receive V, raw modes, component identities, responsibilities or component target logits.

**Nonlinear target head.** Use a width-64 set transformer with two attention blocks, four heads, a width-128 feed-forward sublayer, and a nonlinear width-32 pair-bias MLP. Slot inputs contain allowed shared contextual features, role, mean and diagonal variance. Every ordered slot pair contributes its exact covariance to attention, alongside the two slot means/roles. A query token and native common-neighbor/endpoint features produce one target logit; query-token interactions use context-only bias, while all slot-to-slot interactions use the exact covariance entry. Stream complete pair blocks with exact softmax reductions and qualified backward recomputation; no top-k, diagonal-only input, pair sampling, eigen approximation or degree cap is permitted. All full first/second moments are supplied, not just a projected aggregate neighborhood variance. The density-to-target moment route is detached like native completion; direct contextual features retain main gradients.

Train the single served logit using positive and negative query-mean BCE, plus lambda 1 times the same exact J auxiliary query means. Empty residual support retains zero auxiliary and uses the permitted endpoint/common context. Serve that one logit. The nonlinear head has enough input access to test practical sufficiency of full covariance beyond a terminal affine or Taylor-moment baseline. Width and architecture are concrete starting specifications, not evidence of predictive competence.

**Costs and limit.** Exact moment construction is linear in slots times four; the full contextual pair attention costs quadratic work in support size. Streaming reduces pair storage, not paid pair work. All scorer-head, target-head, encoder, graph and lookup costs must be reported. The model is not claimed to be cheaper, parameter matched or architecture-single throughout. A benefit of J over this control would not isolate higher-order information: their target decoders, private contextual transformations and main objectives differ. Learned shared context may itself carry structural information. This is a strong practical comparator, not a universal information-bottleneck proof.

## Qualification and a path to a defensible claim

No implementation or fit is admitted by this packet. The next source preparation must fix parameter sharing, exact context construction, trace/draw providers, normalizer/sampler or moment interfaces, dropout, optimizer, deterministic replay and memory blocks before outcomes. Require fabricated-support exact-enumeration checks for count likelihood/marginals/samples; direct covariance parity, positive semidefinite/diagonal checks and mode-permutation invariance for S; gradients and label-only teacher checks; then complete native-size batch and selected-state replay qualification. A physical failure remains a resource or implementation finding; no slot/mask/budget truncation silently repairs it.

For a separately admitted comparison, preserve 100 epochs, 17 batches of 65,536 and 1,700 optimizer updates per fit; all 100 complete official VALID Hits50 selector candidates, first exact tie; and two complete selected-state VALID replays. Match the available inputs and realized replay trace across J/F/C/S, use prospectively fixed paired seeds, and report parameters, training/serving wall time, peak memory, every decoder draw/pair pass, selection work and failed attempts. No paper score supplies predictive competence here. No TEST opening or later holdout study is authorized.

A grouped/full-mode single can reproduce the original bank exactly. Keep that equivalence available for later ensemble-specific claims and charge its real work. The retained same-count and same-pair-moment witnesses establish an information possibility only; they do not establish that useful ambiguous configurations occur in Collab. A defensible narrow contribution would need complete served ranking gains, source-pattern competence/coherence and prospective replication, followed by these capable controls and matched association diagnostics. Lower reconstruction loss or favorable strata cannot rescue a null primary comparison. No acceptance probability or positive manuscript claim is made.

## Packet boundaries

`CONTROL_PROPOSALS.json` contains the operational specifications. `PAPER_CONCLUSIONS.json` separates primary-source findings from proposed adaptations. `INPUT_BINDINGS.json` pins the acceptance memo, `index_v41`, reused conclusion/scope files and exact frozen source inspected. `VERIFICATION.json`, `MANIFEST.json` and `SEAL.json` record mechanical custody and source-only checks. The two new reads are scoped methods, not full papers or author-code audits; failures and wrong-ID exclusions remain recorded.
