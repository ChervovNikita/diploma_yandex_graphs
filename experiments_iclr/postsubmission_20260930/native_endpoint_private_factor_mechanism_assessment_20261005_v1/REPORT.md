# Private endpoint factors before native pair compression

5 October 2026. Source and saved-literature analysis, followed by the parent's requested small adapter preparation. No numerical model import, data/checkpoint/outcome payload access, experiment, server action or launch. Saved reports and parent messages contain qualitative project status; this is not total outcome-summary blindness. This packet proposes one attributed utility hypothesis; it supplies no novelty assessment or execution authorization. The CPU fixture is prepared but unexecuted.

## Bounded proposal

Add one learned **Householder factor** per member immediately before the outer target endpoint product in NCNC F4:

`H(v_m) = I − 2 v_m v_mᵀ/(v_mᵀv_m)`

`p_m(u,v) = (H(v_m) h_u) ⊙ (H(v_m) h_v)`.

Pass this vector through the existing member `xijlin` and native decoder. Change only the outer product, not node states, graph supports, common-neighbor features or detached recursive completion. Keep mean raw-logit serving and the existing mean-member TRAIN loss. Each vector has d stored floats and d−1 effective continuous degrees of freedom; its scale and sign do not change H. At the parent's implementation request, initialize member m to axis e_m (zero-based). This changes one channel's sign at both endpoints, so every initial product is exactly the original product algebraically. It needs no new random draw, diversity penalty, coefficient or seed search. The adapter rejects nonfinite/zero vectors and norms, with no epsilon or tunable clipping; the CPU fixture remains for later root execution.

**Hypothesis:** the common encoder and product force all outer member branches to start from one interaction basis. A compact private factor before compression can retain useful mixed-coordinate endpoint compatibility that post-product scales cannot recover, improving served ranking at useful total cost over a common factor and capable ordinary predictors. Useful private bases are unmeasured; the shared encoder or common-neighbor branch may already supply everything needed.

## Exact source restriction and forward path

The portable v2 source has one `SharedEncoder` with no member parameter axis (`prototype.py:48–76,224–233`). At width64 its output is shared GCN → LayerNorm → dropout → ReLU; the dimension-conditional residual is inactive for 128 input features. In every member, line198 forms **the identical** `h_u⊙h_v` before the private outer `xlin` at199. Existing input/output factors act as `(W(x⊙r_m))⊙s_m + shared_bias` (79–88).

Decoder order is endpoint linear → private LayerNorm → dropout → ReLU (149–166), then combination with the CN branch and native final linear/LN/dropout/ReLU/linear. The endpoint product is not an isotropic dot readout. Original pinned native code agrees: endpoint extraction/product at718–720, outer xlin at721, detached recursive scoring at745–754, and xijlin before xcnlin/final lin at781–785. Its `xijlin` definition is557–560. Native source commit is `11d597013750da17ce7468e344bec756a7af39a4`.

The current private two-layer BE `xlin` can mix channels in `hm`. It feeds common/residual-neighbor sums and the recursive scorer, whose endpoint product already uses `hm` (`prototype.py:168–182,199–215`). It does **not** supply the outer target product. There is no private encoder factorization here that can silently absorb four different pre-product transforms. Shared encoder learning may choose one useful common basis; private context paths may compensate. Accordingly this is an interface restriction, not whole-network function-class separation.

## Algebra, cross-coordinate terms and nulls

Let a,b be shared endpoint states, q=vᵀv, A=vᵀa and B=vᵀb. One reflection gives

`p_j = a_j b_j − (2v_j/q)(a_j B+b_j A) + (4v_j²/q²) A B`.

The last two terms contain symmetric cross-coordinate products `a_i b_k+a_k b_i`. An ordinary private diagonal before both operands instead gives `t_m²⊙(a⊙b)`; separate left/right diagonals also collapse to a post-product scale. A common endpoint permutation gives `P(a⊙b)`. Neither recovers lost information.

For a first-map row with effective product weights c (including its input factor), the bilinear matrix becomes

`H(v) D_c H(v) = D_c − 2(vvᵀD_c+D_cvvᵀ)/q + 4(vᵀD_cv)vvᵀ/q²`.

For i≠j its entry is `v_i v_j[−2(c_i+c_j)/q + 4(vᵀD_cv)/q²]`. These are compact symmetric bilinear features, not a new matrix principle.

Three exact boundaries matter:

- **Axis-aligned reflection:** v∝e_k only flips one coordinate. Both flips cancel in the product, so the entire intervention is null. Signed permutations likewise only relabel the original product.
- **Isotropic dot readout:** `sum_j p_j = aᵀHᵀHb = aᵀb`. Any orthogonal factor is null for a purely isotropic linear readout. Native anisotropic weights and nonlinear processing permit a difference but do not guarantee one.
- **Linear pooled-bank containment:** averaging arbitrary weighted orthogonal-product scalar branches yields one symmetric bilinear matrix, `B̄=mean_m H_mᵀD_(c_m)H_m`. An ordinary full symmetric bilinear single reproduces that score exactly. This does not assert equality to the actual nonlinear NCNC bank with private context trajectories.

An axis initializer is not necessarily a dead parameterization. Using e₁ to display the algebra, for tangent perturbation δ with δ₁=0 the first-order changes are `dp₁=2 sum_(j>1) δ_j(a₁b_j+a_jb₁)` and `dp_j=−2δ_j(a₁b_j+a_jb₁)`. Thus an anisotropic linear row responds by `2 sum_(j>1) δ_j(c₁−c_j)(a₁b_j+a_jb₁)`. Relabel the axis for each e_m. Actual native gradients remain unmeasured by the preparer.

A hand-algebra collision uses nonnegative states compatible with the encoder boundary: both `(e₁,e₂)` and `(e₁,e₃)` have original product0. For `v=(1,1,1)/√3` on three channels, their reflected products are respectively `(-2,-2,4)/9` and `(-2,4,-2)/9`. A suitable anisotropic map distinguishes them. This is a local information witness, not a synthetic performance proxy or a claim that both full native contexts coincide.

## Saved overlap and exclusions

Targeted reuse of index_v58 and the following saved reports supplies ancestry, not a new primary read or exhaustive semantic index review:

| Saved evidence | Consequence |
| --- | --- |
| `structured_coordinate_ensemble_gap_v1/REPORT.md` | Parameter superposition,1902.05522v2, already uses orthogonal contexts and explicitly random permutation contexts. That packet excludes learned dense rotations from its own candidate, not from prior art. Same-site adapters reproduce context operations. |
| `structured_candidate_assessor_v1/literature_gap_round10_factor_expressivity_v1/REPORT.md` | Diagonal orbits retain labeled cross ratios; moving adaptation before lossy compression may matter. Ordinary adapters and jointly learned representations prevent factor-exclusive claims. |
| MMSB,0705.4485v1, retained record in index_v58; HGT CP closest-prior report's DRSA,2605.00731v1 | Latent role compatibility and bilinear endpoint alignment are established graph mechanisms. Orthogonal product factors are a restricted symmetric bilinear pair adapter. No paper benefit transfers. |
| Parent's fresh primary scopes: HousE,2202.07919v1 §§3.1–3.2; GoldE,2405.08540v1 §3.1/3.2.1 and AppendixG | Relation-specific Householder transforms of KG endpoints, quadratic-metric generalizations and efficient vector operations are already graph prior. These were read by the parent and communicated as scoped conclusions, not reread here. Householder endpoint adaptation is not a new operator. |
| Parent's OMoE,2501.10062v1 §3, and abstract-only HTKGE discovery | OMoE orthogonalizes LoRA expert output vectors across experts with a router, rather than independently preserving endpoint feature norms. Its printed Eq7 lacks normalization for its stated unit-column claim. HTKGE's inaccessible publisher source leaves an abstract-only Householder-decoder lead. No expert orthogonality, predictive-diversity guarantee or exhaustive prior clearance follows. |
| `hgt_cp_residual_closest_prior_art_20261003_v1/REPORT.md`; existing HGT modulation implementation | Relation-conditioned CP/FiLM factors and typed block refinements already exist locally; they are not this proposal. |
| Graph-conditioned low-rank specialization, repulsion, mixed-block objective and shared-gradient-routing packets | Conditional structural adapters, generic contrastive separation, shared-pool/private-own CE and projected shared updates already have saved scope. They are not reopened. |

GAT attention, DIVE masks, private return/graph-operator banks, TabM initialization, posterior weighting, contrastive repulsion and the closed curvature initializer are excluded. The only retained change is one directly served pre-product factor. Its graph connection is the native link-compression site; H is not itself a graph-conditioned operator.

## Controls that can defeat the explanation

1. **Unchanged F4** and **F4 with one shared learned H** at precisely the same site. If shared H explains the gain, private interaction bases are unsupported.
2. **Ordinary independent native4**, plus **independent native4 with the identical H extension** in each predictor. Initialize bank member m at the same axis e_m without RNG consumption, not e₀ for every independent predictor. Each independent model keeps its own encoder, decoder, native loss scale, RNG and validation selection. A benefit equally available to the ordinary extension supports generic pair-head utility; it does not identify BE tying as the cause.
3. **Native single**, and a capable **single with two interaction frames**: `[a⊙b; (A a)⊙(A b)]`, A a fully learned dense d×d matrix, followed by native xijlin with2d inputs and the original LN/dropout/ReLU order. It retains the original interactions and adds a richer mixed basis. One prospective initialization is A=I, original native pair-map columns copied into the first block and zero second-block columns; this preserves the initial native scorer algebraically and introduces no random sweep. It still requires separate source qualification.

The capable single adds2d² parameters: one A and one extra block of pair-map columns. From the saved native count formulas, its total is `9d²+148d+3`, active `8d²+146d+2`. Width63 is the smallest integer exceeding both private-H F4 counts: **45,048/40,952** versus **44,046/39,049** total/active. Original F4 is43,790/38,793; shared-H F4 is43,854/38,857. Counts retain unused fixed-pt state and count all H floats. Existing native width70 is44,663/39,622; widening alone does not identify this pair-interface mechanism.

The two-frame single is not guaranteed to emulate the complete four-route nonlinear model. The full symmetric bilinear single already defeats any claimed linear necessity; a deterministic grouped network can spell the complete committee identically. Parameter counts do not match graph work, optimizer storage, activation memory or expressive capacity.

## One prospective representative screen

If separately adopted, use **complete ogbl-collab, base block0 only**, native width64/depth1/full supports/100 epochs (1,700 native updates), the existing masking, negative sampler, native Adam rates and full VALID Hits@50 selector. No graph subset, coefficient sweep, early proxy, new loss or TEST access. Keep private completion routing fixed; do not rerun a private/pool hypothesis.

Seven served arms are native single64 (member0 donor), F4, shared-H F4, private-H F4, the capable single63, independent native4, and independent native4+H. This is **12 unique fits** before compatible predetermined baseline reuse: three F4, one capable single, and two four-model banks. Independent seeds are0,5,10,15; both banks retain the existing100 synchronized candidates plus candidate101 assembled from individual validation winners. Use all eligible original baseline fits only if the parent establishes exact source/recipe/selection compatibility, never favorable-checkpoint reuse. No original outcome artifact was opened here.

Compare the **served mean raw logits**, validation-selected Hits@50, and a fixed diagnostic balanced predictive log loss from the same served score. Retain individual native competence and all selections. Charge H application/caching for every queried endpoint, its backward/optimizer state, all recursive/native member work, failures, wall time and peak memory; credit independent and single controls for their own eligible caching. H costs O(d) per endpoint, not zero work.

A null private-versus-shared contrast, an equally good capable single, or utility available only with extra selection/tuning defeats the proposed explanation. A one-block positive result could justify only a later paired assessment; it does not establish generality, factor necessity or a quality–cost advantage. The packet now contains one standalone adapter and one bounded CPU engineering fixture at the parent's request. It creates no experiment freeze, fit runner or launch; ordinary and single controls remain specifications.
