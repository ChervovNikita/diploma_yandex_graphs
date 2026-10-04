# Focused graph-specific prior assessment of the count-conditioned auxiliary

## Decision

A narrow adaptation hypothesis survives the inspected prior work: use an exact count-conditioned identity law on the **actual two-side native TRAIN residual support**, share one component responsibility across those sides, and transfer that supervision through the native scorer while retaining its existing count-free completion and target readout. Utility is unestablished. The evidence does not clear a generic novel graph likelihood, coherent completion method, shared latent pattern model, or count-free serving claim.

New primary code reading sharpens two exclusions. MaskGAE already updates a served graph encoder using masked topology and degree supervision, then scores edges without a degree/count input. GRAN already uses graph-conditioned mixture components and samples one component for a complete generated block. The composition with the specified native residual support remains a falsifiable adaptation question; absence of an inspected duplicate is not its scientific contribution.

The declared sequential Single is normalized and identity-sensitive, but sorting numerical node IDs does not make it equivariant to global node relabeling. Its failure alone therefore cannot establish an ordering-independent or equivariant bank advantage.

## What was newly read

The retained assessments and exact canonical records in `literature_memory/index_v44/LITERATURE_INDEX.json` were checked before retrieval. PIFM's method, Algorithms 1–5, refinement and stitcher were already covered; CAM, KREPE and ARK method scopes were also already covered. Those conclusions are reused with their original limitations. There are **zero new primary paper identities, zero new paper-method reads and zero full-paper reads** in this packet.

The new primary scopes are static author code from two repositories:

- **MaskGAE**, official `EdisonLeeeee/MaskGAE` snapshot `343ced7028681f052b7e9ea47945be7fc5166213` (commit date 2024-10-29): encoder/input interface, edge/degree decoders, objective dispatch, masked training update, edge scoring, edge/path mask provider, selectable losses, and the OGB construction/optimizer binding. Its code had not been semantically read in the consulted index. Exact ranges are in `READ_SCOPES.json`; excerpts are retained.
- **GRAN**, `lrjconan/GRAN` snapshot `70e0dd24bb357aeaacfcd585e77e4ebd0c55f8f8`: data-loader order/support construction, generator inference/sampling/forward, and the empirical graph-node count PMF binding. Prior code reading covered only constructor loss lines 190–203 and mixture loss lines 461–550. The new generator scope is lines 204–458, with no overlap. The previously saved generator file was copied and rehashed; this copy is not counted as a new retrieval.

All new claim-bearing code ranges were read in full. Earlier symbol/regex locators had some truncated outputs, which are not used as semantic evidence. Full snapshot retrieval, tree metadata and hashing are mechanical operations. Neither the author modules nor project science modules were imported or executed. No dataset, graph tensor, model state, project score, official TEST artifact, scientific server or experiment was accessed. Source containing public evaluation routines was read only to establish the scoring interface.

## Primary operational findings

### MaskGAE: count/degree training and count-free edge scoring are established

`MaskGAE.train_step` obtains remaining and masked edges, encodes the remaining graph, scores minibatched masked positives and sampled negatives, then adds masked-degree MSE and backpropagates the combined loss ([model.py lines323–381](https://github.com/EdisonLeeeee/MaskGAE/blob/343ced7028681f052b7e9ea47945be7fc5166213/maskgae/model.py#L323)). The degree target is `degree(masked_edges[1], num_nodes)`, built from the masked edge set; it is not the proposed per-query pair of residual side counts. The author training binding optimizes `model.parameters()` ([train_linkpred_ogb.py lines16–35](https://github.com/EdisonLeeeee/MaskGAE/blob/343ced7028681f052b7e9ea47945be7fc5166213/train_linkpred_ogb.py#L16)). Thus training auxiliary information can update the same encoder used for edge prediction.

`batch_predict` calls only the edge decoder. The OGB scoring routine encodes `data.x,data.edge_index` and calls that edge scorer for each candidate; it never invokes the degree decoder or supplies teacher counts ([model.py lines383–424](https://github.com/EdisonLeeeee/MaskGAE/blob/343ced7028681f052b7e9ea47945be7fc5166213/maskgae/model.py#L383)). This directly defeats **count-free serving alone** as a useful distinguishing claim. It also defeats the standalone claim that topology/count supervision can improve a served graph representation without serving the auxiliary decoder.

The new source read adds a recipe qualification to the saved paper assessment: the class default is sampled-positive/negative BCE, but this OGB entry point selects **paired AUC loss**, a dot-product edge decoder, learned per-node embeddings, and the simple random negative sampler ([construction lines158–179](https://github.com/EdisonLeeeee/MaskGAE/blob/343ced7028681f052b7e9ea47945be7fc5166213/train_linkpred_ogb.py#L158); [loss.py](https://github.com/EdisonLeeeee/MaskGAE/blob/343ced7028681f052b7e9ea47945be7fc5166213/maskgae/loss.py)). The simple sampler draws endpoint IDs and does not exclude observed edges (`model.py:273–275`). Therefore the paper's BCE description cannot be substituted silently for this code recipe or treated as a native NCNC reproduction. This bounded read does not audit the entry point's complete split/context provenance.

Its edge/path mask provider operates on the supplied edge list, with path sorting and optional undirected reconstruction of the remaining context ([mask.py lines16–107](https://github.com/EdisonLeeeee/MaskGAE/blob/343ced7028681f052b7e9ea47945be7fc5166213/maskgae/mask.py#L16)). This is different from the permitted native record-removal process, surviving duplicates/coalescing, query-specific residual supports and complete source-incidence bit teacher. No matched adapter is established. The possible adaptation distinction lies in that exact support, teacher and conditional association objective, not in generic graph masking or omitted degree inputs at serving.

### GRAN: shared graph-pattern components are established; its support and readout differ

The new generator source constructs graph-conditioned component logits using a GNN and ordered block attention features (`model/gran_mixture_bernoulli.py:204–262`). During sampling it averages component gate logits over the block, draws one component, selects that component's probabilities for every block edge, then draws Bernoulli edges ([lines348–365](https://github.com/lrjconan/GRAN/blob/70e0dd24bb357aeaacfcd585e77e4ebd0c55f8f8/model/gran_mixture_bernoulli.py#L348)). Combined with the already read mixture likelihood, this directly establishes graph-pattern component responsibilities and shared component selection as prior operations.

The data loader builds labels and context for ordered graph-growth blocks, including all pair positions for the newly generated rows (`dataset/gran_data.py:188–282`). It chooses original, degree, BFS, DFS and k-core orderings ([lines58–141](https://github.com/lrjconan/GRAN/blob/70e0dd24bb357aeaacfcd585e77e4ebd0c55f8f8/dataset/gran_data.py#L58)). These are neither native target-query residual slots nor two teacher-count-conditioned side subsets. Degree ties, graph iteration, component order and BFS/DFS neighbor traversal are not resolved by a certified graph-isomorphism canonicalizer in this scope. A finite heuristic order bank does not itself prove invariance to every equivalent graph representation.

GRAN's readout generates an adjacency, then samples a **number of graph nodes** from an empirical TRAIN PMF and truncates the adjacency (`gran_mixture_bernoulli.py:448–458`; `runner/gran_runner.py:139–141`). This is not teacher residual count use at native link inference. Its sampler is part of served graph generation; the proposed auxiliary is absent from served native edge scoring. The actual residual adaptation and unchanged recursive ranker are the remaining differences. No untested runtime or predictive advantage is inferred from them.

### PIFM and recent coherent methods: stronger alternatives already exist

The retained [PIFM v2 method](https://arxiv.org/html/2601.22107v2) uses an NCNC-informed adjacency prior and graph-conditioned flow refinement. Previously read author inference performs repeated dense subgraph denoising and anchors observed entries; its stitcher averages logits over subgraphs covering an edge. The inspected source optimizes the denoiser/feature adapter and uses cached prior embeddings; it does not establish an end-to-end auxiliary gradient into a tied native NCNC bank. This source snapshot also has the retained TRAIN ownership, candidate coverage and recipe qualifications. These are adaptation requirements, not evidence of a scientifically weaker prior.

The retained [CAM](https://arxiv.org/html/2405.19375v4), [KREPE](https://arxiv.org/html/2605.24064v1) and [ARK/SAIL](https://arxiv.org/html/2602.06707v1) methods establish globally contextual linksets, any-order masked categorical completion, and autoregressive/shared-latent graph completion. Per-step BCE or categorical CE does not make their integrated contextual laws factorial. Their native supports, readouts and author-code qualification differ from the proposed residual auxiliary. They block generic coherent-completion novelty and justify a capable structured Single. Existing ARK random-order training does not supply an exact permutation-invariance theorem for the present Single.

## The precise association question

For side s and component m on the permitted visible context C, write

`q_ms(z_s | k_s,C) = exp(eta_ms · z_s) / e_ks(exp eta_ms)`, for `sum(z_s)=k_s`.

This is the attributed classical conditional Bernoulli/fixed-size-subset law. The current contract mixes the already conditioned components uniformly:

`Q_joint(z_L,z_R) = (1/M) sum_m q_mL(z_L) q_mR(z_R)`;

`Q_sep(z_L,z_R) = [(1/M) sum_m q_mL(z_L)] [(1/M) sum_m q_mR(z_R)]`.

At **fixed logits**, every whole-side distribution and unary marginal is exactly shared by these two laws. Their difference is cross-side association of the component identity. Separately fitting them changes the logits, so learned marginal matching is not guaranteed. Neither `W_K` nor a count-only potential supplies this same association isolation: memberwise NLL changes component training, while any potential depending only on the conditioned counts cancels from the normalized identity law.

Let `rho_joint,m` be the responsibility proportional to `q_mL(z_L)q_mR(z_R)`, `rho_s,m` the side responsibility proportional to `q_ms(z_s)`, and `mu_ms` the component's conditional bit expectation. With the contract's common per-query denominator d, the local logit-gradient difference is

`(rho_joint,m - rho_s,m) (mu_ms - z_s) / d`.

This describes the proposed transfer signal; it does not establish its size, relevance to the target gradient, optimizer behavior or served ranking effect. If either side is empty/extreme, its observed subset has probability 1 for every member and `Q_joint=Q_sep`. The same equality holds whenever one whole-side component law is identical across members. Nonextreme support geometry alone is insufficient.

Count conditioning removes the direct count likelihood and member/side common-offset signal. It does not calibrate counts or native sigmoid offsets. Native target supervision can still determine offsets, and nonlinear shared parameter updates can affect count-free ranking. **No count predictor or inference-count adapter is required by this auxiliary-only contract.** Uniformly mixing conditioned components is also not, in general, conditioning the old unconditioned mixture: the latter would reweight members by their count evidence.

The restricted law has limitations: component swap odds depend only on the swapped unary logits, and the two-side probability matrix has nonnegative rank at most M. Count-one/complement-one cases reduce to categorical identity laws. Local overlapping residual laws impose no cross-query graph-posterior compatibility. TRAIN zero labels remain unobserved TRAIN incidence, not verified latent nonedges.

## Permutation and Single-control conclusion

The exact bank subset law is invariant to consistently permuting candidate slots and teacher bits at fixed corresponding logits: the dot product and elementary symmetric normalizer do not depend on slot order. Turning this into full graph-relabel equivariance additionally requires the native encoder, scorer, support provider and record masks to transform consistently. This packet does not certify those upstream properties.

For `S_K`, distinguish three operations:

1. **Storage permutation with the same node IDs.** If canonicalization restores the same ascending ID order, first-slot reference and legal histories, the operation can be implementation-consistent.
2. **Global node relabeling.** Relabel graph, node features, endpoints, candidate identities and teacher consistently, then sort the new numerical IDs. This can change the traversal and first-slot reference. Numerical ID sorting is an order convention, not graph-isomorphism canonicalization.
3. **Endpoint exchange.** The original endpoint orientation, left-before-right sequence and unsymmetrized side features give no guaranteed endpoint-exchange invariance.

A direct analytic counterexample requires no experiment. Give one side three candidates `{a,b,c}`, count one, and an empty other side. Set all parameters of the declared head to zero, so every unforced choice has logistic probability 1/2. In visit order `(a,b,c)`, the normalized subset probabilities are

`P({a})=1/2`, `P({b})=(1/2)(1/2)=1/4`, `P({c})=(1/2)(1/2)=1/4`.

The last selection is forced if the first two were skipped. A consistent global relabeling that moves b to the smallest numerical ID changes the corresponding probability of `{b}` from 1/4 to 1/2. The declared family therefore **does not guarantee node-relabel equivariance**, even when the head ignores node features and history. This is a property of an allowed parameter setting, not a claim about a fitted run. First-slot unary centering can introduce another order-dependent reference; common-offset invariance of that feature does not remove this issue.

A teacher-forced ordered law can represent useful joint identities, and specifically learned chain-rule conditionals can represent an invariant subset law. The current engineering capability witness does not qualify all native supports, optimization, resource feasibility or symmetry. A competent order-dependent Single matching the bank can refute generic bank necessity. Its failure cannot isolate structural/equivariant necessity from order policy, prefix compression, capacity and optimization differences. Any future symmetry-matched comparison needs an explicit prospective policy and its qualification; this assessment neither redesigns the Single nor freezes extra arms.

## Falsifiable contribution boundary

The defensible claim is narrow: **sharing count-conditioned two-side identity responsibility on the declared native TRAIN residual supports yields useful transfer into the unchanged count-free native link ranker beyond independent-side mixtures**. A bank-specific necessity claim additionally requires an adequately matched, competent structured Single.

Better auxiliary identity NLL without better served ranking does not establish useful transfer. No material joint-over-independent-side benefit under competent fitting defeats the association claim in that recipe. A qualified Single matching the bank defeats necessity. A failed or unqualified Single, or one confounded by order, does not establish necessity. A difference in learned marginals after independent fitting cannot be advertised as a pure fixed-marginal association effect.

Those are evidentiary implications, not execution authorization or a new science screen. No fit, experiment, numeric result, manuscript/index edit, new arm, source redesign or science freeze was performed. The packet supplies new scoped primary code evidence, an exact symmetry counterexample and a bounded hypothesis whose predictive utility remains unresolved.
