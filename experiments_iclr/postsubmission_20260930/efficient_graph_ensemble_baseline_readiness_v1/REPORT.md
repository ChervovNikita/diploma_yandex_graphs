# Efficient graph ensemble baseline readiness v1

Source-only assessment, 2 October 2026. Consulted `literature_memory/index_v13` first. This packet proposes a prospective comparison; it does not amend, execute, or evaluate the current cohort.

## Recommendation

**First priority: four untied smaller native models packed without cross-member mixing, following Packed-Ensembles with gamma = 1.** Apply this separately to PolyFormer-Mono/SquirrelFiltered and Polynormer-r/AmazonPhoto. This is a **declared graph adaptation of a published efficient ensemble construction**, not a published native PolyFormer/Polynormer ensemble or an exact reproduction of the image experiments.

**Second priority: MIMO on complete cached polynomial-token rows for PolyFormer.** Its input/output construction can faithfully implement the published MIMO algorithm because the inspected learned PolyFormer body acts independently on each cached node row. Label it **MIMO on PolyFormer polynomial-token examples, a declared graph-data/backbone port**. A matching native MIMO comparison on Polynormer's continuing graph/global interactions is not ready: independently permuting node features while keeping its graph fixed is invalid, and repeating the same graph/features in every slot throughout training gives a multihead model rather than MIMO's independent-input training.

The strongest practical cross-backbone comparator is therefore the Packed-Ensembles port. The saved graph-native precedents remain important, but none of their inspected native recipes is presently a faithful drop-in efficient comparator on these two backbones. This is an applicability conclusion about the inspected sources, not an absence claim about the literature.

## Published construction versus graph port

| Comparator | Primary evidence retained | Fidelity boundary and readiness |
|---|---|---|
| **Packed-Ensembles**, Laurent et al., ICLR 2023; [arXiv 2210.09184v4](https://arxiv.org/html/2210.09184v4), 23 September 2025 | Cached current-history v4 §§3.1–3.3; Appendix B subgroup constraints; Appendix K training/selection distinctions. Current author-linked library files are immutable-pinned and byte-identical to the retained 30 September sources. | Published method: disjoint smaller subnetworks, grouped operations propagated through the network, independent actions for normalization, and probability averaging. Faithful graph port requires private attention, norms, biases and all learned maps. No runtime qualification or implementation is supplied by this note. |
| **MIMO**, Havasi et al., ICLR 2021; [arXiv 2010.06610v2](https://arxiv.org/html/2010.06610v2), 4 August 2021 | Cached current-history v2 §2 and §3.5; input permutation, matching-label, member-summed CE and probability-pooling author source. Current Uncertainty Baselines files match saved bytes. | Published method: concatenate independently paired inputs, unchanged hidden body, multiple matching outputs, repeated input at inference. Apply to complete cached PolyFormer token examples; disclose the backbone, graph preprocessing, optimizer/schedule and any repetition choices as the port's contract. Original image timing or reported independence does not transfer. |
| **Kim 2023, Efficient ensemble for graph neural networks**, KAIST thesis, [handle 10203/308201](https://koasas.kaist.ac.kr/handle/10203/308201) | Reused official-viewer method/setup conclusions: shared W with member-private rank-one modulation in GCN/GIN/GAT; member-indexed states; duplicated full graph; probability pooling. | Direct native graph ensemble predecessor. Different backbones/tasks. Exact factor initialization, normalization statistics, objective details and author implementation remain unresolved. All-layer factors on these transformers must be called a **Kim-inspired graph-layer BatchEnsemble port**. The existing boundary-only arm does not reproduce its graph-layer placement. No repeat generic BatchEnsemble or TabM paper read was performed. |
| **GEENI**, Nagarajan et al., DAC 2022, [DOI 10.1145/3489517.3530416](https://doi.org/10.1145/3489517.3530416) | Reused publisher/indexed abstract and access disposition: likely error-node isolation, suppression of outgoing messages, approximate ensemble constituents. | Published efficient node-graph lead. Native node selection, approximation, diversity construction and executable recipe remain unverified. Do not invent a heuristic and label it GEENI. No current method-body reread was possible or claimed. |
| **FAGEL**, Shen et al., ECML PKDD 2026, [DOI 10.1007/978-3-032-37657-2_35](https://doi.org/10.1007/978-3-032-37657-2_35) | Reused abstract/README and pinned GCN source assessment: staged sampling, snapshots, learned residual logit fusion; source alternates samplers immediately and stops at first plateau. First online 8 September 2026; publisher citation/copyright says 2027. | Native graph lead, not source-qualified as the described published method. Full paper and prose/code discrepancy remain unresolved. Its sampled GCN trajectory/fusion is not the full-graph transformer recipe; transplanting it changes architecture, sampling, selection and fusion costs. |
| **Graph ensemble neural network**, Duan et al., Information Fusion 2024, [DOI 10.1016/j.inffus.2024.102461](https://doi.org/10.1016/j.inffus.2024.102461) | Reused metadata-only disposition. A bounded discovery found the SSRN precursor DOI `10.2139/ssrn.4535927`; both new SSRN landing/PDF attempts returned HTTP 403. | No method applicability inferred from the title. The SSRN lead does not close the earlier primary-access gap. |
| **Graph shallow ensembles**, Vinchurkar et al., [arXiv 2504.12627v1](https://arxiv.org/html/2504.12627v1); Schäfer et al., [arXiv 2602.15747v1](https://arxiv.org/html/2602.15747v1) | Reused scoped graph/atomistic conclusions: one graph backbone plus a last-layer committee, regression/UQ objectives. Original DPOSE source is already pinned separately. | Useful one-trunk/private-head architectural control. Node CE and class-probability pooling are declared adaptations; atomistic Gaussian NLL is not a faithful class-label recipe. These are secondary to the two requested efficient constructions. |

## PolyFormer: a valid MIMO example boundary

The retained monomial preprocessing produces the feature-only cache

\[
U_v=(X_v,(AX)_v,\ldots,(A^{12}X)_v)\in\mathbb R^{13\times F}.
\]

`native_polyformer_preprocess.py:17–30` constructs this cache. `native_polyformer_outer.py:29–43` applies a shared input map to each token, runs token attention/FFN blocks, sums the token axis and classifies. `native_polyformer.py:39–60,89–110` shows that attention is over the 13 tokens of each row, without a node-to-node learned operation or normalization across node rows.

For TRAIN indices independently paired according to the predeclared MIMO repetition contract, build each tuple by concatenating **all feature channels of each complete token row**:

\[
\widetilde U_{k,:}=\operatorname{concat}(U_{v_1,k,:},\ldots,U_{v_M,k,:}),
\qquad k=0,\ldots,12.
\]

Replace only the first map `F -> H` by `MF -> H` and final map `H -> C` by `H -> MC`; preserve token count/order and the hidden body. Head m predicts `y[v_m]`; training uses the published mean-over-tuples, **sum-over-heads** CE plus its declared regularization. Inference repeats the whole target-node token row across all slots and averages class probabilities. At input repetition rho = 1 throughout training, this construction loses the independent-input distinction; it must not be the only purported MIMO control. Optional nonzero rho and batch repetition require prospective choices and counted tuning/example exposure.

Graph caching uses the allowed graph/features, including transductive feature context when the existing protocol allows it. Tuple supervision uses TRAIN labels only. Cache rows must be paired after preprocessing; separately shuffling raw node features against the original adjacency does not preserve a complete graph-context example. No warm learned embedding is needed or proposed for this comparator.

This establishes fidelity of the **MIMO construction**, not statistical independence, model quality, or a native graph publication. Switching the current full-row schedule to a minibatch schedule, changing loss reduction from summed to averaged heads, or introducing a new optimizer requires a declared port choice; it cannot be hidden as an implementation detail.

## Polynormer: why a direct MIMO port is unresolved

The inspected native local path uses `GATConv(x, edge_index)` at every local layer (`native_polynormer.py:148–168`). Its global path forms K/V reductions and denominators over the node axis (`54–81`). For an independent graph permutation P, a coherent view requires **both** `PX` and `PAP^T`, matching labels and all associated indexing. One shared unchanged adjacency acting on independently permuted feature slots cannot represent those coherent views.

Duplicating the same intact graph in disjoint components does not supply independent training examples for a single-graph transductive task. Moreover, flattening components into the global node axis makes attention sum across graphs unless additional masking or per-component reductions are introduced. A post-trunk MIMO head changes where the independent-input method is applied and does not test end-to-end MIMO on the native backbone. Whole sampled subgraphs or a decoupled feature encoder could define another port, but both change the native task/context or learned architecture. Neither is recommended for this bounded comparison.

## Packed-Ensembles adaptation contract

Use M = 4 and gamma = 1 initially. For baseline hidden width H, the standard alpha = 2 point uses private width `w = alpha H / M = H/2`. Reduce every width-dependent map consistently: PolyFormer FFN width 128 becomes 64 at H = 128; preserve 13 tokens, two blocks, four attention heads and all filter preprocessing. Polynormer keeps eight heads with hidden channels per head 32 (total width 256), seven local layers, two global layers, qk sharing **within each member**, beta strategy, and the complete local-to-global schedule. These are declared width changes, not additional learned sharing.

- **PolyFormer:** share only the feature-only polynomial cache. Keep separate member token representations, token-wise MLPs, Q/K projections, attention scores, bias scales, FFNs, LayerNorm statistics/affines, intermediate readout and classifier. A LayerNorm over concatenated member channels would mix members and violate the independent subnetwork construction.
- **Polynormer:** reuse topology/indexing but keep member-specific GAT projections, attention parameters/coefficient softmax, beta vectors, dense transforms, norms, local/global heads and optimizer slots. K/V global statistics and denominators must be computed separately for each member. Sharing learned attention coefficients or flattening member/node axes is not faithful packing.
- Grouped maps or a batched member axis are execution choices for the same four untied functions. One Python `forward` call is not evidence of one model's compute. The packed output heads must map to the correct member/class order; the official `PackedLinear(last=True)` places estimators into the batch axis, which is not an existing native transformer tensor contract.
- Independent initialization and dropout must be declared. Same training node set/batch is consistent with the paper's Appendix K, which also selects a synchronized ensemble checkpoint. Probability averaging is the published reducer. There is no requirement to fabricate bootstrap graph samples or add a learned gate.

## Actual budgets, not a hidden-layer proxy

The paper's interior dense/convolution count scales as

\[
P_{\rm interior,PE}/P_{\rm interior,single}=\alpha^2/(M\gamma).
\]

At (alpha,M,gamma) = (2,4,1) this equals one, but the first and last layers, biases, norms and head-specific terms do not all obey that quadratic formula. If the complete scaled native model has `P(w) = a w^2 + b w + c`, then four half-width copies have `a H^2 + 2 b H + 4 c`, rather than `a H^2 + b H + c`. Cache and activation bytes are additional resources.

As a source-only algebra example, the inspected PolyFormer configuration with K+1 = 13, two blocks, four heads, FFN width H/2 and token-MLP multiplier 1 gives

\[
P_{\rm PolyFormer}(H)=59H^2+(F+C+65)H+104+C.
\]

With the saved configuration F = 2089, C = 5, H = 256, this is **4,419,437** native parameters. Four H = 128 copies have **4,972,468**, about **12.5% more**. This is a manual source-derived count, not a model execution or a current-cohort measurement; a future implementation must bind its actual admitted parameter contract. Attention tensors/edge work and optimizer memory also need accounting.

MIMO changes its stem and head sizes by

\[
\Delta P=(M-1)(FH+HC+C).
\]

For the same PolyFormer example at M = 4 this is **1,608,207 additional parameters**, about **36.4%** of its native model. The stem runs on every polynomial token. The image paper's negligible-overhead example therefore cannot justify a near-free claim here. Batch/input repetition changes training exposure and work even when the body has one tuple forward.

## Practical prospective matched-budget comparison

1. **Retain the single native model, current boundary-factor family, and four same-width independent models as references.** Packing the same-width independent family is a serving/training implementation comparison for identical functions, not a new statistical method arm. Do not count it twice as method evidence.
2. **Add one primary PE graph-port arm on each backbone if continuation is authorized.** Start from the interpretable half-width operating point, but distinguish it from an exact budget match. For a strict parameter/storage cap, precompute the complete source-derived count and choose the largest width satisfying the incumbent cap and head/FFN divisibility. This width rule uses no validation performance. Do not equate an approximate count with equal runtime.
3. **Use a separate resource frontier for runtime/memory.** Charge the complete native training/stage schedule, member-example exposures, tuple repetitions, preprocessing/caches, optimizer state, all selector forwards and inference aggregation. Measure cold and cache-reuse serving costs under the same hardware, precision and batching. Width-based parameter matching, fixed-example training and measured time caps answer different questions; report which constraint each operating point matches. Apply the same optimized backend effort to the untied controls.
4. **If a second construction is affordable, add only the valid PolyFormer-token MIMO port.** Preserve the native hidden body initially; declare rho and batch repetition in advance (rho = 0, one batch copy is the clean construction default). Native head-summed CE has a different scale from mean member CE; record that scale and its optimizer/regularization consequences. Any repetition or LR alternatives consume the same prospective configuration budget as other families, rather than an extra free search.
5. **Keep label roles, seeds, configuration allocation and family checkpoint selection matched.** The saved amendment uses predictor-validation pooled NLL and synchronized family checkpoints. Preserve those principles in a new prospective document if continued; do not select from final-pool labels. Source image split ratios, augmentations, SGD schedules and speed measurements are not claims for the current graph roles.
6. **Predeclare pooling.** A native-construction comparison uses probability averaging for PE/MIMO and applies that same reducer to all comparison families. A separate common raw-logit-pooling analysis must be labeled a protocol adaptation for PE/MIMO. Do not optimize the pooling rule independently on held-out outcomes or change the current cohort's fixed reducer.

This is a practical ordering of source-ready adaptations, not an execution request. No numerical model utility, statistical superiority, runtime advantage, acceptance judgment or novelty absence conclusion follows from this packet.

## Source custody and bounded retrieval

- MIMO current repository head: `google/uncertainty-baselines@e3d4be1ca05604ebe129d32a53c3b314a05ce232` (29 September 2026). Two immutable raw files match the saved bytes.
- Packed-Ensembles author-linked library head: `torch-uncertainty/torch-uncertainty@3f82fe5d15a7bf877a821731baadef1e4731c31d` (7 August 2026). Two immutable raw files match the saved bytes. This pins the inspected library implementation, not the exact ICLR-2023 experiment checkout.
- Native architectures reuse the retained author pins `air029/PolyFormer@d390f39e88d0eaac80318fdc7704bd3bf3cf8b13` and `cornell-zhang/Polynormer@fc8c276c9c5dfbd616d83f65338a3392188a5e08` and their bound local bodies. They are the study's retained identities; no update to their upstream heads was requested.
- Cached paper history identifies MIMO v2 and PE v4 as the latest versions in the retained 30 September history. Their scoped saved primary passages were used for the new applicability question; no new paper/version claim as of 2 October is inferred from an unchecked archive history.
- New discovery was bounded: two Crossref queries, two Bing queries, two DBLP queries and one OpenAlex query. Bing returned unrelated partial-keyword results; DBLP returned bot challenges despite HTTP 200; the other searches were noisy. Only the SSRN precursor was a concrete additional lead; it remained inaccessible. These failures are preserved, not interpreted as absence evidence.

`READ_SCOPES.json` identifies every relied-upon primary/code slice and reused conclusion. `SOURCE_PINS.json`, `INPUT_BINDINGS.json` and `RETRIEVAL_LOG.json` bind immutable files, local custody and acquisition outcomes. All outputs are new in this directory. No author code was run, and no labels, arrays, checkpoints, SSH, GPU, tests, manuscript or live study state were accessed or changed.
