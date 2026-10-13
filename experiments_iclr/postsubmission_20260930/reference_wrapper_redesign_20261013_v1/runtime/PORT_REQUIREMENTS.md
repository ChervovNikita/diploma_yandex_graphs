# Focused source review: native-preserving boundary ports

This is one source review of the portable core against the saved native models. It adds port requirements only. No runtime module/model import, model construction, numerical match/tolerance work, fit, remote access or reviewer was used. No core/training code, roster, build or publication action was changed.

## Finding

The [core](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/reference_wrapper_redesign_20261013_v1/runtime/boundary_core.py>) preserves native objects and equations for its supported exact Linear/Embedding leaves: it adds hooks rather than rebuilding maps, retains native b, does not change categorical IDs, and leaves native forward/output activation in place. The [four training helpers](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/reference_wrapper_redesign_20261013_v1/runtime/training_rules.py>) collect block gradients at one old parameter state and make one optimizer transition. Those properties do **not** establish that every saved model has two suitable leaf boundaries or a compatible native task-call pattern.

**PolyFormer has a direct minimal binding candidate. OGB's categorical encoder, NCN's input/call pattern, NCNC's reused scorer, and BUDDY's parallel input stems require explicit source ports.** Saved models support preparing those ports; their current per-preset adapters/providers remain unqualified. No architecture or placement search is needed, and no extra hypothesis/reference is proposed.

## Semantic risks and required treatment

| Risk | What can change silently | Required source treatment |
| --- | --- | --- |
| Categorical multi-field encoder | One chosen AtomEncoder field becomes the supposed whole input boundary; integer IDs or native encodings can be changed | Keep all native fields and encoding/sum; pin dependency and logical boundary before implementing its factors |
| Reused output leaf | NCNC output factors also alter detached internal completion; last captured tensor may belong to an auxiliary call | Apply/capture the final task boundary in an explicit call scope; preserve recursion and source task-call order |
| Native regularizer/reduction | R_native or weight decay is counted twice; flat loss replaces native positive/negative sums or observed-target masks | Separate task loss from explicit regularizer; source optimizer decay remains optimizer decay; reuse exact masks/reductions |
| Mean-logit serving | Probability pool or an internal clamp replaces final logits; activation runs twice; rows/classes are misaligned | Fixed final-task raw-logit mean for GNNM/I4, authenticated IDs/axes, native M1 output unchanged |
| BN state | Shared running buffers update four times; private copies, freezing or concatenated routes change the native state law | Shared original BN and exact declared route/call order, unchanged momentum/counters/eval behavior, full-state checkpoints |
| Parallel/input projection boundaries | One convenient branch or newly invented Linear becomes the input stem | Pin the full logical source input boundary, with required native branch/projection interfaces and no placement search |

### Regularizer and aggregation details

The helper's algebra is correct **if** F/L callbacks contain task loss only and `native_regularizer` contains the source explicit penalty only: each F/L includes that scalar once, so `0.5F+0.5L` includes it once. A callback that already includes it doubles the penalty. Optimizer weight decay must not be supplied again as an explicit loss regularizer. Native positive/negative or observed-target reductions remain callback responsibilities.

The reviewed PolyFormer, OGB graph head, NCN/NCNC task predictor and BUDDY final head return task logits. For these sources, source CE/BCEWithLogits or log-sigmoid loss consumes those logits directly; add no extra sigmoid/softmax before a logits loss. Internal NCNC clamp probabilities are a different object. Mean raw task logits is the predetermined new comparison law for both GNNM and ordinary I4; it must not be presented as a reproduction of an unspecified published ensemble law. Native M1 and historical scores stay untouched.

### BN details

Keeping the same BN module preserves its equation and parameter/buffer objects, but four route forwards advance its native running state four times. Native repeated updates and route order must be declared rather than hidden. Source-preserving preparation keeps the original shared BN with its native call semantics; it does not transplant private BN from an earlier all-map implementation. Independent ordinary I4 retains four complete independently trained native BN states. The core has not verified state/checkpoint/selection behavior for any such frontend.

## Minimal ports supported by saved sources

### PolyFormer: direct leaf candidate

[PolyFormer native model](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/coordinate_source_independent_review_v1/strong_backbones_v1/sources/polyformer_code/node_classification/mymodels.py>) stacks `data.list_mat` into [N,K,D], calls **lin1** once, retains all attention blocks, then uses lin2/ReLU/native dropout before **lin3** once. Both boundaries are native nn.Linear; the core's feature factor broadcasts across the native polynomial slots. lin3 returns raw class logits and supplies a clear passive prehead capture. No graph/attention/width/depth change is required.

A minimal frontend can bind lin1/lin3 around the original instance and keep [its author training program](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/coordinate_source_independent_review_v1/strong_backbones_v1/sources/polyformer_code/node_classification/training.py>), including ordinary/attention parameter groups, preprocessing, splits, RNG policy, horizon and selector. This is PolyFormer; deferred Polynormer head/horizon issues are not resolved by this finding. No runtime qualification was performed.

### OGB atom encoder: logical categorical-input port

[The saved convolution/encoder source](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/training_pipeline_v2_20261012/preparation/ogb_native_source_readonly_root_20261012_v1/source/conv.py>) calls `gnn_node.atom_encoder` on full categorical `batched_data.x`, with native BondEncoder, BN and optional virtual-node behavior. [The graph model](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/training_pipeline_v2_20261012/preparation/ogb_native_source_readonly_root_20261012_v1/source/gnn.py>) has a clear **graph_pred_linear** output leaf. The input is a composite package AtomEncoder, not a single supported Embedding leaf. Its dependency definition/version and all field embeddings/summation must be pinned; that definition is not included in the three-file author-model capture read here.

A small dedicated logical-encoder adapter is feasible while retaining native categorical encoding, but choosing one field or silently replacing the encoder with a float stem is not a valid port. Internal bond/virtual-node maps remain native. Preserve [the source training loop](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/training_pipeline_v2_20261012/preparation/ogb_native_source_readonly_root_20261012_v1/source/main_pyg.py>): observed-target mask, FP32 loss casts, skipped-batch conditions, loader and Adam settings. Hidden node SupCon is ineligible for these molecular tasks. Current core alone does not implement this input boundary.

### NCN: composite/call-order and input-projection port

[The NCN model source](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/graph_link_competing_family_source_20261003_v1/source/ncn__model.py>) often gives its Collab GCN an `xemb` containing dropout only; the first learned projection is inside the native PyG GCNConv, outside the core's exact nn.Linear type support. The final `predictor.lin` Linear is clear, but input placement must be pinned to the native source interface, not invented by adding a new stem or factoring internal decoder maps.

[The source training call](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/graph_link_competing_family_source_20261003_v1/source/ncn__NeighborOverlap.py>) computes the encoder once, then calls positive and negative predictors separately, and sums their two mean log-sigmoid losses. A frontend must preserve those calls, their RNG/support sampling and the separate native optimizer groups. The current exactly-once leaf check cannot be satisfied by concatenating queries without changing source execution. A minimal fixed PyG/native-input and task-call capture adapter is feasible from the saved sources; it is not implemented/qualified here.

### NCNC: reused scorer needs task-only scope

In the same saved source, IncompleteCN1Predictor recursively calls `self.forward` under `no_grad` for residual-support completion, applies the native clamp, then calls the same final head for the task. Unconditional hooks on that head would apply output factors to internal completion too. This would extend the nominal output boundary into a native internal computation; the current count guard rejects the repeated calls.

A per-source adapter must scope or apply the boundary correction to final task logits using the existing final prehead/W/b, while retaining all native internal scorer calls, recursion depth, no_grad, clamps and support/sample rules. Merely removing the count guard, or using grad-enabled status to distinguish calls, is insufficient because serving also runs under no_grad. Preserve positive/negative task calls and their native order. The source is sufficient to design this adapter, but the portable two-leaf hook core is not that adapter.

### BUDDY: parallel input stems and shared native BN

[The saved author BUDDY](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/buddy_shared_cache_execution_v5/vendor/models_elph.py.txt>) has a structural stem **label_lin_layer**, an enabled feature stem **lin_feat**, possible source-configured embedding/SIGN/RA branches, and final **lin**. The whole input has multiple branches. A complete logical input boundary must list those native enabled branch interfaces ahead of time; binding only a convenient structural branch is a different partial-boundary definition.

The current core's exactly-one-input-leaf API needs a fixed source frontend here. Preserve endpoint products, normalized structural features, enabled branches, source cache/sampling and native width. The old [FactorizedBUDDY port](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/buddy_shared_cache_execution_v5/models.py>) clones W/b, factors every map and privately copies BN; none of that constructor belongs in the new wrapper. Its helper also forces sign_k/RA/embedding options, which cannot override a different adopted author preset. Minimal source-preserving preparation is feasible, but branch mapping, shared BN state and native data frontend remain required.

## Scope of the result

This review does not change the four hypotheses, three primary references, parent roster or fixed native settings. Exact per-source site/call mappings, native construction/optimizer/state frontends, checkpoint/selector interfaces and complete native data providers are still required. No additional diagnostics or tolerance audit is proposed. Source hashes and structured risks/port requirements are in [PORT_REQUIREMENTS.json](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/reference_wrapper_redesign_20261013_v1/runtime/PORT_REQUIREMENTS.json>).
