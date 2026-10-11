# Node-MoE/GNNMoE: authoritative source prerequisite

2026-10-11. **Primary source inspection only.** The GNNMoE reference identity, map ownership, controller and multiclass loss/selection interfaces are resolved. Its public tree is incomplete for an executable reference. Node-MoE code and later publication remain unverified in the bounded routes. No model import, dataset download, fit, nomination, GPU/SSH, source repair or memory-contract change occurred.

## Exact identities and publication status

**GNNMoE:** the paper directly links [GISec-Team/GNNMoE](https://github.com/GISec-Team/GNNMoE). The repository README identifies the same title/authors and arXiv2412.08193. Snapshot commit is **`81b00fcbb7a2c16ceaa2ed6e99eb56d7c74f87a4`**,2025-06-17, on main. Four small source files were fetched by exact commit; their Git blob SHA1s match the published tree and their SHA256s are retained. No clone, dataset or image was fetched. The inspected repository is a reference implementation; equivalence to the paper's experimental revision was not certified.

The author README and publisher-deposited Crossref record agree on **Companion Proceedings of the ACM on Web Conference2025**,pp907–910, published2025-05-08, DOI **[10.1145/3701716.3715462](https://doi.org/10.1145/3701716.3715462)**. This is a published companion-proceedings article; no main-track or independent peer-review-quality claim follows.

**Node-MoE:** verified primary is [arXiv2406.03464v1](https://arxiv.org/abs/2406.03464), Haoyu Han et al.,2024-06-05. The arXiv record supplies no journal reference/comment; inspected HTML supplies no author code URL. Exact/alternate-name/readme GitHub searches located unrelated names or third-party paper lists, not an authoritative release. Crossref's exact-title query returned other titles; OpenAlex's exact match identifies the arXiv preprint only. Thus **author code and later publication are unlocated/unverified**, not proven absent. No similarly named repository is substituted. An incidental MORGAN metadata hit is not a method read or reference substitute.

## GNNMoE state and ownership contract

All locations below refer to the pinned local source; SOURCE_CONTRACT.json records URLs, hashes and line scopes.

| Interface | Primary-code fact |
| --- | --- |
| Common evolving state | `MoE.forward`,model/MoE.py298–322, starts one feature state and passes each weighted expert mixture into the next MoELayer. Experts receive the same incoming state within a block; their temporary outputs are combined and no separately evolving expert history survives that mixture. One end classifier emits the final scores. |
| Node controller | `GatingNetwork`,169–185, is Linear→ReLU→Linear→row-Softmax. `MoELayer.forward`,229–240, applies it to the current node state x before evaluating the four experts. Later controllers therefore see the preceding **mixed** state. There is no separate degree/homophily/random-walk pattern input in this gate. |
| P/T map ownership | `MoELayer.__init__`,211–221, constructs fresh PTNorm/TPNorm/TTNorm/PPNorm objects; `MoE.__init__`,272–280, constructs fresh blocks. PT's Linear, TP's Linear and both TT Linears are independently created objects; transformations are not tied between experts or layers. GAT propagation uses separate learned GATConv objects; SAGE PP also uses separate learned SAGEConv objects. Consequently formal “P” is not universally a weight-free operator across variants. |
| Class supervision | There is one combined classifier and one classification loss. `main.train`,13–22, backpropagates through the whole model with one AdamW. No independently pretrained bank or per-expert class CE appears in this script. |
| FFN ownership | Three independently constructed GLU modules own distinct w1/w2/w3 maps,243–261. This is more parameter freedom than merely changing an activation while tying all feature maps. |

**Loss ambiguity resolved:** model returns raw C-class logits for normal multiclass datasets. `main.py112–117` selects **`nn.CrossEntropyLoss()`**, applied on TRAIN at18. The four named binary datasets instead return squeezed sigmoid output and use **BCELoss**. The paper's blanket binary-CE prose therefore does not describe the multiclass native interface. The binary validation functions still use `argmax(dim=1)` despite that squeezed output; binary evaluation would require separate resolution. No binary correctness is certified here.

**Native selection interface:** main.py131–142 runs TRAIN then raw-logit argmax validation. A **strictly larger** validation accuracy resets patience and triggers another TEST forward; ties consume patience. Defaults are500epochs/patience100, but Linux NNI parameters can overwrite parsed settings. The script records best validation/test scalars, **does not save/restore a selected model state**, and evaluates TEST during training improvements. Future protected-role use cannot run this script unchanged: selection, state/RNG custody and final evaluation roles need a separately declared wrapper.

## Exact release limitations

1. **Missing P helpers:** MoE.py6 imports `SAGENorm,GCNNorm` from model, while published `model/__init__.py` contains only `from model.MoE import *`. Neither helper is defined or imported by the two fetched model-package files, and the published tree has no separately named helper modules. Their normalization/propagation semantics cannot be recovered from this release. This is a static unresolved dependency; no import was attempted and no replacement was invented.
2. **Intermediate activation results discarded:** PT38,TP69,TT101/103 call `F.dropout(F.relu(x),…)` without assigning its return. These calls do not alter x in normal functional semantics. Expert LayerNorm/router/FFN remain nonlinear; the entire model is not thereby linear. The code does not implement the intended intermediate ReLU/dropout path as the prose might suggest.
3. **Evaluation remains stochastic:** MoEFFNLayer259 calls hard `F.gumbel_softmax` unconditionally, including eval. Validation and the separate TEST call can therefore use different draws. Static inspection gives no restored-output identity or deterministic serving guarantee.
4. **Paper/release residual and configuration differences:** per-expert alpha residual expressions are commented out; the active native initial-state residual occurs after the FFN at316. `main.py` does not define n_layers/type required by the model; NNI may supply them. Unused registered adjacency-size Linears and other coordinates also exist. Do not infer a complete native configuration, effective parameter budget or paper-faithful runtime from the README alone.

These are fidelity/qualification boundaries, not requested repairs or adverse predictive outcomes. A later capable reference must pin an authoritative complete release or transparently attributed reimplementation, declare any changes before results, preserve the chosen loss/selector/roles, and qualify restored serving. This snapshot is usable source custody and interface evidence; it is **not yet an admitted executable benchmark**. Node-MoE's T ownership/native optimizer/selector remain unknown without authoritative code.

## Nearest collision and one distinct future question

The nearest verified collision is **GNNMoE's current-state controller over P/T expert operations followed by one mixed continuation**, with **independent expert transformations** in its public code. Node-MoE adds graph-conditioned learned-filter routing in the saved primary prose, but its unavailable code cannot establish transformation tying or native class-loss semantics.

The retained question is narrower: **could four private continuing states, each with an own class loss and node-conditioned P/T controller, acquire competent alternatives that are lost in one finite-width mixed continuation while sharing the dense transformations across routes?** Neither private evolving states nor graph-adaptive filters are new: native BE and the completed private-kernel study already contain them. The precise distinction is continuing separate predictions through repeated operation choices under explicit transformation tying, rather than immediately mixing expert outputs into one state. It can also be represented as one wider grouped network; no ensemble expressivity or originality claim follows.

A later answer needs capable Node-MoE/GNNMoE-style ordinary singles with the full same operator/continuation opportunity, plus genuinely fresh equally equipped independently selected I4 bodies. This code audit does not justify a new fit or define a recipe. Current memory execution retains priority.

## Custody

Versioned public metadata, source fetches, Git-tree/blob checks and exact inspected line scopes are retained. Only four code files and one README were fetched; dataset paths seen in tree metadata were not opened. Public search success/empty relevance prefixes are not exhaustive absence evidence. No numerical library import, model, dataset, optimizer, TEST truth, current partial score, reviewer or new agent was used; all earlier packets remain unchanged.
