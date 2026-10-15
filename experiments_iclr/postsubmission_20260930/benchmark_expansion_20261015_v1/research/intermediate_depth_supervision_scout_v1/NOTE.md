# Intermediate-depth supervision: a prospective competence mechanism

15 October 2026. **One optional research route: literature and static-source scout, with separately labelled later completed scalar context.** No candidate implementation, model import, fit, dataset/GPU access, remote scientific execution or campaign change occurred. PubMed remains next. This package does not reopen the completed structural study or change the stochastic-adapter package.

## Recommendation and limits

Test whether a small supervised readout of each live native GIN prefix improves the original final predictions. Add four shared affine training heads to the existing five-layer, width-300 MolBACE residual-only/private-BN bank. They add **1,204 trainable coordinates**, while selection and serving retain the original final head and four-route raw-logit mean. This is an adaptation of established deep supervision, not a new primitive or a proven project gain.

The mechanism hypothesis is modest: direct label gradients at intermediate native states may improve the shared feature basis and member competence. It does not require members to disagree. Common errors do not establish over-smoothing, vanishing gradients, or a particular optimization defect. Five GIN layers are not evidence of either diagnosis. Early label separability can also constrain useful later compositional features, overfit a small labelled sample, or make the members more correlated.

## Retained history and primary ancestry

The terminology/history search and its exclusions are saved in `HISTORY_SEARCH_SCOPE.json`; exact source identifiers were checked in literature index v72. This is a bounded overlap search, not global novelty clearance.

- The retained 5 October member/depth scout already proposes a **frozen member-by-depth nonlinear fusion readout**, with JK/DAGNN/GAMLP/VFusion ancestry. Here the native prefix is live, the auxiliary task changes its training gradient, and served predictions do not fuse depths.
- The 11 October private-memory attribution memo explicitly calls Physics GM4 graph-tour branches “deep supervision.” Its v1 specification and source README prescribe own final view CE and averaged final predictions; the inspected scope does not prescribe a classifier at every native GIN depth. Thus deep supervision is already in project vocabulary and architecture reasoning. This particular live-prefix placement was not found in the bounded retained search.
- Shared branches with own supervision and peer learning already have retained collaborative-learning ancestry. No fresh primary-paper credit is taken for those saved notes.

Two new primary method scopes were inspected:

| Primary source | Inspected mechanism | Difference from this proposal |
| --- | --- | --- |
| Lee et al., **Deeply-Supervised Nets**, [arXiv 1409.5185v2](https://arxiv.org/html/1409.5185v2), §2–2.2/Eqs. 1–7; [AISTATS metadata](https://proceedings.mlr.press/v38/lee15a.html) | Joint intermediate companion classifiers/losses and final classifier; acknowledges that early discriminativeness can conflict with final performance | Original displayed objective uses squared hinge, head norms, threshold and optional decay. Fixed native BCE with coefficient one, zero head starts, and no threshold/decay is an adaptation. No proof or result transfer is claimed. |
| Elinas and Bonilla, **Addressing Over-Smoothing in Graph Neural Networks via Deep Supervision**, [arXiv 2202.12508v1](https://arxiv.org/html/2202.12508v1), §4–4.2/Eqs. 5–9 | Graph pooling and linear predictors at each layer, mean supervised task losses, no companion-loss decay | DSGNN also serves an average across layer classifiers. This proposal serves only the original final head. Eq. 7 uses softmax outputs while the serving prose says average then softmax; exact logit/probability semantics remain unaudited without author code. |

One bounded author-code scope corroborates DSN wiring: [s9xie/DSN](https://github.com/s9xie/DSN), commit `065e49898d239f5c96be558616b2556eabc50351`, CIFAR train auxiliary heads/losses and complete validation graph. Training attaches `ip_svm1/2/3` losses to intermediate conv states; validation omits those heads and computes accuracy from final `pool3`. Its README also reports using softmax losses for other datasets. No Caffe operator/runtime, training result, downloaded weight, dataset or full-repository audit was performed. A bounded DSGNN repository search found no verified author implementation; that is not a claim that none exists. Saved bytes are broader than the inspected scopes. See `SOURCES.json`, `READ_SCOPE.json` and `RETRIEVAL_LEDGER.json`.

## Fixed method sketch

Use the paid residual-only/private-BN identity **F**, its rank-8 residuals, native five-layer GIN, width 300, four routes, native sampler/BN/dropout/Adam/data roles/controller, and final-head selection/serving. This proposal changes only the following training objective and its auxiliary parameters. It is not an executable or admitted specification.

For route m and intermediate depth l in {1,2,3,4}, let h[m,l] be the exact existing native post-BN/post-ReLU/post-dropout state after that layer (including any enabled native residual addition). Pool it with the existing native mean graph pooling and original graph batch IDs. Head a[l]: R^300 → R is shared across routes and private to its depth. Four heads, including biases, start at zero without consuming any native RNG.

Let B be exactly the native masked TRAIN BCE reduction over observed labels, not a new per-example reweighting. Then:

```
F_loss  = (1/4)  Σ_m B(final_logit[m], y)
A_depth = (1/16) Σ_m Σ_l=1..4 B(a[l](meanpool(h[m,l])), y)
J_depth = F_loss + A_depth
```

Every observed TRAIN graph still supervises every final route. No private target assignment, teacher, distillation target, feature repulsion, gating, additional graph input or auxiliary dropout is introduced. Use one scalar backward and the same native Adam transition for native/private/auxiliary parameters. Auxiliary weights and biases use the existing native output-anchor Adam options, explicitly recorded by future source preparation. Coefficient one and depths 1–4 are the single prospective choice; no grid is proposed.

The 1,204 extra FP32 parameters occupy 4,816 bytes, and two dense FP32 Adam moments occupy 9,632 bytes, excluding gradients, optimizer scalars, activations and allocator overhead. There is additional pooling/head/loss/backward work. No extra native forward, convolution, BN update, dropout draw or graph sample should occur. No latency or memory-efficiency claim follows until measured. The extra heads can be omitted from a serving export but must be checkpointed with optimizer state for training continuation. At the zero start, auxiliary head weights give zero extra backbone gradient on the first update; head gradients generally activate the feature path on later updates.

**Static implementation interface:** in the retained native `GNN_node.forward`, pre-hooks on `convs[1]` through `convs[4]` can observe the first positional input, which is the previous layer's complete native state. Raw convolution outputs before BN/dropout would be the wrong placement. Preserve the autograd connection; pool during the same route forward; clear captured tensors per forward. `gnn.py` provides the existing `pool` and final `graph_pred_linear` graph feature. Do not rewrite native execution or redraw stochastic operations. Hooks, route ownership, backward coverage, zero-start RNG identity, restore and auxiliary-free serving all require future source qualification; static inspection is not a runtime pass.

## Discriminating final-state control

**FINAL_AUX** uses the same four shared affine heads, zero starts, nominal coordinates, BCE reductions, coefficient, native route forwards and optimizer options. All four heads read the final depth-5 graph feature:

```
A_last = (1/16) Σ_m Σ_k=1..4 B(a[k](meanpool(h[m,5])), y)
J_last = F_loss + A_last
```

Use the original final head alone for selection and serving in both arms. F is the retained no-auxiliary baseline. DEPTH_AUX versus FINAL_AUX asks whether intermediate placement helps beyond extra supervised heads and loss mass. It does **not** norm-match backbone gradients, match every optimization path, or establish an ensemble-specific benefit. The four zero-initialized final-state heads have identical inputs, labels and optimizer options and therefore stay symmetric in exact deterministic arithmetic: they match nominal parameters, not four independent effective predictors. This control limit must remain visible in any interpretation.

## Optional staged representative protocol

This protocol is a prospective research recommendation, not queue admission. Exact source/custody, source qualification, pilot decision rules and the accounting manifest must be frozen by root before any execution.

1. **Placement pilot:** DEPTH_AUX and FINAL_AUX at the existing representative seeds 101/202/303: **six new fits**, with the paid F, native M1 and I4 retained as context. Use exactly the existing final-head selector and raw-logit mean. Report every seed, mean/sample SD, final served AUC, mean/worst member AUC, native BCE and graph-uniform BCE, Brier score, common member reversals, pooled reversals despite correct members, and paired whole-population repairs/harms. No auxiliary loss selects checkpoints or rescues a final-head failure. Auxiliary diagnostics may use TRAIN only and must be charged if they add work.
2. **Equipped native follow-up, if root selects it:** apply the same four depth heads, initialization and objective coefficient to native M1 and each fully independent I4 body. Each ordinary body uses `J_native = B(final_logit,y) + (1/4) Σ_l=1..4 B(a[l](meanpool(h[l])),y)` with its one native forward, not four artificial routes. M1 is exactly I4 body 0; four bodies × three seeds require **12 additional fits**, not 15. Preserve each body's native seed offsets, own optimizer, selector, selected state and ordinary final serving. There are no residual adapters or private-BN route bank in an ordinary native body. This tests whether the auxiliary treatment is simply helpful ordinary-GNN training. The unequipped native references alone cannot attribute an advantage to sharing.
3. **Claim limits:** stage 1 can nominate useful prefix supervision in F. A positive stage 2 comparison supports the tested shared package relative to equipped native alternatives, with its remaining architecture/capacity differences. A depth-specific *sharing interaction* would additionally require corresponding native FINAL_AUX controls and a predeclared interaction contrast. Those are not silently added to this roster. There is no universal transfer, primitive novelty or speed conclusion.

A credible competence signal requires improvement in **served final predictions and member competence**, with fewer common wrong pairs and useful repairs beyond both F and FINAL_AUX; lower auxiliary training loss alone fails. Preserve all contrary seeds, pooling harms and BCE/Brier deterioration. Requiring intermediate linear separability may fail outright; neither changed diversity nor lower shared error counts alone establishes success. Three seeds on the small reused VALID set provide screening evidence, not a confirmatory population estimate. Keep TEST sealed and do not tune depth/coefficient/initialization after seeing the pilot.

## Later complete structural context

The candidate direction and depth/control choice were developed before the complete structural analysis was supplied. Afterward, root released the complete local structural summary; a bounded scalar read is logged separately. Structured risk had 74.469% mean served AUC versus 74.518% for F (−0.049 pp), despite reducing mean common member reversals 664.7→519.3 and improving pool-minus-mean-member AUC −0.043→+0.455 pp. Mean member AUC fell 74.560%→74.014%; whole-population strict repairs 326.0 and harms 327.3 approximately offset. Its frozen primary therefore fails, even though it beats shuffled risk on average. These completed results make member competence worth protecting; they do not show that deep supervision will fix it or diagnose a feature-depth defect. No structural outcome changes the proposed heads, depth set, coefficient, serving rule or frozen PubMed queue.
