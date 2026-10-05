# Disabled TRAIN-only preparation

Source preparation only. No prepared code was imported or run, no data/label/result/checkpoint/logit payload was opened, and no fit or SSH was performed. V1 protocol and every predecessor remain unchanged.

## Boundary and accessor

`train_only_accessor.py` supplies disabled `prepare_train_roles(project_root, destination)` and `load_public_b(project_root, public_b_dir, *, device="cpu")`.

The existing public NPZ contains features, edges and separate mask members, with no labels. Custody preparation opens only its TRAIN mask, derives the exact seed17 split0 A/B/S/R membership by the two prescribed public-ID hash orders, and then decodes the compact TRAIN0 labels member. That single member necessarily includes A as well as B. The historical producer also declares that it decoded the raw all-node labels. These are disclosed explicitly in `PREFIT_AMENDMENT.json`; this is a holdout from new fitting and selection, not a claim of physically unread or historically untouched A labels.

The custodian writes separate `public_b/B_LABELS.npz` and `evaluator_a/A_LABELS.npz`, no scores or label-dependent role choices. It fails for unsupported fixed S without reseeding or rebalancing. The fitting accessor reads B-only labels and the pinned public features/edges; it reads no mask member, original compact TRAIN label array, or A artifact. Neither path decodes VAL/TEST labels or VAL/TEST mask members.

The accessor returns FP32 `features[24492,300]`, long `edge_index[2,E]`, long `B_ids/B_labels[9797]`, `inner_indices/inner_labels[4898]`, `query_indices/query_labels[4899]`, `A_ids[2449]` (IDs only), and JSON provenance. These cardinalities remain source expectations: no role derivation or payload observation occurred here. Membership follows V1 exactly; stored tensor row order is increasing node ID.

## Concrete fixed worker and evaluation

`six_arm_worker.py` prepares the fresh native constructor/device reset/boundary wrap at seed17, mean own CE on every B label, Adam defaults at lr.001/wd0, exactly200 local then200 global updates, no selector or rollback. It saves update400 and reuses that exact common bank for the six serial H16 arms in `QUEUE.json`. G0 coefficients, probe/private/core steps, complete ownership, all controls and the one class-preserving permutation remain fixed. Continuation uses the native port's public gated APIs; it neither flips guards nor calls private learner helpers. Dormant global-stage local-head tensors must remain unchanged.

The worker accepts no A label artifact and produces no A metric. It freezes common plus six complete endpoint states and B-only episode diagnostics before writing `COMPLETE.json`; failure leaves an incomplete record. `held_a_evaluator.py`, in a separate process, verifies all seven immutable state hashes and exact endpoint metadata, validates complete native state tensors, and completes all seven served predictions before opening A. It saves native FP32 logits, native-device FP32 softmax/mean probabilities, full-A and complete member/class metrics, and every fixed comparison. Accuracy uses first-index argmax ties; NLL uses stable FP64 log-softmax/log-sum-exp from saved FP32 logits without clipping; Brier accumulates saved served FP32 probabilities in FP64. Zeros, ties and raw unanimity remain separate counts; unsupported class cells retain n=0 and null summaries.

## Support risk before fitting

All-B warming may saturate S own CE, making response costs epsilon-dominated. H16 at eta_core=.001 can also make live/stop-Q differences tiny. A failure at this finite recipe cannot reject the broader hypothesis. S∪R=B, so warming a disjoint B subset while leaving both fixed roles label-unseen is impossible without a new partition; warming only R would change V1 and expose the outer set. No alternative was silently selected.

`warm_response_probe.py` is a separately disabled read-only helper for the authenticated unchanged common theta and original phi. The root-requested exact V2 private-response call computes and discards probe/main virtual states using S labels only, commits nothing, and returns response-scale/Q diagnostics. It adds16 complete member forwards,8 private derivative constructions,4+4 virtual private steps and80 pair solver iterations. A numerical nonvacuity/admission convention is still a root pre-fit decision. Episode diagnostics are measured at recomputed theta+, and are not mislabeled unchanged-warm diagnostics.

The scientific plan remains5084 complete member forwards including later evaluation; this optional diagnostic raises that accounting to5100, before other engineering and I/O. No native wall time or peak memory was measured here.

## Remaining work

Every new entry point is guarded false before numerical imports or I/O. The exact bound accessor and native port are also disabled, so merely changing the worker flag cannot reach acquisition. Future reviewed successor bindings, native higher-order/sparse qualification, whole-process resources/runtime and the prospective support convention remain unresolved. Static parsing/hashing does not qualify numerical correctness or predictive usefulness. Later capable ordinary whole-bank, nonlinear single, independent4 and same-operation untied4 comparisons remain necessary.
