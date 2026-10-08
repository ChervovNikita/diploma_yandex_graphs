# Direct12 source: scientific semantics review

8 October 2026. Reviewed source manifest `ec74b597d7ced14941098e20eb6e2e6b66bc2fc9604f902f5af3051baf210faa` once. Static source only; no scientific score/array access, model imports, execution or server work.

## Sole material release blocker

`cohorts.masks` derives member correctness/coverage from `probability.argmax`. `read_family.metrics` does so for member accuracies and global any-correct coverage. Original Wiki24 member prediction authority is **logits.argmax** (`run.py` member metrics and the original collector's `member_prediction`). Softmax rounding can turn a strict logit order into a probability tie, so these source-v1 definitions can change native all-wrong cohorts or count a fitted member as correct under a different rule.

Use logits.argmax for native/fitted member correctness, mean/worst member accuracy and global any-correct coverage. Keep pooled serving correctness on mean-probability argmax. This repairs the decision rule directly; no probability-parity rejection, tolerance search or score rerun is required. Root accepted the finding and requested a minimal sealed successor preserving v1.

## Other requested contracts: no further material blocker found

- Exact original checkpoints and per-member local/global modes are pinned; the existing restore helper enforces snapshot job/config and mode identity. Collection hooks capture the actual active final-head input before r/s, verify same-call ID/representation/logit correspondence and the factor member index, and keep upstream evaluation state frozen.
- The original FactorLinear adds bias after output scaling. Shared TRAIN RMS without centering, scaled W followed by r/s materialization, copied full A0 and common bias preserve the matched BE starting family. Both arms evaluate with the same FP64 affine kernel. The paired-start check is distinct from original native FP32 replay; no original microparity gate is present.
- The objective is mean-member/mean-TRAIN softmax CE plus the fixed class-centered effective-map L2. Ordinary private bias ownership and single bias remain intact. The optimizer receives only TRAIN features/labels; development labels do not enter fitting or stopping.
- All twelve endpoints/failures are retained and sealed before separate comparative opening. Finite nonconvergence is reported and blocks affirmative complete screens. Frozen primary cohorts are from untouched native unit+contrast. Actual pooled and coverage transitions, all-old-rival order diagnostics, same-member correctness and introduced common-error diagnostics are separated.
- The hardcoded screens match the bound prospective decision and node-count clarification, subject to the argmax correction above. Original historical selected accuracy and stored closed native pooled NLL are retained as authority; new native replay is labeled separately. No selected score is replaced by a reconstruction.

No additional variant, helper scaffold or solver change is requested. Root's one short standard-backend artificial qualification remains pending and is separate from this scientific source review. Review of a successor should be confined to the blocker-closing diff and its bindings.
