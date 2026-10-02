# Shared path accounting and scientific boundary

For a nonzero common gradient, the helper reuses one VJP primal and the unpermuted band bank across the remasked/full arms. It repeats each arm's band pullbacks and JVP geometry explicitly.

| Operation | Count |
|---|---:|
| Common VJP primal forward | 1 |
| Pullbacks | 13: one TRAIN CE and three sets of four bands |
| Graph sparse products | 6: original and permuted degree-3 banks |
| JVP primal forwards | 0–12; four per nonzero graph arm |
| Candidate forwards per shared trial | 16; four members in each of four arms |
| Candidate forwards for six rejected trials | 96 |
| Extra same-alpha diagnostic forwards | 0; common-arm member 0 is reused |
| Maximum total closure forwards | 109 |

A normal first-step joint acceptance with twelve JVPs costs 29 closure forwards. No interface qualification/warm acquisition/install/continuation/prediction is included. VJP/JVP/trial counters increment before their respective calls. Per-arm trial counts include every raised call. Geometry exceptions preserve counters through `PairedGeometryError.report`. Three scheduled sparse products are charged before calling each bank primitive; a bank abort can have performed fewer products, so its error receipt must retain that distinction rather than claim three completed products. Pullbacks count reverse calls, not closure forwards. Do not equate these counters with wall time, rematerialization, kernels, native activation memory or optimizer work.

Both dense four-band banks are held, along with q fields for three graph arms, full JVP fields while constructing each arm's diagnostics, and the current sixteen member outputs. Geometry retains four `[4,d]` directions/tangents and its q fields. Rejected trial logits are replaced at the next step; scalar/Gram/error receipts persist. The VJP tape is released before JVP/trial calls. Temporary float64 Gram/slope/finite-contrast arrays and model activations can dominate. No full Jacobian/NTK is formed by the helper, and no streaming/rematerialization implementation is claimed. A small private slice does not guarantee low full-graph memory.

The paired path removes branch-specific alpha selection and silent fallback from this separately named operation. All arms can satisfy the radius bound with different actual norms. The null arm uses the supplied permuted topology and does not move labels/features. Successful algebra, nonzero support delta, full output spread or positive signed construction do not establish predictive utility. The TRAIN separation guard still rejects pure unlabeled-root diversity, and failures remain in the comparison's denominator.

The primitive has the saved detached complementary-soft-target distillation representation. Mean first-order output movement remains common descent. Frozen linear squared-loss SGD with shared fixed preconditioning preserves the pooled common trajectory from zero-mean initial offsets. CE, nonlinear body evolution or nonlinear optimizer moments can change later behavior but require a measured, predeclared continuation comparison. This packet adds no scientific outcome, novelty or gain claim.

Native AD/source qualification and resource feasibility are separate from merit. Inability to fit an exact declared closure on current hardware leaves execution source/resource-deferred; it does not reject the quality hypothesis. The helper neither launches GPU work nor interprets current compute as a scientific verdict.
