# Independent V2 amendment review

Bounded review of the supplied immutable V2 protocol and preserved V1 specification/review. No data/logits, fitting, scientific imports, SSH, agents or manuscript review occurred.

## Amendment disposition

| V1 finding | V2 disposition |
|---|---|
| F1 graph mask | Addressed: native FP32 classes are operative; single uses its own native mask; orientation, self-loop degree, duplicate exclusion, sorted FP64 CSR and hashes are fixed. |
| F2 strong single | Addressed: fixed member0 receives native+C&S and two capable 17-input/156-width residual-head settings; 3,678 parameters and own graph are specified, with the same label/fold opportunities. |
| F3 fitted-control competence | Addressed: exactly 150 updates; best permitted B-fit penalized objective among steps 0–150, initialization included, strict improvement/earliest tie; no D iterate selection. Initial/final/selected objectives and gradients are recorded, and unqualified-control losses cannot establish analytic superiority. |
| F4 processed comparisons | Addressed: shared versus best processed independent and single is operational, with inherited practical thresholds, complete-pipeline interpretation, matched-rho contrasts and mean three-split match tolerance. One stale descriptive JSON reference remains below. |
| F5 numerics | Addressed at specification level: stable log-skip, exact NLL contamination, ddof0/Xavier/Adam/thread/environment choices, scaled active-face solver and reported checks. Numerical qualification remains separate. |
| F6 costs | Addressed: single pilot costs, propagation batching, failed/successful historical evaluator costs and separate future refits are explicit. No refit is admitted in this development run. |

Scope, blocks, native reference, folds/exclusions, moments/rho, nine bank operators, C&S, metrics, confirmation boundary and original tensor-byte estimates are unchanged from V1. No new label access or independent-confirmation claim was introduced. Best B-objective iterate selection is training selection; outer OOF setting/operator selection remains explicitly biased development evidence.

## Minimum-norm active-face procedure

The mathematical procedure is valid for exact PSD systems. There are 15 nonempty free faces of the four-variable density simplex. Inactive weights equal the fixed floor; each face has one equality constraint. Enumerating all faces includes a global optimizer and the face of the feasible minimum-weight-norm optimizer.

For a singular face, suppose two consistent KKT solutions differ by (d,delta-lambda). Their equations imply 1' d=0 and A_FF d+delta-lambda*1=0. Multiplying by d gives d' A_FF d=0; PSD then implies A_FF d=0 and delta-lambda=0. Thus the face multiplier is unique, and minimizing augmented KKT-vector norm also minimizes free-weight norm. Fixed inactive coordinates add a constant to the full weight norm. The pseudoinverse is therefore suitable for the specified numerical minimum-norm tie rule. The projection right-hand side is consistent with the PSD quadratic construction.

The rcond and objective/weight-norm tie tolerances make this a numerical approximation, as V2 states. They do not prove exact equality in near-singular floating-point cases. Qualification must report residuals and the resulting probability projection; no new statistical axis or search is warranted.

For implementation audit, “KKT checks” must mean free-coordinate stationarity **and inactive dual inequalities**, with complementary slackness. An inactive coordinate need not have zero gradient plus equality multiplier. For gradient g=2(Aw-b), the conditions are g_i+lambda=0 on free coordinates and g_i+lambda>=0 at the floor, after using a consistent multiplier convention. Roundoff clipping/renormalization must be followed by these checks. This states the existing constraint's KKT condition; it adds no gate or scientific constant. Objective ties should be measured against the global lowest feasible objective, then weight norm/face order applied.

## One remaining JSON consistency gap

`selection.strongest_cheap` includes the best processed member0 single, and REPORT F4 explicitly retains that addition in the per-bank cheap gate. However, `go_no_go.bank_route_reference` still says “best tuned corrected cheap comparator IDs 1..5 and uncorrected native probability pool,” omitting the processed single.

**Correction:** make that reference name `selection.strongest_cheap` plus the uncorrected native pool. The new shared-versus-independent/single gate already explicitly requires both processed references, so this is a stale per-bank gate description rather than a missing shared-superiority gate. Resolve it before execution admission to prevent an implementation from choosing conflicting strings. No additional comparison, fit or threshold is needed.

## Count verification

| Quantity | Derivation | Count |
|---|---|---:|
| MLP fits | 72 bank + 3 splits × 3 folds × 2 single settings | 90 |
| Calibration fits | 6 bank/splits × 3 folds × 2 settings | 36 |
| Total fitted pilot runs | 90 + 36 | 126 |
| Outer configuration predictions | 6×3×15 + 3×3×3 | 297 |
| Logical C&S H applications | 2×297 | 594 |
| Bank seed-context H applications | 6×3×3 | 54 |
| Single mass-context H applications | 3×3×3 | 27 |
| Label-free H applications | 6 bank/splits + 3 single/splits | 9 |
| Total logical H applications | 594+54+27+9 | 684 |
| Logical sparse steps | 684×20 | 13,680 |
| Batched H calls | 9 graph/splits × [1 label-free + 3 folds × (3 seed contexts + 2 batched C&S)] | 144 |
| Batched sparse calls | 144×20 | 2,880 |

The 144 count assumes all configurations' correction fields and then smoothing fields are batched within a graph/outer fold and contexts are reused. Differing field widths, solves, head work, objective/gradient diagnostics, cache I/O and actual memory/time remain paid work. Batching reduces call count; it does not remove logical output work or certify wall time/RSS. The single parameter count is 5×17+5×156+156×17+156+5=3,678. Later selected refits remain separately released and charged, with at most six learned bank fits and three single-head fits.

## Conclusion and scope

All six original findings are operationally addressed. The stale per-bank gate reference needs a narrow correction; the explicit KKT conditions above are the implementation audit interpretation of the existing procedure. No further scientific axes, label conditions, fitting budget or statistical gates are requested. Payload availability, code implementation, numerical qualification, resource sufficiency and execution authority remain unverified by this protocol review. V1 and its independent review were preserved.

## Additive correction observed after base review

The parent preserved the base V2 PROTOCOL/REPORT and added `REFERENCE_CORRECTION.json`, SHA256 `d5ea58d79f5183c36fc199bbcb895c731091b1c9f2b75385dfb1a551bf1e61c9`. It binds the reviewed base PROTOCOL hash and replaces only `go_no_go.bank_route_reference` with “both selection.strongest_cheap (including best processed member0 single) and uncorrected native probability pool.” This resolves the descriptive inconsistency identified above without a new gate, setting or outcome-dependent choice. Source admission must bind the base plus this correction, as the parent specified. With the correction applied, **no protocol amendment blocker remains in this bounded V2 review**; implementation and numerical qualification remain separate.
