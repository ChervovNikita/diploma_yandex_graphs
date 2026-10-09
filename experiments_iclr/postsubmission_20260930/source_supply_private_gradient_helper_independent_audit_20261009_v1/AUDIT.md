# Independent source audit: source-supply helper

## Verdict

**Focused source review accepted after one finite-safeguard repair. Native/runtime qualification remains pending.** Runtime stays disabled. This audit authorizes no scientific launch and supplies no prediction-quality or novelty evidence.

Reviewed `source_supply.py` SHA256: `b8bcbbede340b7850a09bbe01cd1c62d013d569abda34116c34cefdac22382e9`. The helper was unsealed during review. Source, integration/score/prior documents and source-bound synthetic fixture records were reviewed; no scientific data/outcomes/checkpoints, Torch models or server actions were used.

## Minimal correctness repair

**P2, fixed before seal:** squared gradient norms could overflow/underflow and silently remove a nonzero cone constraint. For target `[1,-1]` and positive risk row `[a,0]`, the projection is `[0,-1]` at every positive scale. The original squared norm turned `a=1e200` into an infinite norm and a zero normalized row.

The author replaced risk/own/projected norms with stable `hypot`, added finite checks for dot/solve/projection/dose/native-representable direction intermediates, and requires a finite negative descent slope before trial copying (`source_supply.py:358–505,551–563`). One analytical scale-invariance fixture covers overflow/underflow scales; the audit added no float sweep. The probe-gradient wording was also clarified to avoid a double negative.

## Exact score and credit assessment

For categorical target y, let `B=(p_m(y)+sum_peer p_k(y))/M` and `rho=p_m(y)/(M*B)`. The implemented factual-minus-absent objective has

`dJ/dz_full = rho_supply*(p_full−onehot(y))`,

`dJ/dz_probe = −rho_absent*(p_probe−onehot(y))`.

Categorical log-probability mode preserves the native log-probability Jacobian instead of applying another softmax. Runtime differentiates the actual mixture score; no responsibility coefficient is detached into a surrogate scalar objective.

For hard observed Bernoulli outcomes, each label has `q=mean_m sigmoid(z_m)`. Positive responsibility is `p/(M*q)`; negative responsibility is `(1−p)/(M*(1−q))`. Each entry's CE/BCE gradient is `rho*(p−y)`, with the probe term subtracted, followed by the query/label and active-recipient means. The implementation mixes entrywise observed-outcome probabilities over members, then averages labels. It forms no joint-label product and performs no across-label normalization.

The audit's exact Fraction fixtures verify the direct mixture chain rule against the responsibility form for categorical and positive/negative Bernoulli entries. They also verify the guard implication `Delta supply = Delta J + Delta absent <=0` and a denominator-gaming counterexample. These fixtures do not execute the Torch/native replay.

## Ownership, removal and safeguards

- Parameter coverage, actual common-object identity and private/nonsteered object/storage separation are checked. Source replay uses `autograd.grad` only for declared private parameters and preserves `.grad` fields; private trials receive no optimizer object or moments.
- For recipient m assigned a, every frozen peer uses `train_probe(k,a)`, regardless of peer assignment. Native view declarations are checked for consistency; they do not prove actual family/reverse/path removal.
- Finite guards protect aggregate TRAIN factual own/pool, assigned-probe own and frozen-peer absent risks, together with aggregate Armijo/per-recipient J. With zero tolerance they imply aggregate supplied-risk nonincrease. They do not protect individual rows, other-source views, the all-updated absent committee or heldout quality. A nonzero tolerance weakens exact nonincrease.
- The actual serving helper calls every factual full-input member; no virtual supplied-source pool is served. All extra reference/replay/VJP/trial work and zero/rejected/accepted dose require reporting.

## Required unqualified native conditions

**Parameter/version stamps cannot certify native prediction-state restoration.** Shared and member-owned buffers, including SeHGNN BatchNorm running statistics, must be frozen or exactly snapshotted/restored before and after every reference, replay, trial and evaluation guard. Source calls must not become hidden normalization training. Native graph caches/normalizers and exact per-view dropout-token/RNG/mode restoration require real integration qualification.

Actual TRAIN/query/output custody, complete hard outcomes/native loss reduction, source removal/support rebuilding, Torch VJP correctness, atomic private rollback, dtype/memory feasibility and competent native/shared/untied references also remain pending. If an eligible factor acts only on a zeroed removed channel, its probe Jacobian can vanish and its update reduces to positive weighted factual CE/BCE. Two outputs do not prove two nonzero private gradient terms.

**Analytical fixtures are not evidence of prediction quality, generalization, useful complementarity, causal source/geometry specialization or methodological novelty.** The documentation states these limits and retains the established proper-score/responsibility prior relationship.

See `AUDIT.json` for input hashes, finding disposition and exact activity limits; `EXACT_ALGEBRA_VERIFICATION.json` and `verify_exact_algebra.py` contain the independent rational fixtures.
