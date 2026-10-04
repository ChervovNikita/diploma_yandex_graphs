# Actual native TRAIN gradient diagnostic

The exact corrected source completed one full native TRAIN batch (65,536 positives and65,536 negatives) on18.77 GPU1. One retained forward graph supplied target, joint J_K and separate-side J_K_sep gradients. The original supervisor exited, its child was reaped with exit0, collection/custody matched, and the physical session closed. No optimizer or VALID/TEST evaluation ran.

## Signal

Joint loss0.047556948; separate-side loss0.047517922; target loss1.377547860.

Across disjoint model blocks, joint auxiliary gradient norm is2.785% of target-gradient norm. Joint-minus-separated norm is0.0561% of target norm and2.014% of joint-auxiliary norm. It reaches both encoder and member factors. These are descriptive values at one initialization, not predictive improvements.

Positive queries:3,800 have both sides variable and886 have genuine subset choices on both sides. Negative queries have no both-variable support. Forced-side gradients are exactly zero, and forced-counterpart joint/separate gradients exactly agree. Every mean-reduced per-slot delta is within the unchanged tolerance; that absolute tolerance exceeds these small derivatives. Parameter-block deltas are independently visible in the recorded scalar norms. No rule was loosened.

## Cost and decision

Child work took87.30s; forward10.69s; joint reverse30.13s; separate reverse31.14s. Peak allocator:allocated28.63GB,reserved42.42GB. The implementation dispatched1,505 genuine ESP groups and132,447 slot loops. These are this diagnostic's costs, not a full-fit estimate.

The mechanism is connected and produces a small parameter-gradient distinction. Predictive usefulness remains untested. Investigate the separately disabled exact-law batching optimization before paying for a broad study. Preserve normalization, coefficient, complete batch, precision and teacher semantics. Representative paired target-only/joint/separate fits and an independent competitive benchmark are still needed. No novelty, superiority or acceptance follows from this diagnostic.

The prior startup failure and all source/review versions remain preserved. Original paper and current predictive scores are unchanged.
