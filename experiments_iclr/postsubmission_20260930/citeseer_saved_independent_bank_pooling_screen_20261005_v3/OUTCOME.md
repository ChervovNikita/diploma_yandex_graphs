# Fixed saved-bank pooling screen: outcome

## What was tested

The complete three-block Citeseer-HeaRT development bank contains four independently trained members per block. Each selected member predicts all 227 validation positives and each positive’s unchanged 500 hard negatives. The screen compared the original mean raw score with equal Borda ranks of those same candidates. Borda ignores member score scale and fits no parameters. No backbone inference or training was needed.

All 36 cohort terminals and all 12 saved member score identities were authenticated before score loading. The original separate positive/negative reduction recipe replayed all 681 native per-query ranks and all 3 MRR values exactly. Original paper scores, checkpoints, selectors, candidate populations and promotion contracts remain unchanged.

| Seed block | Original mean-score MRR (%) | Borda MRR (%) | Difference (percentage points) |
| --- | ---: | ---: | ---: |
| 0 | 27.9145 | 27.9897 | +0.0751 |
| 1 | 28.8375 | 29.4389 | +0.6013 |
| 2 | 27.0093 | 26.8478 | −0.1614 |
| Mean | 27.9204 | 28.0921 | +0.1717 |

The descriptive 95% interval across training seeds for the difference is [−0.7982, +1.1416] percentage points; the exact two-sided sign-flip p is 0.75. Three seeds on one checkpoint-selected validation split do not support significance, graph generality or independent confirmation. The full unrounded values remain in RESULTS.json.

## What this means

A strict majority of members correctly ordered 859 positive/negative pairs that the raw mean reversed. Borda corrected 633 of these, but it also introduced harms: across the complete bank it reversed 2016 previously correct strict pair orders and repaired 2010 strict wrong ones. These pairs are dependent, and the ranking metric weights different query positions differently. The counts demonstrate magnitude domination in the existing scores; they do not establish miscalibration, a causal explanation of the small metric gap or a superior learned combiner.

A label-informed candidate-specific convex score interval calculation is recorded solely as an oracle upper bound. It chooses the most favorable member separately for each labelled positive and negative. It supplies no achievable generalization result or evidence for a learned router.

The old shared-four runner retained pooled scores without separate members. This screen therefore evaluates the independent bank only; it supplies no new shared-bank quality or a sharing-specific benefit. Shared extraction remains a separate cost/availability question.

## Decision

Retain Borda as a cheap aggregation control. The small, mixed, uncertain development difference does not justify presenting it as our methodological extension or opening TEST. Continue assessing graph-conditioned error weighting and hidden-state fusion against competent score stackers and the closest published operators. This does not alter any earlier failed candidate decision.

The owned CPU child exited0 in 3.81 seconds, including startup; analyzer time was 3.37 seconds. Only compact results and receipts returned to the Mac. Scientific tensors remain in the allocation repository. There were no model forwards, fits, optimizer updates, checkpoints/data exports or TEST accesses. The sealed original source findings and v1 and v2 repairs are preserved alongside accepted v3.
