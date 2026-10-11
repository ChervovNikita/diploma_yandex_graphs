# Selected-checkpoint TRAIN gradient result

The complete15-bank/24-state/51-backward diagnostic closed with no model updates. Every route used all580TRAIN nodes in the full11701-node WikiCS graph and one original next-dropout realization. Actual owner exit/reap and CUDA-child absence are recorded. No VALID labels, new validation scores, TEST or current Physics/Penn outcomes were accessed. Old closure/checkpoint deserialization incidentally carries historical score metadata; only identity/receipt/step fields were used.

| Arm | Seed | Shared mean/route energy ratio | Negative whole-shared pairs (of6) | Mean TRAIN CE |
| --- | --- | ---: | ---: | ---: |
| BASE | 7301 | 0.270802 | 3 | 0.005279 |
| E4 | 7301 | 0.301913 | 2 | 0.172826 |
| N4 | 7301 | 0.287516 | 1 | 0.224723 |
| BASE | 7403 | 0.258091 | 1 | 0.005998 |
| E4 | 7403 | 0.197579 | 5 | 0.214590 |
| N4 | 7403 | 0.252188 | 3 | 0.206735 |
| BASE | 7507 | 0.260580 | 3 | 0.022154 |
| E4 | 7507 | 0.261446 | 1 | 0.144264 |
| N4 | 7507 | 0.232216 | 2 | 0.196435 |

## Interpretation

Whole-shared ratios are near0.25 for most selected states. With four exactly orthogonal route gradients, this ratio equals0.25 regardless of their relative nonzero magnitudes. Consequently, a ratio near0.25 is not evidence that a useful gradient has vanished. Negative pair dots occur in BASE, E and N, with different patterns across seeds. E has2/5/1 negative whole-shared pairs; its conditional-builder ratios are0.268448/0.251255/0.230605. There is no uniform overwhelming opposing-direction pattern that explains E member weakness.

This does not rule out harmful interference at other states or in actual optimizer dynamics. It measures local Euclidean gradients, not the full training history, dropout uncertainty or AdamW updates. The single7403 E whole-shared ratio0.197579 cannot supply a general causal claim. Overlapping aggregate groups are summaries of the same coordinates, not additional independent evidence.

E own-loss TRAIN CE is much higher than BASE under these selected-state dropout draws. The already completed validation study nevertheless favors E over BASE. This is consistent with different fitting/regularization/selection behavior, but it does not isolate overfitting or prove why E improves. Different selected checkpoints, parameterizations, conditional capacity and route dynamics remain entangled. M1/I4 own losses and gradient magnitudes are size-only references; no independently learned coordinate cosines are reported.

No gradient-surgery fit or reviewer is admitted from this result. Unused full-pipeline confirmation and the already queued FiLM comparison remain the quality priorities. The separately prepared attributed stochastic-factor acquisition question remains source-only.

## Complete retention and costs

All144shared group summaries and864pair records are retained, alongside816route-group norm rows and612private-row norm rows. Undefined ratios:15; undefined pair cosines:90; present allzero groups:0; routes with unused parameters:0. Every undefined case remains in SUMMARY.json and the full output. Disjoint private rows are never called useful gradient diversity.

Child measurement wall:71.40s; owner wall:72.30s; peak allocated CUDA:4.330GiB; lifetime peak RSS:0.855GiB. Execution overlapped Physics/Penn acquisition, so costs are not isolated speed evidence.

BANKS.csv, GROUPS.csv, PAIRS.csv, ROUTE_GROUP_NORMS.csv and PRIVATE_ROW_NORMS.csv provide all scalar rows. The complete bound JSON additionally retains checkpoint receipts, parameter inventories, every RNG before/after hash and unused-name identities.
