# Complete shared/private objective comparison

The declared extension did not meet its predefined practical improvement criteria on either graph. It is closed without tuning or heldout promotion. This result concerns the fixed HGT setting; it does not establish that every possible block objective is ineffective.

All 40 cases and the 15 required native secondary rows are retained. The evaluator exited 0 in 208.063467 seconds. Selected checkpoints were restored, and the paired vectors below were recomputed from the served raw scores and checked against the complete evaluator. TEST remains closed.

The candidate sends pooled-prediction gradients to shared parameters and member-specific gradients to private factors. Each graph/control comparison required an average NLL reduction of at least 0.005 nats, at least four strict wins out of five paired blocks, and no reduction in mean macro-F1. These are practical continuation criteria, not significance or conference-acceptance thresholds.

| Graph | Control | Mean NLL delta | NLL wins | Mean macro-F1 delta | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| HGB-ACM | own/own | -0.0017315 | 3/5 | -0.0010163 | fail |
| HGB-ACM | pool/pool | -0.0000358 | 2/5 | -0.0011093 | fail |
| HGB-DBLP | own/own | +0.0014742 | 2/5 | +0.0008288 | fail |
| HGB-DBLP | pool/pool | +0.0000241 | 1/5 | +0.0000000 | fail |
| HGB-ACM | own/pool | -0.0016679 | 3/5 | -0.0010163 | fail |
| HGB-DBLP | own/pool | +0.0014351 | 2/5 | +0.0017751 | fail |

Deltas are candidate minus control; smaller NLL is better and larger macro-F1 is better. The reverse role control is own/pool. All vectors, descriptive intervals and leave-one-block-out means remain in the original complete evaluation. Overlapping graph splits do not support independent population-level inference. Calibration uses the same source validation labels and cannot rescue the failed raw primary gate. No selected subset, final checkpoint, member-level metric or changed threshold replaces the declared endpoint.

This study supplies no new predictive winner, established novelty or manuscript acceptance verdict. Original paper scores are unchanged. Earlier failed evaluation attempts and costs remain preserved.
