# Complete fixed J/F development pair

The experiment tests whether learning a whole observed connection pattern helps link ranking beyond learning each connection separately. Architecture, training labels, native masks and main loss are matched.

One seed and one validation-selected time split provide a screening result. These results are not heldout confirmation or a comparison proving superiority over independent ensembles.

| Arm | Selected epoch | Validation Hits@50 (%) |
| --- | ---: | ---: |
| J | 97 | 64.3066 |
| F | 8 | 64.0903 |

Joint minus individual-incidence supervision: **+0.2164 percentage points**.
Positive single-pair development screen; prospective replication and heldout confirmation still required

## Fixed routing diagnostics

Crossed routes keep recipient features and decoders fixed while giving them another member’s completion weights. Pooled weights average the clamped completion bank before each recipient decoder. Every route uses mean raw target logits. These are fixed-bank diagnostics, not newly trained methods or checkpoint selectors.

| Route | J Hits@50 (%) | F Hits@50 (%) |
| --- | ---: | ---: | ---: |
| own | 64.3066 | 64.0903 |
| crossed_cyclic_1 | 64.2467 | 64.3399 |
| crossed_cyclic_2 | 63.6276 | 64.4132 |
| crossed_cyclic_3 | 63.5693 | 64.3948 |
| pooled_clamped_weights | 65.7596 | 64.6212 |

## Pattern and member evidence

ANALYSIS.json retains every fixed stratum, marginal observation NLL/Brier statistic, normalized joint/factorial pattern loss, component spread and responsibility statistic, and every member score/error-overlap diagnostic. Denominators are explicit. No mask events or queries are treated as independent graphs.

The labels describe connections observed in TRAIN. Zero means unobserved in TRAIN. New mask events reuse that source teacher and do not establish latent-link posterior accuracy. Target BCE is measured on the fixed balanced validation query pool and does not certify population calibration.

Better joint reconstruction alone is insufficient. A dependence interpretation also needs useful target quality, matched marginal competence, capable structural-single controls and prospective replication.

## Work and interpretation limits

Both scientific fits completed 100 epochs and 1,700 optimizer updates, with 100 validation checkpoint candidates each. Together, the fitted selections, replays and diagnostics account for 206 full validation traversals. The machine-readable report retains bound stage accounting, closure accounting and normal-supervision envelopes. Nested driver/supervisor durations overlap and are not summed. Unlisted interruptions and terminal-write tails remain outside certified costs.

There is no confidence interval from this one seed, no heldout evaluation, no new state-of-the-art claim and no manuscript acceptance claim.
