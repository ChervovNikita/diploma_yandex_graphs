# Amazon native recipe study: complete six-fit audit

All six selected checkpoints reproduced saved validation logits bitwise and selected fit/control/validation metrics exactly on the original device/runtime. The accuracy selector and patience trace were independently checked. Both recipes and all three split/seed blocks remain. TEST was not scored.

| Split / seed | Recipe | Selected epoch | Fit accuracy | Control accuracy | Validation accuracy | Validation NLL |
|---|---|---:|---:|---:|---:|---:|
| 0 / 17 | source_defaults | 361 | 72.81% | 43.17% | 42.48% | 2.822 |
| 0 / 17 | roman_mono | 535 | 78.55% | 43.08% | 43.25% | 2.437 |
| 1 / 29 | source_defaults | 177 | 61.36% | 43.02% | 43.21% | 1.802 |
| 1 / 29 | roman_mono | 391 | 80.97% | 44.65% | 44.13% | 3.932 |
| 2 / 43 | source_defaults | 82 | 44.58% | 40.35% | 42.41% | 1.329 |
| 2 / 43 | roman_mono | 296 | 83.69% | 44.88% | 44.88% | 3.080 |

Roman transfer improves mean validation accuracy by 1.383 percentage points, but its mean validation NLL is 1.165 nats higher. Its fit accuracy substantially exceeds control accuracy. The higher-accuracy recipe is therefore not evidence of calibrated quality or GNNM superiority. The paired values and descriptive intervals are retained in EVALUATION_ADOPTION_v1.json.

The three official split/seed blocks overlap on one graph. Validation selects checkpoints and the recipe; these are development results. The 80% official-TRAIN fit subset differs from published full-TRAIN protocols, so published test means supply context rather than an interchangeable threshold.

Next priority is the previously specified source-authored Amazon Polynormer-r comparison, including its independent ensemble. No further recipe grid or intervention is authorized by this audit. Original paper scores remain unchanged.
