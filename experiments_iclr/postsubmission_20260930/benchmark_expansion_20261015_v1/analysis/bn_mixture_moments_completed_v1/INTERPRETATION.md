# Mixture moments narrow the fixed-weight normalization effect

**Mixture moments improve AUC over arithmetic tying in all three seeds, while
own banks retain an AUC advantage in all three.** The mean own-minus-common
gap falls from 14.554 to 5.165 percentage points. Missing between-route
variance materially affected the original tied-bank contrast; it did not
account for the entire observed contrast.

All conditions use the same selected private-BN checkpoints, learned weights,
151 MolBACE VALID graphs and fixed mean-logit readout. The mixture common mean
matches arithmetic tying; its variance adds the between-route-mean term:
`v_mix = mean(v_m) + mean((mu_m - mean(mu_m))^2)`.

| Seed | Own AUC (%) | Mixture AUC (%) | Arithmetic tied AUC (%) | Own minus mixture (pp) |
| --- | ---: | ---: | ---: | ---: |
| 101 | 74.835 | 69.048 | 56.410 | 5.788 |
| 202 | 73.004 | 67.289 | 59.194 | 5.714 |
| 303 | 73.004 | 69.011 | 61.575 | 3.993 |
| Mean | **73.614** | **68.449** | **59.060** | **5.165** |

Mixture improves on arithmetic tying by 9.389 points on average. The prior
cyclic evidence remains: own beats the average of the three shift scores in
each seed, although shift1 exceeds own AUC for seed101. Own assignment therefore
is not demonstrated to be uniquely optimal.

## Proper losses qualify the ranking result

Mean BCE is 1.0855 for own, 1.4075 for mixture and 1.2504 for arithmetic tying.
Mixture improves BCE over own in seed101 (0.4246 versus 0.5605), but worsens it
in seeds202 and303. Mean Brier is 0.2983 for own, 0.3827 for mixture and 0.3977
for arithmetic tying; mixture again beats own only in seed101. Improving AUC
over arithmetic tying in every seed does not imply uniformly better
probabilities.

The largest relative correction is at the first MLP BN: between-route means
contribute 42.1–48.9% of its mixture variance summed across channels. All BN
layers changed together, so this does not identify that layer as the cause of
the AUC improvement.

## Interpretation and limits

At these fixed selected weights, route-specific normalization retains a
smaller but consistent AUC advantage over a coherent common-mixture
approximation. The arithmetic control remains a valid saved-state
intervention; the new control supplies a stronger common-moment comparison.
Neither reconstructs G0's sequential shared EMA. Saved EMA moments omit
temporal drift and exact sample-size information, and changing earlier banks
changes later-layer inputs. The mixture is a plug-in approximation, not
recalibrated population normalization.

The remaining 5.165-point gap does not estimate private BN's full training
benefit: training history, coadaptation and checkpoint selection remain
unseparated. Three development-selected models on one small scaffold VALID
split establish no independent TEST gain, significance or novelty. This result
selects no new serving rule and changes neither the nine-fit plan nor the
campaign roster.

## Completion and evidence

All three seeds completed: six full VALID calls, 120 CPU native forwards,
25.764 seconds supervisor time, zero TRAIN/optimizer/TEST calls. Own AUCs and
proper losses exactly match the prior diagnostic; maximum saved-export logit
difference is below 0.000002. Learned parameters were unchanged and banks
restored. Prior tied/cyclic conditions were reused without new forwards.

`EVIDENCE_MANIFEST.json` binds this interpretation to the completed scalar
summary, condition table, launch/result receipts and unchanged frozen sources.
The summary is byte-identical to its remote receipt. The local CSV has LF
instead of CRLF line endings; reversing only that conversion reproduces the
remote hash, and all 21 numeric condition rows match the result receipt.
This interpretation step performed no new execution or scoring.
