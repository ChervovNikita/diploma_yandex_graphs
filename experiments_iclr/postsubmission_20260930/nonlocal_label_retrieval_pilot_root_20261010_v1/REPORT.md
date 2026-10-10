# Complete live TRAIN-label retrieval comparison

10 October 2026. All three owners closed successfully before the CPU reader
opened outcomes. The comparison covers **81 selected banks**: 36 new banks from
63 native fits, plus 45 reused reference banks from 99 acquisitions. All 30 local
contrast/detail partitions match the admitted hashes; all 27 arm-detail packets
were read. No new training, model forward, source tuning, raw prediction/data
access, TEST or remote operation contributed to this report. Original paper
scores, reference archives and unsuccessful studies remain unchanged.

## Decision

**Close this exact broadly improving retrieval recipe.** Every backbone fails
its frozen accuracy conjunction and its NLL-protection conjunction. The fixed
live similarity-learning increment is unsupported across backbones: live shared
minus identical detached retrieval is −0.107445 / −0.214891 / +0.063203 accuracy
points for SAGE / GCN / GAT, with insufficient seed consistency or magnitude.
This is a predictive failure of the full fixed recipe, not an implementation
failure or a theorem against label retrieval.

Preserve the narrower positives. SAGE retains its original shared-bank accuracy
advantage over ordinary/factorized I4; GAT improves its original shared predictor
by **0.265453 pp at all three seeds**, with lower pooled NLL. Neither establishes
portable shared superiority. Member weakness alone is not a rejection rule: a
useful pool can contain weaker members. The actual pooled comparisons fail here.

## Fixed operation and complete served quality

The same factual native forward supplies logits and preclassifier H. A common
class-stratified half-TRAIN Q is excluded from every support S. Cosine attention
at temperature 0.1 retrieves TRAIN one-hot values. Native CE on all TRAIN plus
direct retrieval CE on Q both have coefficient 1; gradients reach query/support
H. The detached arm removes those similarity gradients while retaining the same
forward retrieval and serving. All 580 TRAIN anchors serve VALID; each route
mixes native/retrieval probabilities 0.5/0.5, followed by a probability mean.
There are no teachers, label inputs to H, label feedback or propagated-label cache.

Means over seeds 7301 / 7403 / 7507 on all 5,274 VALID nodes. Each cell is served
accuracy (%) / NLL. Raw references retain their original selected predictions.

| Predictor | SAGE | GCN | GAT |
|---|---:|---:|---:|
| Ordinary M1 | 79.319934 / 0.754213 | 80.697763 / 0.714089 | 81.234989 / 0.701435 |
| Ordinary genuine I4 | 80.021489 / 0.695762 | 81.032739 / 0.660404 | 81.980786 / 0.642380 |
| Factorized M1 | 79.130325 / 0.778711 | 80.457591 / 0.785566 | 81.076981 / 0.701012 |
| Factorized genuine I4 | 80.084692 / 0.710204 | 80.887372 / 0.674790 | 81.816458 / 0.651765 |
| Original shared4 | 80.571356 / 1.001493 | 80.735684 / 0.759718 | 80.944255 / 0.707684 |
| Live shared4 retrieval | 80.577677 / 0.835494 | 80.634559 / 0.725898 | 81.209708 / 0.695858 |
| Detached shared4 retrieval | 80.685122 / 0.945895 | 80.849450 / 0.697994 | 81.146505 / 0.685016 |
| Ordinary M1 live retrieval | 79.212489 / 0.780449 | 80.704083 / 0.711744 | 80.937935 / 0.716992 |
| Genuine ordinary I4 live retrieval | 79.882442 / 0.715728 | 81.152825 / 0.668771 | 81.784857 / 0.635158 |

The three [complete contrast partitions](SAGE_COMPLETE_RESULTS.json) preserve
every seed, class, comparator and atomic criterion; GCN and GAT are in the
corresponding `GCN_COMPLETE_RESULTS.json` and `GAT_COMPLETE_RESULTS.json` files.
No best arm, seed or class is promoted.

## How much survives the controls

All values below are **live shared minus reference**. Accuracy thresholds are
0.2 pp versus live I4 and 0.1 pp versus live M1/detached, with all three seeds
nonnegative and at least two positive. NLL deterioration must be at most 0.02
mean and 0.05 each seed.

| Backbone / reference | Accuracy mean pp | Accuracy by seed pp | NLL mean | Accuracy / NLL gate |
|---|---:|---|---:|---|
| SAGE / live I4 | +0.695234 | +0.891164, +0.663633, +0.530906 | +0.119766 | pass / fail |
| SAGE / live M1 | +1.365188 | +1.687524, +1.251422, +1.156617 | +0.055044 | pass / fail |
| SAGE / detached | −0.107445 | +0.246492, −0.417141, −0.151688 | −0.110402 | fail / fail |
| GCN / live I4 | −0.518266 | −0.739477, −0.758438, −0.056883 | +0.057127 | fail / fail |
| GCN / live M1 | −0.069523 | −0.037922, −0.322336, +0.151688 | +0.014153 | fail / pass |
| GCN / detached | −0.214891 | −0.246492, −0.474024, +0.075844 | +0.027904 | fail / fail |
| GAT / live I4 | −0.575149 | −0.474024, −0.777399, −0.474024 | +0.060700 | fail / fail |
| GAT / live M1 | +0.271773 | +0.379219, +0.284414, +0.151688 | −0.021134 | pass / pass |
| GAT / detached | +0.063203 | 0, −0.094805, +0.284414 | +0.010842 | fail / pass |

SAGE's mean NLL improvement versus detached passes its mean condition, but seed
7507 deteriorates by 0.083478 and fails the per-seed limit. Against live I4,
SAGE NLL worsens at every seed, especially 7507 (+0.339339). GCN detached
protection fails only the mean condition; its per-seed limit passes. Preserve
these distinctions rather than claiming every atomic condition failed.

SAGE live minus original shared is only +0.006320 pp, with one positive/two
negative seeds; detached already gains +0.113766 pp. The larger live-I4 gap
includes live I4's −0.139047 pp versus its original I4, alongside persistence of
the old shared accuracy advantage. Live SAGE still has worse NLL than both raw
I4 references (+0.139732 / +0.125290).

GAT's +0.265453 pp original-shared gain is real; detached already supplies
+0.202250 pp. Live's remaining +0.063203 pp is small/mixed. The live predictor
also trails raw ordinary/factorized I4 by −0.771078 / −0.606750 pp and even raw
ordinary M1 by −0.025281 pp. GCN loses original shared accuracy by −0.101125 pp
and both raw I4 references by −0.398180 / −0.252813 pp. Thus weaker live controls
cannot manufacture a strongest-reference success.

All three improve mean NLL over original shared: −0.165999 / −0.033821 /
−0.011826. GCN and GAT improve it at every seed; SAGE has one small adverse seed.
Those gains do not remove the same-information I4 NLL deficits.

## Member quality and the selected-state limitation

Values are mean-member accuracy / worst-member accuracy / mean-member NLL,
averaged over the three selected banks. Native components are reconstructed at
each arm's **mixed-selected checkpoint**, not its own native-best checkpoint.

| Backbone / bank | Native component | Actual mixed members |
|---|---|---|
| SAGE live shared | 78.510934 / 78.138036 / 1.092968 | 78.634180 / 78.188598 / 1.058850 |
| SAGE live I4 | 78.863292 / 78.548856 / 0.794693 | 79.024460 / 78.833270 / 0.790688 |
| GCN live shared | 78.838010 / 78.352926 / 0.926242 | 79.117684 / 78.542536 / 0.828437 |
| GCN live I4 | 80.198458 / 79.787637 / 0.743676 | 80.460751 / 80.109973 / 0.713948 |
| GAT live shared | 79.654911 / 79.237770 / 0.819041 | 80.016749 / 79.774997 / 0.761300 |
| GAT live I4 | 80.827329 / 80.476552 / 0.713161 | 81.149665 / 80.843130 / 0.690710 |

Live mixed members trail live I4 mean accuracy by 0.390279 / 1.343067 / 1.132916
pp and worsen mean-member NLL by 0.268162 / 0.114489 / 0.070591. Relative to
detached, live mean-member accuracy changes +0.192770 / −0.014221 / −0.055303
pp. Better SAGE members coexist with a worse live pool than detached.

Native pool live-minus-detached accuracy at those selected states is
−0.145367 / −0.252813 / +0.012641 pp. Selection, stopping and learned parameters
all change. Neither these native contrasts nor original-native comparisons
is a pure causal estimate of retrieval training. Same-state mixed-minus-native
is a conditional terminal-readout contrast, also evaluated at mixture-selected
states on encountered VALID.

## Exact coverage, repairs and harms

Counts sum three dependent readouts of the same nodes. The identity is
`Δpool correct = Δcoverage − Δlost alternatives + Δaggregation-only correct`.

| Live shared minus live I4 | Repairs / harms | Δcoverage | Δlost alternatives | Δaggregation-only | Δpool correct |
|---|---:|---:|---:|---:|---:|
| SAGE | 600 / 490 | +287 | +173 | −4 | +110 |
| GCN | 331 / 413 | +112 | +194 | 0 | −82 |
| GAT | 287 / 378 | −123 | −33 | −1 | −91 |

SAGE creates 670 newly covered readouts while removing 383; it serves 271 of the
newly covered and loses 399. GCN creates 393/removes 281, serves 118/loses 275;
its increased coverage is outweighed by additional pooling loss. GAT creates
241/removes 364, serves 58/loses 183; lost coverage remains material. Newly covered
and newly repaired sets can differ when a reference pool is correct without a
correct member, so those counts are not interchangeable.

Against original shared, repairs/harms are 399/398, 226/242 and 282/240:
net +1/−16/+42 correct. Against detached they are 409/426, 203/237 and 298/288:
net −17/−34/+10. The exact flows prevent interpreting churn or larger hidden
similarity gradients as retained useful diversity.

### What terminal memory actually repairs

At the same mixed-selected live states, adding the .5/.5 terminal memory gives
+0.164328 / +0.164328 / +0.316016 pp and NLL −0.017099 / −0.060166 / −0.035842.
Its pooled repairs/harms are 79/53, 85/59 and 92/42. It creates 40/41/56 new
correct-member alternatives but serves only 0/2/3; 79/83/89 of its repairs instead
serve correct alternatives already present in the native bank. Net coverage
falls 27/26/19 while pooling losses fall 53/53/69.

**It corrects zero native common strict false-rival cases in all nine live shared
banks.** Native cohort sizes by seed are SAGE 671/652/657, GCN 749/713/701 and
GAT 777/778/734. Every live mixed bank also has zero aggregation-only rescues.
This rejects the hoped-for common-error acquisition effect in these conditional
serving diagnostics. It does not show that retrieval can never repair such an
error, nor measure whether training repaired a strict rival from a different
original checkpoint. Live M1 terminal memory repairs 102/105/103 native wrong
cases across SAGE/GCN/GAT; detached shared repairs 1/1/5 native strict-rival cases.
The wider operation permits new rankings; this exact shared rule did not use
them effectively enough.

## Costs, failures and custody

The new families ran concurrently; observed owner walls are not summed into an
isolated elapsed-time or speed claim. Acquisition subtotals are nested inside
their owners. All per-fit parameters/checkpoints, CUDA/RSS peaks, selected steps,
actual stops, serving and reference costs remain in the detail partitions.

| Backbone | New acquisitions s | New owner s | New selected readout s | Backwards/Adam steps | Native/retrieval member paths | Similarity MACs |
|---|---:|---:|---:|---:|---:|---:|
| SAGE | 2,418.907 | 2,433.868 | 3.640 | 7,873 | 32,705 | 6,586,143,173,120 |
| GCN | 2,388.183 | 2,403.089 | 4.195 | 7,518 | 28,791 | 5,798,829,235,200 |
| GAT | 2,403.905 | 2,418.647 | 3.793 | 7,530 | 28,695 | 5,779,518,520,320 |

There are 90,191 actual new native/retrieval member paths and 22,921
backwards/updates in total. Retrieval uses the same native forward, adding no
second native call; that does not make its similarities, losses or backward free.
Shared stored parameters are 216,674 / 148,066 / 148,578 versus four native bodies
820,776 / 558,632 / 560,680. The helper adds zero trainable parameters. Maximum
live shared allocated GPU bytes are 1,203,675,648 / 1,242,091,008 / 3,817,856,000;
sequential live-I4 body peaks are 358,940,672 / 569,702,400 / 1,390,908,928.
These acquisition peaks are not simultaneous four-body inference benchmarks.

The 33 reused reference acquisitions per backbone retain acquisition subtotals
237.862 / 486.035 / 583.657 s. Their full original 39-acquisition parents retain
427.808 / 754.036 / 857.529 s; these include overlapping reference work and are
not added again. TRAIN-only qualification v2 cost 8.763 s, 18 forwards/54 member
paths, nine discarded updates and six retrieval VJPs. The earlier missing
`backbone` configuration failure remains preserved separately.

The successful CPU reader made zero new model forwards. Its observed readout/
terminal interval is 4.128596 s (17:09:03.236605–17:09:07.365201 UTC); the deferred
process's 965.033608 s includes waiting for owners and is not reader CPU time.
The failed initial large readout fetch and successful compact recovery are
preserved; recovery reran no training, reader or model forward.

The full report remains on the authorized host: 7,613,237 bytes, SHA256
`be58d3207df6c0071c457762d180fa1397f4fb5dbf1adcbbe08a1d0cd2fe8ed7`.
[COMPLETE_ANALYSIS_SUMMARY.json](COMPLETE_ANALYSIS_SUMMARY.json) binds all 30
local partitions, owners, source/config/freezes and historical references.

## Scope and closure

Cosine support-one-hot attention is the known Matching Networks decoder, with
NCA stochastic-neighbor ancestry; TPN and UniMP already supply related learned
similarity/label-aware graph operations. This study evaluates its joint native
learning and shared-route composition, with no new-primitive claim.

Whole-Q exclusion prevents literal lookup, but native CE still supervises Q;
class stratification uses TRAIN truths. There is no uniform-mask unbiasedness or
withheld-label representation claim. Half-support fitting and all-anchor serving
differ. Repeated fitting/selection on this connected WikiCS graph is encountered
development; three seeds are optimization repeats, not independent graph
populations. No TEST score or unused confirmation is earned.

Preserve GAT's old-shared gain, SAGE's accuracy persistence and all same-state
serving/NLL improvements. Retire the exact portable live-retrieval superiority
rule, with no temperature, mixture, coefficient, support-fraction or optimizer
rescue. The ready UniMP source remains an unexecuted reference packet; this
negative pilot does not activate it or any new decoder. Root owns a separately
frozen next decision after this whole-family report.
