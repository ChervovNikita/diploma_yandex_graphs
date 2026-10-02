# Independent interpretation of the fixed Stage1 result

## Decision and evidence scope

The frozen stdlib assessment records **STAGE1_NO_GO**: all 54 expected fits are retained and marked `COMPLETE_COMPETENT`, with no global issues. This means the logged schedules, selection, provenance, storage and supervision passed the stdlib checks; it does not establish convergence or certify tensor forward semantics. All 54 fits completed 1,000 updates under the same frozen minimum, maximum, patience and evaluation schedule.

The separate native audit verified **15/54** selected-state replays and marked **39/54 inconclusive**, so `complete=false`. Every failure reason is the independent selected-state forward logit comparison; reported maximum absolute discrepancies range from 2.86102294921875e-6 to 8.58306884765625e-6. The retained source flow reaches this comparison after the selected/deployment checkpoint, node-order and CPU64 stored-metric checks. Passing those earlier checks is narrower than a complete forward certificate. The maximum absolute discrepancies alone are not a replacement for the frozen combined absolute/relative comparison.

Keep the adopted tolerances exactly: logits atol=1e-6, rtol=1e-5; NLL atol=2e-6, rtol=1e-5; fraction atol=1e-7; exact pooled/member argmax equality. No tolerance widening, retry, favorable subset or integrity-metric replacement is warranted. The numerical interpretation below describes the retained primary scores used by the fixed stdlib screen. In particular, every coordinate, factor and permutation case is among the inconclusive forward replays. The recorded no-go remains; the candidate is closed and no continuation is admitted.

## Do factors add value beyond permutation-only?

Use the prescribed ratio of three-seed mean NLLs per graph, then equal graph weights. `coordinate` (PF) improves over `permutation` (P0) by only **0.100927% on AmazonPhoto** and is **0.034397% worse on CoauthorCS**. The equal-graph gain is **0.033265%**, far below the frozen 1% requirement. The paired descriptive PF/P0 gains for seeds 17/29/43 are −0.003040%, +0.160076%, +0.147285% on Photo, and −0.030745%, −0.049887%, −0.022294% on CS. These include every seed; the modest Photo effects do not rescue the required two-graph screen.

Factor-only (F) improves over bias-only by 0.186915% on Photo and 0.154037% on CS, and over the original Stage1 arm by 0.223746% and 0.206393%. These are descriptive differences, not a new gate or a general claim of no factor effect. The central proposed addition has not demonstrated the required contribution beyond the corresponding permutation arm in this fixed recipe/split setting. The conspicuous CS improvement over F/original is already almost entirely present in P0; attributing it to the extra factors would misstate this ablation.

## Why the frozen screen fails

| Requirement | AmazonPhoto | CoauthorCS | Fixed consequence |
|---|---:|---:|---|
| PF gain over F | 1.285226% | 13.486769% | Photo fails the 2% requirement |
| PF gain over original Stage1 arm | 1.506096% | 13.665326% | Photo fails the 2% requirement |
| Strict paired PF wins over F | 1/3 | 3/3 | Photo fails the minimum 2/3 |
| Strongest byte control | untied | single | Prescribed minimum of S/U/H mean NLL |
| PF gain over strongest byte control | −6.995769% | 14.823889% | Photo also fails the allowed −1% floor |
| PF/F full-training cost ratio | 1.108853 | 1.120592 | Both pass the 1.25 cap |
| PF/F warm-inference cost ratio | 1.113097 | 1.129572 | Both pass the 1.25 cap |

The equal-graph PF/bias-only gain is 7.544885%, but the equal-graph PF/P0 gain fails; both were required. Accuracy and the PF/F trainable/dense-parameter match gates pass on both graphs. Passing these gates cannot override the failed NLL, paired-win and byte-control gates.

## Are the byte and cost comparators fair?

The comparisons are fair for the specific prospectively frozen resource question. The S/U/H widths were fixed to match PF total model tensor bytes within 1%, including its 221,184 fixed index-buffer bytes. The observed S/U/H deviations from PF are −0.073033%, −0.533801%, +0.012538% on Photo and −0.101381%, +0.634151%, +0.236195% on CS. All are admitted by the original tolerance. The non-index controls can spend the same storage budget on more learned parameters. Thus equal total storage is established; equal trainable capacity, width, member independence and computation are not established by that byte constraint.

Single is one actual member; untied uses four independent narrower models; heads uses four output heads on a shared representation. The prescribed strongest-control selection over all S/U/H is an explicit prospective comparator rule, with fixed tie order S then U then H. It prevents choosing a weak favorable baseline; its minimum on this same validation split is not an unbiased estimate of performance on unseen graphs or partitions. Related non-GT arms have the same within-graph LR, dropout and schedule, so these are fixed-recipe comparisons. They do not establish equal tuning-budget optimality for each architecture.

PF and F have equal learned parameter counts and dense bytes, with only the admitted index-buffer difference. Their cost comparison therefore answers the incremental permutation overhead under the same factor model. The frozen full-training cost includes scientific setup, complete updates, scheduled validation, selected/final checkpoint I/O, reload, deployment preparation and scientific residual bookkeeping; it excludes the three measured profiling blocks. The mean ratio is computed per graph, with no selective timing exclusions. Warm inference is the selected-checkpoint all-member full-graph forward, gather and probability pooling. This provides an internally consistent machine/protocol comparison, not an end-to-end cold-process deployment or actual GPU busy-time claim.

PF/P0 full-training ratios are 1.130954 (Photo) and 1.115266 (CS); warm ratios are 1.051946 and 1.062462. The additional factor cost accompanies negligible recorded NLL changes. Passing PF/F cost caps does not imply superiority to the byte controls: against the strongest control, PF is 1.407177× training and 2.088046× warm inference on Photo, where its NLL is worse; on CS it is 4.851279× training and 4.723455× warm inference, in exchange for lower NLL. These are descriptive tradeoffs, without new cost gates or a strict multi-metric dominance claim.

The GT separate single arm remains useful as a retained architectural reference. It has one actual member, width 512 and five layers, 43,604,000/56,029,244 bytes on Photo/CS, and uses the inherited 3e-5 LR on CS while the related Stage1 controls use 3e-4. It is neither the matched-byte control nor a pure factor/permutation ablation. Its Photo mean NLL is worse than PF; its CS mean NLL is close to PF at substantially larger storage. Neither observation changes the prescribed decision.

## Complete retained score and resource inventory

Each table row contains seeds **17, 29, 43**, in that order. All 54 rows of the fixed assessment are competent under stdlib checks; `I` means inconclusive native forward replay and `V` means verified native replay. Cost means use all three seeds. Display rounding is descriptive; the frozen gates used unrounded values.

### AmazonPhoto

| Arm | NLL 17 | NLL 29 | NLL 43 | Mean NLL | Training mean, s | Warm mean, ms | Model tensor bytes | Replay 17/29/43 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| original | 0.144509032 | 0.170059189 | 0.164375558 | 0.159647927 | 62.967749 | 21.720120 | 4,745,104 | I/I/I |
| factor | 0.144295141 | 0.169532120 | 0.164044902 | 0.159290721 | 71.586787 | 22.298481 | 4,883,344 | I/I/I |
| permutation | 0.158308655 | 0.149099335 | 0.164799020 | 0.157402337 | 70.187835 | 23.594718 | 4,993,936 | I/I/I |
| bias_only | 0.144474864 | 0.169943899 | 0.164348289 | 0.159589017 | 61.953387 | 21.062365 | 4,772,752 | I/I/I |
| coordinate | 0.158313468 | 0.148860663 | 0.164556295 | 0.157243475 | 79.379247 | 24.820372 | 5,104,528 | I/I/I |
| single | 0.198094532 | 0.215147391 | 0.210039422 | 0.207760448 | 20.979853 | 5.966805 | 5,100,800 | I/I/I |
| untied | 0.143376961 | 0.151106045 | 0.146403983 | 0.146962330 | 56.410291 | 11.886888 | 5,077,280 | V/V/V |
| heads | 0.194697797 | 0.189206511 | 0.193897307 | 0.192600538 | 26.998934 | 5.903620 | 5,105,168 | V/V/V |
| gt_sep_single | 0.173295811 | 0.174891949 | 0.162101626 | 0.170096462 | 141.349278 | 50.115969 | 43,604,000 | I/I/I |

### CoauthorCS

| Arm | NLL 17 | NLL 29 | NLL 43 | Mean NLL | Training mean, s | Warm mean, ms | Model tensor bytes | Replay 17/29/43 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| original | 0.156951874 | 0.163536504 | 0.152003899 | 0.157497426 | 130.926400 | 43.292609 | 11,054,896 | I/I/I |
| factor | 0.156522036 | 0.163217172 | 0.151777878 | 0.157172362 | 148.857352 | 46.684453 | 11,193,136 | I/I/I |
| permutation | 0.135078758 | 0.137492195 | 0.135213450 | 0.135928134 | 149.568237 | 49.633283 | 11,303,728 | I/I/I |
| bias_only | 0.156841114 | 0.163455620 | 0.151947781 | 0.157414839 | 131.385048 | 43.609283 | 11,082,544 | I/I/I |
| coordinate | 0.135120288 | 0.137560785 | 0.135243595 | 0.135974889 | 166.808313 | 52.733462 | 11,414,320 | I/I/I |
| single | 0.156502202 | 0.159928486 | 0.162488416 | 0.159639701 | 34.384400 | 11.164172 | 11,402,748 | I/I/I |
| untied | 0.163695499 | 0.160585806 | 0.157478809 | 0.160586705 | 59.777680 | 16.009266 | 11,486,704 | V/V/V |
| heads | 0.192824990 | 0.202245727 | 0.193253830 | 0.196108182 | 38.539292 | 12.143843 | 11,441,280 | V/V/V |
| gt_sep_single | 0.133896992 | 0.140548050 | 0.135932103 | 0.136792382 | 181.793482 | 61.550052 | 56,029,244 | V/V/V |

## Limits and next review

This is an exploratory result from three model seeds on one prospectively fixed core0 partition in each of two specific graphs. Checkpoint selection and reporting use the same held-out validation partition. No test labels were read, no intervals were computed, and no graph-population significance, scientific novelty or paper verdict is inferred. Keep all arms, seeds, graphs and integrity failures in any subsequent accounting.

A forthcoming conformal pilot must have a separate scientific contract and cannot rescue this failed coordinate-factor screen. Once its source packet is supplied, the focused source-only review should check the score/quantile implementation, model-selection versus calibration versus evaluation label use, graph-dependence assumptions behind coverage language, prospective coverage/size/cost estimands, and the exact retained pilot population. No pilot source has been supplied to this reviewer yet; no implementation verdict is made here.

## Inputs inspected

- `coordinate_ensemble_execution_root_v1/stage1_assessment_run01/ASSESSMENT.json`
- `coordinate_ensemble_execution_root_v1/tensor_replay_selected_v1_run01/TERMINAL.json`
- Frozen Stage1 protocol and adopted assessment/replay controls retained from source preparation.

Inspection used JSON reads and arithmetic only. No evidence was rehashed, no tensor deserialization or model execution occurred, no validator was run, and no SSH was used.
