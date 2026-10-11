# Closed SAGE endpoint VALUE results and confirmation design

E4 improves raw combined accuracy over every reference in the complete WikiCS development family and passes all frozen primary, endpoint-attribution and confidence caps. The gain over a genuine independent ensemble is small and uncertain, with worse calibrated NLL and Brier. The next useful experiment is one unused whole-pipeline confirmation of this fixed operation. A sequential own-CE implementation can reduce retained training tapes before that confirmation; it is an execution change with a mathematical equivalence condition.

## Complete procedure and established ancestry

The fixed family has 15 banks, 24 optimizer/body fits and seeds 7301, 7403 and 7507. BASE is an unchanged four-route shared internal-factor SAGE bank, not an ordinary single. E4 changes each nonself neighbor VALUE before the original SAGE mean. Two affine builders per layer consume live source and target states, create bounded 4 by 4 endpoint maps and apply their paired transport to the neighbor's four 32-channel blocks. N4 has the same parameters and matrix action, conditioned on the receiver state and its SiLU transform. Equipped M1 has a full endpoint-equipped factorized body; equipped I4 has four independently initialized bodies/builders, unscaled own losses, optimizers and selectors.

All arms use two residual SAGE blocks, width 128, LayerNorm, dropout 0.2, AdamW 0.001 with zero decay, 1,000 updates maximum, 300 stale-update patience and the earliest strict development-accuracy maximum. Shared banks optimize the mean of four own cross-entropies and serve the probability mean. The support, self VALUEs, native root/neighbor maps, FFN and original native parameters remain. Identity builder initialization preserves matched starting routes; private builder gradients become active after shared weights move. Selector and calibration reuse the same 5,274 WikiCS development nodes. The five label-free folds have sizes 1,055/1,055/1,055/1,055/1,054. All 75 fixed CPU float64 temperature fits completed 500 updates, without retries or a final whole-development refit.

The implementation inherits the existing native SAGE/FactorLinear owner, member RNG streams, own-CE training and probability reader. ECC/NNConv provide edge-conditioned VALUE matrices, GNN-FiLM provides receiver-conditioned modulation, NSD/DNSD provide endpoint incidence transport, and BatchEnsemble provides shared matrices and private multiplicative factors. The earlier Tolokers normalized-sheaf failure remains a failure under its different operator and task. A native scalar-SAGE placement result does not establish a new primitive, a benefit of private builder ownership, or a causal optimization explanation. E4 versus N4 tests endpoint dependence of the complete learned procedure; a tied-builder intervention or nonlinear joint four-path single would be needed for those broader claims. Neither is a next experiment in this plan.

## Every fitted reference

All entries are three-seed means. Accuracy and member mean are percentages. Higher NLL/Brier is worse.

| Arm | Raw accuracy | Member mean | Raw NLL | Calibrated accuracy | Calibrated NLL | Calibrated Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BASE | 80.6282 | 78.3766 | 1.137994 | 80.7294 | 0.685842 | 0.292675 |
| E4 | 81.1971 | 79.4495 | 0.761107 | 81.1971 | 0.652050 | 0.284499 |
| N4 | 81.0517 | 79.5174 | 0.737769 | 81.0201 | 0.649441 | 0.284886 |
| equipped_M1 | 80.2174 | 80.2174 | 0.765341 | 80.2174 | 0.678221 | 0.297737 |
| equipped_I4 | 80.8937 | 80.2127 | 0.688787 | 80.9506 | 0.636007 | 0.283364 |

| E4 minus reference | Raw gain pp | Descriptive raw 95 percent interval | Calibrated gain pp | Calibrated NLL change |
| --- | ---: | --- | ---: | ---: |
| BASE | +0.568828 | [+0.114597, +1.023059] | +0.467703 | -0.033791 |
| equipped_I4 | +0.303375 | [-0.356047, +0.962797] | +0.246492 | +0.016044 |
| N4 | +0.145367 | [+0.001469, +0.289265] | +0.176969 | +0.002609 |
| equipped_M1 | +0.979649 | [+0.212408, +1.746889] | +0.979649 | -0.026171 |

Primary raw gains against BASE and I4 pass the frozen mean margin of 0.2 points, all-three-nonnegative and at-least-two-positive requirements. The E4 minus N4 endpoint contrast passes its 0.1-point rule with all three gains positive. Equipped M1 is reported without post hoc reference selection. These three-seed intervals condition on one encountered graph and its selected procedures. They do not account for graph variation or prior design selection; raw sign-flip p is 0.25 for all four E4 contrasts.

I4 is the important remaining limitation. Raw E4 minus I4 seed gains are +0.303375/+0.568828/+0.037922 points, and its interval is [−0.356047,+0.962797]. After calibration one difference turns negative: +0.284414/+0.474024/−0.018961 points, with mean +0.246492 and interval [−0.371239,+0.864224]. Calibrated NLL harm is +0.016043762345, with seed harms +0.010781994673/+0.024974662102/+0.012374630260 and interval [−0.003271,+0.035359]. All caps pass (mean at most 0.02; each seed at most 0.05), although all three I4 harms are positive. Passing this practical screen does not establish statistical noninferiority. Calibrated Brier also deteriorates by +0.001134758181 against I4. E4 improves calibrated NLL against BASE by −0.033791 and against M1 by −0.026171; it worsens it against N4 by +0.002609.

## What the observed gain contains

Against BASE, E4's mean member accuracy increases by 1.072873 points, while its additional accuracy from pooling decreases by 0.504045 points, leaving the 0.568828-point bank gain. Coverage falls from 13,768 to 13,662 cases; pooling losses fall from 1,011 to 815. With no aggregation-only correction in either raw bank, that produces 90 extra correct decisions. Direct repairs/harms are 512/422. Broader coverage is therefore not the explanation of this comparison.

Against N4, E4 has slightly weaker average members (−0.067943 points), 12 fewer covered cases and 35 fewer pooling losses, leaving 23 extra correct decisions and repairs/harms 213/190. Against I4, E4's average members are weaker by 0.763178 points, but coverage grows by 44 and pooling losses fall by four, leaving 48 extra correct decisions; repairs/harms are 475/427. Against M1 the bank supplies 970 extra covered cases and loses 815 in pooling, leaving 155 extra correct decisions; repairs/harms are 595/440. These are decompositions of separate fitted, stopped and selected procedures, not isolated causes.

The 15,822 readouts repeat the same 5,274 nodes three times. Current E4 strict-rival cohorts contain 679/689/696 cases with zero current-pool corrections, as expected when all current routes share the same wrong rival. E4 still corrects 28/28/23 cases from the fixed BASE strict-rival cohorts. Removal of a BASE rival and creation of a new E4 rival are also both present. A zero current-rival correction count therefore does not establish zero repair of prior failures. Archived raw-pool and raw-member argmax checks have zero FP64 disagreements; stable NLL differs from original reporting by at most 1.11e−16. Calibration does not change member argmaxes.

E4's selected steps are 69/58/74 (completed 369/358/374), while BASE selects 467/334/165 (completed 767/634/465). N4 selects 61/59/65. I4's twelve own selectors choose steps 30–53. Different acquisition histories are part of the comparisons; this is not an equal-update or isolated-learning attribution.

## Actual costs and resource uncertainty

Parameters are per served bank. Timings below are sums over three seed groups, not per-forward latency or concurrent family wall time. Selected readout includes forward, metrics, CPU transfer and probability pooling.

| Arm | Parameters | Acquisition seconds | Restore seconds | Selected readout seconds | CUDA allocated / reserved GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| BASE | 216,674 | 234.671 | 0.614 | 0.161034 | 1.162 / 1.464 |
| E4 | 237,474 | 1566.748 | 0.862 | 1.325601 | 16.120 / 17.260 |
| N4 | 237,474 | 636.351 | 0.957 | 0.311309 | 4.018 / 4.238 |
| equipped_M1 | 225,600 | 372.650 | 0.467 | 0.351277 | 4.652 / 5.719 |
| equipped_I4 | 902,400 | 1487.302 | 3.190 | 1.293800 | 4.652 / 5.719 |

Recorded lifetime process RSS peaks are about 1.062 GB for BASE/E4 and 1.063 GB for the later N4/M1/I4 arms; they are process lifetime peaks, not isolated arm CPU allocations. Checkpoint and optimizer bytes retain their exact recorded definitions in the source extract.

The family owner records 4,315.771 seconds (71.93 minutes), with concurrent Penn94 acquisition on the sole allocation GPU. The five acquisition sums total 4,297.723 seconds; these nested timers must not be added to owner wall time. Calibration separately records 111.012 seconds summed fits and 112.224 seconds wall time, for 37,500 successful scalar updates. Saved disposable WikiCS qualification cost 16.911 seconds, 34 native trajectories and 16 discarded backwards, with a 16.107 GB allocated peak. All preparation, discarded work and collection costs belong in resource accounting; these observations do not establish isolated serving latency or speed superiority.

Completed logical backwards/Adam steps are BASE 1,866, E4 1,101, N4 1,085, M1 1,018 and I4 4,088: 9,158 total. TRAIN trajectories total 21,314, with the same selector trajectory count and 51 selected trajectories. E4's full acquisition/selection/selected builder count is 64.003 trillion MACs, versus N4's 1.666 trillion; these are hook-defined operation counts, not measured FLOPs or a future runtime estimate. E4 stores fewer weights than I4 but takes more recorded acquisition time and retains much larger training tapes.

The completed E4 peak is 16,119,818,240 bytes (16.120 decimal GB, about 15.013 GiB) on 442,907 WikiCS edges. Scaling all of that peak only by Penn94's published planning bound of 2,724,458 edge entries gives about 99.16 GB. It is a warning that unchanged four-route tape retention may exceed an 80 GB device, not a measured Penn94 requirement. Dividing that proxy by four gives an optimistic one-route 24.79 GB; scaling the measured equipped-single peak gives about 28.62 GB. Fixed graph, node, parameter, temporary and allocator terms do not divide by four, and both proxies remain unqualified.

Official ogbn-arxiv metadata gives 169,343 nodes and 1,166,243 directed edges. A directed-edge proxy is about 42.45 GB for four retained routes; a bound after reciprocal edges and one loop per node is 2,501,829 entries and about 91.06 GB. The graph has far more node states than WikiCS. Neither figure certifies an 80 GB fit. Use sequential route backward for the fixed mean-own-CE objective and determine the actual support, peak and owner cap from the complete graph before admitting fits. Do not change directedness, sampling or graph size after an outcome or failed allocation merely to preserve a success claim.

## Sequential route backward

For the fixed shared objective, L(theta)=one quarter of the sum of the four route cross-entropies. At one unchanged parameter state, the gradient is the sum of their quarter-scaled gradients. Thus one zero_grad, then each route forward and backward on CE/4, then one AdamW step preserves the mathematical update. Routes have distinct persistent dropout streams; route order and stream advancement must be identical. Parameters, optimizer moments and native buffers cannot update between route losses. Clip once after accumulation if clipping is introduced; the present source has none. Singles/I4 keep their unscaled own losses and independent updates.

Practical equivalence requires releasing every route's graph and hidden-state references after its backward, keeping only detached TRAIN logits for diagnostic probability pooling, and preserving native/builder ownership, saved stream state, selector order and strict restoration. LayerNorm has no BatchNorm running-statistic complication in this recipe. Separate reduction order can change floating-point rounding, so mathematical equivalence is not bitwise replay evidence. A future narrow source comparison must check gradients, one optimizer update, route RNG/state restoration and the complete graph's TRAIN/selector/serving memory at identical starts. There is no pooled-loss gradient, auxiliary objective or cross-route differentiable term here; adding one would require a different argument. Sequential backward remains an implementation control, not novelty.

The installed source currently requires exactly 11,701 nodes, 300 input features and ten classes. A confirmation port must replace all factual dimension/provider guards with the chosen dataset's fixed dimensions, retaining width 128 and the declared 256-to-16 builders. This is a source change to freeze before fitting, not a rename of the existing WikiCS executable.

## Choice of confirmation population

Prefer one ogbn-arxiv official temporal confirmation while preserving the running Penn94 F programme. The saved official OGB page specifies 128 input features, 40 classes, TRAIN through 2017, VALID in 2018 and TEST from 2019 onward. It supplies a substantial citation task with fixed official evaluation membership. Root identifies it as untouched for this task; the saved public documentation establishes the split, while actual target-identity custody and safe provider hashes remain root freeze inputs. Do not turn source availability into an unqualified unused-input certificate.

Penn94 is the strongest already prepared social-graph option, with a running ordinary-sharing confirmation and published fixed split 0. E could be frozen prospectively for that same population only while its heldout outcomes remain closed and without changing the running F protocol; it would answer an additional comparison on one graph. Its unchanged four-tape resource risk is the main reason to resolve source retention first. Root controls any priority, reservation or release decision. No partial Penn metrics were used.

Coauthor CS has a saved 3,660-node reserved custom holdout, 18,333 nodes, 163,788 directed edge entries and 6,805 features. It is attractive for edge memory, but its existing development fits, missing physical-host receipt, later/external raw-provider access account and old reservation prevent automatic unused admission. It is not selected as an outcome-dependent fallback. New WikiCS masks do not create an unused task, and WikiCS TEST history remains unresolved.

## Two bounded next actions

1. Prepare an inactive source amendment that accumulates CE/4 gradients one route at a time for E4 and matched shared controls. Retain the exact logical optimizer, route RNG, selector, checkpoint and reference contracts. Root can admit only the specific equivalence and complete-graph resource checks needed to establish that execution change; no new objective, factor-placement grid or second context family is proposed.

2. Root may freeze one ogbn-arxiv confirmation of E4, BASE, N4, equipped M1 and genuine equipped I4, with ten paired fresh seed blocks: 50 banks and 80 body/optimizer fits. Keep depth/width/dropout/AdamW/CE/pooling and 1,000-update/300-stale ceilings fixed. Separate SELECT and CAL label-free inside official VALID (seed 11709, floor-half CAL), preserve official TEST, and fit one fixed 500-update temperature per selected bank: 50 scalar fits/25,000 scalar updates. The ceiling is 80,000 logical Adam updates, 170,000 TRAIN and 170,000 SELECT route trajectories, and 170 final selected trajectories; sequential shared backward changes the autograd-call ceiling to 170,000 without adding logical updates. Use the declared native preprocessing consistently for all arms, freezing direction handling, coalescing and self loops before any fit. Root locks actual seeds, support, provider/dimension/source identities, scorer/calibrator, missing-pair handling, multiplicity, scheduling and finite wall/RSS/CUDA caps before fitting. A suggested primary family is E4 minus BASE/I4, Holm one-sided paired-t alpha 0.05 for a positive effect, alongside the separate observed mean ≥0.2-point screen. Endpoint attribution versus N4 has a predeclared ≥0.1-point screen and separate multiplicity-controlled uncertainty; M1 remains fully reported. Retain calibrated NLL mean/per-seed caps 0.02/0.05 as a separate practical screen, with raw/calibrated NLL and Brier intervals. Complete every native state and CAL endpoint before one full TEST readout. Ten seeds remain one graph and provide no power guarantee or proof that the true gain exceeds the practical margin. Failures and wide intervals remain results; no TEST-selected top-up, grid or replacement population follows.

## Primary source custody

The saved official OGB documentation and original retrieval receipt are copied under `primary_sources/` (official URL https://ogb.stanford.edu/docs/nodeprop/). The exact E protocol, family/installer/helper source, completed configuration, saved resource observation, ancestry records and conditional graph-opportunity records are preserved there. `SOURCE_BINDINGS.json` identifies original local paths, byte counts, hashes and read scopes. `CLOSED_E_SOURCE_EXTRACT.json` preserves all five arms, all eight E4 raw/calibrated comparisons, every selected step/cost and all diagnostics. The original complete outputs retain all 20 ordered contrasts per readout. `RESOURCE_UNCERTAINTY.json` labels every scaling scenario as metadata arithmetic. `NEXT_ACTIONS.json` contains only the two inactive proposals; root owns freeze and execution.
