The complete nine-fit packet supports a meaningful exploratory comparison of context routing under a frozen design. It shows changes in which development nodes are correct and in how members rank particular wrong rivals. It does not demonstrate a method advantage: ROUTE ties COMMON on pooled accuracy at one seed and loses at the other two, has a negative mean accuracy contrast against PERMUTED, and fails the unchanged prospective Stage1 rule. Member accuracy remains comparable to COMMON within the specified tolerances, but correct-member coverage falls and newly acquired common wrong rivals offset the clearing of original ones.

This assessment was made on 2026-10-08 using only `context9_closed_result_independent_packet_20261008_v1`. I verified the byte sizes and SHA-256 hashes of all 14 files listed in `MANIFEST.json`; every entry matched, and the directory contained no unlisted file other than the manifest itself. I used Python standard-library arithmetic on the supplied aggregates. No allocation server, raw data, checkpoint, model, other research history, or external source was accessed, and the packet was left unchanged.

The hash check establishes the integrity of the supplied packet against its manifest. It does not independently authenticate the server artifacts referenced by hashes inside the packet. Raw logits, representations, truth, IDs, masks, frozen targets, and selected checkpoints are absent. Consequently this is an independent assessment of aggregate evidence and internal consistency, not an independent reproduction of inference, training, cohort construction, or data provenance.

The frozen comparison has three fixed optimizer seeds, 8101, 8203, and 8307, with four shared members in each of COMMON, ROUTE, and PERMUTED. Pooling is the arithmetic mean of member probabilities. COMMON assigns each route the mean of four normalized context targets; ROUTE assigns the four context targets separately. The contexts are X, X−PX, PX, and P²X. PERMUTED changes context assignment within the stated class, panel-membership, and restricted-positive-degree constraints. The alignment weight, member count, two-view construction, training recipe, and pooled development-selection rule are held fixed in the protocol. These are useful controls for an exploratory comparison.

The packet reports that all nine original 1100-epoch fits and selected-checkpoint hashes were validated before outcomes were opened, that the owner and children were terminal, and that all three COMMON collections and cohort freezes preceded candidate inspection. All nine cells are present, marked complete, and have four completed member inference calls. The readout uses original selected states, without training, reselection, calibration, or automatic retry. The endpoint and custody statements are server-derived metadata in this packet; the referenced endpoint receipts and checkpoints are not available for my own verification. The collection cell list is arranged by seed, so its list order alone does not establish inference chronology; the chronology is explicitly declared by the COMMON cohort metadata.

All outcomes below use the same 5274-node WikiCS split0 development population that selected the checkpoints. The packet identifies it as the official validation/stopping-mask union. Its interface metadata also says that official source verification was not performed by the interface. Shape, domain, and role checks and inherited data hashes are reported, but upstream provenance and those raw hashes cannot be independently checked here. There is no independent TEST evidence.

The complete pooled results are as follows. Accuracy is in percent; NLL is in nats; Brier is the sum over ten classes without division by ten. C, R, and P denote COMMON, ROUTE, and PERMUTED.

| Seed | Arm | Selected epoch / global flag | Pooled correct / 5274 | Accuracy (%) | NLL | Brier |
|---|---|---|---:|---:|---:|---:|
| 8101 | C | 149 / true | 4296 | 81.45620 | 1.467124 | 0.335930 |
| 8101 | R | 138 / true | 4296 | 81.45620 | 1.379193 | 0.335055 |
| 8101 | P | 131 / true | 4291 | 81.36140 | 1.243044 | 0.331516 |
| 8203 | C | 62 / false | 4295 | 81.43724 | 1.105396 | 0.318757 |
| 8203 | R | 135 / true | 4293 | 81.39932 | 1.250382 | 0.331778 |
| 8203 | P | 140 / true | 4297 | 81.47516 | 1.279381 | 0.330969 |
| 8307 | C | 117 / true | 4296 | 81.45620 | 1.210477 | 0.321208 |
| 8307 | R | 118 / true | 4293 | 81.39932 | 1.131237 | 0.317742 |
| 8307 | P | 126 / true | 4295 | 81.43724 | 1.101450 | 0.327852 |

Selection at different epochs and at different values of the saved global flag is part of the declared selection procedure. These are selected-model comparisons, rather than comparisons at a common training step. The reported float32 reevaluation accuracies exactly equal stored pooled and member accuracies for all nine cells. The integer-count accuracies round to those float32 values. This verifies the stated aggregate accuracy agreement; it does not establish bitwise prediction or representation parity, which the packet explicitly does not claim.

| Contrast | Accuracy changes at 8101, 8203, 8307 (pp) | Equal-seed mean (pp) | Seed SD (pp) | Descriptive 95% df2 interval (pp) |
|---|---|---:|---:|---|
| R−C | 0.00000, −0.03792, −0.05688 | −0.03160 | 0.02896 | [−0.10355, 0.04035] |
| R−P | +0.09480, −0.07584, −0.03792 | −0.00632 | 0.08961 | [−0.22891, 0.21627] |

R−C corresponds to zero, two fewer, and three fewer correct nodes. R−P corresponds to five more, four fewer, and two fewer. A favorable first-seed comparison against PERMUTED therefore does not survive the complete seed roster. The R−C mean pooled NLL change is −0.00739485, with per-seed changes −0.08793062, +0.14498581, and −0.07923974. Its mean Brier change is +0.00289316. Against PERMUTED, mean NLL is worse by +0.04564568 while mean Brier improves by −0.00192057. These mixed probability-score results do not establish a consistent predictive-quality advantage. The NLL improvement against COMMON passes the specified mean screen, but it is small relative to seed variation and includes a large deterioration at 8203.

The intervals are computed from three paired optimizer-seed differences with sample SD and the df2 t coefficient 4.302652729911275. I checked every supplied contrast metric, including its three values, mean, SD, and interval arithmetic. They are internally consistent. These intervals describe this selected single-graph experiment; they do not supply confirmation on unused data. The three seeds are the replication units. Nodes, four members, overlapping cohorts, and repeated evaluations of the same graph cannot be treated as additional independent model replicates. The evidence also does not establish equivalence or universal failure of context routing.

All 36 individual member accuracies are close: the complete range is 81.19075% to 81.51308%. Exact correct counts are given here in member-index order, each with denominator 5274.

| Seed | Arm | Four member correct counts | Mean member accuracy (%) | Worst member accuracy (%) |
|---|---|---|---:|---:|
| 8101 | C | 4289, 4293, 4297, 4294 | 81.40406 | 81.32347 |
| 8101 | R | 4294, 4294, 4289, 4290 | 81.37562 | 81.32347 |
| 8101 | P | 4290, 4282, 4289, 4289 | 81.29503 | 81.19075 |
| 8203 | C | 4292, 4297, 4290, 4294 | 81.40406 | 81.34243 |
| 8203 | R | 4294, 4296, 4293, 4292 | 81.41354 | 81.38036 |
| 8203 | P | 4294, 4294, 4299, 4295 | 81.44672 | 81.41828 |
| 8307 | C | 4292, 4295, 4292, 4293 | 81.39932 | 81.38036 |
| 8307 | R | 4296, 4291, 4292, 4292 | 81.39458 | 81.36140 |
| 8307 | P | 4297, 4295, 4289, 4294 | 81.41354 | 81.32347 |

The R−C mean-member changes are −0.02844, +0.00948, and −0.00474 pp, averaging −0.00790 pp. The worst-member changes are 0, +0.03792, and −0.01896 pp, averaging +0.00632 pp. These satisfy the frozen competence-loss tolerances and show no gross member accuracy collapse in this comparison. They do not establish that the members are as capable as ordinary independently trained predictors, because those references are absent from Stage1. The worst NLL and Brier member can differ from the member with worst accuracy; the packet correctly treats these extrema separately.

Repairs are baseline-wrong to candidate-correct pooled changes; introduced errors are baseline-correct to candidate-wrong changes. Both must be retained to interpret cohort improvements.

| Contrast | Seed | Repairs | Introduced errors | Net correct-count change |
|---|---:|---:|---:|---:|
| R−C | 8101 | 124 | 124 | 0 |
| R−C | 8203 | 110 | 112 | −2 |
| R−C | 8307 | 148 | 151 | −3 |
| R−P | 8101 | 117 | 112 | +5 |
| R−P | 8203 | 90 | 94 | −4 |
| R−P | 8307 | 140 | 142 | −2 |

R−C thus produces 382 repairs and 387 introduced errors across the three seed–node populations. These sums are bookkeeping over repeated populations, not a larger independent sample. ROUTE changes decisions, but the changes do not yield a net accuracy benefit. On COMMON-frozen pooled-error cohorts, ROUTE repairs 124/978, 110/979, and 148/978 nodes. Introduced errors are structurally absent from those cohorts because every included COMMON pooled prediction was already wrong. Reading only those repairs would omit the compensating losses on COMMON-correct nodes. The frozen all-member-wrong and common-competitor cohorts coincide in counts and reported statistics here, and overlap with pooled-error and unanimous-wrong cohorts. Their repairs cannot be added as separate evidence.

Correct-member coverage is the number of nodes with at least one correct member prediction. It falls in each R−C pair despite broadly preserved individual accuracy.

| Seed | Arm | Any member correct | All members wrong | Common wrong competitor | Unanimous wrong | Pool harm |
|---|---|---:|---:|---:|---:|---:|
| 8101 | C | 4307 | 967 | 967 | 953 | 11 |
| 8101 | R | 4303 | 971 | 971 | 963 | 7 |
| 8101 | P | 4301 | 973 | 973 | 970 | 10 |
| 8203 | C | 4306 | 968 | 968 | 960 | 11 |
| 8203 | R | 4303 | 971 | 971 | 961 | 10 |
| 8203 | P | 4311 | 963 | 963 | 956 | 14 |
| 8307 | C | 4306 | 968 | 968 | 960 | 10 |
| 8307 | R | 4302 | 972 | 972 | 967 | 9 |
| 8307 | P | 4311 | 963 | 962 | 951 | 16 |

Pool harm denotes a pooled error despite at least one correct member. In every cell, pool rescue—pooled correctness when all members are wrong—is zero, and pooling never harms a node where all members are correct. The identity `pooled correct = any-member correct − pool harm + pool rescue` holds throughout. ROUTE lowers pool harm against COMMON by four, one, and one nodes, but loses four, three, and four nodes of correct-member coverage. This exactly explains the pooled changes 0, −2, and −3. The number of nodes with mixed member correctness decreases from 32, 27, and 28 in COMMON to 27, 19, and 17 in ROUTE, while all-member-correct counts increase by one, five, and seven. The supplied hard-prediction evidence therefore does not show an increase in useful complementary correctness. The mean pooled-minus-mean-member accuracy benefit is only 0.02370 pp for ROUTE, compared with 0.04740 pp for COMMON.

The common wrong-competitor definition is demanding: the same nontruth class must strictly outrank truth in every member's logits, with ties excluded. Rank acquisition tracks reversal of the smallest qualifying original rival by at least one candidate member. A reversal of that particular rival need not make truth top ranked or remove every other common wrong rival.

| Seed | COMMON-frozen eligible nodes | At least one strict original-rival reversal | Reversal with pooled repair | Reversal but pooled still wrong | Any common rival cleared | New common-rival nodes in full population |
|---|---:|---:|---:|---:|---:|---:|
| 8101 | 967 | 244 | 116 | 128 | 120 | 124 |
| 8203 | 968 | 196 | 103 | 93 | 106 | 109 |
| 8307 | 968 | 220 | 142 | 78 | 146 | 150 |

Within each frozen original common-rival cohort, the number retaining any common wrong rival falls to 847, 862, and 822. Original-rival reversals occur on approximately 25.23%, 20.25%, and 22.73% of eligible nodes, and the mean member truth-minus-original-rival margin changes are positive at all three seeds: +1.33304, +0.08916, and +0.93813. These are meaningful descriptive observations of changed rankings. They establish neither causal gradient steering nor acquired competent diversity. Many rank acquisitions leave the pooled prediction wrong. More decisively for the complete population, new common-rival support exceeds clearing at every seed: 124−120, 109−106, and 150−146. Full-population common-rival counts therefore rise by four, three, and four. Local support reduction and full-population support reduction are different observations; only the former is supported here.

The desired-mechanism wording has no additional numerical threshold in the packet, so I have not invented a new binary mechanism test. A reading confined to the COMMON-frozen cohort has evidence of clearing and rival reversal. A claim of a net reduction in common wrong-rival support across the complete populations, or of an advantageous competence-preserving diversification, is unsupported by the reported flows and coverage. The failed scalar continuation rule settles Stage2 eligibility regardless of how that qualitative screen is interpreted.

For R−P, all cohort membership still comes from COMMON. However, the rank-acquisition field selects rivals from the comparison baseline, PERMUTED. Its 218, 145, and 223 full-population acquisitions are therefore PERMUTED-to-ROUTE changes, not additional COMMON-to-ROUTE acquisitions. Within the COMMON-frozen common-rival cohort, R−P pooled net changes are +17, −11, and −2; these are subgroup results and cannot replace the whole-population contrast. ROUTE has two fewer common-rival nodes than PERMUTED at 8101 but eight and ten more at the other seeds.

The prospective Stage1 thresholds in `FROZEN_PROTOCOL.json` and `PREREAD_CONTROL_NOTE.md` agree with the thresholds applied in `STAGE1_FIXED_GATE.json`. I rederived every check from the complete fixed seed roster.

| Frozen requirement | Observed result | Check |
|---|---|---|
| R−C accuracy strictly positive at every seed | 0, −0.03792, −0.05688 pp | Fails |
| Mean R−C accuracy gain at least 0.2 pp | −0.03160 pp | Fails |
| Mean R−P accuracy gain strictly positive | −0.00632 pp | Fails |
| Mean-member R−C change at least −0.1 pp | −0.00790 pp | Passes |
| Worst-member R−C change at least −0.2 pp | +0.00632 pp | Passes |
| Mean pooled NLL for ROUTE no worse than COMMON | −0.00739485 nats | Passes |

The reported Stage2-ineligible result follows exactly, with no threshold change or seed exclusion. `GATE.json` having `passed: true` denotes collection/closure eligibility; it does not mean the scientific Stage1 continuation rule passed. The supplied fixed gate explicitly records Stage2 eligibility and superiority/novelty establishment as false. The thresholds are exploratory engineering and quality screens, not significance tests or guarantees of generalization.

Across the packet I checked 3180 aggregate consistency identities, with no failures at numerical tolerance. These cover the nine-fit roster, equality of PER_CELL and PER_SEED, integer accuracy denominators, confusion-matrix totals and diagonals, class totals, member means and extrema, count/rate relations, every paired cohort's partitions and deltas, repairs minus harms, common-rival clearing/acquisition balances, rank-acquisition partitions, full-population agreement between paired and per-cell results, all contrast summaries and intervals, and collection/storage accounting. The gate checks were separately rederived and matched. Absolute NLL, Brier, masks, rival definitions, and rank margins remain raw-array-dependent quantities; their internal arithmetic consistency cannot verify their computation from original predictions.

The cost record describes collection and readout: 36 attempted and 36 completed member forwards, nine checkpoint deserializations, 29.162 seconds inclusive process wall time, and 30.003 seconds including the external wrapper. Peak CUDA allocation is approximately 1.862 GiB and reservation 2.711 GiB; the external sampled owned-GPU peak is 3,435,134,976 bytes. The wrapper reports exit code zero, successful reap, no remaining owner, and no hard-cap exceedance. Training updates, backwards calls, and optimizer constructions during collection are zero. These figures establish a bounded reported readout, but contain no complete training-cost comparison or evidence of an efficiency advantage over a single model or independent ensemble.

Several scientific questions remain unresolved. COMMON matches aggregate target mass but does not match route concentration or entropy, so target assignment and concentration are not separately identified. PERMUTED preserves class, self/panel membership, and scored-anchor positive counts, but does not preserve public graph degree. X is itself one context. These controls do not isolate a topology-specific causal explanation, and the negative complete R−P accuracy result supplies no positive performance evidence for graph semantics. Target archives are referenced only by hash; actual target-TV qualification, gradients, representation geometry, and source/runtime equivalence cannot be independently inspected from this packet. Nontrivial target differences alone do not guarantee useful gradient effects, including the disclosed stationary case of parallel normalized same-class embeddings.

The 12 conditional ordinary and objective-matched single/independent reference fits are not supplied and are not admitted by this failed Stage1 screen. Thus the evidence cannot establish an advantage over a competent single predictor, four independently selected predictors, or a capable diversity/efficient-ensemble control. Unused split/task confirmation is also absent and remains required by the frozen protocol before a general superiority claim. Reusing the selection population makes this an exploratory development result even though the readout and gate were prospectively fixed.

The defensible scientific statement is that this specified route treatment changes some original wrong-rival rankings and corrects a substantial set of original errors while maintaining approximately the same individual member accuracy. On the complete repeated development populations, those local changes are offset by introduced errors, lower correct-member coverage, and new common wrong rivals. This is useful evidence about the behavior and limits of the tested hypothesis. A demonstrated predictive, mechanistic, graph-specific, efficiency, or generalization advantage does not follow from it.
