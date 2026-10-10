# Complete bootstrap acquisition and graph serving comparison

10 October 2026. All 36 newly exported banks and 900 fixed small CPU endpoints closed at 16:24:26 UTC with no endpoint failure. The complete comparison reuses 30 immutable original-reference OOF banks. No new GNN training or reference refit occurred. Close the frozen superiority recipe: every block fails its full accuracy-transfer and pooled-NLL-protection rules, and all whole-roster flags are false.

A protected local pooled win remains visible. Paired correction plus bootstrap and graph serving improves over the equally graph-processed unweighted paired bank by **+0.158008 pp on GAT** and **+0.233852 pp on SAGE**, with all three seed accuracy deltas positive and NLL protection passing. Original bootstrap member-protection flags remain false. These separate facts coexist; the member failure does not erase the pooled gain. Both candidates nevertheless lose to each equally graph-processed genuine I4 reference at every seed.

## Four paired cells

C is the unweighted native pool; A is its graph-scored pool; B is the corresponding bootstrap native pool; AB is its graph-scored pool. Mean accuracy is percent over seeds 7301, 7403, 7507. All three declared blocks are retained.

| Backbone | Block | C | A | B | AB | AB NLL | Accuracy gate | NLL gate |
|---|---|---:|---:|---:|---:|---:|---|---|
| GAT | coherent | 81.171786 | 81.171786 | 81.190747 | 81.253950 | 0.693120 | F | F |
| GAT | paired correction | 81.171786 | 81.184427 | 81.291872 | 81.342435 | 0.699715 | F | F |
| GAT | rank one LoRA | 81.190747 | 81.178106 | 81.234989 | 81.310833 | 0.695131 | F | F |
| SAGE | coherent | 79.364176 | 79.383137 | 79.383137 | 79.477942 | 0.743146 | F | F |
| SAGE | paired correction | 79.376817 | 79.357856 | 79.402098 | 79.591708 | 0.736828 | F | F |
| SAGE | rank one LoRA | 79.389458 | 79.402098 | 79.389458 | 79.515864 | 0.740454 | F | F |

Equally graph-processed references receive the identical frozen fitting opportunity. Their existing OOF outputs were reused unchanged.

| Backbone | Reference | Graph accuracy % | Graph NLL |
|---|---|---:|---:|
| GAT | ordinary genuine I4 | 82.088232 | 0.631041 |
| GAT | factorized genuine I4 | 81.942864 | 0.642227 |
| SAGE | ordinary genuine I4 | 81.064341 | 0.671709 |
| SAGE | factorized genuine I4 | 80.887372 | 0.681870 |

## Paired cell differences

Accuracy differences are percentage points. Seed vectors always follow 7301, 7403, 7507. Positive interaction is descriptive and does not substitute for final quality or the fixed reference gates.

| Backbone | Block | Contrast | Mean accuracy delta | Three seed deltas | Mean NLL delta |
|---|---|---|---:|---|---:|
| GAT | coherent | A − C | +0.000000 | +0.000000, +0.018961, -0.018961 | -0.001141 |
| GAT | coherent | B − C | +0.018961 | +0.037922, -0.075844, +0.094805 | -0.000970 |
| GAT | coherent | AB − C | +0.082164 | +0.189609, -0.056883, +0.113766 | -0.013014 |
| GAT | coherent | AB − A | +0.082164 | +0.189609, -0.075844, +0.132727 | -0.011873 |
| GAT | coherent | AB − B | +0.063203 | +0.151688, +0.018961, +0.018961 | -0.012044 |
| GAT | coherent | AB − A − B + C | +0.063203 | +0.151688, +0.000000, +0.037922 | — |
| GAT | paired correction | A − C | +0.012641 | +0.018961, -0.018961, +0.037922 | -0.003958 |
| GAT | paired correction | B − C | +0.120086 | +0.132727, +0.056883, +0.170648 | +0.002198 |
| GAT | paired correction | AB − C | +0.170648 | +0.322336, +0.000000, +0.189609 | -0.014187 |
| GAT | paired correction | AB − A | +0.158008 | +0.303375, +0.018961, +0.151688 | -0.010228 |
| GAT | paired correction | AB − B | +0.050563 | +0.189609, -0.056883, +0.018961 | -0.016385 |
| GAT | paired correction | AB − A − B + C | +0.037922 | +0.170648, -0.037922, -0.018961 | — |
| GAT | rank one LoRA | A − C | -0.012641 | -0.018961, +0.000000, -0.018961 | -0.001733 |
| GAT | rank one LoRA | B − C | +0.044242 | +0.094805, +0.000000, +0.037922 | -0.001787 |
| GAT | rank one LoRA | AB − C | +0.120086 | +0.341297, -0.056883, +0.075844 | -0.014551 |
| GAT | rank one LoRA | AB − A | +0.132727 | +0.360258, -0.056883, +0.094805 | -0.012818 |
| GAT | rank one LoRA | AB − B | +0.075844 | +0.246492, -0.056883, +0.037922 | -0.012764 |
| GAT | rank one LoRA | AB − A − B + C | +0.088484 | +0.265453, -0.056883, +0.056883 | — |
| SAGE | coherent | A − C | +0.018961 | +0.037922, +0.000000, +0.018961 | -0.000922 |
| SAGE | coherent | B − C | +0.018961 | -0.018961, +0.113766, -0.037922 | -0.001446 |
| SAGE | coherent | AB − C | +0.113766 | +0.227531, +0.094805, +0.018961 | -0.011527 |
| SAGE | coherent | AB − A | +0.094805 | +0.189609, +0.094805, +0.000000 | -0.010605 |
| SAGE | coherent | AB − B | +0.094805 | +0.246492, -0.018961, +0.056883 | -0.010081 |
| SAGE | coherent | AB − A − B + C | +0.075844 | +0.208570, -0.018961, +0.037922 | — |
| SAGE | paired correction | A − C | -0.018961 | +0.037922, -0.037922, -0.056883 | -0.003656 |
| SAGE | paired correction | B − C | +0.025281 | -0.018961, +0.075844, +0.018961 | -0.002567 |
| SAGE | paired correction | AB − C | +0.214891 | +0.284414, +0.208570, +0.151688 | -0.017651 |
| SAGE | paired correction | AB − A | +0.233852 | +0.246492, +0.246492, +0.208570 | -0.013995 |
| SAGE | paired correction | AB − B | +0.189609 | +0.303375, +0.132727, +0.132727 | -0.015084 |
| SAGE | paired correction | AB − A − B + C | +0.208570 | +0.265453, +0.170648, +0.189609 | — |
| SAGE | rank one LoRA | A − C | +0.012641 | +0.000000, +0.037922, +0.000000 | -0.001567 |
| SAGE | rank one LoRA | B − C | -0.000000 | -0.018961, +0.056883, -0.037922 | -0.001948 |
| SAGE | rank one LoRA | AB − C | +0.126406 | +0.246492, +0.170648, -0.037922 | -0.014127 |
| SAGE | rank one LoRA | AB − A | +0.113766 | +0.246492, +0.132727, -0.037922 | -0.012560 |
| SAGE | rank one LoRA | AB − B | +0.126406 | +0.265453, +0.113766, +0.000000 | -0.012178 |
| SAGE | rank one LoRA | AB − A − B + C | +0.113766 | +0.265453, +0.075844, +0.000000 | — |

Paired AB minus LoRA AB is +0.031602 pp on GAT (0, +0.018961, +0.075844) and +0.075844 pp on SAGE (+0.018961, +0.037922, +0.170648). The mean NLL differences are +0.004584 and −0.003625. This does not establish a general paired-capacity advantage or rescue the failed I4 comparisons.

## Every fixed accuracy and NLL flag

All controls are available at all three seeds. **P** means pass and **F** fail. Accuracy flags are **M/N/2**: the mean threshold (+0.2 pp versus B native, +0.1 otherwise), nonnegative at all three seeds, and at least two positive seeds. NLL flags are **mean/seed**: deterioration at most 0.02 mean and 0.05 at every seed. The NLL column gives mean and worst-seed deterioration. No NLL protection flag was specified against B native.

### GAT

| Block | AB reference | Mean accuracy delta | Three seed deltas | M/N/2 | NLL mean / worst | NLL mean/seed |
|---|---|---:|---|---|---|---|
| coherent | B native | +0.063203 | +0.151688, +0.018961, +0.018961 | F/P/P | -0.012044 / -0.010541 | — |
| coherent | B global temperature | +0.063203 | +0.151688, +0.018961, +0.018961 | F/P/P | +0.047691 / +0.054979 | F/F |
| coherent | B member temperature | +0.063203 | +0.151688, +0.018961, +0.018961 | F/P/P | +0.047688 / +0.054969 | F/F |
| coherent | B identical self scorer | +0.037922 | +0.113766, -0.018961, +0.018961 | F/F/P | -0.000645 / +0.000004 | P/P |
| coherent | B stacking | -0.056883 | +0.018961, -0.170648, -0.018961 | F/F/F | -0.006026 / -0.000880 | P/P |
| coherent | A graph | +0.082164 | +0.189609, -0.075844, +0.132727 | F/F/P | -0.011873 / -0.006166 | P/P |
| coherent | ordinary I4 graph | -0.834281 | -0.587789, -1.535836, -0.379219 | F/F/F | +0.062079 / +0.074435 | F/F |
| coherent | factorized I4 graph | -0.688914 | -0.796359, -0.910125, -0.360258 | F/F/F | +0.050893 / +0.063962 | F/F |
| paired correction | B native | +0.050563 | +0.189609, -0.056883, +0.018961 | F/F/P | -0.016385 / -0.013271 | — |
| paired correction | B global temperature | +0.056883 | +0.208570, -0.037922, +0.000000 | F/F/F | +0.054505 / +0.077528 | F/F |
| paired correction | B member temperature | +0.063203 | +0.208570, -0.037922, +0.018961 | F/F/P | +0.054517 / +0.077533 | F/F |
| paired correction | B identical self scorer | +0.183289 | +0.360258, +0.056883, +0.132727 | P/P/P | -0.002282 / -0.000910 | P/P |
| paired correction | B stacking | +0.031602 | +0.132727, -0.151688, +0.113766 | F/F/P | +0.001851 / +0.024512 | P/P |
| paired correction | A graph | +0.158008 | +0.303375, +0.018961, +0.151688 | P/P/P | -0.010228 / +0.016219 | P/P |
| paired correction | ordinary I4 graph | -0.745797 | -0.417141, -1.554797, -0.265453 | F/F/F | +0.068673 / +0.104690 | F/F |
| paired correction | factorized I4 graph | -0.600430 | -0.625711, -0.929086, -0.246492 | F/F/F | +0.057487 / +0.094217 | F/F |
| rank one LoRA | B native | +0.075844 | +0.246492, -0.056883, +0.037922 | F/F/P | -0.012764 / -0.011595 | — |
| rank one LoRA | B global temperature | +0.082164 | +0.246492, -0.037922, +0.037922 | F/F/P | +0.049260 / +0.067629 | F/F |
| rank one LoRA | B member temperature | +0.063203 | +0.208570, -0.037922, +0.018961 | F/F/P | +0.049263 / +0.067628 | F/F |
| rank one LoRA | B identical self scorer | +0.132727 | +0.379219, -0.018961, +0.037922 | P/F/P | -0.000787 / -0.000193 | P/P |
| rank one LoRA | B stacking | +0.018961 | +0.132727, -0.075844, +0.000000 | F/F/F | -0.005140 / +0.007827 | P/P |
| rank one LoRA | A graph | +0.132727 | +0.360258, -0.056883, +0.094805 | P/F/P | -0.012818 / +0.000663 | P/P |
| rank one LoRA | ordinary I4 graph | -0.777399 | -0.417141, -1.573758, -0.341297 | F/F/F | +0.064090 / +0.076431 | F/F |
| rank one LoRA | factorized I4 graph | -0.632031 | -0.625711, -0.948047, -0.322336 | F/F/F | +0.052904 / +0.068770 | F/F |

### SAGE

| Block | AB reference | Mean accuracy delta | Three seed deltas | M/N/2 | NLL mean / worst | NLL mean/seed |
|---|---|---:|---|---|---|---|
| coherent | B native | +0.094805 | +0.246492, -0.018961, +0.056883 | F/F/P | -0.010081 / -0.006432 | — |
| coherent | B global temperature | +0.094805 | +0.246492, -0.018961, +0.056883 | F/F/P | +0.047479 / +0.058213 | F/F |
| coherent | B member temperature | +0.101125 | +0.265453, -0.018961, +0.056883 | P/F/P | +0.047489 / +0.058217 | F/F |
| coherent | B identical self scorer | +0.132727 | +0.208570, +0.000000, +0.189609 | P/P/P | -0.002334 / -0.001776 | P/P |
| coherent | B stacking | -0.075844 | +0.132727, -0.341297, -0.018961 | F/F/F | -0.003504 / +0.004715 | P/P |
| coherent | A graph | +0.094805 | +0.189609, +0.094805, +0.000000 | F/P/P | -0.010605 / -0.006611 | P/P |
| coherent | ordinary I4 graph | -1.586399 | -1.251422, -1.858172, -1.649602 | F/F/F | +0.071438 / +0.072517 | F/F |
| coherent | factorized I4 graph | -1.409430 | -1.175578, -1.990899, -1.061813 | F/F/F | +0.061276 / +0.069450 | F/F |
| paired correction | B native | +0.189609 | +0.303375, +0.132727, +0.132727 | F/P/P | -0.015084 / -0.011134 | — |
| paired correction | B global temperature | +0.189609 | +0.303375, +0.132727, +0.132727 | P/P/P | +0.041427 / +0.053708 | F/F |
| paired correction | B member temperature | +0.189609 | +0.303375, +0.132727, +0.132727 | P/P/P | +0.041446 / +0.053728 | F/F |
| paired correction | B identical self scorer | +0.297055 | +0.379219, +0.208570, +0.303375 | P/P/P | -0.004517 / -0.003257 | P/P |
| paired correction | B stacking | +0.037922 | +0.170648, -0.113766, +0.056883 | F/F/P | -0.007886 / +0.001154 | P/P |
| paired correction | A graph | +0.233852 | +0.246492, +0.246492, +0.208570 | P/P/P | -0.013995 / -0.009971 | P/P |
| paired correction | ordinary I4 graph | -1.472633 | -1.194539, -1.687524, -1.535836 | F/F/F | +0.065120 / +0.065857 | F/F |
| paired correction | factorized I4 graph | -1.295664 | -1.118695, -1.820250, -0.948047 | F/F/F | +0.054958 / +0.064062 | F/F |
| rank one LoRA | B native | +0.126406 | +0.265453, +0.113766, +0.000000 | F/P/P | -0.012178 / -0.008535 | — |
| rank one LoRA | B global temperature | +0.126406 | +0.265453, +0.113766, +0.000000 | P/P/P | +0.044961 / +0.056165 | F/F |
| rank one LoRA | B member temperature | +0.120086 | +0.265453, +0.113766, -0.018961 | P/F/P | +0.044977 / +0.056176 | F/F |
| rank one LoRA | B identical self scorer | +0.202250 | +0.322336, +0.151688, +0.132727 | P/P/P | -0.003121 / -0.002380 | P/P |
| rank one LoRA | B stacking | -0.018961 | +0.208570, -0.151688, -0.113766 | F/F/F | -0.005252 / +0.003505 | P/P |
| rank one LoRA | A graph | +0.113766 | +0.246492, +0.132727, -0.037922 | P/F/P | -0.012560 / -0.008155 | P/P |
| rank one LoRA | ordinary I4 graph | -1.548477 | -1.213500, -1.725446, -1.706485 | F/F/F | +0.068745 / +0.069282 | F/F |
| rank one LoRA | factorized I4 graph | -1.371508 | -1.137656, -1.858172, -1.118695 | F/F/F | +0.058583 / +0.067097 | F/F |

Temperature and I4 NLL protection fail in every block. Protection against the identical self scorer, stacking and own unweighted graph pool passes in every block. Accuracy cannot be summarized as every atomic flag failing: paired GAT/SAGE pass the self-scorer and own-graph accuracy gates, and several SAGE temperature/self gates pass. The conjunction required by the frozen decision fails everywhere.

For every block, the unchanged original bootstrap criteria are: mean-member accuracy nonnegative **F**; every-seed member decline at most 0.1 pp **F**; own-counterpart mean NLL deterioration at most 0.02 **P**; each-seed deterioration at most 0.05 **P**. Thus original member/protection conjunctions stay **F** without forbidding a separate pooled win.

## Exact repair survival and acquired alternatives

Counts sum three readouts of the same 5274 nodes. They are not independent node or graph samples. Repairs/harms in the first table use C as the common baseline; survival counts retain the corresponding ingredient repair under AB. A lost ingredient repair is a harm versus that ingredient, not necessarily a harm versus C.

| Backbone | Block | A repairs / harms | B repairs / harms | AB repairs / harms | A repairs survive | B repairs survive | New AB repairs | New AB harms where A and B correct |
|---|---|---|---|---|---:|---:|---:|---:|
| GAT | coherent | 1 / 1 | 126 / 123 | 130 / 117 | 1 / 1 | 110 / 126 | 19 | 14 |
| GAT | paired correction | 8 / 6 | 167 / 148 | 189 / 162 | 4 / 8 | 140 / 167 | 46 | 34 |
| GAT | rank one LoRA | 0 / 2 | 100 / 93 | 111 / 92 | 0 / 0 | 85 / 100 | 26 | 16 |
| SAGE | coherent | 3 / 0 | 11 / 8 | 50 / 32 | 3 / 3 | 9 / 11 | 39 | 27 |
| SAGE | paired correction | 11 / 14 | 18 / 14 | 83 / 49 | 11 / 11 | 13 / 18 | 61 | 34 |
| SAGE | rank one LoRA | 6 / 4 | 15 / 15 | 60 / 40 | 6 / 6 | 14 / 15 | 46 | 34 |

The second table isolates acquisition and serving. Newly covered nodes were absent from C member coverage. Rescue counts refer specifically to those nodes that B acquired but its native pool lost; newly served alternatives can also be lost again under AB.

| Backbone | Block | New / removed coverage | New coverage lost by B | Rescued by AB | New B-served alternatives lost by AB | New coverage served / still lost by AB | Other B pooling losses rescued | All B-served alternatives lost |
|---|---|---|---:|---:|---:|---|---:|---:|
| GAT | coherent | 247 / 52 | 136 | 18 | 13 | 116 / 131 | 23 | 31 |
| GAT | paired correction | 314 / 71 | 183 | 43 | 20 | 154 / 160 | 28 | 63 |
| GAT | rank one LoRA | 219 / 27 | 143 | 23 | 12 | 87 / 132 | 20 | 31 |
| SAGE | coherent | 108 / 0 | 106 | 33 | 0 | 35 / 73 | 11 | 29 |
| SAGE | paired correction | 127 / 3 | 126 | 49 | 0 | 50 / 77 | 29 | 48 |
| SAGE | rank one LoRA | 123 / 0 | 119 | 36 | 0 | 40 / 83 | 21 | 37 |

| Backbone | Block | AB repairs / harms by seed | Acquired-loss rescues by seed | Newly served alternatives lost by seed |
|---|---|---|---|---|
| GAT | coherent | 79/69; 10/13; 41/35 | 13, 3, 2 | 11, 2, 0 |
| GAT | paired correction | 67/50; 98/98; 24/14 | 24, 8, 11 | 4, 13, 3 |
| GAT | rank one LoRA | 54/36; 48/51; 9/5 | 20, 1, 2 | 7, 4, 1 |
| SAGE | coherent | 22/10; 8/3; 20/19 | 16, 0, 17 | 0, 0, 0 |
| SAGE | paired correction | 35/20; 20/9; 28/20 | 24, 8, 17 | 0, 0, 0 |
| SAGE | rank one LoRA | 25/12; 14/5; 21/23 | 16, 6, 14 | 0, 0, 0 |

Graph serving corrects **zero** remaining bootstrap common strict wrong rivals across all six blocks. Acquiring a correct member alternative can remove an earlier C strict-rival obstruction, after which AB may serve that alternative. This is distinct from reversing a common strict rival within the fixed B bank. AB has one aggregation-only correct readout in GAT LoRA and zero in the other five blocks; all-member-wrong alone is therefore not the mathematical obstruction.

SAGE paired provides the clearest local mechanism evidence: AB preserves 11/11 A repairs and 13/18 B repairs, adds 61 new repairs versus C, and introduces 34 new harms where both ingredients were correct. It rescues 49/126 previously lost acquired alternatives, but still loses 77/127 newly covered nodes and 48 alternatives that B had served. Its +0.189609 pp gain over B narrowly misses the predeclared +0.2 pp threshold; no rounding or interaction estimate changes that result.

## Costs provenance and scope

Owner cost was **1425.651 s**, including **1405.673 s** of new small-head fitting and **15.416 s** of export. The fixed work was **900 fits / 450000 updates**, **36 selected forwards / 144 native member trajectories**, **zero new GNN acquisitions** and **zero reference refits**. The original complete **855 fits / 427500 updates / 1181.521 s** of head work remain preserved, together with all original export records and the earlier failed-prefix costs. All per-rule, per-fit, per-bank and acquisition records remain in the bound full server report. Concurrent observed times are not isolated speed or inference-saving evidence.

The first complete-reader attempt stopped at a Path API mismatch before numerical imports or scientific metric interpretation. The successor readout completed server-side; a later SSH fetch timed out and the already completed report was recovered without a repeat readout, head fit, export or training run. These engineering events do not change the scientific result. Root retains their source/status records.

All checkpoints were selected using the full VALID role before the fixed aggregator folds. These are **encountered-development** results, not whole-pipeline cross-fitting or unused confirmation. Three seeds on one connected graph describe optimization repeats; descriptive intervals do not account for method search or independent graph variation. Original paper scores remain unchanged and TEST stays closed.

Retain the conditional serving recovery and close this exact superiority recipe without a head, fold, weight-strength or correction rescue grid. Neither negative ingredient means nor failed member criteria constitute a theorem against all shared ensembles. A broader positive claim still requires capable matched references, a frozen whole pipeline and unused confirmation.

## Complete evidence

`COMPLETE_ANALYSIS_SUMMARY.json` binds all six block partitions and ten reference partitions by byte count and SHA256. Every partition identity was checked before this interpretation. Full class/member/seed contrasts, NLL seed vectors, repair cohorts and original records remain in those files and the sealed server report.

Complete study SHA256: `1ce653ef6e2f295a79031bb6ac721b089d49a4c1cbfff33aa9b793190ea9f163`. Complete readout SHA256: `592eb19a2dd65f14041e398c61f575183e8df0fed2a9afe78cfb6cfce458b519`. Prospective decision SHA256: `f98fbffe734c3b2583f41620210bae5f11bbd8884c74fa50b97e6e1e109649b7`.
