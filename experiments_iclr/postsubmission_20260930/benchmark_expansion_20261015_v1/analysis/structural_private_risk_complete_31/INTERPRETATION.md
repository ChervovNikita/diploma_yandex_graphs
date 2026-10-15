# Complete structural specialization result

15 October 2026. All six new fits and six selected-state collections completed; the unchanged analyzer combines 39 prediction banks from 45 unique fits. No reconstruction warning or TEST access occurred. Original controls were reused.

## Outcome

**The frozen primary accuracy criterion was not met.** Structured responsibilities had to improve mean AUC over both uniform responsibilities and the within-class shuffled control. Structured minus uniform was −0.0488 percentage points; structured minus shuffled was +2.3565 points. All outcomes remain in the study.

| Method | VALID AUC mean ± sample SD (%) | Uniform-graph BCE |
| --- | ---: | ---: |
| Native single | 73.321 ± 1.172 | 0.7795 |
| Independent ensemble (four bodies) | 74.847 ± 0.500 | 0.8902 |
| Original boundary GNNM | 67.778 ± 4.390 | 0.7193 |
| Uniform private residuals | 74.518 ± 2.660 | 1.9520 |
| Graph-size private responsibilities | 74.469 ± 0.810 | 1.3300 |
| Within-class shuffled responsibilities | 72.112 ± 2.901 | 1.6733 |

The signed structured-minus-uniform seed differences are **−3.956, +1.136, +2.674 points**. Against shuffled assignments they are **−1.832, +4.908, +3.993**; against I4 they are **−0.842, −0.623, +0.330**. Positive means against the shuffled control do not establish a robust advantage across seeds. The independent ensemble remains higher in mean AUC and lower in mean BCE.

## Mechanism observation

Specialization made route errors less redundant and averaging more useful:

| Diagnostic (mean over three seeds) | Uniform | Structured |
| --- | ---: | ---: |
| Mean individual-route AUC (%) | 74.560 | 74.014 |
| Pool minus mean route AUC (points) | −0.043 | +0.455 |
| Ranking reversals shared by every route | 664.7 | 519.3 |
| Pool ranking errors despite a correct route | 31.0 | 177.7 |

The pooling benefit was positive for all three structured seeds (+0.522, +0.540, +0.302 points). Yet mean route quality fell by 0.546 points. Averaging recovered much of that loss, yielding an almost unchanged final AUC. This is a measured competence–complementarity tradeoff in this setting, not proof of its cause or universality.

Across all positive–negative pairs, structure repaired a mean 326.0 strict pool errors but introduced 327.3 harms. It repaired 312.3 of the uniform bank's common errors while introducing 232.0 new common errors; the latter is only one subset of all harms. Counting repaired old errors alone would overstate progress. The 2,730 pairs per seed share just 151 molecules and are not independent statistical repetitions.

BCE improved from 1.9520 to 1.3300 on average, with seed differences −2.8465, +0.3280, +0.6522. The mean improvement is driven by one seed; it is not a consistent probability-quality repair. I4 BCE is 0.8902.

## Training and evidence limits

Each new fit completed 149 epochs and stopped under the unchanged minimum-100/patience-50 controller. Structured selected epochs were 18/31/24; shuffled selected epochs were 10/29/39. None hit the 1,000-epoch cap. Validation stopping does not establish optimization stationarity. The six-fit owner took 12,162.21 seconds (3.38 hours); full per-fit and collection costs are retained separately in COSTS_AND_TERMINAL.json.

The full official MolBACE TRAIN/VALID roles were used, with 1,210 TRAIN and 151 VALID molecules (130 positives/21 negatives). There are three optimizer seeds on one repeatedly exposed scaffold split. This is development evidence; no independent generalization, significance, equivalence or methodological novelty claim follows. The method still adds 19,200 parameters (1.0183% of the native model).

## Next decision

Keep the already frozen PubMed/Cheb transfer protocol unchanged. It tests the complete uniform/structured/shuffled procedures on another task and native backbone without BN, with M1, I4 and original GNNM controls. Its role is to test whether the observed tradeoff transfers; this molecular result does not predict a win.

For future protocols, prioritize interventions that improve member competence while retaining useful specialization. Intermediate task supervision is a literature-supported candidate under investigation. It requires a matched final-state supervision control and native independent-ensemble objective controls before an ensemble-specific claim. Learned private noise and task-ranking supervision remain separately prepared hypotheses. Do not silently combine or tune them inside the frozen transfer study.

Source RESULT.json: c1ce9e1acae4aeffb4349177de33cbbe62d9dfb9f3b3bc3a28d1632622ebbf50
Source path: `/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/analysis_outputs/structural_private_risk_complete_v1/RESULT.json`
