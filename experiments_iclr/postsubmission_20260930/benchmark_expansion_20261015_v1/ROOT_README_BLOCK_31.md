## Shared-backbone ensemble research — 15 October 2026

Our goal is a trainable ensemble around one native GNN, with small parameter
overhead and better prediction quality than both a single model and four
independently trained native models. Native source settings remain the reference.
**The current molecular studies do not establish a reliable advantage over the
independent ensemble.** Earlier setting-specific findings remain in the history.

### Latest complete result: useful specialization has a competence cost

The completed MolBACE/GIN comparison contains **45 unique fits and 39 prediction
banks**, with three paired optimizer seeds on the full official TRAIN/VALID
roles. TEST remains closed. These are development results on one repeatedly
examined scaffold split: 1,210 training and 151 validation molecules.

| Method | VALID AUC mean ± sample SD (%) | Mean BCE |
| --- | ---: | ---: |
| Native single | 73.32 ± 1.17 | 0.779 |
| Independent four-model ensemble | 74.85 ± 0.50 | 0.890 |
| Original boundary GNNM | 67.78 ± 4.39 | 0.719 |
| GNNM + private BN state | 73.61 ± 1.06 | 1.086 |
| GNNM + private BN + residual corrections | 72.19 ± 1.41 | 0.816 |
| Residual-only routes + private BN | 74.52 ± 2.66 | 1.952 |
| Residual routes + graph-size responsibilities | 74.47 ± 0.81 | 1.330 |
| Residual routes + shuffled responsibilities | 72.11 ± 2.90 | 1.673 |

Higher AUC means better ranking; lower BCE means better probability loss.
All other completed combinations, individual seeds and failures are retained
in the linked complete reports.

The residual-only method replaces the boundary BatchEnsemble factors with
four rank-eight additive corrections after the native input encoder. Their
initial output is zero, preserving the native initial function. GNN weights
and BN affine parameters are shared; BN running statistics are route-specific.
The corrections add **19,200 learned parameters, or 1.0183%**.

The new specialization experiment assigns private learning responsibilities
using TRAIN graph-size quartiles within each class. Shared parameters receive
the ordinary loss. A within-class shuffle provides a matched control for the
association with graph structure. Architecture, optimizer, serving and selection
are fixed across these three residual arms.

**The declared primary criterion failed:** structure improved mean AUC over the
shuffle (+2.36 points), but not over uniform responsibilities (−0.05 points).
The signed structured-minus-uniform changes were −3.96, +1.14 and +2.67 points.
No statistical equivalence, generalization or novelty claim follows.

The analysis nevertheless identifies a specific tradeoff. Averaging improved
from −0.04 to **+0.45 AUC points** relative to mean route quality, and common
member ranking errors fell from 664.7 to 519.3. Mean route AUC fell from 74.56%
to 74.01%, cancelling the pooling benefit. Across all aligned pairs, mean strict
repairs (326.0) were offset by harms (327.3). The next research question is how
to preserve competent routes while creating useful complementary errors.

Each new fit completed 149 epochs and stopped by the unchanged validation
patience rule; none hit its cap. Validation stopping does not prove optimization
stationarity. The six-fit training owner took 3.38 hours. Complete individual
training, collection and analysis costs remain available.

- [Complete interpretation, signed effects and limits](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/analysis/structural_private_risk_complete_31/INTERPRETATION.md)
- [All 39 banks and per-seed comparisons](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/analysis/structural_private_risk_complete_31/REPORT.md)
- [Frozen specialization protocol](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/pilot_20261015_structural_private_risk_v1/PROTOCOL.md)
- [Convergence audit of the preceding 36 fits](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/analysis/completed_molbace_convergence_v1/FINAL_INTERPRETATION.md)
- [Probability-loss diagnosis](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/analysis/completed_probability_failure_v1/INTERPRETATION.md)
- [Fixed probability pooling did not repair the residual bank](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/research/residual_positive_bce_pooling_v1/complete_results_v1/INTERPRETATION.md)

### Next scientific comparisons

**PubMed / native PolyFormer-Cheb:** the frozen 24-fit study compares native
M1/I4, original GNNM and uniform/structured/shuffled residuals. It changes the
prediction task and backbone and has no BN. All three CPU and all seven CUDA
lifecycle/cost checks passed. The shared residual adds 8,192 parameters (1.38%).
Minimum 2,000 / cap 10,000 epochs and patience 100 are fixed; full validation
runs every epoch. Conditional TRAIN+VALID timing projections are roughly
10–12 hours at the minimum, excluding checkpoint I/O and setup. Scientific
launch remains separate from these checks. PubMed is previously encountered
development data; its minority VALID class has only 72 nodes.

Two other prepared hypotheses remain distinct: task-ranking supervision with
matched independent-ensemble objective controls, and learned private weight
noise against fixed noise. The latter passed real CPU update/restore/noise
replay checks; it has no accuracy result. A literature review is also examining
intermediate task supervision to improve member competence. Existing ideas are
credited to prior work; combining them does not itself establish novelty.

- [Frozen PubMed comparison and diagnostics](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/research/pubmed_cheb_transfer_driver_v1/README.md)
- [Measured PubMed costs and current admission recommendation](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/analysis/pubmed_cheb_fit_admission_candidate_v2/REPORT.md)
- [Pairwise competence proposal and controls](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/research/molbace_pairwise_study_v1/README.md)
- [Learned private-noise proposal and prior work](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/research/stochastic_private_adapter_audit_v1/NOTE.md)

### Broad benchmark: prepared roster, runtime completion pending

The proposed campaign has **6,460 physical fits across 7 datasets, 8 architecture
families and 27 fixed source presets**. It includes 270 separate paper-source
references, 1,080 independent native fits and 5,110 shared fits. M1 reuses I4
body 0. Another 270 ensemble-export tasks produce 6,730 total tasks.

Datasets are CiteSeer, PubMed, Coauthor-CS, Coauthor-Physics, MolBACE, MolHIV and
ogbl-collab. Architectures are PolyFormer, GCN, GCN with a virtual node, GIN, GIN
with a virtual node, NCN, NCNC and BUDDY. Task-specific methods have explicit
controls; method support is not inferred from a successful smoke test.

Normalization migration, reporting and dispatch metadata are generated and
cross-checked. Selected NCNC replay now passes with exact saved predictions and zero training.
Composed BUDDY execution is running. Reporting witnesses, custody review and
measured placement/budget remain outstanding. Two jobs per GPU is a proposed maximum; actual
concurrency requires measured admission. Only the dedicated allocation is
currently enabled. **The full campaign requires the user's review and GO.**

- [Plain-language campaign design](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/BENCHMARK_OVERVIEW.md)
- [Method coverage and controls](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/benchmark_review_map_v1/README.md)
- [Native source presets and hyperparameters](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/expanded_study_design_v1/NATIVE_SETTINGS.json)
- [Remaining runtime checks](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/READY_NEXT_RUNTIME_CHECKS_30.md)

The dated sections below preserve earlier project states.
