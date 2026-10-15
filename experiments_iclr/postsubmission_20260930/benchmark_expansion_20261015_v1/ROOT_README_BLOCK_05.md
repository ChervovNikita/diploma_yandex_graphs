## Shared-backbone ensemble research — 15 October 2026

The aim is better predictions from several learned routes through one shared
native GNN. Every compatible setting compares the native single model, original
GNNM and four genuinely independent native models. Separate reference-setting
runs check the source papers. Native operators, features, LR and WD are fixed;
source ambiguities and longer training controllers are disclosed.

### Complete molecular pilot

All **24 fits** of the MolBACE/GIN pilot finished: three paired seeds, four
independent native models and four shared methods per seed. Each fit used
complete TRAIN and VALID, its own first-best AUC checkpoint, and the frozen
minimum100/cap1000/patience50 controller. Actual training lasted149–210 epochs
per fit:3,680 whole epochs and139,840 optimizer updates in total. Elapsed pilot
time was7.81 hours while another existing job shared the dedicated GPU.

| Method | Mean VALID AUC ± SD (%) | Mean BCE |
| --- | ---: | ---: |
| Native single | 73.32 ± 1.17 | 0.779 |
| Independent ensemble | 74.85 ± 0.50 | 0.890 |
| Base GNNM | 67.78 ± 4.39 | 0.719 |
| Alignment + variance | 71.17 ± 2.66 | 0.551 |
| Private input correction | 71.76 ± 3.91 | 0.941 |
| Both ingredients | 69.28 ± 1.80 | 0.882 |

Private nonlinear input corrections improved base GNNM in all three seeds,
by3.98 AUC points on average, but remained3.09 points below the independent
ensemble on average. Their mean BCE also worsened. Alignment with a variance
floor improved AUC in two seeds and reduced mean BCE; the combination did not
improve on either ingredient's mean AUC. The failed interaction is retained.

These are selected validation measurements on one scaffold split. Three seeds
do not establish a confirmed advantage or generality. The report includes every
seed, paired differences, uncertainty assumptions, training curves and costs.
Test scores remain closed. A frozen three-fit follow-up now tests whether
separate running statistics for each route improve native batch normalization;
it reuses the completed controls and has no complete result yet.

### Proposed broad benchmark

The proposed roster contains **6,460 physical fits across7 datasets,
8 architecture families and27 fixed source presets**:270 source references,
1,080 independent native bodies and5,110 shared-method fits. The matched single
reuses independent body0. Exact method aliases are counted once. The25 method
definitions include controls and known prior-work ingredients.

The design tests task-specific embedding supervision, persistent structural
responsibilities, nonlinear private corrections, common versus route-specific
context targets, learned aggregation, and native BN state ownership. Four
selected combinations measure interactions. Node, molecule and link tasks all
retain the required native/GNNM/independent-ensemble comparisons.

**The giant campaign is still pending preparation and user GO.** Missing native
ports, full-runtime serving/recovery checks, provider/resource measurements and
prospective confirmation decisions remain explicit. Only the dedicated
one-GPU allocation is currently authorized. No77 deployment is claimed.

- [Complete pilot results and limitations](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/pilot_20261015_molbace_v1/complete_analysis_v1/REPORT.md)
- [Expanded design, exact counts and comparisons](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/expanded_study_design_v1/README.md)
- [Exact native settings and replication limits](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/expanded_study_design_v1/NATIVE_SETTINGS.json)
- [Current implementation and outstanding work](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/README.md)
- [Method catalogue and retained decisions](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/catalogue/CATALOGUE.md)
- [Scoped literature conclusions](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/literature/SCOPED_CONCLUSIONS.md)
- [Proposed fresh confirmation study](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/confirmation_protocol_proposal_v1/README.md)

The dated sections below preserve earlier states and proposals.

