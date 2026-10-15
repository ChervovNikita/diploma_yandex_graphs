## Shared-backbone ensemble research — 15 October 2026

The aim is better predictions from several learned routes through one shared
native GNN. Every compatible setting compares the native single model, original
GNNM and four genuinely independent native models. Separate reference-setting
runs check the source papers. Native operators, features, LR and WD are fixed.
Source ambiguities and longer training controllers are disclosed.

### Completed molecular pilots

The MolBACE/GIN study completed 24 fits across three paired seeds. Each seed
trained four independent native models and four shared methods. Every fit used
the complete TRAIN and VALID sets, its own first-best AUC checkpoint, and the
frozen minimum 100 / cap 1,000 / patience 50 controller. Training lasted 149–210
epochs per fit, totaling 3,680 epochs and 139,840 updates. Elapsed time was
7.81 hours while another existing job shared the dedicated GPU.

| Method | Mean VALID AUC ± SD (%) | Mean BCE |
| --- | ---: | ---: |
| Native single | 73.32 ± 1.17 | 0.779 |
| Independent ensemble | 74.85 ± 0.50 | 0.890 |
| Base GNNM | 67.78 ± 4.39 | 0.719 |
| Alignment + variance | 71.17 ± 2.66 | 0.551 |
| Private input correction | 71.76 ± 3.91 | 0.941 |
| Both ingredients | 69.28 ± 1.80 | 0.882 |
| Private BN running state, separate three-fit follow-up | 73.61 ± 1.06 | 1.086 |

Private input corrections improved base GNNM in all three seeds, but remained
below I4 on average. Combining corrections with geometry underperformed each
ingredient's mean AUC. The separate three-fit BN follow-up improved AUC over
base GNNM in all seeds, by 5.84 points on average, but remained 1.23 points
below I4 and increased BCE. It used 464 epochs and 17,632 updates. No fit in
either pilot stopped at its cap. Private BN state is a known normalization
control, not a methodological novelty claim.

Analysis of all 24 parent prediction banks found that the combined method made
fewer ranking mistakes shared by every route, yet its individual routes were
weaker. Mean member AUC was 73.41% for I4, 62.01% for base GNNM, 64.24% with
geometry, 66.07% with corrections, and 61.00% with both. The next design decisions
must consider individual prediction quality as well as diversity.

Reconstructing all three private-BN checkpoints gave mean member AUC 72.63%,
followed by a 0.99-point gain from averaging. The member-quality gap to I4 was
smaller in each seed. A separate nine-fit study started on the allocation at 09:36 UTC: geometry, private
correction, and both together under private BN, each at the same three seeds.
It reuses all 15 completed factorial controls and runs one new fit at a time.
At 13:44 UTC, eight fits were complete and the ninth was running; partial quality scores remain unread. The original 6,460-fit benchmark count is unchanged.

A completed saved-prediction check compared logit and probability averaging
for every completed method across all three seeds. Probability averaging raised
base GNNM AUC to 69.01%, while I4 reached 74.90%; it did not close the gap.
No model was trained or evaluated again. All seven-bank comparisons and losses
are retained in the [aggregation report](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/analysis/complete_fixed_aggregation_v1/REPORT.md).

These are selected validation results on one scaffold split. Three seeds do
not establish a confirmed advantage or transfer. Test scores remain closed.
Every outcome, per-seed difference, training cost and failed combination is
retained. Different selected epochs also limit causal interpretation.

### Stronger fixed-weight normalization control

At the same three selected private-BN checkpoints, own route statistics give
mean VALID AUC 73.61%. A common-mixture control gives 68.45%, compared with
59.06% for the earlier arithmetic tie. The mixture adds variation between route
means to the average within-route variance. That correction narrows the
own-minus-common gap from 14.55 to **5.16 points**, with own banks still ahead
in all three seeds. Probability losses are mixed: the common mixture improves
BCE and Brier over own banks in one seed and worsens them in two.

The stronger control required 120 CPU VALID forwards, zero training and 25.76
seconds. Earlier tied/cyclic outcomes are retained. Saved EMA moments provide
a plug-in approximation, not exact recalibration or reconstruction of original
GNNM's sequential shared BN. These results support a matched normalization
control; they establish neither novelty nor independent-ensemble superiority.

[Complete comparison and limitations](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/analysis/bn_mixture_moments_completed_v1/INTERPRETATION.md).

### What the implementation and theory checks add

The inspected BatchEnsemble author implementation normalizes concatenated
route batches; our current molecular model normalizes routes sequentially.
The two rules have different TRAIN statistics and running-state updates. With
matched batches, weights, RNG and route order, changing only the ownership of
running buffers leaves native TRAIN updates mathematically unchanged; its direct
effect is on evaluation state. Selected epochs can still differ.
This explains why batching and normalization must be stated explicitly when
interpreting the completed molecular pilots. It does not establish a new
normalization method or explain the entire measured gain.

The context-specialization analysis also identified a control limitation:
averaging two valid similarity targets can produce a target that no symmetric
cosine scorer can represent exactly. Common versus route-specific target results
must therefore be interpreted together with target fit and actual prediction
quality. The exact derivation and prior-method comparison are preserved in the
[context analysis](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/literature/context_target_novelty_followup_v1/REPORT.md).

### Proposed broad benchmark

The proposal contains **6,460 physical fits across 7 datasets, 8 architecture
families and 27 fixed source presets**. It includes 270 source references,
1,080 independent native bodies and 5,110 shared-method fits. The matched
single reuses independent body 0. The 25 exact recipe definitions include
controls and known ingredients.

Candidate families include task-specific embedding supervision, persistent
structural responsibilities, nonlinear private corrections, common versus
route-specific context targets, learned aggregation, and native BN state.
Four combinations measure interactions. Node, molecule and link tasks all
retain the required native/GNNM/independent-ensemble comparisons.

The compiler now provides **all 6,460 concrete source descriptions**, with no
remaining source/context blocks. Existing exact descriptions are reused.
The production BUDDY input provider passed its two-update/whole-validation/
restore check at a nonzero seed. NCNC completed two native CUDA epochs with
interruption/resume and selected exports (34 updates, four validations).
The molecular runtime completed the corresponding CPU lifecycle (76 updates,
four validations). These are execution checks, not accuracy measurements.
The dispatcher passed 38 subprocess checks. Compiler 4 and runtime 9 now
propagate the corrected BUDDY shared-body metadata without changing scientific
settings. The final NCN/BUDDY lifecycle checks are prepared but await a free
GPU slot; resource measurement and offline three-GPU deployment remain pending.

The portable allocation metadata and reporting registry now build successfully
for all 6,730 tasks, preserving every scientific configuration and all 6,190
expected prediction rows. No campaign training was started. Two earlier
metadata failures and their repairs are retained. All native runtime modules
are unchanged by the latest repair. The default ownership is a preparation
map, not the final three-GPU workload division.

Before the large run, the complete nine-fit study will inform a consistent
normalization policy for the extensions and their controls. Original GNNM
remains unchanged. Native evaluation-population preparation now covers all 27
presets, 99 preset/split proofs and 43 distinct populations. It accessed TEST
metadata without scoring. The original Collab TEST file was recovered from its
retained archive, and its native OGB array conversion was applied; both failed
attempts and their costs remain. Six successful providers were reused.

The final scorer now preserves original checkpoint provenance and attaches the
explicit BUDDY inference context separately. Three genuine selected molecular
states passed whole-VALID CPU reconstruction, without training or TEST scoring.
This verifies their original producer scope, not exact CUDA parity or all tasks.
The combined one-command orchestration passed eight actual process/ledger
checks, including restart, failure and interrupted-readiness recovery. These
used standard-library fixtures, with zero model work. An initial driver failure
is preserved with its repair.

Three finite CUDA checks remain queued: NCN and BUDDY lifecycles plus a MolHIV
GIN-VN resource measurement. They retain the previously fixed jobs and total
4,233 optimizer updates. They wait for the scientific study to release a slot.
Native final-serving qualification and measured three-GPU placement remain
incomplete. Source-setting training and validation scores alone do not verify
a paper's reported test metric.

**The full campaign is awaiting preparation and user GO.** The launch plan
contains 6,460 fit tasks and 270 export-only tasks but has started none. Only
the dedicated allocation is authorized. Both future 18.77 workers are disabled.
The [budget review](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/resource_budget_draft_v1/README.md) separates exact update counts from measured time and unknown costs.
A staged approval option is 646 fits across every setting and method, then the
remaining 5,814. The complete roster is retained; no campaign-wide ETA is claimed.

- [One-command runner source and remaining execution checks](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/portable_dispatcher_v3/README.md)
- [Exact method-by-task coverage and controls](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/benchmark_review_map_v1/README.md)
- [Plain-language benchmark design](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/BENCHMARK_OVERVIEW.md)
- [Complete 24-fit results](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/pilot_20261015_molbace_v1/complete_analysis_v1/REPORT.md)
- [Completed BN follow-up and limitations](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/pilot_20261015_bn_state_v1/complete_analysis_v1/INTERPRETATION.md)
- [Route quality and ranking-error analysis](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/analysis/complete_route_diagnostic_v1/REPORT.md)
- [Exact native settings and replication limits](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/expanded_study_design_v1/NATIVE_SETTINGS.json)
- [Current implementation and outstanding work](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/README.md)
- [Method catalogue and retained decisions](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/catalogue/CATALOGUE.md)
- [Scoped literature conclusions](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/literature/SCOPED_CONCLUSIONS.md)
- [Proposed fresh confirmation study](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/confirmation_protocol_proposal_v1/README.md)

The dated sections below preserve earlier states and proposals.
