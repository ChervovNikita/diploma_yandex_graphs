# Post hoc same-runtime TIED grid replication

This separate 36-cell study was designed after the original 432-cell study
and optimizer diagnostic outcomes were known, before any replication
training or test scoring. It tests whether the selected TIED comparison on
Cora and WikiCS changes when rerun in the optimizer study's A100 runtime.
It is not an untouched confirmation sample or a new method.
The original 03:56 UTC design record is retained byte-for-byte as
`SAME_RUNTIME_TIED36_DESIGN_ORIGINAL.md` (SHA-256
`06797e5cee6512806b439490167304a7b9755a71ccb4f3dbbb75cd0990f65df4`).

The exact original `tuning.train_one` training function is called with only
its result root redirected to `same_runtime_tied36_results`. No model,
optimizer, gradient, update, epoch, checkpoint-selection, or validation
metric code is copied or changed. The original frozen `tuning.py`,
`models.py`, data fingerprints, and complete 432-cell validation lock are
SHA-bound in this study's prospective freeze. `verify_tuning.audit_one` and
`verify_tuning.independent_metrics` supply a separate replay of traces,
checkpoints, validation metrics, and held-out scores. The GPU/runtime is
recorded by the launch report; the exact original data and graph tensor
hashes are checked through `tuning.check_freeze`.

Fixed matrix: Cora public Planetoid and WikiCS official split 0; TIED only;
learning rates 0.0003, 0.001, 0.003 crossed with AdamW decays 0 and 0.01;
seeds 0, 1, 2. This is 2 graphs × 6 candidates × 3 seeds = 36 fresh cells.
Each uses width 128, two residual SAGE blocks, four boundary-factor paths,
LayerNorm, dropout 0.2, mean member cross-entropy, and exactly 1,000 epochs
if finite. Per cell, choose one whole-model checkpoint by highest pooled
raw-logit validation accuracy, then lowest pooled validation CE, then the
earliest epoch. The prespecified default is `(0.001, 0)`.

Before training, check that every default TIED initialization reproduces
the independently locked optimizer-diagnostic TIED initialization hashes
for the same graph/seed. The initial member-logit digest is retained as a
diagnostic, not an exact gate, because sparse CUDA reductions can differ in
the last bits across identical state/RNG replays. The frozen initialization
anchor contains no test results. After each completed finite training cell,
save a separate validation-only companion of pooled/member logits, indices,
and labels from its selected checkpoint. This does not update the original
training checkpoint. The independent verifier requires bounded logit replay,
exact hard decisions, and matching selected validation accuracy/CE against
that companion before locking. A nonfinite seed invalidates its three-seed candidate, while every
planned cell still runs. An independent validation-only audit must verify
all 36 cell identities, traces, checkpoint hashes and states, and the
selection rule. It then writes a single immutable lock: per graph, select
the candidate with highest three-seed mean selected validation accuracy,
then lowest mean CE, then lower learning rate and decay.

Only after that lock, the separate 72-cell all-layer validation lock, and
the exact original 432-cell lock match their freezes may test labels be
loaded. Score only the selected TIED candidate
and the declared default, once each if identical, for all three seeds.
Independently replay every allowed checkpoint and pooled/member logit
array, compare exact class decisions and bounded float logits, and reject
missing or extra score files. Report all 36 validation cells and every
allowed held-out score, regardless of sign. The original primary and
optimizer studies remain immutable. The operational complete-study audit
cutoff is 06:00 UTC, 26 September 2026; an incomplete study is not used
for a complete comparison.
