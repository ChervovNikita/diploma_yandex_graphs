# Post hoc normalized-Cora parameter-matched comparator study

This exploratory study was proposed after the primary and matched-preprocessing
Cora outcomes and reviews were known. It tests whether the favorable
normalized-feature Cora result for width-128 PRIVATE-LAST persists against
parameter-matched ordinary two-block SAGE BASE and independent four-model ENS.
It uses one public split and is not independent graph confirmation. Report all
three arms, all selected and fixed-default comparisons, and negative outcomes.
The PRIVATE-LAST arm and width 128 were chosen after the earlier favorable
normalized-Cora result was known. BASE and ENS widths are determined by stored
parameter count alone in the frozen scan `CORA_MATCHED54_WIDTH_SCAN.json`;
their choices do not use outcomes from this new study.

Use the exact `tuning.py`, `models.py`, `FROZEN_STUDY.json`, public Cora source
data, and row-normalization rule from the earlier matched Cora study. Its
`train_one` function has SHA-256
`4fbbfed4226f9b9e2edc10d7f91df2b4086f530f3e64ad2401d722f7c635c4f4`,
identical to the primary six-arm training routine. Its model source has
SHA-256 `07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303`,
also identical to the primary source. Normalized nonnegative float32 features
divide each row by `max(row sum, 1.0)`; zero rows remain zero. The earlier
frozen graph descriptor and raw-data hashes bind features, labels, split
indices, and coalesced edges. All source/data are copied into a subfolder of
the existing GPU77 validation-tuning repository, and the original freeze is
rechecked before this new freeze and every run.

The stored-weight target is 604,700 parameters. BASE width 184 has 605,919;
ENS width 70 has 602,868; PRIVATE-LAST width 128 has 604,700. This matches
stored parameters, not FLOPs, latency, optimizer state, or diversity. Train
3 arms x 6 AdamW candidates x 3 seeds = 54 cells under one GPU77 software
runtime, with 1,000 epochs per cell, two residual SAGE blocks, dropout 0.2,
and four ensemble members where applicable. Schedule disjoint partitions on
GPU0/GPU1 by the parity of `candidate_index * 3 + seed`; every arm has nine
cells per partition. This depends only on frozen matrix indices.

Learning rates are 0.0003, 0.001, and 0.003 with weight decay 0 or 0.01.
Within a cell choose the checkpoint by highest pooled validation accuracy,
then lowest pooled validation CE, then earliest epoch. Independently replay
all 54 complete traces, checkpoints, pooled/member validation floats and
class decisions, and write a single global validation-only lock. Within each
arm choose the candidate by highest three-seed mean validation accuracy,
then lowest mean CE, then lower learning rate and decay. Test labels enter
neither gradients nor checkpoint/candidate selection; full labels may be read
for frozen tensor fingerprints. Only after the global lock may the selected and declared
default (0.001, 0) candidates be test-scored. Independently replay every
allowed pooled/member test decision, accuracy, and CE.

The complete audited study must be available by 2026-09-26 08:25 UTC for
this submission. If incomplete, it remains author evidence only. Accuracy
variation across three optimizer seeds is descriptive and does not measure
generalization over graph draws.
