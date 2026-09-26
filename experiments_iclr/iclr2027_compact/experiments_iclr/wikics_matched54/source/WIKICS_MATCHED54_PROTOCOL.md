# Post hoc WikiCS parameter-matched comparator study

This exploratory study was proposed after the primary four-graph outcomes and
reviews were known. It tests whether the favorable WikiCS split-0 result for
width-128 PRIVATE-LAST persists against parameter-matched ordinary SAGE BASE and
four-model independent ENS. It is one public split, not independent graph
confirmation. Report all three arms and both selected and fixed-default
comparisons, including negative outcomes. Do not choose a model or width by
validation or test outcomes: PRIVATE-LAST width 128 is inherited from the
primary study; BASE and ENS widths are chosen by the frozen parameter-only
scan in `WIKICS_MATCHED54_WIDTH_SCAN.json`.

Use the original `tuning.py` and `models.py` source, unchanged, the original
WikiCS data and split-0 masks, features, labels, coalesced edges, two residual
SAGE blocks, dropout 0.2, four ensemble members where applicable, and the
original 1,000-epoch `train_one` routine. Only the output directory and
arm-specific width global are redirected before invoking that routine. The
target is 455,552 stored parameters. BASE width 198 has 456,004; ENS width
92 has 457,464; PRIVATE-LAST width 128 has 455,552. Stored weights are the
matching criterion. FLOPs and latency may differ and should be measured or
stated separately.

Train all 3 arms x 6 AdamW candidates x 3 seeds = 54 cells under one GPU77
software runtime. Schedule disjoint partitions on GPU0/GPU1 by the parity of
`candidate_index * 3 + seed`; every arm has exactly nine cells per partition.
This schedule depends only on frozen matrix indices and covers all 54 cells.
Learning rates are 0.0003, 0.001, 0.003, each with weight
decay 0 or 0.01. Within each cell choose the checkpoint by highest validation
accuracy, then lowest validation CE, then earliest epoch. Independently replay
all 54 complete traces and selected checkpoints, including exact pooled/member
validation class decisions, and write one global validation-only lock.
Within each arm choose the candidate by highest three-seed mean validation
accuracy, then lowest mean CE, then lower learning rate and decay. Training
and selection expose or use no test labels; full labels may be read for
frozen tensor fingerprints. Only the selected and
declared default (0.001, 0) candidates may be test-scored; independently
replay every allowed pooled/member test decision, accuracy, and CE.

The complete audited study must be available by 2026-09-26 08:15 UTC for
this submission. If incomplete, it remains author evidence only. Accuracy
variation across three optimizer seeds is descriptive and does not measure
generalization over graph draws. Matching stored parameters does not match
training FLOPs, inference latency, optimizer state, or ensemble diversity.
