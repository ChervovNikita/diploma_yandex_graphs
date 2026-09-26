# Prospective CiteSeer and PubMed confirmation protocol

## Status and question

These two public graphs were chosen after outcomes on Cora, WikiCS, Actor,
filtered Chameleon, and Roman Empire had been seen. They are new graph
settings for the same six-arm selection procedure, but they are not a
prespecified independent benchmark from the original study. The question is
whether the earlier validation and test patterns also appear on these graphs.
The answer must include all arms and negative contrasts.

## Data and model

Use PyTorch Geometric `Planetoid` with `split="public"`, names `CiteSeer`
and `PubMed`, and the raw files downloaded under this study folder only.
Convert the source feature table to float32. For each nonnegative row, divide
every entry by `max(row sum, 1.0)`. A zero row remains zero. Keep the public
train, validation, and test masks. Coalesce the raw edge index and make it
undirected, with no added self-loops. The source freeze records hashes of
every raw file and the resulting feature, label, edge, and split tensors.

Use the byte-identical `train_one` function from the earlier six-arm grid
(SHA-256 `4fbbfed4226f9b9e2edc10d7f91df2b4086f530f3e64ad2401d722f7c635c4f4`).
The model source is also copied unchanged. All arms have two residual SAGE
blocks, width 128, dropout 0.2, and seed values 0, 1, and 2. The six arms
are BASE, independent four-model ENS, TIED graph stack with four boundary
projector members, PRIVATE-FIRST, PRIVATE-LAST, and UNTIED. Every arm uses
all six AdamW settings from learning rates 0.0003, 0.001, and 0.003 crossed
with weight decays 0 and 0.01. Each cell trains for 1,000 epochs. No early
stopping changes the budget. There are 216 cells total, 108 per graph.

## Validation selection and test firewall

The training bundle contains train and validation IDs and labels. Its test
IDs and labels are exposed only to `score`, after the independent verifier
has checked all 216 traces and checkpoints, freshly replayed every selected
validation checkpoint, and written one global validation-only lock. Fresh
replay must reproduce accuracy within `1e-7` and cross-entropy within
`1e-5`. Its pooled and member validation class decisions are saved with
hashes in the lock. Exact logit-byte hashes are not required across GPU
forwards. Within a cell, choose the checkpoint
by highest validation accuracy, then lowest validation cross-entropy, then
earliest epoch. For each graph and arm, choose the hyperparameter setting
by highest mean selected validation accuracy across the three seeds, then
lowest mean validation cross-entropy, lower learning rate, and lower decay.
For the partial-sharing family, select FIRST or LAST by the same validation
comparison, with LAST on an exact remaining tie. In addition to the selected
setting, score the fixed default `(0.001, 0)` for every arm and seed. Score
no other setting. Independently replay every scored checkpoint and saved
test prediction. Report all comparisons, including negative ones.

Freeze source, raw data, processed tensor hashes, exact cell matrix, and
selection rule before any cell trains. Abort scoring if any cell is missing
or the global validation lock is incomplete. A complete audit after
2026-09-26 07:50 UTC is recorded as a later result and omitted from the
current submission.
