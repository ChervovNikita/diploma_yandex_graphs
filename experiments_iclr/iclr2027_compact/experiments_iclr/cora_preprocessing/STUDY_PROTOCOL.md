# Matched Cora feature preprocessing sensitivity

## Question and status

This sensitivity study was proposed after earlier Cora and other graph results were
known. It asks whether the six-arm ordering on Cora changes when the same
source features are row-normalized. This is a post hoc question, not an
independent prespecified benchmark. Freeze both conditions, all candidates,
the validation rule, data fingerprints, and validation audit code before any
cell in this study trains. Freeze the test audit/export scripts before test
scoring. Report both conditions and every arm, including any
negative contrasts. Do not select the more favorable preprocessing condition
using its test accuracy.

## Matched inputs and model

Use PyTorch Geometric `Planetoid(name="Cora", split="public")` from one
shared source directory inside this study folder. Both conditions share the exact public
train/validation/test masks, labels, coalesced undirected edges, and raw
feature source. `cora_raw` uses the source float32 feature table unchanged.
`cora_normalized` divides each nonnegative float32 row by
`max(row sum, 1.0)`; a zero row remains zero. Data and processed tensor
SHA-256 hashes are included in the source freeze. Add no self-loops.

Use the byte-identical `train_one` routine from the earlier six-arm tuning
grid, SHA-256 `4fbbfed4226f9b9e2edc10d7f91df2b4086f530f3e64ad2401d722f7c635c4f4`.
Use the same model source. For each condition train BASE, independent
four-model ENS, TIED four-member boundary projector model, PRIVATE-FIRST,
PRIVATE-LAST, and UNTIED four-member graph propagation. Models use two
residual SAGE blocks, width 128, dropout 0.2, optimizer seeds 0/1/2, and
1,000 epochs per cell. Train each arm under all six AdamW configurations:
learning rate `0.0003`, `0.001`, or `0.003`, each crossed with weight decay
`0` or `0.01`. Total: 2 conditions x 6 arms x 6 candidates x 3 seeds =
216 cells. Each arm receives the same candidate budget. The partial-sharing
family searches both positions and therefore receives twice the search
budget of an individual comparator arm; retain that caveat.

## Selection and test firewall

Training exposes train and validation indices and labels only. Within a
cell, select the checkpoint with highest pooled validation accuracy, then
lowest pooled validation cross-entropy, then earliest epoch. Within each
condition/arm select the candidate with highest mean selected validation
accuracy across its three seeds, then lowest mean validation cross-entropy,
then lower learning rate, then lower decay. Select the partial-family position
using the same validation comparison, with PRIVATE-LAST on any remaining
exact tie. An independent verifier checks all 216 traces and checkpoints,
freshly replays each finite selected checkpoint, and writes one global
validation-only lock before any new test score is opened. Only then score
each arm's selected candidate and the fixed default `(0.001, 0)` in each
condition, each seed. Independently replay all scored checkpoints and exact
pooled/member test decisions.

Report per-arm mean and seed SD, paired seed differences to BASE, ENS,
TIED, and UNTIED, selected candidate, parameter counts, and the fixed-default
results for both conditions. The fixed-default comparison isolates the
preprocessing choice at the same learning rate and decay; the selected
comparison estimates performance under separately tuned candidates.
Interpret both as one public split with optimizer-seed variability, not
confidence intervals over graph draws or evidence of a general causal
advantage. A complete audit after 2026-09-26 07:50 UTC is a later result
and omitted from the current submission.
