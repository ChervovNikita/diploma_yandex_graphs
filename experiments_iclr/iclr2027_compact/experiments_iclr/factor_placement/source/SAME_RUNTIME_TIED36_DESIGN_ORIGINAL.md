# TIED grid replication in the optimizer-study runtime

Decision time: 2026-09-26 03:56:31 UTC (prospective design, before this replication is trained or test-scored).

Reason: the original 432-cell study trained Cora/WikiCS on the original A100 runtime, whereas the SYNC96 study uses the second A100 runtime. Repeat all six TIED optimizer candidates for those two graphs in the exact SYNC96 environment to remove that runtime difference from the selected SYNC-versus-selected TIED comparison. Actor and filtered Chameleon already use the same runtime. This is a replication of an existing grid, not a new optimizer search.

Matrix: Cora public Planetoid split and WikiCS official split0, TIED only, learning rates0.0003/0.001/0.003 crossed with weight decays0/0.01, seeds0/1/2, all1000 epochs, width128, two SAGE blocks, four boundary-factor members, LayerNorm, dropout0.2. Use original data, preprocessing, model, initial seeding, mean member loss, AdamW defaults, and pooled-logit checkpoint selection. No new hyperparameters. Total36 fresh cells.

Freeze executable source and data before training. Preserve a line-level diff against the original tuning runner. Changes may restrict matrix, names and result paths and adapt matrix audits, but must not alter models, optimizers, training updates, or selection rules. Confirm same runtime and default-arm initialization against the SYNC96 TIED controls.

Lock all36 completed validation cells before any held-out scoring. Pick one candidate per graph using mean selected validation accuracy, then mean cross-entropy, then lower learning rate and lower decay. Score only selected and declared default candidates. Replay all scored checkpoints and compare exact hard decisions and bounded logits. Retain every selected/default result, original validation traces and complete candidate table. State that graphs and settings were reused after earlier exploratory studies, and do not call this an untouched confirmatory dataset.

Required scoring/audits/profiling of already completed studies take priority. Run only after a GPU is released, with completion and audit expected by06:00 UTC. If incomplete by that cutoff, preserve the whole study as incomplete evidence and do not integrate partial test results. All comparisons and any numerical differences must be reported honestly, including failed improvement or changes in selected candidates.
