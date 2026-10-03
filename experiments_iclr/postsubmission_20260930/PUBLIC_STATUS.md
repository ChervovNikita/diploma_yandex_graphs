# Current research status

Updated: 2026-10-03T02:59:59.114086+00:00. The goal remains incomplete: no new verified practical predictive improvement, methodological novelty, revised manuscript or independent acceptance verdict. Original paper scores are unchanged.

## Completed finding

The Squirrel PolyFormer support comparison completed twelve fits across three paired seed/split blocks. Mean validation NLL differences are only about 0.000027–0.000053 nats; illustrative paired intervals include zero. This is a practical tie, not predictive superiority. The failed and unpromising evidence remains retained. Photo follow-up support fits are prepared but have not started.

## Active quality hypothesis: relation-conditioned shared ensembles

Four HGT ensemble members share their main weight matrices while adapting differently to graph relation types. Conditional diagonal modulation and CP factor sharing are established ingredients; no new-operator claim is justified. The empirical question is whether the shared ensemble produces better predictions than ordinary BatchEnsemble, common relation conditioning, unrestricted conditioning, wider capacity and independent members.

The unchanged prospective DBLP study has seven methods across five paired development splits (35 cases), on the complete graph. Continuation requires at least 0.005-nat mean validation NLL improvement against each primary control, at least four of five paired wins, and macro-F1 nondecline. All 35 valid cases are required. A pass would justify independent dataset/heldout confirmation, not establish success by itself.

The first CPU attempt failed under an 8 GiB virtual-memory cap: five native-HGT fits selected, thirty other cases had allocation failures. No subset quality comparison was made. All original states, terminals and source history remain preserved. A six-update resource probe with actual fit-style tensor lifetimes passed all seven families under 16 GiB AS / 14 GiB RSS; peak virtual use exceeded the former cap. The fresh complete 35-case rerun is live on five CPU workers. At 02:55 UTC, all had advanced to ordinary BE training (18–21 epochs), with no worker errors. No old checkpoints were reused, and the scientific design did not change. Probe-based forecasts are 4.63 hours nominal / 9.26 hours with a factor-two allowance; these are not guarantees.

The separate native GAT, Simple-HGN and SeHGNN comparison is now training on the same five splits (15 cases). Actual-author numerical, gradient, preprocessing and continuation checks already passed. Full-graph resource qualification passed. At 02:58 UTC, the unchanged native driver had completed seven GAT epochs. This serial CPU study uses one thread, 16 GiB AS / 12 GiB sampled RSS, and an 18-hour wall cap. The measured forecast is 7.41 hours nominal / 14.82 hours with a factor-two allowance; a one-update resource probe cannot guarantee full-fit residency. SeHGNN uses TRAIN-only propagated labels and the explicitly frozen whole-target evaluation composition. Heldout label bytes remain closed.

## Other live representative studies

The original graph-initialization cohort's last fully audited count is 64/72 phases. The unchanged remaining fits are running on the authorized one-GPU allocation. At 02:55 UTC, Photo seed29/topology-permuted reached update605 (continuation355), with GPU utilization100%. Old coordinator monitoring counts are stale and are not canonical closure.

The full ogbl-collab BUDDY comparison on 18.77 comprises fifteen cells / twenty-four optimizer fits. At 02:55 UTC, seed0 native1024 and single256 had100 epoch records; factorized4 and independent4 had96 and65. Both authorized GPUs were98–100% utilized, with no reported errors. No incomplete-family comparison is open.

## Analysis, literature and publication

A saved-logit analysis is prepared to distinguish stronger individual members from improvements attributable to pooling, using the established loss/pooling-matched ambiguity decomposition. Embedding separation alone can leave the pooled prediction unchanged; this constrains unstructured contrastive-diversity proposals and is not a new theorem. The analysis runs after complete valid HGT closure.

Literature index_v27 contains119 conclusion records across75 normalized paper identifiers and two software identifiers. Scoped methods reads/revisits are not counted as full-paper reads; the cumulative full-paper total remains uncertified. A distinct-idea agent is examining one next label-relevant quality modification and its closest prior work. Ideas are prioritized over broad hyperparameter searches.

Latest verified pushed/synced commit before this update: `37316eb0751313a913d6aecdcddb01a8940e70cc`, branch `codex/postsubmission-research-20260930`. Compact runtime corrections, native launch, reviews and current status are being published. Large evidence, states, data and wheels stay on servers. Unlicensed raw third-party author source is excluded from Git redistribution. Previous decisions and failed attempts remain preserved.

Science stays inside the authorized repositories; normal incidental caches are allowed. No sudo, PDF compilation, GENLINK or unrelated-file changes. The seven-GPU account is solely the authorized MacLink forwarding relay. Fresh skill-based manuscript reviews will use immutable evidence without author history or a requested verdict. Source/engineering reviews do not count as paper acceptance.
