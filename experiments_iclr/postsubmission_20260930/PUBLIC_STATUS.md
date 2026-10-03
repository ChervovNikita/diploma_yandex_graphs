# Current research status

Updated: 2026-10-03T03:43:08.674672+00:00. The goal remains incomplete: no new verified practical predictive improvement, methodological novelty, revised manuscript or independent acceptance verdict. Original paper scores are unchanged.

## Completed finding

The Squirrel PolyFormer support comparison completed twelve fits across three paired seed/split blocks. Mean validation NLL differences are only about 0.000027–0.000053 nats; illustrative paired intervals include zero. This is a practical tie, not predictive superiority. The failed and unpromising evidence remains retained. Photo follow-up support fits are prepared but have not started.

## Active quality hypothesis: relation-conditioned shared ensembles

Four HGT ensemble members share their main weight matrices while adapting differently to graph relation types. Conditional diagonal modulation and CP factor sharing are established ingredients; no new-operator claim is justified. The empirical question is whether the shared ensemble produces better predictions than ordinary BatchEnsemble, common relation conditioning, unrestricted conditioning, wider capacity and independent members.

The unchanged prospective DBLP study has seven methods across five paired development splits (35 cases), on the complete graph. Continuation requires at least 0.005-nat mean validation NLL improvement against each primary control, at least four of five paired wins, and macro-F1 nondecline. All 35 valid cases are required. A pass would justify independent dataset/heldout confirmation, not establish success by itself.

The first CPU attempt failed under an 8 GiB virtual-memory cap: five native-HGT fits selected, thirty other cases had allocation failures. No subset quality comparison was made. All original states, terminals and source history remain preserved. A six-update resource probe with actual fit-style tensor lifetimes passed all seven families under 16 GiB AS / 14 GiB RSS; peak virtual use exceeded the former cap. The fresh complete 35-case rerun is live on five CPU workers. At 03:39 UTC, all five workers remained live with no runtime errors. Seeds137/139 reached untied-HGT training,149/151 unrestricted conditioning, and131 the CP arm. 26 cases had started; the complete35-case terminal is not yet available. No old checkpoints were reused, and the scientific design did not change. Probe-based forecasts are 4.63 hours nominal / 9.26 hours with a factor-two allowance; these are not guarantees.

The separate native GAT, Simple-HGN and SeHGNN comparison is now training on the same five splits (15 cases). Actual-author numerical, gradient, preprocessing and continuation checks already passed. Full-graph resource qualification passed. At 03:39 UTC, the unchanged native driver had GAT seed131 at226 epoch records and Simple-HGN seed131 at121. The only logged stderr was a TypedStorage deprecation warning; no runtime error was observed. This serial CPU study uses one thread, 16 GiB AS / 12 GiB sampled RSS, and an 18-hour wall cap. The measured forecast is 7.41 hours nominal / 14.82 hours with a factor-two allowance; a one-update resource probe cannot guarantee full-fit residency. SeHGNN uses TRAIN-only propagated labels and the explicitly frozen whole-target evaluation composition. Heldout label bytes remain closed.

## Independent ACM confirmation prepared

The official HGB-ACM release was acquired on the authorized server. It has10,942nodes, eight raw relations and547,872edge records. The original907-label development pool was split into726TRAIN/181validation nodes using each of the same five frozen seeds. Heldout label bytes remain closed. The confirmation scientific design was fixed before DBLP score inspection.

The ACM HGT source reuses the accepted fit, selection, checkpoint and replay code and adapts graph/features and the literal native ACM recipe. Root source adoption is complete; full-graph execution and replay are not yet qualified. A minimal native GAT/Simple-HGN/SeHGNN adapter is prepared for source review. ACM training has not started: it requires the complete DBLP continuation pass and the ACM resource/replay check. This is preparation for a decisive confirmation, not a new result.

## Other live representative studies

The original graph-initialization cohort's last fully audited count is 64/72 phases. The unchanged remaining fits are running on the authorized one-GPU allocation. At 03:40 UTC, Photo seed29/topology-permuted reached update1090 (continuation840), with GPU utilization100%. Old coordinator monitoring counts are stale and are not canonical closure.

The full ogbl-collab BUDDY comparison on 18.77 comprises fifteen cells / twenty-four optimizer fits. At 03:40 UTC, seed0 native1024, single256 and factorized4 had100 epoch records; independent4 and matched-single had75 and28. Both authorized GPUs were97–98% utilized, with no reported errors. No incomplete-family comparison is open.

## Analysis, literature and publication

A saved-logit analysis is prepared to distinguish stronger individual members from improvements attributable to pooling, using the established loss/pooling-matched ambiguity decomposition. Embedding separation alone can leave the pooled prediction unchanged; this constrains unstructured contrastive-diversity proposals and is not a new theorem. The analysis runs after complete valid HGT closure.

Literature index_v27 contains119 conclusion records across75 normalized paper identifiers and two software identifiers. Scoped methods reads/revisits are not counted as full-paper reads; the cumulative full-paper total remains uncertified. A distinct-idea agent is examining the already-scouted mixed gradient policy: train shared weights for the pooled prediction and private factors for individual member prediction. Closest-prior closure and a prospective full-training design are pending; no novelty claim is established. Saved algebraic scouts found that the proposed correction-credit losses collapse to known objectives and constant relation-attention biases cancel under native relation softmax; neither was promoted to training. Ideas are prioritized over broad hyperparameter searches.

Latest verified pushed/synced commit before this update: `a6ba9b127d82acfe1dd15e2dd8ed6fea0410f8dd`, branch `codex/postsubmission-research-20260930`. The runtime correction/native launch package is pushed and synced. Compact ACM preparation, saved scouts and the refreshed status are being published. Large evidence, states, data and wheels stay on servers. Unlicensed raw third-party author source is excluded from Git redistribution. Previous decisions and failed attempts remain preserved.

Science stays inside the authorized repositories; normal incidental caches are allowed. No sudo, PDF compilation, GENLINK or unrelated-file changes. The seven-GPU account is solely the authorized MacLink forwarding relay. Fresh skill-based manuscript reviews will use immutable evidence without author history or a requested verdict. Source/engineering reviews do not count as paper acceptance.
