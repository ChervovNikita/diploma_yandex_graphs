# Current GNNM research status

Updated: 2026-10-03T15:52:44.885263+00:00. Goal active and incomplete. No new audited GNNM predictive winner, established methodological extension, revised manuscript or independent acceptance verdict. Original paper scores remain unchanged.

## Current scientific decisions

The graph-based initialization variant is closed without more tuning or heldout promotion. All 30 fits and 72 phases finished. The numerical audit recomputed selected-state validation NLL within the original 1e-6 tolerance and the supervisor reaped its child with exit 0. On Photo, graph initialization averages 0.35047 nats NLL versus 0.32173 for unchanged warm copying, with the same mean accuracy. On Squirrel, its differences from random or topology-permuted initialization are very small. These are selected-checkpoint development results on three overlapping split blocks. Serialized-checkpoint inference replay remains outstanding. All failures and costs are retained. See [complete results](graph_init_analysis_companion_v10_cpu_release_v1/RESULTS_SUMMARY_v1.md).

The six native Amazon Ratings fits and independent serialized-checkpoint replay are complete. Defaults average 42.7024% validation accuracy and the Roman transfer 44.0851%, with worse mean NLL and substantial overfit. These scores use the disclosed 80/20 official-TRAIN fit/control adaptation. They establish neither a GNNM gain nor competence under an interchangeable published protocol. Both recipes remain recorded. See [results](amazon_ratings_native_warm_execution_root_20261003_v3/RESULTS_SUMMARY_v1.md).

The next fixed comparison uses source-authored Amazon Polynormer-r, raw features, 200 local epochs followed by 2500 global epochs, and the native state transition. Three blocks compare GNNM4 boundaries with four genuinely independent native members. A native single aliases the immutable first independent member, giving 15 distinct fits. The source is being finished before independent review and real numerical qualification. The one-GPU allocation was verified idle and available at this turn's check. Training has not started.

The completed DBLP comparison includes GAT, Simple-HGN and SeHGNN over five paired split blocks. All 15 native checkpoints passed independent restoration and validation-logit replay. Mean validation accuracy was 93.169% for global BatchEnsemble HGT, 93.909% for GAT, 94.239% for Simple-HGN and 93.992% for SeHGNN. All 30 requested score rows and nine paired comparisons remain available. VALID selects checkpoints and has 243 nodes on each overlapping split. These results do not show GNNM superiority. The completed relation-conditioned HGT study, Squirrel shared-ensemble study and PPI spectral study also supplied no promoted predictive winner.

## Training confirmed live

At 15:37:35 UTC, mixed shared/private objectives had 32 of 40 selected cases, with eight ACM slots remaining. Original supervisor 379192 and child 379193 matched their recorded start times and scripts. No study closure exists and no partial quality comparison was performed.

At 15:40:39 UTC, both original NCNC queues on 77 were live. Seeds 0/1 had completed their private, pooled, native-bank and width70 units. Seed2 private training had reached epoch61 and seed3 private epoch3. The fixed family retains five seeds, 35 distinct fits and 25 served cells. TEST remains locked and no partial outcomes were scored.

At 15:32:37 UTC, original BUDDY supervisors and workers on77 were live. Independent seed1 was at epoch96, matched-single seed1 had finished100 epochs, and native1024 seed2 was at epoch32. The complete family requires15 cells and24 fits. No partial quality comparisons were performed.

## Comparator qualification

FoRDE's Gram backend failed severe float32 cancellation and remains excluded. The explicit streamed backend passed its tiny CPU checks and both full-Amazon M4/B128 derivative resource profiles, with peak allocated memory1.350/2.319 GiB. Q03 then passed every declared value and private-gradient comparison for two fixed full-input M4/B2 CPU-cache/CUDA endpoints. Its console exited0 in279.111 seconds. There were252/860 comparison records, with largest scaled errors0.001808/0.001860 against a required maximum1. This establishes those B2 endpoints. B128 numerical equivalence and predictive benefit remain unproven. The graph adapter epsilon1e-24 differs from the upstream default1e-12. No trained profile state becomes a donor. See [Q03 report](forde_graph_small_real_B2_oracle_execution_root_20261003_v1/RESULTS_SUMMARY_v1.md).

## Literature and provenance

Literature memory [index_v37](literature_memory/index_v37/LITERATURE_INDEX.json) retains157 scoped conclusion records across108 normalized paper identities and2 software identities. These are not whole-paper-read counts. The latest new method read is Dynamic Negative Correlation Learning. Adaptive scalar loss balancing is prior, and fixed graph-frequency/neighbor-error transforms reduce to graph-kernel NCL. This follow-up promoted zero pilots. A separate current scout examines whether a concrete shared-factor operation can represent dependent missing-neighbor configurations beyond marginal uncertainty. It has no adopted method or experiment yet.

Latest verified pushed and synchronized head: `870bfae124e0606c01039bddccbeba015656fbf0` on `codex/postsubmission-research-20260930`. This turn's new audits, reviews and records await the next publication. Earlier status bytes are preserved in `coordination_snapshots/20261003_closed_audits_v1` and Git history.

Normal execution stays inside authorized repositories. Ordinary incidental caches are allowed. The seven-GPU account is forwarding only. No filesystem isolation, sudo, driver changes, PDF compilation, GENLINK or unrelated changes. The earlier unnecessary isolation and restart of our four BUDDY processes were agent mistakes, recorded in [server accountability](gpu77_connection_recovery_v1/SERVER77_ACCOUNTABILITY_20261003.md). Fresh manuscript reviewers receive immutable paper/evidence without author history or a requested verdict. Source approvals and resource checks do not count as acceptance.
