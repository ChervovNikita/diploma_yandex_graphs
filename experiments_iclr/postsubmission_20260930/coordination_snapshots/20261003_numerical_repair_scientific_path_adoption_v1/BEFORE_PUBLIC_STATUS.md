# Current GNNM research status

Updated: 2026-10-03T18:02:36.698895+00:00. Goal active and incomplete. No new audited GNNM predictive winner, established methodological extension, revised manuscript or independent acceptance verdict. Original paper scores remain unchanged.

## Current scientific decisions

The graph-based initialization variant is closed without more tuning or heldout promotion. All 30 fits and 72 phases finished. The numerical audit recomputed selected-state validation NLL within the original 1e-6 tolerance and the supervisor reaped its child with exit 0. On Photo, graph initialization averages 0.35047 nats NLL versus 0.32173 for unchanged warm copying, with the same mean accuracy. On Squirrel, its differences from random or topology-permuted initialization are very small. These are selected-checkpoint development results on three overlapping split blocks. All30 serialized checkpoints now passed fresh complete-logit replay within the original atol1e-6/rtol1e-5 bounds. Fifteen cases reproduced tensor and NPY bytes exactly. The root supervisor reaped its inference child with exit0 in269.46seconds. This verifies restoration and preserves the no-promotion decision. All failures and costs are retained. See [complete results](graph_init_analysis_companion_v10_cpu_release_v1/RESULTS_SUMMARY_v1.md).

The six native Amazon Ratings fits and independent serialized-checkpoint replay are complete. Defaults average 42.7024% validation accuracy and the Roman transfer 44.0851%, with worse mean NLL and substantial overfit. These scores use the disclosed 80/20 official-TRAIN fit/control adaptation. They establish neither a GNNM gain nor competence under an interchangeable published protocol. Both recipes remain recorded. See [results](amazon_ratings_native_warm_execution_root_20261003_v3/RESULTS_SUMMARY_v1.md).

The next fixed comparison uses authored Amazon Polynormer-r with raw features, 200 local epochs, 2500 global epochs and its native state transition. Three blocks compare GNNM4 with four genuinely independent native members. A native single aliases the first independent member, giving 15 distinct fits. V4 failed its complete next-update replay gate with exit 1. The installed Adam implementation allowed its saved step tensor to be aliased between restores. V5 gives each optimizer an owned saved-state copy and checks image immutability. Independent source review passed, but repaired numerical equivalence remains unverified. The original gates, scientific recipe, and all failed costs are preserved. No predictive fit has started.

The completed DBLP comparison includes GAT, Simple-HGN and SeHGNN over five paired split blocks. All 15 native checkpoints passed independent restoration and validation-logit replay. Mean validation accuracy was 93.169% for global BatchEnsemble HGT, 93.909% for GAT, 94.239% for Simple-HGN and 93.992% for SeHGNN. All 30 requested score rows and nine paired comparisons remain available. VALID selects checkpoints and has 243 nodes on each overlapping split. These results do not show GNNM superiority. The completed relation-conditioned HGT study, Squirrel shared-ensemble study and PPI spectral study also supplied no promoted predictive winner.

## Training confirmed live

At 17:56:40 UTC, the mixed shared/private objective family had 37 of 40 selected cases. The original supervisor and child matched their recorded scripts and start times. Three ACM cases remained, with no study closure and no partial quality comparison.

At 17:56:40 UTC, both original NCNC queues on 18.77 remained live with exact recorded start times. Native bank seed 2 finished 100 epochs. Native70 seed 2 reached epoch 14 and native bank seed 3 reached epoch 71. The complete family requires five seeds, 35 fits and 25 served cells. TEST remains locked.

At 17:54:08 UTC, the original BUDDY supervisors on 18.77 were live. Native1024 and single256 seed 2 finished 100 epochs. Factorized4 and independent4 seed 2 reached epochs 29 and 3. The complete family requires 15 cells and 24 fits. No partial quality comparisons were performed.

## Comparator qualification

FoRDE's Gram backend failed severe float32 cancellation and remains excluded. The explicit streamed backend passed its tiny CPU checks and both full-Amazon M4/B128 derivative resource profiles, with peak allocated memory1.350/2.319 GiB. Q03 then passed every declared value and private-gradient comparison for two fixed full-input M4/B2 CPU-cache/CUDA endpoints. Its console exited0 in279.111 seconds. There were252/860 comparison records, with largest scaled errors0.001808/0.001860 against a required maximum1. This establishes those B2 endpoints. B128 numerical equivalence and predictive benefit remain unproven. The graph adapter epsilon1e-24 differs from the upstream default1e-12. No trained profile state becomes a donor. See [Q03 report](forde_graph_small_real_B2_oracle_execution_root_20261003_v1/RESULTS_SUMMARY_v1.md).

## Literature and provenance

Literature memory [index_v39](literature_memory/index_v39/LITERATURE_INDEX.json) retains 163 scoped conclusion records across 114 normalized paper identities and two software identities. Five new primary method scopes were read. These counts do not certify whole-paper reads. Two close primary sources remain inaccessible and unresolved.

The new NCNC pilot compares joint reconstruction of TRAIN observation patterns with equally supervised reconstruction of their marginal incidences. Its two fresh 100-epoch fits use the same masking, completion bank and main target loss. GRAN supplies the mixture-of-Bernoulli ancestry. Source V2 passed independent review after a preserved closure-custody rejection. Numerical and complete-graph qualification remain required before training. Observation absence is not treated as verified absence of a latent link. No novelty or predictive value is established.

A capable count-aware single-model comparator is specified prospectively. It can model dependence through the total residual count and keeps that distribution through four paid decoder draws. A matched-marginal diagnostic removes dependence from the same selected bank. This is a source plan, not an implemented or qualified method. Its complete-support dynamic programming may be expensive. The analytical higher-order witnesses do not establish that such patterns occur usefully in the real graph.

Latest verified pushed and synchronized head: `e2bcb4a61f9e5d8667f4823a102ea69bb70360fb` on `codex/postsubmission-research-20260930`. This turn's replay, repaired-source and research records await the next publication. Earlier status bytes are preserved in `coordination_snapshots/20261003_replay_and_source_resume_v1` and Git history.

Normal execution stays inside authorized repositories. Ordinary incidental caches are allowed. The seven-GPU account is forwarding only. No filesystem isolation, sudo, driver changes, PDF compilation, GENLINK or unrelated changes. The earlier unnecessary isolation and restart of our four BUDDY processes were agent mistakes, recorded in [server accountability](gpu77_connection_recovery_v1/SERVER77_ACCOUNTABILITY_20261003.md). Fresh manuscript reviewers receive immutable paper/evidence without author history or a requested verdict. Source approvals and resource checks do not count as acceptance.
