# Comprehensive new-data study: preparation roster

Status: disabled. No new scientific fits or approval receipt have been created. Resource qualification, inventory closure, transfer verification and the user's explicit go-ahead remain required.

The current frozen plan contains 79,485 model-fit groups in 92 studies. A fit group may contain multiple route owners or sequential paid stages; physical body and optimizer counts are listed separately in the frozen summary. Native author CLI groups are listed separately in the CSV and are not included in this fit-group count.

The original GNNM paper's five datasets are excluded. Original GNNM remains a comparison on compatible new tasks. Ordinary independent ensembles own separate models, optimizers and selectors; exact member-zero reuse is declared where implemented. Native paper-model tracks and deliberately adapted comparative tracks retain their distinct protocols.

Training budgets are source specific. Full-budget trajectories keep their full declared budget and save selected states separately from final states. Native-stopping trajectories retain their source patience and cap; exact executed work can only be known after training. Convergence traces, failures and all planned outcomes are retained.

The launcher requests two fit slots per GPU. Its memory reservation policy can serialize jobs that cannot safely coexist. Current scheduling weights are planning inputs, not a runtime estimate. No experiment is assigned to the seven-GPU allocation.

The same raw Amazon Photo graph is used for multiclass and an explicitly declared binary source adaptation. This is one raw dataset with different tasks. Study counts and dataset counts must not be confused.

## Study inventory

| Study | Dataset and task | Methods | Fit groups |
|---|---|---:|---:|
| native-practical-a120261012-citeseer-sage | citeseer / node_classification | 13 | 326 |
| native-practical-a120261012-citeseer-gcn | citeseer / node_classification | 5 | 133 |
| native-practical-a120261012-amazon-photo-sage | amazon-photo / node_classification | 13 | 920 |
| native-practical-a120261012-amazon-photo-gcn | amazon-photo / node_classification | 5 | 430 |
| coordinate-practical-a1-20261012-amazon-photo | amazon-photo / node_classification | 8 | 1,190 |
| polyformer-coauthor-practical-v1-coauthor-cs-mono | coauthor-cs / node_classification | 36 | 3,438 |
| polyformer-coauthor-practical-v1-coauthor-physics-mono | coauthor-physics / node_classification | 36 | 3,438 |
| fixed-context-keys-wikics-v1 | wikics / node_classification | 13 | 1,770 |
| context-exclusion-v2-wikics | wikics / node_classification | 9 | 1,242 |
| polyformer-practical-v1-citeseer-mono | citeseer / node_classification | 36 | 927 |
| polyformer-practical-v1-pubmed-mono | pubmed / node_classification | 36 | 927 |
| practical-molecular-v4-ogbg-molhiv-e300 | ogbg-molhiv / graph_binary_classification | 13 | 225 |
| practical-molecular-v4-ogbg-molbace-e300 | ogbg-molbace / graph_binary_classification | 13 | 225 |
| original-boundary-ncn-ncnc-collab-v1 | ogbl-collab / link_prediction | 21 | 375 |
| retained-pt-v1-wikics | wikics / node_classification | 12 | 2,020 |
| retained-pt-v1-amazon-photo | amazon-photo / node_classification | 12 | 1,120 |
| private-gru-v1-wikics | wikics / node_classification | 11 | 1,942 |
| private-gru-v1-amazon-photo | amazon-photo / node_classification | 11 | 1,072 |
| continuous-affinity-standardized-v1-wikics | wikics / node_classification | 10 | 1,554 |
| continuous-affinity-standardized-v1-amazon-photo | amazon-photo / node_classification | 10 | 894 |
| landmark-memory-v1-wikics | wikics / node_classification | 12 | 2,020 |
| landmark-memory-v1-amazon-photo | amazon-photo / node_classification | 12 | 1,120 |
| private_cmcl-standardized-v1-wikics | wikics / node_classification | 9 | 1,062 |
| private_cmcl-standardized-v1-amazon-photo | amazon-photo / node_classification | 9 | 612 |
| private_adam-standardized-v1-wikics | wikics / node_classification | 5 | 750 |
| private_adam-standardized-v1-amazon-photo | amazon-photo / node_classification | 5 | 420 |
| masked_core-standardized-v1-wikics | wikics / node_classification | 11 | 1,218 |
| masked_core-standardized-v1-amazon-photo | amazon-photo / node_classification | 11 | 708 |
| endpoint-value-standardized-v3-wikics | wikics / node_classification | 6 | 828 |
| endpoint-value-standardized-v3-amazon-photo | amazon-photo / node_classification | 6 | 468 |
| private-bootstrap-standardized-v1-gat-wikics | wikics / node_classification | 15 | 2,250 |
| private-bootstrap-standardized-v1-sage-wikics | wikics / node_classification | 15 | 2,250 |
| private-bootstrap-standardized-v1-gat-amazon-photo | amazon-photo / node_classification | 15 | 1,260 |
| private-bootstrap-standardized-v1-sage-amazon-photo | amazon-photo / node_classification | 15 | 1,260 |
| ogb-batchnorm-declared-seed-author-20261012-v1 | ogb-two-molecule-native-author-roster / graph_binary_classification | 0 | 0 |
| common-native-wrapper-v1-wikics-gcn | wikics / node_classification | 6 | 948 |
| common-native-wrapper-v1-wikics-gat | wikics / node_classification | 6 | 948 |
| common-native-wrapper-v1-amazon-photo-gcn | amazon-photo / node_classification | 6 | 528 |
| common-native-wrapper-v1-amazon-photo-gat | amazon-photo / node_classification | 6 | 528 |
| common-native-wrapper-v1-wikics-sage | wikics / node_classification | 11 | 1,698 |
| common-native-wrapper-v1-amazon-photo-sage | amazon-photo / node_classification | 12 | 1,068 |
| retained-polynormer-mechanisms-v1-wikics | wikics / node_classification | 10 | 960 |
| retained-polynormer-views-v1-wikics | wikics / node_classification | 6 | 648 |
| retained-polynormer-mechanisms-v1-amazon-photo | amazon-photo / node_classification | 10 | 570 |
| retained-polynormer-views-v1-amazon-photo | amazon-photo / node_classification | 6 | 342 |
| independent-preaggregation-growth-citeseer-heart-v2 | citeseer-heart / link_prediction | 8 | 147 |
| native-combinations-v1-gncl-supcon-wikics-sage | wikics / node_classification | 9 | 1,242 |
| native-combinations-v1-nonlocal-label-retrieval-wikics-gcn | wikics / node_classification | 7 | 1,086 |
| native-combinations-v1-nonlocal-label-retrieval-wikics-gat | wikics / node_classification | 7 | 1,086 |
| native-combinations-v1-nonlocal-label-retrieval-wikics-sage | wikics / node_classification | 7 | 1,086 |
| native-combinations-v1-shared-recurrent-class-decoder-wikics-gcn | wikics / node_classification | 10 | 1,320 |
| native-combinations-v1-shared-recurrent-class-decoder-wikics-gat | wikics / node_classification | 10 | 1,320 |
| native-combinations-v1-shared-recurrent-class-decoder-wikics-sage | wikics / node_classification | 10 | 1,320 |
| native-combinations-v1-gncl-supcon-amazon-photo-sage | amazon-photo / node_classification | 9 | 702 |
| native-combinations-v1-nonlocal-label-retrieval-amazon-photo-gcn | amazon-photo / node_classification | 7 | 606 |
| native-combinations-v1-nonlocal-label-retrieval-amazon-photo-gat | amazon-photo / node_classification | 7 | 606 |
| native-combinations-v1-nonlocal-label-retrieval-amazon-photo-sage | amazon-photo / node_classification | 7 | 606 |
| native-combinations-v1-shared-recurrent-class-decoder-amazon-photo-gcn | amazon-photo / node_classification | 10 | 750 |
| native-combinations-v1-shared-recurrent-class-decoder-amazon-photo-gat | amazon-photo / node_classification | 10 | 750 |
| native-combinations-v1-shared-recurrent-class-decoder-amazon-photo-sage | amazon-photo / node_classification | 10 | 750 |
| retained-curvature5-photo-paid-lifecycle-v2 | amazon-photo / node_classification | 8 | 762 |
| retained-lowrank-gine-source-molbace-e100-v1 | ogbg-molbace / graph_binary_classification | 4 | 21 |
| retained-original-BSNN-deterministic-d2-twitch-de-v1 | twitch-de-musae-raw / node_classification | 2 | 150 |
| native-buddy-collab-newdata-v1 | ogbl-collab / link_prediction | 5 | 114 |
| literal-FiLM-source-wikics-mask0-v1 | wikics / node_classification | 3 | 18 |
| literal-E-Rank1-BNN-source-wikics-mask0-v1 | wikics / node_classification | 6 | 36 |
| independent-neighborhood_distribution-v3-wikics | wikics / node_classification | 7 | 1,086 |
| independent-neighborhood_distribution-v3-amazon-photo | amazon-photo / node_classification | 7 | 606 |
| independent-sparse_feature_kernel-v3-wikics | wikics / node_classification | 8 | 1,344 |
| independent-sparse_feature_kernel-v3-amazon-photo | amazon-photo / node_classification | 8 | 744 |
| independent-shared_message-v3-wikics | wikics / node_classification | 6 | 828 |
| independent-shared_message-v3-amazon-photo | amazon-photo / node_classification | 6 | 468 |
| independent-private_hop-v3-pubmed | pubmed / node_classification | 5 | 123 |
| paid-native-label-stage-newdata-v1-wikics | wikics / node_classification | 11 | 932 |
| paid-native-label-stage-newdata-v1-amazon-photo | amazon-photo / node_classification | 11 | 512 |
| native-nsd-legacy-twitch-de-musae-raw-v2 | twitch-de-musae-raw / node_classification | 8 | 1,578 |
| native-nsd-geometry-twitch-de-musae-raw-v2 | twitch-de-musae-raw / node_classification | 8 | 1,308 |
| native-SeHGNN-source-seven-conditions-newdata-v1 | hgb-imdb-native5-development / multilabel_node_classification | 20 | 1,230 |
| native-meansage-peer-photo-v1 | amazon-photo / node_classification | 10 | 660 |
| native-meansage-tour-photo-v1 | amazon-photo / node_classification | 10 | 570 |
| native-HGT-mixed-DBLP-newroles-v2 | HGB-DBLP / node_classification | 13 | 570 |
| native-polynormer-qk-standardized-v1-wikics | wikics / node_classification | 10 | 1,680 |
| native-polynormer-qk-standardized-v1-amazon-photo | amazon-photo / node_classification | 10 | 930 |
| polyformer-unchanged-author-cli-20261012-v1 | polyformer-four-source-dataset-author-roster / node_classification | 0 | 0 |
| native-nsd-cold-geometry-source-twitch-v1 | twitch-de-musae-raw / node_classification | 3 | 180 |
| native-Bayesian-bsnn_cayley_d2-new-TwitchDE-v1 | twitch-de-musae-raw / node_classification | 2 | 150 |
| native-Bayesian-bsnn_richer_cayley_d2_l4-new-TwitchDE-v1 | twitch-de-musae-raw / node_classification | 2 | 150 |
| literal-private-response5-source60-full180-citeseer-heart-v1 | citeseer-heart / link_prediction | 7 | 30 |
| literal-label-corrector4-WikiCS-Session-mask0-v1 | wikics / node_classification | 6 | 27 |
| literal-typed-label-context6-SeHGNN-source200-three-role-v1 | hgb-imdb / node_classification | 9 | 45 |
| native-Physics-joint-dropout-newroles-v1 | Coauthor-Physics / node_classification | 11 | 54 |
| native-WikiCS-OIPG-coupled-newroles-v1 | WikiCS / node_classification | 10 | 570 |

## Exact details

`METHOD_ROSTER.csv` lists every study/method, backbone, split, seed, fit count, budget, selector and hyperparameter-point count. `EXACT_PROTOCOLS.json` lists the registered hyperparameter candidates and source learning/constructor settings. `STUDIES.json` binds each entry to its immutable full study specification. The full master summary separately counts bank exports, calibration fits, evaluation jobs, body instances, paid optimizer starts and bounded native work.

Canonical master hash: `8f691a4d7b9d833595d531b75b7cb78a3cba0cdb06d00015a6e15355e1883957`.
