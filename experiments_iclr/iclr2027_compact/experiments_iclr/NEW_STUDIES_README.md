# Additional matched studies in the anonymous supplement

These files accompany the paper's completed Roman depth, no-loop depth, and longer-budget studies, WikiCS/Actor depth extension, sharing-position tests including `ogbn-arxiv`, `ogbn-arxiv` 1,000-epoch, and `ogbl-collab` experiments. They contain frozen study source, selected-run metadata, complete validation traces, and compact decision arrays derived from the original float32 member logits. The original selected-logit files and training checkpoints were audited separately but are omitted here to keep the upload small. Each `result.json` or `selected.json` retains the SHA-256 hash of its original prediction file where the source runner recorded it. The compact arrays reproduce the paper's accuracies, Hits@50 values, member/pool decompositions, and coverage/utilization counts, but cannot replay the original CUDA checkpoint or reconstruct every floating-point logit.

From the extracted archive root, run:

```text
python experiments_iclr/verify_new_compact.py
python experiments_iclr/verify_roman_budget1000_compact.py
python experiments_iclr/roman_noloop_depth/verify_compact.py
python experiments_iclr/external_depth_sage/verify_compact.py
python experiments_iclr/sharing_position/verify_decisions.py
python experiments_iclr/ogbn_arxiv_sharing/verify_decisions.py
python experiments_iclr/verify_ogb1000_compact.py
```

These commands use NumPy only. Together they check 40 Roman depth cells, 12 Roman no-loop depth cells, 12 Roman longer-budget cells, 24 external-depth cells, 56 sharing-position arms, nine OGB node arms, and 12 link-prediction runs against the independently audited score manifests. They also check official labels and held-out indices where supplied. The full selected-logit and checkpoint audits were run before packaging and are identified by hashes in the included audit JSON files.

## Roman depth grid

`experiments_iclr/roman_depth_grid/grid_index.json` maps every depth, seed, and tied/untied arm to its frozen study directory. The two- and five-block seeds 0–2 were existing endpoint studies. The other 28 cells were frozen before their outcomes were read. `ROMAN_DEPTH_GRID_FREEZE.json` and the two audit JSON files state the graph fingerprints, paired initial-function checks, scores, and decision counts. Every study directory contains its original model and runner source, source manifest, protocol, validation trace, result, and compact selected decisions. To rerun a study, copy the included public `data/roman_empire.npz` to the study directory's `data/roman_empire.npz` path, install the listed dependencies, and use that directory's documented runner command with a new empty result root.

The compact `selected_decisions.npz` files hold valid/test official indices and labels, four member class predictions, and the class selected after averaging the original raw float32 member logits. A class-only record can verify the published decision statistics, but it does not regenerate the pooled class from raw logits. The original float32 selected-logit SHA-256 is in each run's `result.json`.

## Link prediction

`experiments_iclr/ogbl_collab_frozen.py` contains the frozen runner and `verify_ogbl_collab_frozen.py` the original full-artifact verifier. The compact result directories contain all three seeds for TIED, UNTIED, ENS, and BASE. Their `selected_decisions.npz` files contain, for each official positive edge, whether each member and the raw-logit pool beat the official 50th-highest negative threshold on validation and test. They also retain official test positive edge identifiers. These Boolean arrays reproduce Hits@50 and the member/pooling accounting. The original float32 edge-logit hash, checkpoint hash, 20 validation points, data fingerprints, selection, and training configuration are in the corresponding JSON and CSV files. A fresh full run requires OGB 1.3.6 and its public `ogbl-collab` dataset, which is downloaded under a user-chosen local directory by the runner.

The training loss samples pairs absent from the observed training graph. Such pairs are unobserved candidates, not verified biological or social nonedges. The paper uses the official fixed 100,000-edge negative pool only for validation and test Hits@50. No validation edge is added to the test message graph. This fixed recipe is not a leaderboard comparison with tuned link-prediction systems.

## Longer Roman budget

`experiments_iclr/roman_budget1000/` contains the from-scratch 1,000-epoch depth-two and depth-five endpoint repeat on Roman mask 0 with explicit self-loops. Its freeze manifest predates the complete 300-epoch grid outcomes. Twelve run records, full validation traces, source manifests, initial state checks, and selected member and pooled class decisions are included. `verify_roman_budget1000_compact.py` checks the 12 scores and paired means. The result includes one negative depth-five tied-minus-untied seed.

## Roman depth without added self-loops

`experiments_iclr/roman_noloop_depth/` repeats the tied/untied depth-two and depth-five endpoints on official Roman mask 0 using the same no-loop edge convention as WikiCS and Actor. Its six depth-two cells reproduce the earlier Roman bridge. The three depth-five tied-minus-untied test differences are +0.971, +0.600, and +0.600 percentage points. This was chosen after the Roman depth pattern and is one split with late selected checkpoints. The compact verifier checks all 12 selected scores, traces, and paired differences. Full logits and checkpoint replay remain author evidence.

## Sharing position

`experiments_iclr/sharing_position/` is a compact, separately audited four-arm study of where to place one private SAGE block. Its own README and `verify_decisions.py` explain the two graph edge conventions, all 44 selected results, and the decision-derived score checks. Roman Empire, WikiCS, and Actor each use one official split. Private-first and private-last have equal parameter counts within a graph, but their test ordering reverses across graphs. The full logits and checkpoint replay remain author evidence.

`experiments_iclr/ogbn_arxiv_sharing/` adds a separately audited four-arm extension on the official `ogbn-arxiv` time split. All 12 arms use three optimizer seeds, a 300-epoch budget, no explicit self-loops, and the same 128-wide two-block BatchEnsemble SAGE family. Private-first and private-last each have 436,928 parameters. Private-first exceeds private-last by +0.451, +0.547, and +0.403 test percentage points, respectively; all selected epochs lie between 293 and 300, so the result does not establish a converged ordering. The graph was chosen after the three earlier sharing-position outcomes were known. The nested `verify_decisions.py` checks retained official node IDs, predictions, scores, trace selection, and source hashes. Full logits and checkpoint replay remain author evidence.

## External depth extension

`experiments_iclr/external_depth_sage/` holds the frozen two-versus-five-block SAGE extension on WikiCS and Actor. Its 24 tied/untied arms use the published split 0 of each graph, three seeds, and the same 300-epoch recipe. The fresh depth-two results reproduce the older controls. Depth five does not reproduce the Roman self-loop graph's positive tying contrast. `verify_compact.py` checks complete selected decisions and trace selection. The source/data hashes and CUDA replay audit summaries are retained; full checkpoints and float32 logits remain author evidence.

## Longer OGB node budget

`experiments_iclr/ogbn_arxiv_1000_results/` contains all nine BASE, ENS, and GNNM arms from a from-scratch fixed 1,000-epoch `ogbn-arxiv` repeat. The source lock and protocol are beside it. `verify_ogb1000_compact.py` recalculates selected validation and test accuracies and the three paired test contrasts from class decisions. It checks validation-only selected epochs against all 1,000 trace rows. Class decisions do not reconstruct the raw logits or checkpoint weights. The independently audited full files remain in author evidence.
