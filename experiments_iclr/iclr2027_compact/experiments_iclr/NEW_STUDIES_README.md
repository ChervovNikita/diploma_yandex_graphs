# Additional matched studies in the anonymous supplement

The primary evidence is the four-graph, six-arm, 432-cell optimizer grid in `validation_tuning/`. Every selected primary model also has its original test member logits in `hpo_selected_test_logits/`. The additional studies below examine optimizer histories, factor placement, temperature scaling, depth, parameter budgets, and other graph tasks. Each study states its own scope, full matrix, data fingerprints, and verification boundary. These are different protocols, so their training runs are not exchangeable replications.

## Find the decisive evidence

| Question | Directory | Publicly recomputable evidence |
| --- | --- | --- |
| Sharing after optimizer search | `validation_tuning/`, `hpo_selected_test_logits/` | All 432 validation traces and selections, all allowed selected/default decisions, all 72 selected raw member test arrays |
| Two additional graph settings | `planetoid_confirmation/` | All 216 CiteSeer/PubMed validation cells, 57 selected/default test records and member/pooled classes |
| Feature preprocessing on Cora | `cora_preprocessing/` | All 216 raw/normalized validation cells, 60 selected/default records and member/pooled classes |
| Accuracy versus execution cost | `inference_profile/` | All 7,200 timings, peak-allocation records, 24 selected checkpoint identities |
| Common weights with separate optimizer histories | `optimizer_history/`, `initial_member_gram/` | Full planned outcome records, decision metrics, retained Gram matrices and exact algebraic identities |
| Factors inside propagation | `factor_placement/` | All 108 new validation records, 36 new pooled test arrays, exact selected/default comparisons |
| Probability quality | `temperature_sensitivity/` | All 72 validation fits and unscaled/scaled test metrics |
| Width at approximately equal parameter count | `roman_narrow/` | All six new narrow cells, original comparator records and paired differences |
| Graph-only link baselines | `ogbl_collab_topology/` | All eight complete official-pool score arrays and four strict Hits@50 values |
| Preparation from public inputs | `fresh_primary/` | Frozen source closure, exact public input downloads and graph-tensor gate, tested CPU preflight |

Large training checkpoints are omitted. Where a stage retains only member class decisions, it cannot reconstruct the member float logits or the raw-logit pool. Other stages retain exact pooled logits or full member logits, as specified below. Full checkpoint replays were audited before packaging and are retained author records.

## Prepare the primary benchmark from public inputs

`fresh_primary/README.md` gives commands to create an absent study directory, download the exact public inputs, check their raw hashes and graph tensors, and run the frozen primary runner. The download, input checks, and CPU preflight passed in a new directory without copying any author checkpoints, results, or test scores. Full fresh training was not run in this preparation check. The complete 432-cell training and validation-lock commands are included. A read-only source check is:

```sh
python experiments_iclr/fresh_primary/prepare_primary_fresh.py --source experiments_iclr/validation_tuning --target fresh_primary_432 --check-only
```

`VERIFICATION_SCOPE.md` gives a command-by-command account for the additional node-graph and decision-mechanism studies below: what a reader can recompute from selected decisions and traces, and which CUDA or full-logit audits are retained records because checkpoints and original float32 logits are omitted.

From the extracted archive root, run:

```text
python experiments_iclr/verify_new_compact.py
python experiments_iclr/verify_roman_budget1000_compact.py
python experiments_iclr/roman_additional_masks/verify_packed_results.py
python experiments_iclr/roman_noloop_depth/verify_compact.py
python experiments_iclr/external_depth_sage/verify_compact.py
python experiments_iclr/sharing_position/verify_decisions.py
python experiments_iclr/ogbn_arxiv_sharing/verify_decisions.py
python experiments_iclr/new_graph_studies/verify_decisions.py
python experiments_iclr/filtered_chameleon_study/verify_decisions.py
python experiments_iclr/analyze_decision_mechanism.py
python experiments_iclr/analyze_selected_sharing_tradeoffs.py
python experiments_iclr/verify_ogb1000_compact.py
```

These commands use NumPy only. Together they check 40 Roman depth cells, 12 Roman no-loop depth cells, 12 Roman longer-budget cells, 24 external-depth cells, 56 earlier sharing-position arms, 48 Cora/legacy-Chameleon depth and position arms, 24 filtered-Chameleon depth and position arms, nine OGB node arms, and 12 link-prediction runs against the independently audited score manifests. They also check official labels and held-out indices where supplied. The full selected-logit and checkpoint audits were run before packaging and are identified by hashes in the included audit JSON files.

## Roman depth grid

`experiments_iclr/roman_depth_grid/grid_index.json` maps every depth, seed, and tied/untied arm to its frozen study directory. The two- and five-block seeds 0–2 were existing endpoint studies. The other 28 cells were frozen before their outcomes were read. `ROMAN_DEPTH_GRID_FREEZE.json` and the two audit JSON files state the graph fingerprints, paired initial-function checks, scores, and decision counts. Every study directory contains its original model and runner source, source manifest, protocol, validation trace, result, and compact selected decisions. To rerun a study, first obtain the pinned public `data/roman_empire.npz` with `python experiments_iclr/fetch_datasets.py` from the archive root, then copy it to the study directory's `data/roman_empire.npz` path, install the listed dependencies, and use that directory's documented runner command with a new empty result root.

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

## Cora and legacy Chameleon extension

`experiments_iclr/new_graph_studies/` contains 48 Cora and legacy Geom-GCN Chameleon depth and sharing-position arms, frozen before their outcomes were read. The public Cora split and legacy Chameleon split 0 are each used with three optimizer seeds. The 300-epoch protocol, pretraining freezes, source/data fingerprints, complete validation traces, and decision arrays are retained. Chameleon contains 50 self-loops in its source graph; none were added. Legacy Chameleon has known duplicate-node and evaluation concerns, so its outcomes are descriptive rather than a clean replication on the filtered benchmark. The nested `verify_decisions.py` checks the complete compact stage, including trace selection, node IDs, predictions, parameter equality, paired contrasts, and hashes. Full selected logits and checkpoints passed separate CUDA replay but remain author evidence. On both graphs, tied propagation underperforms untied propagation at depths two and five in mean held-out accuracy. The predeclared prediction that private-first would underperform private-last on both graphs fails: mean private-first-minus-private-last test differences are +0.367 and +0.292 percentage points on Cora and legacy Chameleon, respectively. These are one-split, three-seed outcomes and do not establish a universal position ordering.

`experiments_iclr/filtered_chameleon_study/` contains a separately frozen 24-arm study on official filtered Chameleon split 0 from Platonov et al. The raw public NPZ is identified by pinned repository commit and SHA-256 in its README. The dataset/protocol proposal and filtered study freezes preceded inspection of the Cora and legacy-Chameleon outcomes. Its no-loop, symmetrized-edge model convention differs from the directed-edge evaluation in the original filtered-graph paper. Tied-minus-untied mean test accuracy is −2.577 percentage points at depth two and −2.921 at depth five; private-first-minus-private-last is +0.172 points across three seeds. The frozen validation rule selects private-last, with +0.172 points retrospective test regret. The selected checkpoints are all at epochs 6–17, limiting any optimization interpretation. All 24 full CUDA checkpoint/logit replays and an independent Mac audit of the original data, float32 logits, validation traces, selected scores, and cross-entropies passed before compacting. The nested verifier checks decision-level accuracy, source and stage hashes, node IDs, parameter equality, and the paired contrasts. Original selected logits and checkpoints remain author evidence.

`experiments_iclr/analyze_decision_mechanism.py` recalculates per-seed and mean member accuracy, pooled accuracy, gain from raw-logit pooling, six-pair member prediction disagreement, and any-member-correct coverage from the verified Cora, legacy-Chameleon, filtered-Chameleon, WikiCS, and Actor depth studies. It writes `DECISION_MECHANISM_CROSSGRAPH.json`; the supplied file has SHA-256 `f80770ea415b0ad2d8b100ceeb93c50a9c415303cc2121708d42e5adb6ab0bcb`. The results are descriptive one-split decision accounting. For example, Cora tied arms have more member prediction disagreement than untied arms despite lower pooled accuracy, so disagreement alone cannot explain the storage/accuracy contrast.

## Longer OGB node budget

`experiments_iclr/ogbn_arxiv_1000_results/` contains all nine BASE, ENS, and GNNM arms from a from-scratch fixed 1,000-epoch `ogbn-arxiv` repeat. The source lock and protocol are beside it. `verify_ogb1000_compact.py` recalculates selected validation and test accuracies and the three paired test contrasts from class decisions. It checks validation-only selected epochs against all 1,000 trace rows. Class decisions do not reconstruct the raw logits or checkpoint weights. The independently audited full files remain in author evidence.

## Validation-selected partial sharing

`analyze_selected_sharing_tradeoffs.py` applies the stated validation-accuracy, validation-CE, then private-last tie rule to all seven completed 300-epoch position settings. It regenerates `SELECTED_SHARING_TRADEOFFS.json`, including both partial arms, selected test scores, test regret, and the exact reduction in trainable parameters versus the fully untied arm. This analysis reads existing audited `result.json` records. It does not train a model or independently replay logits. The first four settings are retrospective applications, and the last three use a rule recorded before their outcomes were opened. The two Chameleon settings are related variants.

## Width-512 Roman fixed-mask optimization repeat

`fixed_mask_v4/` supplies all six arms of the separate mask-0 repeat, using three optimizer seeds, five SAGE blocks, width512, and the original component-study checkpoint rule. Run `python experiments_iclr/fixed_mask_v4/verify_fixed_mask_v4_compact.py` from the archive root. It verifies frozen source/calibration records, complete validation traces, retained official labels and selected class decisions, and paired score arithmetic. Tied-minus-untied differences are +3.6004, +3.7769, and +2.6121 percentage points. The mean is +3.3298. This repeats optimization on a post hoc favorable graph and does not establish a general graph effect. Its README explains earlier numerical-gate failures and the final pretraining tolerance calibration. Full logits, checkpoints, raw graph bytes, and identifying environment files remain author evidence. Their omissions are explicit in the compact verifier.

## First graph-update diagnostic

`initial_update_diagnostic/` records all 12 fixed CPU comparisons of TIED and synchronized-copy AdamW updates at matched initial weights and training random draws. Run `python experiments_iclr/initial_update_diagnostic/verify_compact.py` from the archive root. The command checks compact hashes, the complete graph/seed matrix, formula/result/audit links, CSV values, and plot provenance. It does not recompute gradients from the omitted raw arrays. The supplied source and public-graph hashes support recreation in the documented full study environment. Cosine0.641–0.857 and norm ratio0.517–0.784 describe the first graph update, without a prediction-quality claim. The source, prior AdaTask relation, gradient scaling, and post hoc sign-threshold sensitivity are documented in its protocol and README.

## Additional Roman masks at two depths

`roman_additional_masks/` supplies the complete 48-cell study on official masks 1–4, depths 2 and 5, and three optimizer seeds. Both arms use width 128, four boundary-projector members, no added self-loops, and 1,000 epochs. TIED and UNTIED have copied initial parameters and matched random-generator states. Every paired depth-two test difference is negative. The depth-five differences have eight positive signs, three negative signs, and one tie. Their means are −0.6118 and +0.1397 percentage points, respectively. Thirty-five of 48 selected checkpoints are after epoch 900. The masks overlap on one graph and were chosen after earlier Roman studies.

Run `python experiments_iclr/roman_additional_masks/verify_packed_results.py` from the archive root. The upload wrapper verifies the exact hashes of all 210 original `results/` files in `RESULTS_RECORDS.tar.xz`, temporarily extracts them, then runs the unchanged verifier. This checks source/result hashes, all 48 validation traces, selected decisions and accuracies, initialization records, eight CUDA replay summaries, the complete-grid audit, and all 24 paired differences. The optional `--public-npz /path/to/roman_empire.npz` argument verifies the supplied official label/mask anchor against separately downloaded public graph bytes. Full float32 logits, initial-logit arrays and checkpoints are omitted, so the compact command does not reconstruct logit pooling, cross-entropy or CUDA forwards.

## Fixed MC Dropout baseline on five Roman masks

Run `python experiments_iclr/verify_mc_dropout_compact.py` from the archive root. All five BASE checkpoints use four predetermined dropout draws at probability0.2, with no new training or tuning. Both raw-logit averaging and the arithmetic mean of member softmax probabilities are reported for every mask. The probability analysis was specified after the original logit-pooling results, using the same unselected draws. The compact stage retains deterministic BASE classes, all member classes, both pooled classes, official IDs and labels, and source/audit hashes. It verifies accuracies. Raw logits, checkpoints, and probability tensors are omitted, so the compact verifier cannot regenerate pooling or replay checkpoints. The full author evidence passed independent CUDA replay and full-array arithmetic audits before compaction.


## Primary 432-cell validation grid

`validation_tuning/` supplies the frozen six-arm, six-candidate, three-seed matrix on four graph settings. All 432 validation traces and result records are retained, as are all 141 unique selected/default score records and decisions. Run `python experiments_iclr/validation_tuning/verify_trace_archive.py`. The upload packs the original result and validation-trace bytes in `RESULTS_RECORDS.tar.xz`; the wrapper verifies their original hashes and runs the unchanged selection verifier after temporary extraction. Its README states the arithmetic and provenance checks and the omitted checkpoint/logit boundary. The candidate selection uses validation only, after the complete matrix is locked. Choosing between two tuned partial placements costs twelve configurations, versus six for each individual comparator.

## Separate optimizer histories and update norms

`optimizer_history/` supplies all 96 validation-grid cells and the separate 12-cell NORM-SYNC V2 extension. Run `python experiments_iclr/optimizer_history/verify_compact_optimizer_v2.py`. All 48 allowed SYNC/default comparison scores and all 12 NORM scores are included. The README records the numerical correction chronology, which arrays are available, and which complete CUDA replay records can only be checked by provenance. The diagnostic preserves TIED's inference function class but requires additional training memory. It does not consistently improve accuracy.

## Roman depth and optimizer histories

`roman_optimizer/` contains all 24 official-mask-0 cells at depths two and five with three optimization seeds. Run `python experiments_iclr/roman_optimizer/verify_roman_mechanism_compact.py`. An optional public NPZ path checks labels and node IDs against separately downloaded data. The package keeps every validation trace, class decision, original score record, amendment, and full CUDA audit summary. The compact verifier checks accuracy and provenance. Omitted logits and checkpoints prevent it from re-running model forwards or reconstructing pooling. A numerical verification amendment preceded any test score and is described in the nested README.

## Inference profiles for all selected primary arms

`inference_profile/` contains every one of the 7,200 synchronized timings and all 72 memory-allocation blocks for the 24 selected seed-0 checkpoints. Run `python experiments_iclr/inference_profile/verify_inference_profile_public.py`. This NumPy-only command recomputes each median, quartile and peak allocation, and checks each selected result against the 432-cell validation lock. GPU UUID and identifying process context are omitted. Model weight bytes, inference latency, and PyTorch peak allocated bytes are distinct measurements. The profile does not measure training memory or provide an equal-runtime accuracy comparison.


## Initial member-gradient cross terms

Run `python experiments_iclr/initial_member_gram/verify_gram.py`. It checks all twelve retained4×4 Gram matrices and the exact first-order graph-only SGD training-loss identities. The original gradient arrays were independently audited but are omitted. Source for recreating them is supplied in `initial_update_diagnostic/`, with the required frozen training modules in `optimizer_history/`. This is a post hoc initial training-gradient calculation, without an AdamW or test-performance claim.


## Original test logits for all primary selected models

`hpo_selected_test_logits/` adds original float32 member test logits for every one of the72 primary selected models. Run `python experiments_iclr/hpo_selected_test_logits/verify_public.py`. It reconstructs pooled float32 logits bitwise, checks scores and cross-entropy, and binds each cell to the432-cell validation lock,141-score audit, original-array hashes, official test references and independently exported classes. No model was selected by its test score. Original validation float logits and checkpoint weights remain outside the compact upload.

## Roman parameter-count width sensitivity

The six new narrow UNTIED cells use widths chosen by parameter count before
training and compare against the existing Roman optimizer study's TIED cells.
Run from the archive root:

```sh
python experiments_iclr/roman_narrow/verify_roman_narrow_compact.py --roman24 experiments_iclr/roman_optimizer
```

This check uses retained pooled test logits and validation/member decisions,
not omitted checkpoints or individual member float logits. The study README
records the validation-only numerical amendment and all six paired results.

## Post hoc probability-quality sensitivity

`temperature_sensitivity/STAGE_SCOPE.md` gives commands to refit all 72
validation temperatures and regenerate the complete unscaled/scaled test table.
The retained `test_results/TEMPERATURE_ALL24_MEANS.csv` reports means and
sample standard deviations, with its deterministic aggregation source supplied.
This uses standard temperature scaling with reused validation nodes and does
not change any predicted class.

## All-layer factor placement and same-runtime TIED control

`factor_placement/` retains all 108 validation cells across the 72-cell
all-layer-factor study and the 36-cell Cora/WikiCS same-runtime TIED control.
It also retains all 36 selected/default test score records with exact pooled
float32 test logits, member hard decisions, and the two complete independent
final score audits. Run `python experiments_iclr/factor_placement/verify_factor_compact.py --stage experiments_iclr/factor_placement --primary experiments_iclr/validation_tuning`.
For six Actor/filtered-Chameleon original TIED default predictions, run
`python experiments_iclr/factor_placement/legacy_tied_default/verify_legacy_default.py --stage experiments_iclr/factor_placement/legacy_tied_default --primary experiments_iclr/validation_tuning`.
The nested README explains the full-array projection audit and the omitted
checkpoint/member-float-logit boundary. Its `provenance/` directory retains
the pre-test scoring amendment, separate GPU preflights, all eight comparison
rows, and hashed independent audits. Selected all-layer minus TIED mean test
differences are +2.233, -0.239, -0.132, and +1.890 percentage points on
Cora, WikiCS, Actor, and filtered Chameleon, respectively. Factor placement
was studied after earlier results and adds parameters, so these comparisons
do not isolate a causal placement effect.

## Paired selected decisions on eight settings

`paired_decisions/` compares the validation-selected partial-family arm with
validation-selected ENS on the same test nodes within each optimizer seed.
Across the four primary graphs, CiteSeer/PubMed, and both new Cora feature
conditions, it retains all 24 seed-level partitions into both correct,
partial only correct, ENS only correct, and neither correct. Run
`python experiments_iclr/paired_decisions/analyze_paired_decisions.py --check`.
The script checks source hashes, selected candidates, official test references,
and recorded accuracies before checking the exact CSV/JSON outputs. Seeds
reuse test nodes and the Cora settings are related, so these counts carry no
independent-node interval or significance claim.
An optional `python experiments_iclr/paired_decisions/conditional_seed_intervals.py --check`
recomputes three-seed t intervals conditional on each fixed graph split and
selected candidate. It assumes independent, approximately normal optimizer
seed differences and does not account for validation selection across those
same seeds. It is not an interval over new graphs or individual test nodes.

## Post hoc width and approximate parameter matching

`narrow_capacity/` supplies 72 UNTIED cells across the four primary graph
splits, with each width chosen from parameter counts to approach TIED width
128, plus an 18-cell UNTIED width-128 WikiCS control. All cells receive the
same six optimizer candidates, three seeds, and 1,000-epoch budget. Run
`python experiments_iclr/narrow_capacity/verify_narrow_wide_compact_v2.py experiments_iclr/narrow_capacity`
and `python experiments_iclr/narrow_capacity/verify_comparator_bindings.py`.
The first command checks all 90 validation cells and 27 allowed selected/default
test scores, with exact pooled float32 test logits. The second binds the
published contrasts to the independently retained comparator score records.
The selected narrow-minus-TIED differences on Cora, WikiCS, Actor and filtered
Chameleon are +6.133, +0.331, +0.307 and 0.000 points. Wide UNTIED is nearly
tied with the same-runtime TIED WikiCS control (+0.006 points). These studies
were designed after earlier outcomes. Changing width changes the initial
function as well as parameter count, so the contrasts are sensitivity checks,
not causal estimates of a storage-capacity effect.

## Storage-matched WikiCS controls

`wikics_matched54/` is a post hoc, 54-cell WikiCS split-0 comparison among a
single BASE width 198, pooled ENS width 92, and private-last width 128. Their
stored trainable parameter counts are 456,004, 457,464, and 455,552. Six
AdamW candidates and three optimizer seeds per arm were completed and locked
before selected/default test scoring. Selected mean test accuracies are
77.926, 78.770, and 79.454%. Run
`python experiments_iclr/wikics_matched54/source/verify_wikics_matched54_compact.py experiments_iclr/wikics_matched54`.
The compact stage retains all 54 validation traces and exact float32 pooled
test logits for all 18 allowed scores, but not original weights/member floats.
`python experiments_iclr/paired_decisions/matched_wikics_seed_intervals.py --check`
recomputes descriptive paired seed differences conditional on this one fixed
split and the selected candidates; it does not account for selection on the
same seeds. The separate post hoc profile retains all 900 timings for three
selected seed-0 checkpoints. Run
`python experiments_iclr/wikics_matched54_profile/verify_wikics_matched54_profile.py experiments_iclr/wikics_matched54_profile experiments_iclr/wikics_matched54`.
On one A100, private-last is slower than ENS in this fixed full-graph profile
(11.133 versus 8.562 ms median), despite nearly equal stored parameters.

## Storage-matched normalized Cora controls

`cora_matched54/` is a separate post hoc comparison on the same normalized
public Cora split that was favorable in the feature-sensitivity study. BASE
width 184, ENS width 70, and private-last width 128 store 605,919, 602,868,
and 604,700 trainable parameters. All 54 validation cells and 18 allowed
selected/default exact float32 pooled test logits are retained. Selected
means are 70.267, 72.867, and 74.500%, with private-last minus ENS +1.633
points across three paired optimizer seeds. Run
`python experiments_iclr/cora_matched54/source/verify_cora_matched54_compact.py experiments_iclr/cora_matched54`.
The sibling `python experiments_iclr/paired_decisions/matched_cora_seed_intervals.py --check`
recomputes a descriptive conditional three-seed t interval. It ignores the
dependence induced by validation selection and says nothing about unseen
graphs or splits. The original model weights and individual member float
logits are omitted from the compact upload.

## Deterministic graph-only link context

`python experiments_iclr/ogbl_collab_topology/verify_public.py` checks every official-pool score and all four metrics for Common Neighbors and Adamic–Adar. The original full graph replay source and audit are preserved. Complete graph regeneration requires the official OGB data and full-study dependencies. The baselines were fixed after the learned outcomes were known, before their own scoring, and both are reported. Test Hits@50 is 41.700% for Common Neighbors and 52.401% for Adamic–Adar, versus the learned tied-model mean of 47.371% in the 400-step recipe.
