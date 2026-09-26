# What a reader can verify from the compact supplement

Run the commands below from the extracted anonymous code-supplement root. They require Python and NumPy and do not need a GPU. Each study supplies raw member logits, pooled logits, or class decisions, as documented below. When only classes are retained, the verifier checks the saved pooled class against labels and run records but cannot independently reconstruct it from the member classes.

The fresh-start preparation command in `fresh_primary/README.md` is separate from score verification. Its `--check-only` mode checks frozen local sources without downloads or writes. Full preparation additionally requires PyTorch and PyG, downloads the public data, and checks the original raw-file and graph-tensor fingerprints. Preparation and CPU preflight were actually tested in a new directory. The supplied scope record does not claim a fresh training run or reproduction of a model's accuracy.

| Study and public command | Checks from included files | Not included in the anonymous compact upload; author-only verification record |
|---|---|---|
| `python experiments_iclr/ogbn_arxiv_sharing/verify_decisions.py` (12 arms) | Complete three-seed/four-arm matrix; frozen source hashes; 300-row validation-checkpoint selection; retained official node-ID fingerprints and labels; selected pooled and member accuracies; equal private-arm parameter counts; paired position differences; stage file hashes. | Public OGB graph bytes, float32 member logits, training checkpoints, and initial-logit arrays. `completion_audit.json` reports the earlier CUDA checkpoint/logit replay; the reader cannot repeat that replay with this compact upload. The OGB dataset must be fetched to rerun training. |
| `python experiments_iclr/new_graph_studies/verify_decisions.py` (48 Cora and legacy-Chameleon arms) | Full two- and five-block tied/untied and two-block position matrices; pretraining freeze chain; frozen source/split fingerprints; 300-row validation selection; retained node IDs, member/pooled class decisions and accuracies; paired contrasts; parameter equality; stage hashes. | Public raw Planetoid/Geom-GCN graph files, float32 selected logits, initial-logit arrays, and checkpoints. The six `completion_audit.json` files record earlier full CUDA replay, which is not reproducible from the compact files. The included runners can retrain after public data download and dependency setup, producing new checkpoints rather than the original ones. |
| `python experiments_iclr/filtered_chameleon_study/verify_decisions.py` (24 arms) | Frozen 24-cell depth/position matrix and source/NPZ hash references; 300-row validation selection; official split-0 node-ID fingerprints; retained decisions and scores; equal partial-arm parameters; paired contrasts; stage hashes. `python experiments_iclr/filtered_chameleon_study/analyze_filtered_validation_choice.py` reproduces the frozen two-arm validation choice and retrospective test regret from included records. | Official filtered NPZ bytes, raw float32 logits, initial-logit arrays, and checkpoints. The NPZ is pinned by public repository commit and SHA-256 in the stage README. `completion_audit.json` records CUDA replay; `MAC_FULL_LOGIT_AUDIT.json` records an independent audit of transferred original logits and NPZ. Neither is a reader-runnable replay from the compact stage. The original logit bundle is held separately in author evidence, outside the anonymous compact upload. |
| `python experiments_iclr/analyze_decision_mechanism.py` (60 depth cells across five graphs) | Recalculates member accuracy, pooled accuracy, pooling gain, pairwise hard-prediction disagreement, and any-member-correct coverage directly from the included compact decision arrays. Its supplied JSON must be reproducible byte-for-byte. | These descriptive statistics inherit the limitation that the saved pooled hard class cannot be regenerated from omitted logits; they do not prove a causal training mechanism or performance on additional graph splits. |

The underlying model runners require PyTorch, PyTorch Geometric, and, for `ogbn-arxiv`, OGB and the public datasets. Their source/protocol files state the training commands and versions used for the author experiments. The compact verification commands above are narrower: they substantiate selected decisions, validation traces, scores, source/data identifiers and hashes retained in the upload. Full checkpoint replay remains author evidence, as stated in each stage README. This distinction also applies to the independent Mac audits, whose JSON summaries are records of work performed on the original files rather than a reconstruction from class-only arrays.

### First-update compact evidence

The `initial_update_diagnostic/verify_compact.py` command checks retained file integrity, the fixed 12-case matrix and the links between source freezes, results, CSVs, raw-array manifest, independent audit records, and figure. Raw gradient/update arrays are omitted to meet the supplementary size limit. This command therefore verifies the retained records and links, rather than rerunning the NumPy gradient-array audit. Source for recreating those arrays is included, with full-study dependencies described in the nested README.

### Roman additional-mask compact evidence

`python experiments_iclr/roman_additional_masks/verify_packed_results.py` verifies and temporarily extracts all 210 original `results/` files, then runs the unchanged verifier. This checks all 48 source/result records, 1,000-row validation traces, retained official mask/label anchors, selected member and pooled hard classes, paired initialization records, replay-audit hashes and 24 paired accuracy differences. An optional `--public-npz` independently anchors the retained labels and masks to public data with the frozen SHA-256. The pooled hard class was derived from original raw logits. It cannot be recalculated from member classes alone. Original full-logit arrays, initial logits and checkpoints remain author evidence. The included CUDA replay summaries record the completed eight setting-level replay checks without making those checks executable from omitted weights.

## Roman bridge upload representation

The anonymous upload retains all nine Roman bridge rows with exact pooled float32 validation/test logits, every member's hard class, and the original node IDs and labels. Run `python roman_bridge/verify_compact_pooled.py`. This recalculates pooled accuracy and cross-entropy, member accuracy, pooling gain, trace selection, and artifact links. It cannot reconstruct the logit average from the omitted individual member float arrays. Those arrays remain in the full author bundle. `verify_full_original.py` preserves the earlier verifier but requires those omitted arrays. Checkpoints are omitted from both compact bundles.

## MC Dropout baseline

`python experiments_iclr/verify_mc_dropout_compact.py` checks all five masks, source and audit hashes, official-label anchors, deterministic BASE classes, four fixed member classes, and both logit-pooled and probability-pooled accuracies. Raw member logits, probabilities and checkpoints are omitted. The CUDA replay and full-array audit JSONs report separate checks on those full artifacts. The compact upload cannot reconstruct a pooled class from member hard classes or recompute cross-entropy.


## Primary selected 72 test logits

`python experiments_iclr/hpo_selected_test_logits/verify_public.py` checks all 72 selected primary models. The original float32 member test arrays are retained without quantization. Their float32 mean reconstructs every original pooled array bitwise. The verifier binds each cell to the complete 432 validation lock and 141-score audit, official test labels/indices, source prediction and score hashes, and independent class exports. It recalculates accuracy and stable cross-entropy. Original pooled validation logits are supplied in the temperature companion. Individual validation member logits and model weights remain author evidence. `python experiments_iclr/validation_tuning/verify_trace_archive.py` checks the complete validation/selection and selected/default decision records.

## Earlier external-SAGE upload representation

`python external_sage/verify_pooled_classes.py` verifies all 18 earlier external-SAGE rows from exact pooled float32 validation/test logits, every member class, and unchanged IDs/labels. It recalculates pooled accuracy and cross-entropy, member accuracy and pooling gain, and checks source, traces, selection and replay records. Individual member float logits remain in the full author bundle. The original `verify_compact.py` requires those omitted arrays. This change applies uniformly to all 18 rows and allows the primary 72 member logits to fit the attachment limit.

## Selected-checkpoint timing records

`python experiments_iclr/inference_profile/verify_inference_profile_public.py` recomputes medians and quartiles from all 7,200 raw timings and checks all 72 allocation blocks, 24 selected checkpoint identities, runtime and original profiler source. Each block's peak is measured in one extra forward after warmup. It is not the maximum over the 100 timed calls. UUID/host/process context are omitted.

## Post hoc probability-quality sensitivity

The `temperature_sensitivity/` stage supplies all 72 validation-logit arrays,
labels, inverse-temperature fits, and unscaled/scaled test metrics. Its
`STAGE_SCOPE.md` gives commands that re-fit every validation temperature and
then regenerate all test rows from the primary member-logit companion. The
validation set was reused after checkpoint and hyperparameter selection.
Test outcomes had already been inspected before this diagnostic was designed.

## Roman narrow-width sensitivity

`python experiments_iclr/roman_narrow/verify_roman_narrow_compact.py --roman24 experiments_iclr/roman_optimizer`
checks all six narrower UNTIED cells against the immutable TIED baseline,
including selection, pooled test logits, member classes, and all paired signs.
Individual member float logits and model checkpoints remain author evidence.
`FRESH_ROMAN_REPRODUCTION.md` explains why the historical overlays need the
original full artifacts and are distinct from compact arithmetic checks.

## Factor placement and same-runtime TIED control

`python experiments_iclr/factor_placement/verify_factor_compact.py --stage experiments_iclr/factor_placement --primary experiments_iclr/validation_tuning`
checks the complete 108-cell validation matrix, frozen source/lock/audit
relationships, 1,000-epoch checkpoint selection, all hard-decision validation
accuracies, and all 36 selected/default pooled test logits, accuracies, and
cross-entropies. The separate
`python experiments_iclr/factor_placement/legacy_tied_default/verify_legacy_default.py --stage experiments_iclr/factor_placement/legacy_tied_default --primary experiments_iclr/validation_tuning`
checks six original TIED default projections used for Actor and filtered
Chameleon comparisons. `python experiments_iclr/factor_placement/provenance/verify_provenance.py experiments_iclr/factor_placement/provenance`
checks supplemental records and their SHA-256 hashes. The full-to-compact
audit was run on every original validation/test array before packaging; the
compact reader cannot rerun that audit without the omitted originals. Original
checkpoint weights and member float logits remain author evidence. Thus the
public verifier cannot replay weights, recompute individual member
probabilities, or recover validation cross-entropy from retained hard classes.

## Deterministic graph-only link scores

`python experiments_iclr/ogbl_collab_topology/verify_public.py` recalculates all four strict Hits@50 metrics from eight complete raw official-pool score vectors, including negative thresholds and positive tie counts. It also checks source, frozen fingerprints, and saved full replay-audit bindings. This arithmetic verifier needs NumPy. It does not reconstruct graph neighbor intersections from the omitted 116 MB official data archive. The original runner and independently implemented full-score verifier are included for regeneration with the public OGB dataset.
