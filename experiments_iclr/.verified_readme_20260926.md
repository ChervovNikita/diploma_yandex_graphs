# GNNM: shared propagation in graph ensembles

GNNM makes four predictions for each node. Each member changes the input and output maps through BatchEnsemble factors, while all members reuse the same stored graph-layer parameters. Their hidden states remain separate throughout message passing. Training minimizes the mean member loss. Inference averages **raw logits**, then chooses the largest class logit or uses the binary logit as a ranking score.

```text
node features and graph edges
             |
  four member input projectors
             |
  four separate hidden-state paths
  using shared graph-layer weights
             |
  four member output projectors
             |
       average raw logits
```

Sharing weights reduces trainable-parameter storage. It still requires four propagation paths and does not imply lower latency. The method uses uniform averaging, without an expert-routing gate.

BatchEnsemble is prior work ([Wen et al., 2020](https://arxiv.org/abs/2002.06715)). [Kim (2023)](https://koasas.kaist.ac.kr/handle/10203/308201) previously placed its factors inside GNN layers. This repository studies factors at both boundary projectors, tied propagation weights, partial sharing, and the limits of these constructions. It does not claim the first graph ensemble or the first use of BatchEnsemble in GNNs.

## Repository contents

| Location | Contents |
| --- | --- |
| `models.py`, `datasets.py` | Residual backbones, projectors, graph loading, and official masks |
| `run_base.py`, `run_base_ensemble.py`, `run_tabm.py` | New ordinary-model, explicit-ensemble, and GNNM training runs |
| `results/`, `results2/`, `results3/` | Archived benchmark CSVs |
| `experiments_iclr/recomputed_main/` | Reconstructed tables, complete source manifest, and selection sensitivities |
| `experiments_iclr/` | Additional training, analysis, and author-artifact verification scripts |
| `external_sage/`, `roman_bridge/` | Frozen compact three-arm studies with selected member logits |
| `experiments_iclr/iclr2027_compact/` | Anonymous-size compact records for completed ICLR extension studies, with public verification scripts |

Each frozen study has its own protocol, source hashes, data identifiers, and interpretation limits. The current general runners are not an exact replay of the historical software environment.

## What the completed experiments show

The archived comparison covers eight backbones and five graph datasets. GNNM has a higher recorded mean than ENS in 28 of 40 backbone–graph cells. Those cells reuse five graphs. Archived ENS selected member checkpoints separately, while GNNM selected a pooled checkpoint, and their projector initializations differ. These descriptive results do not isolate parameter sharing.

New experiments copy initial propagation parameters and match random-generator states between tied and untied arms. The study-specific protocols record numerical initial-logit checks. Both arms use the same mean member loss and pooled-validation checkpoint rule. Untying also increases capacity. The first-private and last-private comparisons instead keep parameter count equal.

### Frozen validation grid and measured costs

The primary extension fixes four graph settings, six arms, six learning-rate/weight-decay candidates, and three optimizer seeds: 432 validation-only training cells, each with up to 1,000 epochs. A complete validation lock precedes the selected/default test scores. The table gives the mean selected test accuracy in percent, over three optimizer seeds on one published split per graph. Selection uses validation accuracy, validation cross-entropy, and the stated tie rule; no test score selects a candidate.

| Graph | BASE | ENS | TIED GNNM | Private first | Private last | UNTIED |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 71.167 | 73.600 | 67.700 | 73.667 | 70.833 | 73.633 |
| WikiCS | 77.977 | 78.867 | 79.038 | 78.610 | 79.608 | 79.043 |
| Actor | 37.193 | 37.632 | 36.469 | 37.193 | 37.522 | 36.447 |
| Filtered Chameleon | 37.113 | 41.924 | 37.113 | 39.347 | 38.832 | 36.082 |

The selected partial family compares two equal-parameter placements, but choosing one searches twelve configurations across those arms versus six per individual comparator. All 432 result records and validation traces, the 141 unique allowed selected/default score records, the final score audit, and the selected hard decisions are in `validation_tuning/`. The public compact verifier recomputes selection and accuracy. It cannot replay omitted weights or float logits.

Storage, time, and search cost differ. The saved `training_seconds` values sum to 15,367 seconds (4.27 hours) across all 432 fits; this excludes data loading, score audits, and queue time. On Cora, selected TIED has 357,020 trainable parameters versus 1,399,324 for ENS. Its selected three-seed training records sum to 83.9 seconds versus 79.5 seconds for ENS; this is for those three fits, not the full six-candidate search. The separately profiled seed-0 full-graph inference medians are 4.193 versus 3.775 ms on one A100. The profile retains all 7,200 synchronized timings for 24 selected seed-0 checkpoints, plus peak allocation records; it measures inference, not training memory. Reducing stored weights does not make message passing free.

### Depth and sharing position

Numbers below are mean tied-minus-untied test-accuracy differences in percentage points, under the fixed width-128, 300-epoch SAGE recipes.

| Graph / edge convention | Depth 2 | Depth 5 | Scope |
| --- | ---: | ---: | --- |
| Roman Empire, one added loop per node | −0.738 | +0.709 | One split, five optimizer seeds |
| Roman Empire, no added loops | −0.429 | +0.724 | One split, three seeds |
| WikiCS | −0.011 | −0.103 | Published split 0, three seeds |
| Actor | −0.088 | −0.789 | Published split 0, three seeds |
| Cora | −4.900 | −2.967 | Planetoid public split, three seeds |
| Legacy Chameleon | −0.731 | −1.608 | Published split 0, three seeds |
| Official filtered Chameleon | −2.577 | −2.921 | Published split 0, three seeds |

The full Roman depth grid gives −0.738, +0.508, +0.720, and +0.709 points at depths 2–5. All selected epochs are 275–300. A separately frozen 1,000-epoch endpoint repeat gives −0.765 and +0.382 points, with one negative depth-five seed. The grid was chosen after endpoint pilots. It does not establish a general depth threshold or convergence. Added self-loops alone do not explain the Roman reversal, but the external graphs do not reproduce it under these recipes.

Private-first minus private-last accuracy is +0.540 points on Roman, −0.844 on WikiCS, −0.877 on Actor, +0.467 on `ogbn-arxiv`, +0.367 on Cora, +0.292 on legacy Chameleon, and +0.172 on filtered Chameleon. The two partial arms have equal parameter counts within each graph. The last three have mixed seed signs. A label-count heuristic inferred from the earlier settings fails on all three additions.

The frozen validation rule chooses private-last on Cora. This misses the better partial test mean by 0.367 points, yet still exceeds fully untied propagation by 0.300 points with 29.06% fewer stored parameters. The selected WikiCS arm gives +0.399 points with 35.22% fewer parameters. The other settings trade fewer parameters for lower accuracy, and Actor's selected partial arm also loses to full tying. The complete seven-setting analysis includes every arm and the cost of training both partial choices.

Legacy Chameleon has known duplicate-node evaluation problems. Its filtered variant changes both nodes and the 50 raw self-loops retained by the legacy graph. The two are related variants, and their difference cannot isolate duplicate removal. The filtered study uses symmetrized edges, unlike the directed convention of the original benchmark, so its scores should not be compared directly to that leaderboard.

### Other backbones, tasks, and resource measurements

- A selected width-512 Roman SAGE component study gives 89.502% tied accuracy and 86.562% with initially copied untied propagation, across five overlapping masks with one seed per mask. This local +2.940-point difference is positive on every mask. It does not identify the training mechanism.
- A fresh width-512 Roman mask-0 SAGE repeat across three optimization seeds gives 89.358% tied and 86.028% untied accuracy. Paired differences are +3.600, +3.777, and +2.612 points. Full checkpoint replay and an independent score/provenance audit passed. This remains a post hoc check on one favorable graph.
- A matched Roman GAT pair is nearly neutral: tied minus untied is +0.039 points, with two positive and three negative masks. This does not explain the archived GAT deficit against ENS.
- On `ogbn-arxiv`, BASE/ENS/GNNM give 70.639/71.096/69.027% at 300 epochs and 71.500/72.143/70.824% at 1,000. GNNM trails ENS on every seed. Several selected checkpoints remain near the cap. A separate initially matched 300-epoch tying pair gives −1.562 points, all three seeds negative.
- On `ogbl-collab`, a frozen link recipe gives tied/untied/ENS/BASE Hits@50 of 47.371/46.872/44.596/38.549%. Tied minus untied has two positive seeds and one negative seed. This is one untuned recipe, not a ranking against specialized link predictors.
- On the selected Roman setting, a parameter-matched explicit ensemble reaches 88.143% with 6.908 million parameters versus GNNM's 89.502% with 6.738 million. Its measured full-graph inference is faster on one A100: 39.9 versus 109.7 ms. Storage and runtime are different costs.

The member analysis separates mean member accuracy from the extra accuracy obtained by pooling logits. On Cora, tied members disagree more while predicting less accurately. On filtered Chameleon, disagreement and accuracy are both lower. These are descriptions of selected predictions, not proof that disagreement causes the result.

### Optimizer and factor diagnostics

The tied graph stack updates with AdamW on the mean member gradient. Four private stacks with separate AdamW moments can be synchronized after each step and then collapsed to the same inference function class as TIED. The 96-cell SYNC grid and separate 12-cell norm-matched extension include both default and validation-selected comparisons. Their results do not show a consistent accuracy improvement. The first-update audit reports graph-update cosine 0.641–0.857 and SYNC/TIED norm ratio 0.517–0.784 across twelve initial cases. In twelve cross-member gradient Gram matrices, seven of 72 member pairs have negative inner products, all on WikiCS; the total graph direction still descends for each member at these initial points. These first-step calculations do not establish an AdamW trajectory or explain held-out accuracy.

On Roman Empire mask 0, the 24-cell, 1,000-epoch depth-two/depth-five optimizer comparison is post hoc. A separate parameter-count sensitivity narrows UNTIED to width 68 at depth two and width 65 at depth five, with six completed cells. Width changes the initial function, so it does not isolate parameter count. Standard scalar temperature fitting covers all 72 primary validation-selected models; it changes probability scores, not class predictions, and reuses validation nodes after model selection. These studies retain their complete outcomes and numerical amendment chronology in the compact evidence.

An additional post hoc factor-placement study trains 72 all-layer-factor cells and 36 same-runtime TIED control cells. The selected all-layer minus TIED mean differences are +2.233, −0.239, −0.132, and +1.890 percentage points on Cora, WikiCS, Actor, and filtered Chameleon. The default differences are −0.633, −0.257, −0.066, and +0.172 points. It adds parameters and does not identify an isolated effect of factor placement. The compact package retains all 108 validation records and 36 selected/default test records, with exact pooled test logits, original source hashes, and independent audits.

For `ogbl-collab`, the separately fixed, graph-only Common Neighbors and Adamic–Adar baselines obtain 41.700% and 52.401% strict test Hits@50 under the official negative pool. Adamic–Adar exceeds the learned TIED mean of 47.371% in the 400-step recipe. These topology scores were computed after the learned-link outcomes were known; neither baseline is a capacity-matched learned model. They help bound the interpretation of the link result.

## Recalculate the archived tables

Use the repository root as the working directory:

```bash
python experiments_iclr/reproduce_main.py
python experiments_iclr/reproduce_cost.py
python experiments_iclr/selection_sensitivity.py
python experiments_iclr/tag_sensitivity.py
```

The main reconstruction fixes width 512 and learning rate 3e-5, then selects depth 1–5 by validation separately on each official split. Its retrospective tie rule treats validation values within 1e-12 as tied and prefers shallower depth. It uses the unique complete fixed-grid folder for each graph, based on setup and completeness rather than test scores. Every selected input CSV is hashed in `source_manifest.json`.

This coherent manifest matches 102 of 105 earlier printed seven-backbone cells to two decimals. Three Tolokers cells differ slightly, as documented in the reconstruction. Historical launch-time source hashes, exact software versions, and exact Amazon data bytes were not recovered. Reconstructing CSV arithmetic does not recreate that missing provenance.

The archived cost profiler used 50 warmups and 200 synchronized timed training steps. Its AdamW weight decay was 0.01, while archived predictive runs used zero. The measured step times are distinct from whole-training time and inference latency.

## Verify the compact extension evidence

Use the compact root as the working directory, with Python and NumPy. This
namespace is `experiments_iclr/iclr2027_compact/` in the public repository:

```bash
cd experiments_iclr/iclr2027_compact
python experiments_iclr/validation_tuning/verify_compact_tuning.py
python experiments_iclr/hpo_selected_test_logits/verify_public.py
python experiments_iclr/inference_profile/verify_inference_profile_public.py
python experiments_iclr/optimizer_history/verify_compact_optimizer_v2.py
python experiments_iclr/initial_member_gram/verify_gram.py
python experiments_iclr/roman_optimizer/verify_roman_mechanism_compact.py
python experiments_iclr/roman_narrow/verify_roman_narrow_compact.py --roman24 experiments_iclr/roman_optimizer
python experiments_iclr/factor_placement/verify_factor_compact.py --stage experiments_iclr/factor_placement --primary experiments_iclr/validation_tuning
python experiments_iclr/factor_placement/legacy_tied_default/verify_legacy_default.py --stage experiments_iclr/factor_placement/legacy_tied_default --primary experiments_iclr/validation_tuning
python experiments_iclr/ogbl_collab_topology/verify_public.py
python experiments_iclr/verify_new_compact.py
python experiments_iclr/verify_roman_budget1000_compact.py
python experiments_iclr/roman_noloop_depth/verify_compact.py
python experiments_iclr/external_depth_sage/verify_compact.py
python experiments_iclr/sharing_position/verify_decisions.py
python experiments_iclr/ogbn_arxiv_sharing/verify_decisions.py
python experiments_iclr/new_graph_studies/verify_decisions.py
python experiments_iclr/filtered_chameleon_study/verify_decisions.py
python experiments_iclr/analyze_decision_mechanism.py
python experiments_iclr/analyze_selected_sharing_tradeoffs.py
python experiments_iclr/verify_ogb1000_compact.py
python experiments_iclr/fixed_mask_v4/verify_fixed_mask_v4_compact.py
```

These commands check retained records, validation traces, hashes, selected decisions, and the score arithmetic each package supports. The factor and HPO test-logit stages retain exact pooled or member float32 logits and permit stronger test-score checks than class-only stages. Where only hard classes are retained, a verifier cannot regenerate pooled classes from member classes, reconstruct raw logits, or replay omitted checkpoints. Full checkpoint/logit replays were performed before compaction; their included audit reports record those checks, and are not an executable replay from omitted files. The temperature stage has a separate validation-refit and test-recalculation sequence in `experiments_iclr/temperature_sensitivity/STAGE_SCOPE.md`. See `experiments_iclr/NEW_STUDIES_README.md` and `experiments_iclr/VERIFICATION_SCOPE.md` for every study's exact scope and dependencies.

The two earlier compact stages retain selected member logits and support pooled-logit score recalculation from the repository root:

```bash
python external_sage/verify_compact.py
python roman_bridge/verify_compact.py
```

## Run new experiments

The tested original-server environment uses Python 3.11, PyTorch 2.1.2 with CUDA 11.8, PyG 2.7.0, and DGL 2.4.0. Package metadata is in `pyproject.toml`, `uv.lock`, and `environment.yml`. Frozen extensions may record a different runtime, so use their own protocol. No administrator privileges are required.

General runners default to choosing hyperparameters on split 0 and reusing them. For a **new** depth search using the archived selection pattern, pass `--search_each_split --hidden_dim 512 --lr 3e-5 --layers 1 2 3 4 5` to each applicable runner. This matches the stated search dimensions but does not restore the historical runtime.

For fresh component experiments, choose a new empty result directory, because existing complete keys are skipped:

```bash
python experiments_iclr/projector_controls.py --model SAGE \
  --variants gnnm independent_projectors heads_only input_only output_only \
    base gnnm_m1 untied_backbone freeze_output_factors ens_pooled \
  --splits 0 1 2 3 4 --result_root experiments_iclr/new_controls
```

Study-specific full verifiers need their original checkpoints and prediction arrays. They cannot certify a fresh run in a different output directory without adapting its manifest and paths. Avoid running a trainer over packaged selected records expecting it to restore omitted binaries.

## Data and interpretation

The five original node datasets and their official masks come from [Platonov et al.](https://github.com/yandex-research/heterophilous-graphs). New studies pin public graph bytes and parsed tensors in their manifests. Training uses only training-node labels, validation selects checkpoints/configurations, and test labels score the locked selections. Features and graph structure are available transductively where the protocol states this.

Optimizer seeds repeat training on one split. Official masks overlap on a graph. Related graph variants are not independent tasks. This repository therefore reports local comparisons, complete per-seed outcomes, and parameter/time measurements without a universal accuracy, latency, or state-of-the-art claim.

This public repository identifies its contributors. Conference review uses a separately assembled anonymous code archive without Git history or identifying metadata. Ongoing studies are not evidence until their complete audit passes.
