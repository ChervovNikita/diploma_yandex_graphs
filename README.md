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
| `experiments_iclr/iclr2027_compact/` | Compact frozen depth, sharing-position, new-graph, and link evidence |

Each frozen study has its own protocol, source hashes, data identifiers, and interpretation limits. The current general runners are not an exact replay of the historical software environment.

## What the completed experiments show

The archived comparison covers eight backbones and five graph datasets. GNNM has a higher recorded mean than ENS in 28 of 40 backbone–graph cells. Those cells reuse five graphs. Archived ENS selected member checkpoints separately, while GNNM selected a pooled checkpoint, and their projector initializations differ. These descriptive results do not isolate parameter sharing.

New experiments copy initial propagation parameters and match random-generator states between tied and untied arms. The study-specific protocols record numerical initial-logit checks. Both arms use the same mean member loss and pooled-validation checkpoint rule. Untying also increases capacity. The first-private and last-private comparisons instead keep parameter count equal.

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
- A matched Roman GAT pair is nearly neutral: tied minus untied is +0.039 points, with two positive and three negative masks. This does not explain the archived GAT deficit against ENS.
- On `ogbn-arxiv`, BASE/ENS/GNNM give 70.639/71.096/69.027% at 300 epochs and 71.500/72.143/70.824% at 1,000. GNNM trails ENS on every seed. Several selected checkpoints remain near the cap. A separate initially matched 300-epoch tying pair gives −1.562 points, all three seeds negative.
- On `ogbl-collab`, a frozen link recipe gives tied/untied/ENS/BASE Hits@50 of 47.371/46.872/44.596/38.549%. Tied minus untied has two positive seeds and one negative seed. This is one untuned recipe, not a ranking against specialized link predictors.
- On the selected Roman setting, a parameter-matched explicit ensemble reaches 88.143% with 6.908 million parameters versus GNNM's 89.502% with 6.738 million. Its measured full-graph inference is faster on one A100: 39.9 versus 109.7 ms. Storage and runtime are different costs.

The member analysis separates mean member accuracy from the extra accuracy obtained by pooling logits. On Cora, tied members disagree more while predicting less accurately. On filtered Chameleon, disagreement and accuracy are both lower. These are descriptions of selected predictions, not proof that disagreement causes the result.

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

From `experiments_iclr/iclr2027_compact/`, use Python with NumPy:

```bash
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
```

These commands check the retained records, validation traces, hashes, selected decisions, and score arithmetic. Where only hard classes are retained, they cannot regenerate pooled classes from member classes, reconstruct raw logits, or replay omitted checkpoints. Full checkpoint/logit replays were performed before compaction, and the included audit reports are records of those checks. See `experiments_iclr/NEW_STUDIES_README.md` for every study's scope and dependencies.

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
