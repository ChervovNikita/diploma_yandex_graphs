# Prospective external SAGE controls, protocol v3

Frozen before v3 training or inspection of any selected test score. The v1
trial stopped after the first tied arm on each graph because the paired
post-construction RNG audit failed. The v2 smoke stopped before training
because its 1e-6 CUDA logit tolerance was below measured same-model
reduction drift. Their folders are quarantined and excluded from analysis.

## Public data and fixed sample

Two graphs were chosen before outcomes: WikiCS, from
`https://raw.githubusercontent.com/pmernyei/wiki-cs-dataset/master/dataset/data.json`,
and Actor/Film, from `https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/new_data/film/`
plus `https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/splits/film_split_0.6_0.2_0.npz`.
The precise downloaded bytes and parsed tensors are pinned in each dataset's
source manifest. No public raw data will be redistributed in the anonymous
code bundle.

For WikiCS use published train/validation mask 0 and its fixed test mask.
For Actor use published Geom-GCN train/validation/test mask 0. Optimization
seeds are 0, 1, and 2 on that *same* mask for each dataset. Thus the three
observations are optimization repetitions, not independent graph splits.
Construct undirected edges with PyG `to_undirected`, without explicit self
loops. Use full-batch transductive node classification; features and edges of
all nodes enter message passing, while only train labels enter the loss.

## Model, arms, and selection

Use the original repository `models.TABMModel` SAGE, including its input and
output BatchEnsemble maps and residual SAGE blocks. Match the separate
ogbn-arxiv 300-epoch configuration: four members, two residual layers,
width 128, LayerNorm, dropout 0.2, AdamW at 0.001 with zero weight decay,
300 epochs (no early stop), mean member cross-entropy, and pooled raw logits.
Select one joint checkpoint at each epoch by highest pooled validation
accuracy, lowest pooled validation cross-entropy on an accuracy tie, and
earliest epoch on a complete tie. Restore it before scoring the test mask.
There is no hyperparameter search or outcome-dependent arm choice.

Three arms are fixed: `tied`, `untied_propagation` (four deep-copied residual
stacks, boundary factors and output normalization still shared), and
`all_layer_be` (the tied model plus identity-initialized member factors on
each SAGE neighbor/root map and both FFN linear maps in each residual block).
The paired tied/untied arms run first; the all-layer arm follows if time
allows. Each arm has exactly the same initial tied model state and
post-construction Python, NumPy, CPU Torch, and CUDA Torch RNG state. All
initial member logits are checked. Parameter counts are recorded per arm.

## Numeric gate and provenance

On A100, Torch 2.7.1 and PyG 2.4.0, repeated CUDA forward passes of the
*same* tied WikiCS model differed by at most 2.6226043701171875e-6 and tied
versus initially copied untied by 2.5033950805664062e-6. For Actor the
corresponding maxima were 2.384185791015625e-6 and
2.86102294921875e-6. Actor CPU repeat and tied/untied logits were exactly
equal; the WikiCS CPU gate is recorded separately. Before v3 outcomes we set
the CUDA initial-logit maximum absolute tolerance to 1e-5, above the measured
same-model scatter drift. Canonical initial parameter hashes and CPU/CUDA
RNG hashes must match exactly. This tolerance concerns only initial numeric
equivalence; it does not relax mask, label, source, checkpoint, or test-decision
checks.

The runner pins its own bytes, the original model-definition bytes, this
protocol, all raw data bytes, tensor fingerprints, masks, and the software
versions in `results/<dataset>/source_manifest.json` before training. Each
completed arm has a validation trace, initialization audit, selected
checkpoint, selected validation/test member logits, and artifact hashes.
Incomplete or interrupted arms fail closed. Independent completion checks
must recalculate pooled and member decisions from the selected logits and
replay saved checkpoints before either graph is cited. No partial paired
score is reported.
