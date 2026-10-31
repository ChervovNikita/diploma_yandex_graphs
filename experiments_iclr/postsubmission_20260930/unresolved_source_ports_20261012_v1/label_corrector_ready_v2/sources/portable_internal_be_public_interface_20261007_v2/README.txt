Portable internal BatchEnsemble: complete training and public role conversion
===========================================================================

This v2 successor preserves v1. It adds export.py and train.py to the portable
Session library. No hostname, GPU UUID, author path, private approval record or
server connection is required. Copy this directory to the researcher's machine.
The four factor/model/objective/selection files remain byte-identical to V4/V6;
core/data.py and the two molecular packing functions remain identical to V6.
Recipes preserve model dimensions, eight arms, coefficients, batching, pooling,
initialization, full horizons and own-vs-joint selection semantics.

Preparation checks were static source parsing/byte comparisons only. No code
imports, numerical example, download, data conversion, model, GPU/server work
or TEST scoring was executed. Runtime compatibility and numerical parity remain
unverified pending actual representative checks. Public source preparation is
not byte-identical author execution or a scientific reproduction result.

1. Environment and external native sources
------------------------------------------
Use Python3.11. Recorded author versions: NumPy1.26.4, torch2.1.2+cu118,
torch-geometric2.7.0, ogb1.3.6, torch-scatter2.1.2, torch-sparse0.6.18.
requirements.txt records versions; choose Torch/scatter/sparse wheels matching
the researcher's CPU/CUDA ABI. The converter also uses pandas; its CSV parser
is accepted only if the exact original Collab float32 feature digest matches.
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

Molhiv requires no external native repository. Independently fetch these pinned
public dependencies for WikiCS/Collab (commands below were not run here):

  git clone https://github.com/cornell-zhang/Polynormer external/Polynormer
  git -C external/Polynormer checkout fc8c276c9c5dfbd616d83f65338a3392188a5e08
  git clone https://github.com/GraphPKU/NeuralCommonNeighbor external/NCN
  git -C external/NCN checkout 11d597013750da17ce7468e344bec756a7af39a4

NATIVE_SOURCES.json records exact source file hashes. Only the requested task's
source is loaded. The NCN retained tree has no LICENSE/COPYING; redistribution
permission remains unresolved. No third-party native source is bundled. The
Polynormer retained source relationship also grants no new redistribution right.

2. Public raw-to-role conversion
--------------------------------
Obtain the official public source files independently. The converters read only
explicit local paths and never download. PUBLIC_DATA_PINS.json contains public
URLs/documentation and exact source/member/array hashes reused from the actual
official-source export/audit. These are identity references, not author approval
receipts. There are no private host or acquisition-origin conditions.

WikiCS official raw JSON:
https://github.com/pmernyei/wiki-cs-dataset/raw/master/dataset/data.json

  python export.py wikics --raw-json data.json --output roles/wikics

The public raw JSON and all six projected tensor fingerprints must match.
Features are original float32, with no extra normalization. Topology reproduces
PyG WikiCS(is_undirected=True), then the author's to_undirected ->
remove_self_loops -> add_self_loops operations; split0 VALID is the union of
val_mask and stopping_mask. Public JSON decoding traverses the entire raw label
list; only selected TRAIN/VALID labels become role tensors. No TEST labels enter
the trainer and no TEST metric is computed. This is not OS-level label isolation.

Collab official archive:
https://snap.stanford.edu/ogb/data/linkproppred/collab.zip

  python export.py collab --archive collab.zip --output roles/collab

The complete archive hash and four allowlisted member hashes are checked before
deserializing the exact upstream TRAIN/VALID NumPy pickle dictionaries. No TEST
member is deserialized. Full original ordered positive/year/shared-negative/
feature array digests must match. All event multiplicity is preserved, historical
TRAIN/VALID canonical pair overlap is allowed, and the authentic100000 negative
list includes its one self-pair. Raw edges must equal the TRAIN canonical
multiset with duplicates; no future-pair filtering is performed.

Molhiv official raw/scaffold files:
https://ogb.stanford.edu/docs/graphprop/#ogbg-molhiv

Unpack the official dataset so the supplied directory contains raw/*.csv.gz and
split/scaffold/{train,valid}.csv.gz. No processed PyG tensor or TEST split is used.

  python export.py molhiv --raw-root ogbg_molhiv --output roles/molhiv

All eight original raw/split file hashes must match. Frozen numeric_rows and
pack_molecules functions preserve original role order, categorical atom/bond
fields, graph-local edge indices and reciprocal-interleaved bonds. All public
graph counts are parsed to locate offsets. Mixed label gzip traverses skipped
rows as bytes, but only TRAIN/VALID rows are numerically decoded; no TEST target
values/features are supplied to training. DATA_FORMAT.txt specifies every field.

Every converter writes train.npz, valid.npz and DATA.json after complete role/
domain checks and full ordered NPZ round-trip field equality. Files are hashed
for reproducibility. Input/output serialization bytes need not match author
NPZ/checkpoint bytes. DATA.json records actual successful checks when executed;
it is not a prewritten success receipt or author approval. Fresh output required;
failed conversion is retained with FAILURE.json, never silently repaired/retried.

3. Complete fixed-recipe training
---------------------------------
One explicit task/arm/seed per command; output directories must be fresh.
No epoch-shortening, early stopping, automatic next task, restart or resume mode.
The commands below are recipes for the researcher, not runs performed here.

  python train.py --task molhiv --arm be_init_contrastive --seed 6101 --device cpu \
    --train roles/molhiv/train.npz --valid roles/molhiv/valid.npz --output runs/molhiv_be_init_contrastive6101

  python train.py --task wikics --arm be_init_contrastive --seed 6101 --device cuda:0 \
    --polynormer external/Polynormer/model.py --train roles/wikics/train.npz \
    --valid roles/wikics/valid.npz --output runs/wikics_be_init_contrastive6101

  python train.py --task collab --arm independent4 --seed 6101 --device cuda:0 \
    --ncn-model external/NCN/model.py --ncn-utils external/NCN/utils.py \
    --train roles/collab/train.npz --valid roles/collab/valid.npz --output runs/collab_independent46101

Arms: single,
single_contrastive, independent4, independent4_contrastive, be_unit, be_init,
be_unit_contrastive, be_init_contrastive. Original development seeds6101/6203/6307;
unused confirmation seeds7109/7211/7309/7411/7517 remain separate prospective work.

Full horizons: WikiCS1100 epochs (100 local+1000 global), Collab/Molhiv100 epochs.
Complete TRAIN updates: WikiCS1/epoch, Collab18/epoch, Molhiv258/epoch. Preserve
tails. Every update executes two full own-supervision views/member with original
Adam grouping and rates. Auxiliary targets use at most512 evenly spaced TRAIN
positions, temperature0.2 and fixed alignment/residual weights0.05 each. Independent
members use own sum+M*auxiliary; BE uses own mean+auxiliary. No schedule/tuning.

Every epoch evaluates complete VALID: WikiCS5274 targets, Collab60084 positive
and100000 shared-negative queries, Molhiv4113 graphs. All member and pooled
outputs must be finite before metrics/selection. Metrics are split0 accuracy,
official OGB Hits@50 and official scaffold ROC AUC. Serving is mean probabilities
for WikiCS and mean raw logits for Collab/Molhiv.

Strict selection: a strictly larger complete VALID metric replaces the first
maximum joint bank for every arm except ordinary independent4. Ties keep the
first. Ordinary independent4 selects each own model by its own metric and
assembles selected.pt as an EVALUATION-ONLY mixed-epoch bank. Its pooled scores
never select member states. Packed independent4_contrastive remains one joint
trajectory; its own-best artifacts are diagnostics, not the served candidate.

WikiCS epoch101 calls unchanged core/selection.local_transition. Ordinary
independent4 restores own_local_N model+Adam; every other arm restores the entire
joint selected_local model+all Adam states. Live end-local member/global RNG is
retained rather than restored from selected-local RNG. Best records continue
across both phases. Final own-best bank restores each body's selected local/
global serving flag. The original unused pooled selected_local artifact for
ordinary independent4 is retained but never used by its transition or candidate.

Outputs: RUN.json (source/data/device identities), PROGRESS.json,
VALID_TRACE.json, selected.pt, own/local selector artifacts, OWN_BEST_BANK.json
for ordinary independent4, and COMPLETE.json only after the full horizon.
FAILURE.json preserves caught failures/interruptions and costs; absent COMPLETE
means incomplete, including an unhandled termination. There is no silent retry.
The public wrapper exposes VALID trace files without the author's family score
embargo. It does not implement an automatic24-cell orchestration or TEST scoring.
For matched scientific analysis, keep the predeclared family and all failed/
missing cells and defer outcome interpretation until family closure.

4. Checkpoint and portability limits
------------------------------------
Selected joint checkpoints contain full model/Adam/member streams/Python/NumPy/
torch RNG and selection epoch/metrics. Own selector files contain the own model+
Adam and phase flag. Ordinary independent4 selected.pt contains only the assembled
evaluation bank and its per-body flags; it must never become a joint fit start.
Loading PyTorch pickle requires a trusted checkpoint. No binaries are bundled.
There is no full-fit resume CLI or exact-resume claim: iterator/negative-population
and selector history are not a fully resumable scientific-run state.

Session's library snapshot save/restore remains available as a primitive, with
matching source/recipe/backend checks. It does not bind data origin or preserve
an in-progress epoch iterator/Collab negative draw. example.py remains a tiny
synthetic interface exercise, not full population/resource/quality evidence.

Unavoidable adaptations: configurable CPU/CUDA device; public caller paths and
source/hash identity instead of author operational approvals; task-specific
native loading; new metadata/serialization; full public raw conversion instead
of requiring the private safe role manifest; exposed VALID traces; no author
resource-qualifier/supervisor/cap/admission or queue. Process/resource limits are
the researcher's responsibility. Sparse numerical kernels, hardware, dependency
ABI and import schedule may affect trajectories. Core/recipe parity is a source
fact; wrapper runtime, raw-to-role numeric parity and training-selection parity
remain unverified until representative actual checks.

Source provenance: SOURCE_CHECK.json compares core files, molecular helper bodies,
recipe fields and relevant driver branches to the frozen sources without executing
them. MANIFEST.json seals this successor independently. v1, author guards, source
seals, original studies and every frozen run are preserved unchanged.
