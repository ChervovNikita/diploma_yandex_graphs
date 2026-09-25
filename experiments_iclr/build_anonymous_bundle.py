"""Build and scan a code/data supplement with no repository history or identity."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import os
import subprocess
import sys
import statistics
import tempfile
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED, ZIP_STORED

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments_iclr' / 'gnnm_anonymous_code.zip'
DATASETS = {
    'roman_empire': ROOT / 'results2' / 'roman_empire',
    'amazon_ratings': ROOT / 'results2' / 'amazon_ratings',
    'minesweeper': ROOT / 'results' / 'minesweeper',
    'questions': ROOT / 'results' / 'questions',
    'tolokers': ROOT / 'results3' / 'tolokers',
}
CORE_FILES = [
    'models.py', 'datasets.py', 'run_common.py', 'run_base.py',
    'run_base_ensemble.py', 'run_tabm.py',
    'experiments_iclr/UPSTREAM_DATA_LICENCE.txt',
    'ablation/__init__.py', 'ablation/_runner.py',
    'ablation/models_ablation.py', 'ablation/parity_check.py',
    'ablation/run_cost.py', 'ablation/run_init_ablation.py',
    'ablation/run_k_ablation.py',
    'experiments_iclr/data_manifest.json',
    'experiments_iclr/fetch_datasets.py',
    'experiments_iclr/audit_initialization.py',
    'experiments_iclr/projector_controls.py',
    'experiments_iclr/test_projector_controls.py',
    'experiments_iclr/verify_control_artifacts.py',
    'experiments_iclr/verify_control_inference.py',
    'experiments_iclr/verify_control_inference_protocol.md',
    'experiments_iclr/verify_gnnm_split0.py',
    'experiments_iclr/reproduce_main.py',
    'experiments_iclr/selection_sensitivity.py',
    'experiments_iclr/tag_sensitivity.py',
    'experiments_iclr/reproduce_cost.py',
    'experiments_iclr/reproduce_ablations.py',
    'experiments_iclr/reexport_predictions.py',
    'experiments_iclr/analyze_decisions.py',
    'experiments_iclr/summarize_controls.py',
    'experiments_iclr/profile_strong_base.py',
    'experiments_iclr/summarize_strong_base.py',
    'experiments_iclr/verify_strong_base.py',
]
PARAMETER_MATCHED_FILES = [
    'experiments_iclr/select_parameter_matched_ens.py',
    'experiments_iclr/test_parameter_matched_ens.py',
    'experiments_iclr/verify_parameter_matched_ens.py',
    'experiments_iclr/summarize_parameter_matched_ens.py',
    'experiments_iclr/profile_ensemble_inference.py',
]
OGB_FILES = [
    'experiments_iclr/ogbn_arxiv_pilot.py',
    'experiments_iclr/ogbn_arxiv_diagnostics.py',
    'experiments_iclr/test_ogbn_arxiv_pilot.py',
    'experiments_iclr/verify_ogbn_arxiv_results.py',
    'experiments_iclr/test_verify_ogbn_arxiv_results.py',
    'experiments_iclr/ogbn_arxiv_protocol.md',
]
FIXED_MASK_SEED_FILES = [
    'experiments_iclr/fixed_mask_seed_pair.py',
    'experiments_iclr/verify_fixed_mask_seed_pair.py',
    'experiments_iclr/fixed_mask_seed_pair_protocol.md',
]
AMAZON_SAGE_PAIR_FILES = (
    'experiments_iclr/amazon_sage_tied_untied.py',
    'experiments_iclr/verify_amazon_sage_tied_untied.py',
    'experiments_iclr/amazon_sage_tied_untied_protocol.md',
    'experiments_iclr/run_amazon_sage_tied_untied_once.sh',
)
PREDICTION_SPLITS = {
    'gnnm': tuple(range(5)),
    'ens_pooled': tuple(range(5)),
    'independent_projectors': (0,),
    'untied_backbone': (0,),
    'freeze_output_factors': (0,),
}
CONTROL_VARIANTS = (
    'gnnm', 'independent_projectors', 'heads_only', 'input_only',
    'output_only', 'base', 'gnnm_m1', 'untied_backbone',
    'freeze_output_factors', 'ens_pooled',
)
RECOMPUTED_MAIN_FILES = (
    'ablations_recomputed.csv', 'backbone_paired_effects.csv',
    'cost_table_recomputed.csv', 'dataset_level_effects.csv',
    'per_split_selected.csv', 'table_recomputed.csv',
    'all_eight_candidate_summary.csv',
    'all_eight_candidate_dataset_spread.csv',
    'audit.json', 'source_manifest.json', 'init_audit.json',
)
MAIN_RECONSTRUCTED_FILES = (
    'per_split_selected.csv', 'table_recomputed.csv',
    'backbone_paired_effects.csv', 'dataset_level_effects.csv',
    'all_eight_candidate_summary.csv',
    'all_eight_candidate_dataset_spread.csv',
    'audit.json', 'source_manifest.json',
)
OGB_UNTIED_SOURCE_LOCK_SHA256 = 'cd2da82e80f489c7558284937a62f9e5eab4c349df68e65553a4f7ff3c4df469'
OGB_UNTIED_ARTIFACT_MANIFEST_SHA256 = '78e1aa7af58a343b1cf024413ed1f6f9d4b8081e0209c8dee75b5c8e10808f17'
OGB_UNTIED_V2_SHA256 = '0388198e3aa8b9924762d61c6eeaa92ff49b384fdfa9fa8d97df5ffa50ddf3cb'
OGB_UNTIED_AMENDMENT_SHA256 = '302cd493f69dd9e57d7c39c71e6d02398f15fb656e2eac36036a66e5e8286507'
OGB_UNTIED_SOURCE_FILES = (
    'experiments_iclr/ogbn_arxiv_untied_control.py',
    'experiments_iclr/ogbn_arxiv_untied_protocol.md',
    'experiments_iclr/ogbn_arxiv_untied_source_lock.json',
    'experiments_iclr/verify_ogbn_arxiv_untied.py',
    'experiments_iclr/verify_ogbn_arxiv_untied_v2.py',
    'experiments_iclr/ogbn_arxiv_untied_verification_amendment.md',
)
ABLATION_RESULT_FILES = (
    *(f'results_cost/{name}.csv' for name in
      ('amazon-ratings', 'minesweeper', 'questions', 'roman-empire', 'tolokers')),
    *(f'results_init/roman-empire_{name}.csv' for name in
      ('default', 'ones_all', 'xavier_all')),
    *(f'results_k/roman-empire_k{k}.csv' for k in (2, 4, 8)),
)
ALTERNATE_TOLOKERS_FILES = tuple(
    f'{folder}/tolokers/{name}.csv'
    for folder in ('results', 'results2')
    for name in ('base', 'ensemble', 'tabm')
)
README = """# Anonymous GNNM reproduction code

This archive contains implementation code, three benchmark graph files,
archived per-configuration result CSVs, and scripts that recompute the main
comparison and cost summaries. It contains no Git history or author metadata.

## Environment

Use Python 3.11 and a compatible NVIDIA GPU. Install packages from
`requirements.txt`. These pinned CUDA 11.8 packages define a tested
reproduction environment. The full software environment used for the archived
training runs was not recorded.

## Recompute the archived benchmark tables

From this archive's root, run:

```
python experiments_iclr/reproduce_main.py
python experiments_iclr/reproduce_cost.py
```

The main command also regenerates the 15-row backbone summary and five-row
mask-spread summary used in the appendix:
`experiments_iclr/recomputed_main/all_eight_candidate_summary.csv` and
`experiments_iclr/recomputed_main/all_eight_candidate_dataset_spread.csv`.

The retrospective reconstruction fixes hidden width at 512 and learning rate
at 3e-5, searches depths 1 through 5 separately on each of ten official
splits, selects depth by validation score, and summarizes selected test scores.
The exact input files and their SHA256 hashes are in
`experiments_iclr/recomputed_main/source_manifest.json`. The source rule
selects one complete fixed-grid folder for each dataset using setup and row
coverage only. It was reconstructed retrospectively. Archived ENS selects
each member's checkpoint separately, while GNNM uses a pooled validation
checkpoint. The benchmark comparison includes that difference.
The archived fixed-grid results use a protocol different from the current
default runner settings. To run a new fixed-grid experiment with comparable
settings, pass
`--search_each_split` to each of `run_base.py`, `run_base_ensemble.py`, and
`run_tabm.py`, along with `--hidden_dim 512 --lr 3e-5 --layers 1 2 3 4 5`.
Each variant gets the same depth candidates and evaluation rule.
This does not recreate the historical run environment or provenance.

The manuscript's depth-selection sensitivity uses the same archived grid.
Run `python experiments_iclr/selection_sensitivity.py` to reproduce its three
policies. The included `experiments_iclr/recomputed_main/depth_policy_*.csv`
files and `depth_policy_note.md` give the selections and interpretation limits.
The global and split-0 rules are retrospective checks. Official masks of one
graph overlap, so they do not create independent held-out graph tasks.

`python experiments_iclr/tag_sensitivity.py` independently checks the TAG
rows of the eight-backbone main comparison in
`experiments_iclr/recomputed_main/tag_sensitivity.csv`. TAG is present in
the same archived fixed grid and does not add an independent graph task.

The cost profiler uses 50 warmup and 200 timed CUDA-synchronized training
steps on official split 0. The recomputation script emits the main
eight-backbone average and a `without_TAG` subset for comparison with the
earlier seven-backbone table. Peak memory is peak PyTorch allocated memory.
The archived cost profiler used AdamW's default 0.01 weight decay, while
the archived predictive runs used zero weight decay.

The included public graph files are `data/minesweeper.npz`,
`data/roman_empire.npz`, and `data/tolokers.npz`. The public
`data/amazon_ratings.npz` and `data/questions.npz` are omitted to reduce
archive size. Run `python experiments_iclr/fetch_datasets.py` to fetch
and verify both omitted files from the pinned source commit. The pinned
download URL template, byte sizes, and SHA256 hashes of all five files are
in `experiments_iclr/data_manifest.json`. Recomputing the archived tables
above reads only CSVs and does not require either omitted graph file.
The public graph files come from the Yandex Research heterophilous-graphs
repository. Its MIT license notice is included verbatim in
`experiments_iclr/UPSTREAM_DATA_LICENCE.txt`.

## Additional attribution controls

`experiments_iclr/projector_controls.py` trains a post hoc Roman Empire case
study with a fixed within-study protocol. It fixes five layers, hidden width 512, learning rate 3e-5,
M=4 where applicable, and official splits 0 through 4. It records each
variant's validation-selected checkpoint and evaluates the held-out test split
only afterward. Its optional analysis uses test labels descriptively after
selection and does not tune the model.

To train a new comparison from this extracted archive, choose a new empty
result root. The packaged original result CSV already has all 50 keys, so the
runner skips those keys if its default root is used:

```
python experiments_iclr/projector_controls.py --model SAGE \\
  --variants gnnm independent_projectors heads_only input_only output_only base gnnm_m1 \\
    untied_backbone freeze_output_factors ens_pooled \\
  --splits 0 1 2 3 4 --result_root experiments_iclr/replay_controls
```

This command creates a new run for comparison with the archived rows. It does
not restore omitted historical artifacts in `experiments_iclr/results/`.
The included decision analysis and complete verifiers target that original
result root and its saved artifact hashes. This ZIP includes raw prediction arrays for
all official masks 0–4 of GNNM and pooled ENS, and mask 0 only for
independent projectors, untied backbones, and frozen output factors. The exact
included arrays, their byte sizes, and SHA256 hashes are listed in
`experiments_iclr/prediction_manifest.json`. Summary CSVs retain every
measured control and all five official masks, including controls whose raw
prediction arrays are omitted. Recomputing the complete five-mask diagnostic
summary requires regenerating the omitted arrays. The analysis script reads
every nonempty prediction_file in the 50-row CSV and raises FileNotFoundError
if run directly against this partial archive. On a complete original artifact tree, run
`python experiments_iclr/verify_control_artifacts.py --require-complete --check-checkpoints`
and `python experiments_iclr/verify_control_inference.py --require-complete --threads 2`
before `python experiments_iclr/analyze_decisions.py`. Those checks cannot
verify the compact archive or a separate fresh-root run without the original
omitted checkpoints and predictions. `gnnm_m1` is a one-member BE
reparameterization control, not an ordinary BASE network with identical
initialization.
`untied_backbone` starts with member logits identical to GNNM, then updates
four separately copied propagation stacks. Both BatchEnsemble projectors and
the output normalization remain shared.
`verify_gnnm_split0.py` records a predeclared validation-checkpoint
comparison between the early GNNM split-0 pilot and a fresh rerun. Its audit
is included in `experiments_iclr/results/`.
The original first-stage process loaded `projector_controls.py` before a later
edit confined to the then-unused `ens_pooled` branch. The runner hash in that
audit freezes the source used for the fresh rerun, not the exact bytes already
loaded by the original process. No launch-time source hash was preserved for
the original process. The recorded settings, selected artifacts, first-stage
matrix hash, and rerun audit are available within that provenance limit.
`freeze_output_factors` starts from the same GNNM parameters and logits, but
holds the output member factors at identity and zero while training the shared
output weight matrix. The ordinary input-only control has a different initial
output weight matrix and cannot isolate output-factor learning.

`profile_strong_base.py` measures synchronized training-step time for
GNNM and ordinary SAGE widths. It chooses a wider ordinary SAGE by timing
alone. `summarize_strong_base.py` reports its paired five-mask comparison.

## Statistical interpretation

The five datasets are the independent benchmark units. The eight backbones
within each dataset share the same graph and official splits. The script
records the signs of five dataset-level effects. These selected graphs do not
establish a population-wide advantage. The descriptive 40-cell ranks are not
an independent-task significance test.
"""
PARAMETER_MATCHED_README = """

## Parameter matched ensemble

This optional section is included only when the five-mask parameter matched
ensemble study is complete and used in the manuscript. The CPU count-only
selection in `experiments_iclr/select_parameter_matched_ens.py` predeclared
widths 192, 224, 256, 288, and 320. Its manifest records the chosen width
and every model count before any score was read. The training runner uses the
same masks and pooled validation checkpoint rule as width-512 ENS. The paired
summary and, when available, same-device checkpoint-byte and inference-latency
profiles are in `experiments_iclr/parameter_matched_ens_results/`. Only mask-0 raw predictions
from this additional ensemble are included to limit archive size; all five
run rows and paired scores are included.

For a new width-256 training run, use a separate empty result root so the
packaged five selected rows do not make the runner skip every mask:

```
python experiments_iclr/projector_controls.py --model SAGE \\
  --variants ens_pooled --splits 0 1 2 3 4 --num_layers 5 \\
  --hidden_dim 256 --lr 3e-5 --m 4 --num_steps 5000 \\
  --result_root experiments_iclr/replay_matched_ens
```

The included paired summarizer targets the original result roots and does not
summarize this new root without an explicit root-path adaptation.
"""
OGB_README = """

## Fixed ogbn-arxiv pilot

This optional section is included only when the complete three-seed pilot is
reported in the manuscript. `experiments_iclr/ogbn_arxiv_protocol.md` gives
the official graph and split, fixed settings, and interpretation limits.
The command below starts new runs in an empty result root. Using the packaged
original result root would skip all nine runs because its selected JSON records
are already present. The OGB graph is fetched by its loader into this archive's own
`experiments_iclr/data/` directory. The large graph, checkpoints, and raw
selected prediction arrays are omitted from this ZIP. The result directory
contains all nine selected-run records, epoch traces, graph fingerprint,
summary tables, and descriptive diagnostics. The optional bundle gate first
checks all nine checkpoints and saved prediction arrays with the read-only
CPU verifier. Three seeds share one official graph split and are not
independent graph tasks.

```
python experiments_iclr/ogbn_arxiv_pilot.py --device cuda:0 \\
  --seeds 0 1 2 --variants base ens gnnm \\
  --output-root experiments_iclr/ogbn_arxiv_replay_100
```
"""
OGB_300_README = """

## Fixed 300-epoch ogbn-arxiv sensitivity repeat

This post hoc budget check is separate from the original 100-epoch pilot.
All nine BASE, ENS, and GNNM runs start at epoch 1 with the same official
temporal graph split and exactly 300 training/validation epochs. The selected
checkpoint uses pooled validation accuracy, then pooled validation cross
entropy, then earliest epoch. The two budgets must not be selected or combined
based on their test scores. Three seeds on this graph are optimization
replicates, not three graph tasks. A 300-epoch cap does not establish
convergence. This section is present only after both nine-run artifact gates
pass. The graph, checkpoints, and saved logits are omitted; selected records,
epoch traces, and descriptive summaries are included. The command below
writes a new run into an empty root; the packaged original selected JSON
records would otherwise make the trainer skip all nine runs.

```
python experiments_iclr/ogbn_arxiv_pilot.py --device cuda:0 \\
  --seeds 0 1 2 --variants base ens gnnm \\
  --max-epochs 300 --min-epochs 300 --patience 20 --eval-every 1 \\
  --output-root experiments_iclr/ogbn_arxiv_replay_300
```

On the complete original artifact tree, verify the two budgets separately
with `python experiments_iclr/verify_ogbn_arxiv_results.py --profile pilot-100 --require-complete`
and `python experiments_iclr/verify_ogbn_arxiv_results.py --profile fixed-300 --require-complete`.
Those historical gates require the omitted original checkpoints and logits;
they do not certify fresh-root runs made by the commands above.
"""
OGB_UNTIED_README = """

## Post hoc ogbn-arxiv tied versus untied propagation

This optional control pairs three fresh tied and untied-propagation SAGE runs
on the same official temporal graph split. Each arm trains for exactly 300
epochs from matched initial member functions. The tied arm is a separate rerun,
not the earlier 300-epoch GNNM checkpoint. Untying raises the parameter count
from 189,248 to 684,608, so the paired accuracy contrast cannot separate
capacity from the sharing constraint. The three seeds measure optimization
variation on one split.

The pre-run source lock, exact six-arm compact records, output artifact
manifest, original verifier, corrected v2 verifier, and verification amendment
are included. The original verifier used the wrong initialization field name;
the amendment discloses the sole verification-only correction. The builder
requires the corrected verifier to report complete status for six arms and
three seed pairs after replaying all selected checkpoints on CPU. The graph,
checkpoints, saved logits, and original launch logs are omitted. The complete
historical gate cannot run from this compact archive alone. The frozen
training runner requires an empty canonical result root and original source
lock, so reproducing training as a new study needs a fresh root and source
manifest rather than invoking it over the packaged selected records.
"""
FIXED_MASK_SEED_README = """

## Fixed-mask SAGE optimization-seed pair

This optional post hoc diagnostic uses Roman Empire official mask 0 throughout.
Seeds 0, 1, and 2 each start a fresh GNNM and initially matched untied
propagation run. The six selected result records, validation traces,
initialization checks, source manifest, and completion audit are included.
The independent complete gate reloaded all six selected checkpoints and
recomputed their predictions before packaging. Checkpoints, raw predictions,
and initial-logit arrays are omitted from this ZIP. The source manifest is
preserved as an evidence record: it pins original environment lockfiles that
are not distributed here, so the packaged files alone cannot rerun the
historical completion verifier. Fresh training requires a new frozen source
manifest and the original public graph file. The three seeds measure
optimization variation on one graph and one fixed mask, not across-graph
variation.
"""
AMAZON_SAGE_PAIR_README = """

## Amazon Ratings SAGE tied versus untied propagation

This optional second-graph diagnostic uses official masks 0, 1, and 2 of the
corrected public Amazon Ratings graph. The optimization seed equals each mask
index. Every mask has one fresh four-member GNNM and one initially matched
untied-propagation arm under the Roman-like five-layer, width-512, 3e-5
protocol. The bundle includes the source, six selected result records,
validation traces, initialization audits, source manifest, and complete
checkpoint-replay audit. Checkpoints, raw predictions, initial-logit arrays,
training logs, and the 27.7 MB public graph NPZ are omitted from this compact
archive. The pinned data manifest supplies its public download hash.
The historical complete verifier requires the omitted artifacts; reproducing
training requires the public graph and a fresh empty result root with a newly
frozen source manifest. The three masks are correlated observations from one
graph and do not estimate independent-graph uncertainty.
"""
MC_DROPOUT_README = """

## Inference-only MC Dropout comparator

This optional section is included only when the complete five-mask comparison
is reported in the manuscript. The detailed fixed procedure and provenance
checks are in `experiments_iclr/mc_dropout_results/README.md`. It reuses
each selected width-512 ordinary SAGE checkpoint without retraining or checkpoint reselection. For each official
mask, it averages raw logits from four seeded stochastic full-graph passes.
The included `experiments_iclr/mc_dropout_results/split0.json` through
`split4.json` record the selected BASE row, source and checkpoint hashes,
per-mask metrics, and prediction-array hashes. `summary.json` reports the
five-mask descriptive comparison. The large raw prediction arrays and model
checkpoints are omitted from this ZIP. Rerun the comparator after regenerating
the selected BASE checkpoints:

```
python experiments_iclr/mc_dropout_control.py --device cuda:0
```

The five official masks share one graph. Their paired differences describe this
case study and are not five independent graph tasks.
"""
ALL_LAYER_README = """

## All-layer SAGE adaptation control

The optional all-layer study includes the runner, fixed protocol, CPU tests,
five selected result rows, source and upstream hashes, artifact hashes, and
initialization audit. When it began at 2026-09-25 16:25:06 UTC, its frozen
manifest recorded the GAT study as
"incomplete_or_failed_after_launcher_exit", with no GAT result rows or GAT
artifact hashes observed. Any later GAT rerun is separate from that launch-time
status. The server inclusion gate checks this frozen record and replays all
five selected checkpoints on the official CPU graph against saved test logits.

Checkpoints, raw predictions, and the server queue, launcher, and failed GAT
waiter logs are omitted from this anonymous archive. The full server replay
audit therefore requires those selected binaries and historical log evidence;
it cannot be repeated from the archive alone. Reproducing training requires
a fresh run environment and upstream queue state. The five masks overlap on
one graph and are descriptive.
"""
GAT_PAIR_README = """

## Post hoc GAT tying pair

The optional GAT study includes the fixed launcher source, read-only verifier,
five-mask result matrix, paired decision analysis, and source/input hash
manifests. Its original launcher and verifier require the server queue marker
and selected checkpoints and prediction arrays, which are omitted here.
Regenerate those artifacts before rerunning the verifier or decision analysis.
The chosen backbone and masks were selected after archived results were seen;
the five overlapping masks are descriptive, not independent graph tasks.
"""
DECISION_STRATA_README = """

## Retrospective decision strata

The optional degree and local homophily analysis includes its source,
predeclared protocol, 20 group-by-mask rows, four five-mask summaries, and
input hashes. Local homophily uses true labels of neighboring nodes only after
checkpoint selection. It cannot serve as an inference-time input. The archive
omits most selected raw prediction arrays; regenerate them to rerun this
analysis. The groups and masks are descriptive observations on one graph.
"""
LAYERWISE_SAGE_README = """

## SAGE layerwise member diversity

The optional five-mask, label-free SAGE hidden-state analysis includes source,
protocol, per-layer metrics, selected-artifact hashes, and logit checks.
Selected checkpoints and most raw predictions are omitted. Regenerate those
artifacts before rerunning the analysis; the included JSON records its completed
measurements and provenance.
"""
LAYERWISE_GAT_README = """

## GAT layerwise member diversity

The optional five-mask, label-free GAT hidden-state analysis includes source,
protocol, per-layer metrics, selected-artifact hashes, and logit checks. It
requires the completed GAT tying pair. Selected checkpoints and raw predictions
are omitted; regenerate them before rerunning the analysis.
"""
REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cu118
numpy==1.26.4
pandas==2.2.3
scipy==1.14.1
scikit-learn==1.5.2
torch==2.1.2+cu118
torch-geometric==2.7.0
nvidia-cusparse-cu11==11.7.5.86
dgl @ https://data.dgl.ai/wheels/torch-2.1/cu118/dgl-2.4.0%2Bcu118-cp311-cp311-manylinux1_x86_64.whl
"""
IDENTITY = re.compile(
    rb'\b(?:Nikita|Chervov|Aleksei|Shmelev|Shchur|Alex)\b|'
    rb'/home/[A-Za-z0-9_.-]+|/Users/[A-Za-z0-9_.-]+|'
    rb'github\.com/[A-Za-z0-9_.-]+|ai0001053|ssh-sr003', re.IGNORECASE
)


def zipinfo(name, compressed=True):
    info = ZipInfo(name)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = ZIP_DEFLATED if compressed else ZIP_STORED
    info.external_attr = 0o644 << 16
    return info


def require(paths, context):
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f'{context} is incomplete: {missing}')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_digest(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def stable_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def selected_prediction(row, parent, expected_name):
    """Resolve a CSV-selected array without allowing outside or stale paths."""
    raw = row.get('prediction_file', '')
    relative = Path(raw)
    if not raw or relative.is_absolute() or '..' in relative.parts:
        raise RuntimeError(f'Unsafe or absent selected prediction path: {raw!r}')
    path = ROOT / relative
    if path.name != expected_name or path.parent != parent:
        raise RuntimeError(f'Selected prediction is outside its expected directory: {raw!r}')
    require([path], 'Selected member prediction arrays')
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError(f'Selected prediction resolves outside repository: {raw!r}')
    return path


def validate_main_manifest():
    manifest_path = ROOT / 'experiments_iclr' / 'recomputed_main' / 'source_manifest.json'
    require([manifest_path], 'Archived source manifest')
    entries = json.loads(manifest_path.read_text())['files']
    expected = {str((directory / name).relative_to(ROOT))
                for directory in DATASETS.values()
                for name in ('base.csv', 'ensemble.csv', 'tabm.csv')}
    observed = {entry['path'] for entry in entries}
    if len(entries) != 15 or observed != expected:
        raise RuntimeError('Archived source manifest must list exactly the 15 main CSVs')
    paths = []
    for entry in entries:
        path = ROOT / entry['path']
        require([path], 'Archived main CSVs')
        payload = path.read_bytes()
        if len(payload) != entry['bytes'] or hashlib.sha256(payload).hexdigest() != entry['sha256']:
            raise RuntimeError(f'Archived main CSV hash mismatch: {entry["path"]}')
        paths.append(path)
    return paths


def validate_main_tables():
    result_root = ROOT / 'experiments_iclr' / 'recomputed_main'
    expected_backbones = {'GCN', 'SAGE', 'GAT', 'GAT-sep', 'GT',
                          'GT-sep', 'ResNet', 'TAG'}
    expected_datasets = {'roman-empire', 'amazon-ratings', 'minesweeper',
                         'questions', 'tolokers'}
    expected_variants = {'BASE', 'ENS', 'GNNM'}
    expected_comparisons = {'GNNM-ENS', 'GNNM-BASE', 'ENS-BASE'}
    specs = {
        'per_split_selected.csv': (
            1200, lambda r: (r['dataset'], r['backbone'],
                             r['variant'], int(r['split']))),
        'table_recomputed.csv': (
            120, lambda r: (r['dataset'], r['backbone'], r['variant'])),
        'backbone_paired_effects.csv': (
            120, lambda r: (r['dataset'], r['backbone'], r['comparison'])),
        'dataset_level_effects.csv': (
            15, lambda r: (r['dataset'], r['comparison'])),
    }
    for name, (expected_count, key) in specs.items():
        path = result_root / name
        require([path], 'Eight-backbone main tables')
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        keys = [key(row) for row in rows]
        if len(rows) != expected_count or len(set(keys)) != expected_count:
            raise RuntimeError(f'Incomplete eight-backbone main table: {name}')
        if {r['dataset'] for r in rows} != expected_datasets:
            raise RuntimeError(f'Wrong datasets in eight-backbone table: {name}')
        if name != 'dataset_level_effects.csv':
            if {r['backbone'] for r in rows} != expected_backbones:
                raise RuntimeError(f'Wrong backbones in eight-backbone table: {name}')
        if name in ('per_split_selected.csv', 'table_recomputed.csv'):
            if {r['variant'] for r in rows} != expected_variants:
                raise RuntimeError(f'Wrong methods in eight-backbone table: {name}')
        else:
            if {r['comparison'] for r in rows} != expected_comparisons:
                raise RuntimeError(f'Wrong comparisons in eight-backbone table: {name}')
    audit_path = result_root / 'audit.json'
    require([audit_path], 'Eight-backbone main audit')
    audit = json.loads(audit_path.read_text())
    if (int(audit.get('num_selected_records', -1)) != 1200
            or 'descriptive_40_cell_mean_ranks' not in audit):
        raise RuntimeError('Main audit does not describe eight backbones')
    cost_path = result_root / 'cost_table_recomputed.csv'
    require([cost_path], 'Eight-backbone cost summary')
    with cost_path.open(newline='') as stream:
        cost_rows = list(csv.DictReader(stream))
    main_cost = [r for r in cost_rows if r['included_backbones'] == 'all_eight']
    if (len(main_cost) != 25
            or {r['dataset'] for r in main_cost} != expected_datasets
            or any(int(r['n_configurations']) != 40 for r in main_cost)):
        raise RuntimeError('Main cost rows do not contain 40 configurations')



def validate_main_reconstruction():
    """Regenerate the archived main outputs in a disposable repository directory."""
    script = ROOT / 'experiments_iclr' / 'reproduce_main.py'
    require([script], 'Main table reconstruction source')
    result_root = ROOT / 'experiments_iclr' / 'recomputed_main'
    with tempfile.TemporaryDirectory(prefix='bundle-main-check-',
                                     dir=ROOT / 'experiments_iclr') as scratch:
        result = subprocess.run(
            [sys.executable, str(script), '--output-root', scratch],
            cwd=ROOT, capture_output=True, text=True, timeout=180,
        )
        if result.returncode:
            raise RuntimeError(
                'Main table reconstruction failed: '
                + (result.stderr or result.stdout).strip())
        for name in MAIN_RECONSTRUCTED_FILES:
            accepted = result_root / name
            regenerated = Path(scratch) / name
            require([accepted, regenerated], 'Reconstructed main outputs')
            if accepted.read_bytes() != regenerated.read_bytes():
                raise RuntimeError(f'Stale reconstructed main output: {name}')


def validate_depth_sensitivity():
    result_root = ROOT / 'experiments_iclr' / 'recomputed_main'
    paths = [result_root / name for name in (
        'depth_policy_per_split.csv', 'depth_policy_cell_summary.csv',
        'depth_policy_dataset_effects.csv', 'depth_policy_note.md',
    )]
    require(paths, 'Depth-selection sensitivity reported in the manuscript')
    expected_counts = {
        'depth_policy_per_split.csv': 3 * 5 * 8 * 3 * 10,
        'depth_policy_cell_summary.csv': 3 * 5 * 8 * 3,
        'depth_policy_dataset_effects.csv': 3 * 5 * 3,
    }
    for path in paths:
        if path.suffix == '.csv':
            with path.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != expected_counts[path.name]:
                raise RuntimeError(f'Incomplete depth-selection sensitivity: {path.name}')
    return paths


def validate_tag_sensitivity():
    path = ROOT / 'experiments_iclr' / 'recomputed_main' / 'tag_sensitivity.csv'
    require([path], 'Retrospective TAG sensitivity')
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    datasets = {'roman-empire', 'amazon-ratings', 'minesweeper',
                'questions', 'tolokers'}
    if len(rows) != 5 or {r['dataset'] for r in rows} != datasets:
        raise RuntimeError('TAG sensitivity needs exactly five benchmark datasets')
    for row in rows:
        if row['backbone'] != 'TAG' or int(row['n_splits']) != 10:
            raise RuntimeError('TAG sensitivity has a different backbone or split count')
        base = float(row['base_mean_percent'])
        ens = float(row['ens_mean_percent'])
        gnnm = float(row['gnnm_mean_percent'])
        if (not all(math.isfinite(value) for value in (base, ens, gnnm))
                or not math.isclose(
                    float(row['gnnm_minus_ens_mean_pp']), gnnm - ens,
                    rel_tol=0, abs_tol=1e-6)
                or not math.isclose(
                    float(row['gnnm_minus_base_mean_pp']), gnnm - base,
                    rel_tol=0, abs_tol=1e-6)):
            raise RuntimeError(f'TAG sensitivity summary disagrees: {row["dataset"]}')
    with (ROOT / 'experiments_iclr' / 'recomputed_main' /
          'table_recomputed.csv').open(newline='') as stream:
        main_rows = list(csv.DictReader(stream))
    by_key = {(r['dataset'], r['variant']): r for r in main_rows
              if r['backbone'] == 'TAG'}
    if len(by_key) != 15:
        raise RuntimeError('The main score table omits TAG rows')
    for row in rows:
        for variant, prefix in (('BASE', 'base'), ('ENS', 'ens'),
                                ('GNNM', 'gnnm')):
            main_row = by_key[row['dataset'], variant]
            if (not math.isclose(float(main_row['mean_percent']),
                                 float(row[f'{prefix}_mean_percent']),
                                 rel_tol=0, abs_tol=1e-8)
                    or not math.isclose(
                        float(main_row['population_sd_percent']),
                        float(row[f'{prefix}_population_sd_percent']),
                        rel_tol=0, abs_tol=1e-8)):
                raise RuntimeError(f'TAG main score disagrees: {row["dataset"]} {variant}')
    with (ROOT / 'experiments_iclr' / 'recomputed_main' /
          'backbone_paired_effects.csv').open(newline='') as stream:
        paired_rows = list(csv.DictReader(stream))
    paired = {(r['dataset'], r['comparison']): r for r in paired_rows
              if r['backbone'] == 'TAG'}
    if len(paired) != 15:
        raise RuntimeError('The main paired table omits TAG rows')
    for row in rows:
        checked = paired[row['dataset'], 'GNNM-ENS']
        if (not math.isclose(float(checked['mean_paired_delta_pp']),
                             float(row['gnnm_minus_ens_mean_pp']),
                             rel_tol=0, abs_tol=1e-8)
                or int(checked['split_wins']) !=
                    int(row['gnnm_minus_ens_split_wins'])):
            raise RuntimeError(f'TAG main paired effect disagrees: {row["dataset"]}')
    return path


def validate_controls():
    result_root = ROOT / 'experiments_iclr' / 'results'
    verification_root = ROOT / 'experiments_iclr' / 'gnnm_split0_verification'
    required = [result_root / name for name in (
        'projector_controls.csv', 'control_summary.csv',
        'paired_control_effects.csv', 'decision_analysis.csv',
        'decision_by_k.csv', 'decision_summary.csv',
        'gnnm_split0_verification_audit.json',
    )]
    require(required, 'Five-mask control study')
    with required[0].open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    expected = {('roman-empire', 'SAGE', variant, str(split))
                for variant in CONTROL_VARIANTS for split in range(5)}
    observed = [(r['dataset'], r['model'], r['variant'], r['split']) for r in rows]
    if len(rows) != len(expected) or set(observed) != expected:
        raise RuntimeError('Five-mask control study needs exactly 50 unique rows')
    row_by_key = {(r['variant'], int(r['split'])): r for r in rows}
    for (variant, split), row in row_by_key.items():
        expected_m = 1 if variant in ('base', 'gnnm_m1') else 4
        if (int(row['seed']) != split or int(row['num_layers']) != 5
                or int(row['hidden_dim']) != 512
                or not math.isclose(float(row['lr']), 3e-5,
                                    rel_tol=0, abs_tol=1e-12)
                or int(row['m']) != expected_m or int(row['num_steps']) != 5000
                or not 1 <= int(row['best_step']) <= 5000):
            raise RuntimeError(f'Mixed control protocol: {variant} split {split}')
        for field in ('val_metric', 'test_acc', 'test_loss'):
            if not math.isfinite(float(row[field])):
                raise RuntimeError(f'Nonfinite {field}: {variant} split {split}')
    with (result_root / 'decision_analysis.csv').open(newline='') as stream:
        decision_rows = list(csv.DictReader(stream))
    decision_keys = [(r['dataset'], r['model'], r['variant'], r['split'])
                     for r in decision_rows]
    if len(decision_keys) != len(expected) or set(decision_keys) != expected:
        raise RuntimeError('Decision analysis needs exactly 50 unique control rows')
    for decision in decision_rows:
        source = row_by_key[decision['variant'], int(decision['split'])]
        measured = float(decision['logit_ensemble_acc'])
        selected = float(source['test_acc'])
        if not (math.isfinite(measured) and math.isfinite(selected)) or (
                abs(measured - selected) > 1e-6):
            raise RuntimeError('Decision analysis does not match selected control scores')
    summary_keys = {('roman-empire', 'SAGE', variant)
                    for variant in CONTROL_VARIANTS}
    for name in ('control_summary.csv', 'decision_summary.csv'):
        with (result_root / name).open(newline='') as stream:
            summary = list(csv.DictReader(stream))
        keys = [(r['dataset'], r['model'], r['variant']) for r in summary]
        if len(keys) != len(summary_keys) or set(keys) != summary_keys:
            raise RuntimeError(f'{name} needs exactly ten unique variant rows')
        if any(int(r['n_splits']) != 5 for r in summary):
            raise RuntimeError(f'{name} needs five masks for every variant')
        if name == 'control_summary.csv':
            for row in summary:
                scores = [float(row_by_key[row['variant'], split]['test_acc'])
                          for split in range(5)]
                expected_mean = 100 * sum(scores) / 5
                reported_mean = float(row['mean_test_acc_percent'])
                if not (math.isfinite(expected_mean) and math.isfinite(reported_mean)) or (
                        abs(reported_mean - expected_mean) > 1e-6):
                    raise RuntimeError('Control summary does not match selected control scores')
    with (result_root / 'paired_control_effects.csv').open(newline='') as stream:
        paired_rows = list(csv.DictReader(stream))
    paired_expected = {f'{variant}-gnnm' for variant in CONTROL_VARIANTS
                       if variant != 'gnnm'}
    paired_names = [r['comparison'] for r in paired_rows]
    if len(paired_names) != len(paired_expected) or set(paired_names) != paired_expected:
        raise RuntimeError('Paired control effects need exactly nine comparisons')
    for row in paired_rows:
        variant = row['comparison'].removesuffix('-gnnm')
        effects = [float(row_by_key[variant, split]['test_acc']) -
                   float(row_by_key['gnnm', split]['test_acc'])
                   for split in range(5)]
        expected_mean = 100 * sum(effects) / 5
        reported_mean = float(row['mean_paired_delta_pp'])
        if int(row['n_paired_splits']) != 5 or not (
                math.isfinite(expected_mean) and math.isfinite(reported_mean)) or (
                abs(reported_mean - expected_mean) > 1e-6):
            raise RuntimeError('Paired control effects do not match selected scores')
    with (result_root / 'decision_by_k.csv').open(newline='') as stream:
        k_rows = list(csv.DictReader(stream))
    k_keys = {(r['dataset'], r['model'], r['variant'], r['split'])
              for r in k_rows}
    if k_keys != expected:
        raise RuntimeError('Decision-by-k analysis does not cover every control row')
    audit = json.loads((result_root / 'gnnm_split0_verification_audit.json').read_text())
    adopted = audit.get('adopted')
    if adopted not in ('pilot', 'fresh'):
        raise RuntimeError('GNNM split-0 audit needs a pilot or fresh adoption decision')
    selected_row = row_by_key['gnnm', 0]
    if row_digest(selected_row) != audit.get(f'{adopted}_row_sha256'):
        raise RuntimeError('GNNM split-0 selected row differs from its verification audit')
    runner = ROOT / 'experiments_iclr' / 'projector_controls.py'
    if sha256(runner) != audit.get('runner_sha256'):
        raise RuntimeError('Control runner differs from its verification audit')
    verification_files = [verification_root / name for name in (
        'projector_controls.csv', 'runner.sha256', 'verifier.sha256',
        'matrix.sha256',
        'pilot_archive/pilot_row.json',
        'pilot_archive/projector_controls_before_verification.csv',
    )]
    require(verification_files, 'GNNM split-0 verification evidence')
    if (verification_root / 'runner.sha256').read_text().strip() != audit['runner_sha256']:
        raise RuntimeError('Verification runner manifest differs from audit')
    verifier = ROOT / 'experiments_iclr' / 'verify_gnnm_split0.py'
    if (sha256(verifier) != audit.get('verifier_sha256') or
            (verification_root / 'verifier.sha256').read_text().strip() !=
            audit.get('verifier_sha256')):
        raise RuntimeError('Verification decision script differs from its frozen manifest')
    if (verification_root / 'matrix.sha256').read_text().strip() != audit.get('matrix_sha256'):
        raise RuntimeError('Verification matrix manifest differs from audit')
    if sha256(verification_root / 'pilot_archive' /
              'projector_controls_before_verification.csv') != (
                  verification_root / 'matrix.sha256').read_text().strip():
        raise RuntimeError('Pilot matrix snapshot differs from its manifest')
    with (verification_root / 'projector_controls.csv').open(newline='') as stream:
        fresh_rows = list(csv.DictReader(stream))
    if len(fresh_rows) != 1 or row_digest(fresh_rows[0]) != audit.get('fresh_row_sha256'):
        raise RuntimeError('Fresh verification row differs from its audit')
    pilot_row = json.loads((verification_root / 'pilot_archive' /
                            'pilot_row.json').read_text())
    if row_digest(pilot_row) != audit.get('pilot_row_sha256'):
        raise RuntimeError('Archived pilot row differs from its audit')
    archived_checkpoint = verification_root / 'pilot_archive' / 'pilot_checkpoint.pt'
    archived_prediction = verification_root / 'pilot_archive' / 'pilot_prediction.npz'
    require([archived_checkpoint, archived_prediction],
            'Preserved pilot artifacts')
    if (sha256(archived_checkpoint) != audit.get('pilot_checkpoint_sha256') or
            sha256(archived_prediction) != audit.get('pilot_prediction_sha256')):
        raise RuntimeError('Preserved pilot artifacts differ from verification audit')
    for (variant, split), row in row_by_key.items():
        parent = (verification_root if (variant, split) == ('gnnm', 0)
                  and adopted == 'fresh' else result_root)
        stem = f'roman-empire_SAGE_{variant}_split{split}'
        for field, folder, extension in (
                ('checkpoint', 'checkpoints', '.pt'),
                ('prediction_file', 'predictions', '.npz')):
            relative = Path(row[field])
            if relative.is_absolute() or '..' in relative.parts:
                raise RuntimeError(f'Unsafe {field} path: {row[field]!r}')
            path = ROOT / relative
            if (path.parent != parent / folder or path.name != stem + extension
                    or not path.resolve().is_relative_to(ROOT.resolve())):
                raise RuntimeError(f'Unexpected {field} path: {row[field]!r}')
            require([path], f'Selected {field} artifact')
    predictions = []
    for variant, splits in PREDICTION_SPLITS.items():
        for split in splits:
            parent = (verification_root if (variant, split) == ('gnnm', 0)
                      and adopted == 'fresh' else result_root) / 'predictions'
            path = selected_prediction(
                row_by_key[variant, split], parent,
                f'roman-empire_SAGE_{variant}_split{split}.npz')
            if (variant, split) == ('gnnm', 0) and (
                    sha256(path) != audit.get(f'{adopted}_prediction_sha256')):
                raise RuntimeError('GNNM split-0 prediction differs from verification audit')
            predictions.append(path)
    return required + verification_files, predictions


def validate_strong_base():
    run_read_only_verifier('experiments_iclr/verify_strong_base.py',
                           '--require-complete')
    result_root = ROOT / 'experiments_iclr' / 'strong_base_results'
    required = [result_root / name for name in (
        'strong_base_selection.json', 'projector_controls.csv',
        'strong_base_paired_scores.csv', 'strong_base_comparison.json',
        'strong_base_profile.csv',
    )]
    require(required, 'Timing-matched ordinary SAGE study')
    selection = json.loads(required[0].read_text())
    selected_width = int(selection['selected_width'])
    if selected_width not in (896, 1024, 1152):
        raise RuntimeError('Unexpected timing-selected BASE width')
    measurements = selection['measurements']
    with required[4].open(newline='') as stream:
        profile_rows = list(csv.DictReader(stream))
    expected_profiles = {('TABM_k4', 512), ('BASE', 512),
                         ('BASE', 896), ('BASE', 1024), ('BASE', 1152)}
    profile_keys = [(r['variant'], int(r['hidden_dim'])) for r in profile_rows]
    if (len(profile_rows) != 5 or len(measurements) != 5
            or set(profile_keys) != expected_profiles):
        raise RuntimeError('Timing profile lacks a predeclared candidate')
    for profile in profile_rows:
        matching = [r for r in measurements
                    if (r['variant'], int(r['hidden_dim'])) ==
                    (profile['variant'], int(profile['hidden_dim']))]
        if (len(matching) != 1 or not math.isclose(
                float(profile['mean_step_ms']),
                float(matching[0]['mean_step_ms']), rel_tol=0, abs_tol=1e-6)):
            raise RuntimeError('Timing profile differs from its selection manifest')
    candidates = [r for r in measurements if r['variant'] == 'BASE'
                  and int(r['hidden_dim']) in (896, 1024, 1152)]
    if len(candidates) != 3 or {int(r['hidden_dim']) for r in candidates} != {
            896, 1024, 1152}:
        raise RuntimeError('Timing selection lacks a predeclared BASE candidate')
    target = float(selection['target_step_ms'])
    if not math.isfinite(target) or target <= 0:
        raise RuntimeError('Invalid GNNM timing target')
    winner = min(candidates,
                 key=lambda r: (abs(math.log(float(r['mean_step_ms']) / target)),
                                int(r['hidden_dim'])))
    if int(winner['hidden_dim']) != selected_width:
        raise RuntimeError('Selected BASE width differs from timing-only rule')
    with required[1].open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    filtered = [r for r in rows if r['dataset'] == 'roman-empire'
                and r['model'] == 'SAGE' and r['variant'] == 'base'
                and int(r['hidden_dim']) == selected_width]
    if len(filtered) != 5 or {int(r['split']) for r in filtered} != set(range(5)):
        raise RuntimeError('Timing-matched BASE needs five official masks')
    if len(rows) != 5:
        raise RuntimeError('Unexpected extra rows in timing-matched BASE output')
    by_split = {int(r['split']): r for r in filtered}
    for split, row in by_split.items():
        if (int(row['seed']) != split or int(row['num_layers']) != 5
                or not math.isclose(float(row['lr']), 3e-5,
                                    rel_tol=0, abs_tol=1e-12)
                or int(row['m']) != 1 or int(row['num_steps']) != 5000
                or not 1 <= int(row['best_step']) <= 5000):
            raise RuntimeError(f'Mixed timing-matched BASE protocol: split {split}')
    with (ROOT / 'experiments_iclr' / 'results' /
          'projector_controls.csv').open(newline='') as stream:
        gnnm_rows = [row for row in csv.DictReader(stream)
                     if (row['dataset'], row['model'], row['variant']) ==
                     ('roman-empire', 'SAGE', 'gnnm')]
    if len(gnnm_rows) != 5 or {int(row['split']) for row in gnnm_rows} != set(range(5)):
        raise RuntimeError('Timing-matched BASE comparison needs five selected GNNM rows')
    gnnm_by_split = {int(row['split']): row for row in gnnm_rows}
    with required[2].open(newline='') as stream:
        paired = list(csv.DictReader(stream))
    if len(paired) != 5 or {int(r['split']) for r in paired} != set(range(5)):
        raise RuntimeError('Timing-matched BASE paired scores need five masks')
    for row in paired:
        split = int(row['split'])
        base_score = 100 * float(by_split[split]['test_acc'])
        gnnm_score = 100 * float(gnnm_by_split[split]['test_acc'])
        if any(not math.isclose(float(row[field]), expected,
                                rel_tol=0, abs_tol=1e-6)
               for field, expected in (
                   ('base_test_acc_percent', base_score),
                   ('gnnm_test_acc_percent', gnnm_score),
                   ('base_minus_gnnm_test_pp', base_score - gnnm_score),
               )):
            raise RuntimeError(f'Stale timing-matched BASE paired score: split {split}')
    comparison = json.loads(required[3].read_text())
    mean_score = sum(100 * float(row['test_acc']) for row in filtered) / 5
    gnnm_mean = sum(100 * float(row['test_acc']) for row in gnnm_rows) / 5
    if (int(comparison['base_width_selected_from_timing_only']) != selected_width
            or not math.isclose(float(comparison['base_mean_test_acc_percent']),
                                mean_score, rel_tol=0, abs_tol=1e-6)
            or not math.isclose(float(comparison['gnnm_mean_test_acc_percent']),
                                gnnm_mean, rel_tol=0, abs_tol=1e-6)
            or not math.isclose(float(comparison['base_minus_gnnm_mean_paired_test_pp']),
                                mean_score - gnnm_mean, rel_tol=0, abs_tol=1e-6)):
        raise RuntimeError('Stale timing-matched BASE comparison summary')
    return required


def validate_mc_dropout():
    result_root = ROOT / 'experiments_iclr' / 'mc_dropout_results'
    script = ROOT / 'experiments_iclr' / 'mc_dropout_control.py'
    required = [script, result_root / 'README.md', result_root / 'summary.json'] + [
        result_root / f'split{split}.json' for split in range(5)]
    require(required, 'Five-mask MC Dropout comparison')
    with (ROOT / 'experiments_iclr' / 'results' /
          'projector_controls.csv').open(newline='') as stream:
        base_rows = [r for r in csv.DictReader(stream)
                     if r['dataset'] == 'roman-empire' and r['model'] == 'SAGE'
                     and r['variant'] == 'base']
    if len(base_rows) != 5 or {int(r['split']) for r in base_rows} != set(range(5)):
        raise RuntimeError('MC Dropout requires five selected ordinary BASE rows')
    base_by_split = {int(r['split']): r for r in base_rows}
    expected_code_hashes = {
        'script_sha256': sha256(script),
        'models_sha256': sha256(ROOT / 'models.py'),
        'datasets_sha256': sha256(ROOT / 'datasets.py'),
        'control_runner_sha256': sha256(ROOT / 'experiments_iclr' /
                                         'projector_controls.py'),
        'data_sha256': sha256(ROOT / 'data' / 'roman_empire.npz'),
    }
    records = []
    for split in range(5):
        record = json.loads((result_root / f'split{split}.json').read_text())
        if int(record['split']) != split:
            raise RuntimeError(f'MC Dropout record has wrong split: {split}')
        manifest = record['input_manifest']
        if (manifest.get('row') != base_by_split[split]
                or any(manifest.get(key) != value
                       for key, value in expected_code_hashes.items())
                or record.get('input_fingerprint') != stable_hash(manifest)):
            raise RuntimeError(f'MC Dropout source fingerprint differs: split {split}')
        protocol = manifest['protocol']
        if (protocol.get('dataset') != 'roman-empire'
                or protocol.get('model') != 'SAGE'
                or protocol.get('splits') != list(range(5))
                or int(protocol.get('passes', -1)) != 4):
            raise RuntimeError(f'MC Dropout protocol differs: split {split}')
        checkpoint = ROOT / base_by_split[split]['checkpoint']
        require([checkpoint], 'Selected BASE checkpoint for MC Dropout')
        if sha256(checkpoint) != manifest.get('checkpoint_sha256'):
            raise RuntimeError(f'MC Dropout checkpoint changed: split {split}')
        prediction = result_root / f'split{split}_predictions.npz'
        if (record.get('prediction_file') !=
                str(prediction.relative_to(ROOT))):
            raise RuntimeError(f'MC Dropout prediction path differs: split {split}')
        require([prediction], 'MC Dropout prediction provenance')
        if sha256(prediction) != record.get('prediction_sha256'):
            raise RuntimeError(f'MC Dropout prediction changed: split {split}')
        base_score = float(base_by_split[split]['test_acc'])
        mc_score = float(record['mc_test_metric'])
        delta = float(record['delta_vs_same_checkpoint_pp'])
        if (not all(math.isfinite(value) for value in (base_score, mc_score, delta))
                or not 0 <= mc_score <= 1
                or not math.isclose(delta, 100 * (mc_score - base_score),
                                    rel_tol=0, abs_tol=1e-6)):
            raise RuntimeError(f'MC Dropout scores disagree: split {split}')
        records.append(record)
    summary = json.loads((result_root / 'summary.json').read_text())
    if summary.get('splits') != list(range(5)) or len(summary['per_split']) != 5:
        raise RuntimeError('MC Dropout summary does not cover five masks')
    summary_by_split = {int(r['split']): r for r in summary['per_split']}
    if len(summary_by_split) != 5 or set(summary_by_split) != set(range(5)):
        raise RuntimeError('MC Dropout summary has duplicate or missing masks')
    for split, record in enumerate(records):
        reported = summary_by_split[split]
        for key in ('deterministic_base_test_metric', 'mc_test_metric',
                    'delta_vs_same_checkpoint_pp'):
            if not math.isclose(float(reported[key]), float(record[key]),
                                rel_tol=0, abs_tol=1e-6):
                raise RuntimeError(f'Stale MC Dropout summary: split {split}')
    mean_mc = 100 * sum(float(r['mc_test_metric']) for r in records) / 5
    mean_base = 100 * sum(float(base_by_split[i]['test_acc'])
                          for i in range(5)) / 5
    mean_delta = sum(float(r['delta_vs_same_checkpoint_pp']) for r in records) / 5
    if (not math.isclose(float(summary['mean_mc_test_acc_percent']), mean_mc,
                         rel_tol=0, abs_tol=1e-6)
            or not math.isclose(float(summary['mean_base_test_acc_percent']),
                                mean_base, rel_tol=0, abs_tol=1e-6)
            or not math.isclose(float(summary['mean_paired_delta_pp']),
                                mean_delta, rel_tol=0, abs_tol=1e-6)):
        raise RuntimeError('MC Dropout summary differs from selected run records')
    return required



def audit_file(relative):
    path = Path(relative)
    if not relative or path.is_absolute() or '..' in path.parts:
        raise RuntimeError(f'Unsafe provenance path: {relative!r}')
    actual = (ROOT / path).resolve()
    if not actual.is_relative_to(ROOT.resolve()) or not actual.is_file():
        raise RuntimeError(f'Missing or external provenance file: {relative!r}')
    return actual


def read_complete_csv(path, count):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != count or any(None in row or any(
            value is None for value in row.values()) for row in rows):
        raise RuntimeError(f'Incomplete or malformed result table: {path}')
    return rows


def run_cpu_verifier(relative, *arguments):
    script = audit_file(relative)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    result = subprocess.run(
        [sys.executable, str(script), *arguments], cwd=ROOT,
        env=environment, capture_output=True, text=True, timeout=1200,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f'CPU artifact verifier failed: {relative}: {detail}')


def run_read_only_verifier(relative, *arguments):
    return run_cpu_verifier(relative, *arguments)


def check_record(actual, expected, context):
    if set(actual) != set(expected):
        raise RuntimeError(f'{context} columns differ from verified results')
    for key, value in expected.items():
        observed = actual[key]
        if value is None:
            valid = observed == ''
        elif isinstance(value, bool):
            valid = observed == str(value)
        elif isinstance(value, int):
            valid = observed == str(value)
        elif isinstance(value, float):
            try:
                number = float(observed)
            except (TypeError, ValueError):
                valid = False
            else:
                valid = math.isfinite(number) and math.isclose(
                    number, value, rel_tol=0, abs_tol=1e-8)
        else:
            valid = observed == value
        if not valid:
            raise RuntimeError(f'{context} has stale {key}: {observed!r} != {value!r}')


def check_json(actual, expected, context):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise RuntimeError(f'{context} JSON keys differ from verified results')
        for key, value in expected.items():
            check_json(actual[key], value, f'{context}.{key}')
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise RuntimeError(f'{context} JSON list differs from verified results')
        for index, value in enumerate(expected):
            check_json(actual[index], value, f'{context}[{index}]')
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isfinite(actual) or (
                not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-8)):
            raise RuntimeError(f'{context} differs from verified results')
    elif actual != expected:
        raise RuntimeError(f'{context} differs from verified results')


def validate_ogb_aggregates(root):
    """Compare packaged summaries and diagnostics with verified selected runs."""
    import numpy as np
    script_dir = str(ROOT / 'experiments_iclr')
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from ogbn_arxiv_diagnostics import (analyze_predictions, COUNT_FIELDS,
                                        FIELDS)

    pairs = [(seed, variant) for seed in range(3)
             for variant in ('base', 'ens', 'gnnm')]
    selected = [json.loads((root / f'seed_{seed}' / variant /
                            'selected.json').read_text()) for seed, variant in pairs]
    if any(row['seed'] != seed or row['variant'] != variant
           for row, (seed, variant) in zip(selected, pairs)):
        raise RuntimeError('OGB selected rows differ from canonical seed/variant paths')
    summary_fields = (
        'seed', 'variant', 'selected_epoch', 'valid_accuracy', 'valid_ce',
        'test_accuracy', 'test_ce', 'parameter_count',
        'trainable_parameter_count', 'epochs_run',
        'train_step_seconds_total', 'validation_seconds_total',
        'total_seconds', 'mean_train_step_ms', 'median_train_step_ms',
        'peak_allocated_mib', 'checkpoint_sha256',
    )
    with (root / 'summary.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != summary_fields:
            raise RuntimeError('OGB summary CSV columns differ from selected runs')
        rows = list(reader)
    if len(rows) != len(pairs):
        raise RuntimeError('OGB summary CSV needs nine canonical rows')
    for row, source in zip(rows, selected):
        check_record(row, {key: source[key] for key in summary_fields},
                     'OGB summary CSV')
    keyed = {(row['seed'], row['variant']): row for row in selected}
    variants = {}
    for variant in ('base', 'ens', 'gnnm'):
        scores = [keyed[seed, variant]['test_accuracy'] for seed in range(3)]
        variants[variant] = {
            'num_seeds': 3, 'seeds': [0, 1, 2],
            'test_accuracy_mean': statistics.mean(scores),
            'test_accuracy_sample_sd': statistics.stdev(scores),
        }
    paired = {}
    for left, right in (('gnnm', 'ens'), ('gnnm', 'base'), ('ens', 'base')):
        diffs = [keyed[seed, left]['test_accuracy'] -
                 keyed[seed, right]['test_accuracy'] for seed in range(3)]
        paired[f'{left}_minus_{right}'] = {
            'seeds': [0, 1, 2], 'differences': diffs,
            'mean': statistics.mean(diffs),
            'sample_sd': statistics.stdev(diffs),
        }
    check_json(json.loads((root / 'summary.json').read_text()), {
        'unit_of_replication': 'optimization seed on one official temporal graph split',
        'accuracy_units': 'fraction correct; multiply by 100 for percentage points',
        'variants': variants, 'paired_differences': paired,
    }, 'OGB summary JSON')

    diagnostics = []
    by_count = []
    for seed, variant in pairs:
        with np.load(root / f'seed_{seed}' / variant /
                     'selected_predictions.npz', allow_pickle=False) as archive:
            stats, groups = analyze_predictions(
                archive['test_member_logits'], archive['test_labels'])
        if not math.isclose(stats['pooled_accuracy'],
                            keyed[seed, variant]['test_accuracy'],
                            rel_tol=0, abs_tol=1e-6):
            raise RuntimeError('OGB diagnostics disagree with selected score')
        identity = {'seed': seed, 'variant': variant}
        diagnostics.append({**identity, **stats})
        by_count.extend({**identity, **group} for group in groups)
    for filename, fields, expected_rows in (
        ('diagnostics.csv', FIELDS, diagnostics),
        ('diagnostics_by_correct_count.csv', COUNT_FIELDS, by_count),
    ):
        with (root / filename).open(newline='') as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != fields:
                raise RuntimeError(f'OGB {filename} columns differ from verified predictions')
            rows = list(reader)
        if len(rows) != len(expected_rows):
            raise RuntimeError(f'OGB {filename} row count differs from verified predictions')
        for row, source in zip(rows, expected_rows):
            check_record(row, {key: source[key] for key in fields}, filename)
    diagnostic_variants = {}
    for variant in ('base', 'ens', 'gnnm'):
        rows = [row for row in diagnostics if row['variant'] == variant]
        entry = {'seeds': [row['seed'] for row in rows], 'num_seeds': 3}
        for metric in ('mean_member_accuracy', 'pooled_accuracy',
                       'pool_minus_mean_member_accuracy',
                       'pairwise_prediction_disagreement'):
            values = [row[metric] for row in rows if row[metric] is not None]
            entry[metric + '_mean'] = statistics.mean(values) if values else None
            entry[metric + '_sample_sd'] = (
                statistics.stdev(values) if len(values) > 1 else None)
        diagnostic_variants[variant] = entry
    check_json(json.loads((root / 'diagnostics_summary.json').read_text()), {
        'scope': 'Descriptive diagnostics on the official test nodes after validation-only checkpoint selection',
        'accuracy_units': 'fraction correct',
        'replication_unit': 'optimization seed on one graph and one official split',
        'variants': diagnostic_variants,
    }, 'OGB diagnostics summary JSON')


def validate_ogb(profile, directory):
    root = ROOT / 'experiments_iclr' / directory
    required = [root / name for name in (
        'run_config.json', 'dataset_manifest.json', 'summary.csv',
        'summary.json', 'diagnostics.csv',
        'diagnostics_by_correct_count.csv', 'diagnostics_summary.json',
    )]
    artifacts = []
    for seed in range(3):
        for variant in ('base', 'ens', 'gnnm'):
            run_dir = root / f'seed_{seed}' / variant
            required.extend((run_dir / 'selected.json', run_dir / 'epochs.csv'))
            artifacts.extend((run_dir / 'selected_checkpoint.pt',
                              run_dir / 'selected_predictions.npz'))
    require(required + artifacts, f'{profile} ogbn-arxiv results')
    run_read_only_verifier('experiments_iclr/verify_ogbn_arxiv_results.py',
                           '--profile', profile, '--require-complete')
    validate_ogb_aggregates(root)
    return required



def validate_ogb_untied():
    """Gate the corrected six-arm CPU replay and package only compact records."""
    root = ROOT / 'experiments_iclr' / 'ogbn_arxiv_untied_results'
    source = [ROOT / name for name in OGB_UNTIED_SOURCE_FILES]
    metadata = [root / name for name in (
        'run_config.json', 'dataset_manifest.json', 'summary.csv',
        'summary.json', 'artifact_manifest.json',
    )]
    artifacts = []
    manifest_names = {
        'run_config.json', 'dataset_manifest.json', 'summary.csv',
        'summary.json',
    }
    for seed in range(3):
        seed_name = f'seed_{seed}'
        metadata.append(root / seed_name / 'initialization.json')
        manifest_names.add(f'{seed_name}/initialization.json')
        for arm in ('tied', 'untied_propagation'):
            prefix = f'{seed_name}/{arm}'
            metadata.extend((root / prefix / 'selected.json',
                             root / prefix / 'epochs.csv'))
            artifacts.extend((root / prefix / 'selected_checkpoint.pt',
                              root / prefix / 'selected_predictions.npz'))
            manifest_names.update(f'{prefix}/{name}' for name in (
                'selected.json', 'epochs.csv', 'selected_checkpoint.pt',
                'selected_predictions.npz',
            ))
    require(source + metadata + artifacts, 'Six-arm ogbn-arxiv untied control')
    if any(path.is_symlink() for path in source + metadata + artifacts):
        raise RuntimeError('OGB untied source or artifact is a symlink')
    lock = ROOT / 'experiments_iclr' / 'ogbn_arxiv_untied_source_lock.json'
    manifest = root / 'artifact_manifest.json'
    verifier = ROOT / 'experiments_iclr' / 'verify_ogbn_arxiv_untied_v2.py'
    amendment = ROOT / 'experiments_iclr' / 'ogbn_arxiv_untied_verification_amendment.md'
    for path, expected in (
        (lock, OGB_UNTIED_SOURCE_LOCK_SHA256),
        (manifest, OGB_UNTIED_ARTIFACT_MANIFEST_SHA256),
        (verifier, OGB_UNTIED_V2_SHA256),
        (amendment, OGB_UNTIED_AMENDMENT_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f'Frozen OGB untied file changed: {path.relative_to(ROOT)}')
    hashes = json.loads(manifest.read_text()).get('sha256')
    if not isinstance(hashes, dict) or set(hashes) != manifest_names:
        raise RuntimeError('OGB untied artifact manifest differs from six canonical arms')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')
    result = subprocess.run(
        [sys.executable, str(verifier)], cwd=ROOT, env=environment,
        capture_output=True, text=True, timeout=1200,
    )
    if result.returncode:
        raise RuntimeError('OGB untied v2 CPU verifier failed: '
                           + (result.stderr or result.stdout).strip())
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError('OGB untied v2 verifier did not return JSON') from exc
    if (set(report) != {
            'status', 'verified_arms', 'verified_seed_pairs', 'summary',
            'artifact_manifest_sha256', 'source_lock_sha256',
            'dataset_manifest_sha256',
        } or report['status'] != 'complete'
            or report['verified_arms'] != 6
            or report['verified_seed_pairs'] != 3
            or report['artifact_manifest_sha256'] != OGB_UNTIED_ARTIFACT_MANIFEST_SHA256
            or report['source_lock_sha256'] != OGB_UNTIED_SOURCE_LOCK_SHA256
            or report['dataset_manifest_sha256'] != sha256(root / 'dataset_manifest.json')
            or report['summary'] != json.loads((root / 'summary.json').read_text())):
        raise RuntimeError('OGB untied v2 verifier report differs from six-arm result')
    return source + metadata


def validate_fixed_mask_seeds():
    root = ROOT / 'experiments_iclr' / 'fixed_mask_seed_pair_results'
    source = [ROOT / name for name in FIXED_MASK_SEED_FILES]
    metadata = [root / 'source_manifest.json']
    for seed in range(3):
        for variant in ('gnnm', 'untied_backbone'):
            run_dir = root / f'seed{seed}' / variant
            metadata.extend(run_dir / name for name in (
                'result.json', 'validation_trace.csv', 'initialization.json'))
    require(source + metadata, 'Fixed-mask optimization-seed pairs')
    # The current verifier writes completion_audit.json after six complete CPU
    # checkpoint replays. It is deliberately not described as read-only.
    run_cpu_verifier('experiments_iclr/verify_fixed_mask_seed_pair.py',
                     '--device', 'cpu')
    audit_path = root / 'completion_audit.json'
    require([audit_path], 'Fixed-mask completion audit')
    audit = json.loads(audit_path.read_text())
    if (audit.get('verified_runs') != 6 or audit.get('official_split') != 0
            or audit.get('optimization_seeds') != [0, 1, 2]
            or audit.get('source_manifest_sha256') != sha256(root / 'source_manifest.json')):
        raise RuntimeError('Fixed-mask completion audit differs from six-run protocol')
    deltas = []
    for seed in range(3):
        scores = {}
        for variant in ('gnnm', 'untied_backbone'):
            run_dir = root / f'seed{seed}' / variant
            row = json.loads((run_dir / 'result.json').read_text())
            if (row.get('official_split') != 0 or
                    row.get('optimization_seed') != seed or
                    row.get('variant') != variant or
                    row.get('source_manifest_sha256') != audit['source_manifest_sha256']):
                raise RuntimeError('Fixed-mask selected row differs from its canonical path')
            for name in ('validation_trace.csv', 'initialization.json'):
                if row['artifacts_sha256'][name] != sha256(run_dir / name):
                    raise RuntimeError(f'Fixed-mask packaged artifact changed: {name}')
            scores[variant] = float(row['test_acc'])
        deltas.append(100 * (scores['gnnm'] - scores['untied_backbone']))
    check_json(audit.get('paired_test_accuracy_deltas_percentage_points'),
               deltas, 'Fixed-mask paired deltas')
    check_json(audit.get('mean_paired_delta_percentage_points'),
               statistics.mean(deltas), 'Fixed-mask mean paired delta')
    check_json(audit.get('sd_paired_delta_percentage_points'),
               statistics.stdev(deltas), 'Fixed-mask paired sample SD')
    return source + metadata + [audit_path]


def validate_amazon_sage_pair():
    root = ROOT / 'experiments_iclr' / 'amazon_sage_tied_untied_results'
    source = [ROOT / name for name in AMAZON_SAGE_PAIR_FILES]
    metadata = [root / 'source_manifest.json']
    for mask in range(3):
        for variant in ('gnnm', 'untied_backbone'):
            arm = root / f'mask{mask}' / variant
            metadata.extend(arm / name for name in (
                'result.json', 'validation_trace.csv', 'initialization.json'))
    audit_path = root / 'completion_audit.json'
    require(source + metadata + [audit_path], 'Amazon SAGE tied-versus-untied extension')
    if any(path.is_symlink() for path in source + metadata + [audit_path]):
        raise RuntimeError('Amazon SAGE source or compact artifact is a symlink')
    # The frozen completion audit was emitted only after six checkpoint
    # replays on the training device. Recheck the pinned source and compact
    # records here; omitted checkpoints cannot be replayed from the ZIP.
    run_cpu_verifier('experiments_iclr/verify_amazon_sage_tied_untied.py',
                     '--preflight')
    audit = json.loads(audit_path.read_text())
    manifest = root / 'source_manifest.json'
    if (audit.get('protocol') != 'amazon_sage_tied_untied_masks0_1_2_v1'
            or audit.get('verified_runs') != 6
            or audit.get('official_splits') != [0, 1, 2]
            or audit.get('optimization_seed_rule') != 'seed equals official mask index'
            or audit.get('source_manifest_sha256') != sha256(manifest)):
        raise RuntimeError('Amazon SAGE completion audit differs from the six-arm protocol')
    deltas = []
    for mask in range(3):
        scores = {}
        for variant in ('gnnm', 'untied_backbone'):
            arm = root / f'mask{mask}' / variant
            row = json.loads((arm / 'result.json').read_text())
            if (row.get('protocol') != audit['protocol']
                    or row.get('dataset') != 'amazon-ratings'
                    or row.get('model') != 'SAGE'
                    or row.get('official_split') != mask
                    or row.get('optimization_seed') != mask
                    or row.get('variant') != variant
                    or row.get('source_manifest_sha256') != audit['source_manifest_sha256']):
                raise RuntimeError('Amazon SAGE selected row differs from its canonical path')
            for name in ('validation_trace.csv', 'initialization.json'):
                if row['artifacts_sha256'][name] != sha256(arm / name):
                    raise RuntimeError(f'Amazon SAGE packaged artifact changed: {name}')
            scores[variant] = float(row['test_acc'])
        deltas.append(100 * (scores['gnnm'] - scores['untied_backbone']))
    check_json(audit.get('paired_test_accuracy_deltas_percentage_points'),
               deltas, 'Amazon SAGE paired deltas')
    check_json(audit.get('mean_paired_delta_percentage_points'),
               statistics.mean(deltas), 'Amazon SAGE mean paired delta')
    check_json(audit.get('sd_paired_delta_percentage_points'),
               statistics.stdev(deltas), 'Amazon SAGE paired sample SD')
    return source + metadata + [audit_path]


def validate_all_layer_sage():
    source = [ROOT / name for name in (
        'experiments_iclr/all_layer_be_sage.py',
        'experiments_iclr/test_all_layer_be_sage.py',
        'experiments_iclr/verify_all_layer_be_sage_frozen.py',
        'experiments_iclr/test_verify_all_layer_be_sage_frozen.py',
        'experiments_iclr/all_layer_be_sage_protocol.md',
    )]
    root = ROOT / 'experiments_iclr' / 'all_layer_be_sage_results'
    metadata = [root / name for name in (
        'protocol.json', 'artifact_hashes.json',
        'initialization_audit.json', 'projector_controls.csv',
    )]
    require(source + metadata, 'All-layer SAGE control')
    run_read_only_verifier(
        'experiments_iclr/verify_all_layer_be_sage_frozen.py',
        '--threads', '2')
    rows = read_complete_csv(metadata[-1], 5)
    if {int(row['split']) for row in rows} != set(range(5)) or any(
            (row['dataset'], row['model'], row['variant']) !=
            ('roman-empire', 'SAGE', 'all_layer_be') for row in rows):
        raise RuntimeError('All-layer SAGE needs five selected official masks')
    identity = json.loads((root / 'initialization_audit.json').read_text())
    if len(identity.get('checks', [])) != 5:
        raise RuntimeError('All-layer SAGE initialization audit is incomplete')
    return source + metadata


def validate_gat_pair():
    source = [ROOT / name for name in (
        'experiments_iclr/verify_gat_failure_pair.py',
        'experiments_iclr/gat_decision_pair_analysis.py',
        'experiments_iclr/test_gat_decision_pair_verify_only.py',
        'experiments_iclr/gat_decision_pair_analysis_protocol.md',
        'experiments_iclr/run_gat_failure_pair.sh',
    )]
    root = ROOT / 'experiments_iclr' / 'gat_failure_pair_results'
    metadata = [root / name for name in (
        'protocol.json', 'projector_controls.csv',
        'gat_decision_pair_analysis.csv',
        'gat_decision_pair_analysis_audit.json',
    )]
    require(source + metadata, 'Five-mask GAT tying pair')
    run_read_only_verifier('experiments_iclr/verify_gat_failure_pair.py',
                           '--complete')
    run_read_only_verifier('experiments_iclr/gat_decision_pair_analysis.py',
                           '--verify-only')
    rows = read_complete_csv(metadata[2], 5)
    if {int(row['split']) for row in rows} != set(range(5)) or any(
            (row['dataset'], row['model']) != ('roman-empire', 'GAT')
            for row in rows):
        raise RuntimeError('GAT decision analysis needs five paired masks')
    audit = json.loads(metadata[3].read_text())
    hashes = audit.get('input_sha256')
    if not isinstance(hashes, dict) or len(hashes) < 15:
        raise RuntimeError('GAT analysis input manifest is incomplete')
    for relative, expected in hashes.items():
        if sha256(audit_file(relative)) != expected:
            raise RuntimeError(f'GAT analysis input changed: {relative}')
    selected = read_complete_csv(metadata[1], 10)
    selected_paths = {row['prediction_file'] for row in selected}
    if len(selected_paths) != 10 or not selected_paths.issubset(hashes):
        raise RuntimeError('GAT analysis manifest omits a selected prediction')
    return source + metadata


def validate_decision_strata():
    source = [ROOT / name for name in (
        'experiments_iclr/decision_strata.py',
        'experiments_iclr/decision_strata_protocol.md',
    )]
    root = ROOT / 'experiments_iclr' / 'results'
    metadata = [root / name for name in (
        'decision_strata_per_split.csv',
        'decision_strata_five_mask_summary.csv',
        'decision_strata_audit.json',
    )]
    require(source + metadata, 'Five-mask decision strata')
    rows = read_complete_csv(metadata[0], 20)
    expected = {(split, dimension, stratum)
                for split in range(5)
                for dimension, strata in (
                    ('degree', ('degree_2', 'degree_ge3')),
                    ('label_homophily', (
                        'same_label_neighbors_0',
                        'same_label_neighbors_positive',
                    )),
                ) for stratum in strata}
    observed = [(int(row['split']), row['dimension'], row['stratum'])
                for row in rows]
    if len(set(observed)) != 20 or set(observed) != expected:
        raise RuntimeError('Decision strata have missing or duplicate groups')
    for row in rows:
        n = int(row['n_nodes'])
        both = int(row['both_correct_nodes'])
        gnnm_only = int(row['gnnm_only_correct_nodes'])
        ens_only = int(row['ens_only_correct_nodes'])
        neither = int(row['neither_correct_nodes'])
        if n <= 0 or any(value < 0 for value in (
                both, gnnm_only, ens_only, neither)) or (
                both + gnnm_only + ens_only + neither != n):
            raise RuntimeError('Decision strata paired counts do not partition test nodes')
        for field, expected_value in (
                ('gnnm_pooled_acc_percent', 100 * (both + gnnm_only) / n),
                ('ens_pooled_acc_percent', 100 * (both + ens_only) / n),
                ('gnnm_minus_ens_acc_pp', 100 * (gnnm_only - ens_only) / n)):
            if not math.isclose(float(row[field]), expected_value,
                                rel_tol=0, abs_tol=1e-6):
                raise RuntimeError(f'Decision strata paired metric differs: {field}')
    summaries = read_complete_csv(metadata[1], 4)
    if {(row['dimension'], row['stratum']) for row in summaries} != {
            (dimension, stratum) for _, dimension, stratum in expected} or any(
            int(row['n_masks']) != 5 for row in summaries):
        raise RuntimeError('Decision strata summaries are incomplete')
    metrics = [field for field in rows[0] if field not in (
        'dataset', 'model', 'split', 'dimension', 'stratum')]
    for summary in summaries:
        group = [row for row in rows if (
            row['dimension'], row['stratum']) == (
            summary['dimension'], summary['stratum'])]
        if len(group) != 5 or {int(row['split']) for row in group} != set(range(5)):
            raise RuntimeError('Decision strata summary lacks a complete group')
        for field in metrics:
            values = [float(row[field]) for row in group]
            for prefix, expected_value in (
                    ('mean', sum(values) / 5), ('min', min(values)),
                    ('max', max(values))):
                reported = float(summary[f'{prefix}_{field}'])
                if not math.isfinite(reported) or not math.isclose(
                        reported, expected_value, rel_tol=0, abs_tol=1e-6):
                    raise RuntimeError(f'Stale decision strata summary: {field}')
    audit = json.loads(metadata[2].read_text())
    expected_hashes = {
        'source_sha256': source[0],
        'protocol_sha256': source[1],
        'data_sha256': ROOT / 'data' / 'roman_empire.npz',
        'matrix_sha256': root / 'projector_controls.csv',
        'split0_verification_audit_sha256':
            root / 'gnnm_split0_verification_audit.json',
    }
    if any(audit.get(key) != sha256(path)
           for key, path in expected_hashes.items()):
        raise RuntimeError('Decision strata provenance differs from source')
    selected = audit.get('selected_predictions', [])
    keys = [(entry.get('variant'), int(entry.get('split', -1)))
            for entry in selected]
    if len(keys) != 10 or set(keys) != {
            (variant, split) for variant in ('gnnm', 'ens_pooled')
            for split in range(5)}:
        raise RuntimeError('Decision strata prediction manifest is incomplete')
    matrix_rows = read_complete_csv(root / 'projector_controls.csv', 50)
    matrix = {(row['variant'], int(row['split'])): row
              for row in matrix_rows}
    if any(entry.get('path') !=
           matrix[entry['variant'], int(entry['split'])]['prediction_file']
           or sha256(audit_file(entry['path'])) != entry.get('sha256')
           for entry in selected):
        raise RuntimeError('Decision strata selected prediction changed')
    if (int(audit.get('output_rows_per_split', -1)) != 20 or
            int(audit.get('output_rows_five_mask_summary', -1)) != 4):
        raise RuntimeError('Decision strata audit row counts differ')
    return source + metadata


def validate_layerwise(model):
    if model not in ('SAGE', 'GAT'):
        raise ValueError(model)
    source = [ROOT / name for name in (
        'experiments_iclr/layerwise_member_diversity.py',
        'experiments_iclr/layerwise_member_diversity_protocol.md',
    )]
    output = (ROOT / 'experiments_iclr' /
              'layerwise_member_diversity_results' /
              f'{model.lower()}_all.json')
    require(source + [output], f'Five-mask {model} layerwise diversity')
    result = json.loads(output.read_text())
    runs = result.get('runs', [])
    if (result.get('schema') != 1 or result.get('dataset') != 'roman-empire'
            or result.get('model') != model
            or result.get('splits') != list(range(5))
            or len(runs) != 5
            or [run.get('split') for run in runs] != list(range(5))
            or any(not run.get('stages') or not run.get(
                'prediction_verification') for run in runs)):
        raise RuntimeError(f'{model} layerwise diversity is incomplete')
    hashes = result.get('frozen_sha256')
    if not isinstance(hashes, dict) or len(hashes) < 10:
        raise RuntimeError(f'{model} layerwise source manifest is incomplete')
    for relative, expected in hashes.items():
        if sha256(audit_file(relative)) != expected:
            raise RuntimeError(f'{model} layerwise input changed: {relative}')
    matrix_path = (ROOT / 'experiments_iclr' /
                   ('results' if model == 'SAGE' else
                    'gat_failure_pair_results') / 'projector_controls.csv')
    with matrix_path.open(newline='') as stream:
        selected = [row for row in csv.DictReader(stream)
                    if (row['dataset'], row['model'], row['variant']) ==
                    ('roman-empire', model, 'gnnm')]
    if len(selected) != 5 or {int(row['split']) for row in selected} != set(range(5)):
        raise RuntimeError(f'{model} layerwise GNNM selection is incomplete')
    if any(row[field] not in hashes for row in selected
           for field in ('checkpoint', 'prediction_file')):
        raise RuntimeError(f'{model} layerwise manifest omits selected artifacts')
    return source + [output]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-ogb', action='store_true',
                        help='Include completed ogbn-arxiv pilot source and results only when the manuscript reports it')
    parser.add_argument('--include-ogb-300', action='store_true',
                        help='Include the separately verified 300-epoch ogbn-arxiv repeat, together with the 100-epoch pilot')
    parser.add_argument('--include-ogb-untied', action='store_true',
                        help='Include only the amended and fully replayed six-arm 300-epoch tying control')
    parser.add_argument('--include-parameter-matched-ens', action='store_true',
                        help='Include completed five-mask parameter matched ENS source and results only when the manuscript reports it')
    parser.add_argument('--include-mc-dropout', action='store_true',
                        help='Include completed five-mask inference-only MC Dropout source and metadata only when the manuscript reports it')
    parser.add_argument('--include-all-layer-sage', action='store_true',
                        help='Include only the audited five-mask all-layer SAGE study')
    parser.add_argument('--include-gat-pair', action='store_true',
                        help='Include only the audited five-mask GAT tying study and decision analysis')
    parser.add_argument('--include-decision-strata', action='store_true',
                        help='Include only complete five-mask decision strata metadata')
    parser.add_argument('--include-layerwise-sage', action='store_true',
                        help='Include only complete SAGE layerwise diversity metadata')
    parser.add_argument('--include-layerwise-gat', action='store_true',
                        help='Include only complete GAT layerwise diversity metadata')
    parser.add_argument('--include-fixed-mask-seeds', action='store_true',
                        help='Include only the six fully verified mask-0 SAGE optimization-seed pairs')
    parser.add_argument('--include-amazon-sage-pair', action='store_true',
                        help='Include only the complete, independently replayed six-arm Amazon SAGE extension')
    args = parser.parse_args()
    if args.include_ogb_300 and not args.include_ogb:
        parser.error('--include-ogb-300 requires --include-ogb')
    if args.include_ogb_untied and not args.include_ogb_300:
        parser.error('--include-ogb-untied requires --include-ogb-300')
    if args.include_layerwise_gat and not args.include_gat_pair:
        parser.error('--include-layerwise-gat requires --include-gat-pair')
    files = {ROOT / path for path in CORE_FILES}
    files.update(validate_main_manifest())
    validate_main_tables()
    validate_main_reconstruction()
    files.update(validate_depth_sensitivity())
    files.add(validate_tag_sensitivity())
    control_required, prediction_required = validate_controls()
    run_read_only_verifier('experiments_iclr/verify_control_artifacts.py',
                           '--require-complete', '--check-checkpoints')
    run_read_only_verifier('experiments_iclr/verify_control_inference.py',
                           '--require-complete', '--threads', '2')
    strong_required = validate_strong_base()
    files.update(control_required)
    files.update(prediction_required)
    files.update(strong_required)
    ablation_results = [ROOT / 'ablation' / name for name in ABLATION_RESULT_FILES]
    require(ablation_results, 'Archived ablation results')
    files.update(ablation_results)
    alternates = [ROOT / name for name in ALTERNATE_TOLOKERS_FILES]
    require(alternates, 'Archived alternate Tolokers result folders')
    files.update(alternates)
    derived = [ROOT / 'experiments_iclr' / 'recomputed_main' / name
               for name in RECOMPUTED_MAIN_FILES]
    require(derived, 'Recomputed main and ablation outputs')
    files.update(derived)
    data_manifest = json.loads((ROOT / 'experiments_iclr' / 'data_manifest.json').read_text())
    for name in ('minesweeper.npz', 'roman_empire.npz', 'tolokers.npz'):
        item = ROOT / 'data' / name
        require([item], 'Included public graph files')
        entry = data_manifest['files'][name]
        if item.stat().st_size != entry['bytes'] or sha256(item) != entry['sha256']:
            raise RuntimeError(f'Included public graph hash mismatch: {name}')
        files.add(item)
    if args.include_parameter_matched_ens:
        run_read_only_verifier('experiments_iclr/verify_parameter_matched_ens.py',
                               '--require-complete')
        matched_root = ROOT / 'experiments_iclr' / 'parameter_matched_ens_results'
        required = [matched_root / name for name in
                    ('selection.json', 'projector_controls.csv', 'paired_scores.csv',
                     'comparison.json')]
        if any(not item.is_file() for item in required):
            raise RuntimeError('Parameter matched ENS results are incomplete')
        selected_width = int(json.loads(required[0].read_text())['selected_width'])
        with required[1].open(newline='') as f:
            rows = [row for row in csv.DictReader(f)
                    if row['dataset'] == 'roman-empire'
                    and row['model'] == 'SAGE' and row['variant'] == 'ens_pooled'
                    and int(row['hidden_dim']) == selected_width]
        if len(rows) != 5 or {int(row['split']) for row in rows} != set(range(5)):
            raise RuntimeError('Parameter matched ENS needs five official masks')
        files.update(ROOT / name for name in PARAMETER_MATCHED_FILES)
        files.update(required)
        inference_profile = matched_root / 'inference_profile.json'
        if inference_profile.is_file():
            files.add(inference_profile)
        split0_row = next(row for row in rows if int(row['split']) == 0)
        files.add(selected_prediction(
            split0_row, matched_root / 'predictions',
            'roman-empire_SAGE_ens_pooled_split0.npz'))
    if args.include_ogb:
        files.update(ROOT / name for name in OGB_FILES)
        files.update(validate_ogb('pilot-100', 'ogbn_arxiv_results'))
    if args.include_ogb_300:
        files.update(validate_ogb('fixed-300', 'ogbn_arxiv_300_results'))
    if args.include_ogb_untied:
        files.update(validate_ogb_untied())
    if args.include_fixed_mask_seeds:
        files.update(validate_fixed_mask_seeds())
    if args.include_amazon_sage_pair:
        files.update(validate_amazon_sage_pair())
    if args.include_mc_dropout:
        files.update(validate_mc_dropout())
    if args.include_all_layer_sage:
        files.update(validate_all_layer_sage())
    if args.include_gat_pair:
        files.update(validate_gat_pair())
    if args.include_decision_strata:
        files.update(validate_decision_strata())
    if args.include_layerwise_sage:
        files.update(validate_layerwise('SAGE'))
    if args.include_layerwise_gat:
        files.update(validate_layerwise('GAT'))
    files = sorted(files)
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    readme = README
    requirements = REQUIREMENTS
    if args.include_parameter_matched_ens:
        readme += PARAMETER_MATCHED_README
    if args.include_ogb:
        readme += OGB_README
        requirements += 'ogb==1.3.6\n'
    if args.include_ogb_300:
        readme += OGB_300_README
    if args.include_ogb_untied:
        readme += OGB_UNTIED_README
    if args.include_fixed_mask_seeds:
        readme += FIXED_MASK_SEED_README
    if args.include_amazon_sage_pair:
        readme += AMAZON_SAGE_PAIR_README
    if args.include_mc_dropout:
        readme += MC_DROPOUT_README
    if args.include_all_layer_sage:
        readme += ALL_LAYER_README
    if args.include_gat_pair:
        readme += GAT_PAIR_README
    if args.include_decision_strata:
        readme += DECISION_STRATA_README
    if args.include_layerwise_sage:
        readme += LAYERWISE_SAGE_README
    if args.include_layerwise_gat:
        readme += LAYERWISE_GAT_README
    payloads = {'README.md': readme.encode(), 'requirements.txt': requirements.encode()}
    for path in files:
        name = str(path.relative_to(ROOT))
        if any(part.startswith('.') for part in Path(name).parts) or '.git' in Path(name).parts:
            raise ValueError(f'Hidden/history path: {name}')
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError(f'Archive source resolves outside repository: {name}')
        payloads[name] = path.read_bytes()
    prediction_names = sorted(name for name in payloads
                              if '/predictions/' in name and name.endswith('.npz'))
    prediction_manifest = {
        'files': [{'path': name, 'bytes': len(payloads[name]),
                   'sha256': hashlib.sha256(payloads[name]).hexdigest()}
                  for name in prediction_names]
    }
    payloads['experiments_iclr/prediction_manifest.json'] = (
        json.dumps(prediction_manifest, indent=2) + '\n').encode()
    for name, payload in payloads.items():
        if name.endswith(('.py', '.md', '.txt', '.csv', '.json', '.sh',
                          '.toml', '.lock')):
            hit = IDENTITY.search(payload)
            if hit:
                raise ValueError(f'Identity or private path found in {name}: {hit.group()!r}')
    temporary_out = OUT.with_suffix('.building.zip')
    with ZipFile(temporary_out, 'w', allowZip64=True) as archive:
        for name, payload in sorted(payloads.items()):
            compressed = not name.endswith('.npz')
            archive.writestr(zipinfo(name, compressed), payload)
    with ZipFile(temporary_out) as archive:
        names = archive.namelist()
        if set(names) != set(payloads):
            raise AssertionError('Archive manifest mismatch')
        for name in names:
            if name.startswith('/') or '..' in Path(name).parts or '.git' in Path(name).parts:
                raise AssertionError(f'Unsafe archive path: {name}')
    temporary_out.replace(OUT)
    print(f'Created {OUT.relative_to(ROOT)} with {len(payloads)} files, '
          f'{OUT.stat().st_size / 1024**2:.1f} MiB. Identity scan passed.')


if __name__ == '__main__':
    main()
