# Anonymous GNNM reproduction code

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

The included public graph files are `data/minesweeper.npz` and
`data/tolokers.npz`. The public `data/roman_empire.npz`,
`data/amazon_ratings.npz`, and `data/questions.npz` are omitted to keep the
upload below the submission form's 100 MB limit. Run
`python experiments_iclr/fetch_datasets.py` before new training to fetch and
verify the omitted files from the pinned public source commit. The advertised
compact score verifiers use retained selected arrays and do not require the
omitted Roman graph. See `UPLOAD_PACKAGE_SCOPE.md` for exact bytes and hashes. The pinned
download URL template, byte sizes, and SHA256 hashes of all five files are
in `experiments_iclr/data_manifest.json`. Recomputing the archived tables
above reads only CSVs and does not require the omitted graph files.
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
python experiments_iclr/projector_controls.py --model SAGE \
  --variants gnnm independent_projectors heads_only input_only output_only base gnnm_m1 \
    untied_backbone freeze_output_factors ens_pooled \
  --splits 0 1 2 3 4 --result_root experiments_iclr/replay_controls
```

This command creates a new run for comparison with the archived rows. It does
not restore omitted historical artifacts in `experiments_iclr/results/`.
The included decision analysis and complete verifiers target that original
result root and its saved artifact hashes. This upload projects all 14 earlier supplied Roman control prediction files
uniformly. It retains the exact float32 mean logits and all other original
arrays, including member classes, pooled probabilities, labels and node IDs.
Individual member float logits remain in the full author bundle. The original
`prediction_manifest.json` identifies those original arrays and their hashes.
`CONTROL_POOLED_MANIFEST.json` links every derived file to that manifest and
its unchanged selected result row. Run
`python experiments_iclr/verify_control_pooled.py` to recalculate the 14
pooled scores, cross-entropies and member accuracies. The remaining original
control rows still have their summary CSV records and original omission scope.
The original full-array diagnostic scripts require the full author artifacts. Summary CSVs retain every
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


## Parameter matched ensemble

This optional section is included only when the five-mask parameter matched
ensemble study is complete and used in the manuscript. The CPU count-only
selection in `experiments_iclr/select_parameter_matched_ens.py` predeclared
widths 192, 224, 256, 288, and 320. Its manifest records the chosen width
and every model count before any score was read. The training runner uses the
same masks and pooled validation checkpoint rule as width-512 ENS. The paired
summary and, when available, same-device checkpoint-byte and inference-latency
profiles are in `experiments_iclr/parameter_matched_ens_results/`. Only mask-0 pooled logits and member classes
from this additional ensemble are included to limit archive size; all five
run rows and paired scores are included.

For a new width-256 training run, use a separate empty result root so the
packaged five selected rows do not make the runner skip every mask:

```
python experiments_iclr/projector_controls.py --model SAGE \
  --variants ens_pooled --splits 0 1 2 3 4 --num_layers 5 \
  --hidden_dim 256 --lr 3e-5 --m 4 --num_steps 5000 \
  --result_root experiments_iclr/replay_matched_ens
```

The included paired summarizer targets the original result roots and does not
summarize this new root without an explicit root-path adaptation.


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
python experiments_iclr/ogbn_arxiv_pilot.py --device cuda:0 \
  --seeds 0 1 2 --variants base ens gnnm \
  --output-root experiments_iclr/ogbn_arxiv_replay_100
```


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
python experiments_iclr/ogbn_arxiv_pilot.py --device cuda:0 \
  --seeds 0 1 2 --variants base ens gnnm \
  --max-epochs 300 --min-epochs 300 --patience 20 --eval-every 1 \
  --output-root experiments_iclr/ogbn_arxiv_replay_300
```

On the complete original artifact tree, verify the two budgets separately
with `python experiments_iclr/verify_ogbn_arxiv_results.py --profile pilot-100 --require-complete`
and `python experiments_iclr/verify_ogbn_arxiv_results.py --profile fixed-300 --require-complete`.
Those historical gates require the omitted original checkpoints and logits;
they do not certify fresh-root runs made by the commands above.


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


## WikiCS and Actor fixed-split SAGE controls

`external_sage/` contains the frozen source and protocol for the three-arm,
three-seed WikiCS and Actor study, with published split 0 on each graph.
Within each seed, tied propagation, initially copied untied propagation,
and added in-layer factors begin from the same member functions and random
state. Run `python external_sage/verify_pooled_classes.py` to recalculate all
tabulated pooled scores from saved exact pooled logits, check the complete
validation traces and selected epoch, and compare the GPU replay audit.
The full checkpoint replay was completed before packaging, but the large
checkpoints and raw public graph files are omitted from this ZIP. Use
`python external_sage/fetch_data.py` to download and hash-check the exact
public WikiCS and Actor files before rerunning the frozen experiment.
Three seeds on one split are optimizer repetitions, not graph replicates.


## Post hoc Roman Empire configuration bridge

`roman_bridge/` applies the same two-layer, width-128, 300-epoch SAGE
configuration as the external studies to Roman Empire official split 0,
with seeds 0–2 and tied, initially copied untied, and in-layer-factor arms.
Run `python roman_bridge/verify_compact_pooled.py` to recalculate the selected
scores and cross-entropy from retained pooled logits, member accuracy from saved classes, and links to the recorded GPU replay. Individual member float logits are omitted uniformly for all nine bridge rows. See `roman_bridge/COMPACT_SCOPE.md`.
`python experiments_iclr/fetch_datasets.py` first downloads and verifies the public Roman graph. Then `python roman_bridge/fetch_data.py` copies and hash-checks it for a fresh run. This bridge was chosen
after seeing other results. It uses a bidirectional message graph without
explicit self-loops, while the earlier Roman Empire component study added
one self-loop per node. The bridge therefore does not isolate a single
architecture or graph-preprocessing change.


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


## Post hoc GAT tying pair

The optional GAT study includes the fixed launcher source, read-only verifier,
five-mask result matrix, paired decision analysis, and source/input hash
manifests. Its original launcher and verifier require the server queue marker
and selected checkpoints and prediction arrays, which are omitted here.
Regenerate those artifacts before rerunning the verifier or decision analysis.
The chosen backbone and masks were selected after archived results were seen;
the five overlapping masks are descriptive, not independent graph tasks.

## Additional audited experiments

The Roman depth, no-loop depth, and longer-budget tests, external depth tests, sharing-position study, 1,000-epoch ogbn-arxiv repeat, and ogbl-collab link study are documented in `experiments_iclr/NEW_STUDIES_README.md`. Run the listed compact score verifiers from this archive root. The compact arrays retain selected hard decisions or full logits, depending on the study, along with validation traces. Full training checkpoints are omitted. Each study README identifies the supplied arrays and which checks can be rerun from them. See `experiments_iclr/VERIFICATION_SCOPE.md` for the boundary between compact verification and author checkpoint replay.

The upload uses pooled float32 logits plus every member class for all 18 earlier external-SAGE rows. Its individual member float logits remain in the author bundle. See `external_sage/PROJECTION_SCOPE.md`. All 72 selected primary-grid test member logits are supplied in `experiments_iclr/hpo_selected_test_logits/`, enabling bitwise reconstruction of their pooled logits.
