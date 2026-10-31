# F1 Stage 1 native baseline preparation

This is a small runnable source preparation, not an admitted or executed study.
No database, labels, weights, scientific imports, fitting, GPU work, or heldout
evaluation were performed while preparing it. Prior scouts and industrial v2
are unchanged. Source text retrievals and stdlib integrity/AST checks are recorded
in `evidence/`. The parent task owns real filesystem confinement, local data and
asset admission, the complete dependency lock, runtime qualification, and any
later heldout inference.

## Fixed scope

Three **separate** native HeteroGraphSAGE fits at seeds 42, 43, and 44 use the
official RelBench v1.1.0 model at commit
`9aa346267c2e1c560bd92da07d6f4ad1ca2f0639`. All ten epochs run; the strictly
smallest fixed-sampling validation MAE selects the checkpoint. There is no graph
HPT. The profile is `PROFILE.json`. This is not an ensemble: the user's default
member count remains **four**; initializer and ensemble decisions are unadopted.

Controls are the training global median, a fixed prior-year driver mean with
global-median fallback, full-data native raw LightGBM, and a history-engineered
LightGBM view. Each GBDT view has exactly ten native validation-MAE trials plus
the native final refit. No additional search is available in the CLI.

The prospective competence rule is: mean graph validation MAE across the three
seeds must be strictly below the median and no worse than raw LightGBM.
History controls are always reported, including if stronger. Failure is not
rescued by test selection or more HPT. MAE is primary; R² and RMSE are fixed
secondary legacy metrics. No paper result is claimed.

## Custodian export (outside worker filesystem)

Inputs must be already-present native RelBench database parquet files with
consecutive, time-sorted reindexed primary/foreign keys, and already-admitted
native task parquet files. The custodian takes explicit local paths; it does
not call the dataset/task registry, regenerate the cohort, or download data.

```sh
python /path/to/packet/custodian_export.py \
  --source-db /admitted/private/rel-f1/db \
  --train-table /admitted/private/driver-position/train.parquet \
  --val-table /admitted/private/driver-position/val.parquet \
  --test-table /admitted/private/driver-position/test.parquet \
  --output-dir /admitted/public/f1-stage1
```

`--output-dir` must be empty. Test parquet is projected to `driverId,date` only;
its target values are never materialized. Exported train/validation files contain
targets; the test-key file contains no target column. Every split preserves the
original native row order as `row_id` and adds a key/order hash. The public JSON
contains no private file paths. The public database is cut inclusively at
2010-01-01, followed by native dangling-FK correction; IDs are not reindexed.
Counts are required to be 7,453 train, 499 validation, and 760 masked test rows.

Do not give the worker ordinary RelBench caches or private task/data paths.
The published cohort is used without change, including native eligibility's
missing upper time bound; this packet does not claim a production cohort.

## Worker run (only after parent admission)

```sh
python /path/to/packet/run_stage1.py \
  --input-root /admitted/public/f1-stage1 \
  --glove-dir /admitted/assets/average_word_embeddings_glove.6B.300d \
  --output-dir /admitted/output/f1-stage1 \
  --device cuda
```

The worker sees only the public export and a complete already-present local
native SentenceTransformer GloVe asset. The explicit local asset is loaded with
`local_files_only=True` and Hugging Face offline variables; every asset file is
hashed. Asset identity and enforcement of offline filesystem/network boundaries
remain the parent's admission responsibilities. `--device cpu` is available for
parent-directed qualification; no device is selected silently.

The candidate Python/package versions are in `PROFILE.json` and
`requirements-candidate.txt`. The runner fails on drift from critical saved
native source files and requires compiled pyg-lib heterogeneous temporal
sampling. The remaining environment must be locked before scientific execution;
the runtime receipt records all installed distribution versions. These checks
do not prove dependency or runtime qualification.

## Declared independent adaptations

- Timestamped tables fit stypes/statistics/vocabularies only on rows through
  2005-01-01. Their frozen converters transform the stable-ID full public table.
  Untimestamped native metadata is retained as an explicit benchmark assumption.
- Official `get_node_train_table_input` and target attachment preserve repeated
  drivers at different forecast dates. Sampling is uniform, directional and
  disjoint with fanouts 128/64 and zero workers. Every batch checks root identities,
  disjoint components and sampled timestamps against seed times.
- Each graph seed gets a separate validation RNG namespace (`100000 + seed`).
  Python, NumPy and CPU/CUDA Torch training RNG states are preserved. Sampling
  digests must match across all ten comparisons and the restored pass. Restored
  MAE uses a fixed numerical consistency tolerance (relative 1e-6, absolute 1e-7)
  because CUDA reduction arithmetic is not promised bitwise deterministic.
- Raw GBDT follows the native entity-table merge, no subsampling and no AR labels.
  Tabular converters fit only training forecast rows. GBDT objective/domain,
  2,000-round maximum and 50-round early stopping are native. Optuna TPE is seeded
  at 42 for repeatability and retained in a local SQLite study; the native booster
  seed default of zero is preserved and parameters/trial outcomes/costs recorded.
- Historical SQL retains the source's 44 historical predictors and `driverId,date`
  (46 non-target fields), and removes all six upcoming round/circuit fields. The
  query adds deterministic lowest `resultId` duplicate handling and explicit
  past-date join guards. Feature counts, keys, targets and one-row-per-forecast
  coverage must pass. Source race-ID offsets and prior-two-month window remain.
- Frame 0.2.3's native LightGBM converter ignores timestamp tensors. Thus `date`
  is retained/materialized but does not become a booster input; the report records
  effective features. This is not a five-run author reproduction.

## Reviewable outputs

`STAGE1_SUMMARY.json` reports competence and cost accounting. Each native fit
has epoch metrics, the selected state dictionary including buffers, a summary,
and row-keyed validation CSV. Shared frozen graph converters/statistics, stypes,
sampling QA, input/profile copies, GloVe hashes and runtime identity are saved.
Controls save keyed validation predictions, frozen train converters, native
booster models, ten per-trial receipts, and study/summary files. No test predictions
or test metrics are produced. Later test inference requires a separate admitted
stage; it is deliberately absent from this runner.

Expected graph work is 15 batches × 10 epochs × 3 fits = **450 optimizer updates**,
plus validation and sampling QA. No timing or memory measurements exist yet.

## Source-only preparation check

```sh
python /path/to/packet/check_preparation.py
```

This imports only stdlib, checks saved hashes and syntax, and writes no results.
Numerical, SQL, sampler, asset, device and dependency checks remain to be run by
the parent in its admitted environment. A failed run must be reported as a
qualification failure, without silently changing the frozen recipe.
