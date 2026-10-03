# NCNC collab predictive pilot driver

Status: source prepared; numerical synthetic and complete TRAIN/VALID execution qualification remain required. This packet implements the accepted design in `graph_ncNC_collab_predictive_pilot_design_20261003_v1`. The qualified prototype v2 and resource v4 are reused without changes. Preparation imported no numerical library, opened no dataset or actual checkpoint, and launched no remote job.

## Entry point and release

`pilot_run.py --root-release FILE --stage STAGE --unit UNIT --base-seed S --output DIRECTORY [--resume]`

Units: `native_bank4`, `factor_private4`, `factor_pooled4`, `native70`. Base seeds are exactly 0–4. A root release must name the stage and exact unit, seed, and absolute output directory in `authorized_invocations`. The release example is prospective and must be copied to an external path, completed, and bound to the actual sealed manifest SHA. Its schema is documented in `ROOT_RELEASE_SCHEMA.md`.

Each release uses the same `family_id`, `family_lock_output_directory`, source packets, TRAIN/VALID authority, and runtime authority. Normal `CUDA_VISIBLE_DEVICES` selects one of the two root-authorized physical GPU UUIDs; the worker uses logical device 0. Existing dependency overlay and interpreter are used; no installation or namespace setup is performed.

Stages:

| Stage | Required work | Receipt |
|---|---|---|
| `synthetic` | Fabricated metric/model/sign/RNG/update checks, atomic state fault checks, actual 100-epoch orchestration with fabricated work and interrupted-epoch resume | `QUALIFICATION.json` |
| `valid_engineering` | Fresh engineering constructor and every official VALID positive/negative query | `QUALIFICATION.json` |
| `resource_engineering` | One complete 17-batch native TRAIN epoch per model, then every official VALID query; both factor routes and all native bank members are supported | `QUALIFICATION.json` |
| `fit` | Frozen scientific 100-epoch unit, private selection and full states | `COMPLETE.json` |
| `family_lock` | All 20 complete-or-terminal-failed unit entries, 35 planned fits, all 25 cells, paired streams, selected-state custody, costs and failures | `FAMILY_LOCK.json` |

No TEST stage or TEST data accessor exists. Engineering stages never invoke the ranking metric or save model/Adam/RNG donor checkpoints. They report coverage, finite completion, score hashes, timing and memory only. All scientific fits construct fresh states; seed 20261003 and resource states are excluded as scientific donors.

## Ordinary runtime commands

After completing the external release, root runs commands of this form on 77. Set `task_python`, `task_driver`, `task_release`, and `task_research` to the paths below. These variables deliberately do not replace HOME or CODEX_HOME.

```sh
task_python=/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12
task_research=/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930
task_driver="$task_research/graph_ncNC_collab_predictive_driver_preparation_20261003_v1/pilot_run.py"
task_release="$task_research/graph_ncNC_predictive_qualification_execution_root_20261003_v1/ROOT_RELEASE.json"
```

Every invocation needs the release's exact output path. The example release supplies these five paths:

```sh
CUDA_VISIBLE_DEVICES=GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998 PYTHONPATH=/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/.gnnm_runtime/buddy_extra_v1/site PYTHONDONTWRITEBYTECODE=1 "$task_python" "$task_driver" --root-release "$task_release" --stage synthetic --unit native_bank4 --base-seed 0 --output "$task_research/graph_ncNC_predictive_qualification_execution_root_20261003_v1/synthetic/run01"
```

```sh
CUDA_VISIBLE_DEVICES=GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998 PYTHONPATH=/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/.gnnm_runtime/buddy_extra_v1/site PYTHONDONTWRITEBYTECODE=1 "$task_python" "$task_driver" --root-release "$task_release" --stage resource_engineering --unit native_bank4 --base-seed 0 --output "$task_research/graph_ncNC_predictive_qualification_execution_root_20261003_v1/native_bank4/run01"
```

```sh
CUDA_VISIBLE_DEVICES=GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998 PYTHONPATH=/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/.gnnm_runtime/buddy_extra_v1/site PYTHONDONTWRITEBYTECODE=1 "$task_python" "$task_driver" --root-release "$task_release" --stage resource_engineering --unit factor_private4 --base-seed 0 --output "$task_research/graph_ncNC_predictive_qualification_execution_root_20261003_v1/factor_private4/run01"
```

```sh
CUDA_VISIBLE_DEVICES=GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998 PYTHONPATH=/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/.gnnm_runtime/buddy_extra_v1/site PYTHONDONTWRITEBYTECODE=1 "$task_python" "$task_driver" --root-release "$task_release" --stage resource_engineering --unit factor_pooled4 --base-seed 0 --output "$task_research/graph_ncNC_predictive_qualification_execution_root_20261003_v1/factor_pooled4/run01"
```

```sh
CUDA_VISIBLE_DEVICES=GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998 PYTHONPATH=/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/.gnnm_runtime/buddy_extra_v1/site PYTHONDONTWRITEBYTECODE=1 "$task_python" "$task_driver" --root-release "$task_release" --stage resource_engineering --unit native70 --base-seed 0 --output "$task_research/graph_ncNC_predictive_qualification_execution_root_20261003_v1/native70/run01"
```

Root can select the second authorized UUID by changing both release and environment. Run `fit` only after binding the passing synthetic receipt and all four passing resource-engineering receipts. The gate also checks exact F4 initial state, TRAIN stream and final RNG agreement. `valid_engineering` is optional when the complete resource stage has already covered its VALID work. No total family ETA is claimed.

## Scientific semantics

There are 20 native width64 fits, 10 factorized fits, and 5 native width70 fits. `native_bank4` owns four separate learned encoders, decoders, Adam instances, and RNG streams with seeds `s+5*m`. N64 reuses that bank's individually selected member0 state. Each independent model receives its own unscaled native positive mean plus negative mean loss. F4's loss averages over all query/member entries. Parameters are N64 38147/33922 total/active, I4 152588/135688, F4 43790/38793, and N70 44663/39622.

All 235868 nodes, 128 raw features and official TRAIN records are retained. Positive record minibatches are removed before rebuilding an unweighted symmetric graph; remaining duplicate records survive. The native default sampler uses the reciprocal raw TRAIN graph and rejects TRAIN/self-links only. Its negative draw precedes the native shuffled permutation. Each epoch has 17 batches of 65536 and drops the shuffled tail of 64940. No candidate cap or reduced graph is introduced.

VALID uses the complete TRAIN graph and all 60084 positives/100000 negatives in canonical order, including evaluation tails. Serving takes the equal mean of raw logits. The exact source-bound OGB class parses and evaluates scores, with strict ties failing; an independent explicit rule cross-check is required. Root accepted bypassing only the evaluator constructor's unbound master.csv read: the accepted frozen collab metric is assigned to `name`, `eval_metric` and `K`, and the pinned official methods are used. Production metrics reject any incomplete official pool. Synthetic checks cover attributes, the entire negative pool, strict ties, malformed/nonfinite inputs and mean-before-metric behavior.

Single/F4 selectors evaluate epochs1–100, save epoch1 unconditionally and replace only on strict served Hits@50 improvement. I4 evaluates same-epoch banks1–100 and then a fresh VALID evaluation of the bank assembled from four individually selected states. This 101st candidate's four extra encoder/scoring passes are charged. First exact ties remain selected.

The full-family primary summary is the five paired, separately validation-selected private-minus-pooled VALID differences, with mean and sample SD. VALID is used for selection and for this development-pilot contrast. Baseline quality/cost comparisons remain exploratory. No novelty, historical numerical reproduction, broad SOTA, or baseline superiority claim is made.

## Fresh initialization authority

Native N64/N70 call the exact preserved native `GCN` and `IncompleteCN1Predictor` constructors once after the frozen seed. Their embedded PyG/Torch constructors own parameter initialization/reset behavior; no extra reset, donor state, or warm checkpoint is inserted. This follows the saved native run-loop construction order, while the current runtime and explicit Python sampler seeding are prospective adaptations.

F4 calls the qualified prototype constructors, retaining their shared W/bias, LN, beta and fixed buffer defaults. It then initializes every r/s parameter, including unused ptlin factors, with the accepted dedicated CPU Rademacher generator in lexical named-parameter order. The generator does not consume training RNG. The two routes independently replay identical initial model, empty Adam, flags and all RNG states, and closure checks every epoch's streams and RNG consumption. Synthetic source qualification has copied-state mathematical ancestry; it does not establish identical historical constructor draws or bitwise CUDA training.

## Persistence, visibility and failure closure

Scientific states and metrics remain inside mode600 own journals/checkpoints/selection metadata under mode700 output directories. Live status and stdout contain counts and phases only. Each committed epoch contains complete model/Adam/RNG/module flags, all selectors and retained best states. Revisions alternate two slots; the prior journal survives a failed commit. Resume accepts only the current own bound slot, verifies identity/hash, restores its exact epoch and RNG, and repeats any uncommitted epoch without new donors. Adam values are cloned before restoration to avoid retained-best aliasing. Concurrent workers cannot own the same output.

`ATTEMPTS.json` retains completed phases, batch counts and observed costs even after failure. Closed attempt wall time includes setup, custody reads/hashes, transfers, full training, VALID, selection and serialization. Interrupted processes preserve observed lower bounds and mark their unknown remainder; they are never called exact costs. The terminal accounting receipt writes have a disclosed unmeasured tail. No loss or predictive curve is exposed.

A complete closure requires all 20 bound units, 35 complete unique fits and all 25 selected cells. A root-released terminal-failure disposition can instead bind `FAILED.json`, its accounting/journal custody, and an explicit immutable retirement reference. That closure retains the missing cells and costs, emits no success-only subset summary or primary contrast, and authorizes no TEST. Family closure markers and the common lock destination prevent later fit/resume/seed replacement. Retry of a runtime failure, when explicitly released before closure, restores the same scientific state and preserves every attempt cost.

## Files and preparation checks

`SOURCE_BINDINGS.json` records exact accepted design, source/runtime/data authority and prior engineering receipt hashes. `STDLIB_PREPARATION_CHECK.json` verifies AST syntax, absence of top-level numerical imports, unchanged sealed dependencies, source metadata and fabricated complete/failed family closure fixtures. It does not qualify numerical execution. Do not run its default preparation write inside a sealed packet; root can read this receipt and run the separate external-output numerical `synthetic` stage.

The remaining execution gates are the ordinary-runtime synthetic stage and complete TRAIN/VALID engineering for all four units. Then root can release the 35-fit family. TEST requires a separate future release and implementation after immutable family closure.
