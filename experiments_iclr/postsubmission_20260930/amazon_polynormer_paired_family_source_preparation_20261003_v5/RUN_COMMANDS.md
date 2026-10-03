# Exact normal-execution commands

Run from `postsubmission_research_20260930`. `python3` runs the stdlib helper/supervisor; each worker uses the exact interpreter descriptor in its root release. Source review and explicit root authorization must precede numerical execution. These commands were not executed in source preparation.

```sh
amazon_packet="amazon_polynormer_paired_family_source_preparation_20261003_v5"
amazon_run="amazon_polynormer_paired_family_execution_root_20261003_v3"
```

## Root release creation

Create each external `release_specs/<name>.json` by merging the common spec and the relevant kind-specific fields in `ROOT_RELEASE_SPEC_TEMPLATE.json`. Fill the measured/prospective caps, interpreter path, metadata paths, exact output and actual kind. Authorization fields remain false until root explicitly approves. `prepare_release.py` supplies final source descriptors and hashes existing JSON metadata; it copies public/TRAIN/VAL descriptors from the existing data manifest without opening those payloads. It preserves the authorization supplied by root and performs no numerical work.

```sh
python3 "$amazon_packet/prepare_release.py" --spec "$amazon_run/release_specs/runtime_cpu.json" --output "$amazon_run/releases/runtime_cpu.json"
python3 "$amazon_packet/prepare_release.py" --spec "$amazon_run/release_specs/runtime_gpu.json" --output "$amazon_run/releases/runtime_gpu.json"
```

Run the approved runtime captures:

```sh
python3 "$amazon_packet/supervise.py" --kind runtime_capture --release "$amazon_run/releases/runtime_cpu.json" --output "$amazon_run/runtime_cpu"
python3 "$amazon_packet/supervise.py" --kind runtime_capture --release "$amazon_run/releases/runtime_gpu.json" --output "$amazon_run/runtime_gpu"
```

CPU capture uses `device=cpu`; GPU capture uses the explicitly approved indexed GPU. The produced `RUNTIME.json` binds actual interpreter, package/source/binary files, flags and hardware. Select one of the two declared existing profiles in `DEPENDENCIES.json` before qualification; the scientific cohort uses one exact GPU receipt.

## Existing-data consumer and fixed registry

The consumer preparation spec uses the exact existing `DATA_MANIFEST.json` descriptor from DESIGN and the new CPU runtime receipt. It only reads existing public arrays and compact TRAIN/VAL packs, derives roles, and emits proposed metadata.

```sh
python3 "$amazon_packet/prepare_release.py" --spec "$amazon_run/release_specs/consumer_prepare.json" --output "$amazon_run/releases/consumer_prepare.json"
python3 "$amazon_packet/supervise.py" --kind consumer_prepare --release "$amazon_run/releases/consumer_prepare.json" --output "$amazon_run/consumer_prepare"
python3 "$amazon_packet/prepare_release.py" --spec "$amazon_run/release_specs/register.json" --output "$amazon_run/releases/register.json"
python3 "$amazon_packet/supervise.py" --kind register --release "$amazon_run/releases/register.json" --output "$amazon_run/registry"
```

Root independently compares the proposed role hashes to preserved prior role evidence and writes a separate `CONSUMER_RELEASE.json`. Each normalized role receipt must have schema `amazon_fixed_role_identity_v1`, the exact existing data descriptor and `role_record`, plus `prior_artifact_comparison_satisfied=true`. Fresh preparation receipts intentionally contain false and cannot admit qualification. The release requires exact `preprocessing_identity` and `prior_role_receipts` for splits0/1/2. Raw all-node label NPZ is never opened by this consumer. Registration writes the immutable fifteen-fit/nine-family registry and one source claim; a second registry for this sealed source is rejected.

## Bounded complete-Amazon qualifier

The qualification spec uses `kind=qualify`, `device=cuda:0` (or the actual approved index), the new exact GPU runtime receipt, reviewed consumer release, complete preserved attempt registry and prospectively approved whole-process caps. Output is fixed below. It requires no utility threshold.

```sh
python3 "$amazon_packet/prepare_release.py" --spec "$amazon_run/release_specs/qualify.json" --output "$amazon_run/releases/qualify.json"
python3 "$amazon_packet/supervise.py" --kind qualify --release "$amazon_run/releases/qualify.json" --output "$amazon_run/qualification_block0_cuda0"
```

Only a physical exit0 plus unchanged custody and `qualification_block0_cuda0/FREEZE.json` admits the qualification evidence. RESULT records block0/seed17 native seeds17/1026/2035/3044 and GNNM4, both-stage parity, scratch rollback, real dropout updates, explicit transition, safe exact next-step replays and costs. The shortened probe is report-ineligible and cannot initialize any scientific fit. Root then approves the measured complete-schedule forecast in a separate resource admission.

## Exact fifteen-fit study and closure

Create fifteen individually authorized `releases/fits/<registry id>.json` files from the fit spec. Each binds its exact registry row/output, same GPU runtime/consumer, qualification freeze, measured resource admission and all preceding attempt evidence. The `all` spec binds all fifteen release descriptors in exact registry order and the separately authorized `close` release. No recipe/configuration option exists.

```sh
python3 "$amazon_packet/prepare_release.py" --spec "$amazon_run/release_specs/close.json" --output "$amazon_run/releases/close.json"
python3 "$amazon_packet/prepare_release.py" --spec "$amazon_run/release_specs/all_fits.json" --output "$amazon_run/releases/all_fits.json"
python3 "$amazon_packet/supervise.py" --kind all --release "$amazon_run/releases/all_fits.json" --output "$amazon_run/all_fits"
```

For a registered individual fit the exact command is:

```sh
python3 "$amazon_packet/prepare_release.py" --spec "$amazon_run/release_specs/fits/split0_gnnm_boundary_4_seed17.json" --output "$amazon_run/releases/fits/split0_gnnm_boundary_4_seed17.json"
python3 "$amazon_packet/supervise.py" --kind fit --release "$amazon_run/releases/fits/split0_gnnm_boundary_4_seed17.json" --output "$amazon_run/fits/split0_gnnm_boundary_4_seed17"
```

Choose the individual or `all` launch route prospectively; never rerun a claimed physical slot. `all` executes exactly15 registered fits sequentially and stops at the first failure. Native single references independent member0. Closure requires all2700-update traces, source/runtime/data preservation, selected image/logit replays and successful physical freezes; partial cohorts cannot close.

## Separate complete-cohort evaluation

The evaluation spec must bind the successful cohort closure freeze and explicitly authorize TRAIN-control predictive scoring after that closure. Output is fixed below. TEST labels have no reader.

```sh
python3 "$amazon_packet/prepare_release.py" --spec "$amazon_run/release_specs/evaluate.json" --output "$amazon_run/releases/evaluate.json"
python3 "$amazon_packet/supervise.py" --kind evaluate --release "$amazon_run/releases/evaluate.json" --output "$amazon_run/evaluation"
```

Evaluation reopens/replays all15 selected states and raw logits, retains all4 members in each four-member pool, recomputes frozen metrics, and reports all3 paired control-NLL differences and their mean. Ordinary subprocess supervision records wall/RSS/GPU costs, physical exits, immutable failures and an atomic success freeze. It performs no namespace, install, system change, restart or automatic retry.
