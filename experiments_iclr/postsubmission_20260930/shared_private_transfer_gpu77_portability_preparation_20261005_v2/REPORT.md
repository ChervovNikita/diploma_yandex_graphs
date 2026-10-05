# Private-transfer v2: normal repo-local 18.77 portability

## Status

This v2 plan supersedes the sealed v1 preparation. Use a fresh repo-local **conda CPython 3.11.14 environment with the original pinned cp311/cu118 core wheels**, record actual build/provider origins, match the episode sampling transcript, then run the unchanged finite FP32/native-Adam qualification once. Identical interpreter archives or host binaries are not required. Root allocates complete pilot blocks prospectively before new scores.

One authorized metadata-only 18.77 check was completed. No installation, download, numerical/model execution, fit or remote file write was performed. 7GPU was used only as the saved MacLink forwarding relay.

## Observed host and ordinary setup

The bounded check at 2026-10-05T05:13:59Z verified `peptide` and `/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git`. Both GPUs are A100 80GB PCIe, driver 580.126.09:

- GPU0: `GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998`
- GPU1: `GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced`

System Python is 3.10.12. PATH has no python3.11/uv/micromamba/conda. Known absolute Miniconda base Python is 3.11.5 and conda is 23.11.0; existing RAPIDS Python is 3.12.11. The two declared project venv paths checked are absent. These identity observations do not establish present free capacity.

`SETUP_COMMANDS.sh.txt` provides explicit ordinary commands:

1. The known absolute conda executable creates fresh `REPO/.gnnm_runtime/private_transfer_cp311_cu118_20261005_v1` with `python=3.11.14 pip` from declared conda-forge.
2. pip installs Torch2.1.2+cu118 from the official cu118 index, NumPy1.26.4, SciPy1.14.1, PyG2.7.0 and Triton2.1.0, then the original successful cp311/pt21cu118 sparse0.6.18 and scatter2.1.2 wheel URLs.
3. pip checks dependencies and saves install reports, freeze and conda explicit records. Record Python build, resolved package URLs/hashes and imported source/binary origins before admission.

`CORE_REQUIREMENTS.txt` retains all seven core pins. Torch/PyG import dependencies are listed in `ENVIRONMENT_PLAN.json`; ordinary resolved ancillary versions are recorded. Existing Miniconda base, RAPIDS and buddy runtime packages remain unchanged. The numerical child uses the new prefix's `bin/python` and clears inherited `PYTHONPATH` so the old77 overlay cannot shadow providers. Normal incidental caches are acceptable. No sudo, system CUDA installation, mounts, namespaces or host-setting changes are proposed.

Core version/origin mismatches and original native/scientific source hash mismatches must be reported explicitly. Build/driver/library differences are declared; they are not silently substituted or used to promise cross-host bitwise identity.

## Exact host-only source deltas

The concrete candidate `custody77.py` and `HOST_ONLY_DIFF.patch` change only the repository root, physical inventory/explicit selected UUID check and custody error text. Only `authorize` has a changed function AST. Every other custody function, including runtime, input loading, source loading and episode construction, remains identical.

| Binding | 18.77 port |
|---|---|
| Repository | Exact peptide repository above |
| Phase | Unchanged suffix `experiments_iclr/postsubmission_20260930` |
| Job hostname | `peptide` |
| GPU binding | Exact ordered UUID pair; root explicitly selects one UUID in `CUDA_VISIBLE_DEVICES`; one visible GPU remains `cuda:0` |
| Source folder | Fresh `shared_backbone_private_transfer_training_source_gpu77_20261005_v1` |
| Source manifest | New custody hash; all numerical file hashes unchanged |
| Runtime dictionary | Original singleton versions, satisfied by the new environment |
| Output/admission | Fresh77 phase-owned paths and source/runtime/qualification receipts |

`run.py`, `qualify_training_step.py`, `models.py`, `private_adam.py`, `transfer_step.py`, original native modules/heads, geometry and recursive adjoint retain their original bytes. No recipe, seed, objective, mask, tolerance, horizon or selection rule changes are proposed.

The old physical supervisor also has singleton guards. `SUPERVISOR_PORT_PLAN.json` lists its required host-only path/hostname/interpreter/inventory/transport/output deltas. Keep its 600s soft/720s hard bound, 64 GiB RSS/10 GiB owned-CUDA caps, one attempt and owned PID/start-tick closure. No launcher runs here.

## Byte-bound transfer set

`TRANSFER_PLAN.json` binds 38 local source/evidence payloads plus original remote Citeseer roles. Transfer the v2 files with only custody/new-manifest replacement, complete unchanged donor/geometry source packets, recursive adjoint, source reviews and authority receipts. Original native `model.py`/`util.py` are required; the unrelated installed77 NCNC source is not an alternative.

Copy the original acquisition manifest and exactly `train_pos.txt`, `valid_pos.txt`, `heart_valid_samples.npy`, `gnn_feature`, totaling 51,135,022 bytes, at their existing phase-relative paths. Preserve their recorded hashes. No TEST role/full archive is needed. The qualification opens TRAIN/features only; future authorized fits use the same full VALID roles.

Preserve original raw singleton FP32 evidence at `shared_backbone_private_transfer_fp32_execution_root_20261005_v1/result/RESULT.json`: 2,051,914 bytes, SHA256 `8601b1138c3f7ab65cb1525d5d8274a8e14e22731cd060801eec437f6169806d`. Compact JSON is not a byte-identical substitute. Chunk/compress transport may reassemble and hash the original bytes. These old receipts are lineage; they do not admit the new port manifest.

## Required sampling and one FP32 gate

`SAMPLING_TRANSCRIPT_PLAN.json` specifies a finite TRAIN-only engineering comparison: hash the complete native negative bank, outer order including tail, endpoint/random inner indices, outer endpoints and common removed/kept support indices using identical canonical serialization. Compare singleton and77 at the qualification seed/cycle and the prospectively allocated pilot seed/cycle scope root requires for its frozen horizon. No outcome-selected fixture, redraw, model, fit or VALID/TEST values enter this gate. A mismatch is preserved and returned to root, not corrected by silent resampling.

Then execute the **unchanged qualifier once**: seed/factor seed 20261005, first TRAIN episode, outer 64/inner 256, shared F4/capable single/untied four, two reachable recomputed Adam histories and one stale discarded step. Preserve scalar guards, complete native parameter/moment parity, independent chain rule, shared/private commitment, original-donor serving parity and state/RNG/mode checks, with original tolerances. All states are discarded; no scores/checkpoints/fits. Preserve raw result/failure, provider/source/input/UUID bindings and physical terminal receipt. No automatic retry or tolerance adjustment.

`QUALIFICATION_JOB_DISABLED.json` is unreleased. Unchanged `run.admissible` requires a passing new FP32 result bound to the new source manifest, so singleton/old SGD/float64/import metadata cannot replace the77 gate. Sampling equality and within-runtime FP32 parity establish this engineering admission; they do not promise identical trajectories across GPU/driver builds.

## Root-prospective block allocation

**This preparation proposes no host mapping.** During final binding checks root froze a60-cycle horizon and assigned complete blocks in the current study specification: b0singleton, b1on77GPU0 and b2on77GPU1. `ROOT_STUDY_SPEC_SNAPSHOT.json` retains those root-prospective bytes; this packet does not independently freeze or change the assignment. All comparison cells within a block stay on the same host/runtime and preserve their prospective order; all cell types receive the same host distribution or any unavoidable imbalance is disclosed. One owned fit runs per chosen physical GPU, after root checks current availability/ownership.

Preserve the same root-frozen numerical horizon, evaluation every 5 complete cycles, eleven VALID misses and first maximum complete rounded4VALID MRR. Faster execution must not create extra training/selection opportunities. Report actual host/GPU/build/provider origins and inclusive paid costs; disclose seed/host confounding. `COHORT_HOST_PLAN.json` records the current root assignment and allocation constraints; this agent makes no block assignment or edit to `STUDY_SPEC.json`.
