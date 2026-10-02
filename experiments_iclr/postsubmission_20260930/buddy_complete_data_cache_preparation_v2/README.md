# Official collab public/train/valid staging and cache qualification

Source wrapper v2 preserves sealed wrapper v1 and adds the recorded process-local `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` setting for Torch2.6+ compatibility with pinned OGB1.3.6's PyG/NumPy data loads. It patches no functions. Explicit checkpoint `weights_only=True` guards remain in force. The complete official archive SHA is verified before any data deserialization.

Prepared source wrapper around sealed **BUDDY v4**, preserving its recipe and vendor bytes. Local stdlib source/archive fixtures pass; actual complete-data execution is pending root's fresh v4 seven-test CPU certificate and source inspection. The acquired official archive is reused; it is not downloaded again. No real archive member, data, Torch, model/checkpoint, GPU or remote action was accessed by the author.

`SOURCE_ASSESSMENT.md` describes the inspected installed OGB convention. `CONTRACT.json` freezes archive metadata/SHA, the nine extraction members, installed OGB source hashes and v4 source manifest. `prepare_cache.py` verifies those sources plus a fresh exact-v4 numerical certificate, stages a fresh repository dataset root, checks complete raw/processed/train correspondence before native hashing, builds through unmodified v4, validates the complete cache and writes one exclusive source/data/cache-bound qualification receipt. It uses direct calls and patches no runtime source functions.

The only allowed payloads are release metadata, train/valid split files and public raw graph/features/counts/weights/years. Closed test contents remain in the archive; test.pt and split_dict.pt are never staged/read/decompressed. No toy substitute, year filter, isolate removal, validation-edge graph augmentation or future-positive rejection is allowed. A failed assertion leaves a failed receipt and blocks downstream resource/family work; outputs are not overwritten or silently replayed.

## Root command in the already authorized one-GPU checkout

Run after fresh **v4** seven-test/source qualification. This command forces CPU computation inside the wrapper; the existing isolated dependency target is reused.

```sh
BUDDY_REPO=/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
BUDDY_PHASE="$BUDDY_REPO/experiments_iclr/postsubmission_20260930"
PYTHONPATH="$BUDDY_PHASE/buddy_cpu_runtime_preparation_v1/root_setup_v2/site" "$BUDDY_REPO/.venv/bin/python" -B "$BUDDY_PHASE/buddy_complete_data_cache_preparation_v2/prepare_cache.py" \
  --repo "$BUDDY_REPO" \
  --archive "$BUDDY_PHASE/buddy_official_archive_staging_v1/root_transfer_v2/collab.zip" \
  --dataset-root "$BUDDY_PHASE/buddy_complete_data_cache_preparation_v2/root_run_v1/dataset" \
  --cache-output "$BUDDY_PHASE/buddy_complete_data_cache_preparation_v2/root_run_v1/cache" \
  --receipt "$BUDDY_PHASE/buddy_complete_data_cache_preparation_v2/root_run_v1/QUALIFICATION.json" \
  --qualification "$BUDDY_PHASE/buddy_cpu_numerical_qualification_v4/root_run_v1/CPU_QUALIFICATION.json" \
  --source-dir "$BUDDY_PHASE/buddy_shared_cache_execution_v4" \
  --cpu-threads 1
```

Use the actual new root certificate path if root names that run differently. The certificate must match all v4 implementation hashes/Torch version; v3's successful certificate cannot pass. Any other authorized runtime must match the pinned installed OGB sources and use its qualified compatible dependencies. This wrapper has no network/SSH/host/GPU management code.

## Small invocation on authorized 18.77

From `/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git`, use the same qualified Python interpreter as its fresh v4 CPU pass, with the existing archive staged once at the shown relative path. Replace only the certificate path if root names its 77 receipt differently:

```sh
BUDDY_PHASE="$PWD/experiments_iclr/postsubmission_20260930"
PYTHONPATH="$PWD/.gnnm_runtime/buddy_extra_v1" python -B "$BUDDY_PHASE/buddy_complete_data_cache_preparation_v2/prepare_cache.py" \
  --repo "$PWD" \
  --archive "$BUDDY_PHASE/buddy_official_archive_staging_v1/root_transfer_v2/collab.zip" \
  --dataset-root "$BUDDY_PHASE/buddy_complete_data_cache_preparation_v2/root_run_77_v1/dataset" \
  --cache-output "$BUDDY_PHASE/buddy_complete_data_cache_preparation_v2/root_run_77_v1/cache" \
  --receipt "$BUDDY_PHASE/buddy_complete_data_cache_preparation_v2/root_run_77_v1/QUALIFICATION.json" \
  --qualification "$BUDDY_PHASE/buddy_cpu_numerical_qualification_v4/root_run_77_v1/CPU_QUALIFICATION.json"
```

Local author check:

```sh
python3 test_staging.py
```

All six tests use only temporary synthetic archives/modules and stdlib source inspection. They verify allowlist-only extraction and untouched test content, exact hashes, archive identity/fresh output refusals, traversal/duplicate/symlink/nonregular refusals, repository confinement, and qualification ordering without runtime patches. They do not execute the full numerical/data assertions. All-arm resource qualification/admission and the genuine completed 15-cell locked family remain later gates. No final test staging is provided in this package.
