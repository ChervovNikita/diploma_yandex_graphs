# Complete official collab staging/cache wrapper v3 for BUDDY v5

Source only. Wrapper v3 preserves sealed v2 and changes only its default BUDDY source directory and manifest/version bindings to newly sealed v5. All raw/processed/train/valid/cache mathematics, nine-member allowlist, installed OGB/archive pins, process-local Torch compatibility environment and six stdlib fixtures are unchanged. No real archive/data/Torch/GPU/remote execution occurred in this authoring step.

V5 adds only CUDA set_device before peak-memory reset in fit. Whole-source identity requires a fresh v5 seven-test actual77 certificate and rebuilt complete training-only common cache, even though cached scientific contents/recipe are unchanged. Prior successful v4 data/CPU evidence and the failed resource launch remain preserved separately. New v5 CPU/data/cache/resource results are not asserted here.

After root's exact-v5 actual77 all-seven pass, from `/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git`:

```sh
BUDDY_PHASE="$PWD/experiments_iclr/postsubmission_20260930"
PYTHONPATH="$PWD/.gnnm_runtime/buddy_extra_v1/site" /disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python -B "$BUDDY_PHASE/buddy_complete_data_cache_preparation_v3/prepare_cache.py" \
  --repo "$PWD" \
  --archive "$BUDDY_PHASE/buddy_official_archive_staging_v1/root_transfer_v2/collab.zip" \
  --dataset-root "$BUDDY_PHASE/buddy_complete_data_cache_preparation_v3/root_run_77_v1/dataset" \
  --cache-output "$BUDDY_PHASE/buddy_complete_data_cache_preparation_v3/root_run_77_v1/cache" \
  --receipt "$BUDDY_PHASE/buddy_complete_data_cache_preparation_v3/root_run_77_v1/QUALIFICATION.json" \
  --qualification "$BUDDY_PHASE/gpu77_buddy_numerical_qualification_v2/root_run_v1/CPU_QUALIFICATION.json" \
  --source-dir "$BUDDY_PHASE/buddy_shared_cache_execution_v5" \
  --cpu-threads 4
```

The already acquired archive is reused; no download/transfer is provided. Fresh repository-confined outputs are required. The wrapper verifies exact installed source, archive and numerical identities before allowlisted extraction, qualifies complete raw/processed/official weighted training-only correspondence before native hashing, validates the complete common cache, and writes one exclusive qualification receipt. Test.pt/split_dict.pt payloads remain closed. It calls unmodified v5 functions directly, with no monkeypatch, training, selection or GPU stage.

Torch2.6+ OGB1.3.6 compatibility remains process-local TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1. Explicit source checkpoint weights_only=True calls remain unchanged. All complete-data verification/build I/O and CPU work belong in the receipt/cost accounting. All-five resource qualification and separate prospective family admission are later root gates.
