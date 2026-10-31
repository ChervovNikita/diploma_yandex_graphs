# Actual 18.77 BUDDY resource/family launcher v2

Source packet only. No remote call, GPU/Torch/native model/data execution occurred while authoring. This wrapper delegates the sealed, unchanged BUDDY v5 resource and family entry points. It verifies the actual repository, base interpreter, two physical GPU UUIDs, sealed source bytes, isolated dependency RECORDs, actual 77 seven-test certificate/execution receipt and complete official common-cache qualification. Cloud Torch2.1.2 success does not satisfy the actual Torch2.7.1 check.

Use the actual repository `/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git`, base Python `/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python`, and process-local `.gnnm_runtime/buddy_extra_v1/site`. The exact source manifest is v4 `c479cedbd244ff645c7ee822625120b9f809d5082f4d9d4ace265237b5712e0f`. Certificate and common-cache paths are frozen in CONTRACT.json. No numeric GPU index selects a physical device: child environments bind only the full verified GPU UUIDs; `cuda:0` is the resulting local logical device.

From that actual checkout, root can inspect the plan without compute:

```sh
/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python -B experiments_iclr/postsubmission_20260930/buddy_gpu77_resource_family_launcher_v2/launch77.py resource
```

After actual 77 numerical qualification and the staging wrapper v3's complete cache qualification, run all five resource passes:

```sh
/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python -B experiments_iclr/postsubmission_20260930/buddy_gpu77_resource_family_launcher_v2/launch77.py resource --execute
```

The resources are sequential, alternate the two UUIDs, and each uses seed 0 for one complete training epoch plus validation forward only. The source resource mode scores no validation quality, selects no checkpoint, and loads no test arrays. CPU intra-operation threads are verified at four with OMP/MKL/OpenBLAS/NUMEXPR environment settings. All five arms use the same cache. Existing jobs remain active; no scheduler, SSH, kill, stop or dependency-install action is provided. Existing output folders are refused. An unsuccessful resource pass produces a failed receipt and blocks family work.

The exclusive `root_resource_v1/RESOURCE_RECEIPT.json` binds every completion/identity/epoch ledger and actual command wall time, each GPU UUID, peaks, source/cache/runtime identities and preflight/total wrapper timing. Its epoch/validation forecast is 300 times the sum of the five measured training-plus-validation passes, for 15 cells of 100 epochs. Include cold cache build, staging/qualification, source/file hashing, cache reading, resource passes, setup, family checkpoint/metric overhead and later final scoring in total work/cost accounting. The forecast excludes the latter fixed/unmeasured costs and is not a complete wall budget. Timing and memory reflect contention with the other authorized jobs; these observations do not establish isolated speedup.

The wrapper exits after resources. Root then reviews the actual cost and writes a separate, prospective repository-confined admission JSON with schema `buddy77-prospective-family-admission-v1`, decision `admitted`, source/wrapper/cache/certificate/resource-receipt SHA256 identities, `family_cells:15`, `optimizer_fits:24`, `cpu_threads:4`, ordered `physical_GPU_UUIDs` from CONTRACT.json and a nonempty `root_resource_cost_decision`. The wrapper does not create this admission or start the family automatically. Root should record the complete budget/limitations decision in that field.

After that separate root admission, launch the fixed family:

```sh
/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python -B experiments_iclr/postsubmission_20260930/buddy_gpu77_resource_family_launcher_v2/launch77.py family --execute --admission experiments_iclr/postsubmission_20260930/buddy_gpu77_resource_family_launcher_v2/ROOT_ADMISSION.json
```

This delegates the sealed `launch_family.py` to two UUID-bound slots, with four CPU threads per child, for exactly 15 arm/seed cells and 24 optimizer fits. All 100 epochs and 1,500 validation selections remain in the unchanged source. Existing output folders are refused. Failed cells leave the family incomplete and require root replay review. A launch receipt records admission/resource/context identities, argv, exit status and elapsed work; it does not assert a family lock or final-test result.

Torch2.7's official OGB/PyG data loads use the explicit process-local `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` compatibility environment. The conflicting force-weights-only environment is removed only in new child dictionaries. Source checkpoint loads explicitly retain `weights_only=True`. No scientific source/vendor/recipe or global environment is changed. There is no test/finalization/heldout command in this packet.

V2 preserves the sealed v1 packet and its first actual77 failure. Its only executable changes are source version/SHA, numerical certificate path, data-wrapper/cache path and fresh packet resource/family paths. Queue/admission/UUID/thread/cost behavior is unchanged. V5 calls set_device before peak reset; new v5 cache/resource/family success is not claimed by this source packet. Root separately reruns exact-v5 qualification and rebuilds the common cache before resource replay.
