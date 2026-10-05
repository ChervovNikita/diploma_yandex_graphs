# Citeseer-HeaRT acquisition successor: source only

This is a minimal successor of `pubmed_heart_acquisition_native_plan_20261004_v2/acquire_pubmed_on_server.py`. The predecessor remains unchanged. No server connection, archive/data download, scientific import, training or numerical execution occurred in this preparation.

The successor pins the authorized one-GPU repository `/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs`, its `experiments_iclr/postsubmission_20260930` phase, Linux, an explicitly operator-bound expected hostname and the exact singleton UUID `GPU-44039938-fd82-41d2-fefd-de71514e2fac`. The intended SSH route is `anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru:2222`. UUID inspection is metadata only; there is no CUDA computation. Guards run before networking or output creation. The expected hostname must be supplied from root's independent verified route observation; deriving it within the same invocation would defeat its binding.

The archive is the same official Zenodo record 22184581 / `HeaRT.tar.gz`: 880240878 bytes, MD5 `d5086c7456ee98892af00ce00816aba1`, additional prior full SHA256 `7b7042476319a353bdb6c50b5f402b89b9006a2fde2d1258b7adcbd1d22629ba`. Current official metadata must still agree with the exact size/MD5. The entire archive must then satisfy all three byte/hash pins before extraction. Metadata and script digest are retained in receipts.

Root must choose exactly one archive policy:

- `--authenticated-archive /absolute/repo/path/HeaRT.tar.gz` rehashes that explicitly known regular archive. It must remain inside the pinned repository and cannot be a symlink. It is read without relocation or permission changes, and no archive download occurs.
- `--no-known-authenticated-archive` explicitly records that root has no exact known authenticated archive in this repository and permits one download to the new output's `withheld_inputs/HeaRT.tar.gz.partial`. The partial is renamed only after authentication. There is no fallback from a failed explicit reuse to download, no retry, no resume and no overwrite.

Only six regular archive members ending in one common `dataset/citeseer` prefix can be extracted: `train_pos.txt`, `valid_pos.txt`, `test_pos.txt`, `heart_valid_samples.npy`, `heart_test_samples.npy`, `gnn_feature`. Archive traversal rejects path traversal and absolute member names; selected links, duplicates, mixed prefixes or oversized content fail. Existing resource/time bounds are preserved. No other dataset is extracted.

Four TRAIN/VALID files go to `available/citeseer`. TEST positives and TEST pool go to `withheld_inputs/withheld_citeseer`; the whole archive and complete operator receipt remain unavailable to the planned training-loader arguments. This is a path/program contract, not OS-enforced isolation. There are no filesystem namespaces, mounts, permission/account changes, settings changes or sudo calls.

The utility computes exact selected-file hashes, positive counts/endpoint population/disjointness metadata and integer NPY **headers only**, requiring `(native_nonself_positive_count, 500, 2)`. It does not unpickle or interpret `gnn_feature`, load pool arrays or import numpy/torch/model code. Supplied feature key/shape/dtype/donor, feature equivalence and row/endpoint association remain unresolved. Successful acquisition explicitly remains `not_admitted`, with `feature_provenance_admitted=False`; it does not authorize fitting.

`ACQUISITION_START.json` preserves initial authority/policy. Success writes `AVAILABLE_MANIFEST.json` and `withheld_inputs/operator_receipt.json`. Acquisition exceptions, including keyboard interrupt, write `FAILURE_RECEIPT.json`, retain partial/selected files and perform no retry. Failures before repository/host/GPU authority succeeds produce no filesystem write; process exit output is the root's evidence for those guards. A hard kill cannot create a final receipt, but the initial receipt and partial files remain. Root should monitor the same process and not relaunch after an observation timeout.

## Operator template

Stage the reviewed script inside the pinned phase and run with ordinary Python. Bind a fresh `--output-relative` if the default is already present. Replace `VERIFIED_HOSTNAME` with the independently observed authorized hostname, then choose one of the two archive-policy arguments described above:

```text
python3 -B <PINNED_PHASE>/citeseer_heart_acquisition_source_20261005_v1/acquire_citeseer_on_server.py
  --expected-hostname VERIFIED_HOSTNAME --confirm-authorized-one-gpu
  --no-known-authenticated-archive
```

Source preparation inspected the exact successor and performed AST syntax parsing only. No utility import, execution, dataset simulation or server action was used. Runtime acquisition, actual data identities and environment qualification remain root-owned work.
