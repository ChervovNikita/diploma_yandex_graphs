# PPI spectral collection integration v1

This integration reuses six reproduced, tied P-only epoch25 checkpoints: optimizer seeds0/1/2 crossed with observed-node label fractions0.1/1.0. It performs no training or predictive scoring. It does not certify confinement of the older probe executions. The coordinator's reproduction comparison binds the supplied checkpoint bytes, masks, contracts, sources, and fresh fit traces; the audited probe code and recorded artifacts support P-only provenance, without proving an unobserved execution history.

The successful synthetic CPU gate is `../v2_abi_fix_01/CPU_VERIFICATION_run03.json` (file SHA256 `14cabc22d9b5d7aa50440f7c84761ce0b9a0da8343832426158b5ac553f9ceb2`). The v2 ABI repair's implementation manifest remains `d682d44bfd49b357c40888965687c5a5a322c92c6d95ca0e59950af7818c4c39`. Its source code, completed v1 proposal, spectral statistic, selection rule, and replay schema remain unchanged. This sibling supplies checkpoint/data/protocol binding only.

## Fixed graph identities and masks

The original `sharing_policy/ppi_probe.py` is bound at SHA256 `29b67a5ead28fa9f6fbfd98ce51e6480c3d9f1d2dd498858e5abd9c88ce844b6`. It fits P10 only: 25 epochs, ten complete graph steps per epoch, mean over four member BCE losses, AdamW(lr0.001, wd0, eps1e-8). Contract/epoch metadata, the audited fit code, fresh comparison, 25 trace rows of ten steps, and checkpoint optimizer step250 are checked. No older sharing decision is an input to the spectral rule.

Official training split indices use the immutable ID `ppi:train:NN`:

| Group | Official training split indices | Rule |
| --- | --- | --- |
| P10 | 2,3,4,7,11,12,15,16,17,19 | shuffle20 with Python Random700; first10, then sort |
| Q10 | 0,1,5,6,8,9,10,13,14,18 | remaining10, then sort |
| D4 | 0,1,5,13 | shuffle sorted Q10 with Python Random704; first4, then sort |
| E4 | 6,8,10,14 | next4, then sort |
| F2 | 9,18 | final2, then sort; IDs only |

Within each group the frozen collection order is ascending training split index. Every archived mask must exactly equal the original rule: CPU generator seed9701+index, `randperm(num_nodes)`, first `max(1,int(num_nodes*fraction))` nodes. All twenty mask tensors and contract fingerprints are checked. For D/E graph objects supplied to autograd, unobserved targets are physically replaced by NaN and the same boolean row mask is supplied. P/F target tensors are never supplied to the spectral collector.

## Prospective gates

1. `freeze_inputs.py` uses only the standard library. Before any new spectral gradient is collected it locks hashes of all six epoch25 checkpoints, mask files, contracts, fit traces, comparison report, synthetic gate, original source/dependencies, integration code, data manifest, and training provider file identity. It checks the exact P-only source/folds/fit recipe, reproduction comparison, and archived source hashes. It reads no checkpoint tensors, dataset, losses, or older selector decisions.
2. `collect_ppi.py gate` must run successfully in the existing repoenv CPU runtime, with intra/inter-op threads1. It verifies input bytes, strict source checkpoint loading, all masks, optimizer step250, fit trace step counts, and source function identity to the original tied wrapper and rank-zero adapter on the first D graph. It performs no gradient or loss collection. Its sealed `PRECOLLECTION_GATE.json` binds source states, D/E fingerprints, all training input/topology fingerprints, and the environment.
3. `collect_ppi.py collect` requires that prior gate and reconstructs it byte-exactly before collecting. Any mismatch fails. It collects all six cells in fixed order by default. For bounded coordinator parallelism, `--arm` may name one of the six locked cells; it still reconstructs the full six-cell gate. All six cells remain required for the planned study, with no pooling or parameter changes after outcomes.

The provider is opened only with `split='train'`. Only the processed training file is hashed/deserialized as data. Existing val/test processed files receive existence checks to prevent PyG processing. Download/process overrides raise on cache misses. No validation/test dataset is constructed, no held-out provider bytes are hashed, and no predictive metrics are computed. The official `train.pt` contains all training graphs in memory; only D/E target tensors are consulted downstream. F2 has no target fingerprint, loss, gradient, or diagnostic in this integration. Existing frozen metadata may describe other splits; those fields are not used.

The checkpoints must contain only `model.*` source keys and no private block keys. Strip exactly the `model.` prefix and strict-load the frozen original TABMModel, verifying every CPU float32 tensor remains equal. Source SAGE, three h128 blocks, multiplier1, LayerNorm, M4, and original neighbor biases are retained. No optimizer state is advanced; it is checked only as fit evidence. Collector preservation of state, gradients, modes, and RNG remains the verified v2 implementation.

## Fixed collection and outputs

Only the three `residual_modules.{0,1,2}.module.conv.lin_l.weight` matrices are collected. Each D/E graph/member contributes its own unscaled mean BCE over finite observed entries. The covariance is on output rows, using equal complete-graph weights. The frozen selector remains n0=1, positive discovery signed eigenvalue and positive E score, descending minimum score with discovery-index ties, cap16 per target, and rank0 allowed. Matching random/raw-gradient control bases and exact replay use the unchanged v2 code; random-control seed800 is fixed.

For each cell, save the raw float64 matrix record, an exact replay report, and a basis lock with all scores, basis/control bytes, ranks, source/state/input identities, and added parameter/buffer counts. All outputs use new directories and exclusive files. `RESULT.json` means collection and replay passed; it does not establish predictive utility, novelty, unbiased selected scores, or acceptance. Keep failures and use a newly versioned repair if any concrete implementation gate fails. F2 checks or final arm training require a later separately frozen protocol.

## Coordinator commands

Run from the phase directory. The local archived data manifest is byte-identical to the original probe manifest; the remote frozen provider cache remains under `new_tasks/data/ppi`.

```bash
python spectral_privacy/ppi_integration_v1/freeze_inputs.py \
  --data-manifest data_freezes/ppi/frozen_manifest.json \
  --data-cache new_tasks/data/ppi \
  --synthetic-report spectral_privacy/v2_abi_fix_01/CPU_VERIFICATION_run03.json \
  --output spectral_privacy/ppi_integration_v1/INPUT_LOCK.json
```

The prepared `INPUT_LOCK.json` is generated locally before collection; transfer it with the exact integration code. Do not regenerate or alter it after observing new gradients.

```bash
python spectral_privacy/ppi_integration_v1/collect_ppi.py gate \
  --lock spectral_privacy/ppi_integration_v1/INPUT_LOCK.json \
  --output spectral_privacy/ppi_integration_v1/precollection_gate_run01
```

After that gate passes:

```bash
python spectral_privacy/ppi_integration_v1/collect_ppi.py collect \
  --lock spectral_privacy/ppi_integration_v1/INPUT_LOCK.json \
  --gate spectral_privacy/ppi_integration_v1/precollection_gate_run01/PRECOLLECTION_GATE.json \
  --output spectral_privacy/ppi_integration_v1/collection_run01
```

Use the existing repoenv Python and coordinator-supervised CPU launch. The script sets both torch thread counts to1 and has no GPU code path. No server execution was performed by the integration author.

For bounded parallel launches, append `--arm seed0_label10` (or one of `seed0_label100`, `seed1_label10`, `seed1_label100`, `seed2_label10`, `seed2_label100`) and give each cell its own new output directory, such as `collection_seed0_label10_run01`. A single-cell result does not certify completion of the other five cells.
