# DDI paired development pilot source

Prepared prospectively from actual TRAIN cost before any DDI VALID predictive score. This separate source fixes 12 fresh 100-epoch fits: seeds 0/1/2 × native M1, F4 target-only, F4 joint and F4 separate. The original 500-epoch source remains preserved and disabled.100 epochs are a shortened development budget with no author-budget parity claim.

Read `PROTOCOL.md`, `BUDGET_AMENDMENT.json` and `INPUT_BINDINGS.json`. Model/core/support/native sources are reused by exact sibling pins; no old source is edited. Root-adopted complete TRAIN runtime epochs were 12.38 s target-only,226.07 s joint and 212.41 s separate. Linear TRAIN-only scales are 187.9 GPU-hours for nine F4 fits at 500 epochs and 37.6 for nine at 100; M1/VALID/replay and later-epoch/contended-throughput uncertainty are additional.38–40 pilot GPU-hours is preliminary planning, not a forecast guarantee.

`config.json` is release-disabled. Root creates an external release config changing only `release_enabled` and `root_admission`, including supervisor, current capacity evidence and caps. Root approval and empirical qualification are distinct; no full 500 qualification is asserted. Default requested caps are 0.30 CUDA allocator fraction,32 GiB host and 8 h per cell; root supervises host/wall/capacity conditions.

One concrete cell command, inside root's supervisor:

```sh
python -B /disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930/hlgnn_ddi_paired_development_pilot_preparation_20261004_v1/train_cell.py \
  --config /disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930/ddi_paired_development_pilot_execution_root_20261004_v1/ROOT_RELEASE.json \
  --arm joint --seed 0 --device cuda:0 \
  --output-dir /disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930/ddi_paired_development_pilot_execution_root_20261004_v1/fits/joint/seed_0
```

The entry point runs exactly one cell, all 1,700 native updates including the tail, VALID every 5 epochs with strict first-tie Hits@20 selection, and its own full selected replay. The allowed arms are `native_m1`, `target_only`, `joint`, `separate`; seeds are 0/1/2. The output path must end in that arm and seed. Outputs include native fit history/checkpoint/metadata and `cell_summary.json`; F4 cells also contain complete `paired_stream.jsonl`. No TEST, donor, early stopping, automatic family loop or tuning grid is supplied.

After all 12 cells are complete, invoke `compare_pilot.py` with `--family-root ABSOLUTE_PROJECT_FITS_ROOT --output-dir ABSOLUTE_NEW_PROJECT_REPORT_DIR`. It verifies all nine F4 streams and reports every fixed contrast. Missing/failed cells produce an incomplete-family record and no success-only mean. Native M1 differences use the seed blocks and do not claim F4 RNG-stream pairing.

The fixed continuation rule requires a positive joint-minus-separate mean, at least two positive seed differences, and a nonnegative joint-minus-target-only mean. Passing recommends separately admitted fresh 500-epoch competitive confirmation; it neither launches confirmation nor establishes significance, novelty, generalization or acceptance.

Only source/AST/JSON/hash/disabled-guard checks occur during this preparation. No model/numerical imports, training, VALID predictive score reads or server actions are performed.
