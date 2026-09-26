# Packed trace records in the upload

The upload stores every original result and validation trace in `CELL_RECORDS.tar.xz`, with unchanged bytes. Run `python3 verify_trace_archive.py` first. It checks the original hashes, unpacks to a temporary directory, and runs the unchanged original validation verifier there. The pooled-float companion check runs directly. In a full author stage with `results/` already present, the same wrapper directly runs the original verifier. The original source below is retained for inspection and for an unpacked copy.

# Anonymous experiment evidence

This folder contains one complete frozen 216-cell six-arm study. Read
`STUDY_PROTOCOL.md` for its graph setting, post hoc status, model, and
selection rule. `FROZEN_STUDY.json` records source/data fingerprints.
`results/` includes every 1,000-epoch validation trace and selected
checkpoint metadata; the complete independent validation lock records all
six candidates per arm and condition. `scores/` contains only the locked
selected and fixed-default test score metadata. `hard_decisions/` includes
audited pooled/member class decisions and public test-label references.
`summary/` provides descriptive tables for all arms and both score roles.

Run these commands in this folder with Python 3 and NumPy installed:

```sh
python3 verify_trace_archive.py
python3 verify_pooled_float_companion.py
```

The first check replays the validation selection from all traces and
recomputes test accuracy from hard classes. The second verifies the
post-score `pooled_float_companion/` add-on: every allowed checkpoint's
exact float32 pooled test logits, pooled decisions, test accuracy, and
cross-entropy. The add-on is linked to the original compact manifest; it
does not alter the frozen training or test-selection records.

Trained checkpoints, raw datasets, and full per-member float logits are
omitted to keep the submission small. This public folder verifies the
recorded selection and scores. A fresh training run uses the included
source, its dependencies, and the cited public dataset. An independent
forward replay of the **original trained checkpoints** additionally needs
the omitted author weights.
