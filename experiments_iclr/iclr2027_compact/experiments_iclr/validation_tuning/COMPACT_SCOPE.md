# Frozen 432-cell study: compact public evidence

This folder accompanies the anonymous ICLR submission. The fixed study has
four public graph settings, six model arms, six AdamW candidates per
graph/arm, and three optimizer seeds: 432 validation-only training cells.
Every cell runs 1,000 epochs unless a declared nonfinite failure stops it.
The selection rule uses pooled validation accuracy, then validation
cross-entropy and fixed candidate tie breaks. Test inference is allowed
only after a complete 432-cell validation lock.

## What the compact folder supports

`verify_compact_tuning.py` recalculates every checkpoint winner from the
1,000-row validation trace, the three-seed candidate means and tie breaks,
all 24 graph/arm selections, and the four private-block family choices.
It checks the result/trace SHA-256 links to
`VALIDATION_SELECTION_LOCK.json`, the frozen source hashes, and the exact
selected/default test-score allowlist. The hard-decision export retains the
pooled class (argmax of averaged **raw logits**) and each member class for
every allowed score. It includes the official test node indices and public
full label vector once per graph, checked against the graph fingerprints
fixed before test scoring. The verifier recalculates test accuracy from
those class IDs using NumPy and compares it with the score JSON and
independent audit records.

Run this from the study folder after installing NumPy:

```sh
python verify_compact_tuning.py
```

The compact verifier does **not** recompute model forward passes or prove
from the omitted float logits that a pooled class came from logit averaging.
The full independent score audit and strict decision replay did that in the
author evidence before this compact export. Their record hashes and
per-checkpoint decision mismatch counts are included. Reproducing those
checks requires the selected checkpoints and float logits or a fresh run
from the source and public datasets.

## Files to retain in the public package

- Frozen source and protocol: `tuning.py`, `models.py`,
  `verify_tuning.py`, `STUDY_PROTOCOL.md`, `PRETRAIN_REPAIR.md`,
  `FROZEN_STUDY.json`, and the pretraining-aborted freeze record.
- The complete `results/` tree of 432 `result.json` and
  `validation_trace.csv` pairs, without `checkpoint.pt` files.
- `VALIDATION_SELECTION_LOCK.json`, all selected/default `score.json`
  records without `predictions.npz`, `FINAL_SCORE_AUDIT.json`,
  `STRICT_DECISION_AUDIT.json`, and the complete `hard_decisions/` tree
  with `HARD_DECISION_EXPORT_MANIFEST.json`.
- `export_compact_hard_decisions.py`,
  `HARD_DECISION_EXPORT_PROTOCOL.md`, `verify_compact_tuning.py`,
  `strict_score_decision_audit.py`,
  `summarize_tuning.py`, the summary CSVs and provenance, plus the
  pre-outcome postfreeze source-hash manifests.

The public package may include a compact anonymous inference profile table
if that separate timing run and audit finish. Its raw profiler JSON can
contain a GPU UUID and remains in author evidence.

Raw dataset copies, training checkpoints, float prediction arrays,
host-specific launch scripts, caches, and GPU-context logs are omitted to
meet the public upload limit and preserve anonymity. The public datasets
can be fetched by their documented upstream sources and checked against
`FROZEN_STUDY.json`. The study loader reads the complete published label
vector for integrity, but returns a training bundle without test indices
or labels until the locked scoring step explicitly requests them.

Each graph uses one official or published split. Three seeds repeat the
optimizer on that split. The graph settings are not independent samples
from a population of graph tasks, and the filtered Chameleon graph is a
related variant of the Chameleon source rather than an additional
independent graph family.
