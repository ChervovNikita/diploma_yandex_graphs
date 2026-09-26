# Capacity and width sensitivity: public record

This anonymous record contains two exploratory studies on the frozen graph splits:

- `narrow72`: four graphs, six optimizer candidates, three seeds, and an UNTIED width chosen on each graph to approach the parameter count of TIED width 128. Every one of the 72 cells trained for 1,000 epochs.
- `wide18`: WikiCS split 0, the same six candidates and three seeds with UNTIED width 128. Every one of the 18 cells trained for 1,000 epochs. This control was designed after earlier outcomes were inspected.

The studies test sensitivity to width and parameter count together. They do not isolate a causal parameter-count effect or establish a new benchmark ranking. The full selected and declared default comparisons, including unfavorable rows, are in `narrow72/CAPACITY_SENSITIVITY_COMPARISON.json` and `wide18/WIKICS_WIDTH_CONTRAST.json`.

## What is included

`CELL_RECORDS.tar.xz` contains the original result JSON, all 1,000 validation-trace rows, and validation companion manifest for every one of the 90 cells, plus the original score JSON for every one of the 27 permitted selected/default test scores. Study freeze, validation lock, final independent score audit, training source, and verification source are included outside that archive. `MANIFEST.json` binds each record and projected array by SHA-256.

For every validation cell, `validation/` stores the exact pooled hard classes and four member hard-class vectors derived from the original float32 logits. For every permitted test score, `test/` stores the **exact float32 pooled logits** and four member hard-class vectors. `references/` stores labels and split indices, bound to the frozen graph descriptor by tensor hashes. The included verifier recomputes validation selection from all six candidates and three seeds, checks every trace and hard-class accuracy, and recomputes test pooled accuracy, cross-entropy, and member accuracies.

The public projection cannot recompute validation cross-entropy from hard classes or rerun the trained models, because full validation logits, member float logits, and checkpoints are absent. The complete source arrays and checkpoints were independently replayed and retained in the author evidence. `FULL_TO_COMPACT_ARRAY_AUDIT.json` records exact source-to-projection equality for all 90 validation cells and 27 test scores; its audit script is included. The compact verifier checks archive, lock, score, frozen data, and projection identities without those source files.

## Verify

With Python and NumPy, from this directory run:

```sh
python verify_narrow_wide_compact_v2.py .
```

Expected scope: `PASS`, 90 validation cells, 27 test scores. This checks the compact record and its stated numerical replay scope; it is separate from retraining the models.
