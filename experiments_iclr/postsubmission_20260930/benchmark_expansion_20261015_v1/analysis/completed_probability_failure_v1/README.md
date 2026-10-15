# Completed probability-failure diagnostic

One CPU saved-prediction pass over **all 11 methods x 3 seeds** in the
completed33-bank/39-fit residual-only authority. Existing reports contain
aggregate BCE/Brier/threshold accuracy, but not these class/margin and
signed per-example loss diagnostics. This reader adds descriptive scalar
outputs and leaves the original report, serving and selectors unchanged.
No current structural fit or partial outcome is an input.

`run.py` reuses `analysis/residual_gate_feasibility_v1/run.py::verified` for
saved-bank custody. It gates the completed authority,39 fit metadata/export
receipts and27 selected route collections before importing torch. It loads
only saved VALID exports/route tensors on CPU. No checkpoint weights,
models, native inputs, forwards, optimizer steps or TEST records are read
or executed. I4 uses the exact original FP32 mean of four own-selected
exports; shared pools use their authoritative saved selected export.

## Frozen diagnostics

- Pooled and every native member's raw-logit and true-class-margin
  quantiles:0,1,5,25,50,75,95,99,100 percent; linear interpolation at
  `(n-1)q`. Also shown by observed class.
- Per-class mean BCE/Brier and fixed native threshold correct/wrong
  breakdowns (`positive iff logit>=0`). Loss sums/counts are retained.
- Mean and quantiles of probability assigned to the true label, including
  wrong predictions overall and by class. Empty groups are unsupported,
  without replacement or zero-filled means.
- Every graph's signed pooled BCE difference against same-seed I4 and
  residual F, for every method and seed. Both improvements and harms stay
  in the output, including each reference's own exact-zero comparison.
- Top1/top5 mean the largest positive contributions from **1/5 graphs**,
  fixed for every comparison. Concentration uses the total **signed**
  excess BCE, with all negative offsets retained. Ratios can exceed1.
  Net-zero or net-improved comparisons have no excess concentration;
  their signed totals, positive mass, negative offsets and graph-level
  differences remain explicit.

TRAIN has480 positives/730 negatives (39.67% positive); VALID has130/21
(86.09%). The actual TRAIN-support authority and observed VALID counts are
reported alongside class-conditional losses. This observed prevalence
difference does **not** establish pure label shift, explain paired excess
BCE, or authorize a prior correction/refit from VALID. No blanket
overconfidence explanation is presupposed.

Three-seed pooled/class BCE/Brier summaries remain descriptive. There is
no fitted calibrator, changed readout, new ranking score, bootstrap,
significance test, ECE interpretation or claim of causal explanation.
Original AUC/BCE values are retained for context; the diagnostic BCE/Brier
must match the original uniform-graph values within1e-12 numerical error.

## Root review and one bounded execution

Source preparation executed only stdlib/source/scalar checks, no saved
tensors. Root reviews the manifest, reserves host memory alongside current
worker growth envelopes, chooses a finite external watchdog and executes
once with two CPU threads and CUDA hidden:

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  <repository>/.venv/bin/python -B -u <this-folder>/run.py
  --manifest-sha256 <reviewed-manifest-sha256>
```

The source performs one finite pass:39 saved exports,27 route files,
33 banks of151 VALID graphs, and two fixed paired references per bank.
It starts no owner, queue, subprocess worker or automatic retry. Root owns
the outer timeout/resource decision; source preparation grants no launch.

Outputs are fresh `RESULT.json` and `REPORT.md` in this folder, or preserved
`GATE.json` on failure. Existing output prevents overwrite or retry. Main
JSON contains compact scalar distributions/concentration summaries,
original source/result/collection receipts, elapsed CPU wall time and peak
RSS, with zero model/forward/update/checkpoint counts. Graph IDs and all
9966 individual paired contributions are saved separately in allocation-only
`analysis_outputs/completed_probability_failure_v1/PER_EXAMPLE.jsonl`, with
its receipt in RESULT. Do not synchronize that graph-level artifact locally
or commit it. No tensor output is saved. Missing/failed inputs stop the
whole family without a partial diagnostic summary.
