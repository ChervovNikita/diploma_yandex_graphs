# Selected molecular VALID analysis

`selected_molecular.py` analyzes one completed own fit from the frozen24-fit
pilot. It supports a native single body or one four-route GNNM bank, including
the auxiliary/input-residual arms. It constructs the exact native adapter,
loads that fit's **selected** checkpoint and runs only the original whole VALID
role. It never trains, changes selection or opens TEST. Every native I4 member
can be analyzed independently through its own fit directory.

```sh
python -B analysis/selected_molecular.py \
  --project-root PROJECT_REPOSITORY \
  --fit COMPLETED_OWN_FIT_DIRECTORY \
  --output FRESH_ANALYSIS_DIRECTORY
```

Execution requires the authorized dedicated one-GPU allocation. Do not run this
on the Mac, on77 or on the restricted allocation. Root owns GPU scheduling.
All deliberate inputs/outputs must remain in the project repository. An
existing analysis directory is rejected; earlier diagnostics remain intact.

## Authority and reconstruction

The command requires `JOB.json`, complete `RESULT.json`, `selected.pt` and
`selected_VALID.pt` from the same own fit. It verifies their job identity and
checkpoint/export hashes. It verifies the frozen runtime/native dependency
receipts, reconstructs the actual session using the job's native configuration
and pinned extension, restores its selected full state/RNG, and sets eval mode.
Creating the compatible optimizer is needed to restore the complete source
checkpoint; **no optimizer step or TRAIN callback executes**.

The original VALID loader/skip law is retained: skip only batches with one
total node, not every singleton-graph batch. Its graph traversal must equal the
official VALID IDs/order exactly. Evaluated IDs/labels must also align exactly
with the saved selected VALID export. Misaligned roles, identities or tensor
shapes fail rather than being silently repaired.

Native preheads are captured with a passive graph-head input hook. Four-route
models return their already available native graph preheads and raw logits in
the original route order. No new projector, graph operation or extra forward
view is introduced. The representation is the graph prehead **before the
output boundary R factor**.

## Outputs

| File | Contents |
|---|---|
| `ROUTE_VALID.pt` | Authentic route raw logits `[M,N,1]`, pooled raw logits, native graph preheads `[M,N,D]`, graph IDs/labels and selected-checkpoint receipt |
| `RESULT.json` | Source/job/checkpoint hashes, metrics, error overlap, class-wise geometry, physical cost/count records, skip counts and pooled-logit sanity check |
| `REPORT.md` | Compact readable metric table and interpretation limits |

Artifacts are created on the allocation in the specified analysis folder.
The fit, runtime, source, frozen pilot and selected checkpoint remain unchanged.

## Prediction and error diagnostics

- Per-member and pooled ROC-AUC use the source OGB evaluator. BCE and Brier use
  observed native binary targets; missing labels remain excluded.
- Threshold diagnostics use raw logit>=0. They include each member's errors,
  all-members-wrong/common errors, unique correct predictions, pairwise
  disagreement/double faults/error Jaccard/correlation, and pooled repairs and
  harms relative to every member. Zero-denominator correlations/Jaccard are
  reported as unsupported rather than fabricated numbers.
- A native one-member fit has no pairwise route comparison. Its raw exports
  remain available for later aligned comparisons with other own fits.

## Geometry diagnostics

Features are L2-normalized per graph with epsilon1e-12. Within each observed
class, the report gives native prehead norms, normalized population variance,
within-class scatter, entropy/participation ranks of the centered covariance,
and centroid separation. Class variance requires at least two distinct graphs.
The optional auxiliary's fixed std target0.5/sqrt(D) and variance epsilon1e-8
are reported as one descriptive reference for every arm. Pairwise route
statistics include same-graph cosine and centered linear CKA.

These quantify representation geometry. They do **not** establish predictive
improvement, diversity usefulness, generality or methodological novelty.

## Saved-export sanity check and costs

The newly pooled logits are compared with the fit's saved selected export using
defaults rtol1e-4/atol1e-5 (`--rtol`, `--atol`). The report preserves maximum
coordinate/RMS differences and the number outside that disclosed tolerance.
This is a diagnostic, not an exact-equality gate or a reason to retry training.
An outside-tolerance result is retained as `complete_with_sanity_warning` with
both the new metrics and saved selected metrics visible. Graph/label alignment
still requires exact equality. No selected state is rewritten.

Reports count actual forward attempts/completions, raw/eligible VALID batches,
skipped graphs, zero optimizer/TRAIN work, construction/forward/wall times and
own CUDA allocated/reserved peaks. The analysis source is hashed. No predictive
quality or model execution was checked on the Mac during implementation;
syntax/CLI checks do not substitute for this authentic allocation analysis.

All predictive statistics here describe VALID, which was used for checkpoint
selection. They are exploratory diagnostics, not independent heldout results.
