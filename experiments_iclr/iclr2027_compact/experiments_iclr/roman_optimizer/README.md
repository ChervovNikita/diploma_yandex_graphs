# Roman Empire optimizer-history diagnostic, mask 0

This directory contains the complete audited 24-cell V3 source and compact
decision-level evidence for Roman Empire official mask 0, residual depths
2 and 5, three optimization seeds, and four arms: TIED, UNTIED, SYNC,
and NORM-SYNC. The matrix is a post hoc mechanism diagnostic on one graph.
The V1 and V2 numerical preflights were revised before any Roman24 model
was trained or test outcome opened; their chronology and training-only
diagnostic records are in `numeric_revision_evidence/` and pinned by the
V3 design record.

The V3 selected-checkpoint collapse check received a documented numerical
verification amendment after ten cells completed and a separate cell
finished training but stopped at the original `1e-5` GPU logit gate.
The failed checkpoint, epoch choice, 1,000-row trace, and initial logits
were preserved. Five validation-only GPU replays showed arithmetic drift
slightly above the old gate with equal decisions, while the CPU collapse
was bitwise identical. The prospective amendment, its exact source and
hash freeze, the diagnosis, and the transport-interruption record are in
`verification_amendment/`. All 12 selected SYNC/NORM-SYNC checkpoints
passed exact CPU member-logit/state checks and the declared GPU `1e-4`
diagnostic bound with equal decisions before test scoring. The applied
NORM-SYNC update norm gate was unchanged. This chronology is part of the
evidence; the study is descriptive on one graph and one official mask.
After the original once-only scores and full CUDA replay audit, a separate
12-checkpoint test-node audit checked the same CPU identity and GPU bound
for SYNC/NORM-SYNC compact inference. It did not change any selected
checkpoint or score.

The package includes the exact source/data/runtime freeze, all six CUDA
preflight records, the complete 24-checkpoint validation replay lock,
all 24 1,000-epoch validation traces and selected result rows, the
once-only test score rows, complete CUDA test replay audit, and class
decisions derived from saved validation and test logits. The V2 norm
primitive retains private AdamW moments, a virtual mean-gradient AdamW
reference, the original applied-update norm tolerance, and its fixed
bounded FP32 correction. The V3 preflight verifies the reference gradient
against an explicit member mean and compares it with matched TIED
gradients within stated numerical tolerances. It records the observed
TIED-versus-virtual first-update vector discrepancy.

Run the independent compact verifier with Python, NumPy, and the public
Roman Empire NPZ if available:

```bash
python verify_roman_mechanism_compact.py --public-npz /path/to/roman_empire.npz
```

The verifier checks source and retained artifact hashes, mask alignment,
all 24 traces and class decisions, the six numeric preflight records,
validation/test audit linkage, and all six predeclared paired contrasts.
It also checks the amendment hashes and the per-checkpoint CPU/GPU collapse
audit retained in the validation lock, plus the post-score test-node collapse
audit and its score-file links.
Contrasts recomputed from integer decisions may differ from the original
CUDA float32 accuracy report by at most `1e-5` percentage points; the
verifier records the observed maximum difference.
It writes `local_compact_score_audit.json` only after all checks pass.

The public NPZ, initial logits, floating-point selected logits, and
checkpoints are omitted to keep the anonymous upload small. Their original
SHA-256 values remain in the retained manifests. The compact decisions
permit independent accuracy checks; they do not reconstruct cross-entropy
or replay CUDA checkpoints. The full author archive retained those
artifacts and passed the 24-checkpoint validation and test CUDA replay
gates before this package was derived.

For a fresh replication, place the public NPZ with the frozen SHA-256 at
`data/roman_empire.npz`, use the frozen library versions, and run the
24-cell design with the documented amended selected-checkpoint collapse
rule. The supplied amendment overlay records the exact interrupted V3
chronology and expects its ten completed cells plus failed checkpoint;
it is not a clean-start launcher. The original frozen V3 sources remain
available for inspecting model updates, seed resets, and checkpoint
selection. Use non-optimized Python so assertion gates execute.

The two depths and three seeds share one graph and one published mask.
Differences are descriptive optimizer and depth sensitivity, not
independent graph replication or a general selection rule.
