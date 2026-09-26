# Roman Empire additional masks and depths

This directory records the complete post hoc study of official Roman Empire
masks 1–4, residual SAGE depths 2 and 5, three optimization seeds, and two
paired architectures. It contains all 48 selected result rows and 1,000-epoch
validation traces, the frozen source and input hashes, the all-seed CUDA smoke
record, eight CUDA checkpoint replay audits, the complete-grid audit, and
class decisions derived from the selected validation and test logits.

The tied arm uses one residual SAGE propagation stack for four BatchEnsemble
members. The untied arm starts from the same tensors and training RNG state
but trains a separate copied propagation stack per member. Both retain the
same boundary BatchEnsemble projectors. The graph has 65,854 symmetrized
directed entries and no added self-loops. The recipe is width 128, four
members, 1,000 full-batch AdamW epochs, and validation selection by pooled
accuracy, then pooled cross-entropy, then earliest epoch. The published test
mask is scored after checkpoint selection.

Run `verify_roman_multimask_compact.py` with Python, NumPy, and the public
Roman Empire NPZ if available:

```bash
python verify_roman_multimask_compact.py --public-npz /path/to/roman_empire.npz
```

The verifier checks every source and result hash that remains in this
package, all 48 validation traces and decisions, official mask alignment,
paired initialization records, eight replay-audit summaries, and all 24
paired test-accuracy differences. It writes
`local_compact_score_audit.json` only after all checks pass.

To rerun training, copy the source files and freeze manifest into a clean
directory. Obtain the public NPZ whose SHA-256 is in
`ROMAN_MULTIMASK_FREEZE.json` and place it at `data/roman_empire.npz`.
Use the library versions listed in each source manifest, then run
`python roman_multimask_orchestrator.py --stage preflight` followed by
`python roman_multimask_orchestrator.py --stage production` on a CUDA machine.
The orchestrator assigns physical GPU 0 and runs depths sequentially.

The public raw NPZ, initial logits, selected floating-point logits, and
checkpoints are omitted to keep the anonymous upload small. Each omitted
artifact's original SHA-256 is retained. The saved class decisions permit
independent accuracy checks but do not reproduce pooled logit values,
cross-entropy, or CUDA checkpoint replay. The full author archive retains
those artifacts and passed the eight CUDA replay gates before this package
was derived.

All masks share the same graph and most nodes. The four masks and three
optimization seeds are repeated measurements on that graph, not independent
graph datasets. This extension was designed after earlier Roman results
were known, so it is descriptive evidence, not a preregistered confirmatory
test.
