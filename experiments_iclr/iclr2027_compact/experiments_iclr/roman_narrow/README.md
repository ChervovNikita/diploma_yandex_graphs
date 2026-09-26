# Roman Empire narrow UNTIED width sensitivity

This is the anonymous six-cell supplement for a parameter-count-matched
width sensitivity study on Roman Empire official mask 0. It uses depths 2 and
5 and optimization seeds 0, 1, and 2. At depth 2 the narrow UNTIED width is
68 (211,600 parameters) versus width-128 TIED (208,960). At depth 5 it is
65 (451,924) versus TIED (456,640). Widths were selected from parameter
counts alone, before narrow training. The width change also changes the
initial function. This comparison does not isolate a single causal factor.

The study uses the source files in this folder and the Roman24 source/data
package alongside it. All six cells ran for 1,000 epochs. Each selected
checkpoint passed independent validation replay before its test score was
opened. Each once-only test score then passed an independent CUDA replay.
The original validation verifier stopped on one near-tie initial member
decision under nondeterministic GPU summation. `original_validation_failure_redacted.log`
preserves the error without an identifying server path. The separate frozen
amendment permits an initial member decision flip only when both top-two
margins are bounded by twice the observed logit drift. It still requires
the original initial-logit tolerance, exact pooled decisions, exact initial
state and RNG checks, and unchanged selected validation and test decisions.
The original failure log SHA-256 is retained in the amendment freeze and
derivation manifest. No test labels or scores were accessed for that amendment.

`selected_evidence.npz` for each cell contains float32 pooled test logits,
pooled validation class decisions, and the four members' validation and test
class decisions. This package was derived
from the original full member-logit files after the CUDA audit. Their SHA-256
digests remain in each result JSON and `derivation_manifest.json`. The full
original logits and checkpoints remain in the author archive. The compact
package is meant to permit independent checking of labels, accuracy, test
cross-entropy, checkpoint selection, and every paired contrast while staying
within the conference supplement size limit.

Run the independent local check with NumPy installed:

```sh
python verify_roman_narrow_compact.py --roman24 ../roman_optimizer
```

Use `prepare_roman_narrow_compact.py` to rebuild this package from the full
author archive and the Roman24 compact package. See
`FRESH_ROMAN_REPRODUCTION.md` for the original layout, required artifacts,
and a separate fresh repetition procedure. The historical amendment and
the compact package are evidence for this run, not clean-start launchers.
