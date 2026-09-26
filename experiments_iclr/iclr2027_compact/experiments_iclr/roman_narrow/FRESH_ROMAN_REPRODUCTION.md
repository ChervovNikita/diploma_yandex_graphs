# Roman24 and narrow width control: verification versus fresh repetition

## Check the uploaded evidence without CUDA

The anonymous `roman_optimizer` folder contains the frozen Roman24 source,
all 24 validation traces, validation and test decisions, source and score
manifests, and a local NumPy verifier. The `roman_narrow` folder contains the
six-cell narrow width sensitivity source, traces, test pooled logits,
decisions, and its local verifier. From their common parent directory:

```sh
python roman_optimizer/verify_roman_mechanism_compact.py --public-npz /path/to/roman_empire.npz
python roman_narrow/verify_roman_narrow_compact.py --roman24 roman_optimizer
```

The first command requires the corrected public Roman Empire NPZ whose
SHA-256 is `a58ba741d123bf892fe5c872138d07463d75a2e9012360b8dd78ac2d4766d428`.
The second command uses the labels and mask anchor in `roman_optimizer`.
These checks reconstruct reported decisions and accuracies from the compact
files. They do not train models or replay checkpoints. The uploaded compact
folders omit the 24 Roman24 and six narrow checkpoints, initial logits, and
full member logits. Their hashes are retained in the manifests. The full
original artifacts were audited before this compact package was built.

## Reconstruct the original training layout

The frozen narrow source assumes this exact sibling layout:

```text
reproduction/
  roman_mechanism_v3_prepared/
    roman_mechanism.py, verify_roman_mechanism.py, tuning.py,
    roman_multimask.py, models.py, mechanism.py, norm_sync_v2.py,
    ROMAN_MECHANISM_PROTOCOL.md, ROMAN_MECHANISM_DESIGN_FREEZE.json,
    ROMAN_MECHANISM_SOURCE_FREEZE.json,
    data/roman_empire.npz, results/roman/..., test_scores/...
  roman_narrow_untied.py
  verify_roman_narrow_untied.py
  verify_roman_narrow_initial_amendment.py
  ROMAN_NARROW_UNTIED_PROTOCOL.md
  ROMAN_NARROW_UNTIED_WIDTH_LOCK.json
  roman_narrow_untied_width_lock.py
  roman_narrow_untied_prepared/
    SOURCE_FREEZE.json, preflight.json, results/roman/...,
    INITIAL_DECISION_AMENDMENT_FREEZE.json,
    initial_replay_all6_diagnostic.json, validate.log,
    validation_lock.json, test_scores/...
```

Use `PYTHONOPTIMIZE=0` so assertions run. The original Roman24 freeze records
the exact Python, NumPy, PyTorch, CUDA, and PyG versions. Both `check_freeze`
functions require those versions and exact source/data SHA-256 values. The
narrow freeze also requires hashes of the original Roman24 validation lock,
test audit, six TIED results, six checkpoints, and six test result files.
Therefore the uploaded compact folders alone cannot execute the frozen
narrow training or CUDA audit scripts. The full author archive must be used
for an exact replay of those frozen artifacts.

## Run a new scientific repetition

A new repetition should start from a new directory and a new source/data
manifest. Copy the model, optimizer, loss, epoch budget, and checkpoint rule
from the frozen source. Use the same public NPZ and mask. Run preflight,
then all 24 Roman24 cells, independently replay validation checkpoints,
write a complete validation lock, and only then score and replay all tests.
Run the six narrow cells with the same depth-specific widths, schedule, and
selection rule. Regenerate every freeze and baseline hash from this new run.
Do not reuse the original `SOURCE_FREEZE.json` or the narrow
`ORIGINAL_SOURCE_SHA` constant as proof of new outputs: a fresh run can have
different checkpoint bytes even when the method is the same.

The historical Roman24 verification overlay expects a particular completed
ten-cell prefix and one interrupted depth-2/SYNC checkpoint. The narrow
overlay expects its historical initial near-tie diagnosis and failed
validation log. They document the original audit chronology and are not
clean-start launchers. A fresh repetition needs a newly frozen numerical
verification rule before training and an independent validation/test audit.
If the strict original GPU collapse gate stops on harmless FP32 summation
drift, record the failure and pretest diagnosis before changing that rule.

No result from a fresh repetition is claimed by the original hashes. Report
its own full matrix, data/runtime manifest, and negative as well as positive
contrasts.
