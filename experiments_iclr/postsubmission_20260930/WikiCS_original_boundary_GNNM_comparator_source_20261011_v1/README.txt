Original-boundary GNNM WikiCS comparator — separate inactive source

This adapter reuses the exact saved native TABMModel and train_step_tabm
multiclass branch, with the original construction order and global sequential
dropout RNG. The saved Penn runtime was inspected for construction, selector,
checkpoint cadence and strict model/optimizer/global-RNG restoration. No Penn
outcomes, arrays or states were read. No fresh score, fit or runtime occurred.

The planned separate panel has the three paired WikiCS seeds7301/7403/7507,
same11701-node/442907-entry graph and TRAIN580/VALID5274 receipts, native width128,
depth2/dropout.2, AdamW lr.001/zero decay, four multiclass own-CE/4 backwards and
one update. It retains the5000-update ceiling and patience300. VALID is served
every update by native raw-logit mean/argmax; only strict improvements at
multiples of ten are saved. Improving off-cadence steps reset stale without
updating saved best. No probability-pool or every-update-checkpoint substitute.

Boundary R/S/B initialization remains original: first R is native-device
Rademacher, other R/S are ones, B is zero, W is Xavier and has no bias. Only
input/output boundaries are factorized. Native internal graph/FFN and the
original head/bias recipe remain. This is an architecture/recipe contrast;
it is not isolated placement, an exact reproduction of historical scores or
a generic claim about shared-member weakness.

Future entry: comparator.acquire_comparator(spec,root_guard). PLAN.json names
the exact source/input/amendment receipts and new admission callback. Root must
first explicitly amend the future roster and reader, then choose finite
resources and scientific gates. Actual host/GPU/source/runtime/input custody
and external owner exit/reap are required. This packet supplies no admission,
budget, amendment, calibration or automatic launch.

The raw-logit archive has its own named schema. Pooled-logit softmax is not
probability-mean serving and must not be silently passed to the E/Gaussian
probability-pool reader. Root owns the reader/calibration adaptation.

The current seven-arm21/39/105 proposal and v2 acquisition port remain unchanged.
SOURCE_BINDINGS.json pins their exact preserved files. This comparator is not
mixed into those counts or outputs. Preparation was source inspection,
AST/JSON parsing and receipt hashing only; no module/numerical imports, server,
arrays, experiments, admissions, paper edits or reviewers.
