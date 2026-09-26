# Compact selected/default hard-decision export

This postfreeze transport was specified before the 432-cell study produced
test scores. It is not a tuning candidate, checkpoint rule, or model change.

Only after the complete validation lock, independent selected/default test
score audit, and strict per-node logit/argmax replay pass, export one compact
hard-decision file per allowed score. For each file, derive the pooled class
from the argmax of **pooled raw logits**. Derive each member class from that
member's raw logits. Do not substitute a vote over member classes for the
pooled class. The BASE arm has one member, and other arms have four.

Store the fixed official test indices and full public graph label vector
once per graph. The latter allows a reader to verify its SHA-256 against the
pretest frozen graph fingerprint and recover the test labels by indexing.
Store class IDs as signed 16-bit integers after checking that class count
fits. The test indices and full label vector retain signed 64-bit dtype so
their tensor fingerprints can be checked exactly. Keep all four graphs,
six arms, three paired seeds, and both the validation-selected and
predeclared default candidate where distinct. Exported hard decisions must
reproduce each scored test accuracy and the independent audit value.

The public compact verifier checks the 432 validation traces and lock,
the score/audit identity chain, frozen test index and label fingerprints,
and the accuracies calculated from these class IDs. The omitted float
logits and selected checkpoints are required to independently replay model
forward passes and prove that each hard class came from those logits. The
author evidence retains those files and the strict decision audit records
zero mismatches on the complete allowed set.
