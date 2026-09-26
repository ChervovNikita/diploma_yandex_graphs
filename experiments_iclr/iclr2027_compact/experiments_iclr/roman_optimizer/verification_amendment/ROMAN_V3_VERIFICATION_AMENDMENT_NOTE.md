# Roman V3 selected-checkpoint verification amendment

The Roman V3 design, source, preflight, model updates, 1,000-epoch schedule,
checkpoint selection, and NORM-SYNC applied-update norm gate are unchanged.
Ten of 24 validation cells finished under the original V3 source. The
depth-2/seed-2/SYNC cell trained all 1,000 epochs, then stopped while checking
whether its selected checkpoint could be reduced to the shared-stack inference
layout. Its selected epoch was 933. Its original checkpoint, validation trace,
initial logits, and failure log are retained byte for byte. No Roman24 test
score had been computed when this amendment was defined.

The original verification required the maximum GPU member-logit difference
between the private-stack model and its collapsed version to be at most
`1e-5`, and required equal pooled decisions. A validation-only diagnosis of
the frozen failed checkpoint produced these observations:

- The collapsed state is an exact copy of one of four bitwise-equal private
  graph stacks, plus the untouched shared projector and output tensors.
- CPU original and collapsed validation member logits were bitwise equal.
- In five GPU original-versus-collapsed replays, the largest member-logit
  difference was `1.1444091796875e-5`; every member and pooled decision
  agreed. Five same-model GPU replays varied by up to the same magnitude.
- The smallest original pooled top-two margin on the validation nodes was
  approximately `5.37e-4`, far larger than the observed GPU arithmetic drift.

The predeclared amended selected-checkpoint verification requires exact
state mapping, bitwise-equal CPU member logits and decisions, equal GPU member
and pooled decisions, and GPU maximum member-logit difference at most `1e-4`.
This `1e-4` diagnostic bound is the frozen V3 five-step SGD logit bound and
was chosen without test outcomes. GPU drift is recorded as measured. All 24
selected SYNC/NORM-SYNC checkpoints, including the ones originally accepted,
must pass the amended CPU and GPU checks during the independent validation
replay. The original 10 result files retain their original hashes and gates.

The completed failed cell is finalized from a copy of its existing selected
checkpoint, trace, and initial-logit file. The first-step moment audit is
recomputed from the frozen initialization and seed, because it was not written
before the original gate stopped. The source records this distinction. The
remaining cells start from the same frozen initialization with their original
seed resets and run the unchanged optimizer and checkpoint rule. The amendment
source and inputs are hashed before this work begins. A complete 24-cell
validation lock must be written before any held-out test score is opened.

This is a numerical verification amendment after seeing validation behavior.
It does not turn the one-graph, one-mask optimizer-history diagnostic into an
independent confirmation. All contrasts remain descriptive and both positive
and negative signs must be reported.
