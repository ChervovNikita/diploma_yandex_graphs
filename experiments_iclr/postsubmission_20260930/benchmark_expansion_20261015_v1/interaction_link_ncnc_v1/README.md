# Four explicit NCNC interaction recipes

This source ports the four fixed NCN interaction recipes to actual NCNC. It
constructs one original native boundary session and assembles tensor helpers
directly. No standalone transform chain or source operator replacement occurs.

| Method | Private components | Shared/private credit | Serving |
|---|---|---|---|
| `interaction_geometry_input` | BE + portable input correction + original endpoint contrast | shared F; private F+0.05A | Mean raw logits |
| `interaction_responsibility_private_pool` | BE + corrected endpoint responsibility v2 | shared F; private 0.5W+0.5L_mean | Mean raw logits |
| `interaction_input_meanJ` | BE + portable input correction | all-block 0.5F+0.5L_mean | Mean raw logits |
| `interaction_input_conditionalJ` | BE + portable input correction + conditional20 | all-block 0.5F+0.5L_actual | Actual gated raw logits |

F retains all65,536positive+65,536negative original top-level task queries in
each of4routes, using the original separate-group mean sum. Every native
recursive completion remains part of that computation graph. The original
TaskCallScope still owns recursive dispatch and depth-one output-bank scope.
The passive observer authenticates actual source rows and captures only
depth-one task preheads/logits. Recursive rows are not auxiliary or aggregation
objects. The observer's task/recursive counts must match native routing receipts.

The input correction and all coefficients match the NCN/node/molecular recipe:
rank8, encoded width64, isolated QR seed=fitseed+901003, zero up initialization,
4,096 private Parameters appended after retained BE at encoder.convs.0.lin.
They join the original encoder Adam group atLR0.0082. The conditional readout's
20 Parameters join the native head group atLR0.0037. Its detached descriptor
law and0.05-per-route floor are unchanged. No new optimizer group or source
forward is added. Source classes/Parameters, mask, sampler, recursive operators,
route/privateRNG, original Adam/options and topology stay native.

I1 uses the original endpoint link contrast with PRIVATE credit, not a copied
node objective or all-block contrast. It fixes temperature0.2, coefficient0.05,
up to1,024 existing anchors and16 compatible existing opposite-group alternatives.
I2 uses the corrected link-resp-v2 semantic endpoint hash/floor0.25 and native
within-group normalization; no older responsibility law is substituted. I1/I2
collect the two declared parameter-block gradients at one old state and then
perform one source Adam transition. I3/I4 use one audited J scalar backward.
Prediction gates apply only to the original top-level task outputs; they do
not replace or reroute native recursive completion predictions.

Checkpoint/restore pins exact components, data, source, seeds, ownership,
last native query/contrast/weight receipts and complete model/Adam/sampler/
TaskCallScope/private-globalRNG/modes/epoch state. Endpoint reconstruction uses
an index on each tensor's own device, then CPU authentication. Full VALID keeps
all60,084positive+100,000negative records in native order and verifies original
task-only/recursive output scope. Same-call selected-bank arithmetic means are
returned/cached with the current update; the zero-forward getter requires fresh
VALID and is invalidated on TRAIN/restore. Production selected-checkpoint export
remains a separate runtime gate.

Status: source/local algebra and recursive-capture helper checks only. The NCN
native receipts are lineage, not qualification of this NCNC port. A finite
real-data CPU smoke proposal is separate and must be approved/admitted before
execution. No TEST, GPU update, scientific fit, accuracy/convergence result or
novelty claim. CUDA/resources/full-worker/final TEST gates remain pending.
