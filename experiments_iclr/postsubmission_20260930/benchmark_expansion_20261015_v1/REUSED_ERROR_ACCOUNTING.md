# Reused error accounting and current hypotheses

This note reads saved aggregates from the closed
`Actual_GraphSAGE_common_wrapper_complete39_closed_interpreted_20261010_v1`
ledger record via `prior_result_index.json`. No checkpoint, raw predictions,
TEST labels, model or new training was opened. These are old exploratory
results under their original controller, not the new benchmark or a new win.
The current native-preserving boundary design also differs from that historical
all-map wrapper, so those scores cannot substitute for new controls.

The retained shared baseline already had higher mean accuracy than ordinary
I4 in that particular study: 80.5714% versus80.0215%, while its NLL was worse:
1.0015 versus0.6958. Consequently, “every shared ensemble has weak members and
therefore cannot win” is not supported by the history. Nor does that small
three-trial result establish a novel or broadly better method.

## Exact decomposition from saved integer aggregates

For each evaluated node-instance, record whether any member's argmax is correct,
whether the pooled argmax loses such an available correct answer, and whether
the pool is correct despite every individual argmax being wrong. Then:

`pooled_correct = any_member_correct - lost_available_correct + pool_only_correct`.

| Historical arm | Any member correct | Lost by pooling | Pool-only correct | Pooled correct |
| --- | ---: | ---: | ---: | ---: |
| Unchanged shared4 |13740|993|1|12748|
| Ordinary independent4 |13508|847|0|12661|
| Factorized independent4 |13506|835|0|12671|
| Exchange candidate |13032|487|0|12545|

Each identity holds exactly in the saved record. Counts sum over the three
readouts;15,822 node-instances are not15,822 independent statistical replicates.
For shared4 versus ordinary I4, the difference is232 extra covered instances,
minus146 extra cancellations, plus1 pooled-only repair:87 net correct instances.
The shared baseline did not merely lack alternatives; it had more available
correct alternatives while losing more of them at pooling and producing worse
probabilities. The failed exchange candidate reduced cancellations but destroyed
more alternatives, so a better cancellation count alone did not imply progress.

## Consequences for the new study

- Measure both individual competence and available alternatives. A stronger
  pooled prediction does not require every route to beat a native single.
- Test a learned combiner only on authentic saved/current member information,
  with TRAIN supervision and the same-bank mean as its ablation. A union of
  correct member argmaxes is not a general pooling upper bound: the recorded
  shared baseline itself has a pooled-only correct instance.
- Test private capacity and task-aware supervision separately and together.
  Enlarging feature spread or reducing disagreement is not a useful endpoint;
  the result must retain and combine better task predictions.
- Keep proper loss alongside accuracy. Temperature fitting may address some
  overconfidence but is not architectural novelty or an accuracy improvement.
  Those older NLL values do not prove that all of their difference is calibration.
- Reassess under full native training and independently owned reference fits.
  Do not relabel the old exchange failure, transfer its checkpoint, or treat
  the new molecular pilot as confirmation of the historical node result.

The current MolBACE pilot and future link diagnostics therefore save native
member logits and features, rather than reporting only a diversity scalar.
This note proposes mechanisms to investigate; it supplies no new performance
or significance claim.
