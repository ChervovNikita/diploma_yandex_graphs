# Completed-metric interpretation, 30 October 2026

The audit does **not** support saying every completed configuration is healthy.
The primary metrics are finite, but one MolHIV configuration has extreme output
magnitudes, and original GNNM is consistently weak on the completed MolBACE GCN
cohorts. These are research findings to preserve and investigate. No training
job, descriptor, checkpoint, selected epoch or original score was changed.

## MolHIV: real extreme false-positive logits, not loss summation

The affected completed fit is `ogb-ogbg-molhiv-gcn-virtual-drop0.0`, seed 2,
`aggregation_global_globalJ_v1`, route-private normalization, on server51/gpu0.
Its selected epoch 152 has VALID ROC-AUC **83.3796%** and mean binary
cross-entropy (BCE) **194.9571** on 4,113 graphs. Selection maximizes ROC-AUC.

The source `conditional_aggregation_v1/molecular_transform.py:201–229` uses
raw logits, sums BCE over observed labels, then divides by observed-label count.
Thus 194.957 is a per-label mean, not an unnormalized dataset sum. The native
training criterion is `BCEWithLogitsLoss` in
`reference_wrapper_redesign_20261013_v1/smoke_20261014/ogb_port/session.py:226–234`.
The selector receives validation metrics, not loss, in
`server51_campaign_port_v6/composed/runtime/run_fit.py:231–240`.

An exact-hash check followed by CPU-only reductions of this existing selected
VALID export found:

- 4,032 negative and 81 positive graphs; all exported logits finite.
- Logit minimum −16,978.77, median −12.96, maximum +174,114.72.
- Ten negative graphs contribute **86.13%** of total BCE, through extremely
  confident false-positive predictions.
- Mean BCE among negative graphs 198.777; among positives 4.808.
- The same selected bank's uniform mean is also unstable: BCE 294.218 and
  maximum logit 274,700.63. The learned aggregation alone is therefore not an
  adequate explanation.
- Brier score is 0.04016. Its bounded scale explains why a few huge logit
  outliers affect BCE much more strongly; it does not establish good calibration.

The saved 202-epoch curve confirms persistence: validation BCE ranges from 1.49
to 296,428.05, median 551.57; 197 epochs exceed 10. At selected epoch 152, the mean
training update objective is 0.02484. Training and validation losses have
different aggregation semantics, but this gap warrants investigation.

The runtime metadata confirms **13 route-private BN modules**, including all
eight virtual-node MLP BN modules and five GNN layer BN modules. This is not a
case of accidentally omitting the virtual-node BN modules from the declared
private-state method. There is no evidence in this audit that the ranking metric
or its record order is wrong. The exact cause of the extreme outputs remains
unresolved; graph-size sensitivity, virtual-node updates, normalization state
and learned boundary scaling require controlled diagnosis. Do not silently
clamp logits or replace this run. Compare matched native/private-BN controls
when available and inspect the already identified outlier graphs before
proposing a separately frozen repair experiment.

`molhiv_high_loss_RESULT.json`, `molhiv_high_loss_CURVE.jsonl` and
`molhiv_high_loss_VALID_DIAGNOSTIC.json` preserve the evidence. The exported
artifact SHA256 is
`f78419bf7a78fb2174b698cd8a4ce3294d60525119f234434a441ae305167d0c`.
The CPU diagnostic loaded only this 194,284-byte VALID artifact, with zero model
forwards, optimizer steps or TEST access. Gate parameters are not stored in this
export; no full checkpoint was loaded to obtain them.

## BUDDY: loss fields use different reductions

The largest audited BUDDY loss is 19.3490 for seed 0,
`boundary_private_half_pool`, Hits@50=66.2106%. Its source is
`link_port/callbacks.py:234–245`: mean positive BCE **plus** mean negative BCE.
This is a class-balanced diagnostic sum; it is not BUDDY's joint training BCE
mean. Native fits also have substantial values under that convention
(long-convergence range 2.5896–9.5891 across 28 completed native fits).

In contrast, `link_expansion_ports_BUDDY_production_v4/session.py:244–270`
explicitly reports joint mean BCE for expanded methods, including learned
aggregation, while retaining the source split-group diagnostic separately.
Interaction methods retain the split-group form in
`interaction_BUDDY_production_v5/session.py:323–348`.

**Do not rank BUDDY probability quality using the generic `loss` column across
these method families.** Hits@50 continues to use the same official evaluator
and the same positive/negative record groups. These loss conventions do not
invalidate the Hits@50 comparisons or change checkpoint selection. For any
future loss comparison, derive explicitly named joint BCE and balanced BCE
from each frozen VALID export, preserving the existing fields. A high diagnostic
loss still cautions against assuming calibrated probabilities.

## MolBACE GCN: a repeatable limitation of original shared running state

Seven matched completed long-convergence cohorts for GCN/dropout 0.0 are
seeds 0, 2, 3, 5, 6, 8, 9 on the fixed source scaffold split.

| Method | Mean selected VALID ROC-AUC |
| --- | ---: |
| Original boundary GNNM, shared native BN state | 64.9084% |
| GNNM with route-private BN state | 77.5772% |
| Four independently trained native models | 80.4867% |

Private BN improves on original GNNM in all seven paired cohorts, by 12.6688
points on average (range 8.0952–24.8352). It still trails I4 in all seven.
This supports treating normalization-state ownership as material, not claiming
that every wrapper variant is successful.

The distinction is intentional in the benchmark: original GNNM sequentially
updates one shared set of native running moments; private BN uses one running
state per route while keeping all learned BN affine and backbone parameters
shared. Under ordinary TRAIN-mode BN and matched inputs/RNG, these running
states do not enter the current-batch normalization or its gradients. Their
direct effect is on evaluation, selection and stopping. Selected-epoch gains
alone do not prove improved training gradients or learned representations.

This mechanism was recorded before the current large run in
`private_bn_state_v3/README.md`, `analysis/bn_train_semantics_source_v1/NOTE.md`
and `analysis/bn_route_diagnostic_v1/REPORT.md`. It is not a newly discovered
code defect. The current comparison is a stronger multi-seed observation on
the GCN setting; it remains selected VALID evidence, not independent TEST
confirmation or a novelty claim.
