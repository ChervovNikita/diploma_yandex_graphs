# Normalization depends on how ensemble routes are batched

15 October 2026. This scoped inspection answers a new question raised by the
completed molecular normalization result: can sequential route execution and a
vectorized BatchEnsemble implementation be assumed to use the same BN rule?
**They cannot when their normalization batches differ.** This is a source and
algebra finding, not a new accuracy result or a novel normalization method.

## What the sources actually show

The already retained BatchEnsemble paper describes partitioning a training
minibatch among members and repeating inputs for ensemble evaluation. Its
classification appendix discusses per-member batch size. Those paragraphs do
not by themselves specify an exact private-running-buffer implementation.

The current Uncertainty Baselines author implementation is now pinned to commit
`9920c89ead28af02f0ba6a70f3ca7d55b6d329d3`. In
`baselines/cifar/batchensemble.py`, lines 230–243, training tiles images and
labels across members and makes one `model(images, training=True)` call. In
`uncertainty_baselines/models/wide_resnet_batchensemble.py`, ordinary Keras
BatchNormalization layers act on the model's leading batch dimension. No
route-dependent BN state or route-sub-batch normalization appears at those
call sites. Its evaluation also tiles inputs in one call and averages member
probabilities, whereas our current serving rule averages raw logits.

Consequently, this implementation's ordinary BN sees activations from multiple
routes in its local replica batch. This conclusion concerns the inspected
current CIFAR/WRN implementation. It does not establish exact equivalence to
the original 2020 experimental code, every BatchEnsemble implementation,
distributed cross-replica behavior, or our graph setting. No code was executed.

Our molecular source instead calls `run_member` four times sequentially for a
native graph minibatch. During TRAIN, each ordinary BN call normalizes its
own route batch. Original GNNM then uses the same running buffers for all four
calls and all inference routes. The private-BN extension changes only buffer
ownership; learned native/affine parameters and the TRAIN normalization batch
remain shared and unchanged respectively. Exact local source pins are retained.

| Policy | TRAIN statistics for a route | Stored state used at evaluation |
| --- | --- | --- |
| Current sequential GNNM | That route's native minibatch | One EMA updated by every route |
| Current private-state control | That route's native minibatch | Its own EMA |
| Inspected vectorized UB implementation | Combined route batch within the local replica | EMA from those combined batches |

## Why averaging running variances is not pooled normalization

For a scalar activation, suppose route `r` has a stationary population mean
`mu_r` and variance `v_r`. With constant native update coefficient `a`, one
complete sequential pass through `M` routes updates the shared mean as

`mu_next = (1-a)^M mu_previous + a sum_r (1-a)^(M-1-r) mu_r`.

At the end-of-cycle stationary limit, this is a weighted mean with
`w_r = a(1-a)^(M-1-r) / (1-(1-a)^M)`. For `a=0.1`, `M=4`, the effective
per-cycle coefficient is 0.3439; the four weights are approximately
0.212, 0.236, 0.262 and 0.291. Later routes receive greater weight.

Under the same idealized assumptions, the running variance tends toward the
weighted within-route variances `sum_r w_r v_r`. A population mixture with
those weights instead has variance

`v_mixture = sum_r w_r [v_r + (mu_r - sum_s w_s mu_s)^2]`.

The between-route mean term is absent from sequential averaging of within-route
variance estimates. Furthermore, the inspected tiled implementation uses the
batch's actual membership weights, not necessarily the sequential EMA weights.
Thus tying saved means and variances is neither a reconstruction of the
original sequential history nor equivalent to computing pooled-batch moments.

This derivation assumes stationary route distributions, constant momentum,
fixed call order and compatible variance conventions. Finite native batches,
unbiased running-variance updates, variable graph sizes, nonlinear downstream
layers and evolving weights add differences. The equations do not predict
which policy has better AUC or isolate the full observed private-BN gain.

## Consequence for our research and benchmark

This makes the molecular result more interpretable: original sequential GNNM
normalizes a route using its own TRAIN batch but later uses shared history at
evaluation. Private state makes the conditioning of these two operations more
consistent. The completed AUC improvement supports examining that mechanism;
the worse BCE and separately selected epochs still limit the claim.

The same issue affects combinations. Input corrections and geometry losses
may change route activation distributions, so an auxiliary objective can be
judged under a materially different evaluation normalization rule. The running
nine-fit factorial is already the appropriate next predictive comparison. This
note adds no arm or run and does not inspect its partial scores.

For the giant benchmark, preserve original GNNM explicitly and settle a common
normalization policy for extensions and their matched controls after the full
factorial. Record the normalization batch and state ownership separately in
method descriptions. Do not treat replacing four native forward calls with
one concatenated call as a semantics-preserving speed optimization when native
BN is present. That change can alter TRAIN outputs and gradients, not merely
runtime. A vectorized implementation would need route-wise native-equivalent
normalization and suitable RNG handling before such an equivalence claim.

Private statistics with shared affine parameters already have slimmable-network
precedent. The contribution still needs to come from a supported predictive
method and its explanation. This note identifies a concrete experimental
condition to control; it supplies no novelty or superiority theorem.

## Reading record

- No new full-paper read is counted. Existing paper paragraphs were revisited
  for this specific batching question.
- Newly inspected: the pinned UB model's normalization declarations and the
  training/evaluation batching, forward and averaging call sites; our exact
  sequential molecular and private-state call paths.
- `RETRIEVAL.json` records public URLs, hashes and commit. `LOCAL_SOURCES.json`
  pins local evidence. No model, dataset, checkpoint or held-out score was read.

Sources: https://arxiv.org/abs/2002.06715 and
https://github.com/google/uncertainty-baselines/tree/9920c89ead28af02f0ba6a70f3ca7d55b6d329d3.
