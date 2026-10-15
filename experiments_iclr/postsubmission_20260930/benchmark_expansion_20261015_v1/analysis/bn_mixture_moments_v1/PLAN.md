# Frozen follow-up: common mixture moments at fixed private-BN weights

Question: does including between-route-mean variance materially change the
interpretation of the existing arithmetic tied-bank control?

Use exactly the completed MolBACE private-BN seeds101/202/303 and their already
selected epochs92/54/116, as pinned in INPUTS.json. Keep learned parameters,
native BN configuration/epsilon/affine parameters, examples, full151-graph VALID
order, checkpoint selection and fixed mean-raw-logit readout unchanged.

Evaluate two conditions in this order for every seed:

1. Own: restore all saved banks. This also checks the original selected export.
2. Mixture: for every existing BN layer/channel, set every route to the same
   mean=mean(mu_m) and variance=mean(v_m)+mean((mu_m-mean(mu_m))^2). Accumulate
   variance in float64, cast on copying; calculate the common mean in the same
   native dtype/order as the existing arithmetic tie. Leave counters as
   saved; they must agree across routes. Keep BN epsilon outside the variance.

This is a plug-in equal-route mixture of EMA moments, not pooled-history
recalibration or reconstruction of the original sequential shared EMA. No
training data are iterated. Compute only scalar per-layer within/between moment
summaries from already loaded banks; do not use them to select a condition.

Reuse all five conditions from the exact completed original bank-assignment
JSON report for each seed. Verify that its checkpoint hash matches the current
selected checkpoint. Report new own and mixture metrics alongside prior own,
arithmetic tied, and all cyclic shifts. Report paired own-minus-mixture,
mixture-minus-arithmetic, and own-minus-average-cyclic-score differences for
AUC, BCE and Brier. For loss differences retain their literal sign: negative
means lower loss. Do not select a best shift or change serving.

Run only via the retained source-compatible CPU adapter.validate path, one
seed at a time, two threads, at most300 seconds per seed, no retries. Two
conditions x five VALID batches x four routes =40 native forwards per seed,
120 total. Three model constructions; zero TRAIN calls, backward/optimizer
updates, TEST access or new scientific fits. Fresh output only; restore banks
and verify learned parameters unchanged. A failure is preserved and stops the
supervisor; no automatic alternate setting is tried.

Execution entrypoint, when separately dispatched by the parent: supervise.py
EXACT_PROJECT_ROOT FRESH_OUTPUT, with CUDA_VISIBLE_DEVICES empty. Existing
allocation-only route checks remain inherited from the original evaluator.
This packet itself is preparation only; no command has been dispatched.
