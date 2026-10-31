# Prospective CE measurement amendment

This source packet changes only CE finite-difference reduction precision. It was designed after the disclosed Photo17 R17v2 failure and exact-state diagnostic. It does not reverse the failed registry, continue an old claim or authorize training.

The native model, FP32 logits, training CE/gradient, private-factor initializer, Adam transport, arms, schedules, seed/direction generation, epsilons and tolerances remain unchanged. FP32 per-example CE is promoted before its mean and subtraction. All original FP32 CE FD values and their errors remain in every qualification receipt. Every cold and actual-warm state must be requalified under a new complete study/version.

The exact-state diagnostic reproduced the full old failed gate and all six original records. Both precision variants passed unchanged thresholds; reduction-only variant errors for the failing direction were 0.0692% and 0.6029% against the original analytic derivative. This justifies a measurement correction at that state; it does not establish future-state correctness or predictive benefit.

A new independent source audit and a new finite root study admission with the complete prior failure/diagnostic lineage are required. No useful checkpoint or phase result is inherited. The existing Round15 initializer source remains byte-identical; the new qualifier is an explicit driver dependency, not a hidden runtime mutation.
