# Prospective pooling and error analysis

This development diagnostic is fixed before any of the 35 HGT study outcomes.
It requires the exact adopted seven-arm/five-split freeze and all 35 selected,
replayed cases. It verifies native latest-tie checkpoint selection and saved
prediction NLL before calculating diagnostics. It opens only explicitly admitted
development labels, never the dataset archive or heldout labels.

The analysis separates average member NLL from the ambiguity term for the served
mean-logit predictor. Wood et al. (JMLR 2023) is the known prior for this
loss/pooling-matched decomposition. The accounting identity does not establish
a causal mechanism. FP64 diagnostic arithmetic does not replace the native FP32
scores. It reports all controls, node rescues and harms, jointly wrong members,
and pair disagreement. Member correctness coverage is an unserved diagnostic,
not an accuracy upper bound: pooling can classify correctly even when each
member has a different wrong argmax. Selected-validation observations are not independent
heldout confirmation.

Increasing member separation can leave the pooled prediction exactly unchanged.
An increased ambiguity term alone therefore cannot support a useful-model claim;
member quality and the served predictor must be assessed together. The analysis
adds no tuning, arm, model selection, predictor, acceptance gate, or new theorem
claim. The synthetic CPU fixture passed; its mathematical functions are unchanged
by the subsequent source-score provenance repair. Reported source scores retain
the original selection values, with FP32 replay values stored separately. No real
study outcomes have been analyzed. Execution remains conditional on complete
study closure and a separate root release.
