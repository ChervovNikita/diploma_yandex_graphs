# External SAGE public projection

This transport derivative covers **all 18 runs** in the fixed external SAGE
study: WikiCS and Actor, three architecture arms, and three paired optimizer
seeds. The original `selected_predictions.npz` files, with valid and test
member float logits, remain in the author evidence. Their hashes are retained
in each original `result.json` and in `PROJECTION_MANIFEST.json`.

Each `selected_projection.npz` contains, for both validation and test nodes:

- Exact float32 logits from the original four-member mean.
- Exact class predictions for every member, stored as int16.
- The original int64 node indices and labels.

These are sufficient to reproduce validation and test pooled accuracy and
cross entropy, member mean accuracy, and pooling gain. Individual member
probabilities and member cross entropy require the author-retained original
arrays or a rerun. The 18 original NPZ files total 13,068,802 bytes; the 18
projection NPZ files total 3,631,396 bytes. No run was omitted or selected by
its test outcome.

`verify_pooled_classes.py` checks all 18 projected files, frozen source
hashes, paired initial states, complete 300-epoch validation traces,
validation-selected epochs, recorded test metrics, and the previous full
checkpoint CUDA replay audit. The independent author-side old-to-new audit
also compared every projected array with its original array or derived mean
and class predictions; see `ORIGINAL_TO_PROJECTION_AUDIT.json`.

The historical `verify_compact.py` expects the original full member-logit
files. It is preserved for reruns with those files. Use
`verify_pooled_classes.py` for this public projection.
