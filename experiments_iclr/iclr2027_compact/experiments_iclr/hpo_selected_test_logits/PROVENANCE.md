# Selected HPO test-logit companion

This public companion covers **all 72 validation-selected cells**: four graph
settings, six arms, and three optimizer seeds. The candidate for each graph
and arm is determined by the 432-cell validation-only selection lock in the
sibling `validation_tuning/` folder. No case was chosen using test outcomes.

Each `logits/<graph>/<arm>/<candidate>/seed*.npz` contains the original
float32 `test_member_logits` array, with no quantization or rounding. The
original float32 `test_pooled_logits` is reproduced **bitwise** by
`test_member_logits.mean(axis=0)` in NumPy. The author-side derivative audit
checked this equality against every original NPZ before writing the public
files. The `MANIFEST.json` records SHA-256 hashes of both original arrays,
each original predictions NPZ, each original score file, and each derivative
NPZ. The original 72 NPZ files are retained in the author evidence archive
`SELECTED_HPO72_ORIGINAL_NPZ.tar.gz` (SHA-256
`f872087c79abdc20ce36d088a974c96335e02beeac90b395141c17aa237c3410`).

Run `python verify_public.py` from this folder or another directory. It needs
NumPy and the sibling `validation_tuning/` folder. The verifier checks the
selection lock, original score audit, test labels and indices, all test
member and pooled decisions, original-array hashes, and recorded test
accuracy/cross entropy for **all 72 cells**. It also matches each result to
the independent hard-decision export. The compact public evidence omits the
original validation float logits and training checkpoints; reproducing
training requires the frozen study source and public graph datasets.

The hard-decision export already contains validation-selection evidence and
test class decisions for all scored cells. This companion adds exact test
float logits so pooled predictions and member behavior can be examined
without a GPU. It is a post-score transport derivative, not a new experiment.
