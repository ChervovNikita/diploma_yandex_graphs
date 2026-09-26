# Initial cross-member graph gradients

Run `python experiments_iclr/initial_member_gram/verify_gram.py` from the
anonymous archive root with Python and NumPy. It checks all twelve retained
Gram matrices, their source and audit hashes, positive semidefiniteness,
pair cosines, and the first-order graph-only SGD loss identities.

This calculation uses all twelve previously exported training-gradient
cases. No new training or held-out label access was needed. Seven of 72
unordered member pairs have negative inner products, all on WikiCS. The
total graph direction is nevertheless descending for every member at
these initial points. This is neither an AdamW trajectory result nor a
predictor of test performance.

`analyze_initial_member_gram.py` is the immutable original analysis source.
`INITIAL_MEMBER_GRAM_ROOT_AUDIT.json` records an independent computation
from the full float64-converted original arrays using scalar dot products
and direct vector norms. The original float32 gradient arrays are retained
in author evidence and omitted here. The small verifier checks the Gram
matrices and arithmetic, and cannot recreate their underlying gradients.
To recreate the latter, follow the first-update diagnostic's source/runtime
instructions, export all twelve cases, and run the supplied analysis beside
the resulting `initial_update_raw_gradients/` directory.
