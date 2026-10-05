# Bounded V2 repair review request

Root and independent auditor: authenticate the preserved V1 manifest/seal and the fresh V2 packet, then review only SOURCE_V1_TO_V2.patch and REPAIR.json.

P2 fix: exact constant probability/residual columns use a zero centered vector before the existing sums of squares/covariance and Pearson calculation. Require agreement between zero variance/covariance rows and undefined_zero_variance status. Empty support remains undefined_empty. No arbitrary small-variance threshold, new metric or scientific computation was added.

run_error_analysis.py and SOURCE_BINDINGS.json are byte-identical to V1. All remaining scientific source is unchanged. Preserve V1 and use the fresh V2 source seal for any root execution. Root alone executes; preparation performed AST/text/hash checks only, with no payload or numerical runtime access.
