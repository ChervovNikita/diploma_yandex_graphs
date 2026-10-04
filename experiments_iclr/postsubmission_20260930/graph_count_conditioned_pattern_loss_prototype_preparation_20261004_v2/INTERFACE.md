# TRAIN-only interface, no execution

Existing producer identities are pinned in SOURCE_BINDING.json. A future
root-admitted adapter may call the mathematical core after the existing full
native neural schedule has produced the original ordered residual scores:

```python
neighbors = details["neighbors"]
left_rows, left_nodes = neighbors.left
right_rows, right_nodes = neighbors.right
rows, bits = teacher.labels(queries, neighbors)  # authenticated full TRAIN only
assert torch.equal(rows, torch.cat((left_rows, right_rows)))
assert teacher.receipt()["teacher"] == "complete_TRAIN_observation_membership"
t = details["t"]
values = training_pattern_losses(
    t[:, :len(left_rows)], t[:, len(left_rows):],
    left_rows, right_rows,
    TrainPatternLabels(bits, "complete_TRAIN_observation_membership"), len(queries),
)
# A separately frozen future condition selects its prescribed value.
# Retain .mean() over all queries, coefficient1 and both existing query populations.
```

This is an interface illustration, not a patched fit or condition selector.
There is no TRAIN/VALID/TEST loader or graph masking in this prototype. Actual
teacher provenance must be checked upstream against the complete TRAIN source;
a role string alone cannot establish it. Both side counts are derived only
inside the auxiliary branch from the one shared observed pattern. A member
axis, VALID/TEST roles and soft/predicted-member teacher patterns are invalid.
No teacher count or bit vector is a native completion/link-serving input.

The conditional single accepts one native width64 h and its original single
unary left/right scores in visible_context, whose signature contains no teacher.
It validates original unique ascending CSR candidate order and centers unary
features per query/side. Only training_nll accepts the shared TRAIN labels,
derives remaining count budgets and constructs earlier-selected identity
summaries. Its outputs are auxiliary query losses; a separate native target
route handles ordinary link inference without those budgets or teacher bits.
No complete native S_K backbone wrapper/fit adapter is supplied or qualified.

## V2 exact grouped batch source

The same training_pattern_losses signature/results use exact(n,min(k,n-k)) grouping internally. No per-query Python loop or count.item synchronization remains in that CB batch path. All slots, original query order and all-query reduction remain. The single's valid visible_context function and sequential training_nll are preserved except for explicit endpoint bounds. New fabricated float32/ragged QA source uses the actual visible_context path, without a dataset loader or native fit adapter. It remains unexecuted.
