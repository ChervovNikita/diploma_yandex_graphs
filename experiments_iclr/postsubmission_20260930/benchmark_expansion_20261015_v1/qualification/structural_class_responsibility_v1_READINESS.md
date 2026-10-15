# Structural class responsibilities: readiness

Source manifest: `2fe768218083169f3973f78f5a9face1c2f79d75fb2a25cd2256a02fe4d25e98`.

The fixed atom-count quartile method and its exact whole-vector within-class
permutation control are implemented for native molecular GNNM. Fractional tie
handling preserves exact rational class masses and per-object total weights.
There are no new learned parameters, input changes or tunable candidate options.
Shared gradients use native F; private gradients use weighted W, with the
original native reduction, Adam groups and one transition.

Six pure table checks and two discriminating tensor-gradient checks passed.
Actual MolBACE/GIN CPU qualification also passed both variants: two optimizer
transitions per variant using the same 32 real TRAIN graphs, three complete
151-graph VALID passes, serialized checkpoint and fresh-session resumed update.
Initial and restored output differences were 0; native Parameter identities
were retained. All updates reported two reverse collections and one Adam step.
Execution used two CPU threads, Torch 2.1.2+cu118, four native constructions and
four optimizer attempts/completions in 10.16 seconds. No GPU job or TEST access
was added. These checks establish implementation behavior, not predictive gain.

Both tables cover all 1,210 observed TRAIN molecules. Each route retains class
weight totals 730 and 480. The structured and permuted tables preserve the same
classwise multiset of complete member weight vectors. The fixed control moved
all 1,210 object positions; this does not imply every weight vector changed,
because ties can give multiple objects the same vector. No redraw occurred.

The node incoming-degree rule and exact counting helper are defined and tested.
Its native training bridge remains pending and is rejected explicitly. Link
endpoint-correlated random fields are a different hypothesis. Other molecular
cells, CUDA/full-epoch/resource checks and final-serving reconstruction for
this new state wrapper still need qualification.

The method definition can enter the proposal, with original GNNM G0 as the
uniform-risk baseline. No all-cell execution admission, full scientific fit,
novelty, score improvement or giant campaign GO follows from this receipt.
Exact source/evidence hashes and work counts are in the adjacent JSON receipt.
