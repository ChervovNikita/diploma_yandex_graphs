# Molecular alignment-only: active-support gap resolved

A separate frozen qualifier exercised the unchanged callback on the first actual source-shuffled eligible32-graph TRAIN batch containing both classes. The first batch already supplied18class0and14class1graphs; its IDs/classes and source positions are preserved. It was reused for this finite engineering check, not treated as a scientific training trajectory.

Both additional optimizer transitions completed, with128eligible alignment anchors. Weighted alignment was0.039332and0.039812; actual auxiliary gradient norms were0.105866/0.113873for shared source parameters and0.005199/0.005484for private parameters. The final output bank had zero direct auxiliary gradient, as expected from prehead placement. Two extra retained-graph autograd collections measured support without assigning parameter gradients, changing the returned loss or stepping the optimizer; these collections are charged separately.

Fresh-session restore and all three complete151-graph VALID passes succeeded, with restored output difference zero and native optimizer ownership/options/Parameter identity retained. An objective-mode mismatch was rejected. The finite check took4.82seconds with two CPU threads and CUDA hidden.

The earlier two updates on a one-class batch remain recorded as lifecycle/zero-auxiliary evidence. Total cost across both checks is four native constructions, four optimizer transitions and six whole VALID passes, plus two auxiliary gradient collections. No frozen callback/helper/parent source was modified.

The active gradient path is now qualified only for this MolBACE GIN CPU case. This supplies no convergence, accuracy, CUDA/other-cell/full-worker/final-serving or methodological novelty evidence.
