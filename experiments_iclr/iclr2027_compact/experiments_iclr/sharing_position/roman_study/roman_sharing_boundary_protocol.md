# Roman Empire sharing-boundary diagnostic, frozen before scores

Prepared 2026-09-25 UTC. This is a post hoc author-side diagnostic, chosen after earlier Roman Empire and ogbn-arxiv results were known. It does not establish a new general method. All twenty runs are required before the test comparison is opened or used in a paper.

## Question and fixed arms

One published Roman Empire mask (index 0), five optimizer seeds (0, 1, 2, 3, 4), and four model arms form twenty runs. All use the original `models.py` two-block SAGE GNNM architecture: width 128, four BatchEnsemble members, LayerNorm, dropout 0.2, member-specific input and output projectors, and a mean-of-member-cross-entropies training objective. The arms are:

- `tied`: both residual SAGE blocks shared by four members.
- `private_first`: first residual SAGE block copied four times, second shared.
- `private_last`: first residual SAGE block shared, second copied four times.
- `untied_propagation`: both residual SAGE blocks copied four times.

The two partial arms have exactly equal trainable parameter counts and the same number of graph propagation calls per member. Each begins as exact copies of one newly initialized tied reference, including the common BatchEnsemble input/output projectors. Its four member logits are required to match tied before optimization. Each seed/arm starts from the same CPU and CUDA random-number state after model construction. The two SAGE blocks have identical width and module specification. The intervention is where parameter sharing stops, not added parameter quantity. Different training gradients remain an inseparable part of the sharing change.

## Data and optimization

Use the pinned public Roman Empire NPZ (`data/roman_empire.npz`), its official train/validation/test mask 0, undirected/coalesced edges with no explicit self loops, and the original SAGE root transform. This matches the earlier Roman bridge OGB-300 schedule. Train every arm for exactly 300 epochs with AdamW, learning rate 0.001, zero weight decay. At each epoch, update on the mean of four member cross-entropies from training labels. Select the checkpoint by greatest pooled-logit validation accuracy, then least pooled validation cross-entropy, then earliest epoch. Restore that checkpoint and replay validation before scoring test labels. Do not select any hyperparameter, arm, or seed from test results.

The first comparison is `private_first - private_last` in percentage points of test accuracy for each seed. Report all five paired differences, mean and sample standard deviation, even when signs differ. Also report every arm's score, selected epoch, parameter count, mean member accuracy, and pooled-minus-mean-member gain. The unit of replication is optimizer seed on one fixed graph and official mask. These five seeds cannot estimate variation across datasets, graph constructions, or official masks.

## Completion and provenance gates

The runner writes a source/data manifest before production and refuses changed source, data, or protocol bytes. Its smoke test checks all four member functions, initial random states, parameter storage, and exact equality of private-block copies to the tied initialization. The independent verifier requires all twenty immutable result directories, source and data hashes, 300-epoch traces, validation selection, artifact hashes, the initial member logits, equal partial-arm parameter counts, selected checkpoint reload, and validation/test logits and pooled decisions recomputed from the public graph. `completion_audit.json` is written only if every condition passes. Scores may enter the manuscript only after this gate. The raw dataset is not redistributed in a public supplement.

A failed or incomplete arm is reported as incomplete. It is not silently replaced or dropped. If the full gate misses the internal submission freeze, this diagnostic remains author-only and out of the submission.
