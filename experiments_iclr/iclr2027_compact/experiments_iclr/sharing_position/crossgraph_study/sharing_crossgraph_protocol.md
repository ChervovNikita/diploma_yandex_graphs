# Frozen sharing-boundary cross-graph check

Prepared 2026-09-25 UTC after the Roman Empire sharing-boundary result was known. This is a post hoc cross-graph check of that result. The choice of datasets and fixed recipe is made before opening any WikiCS or Actor scores. Both datasets and all 24 runs must be reported together, including null or opposite effects.

## Question and fixed matrix

Repeat the exact two-block SAGE sharing placement from the Roman Empire diagnostic on WikiCS official split 0 and Actor official split 0. Use optimizer seeds 0, 1, 2, and four arms per seed: tied, private-first, private-last, and fully untied propagation. The two partial arms retain identical member-specific input/output BatchEnsemble projectors and differ only in which residual SAGE block has four private copies. Both partial arms have equal trainable parameter counts within each dataset and four graph propagation paths. Each is copied from one newly initialized tied model. Initial parameters, all four initial member logits, CPU/CUDA random states, and parameter storage are checked before training.

For both graphs, use width 128, depth 2, dropout 0.2, four members, LayerNorm, AdamW at learning rate 0.001 and zero weight decay, and exactly 300 epochs. Update on the mean of four member cross-entropies. At every epoch, choose by greatest pooled-logit validation accuracy, then least pooled cross-entropy, then earliest epoch. Restore that checkpoint and replay validation before reading test labels. No hyperparameter or arm is chosen from test performance.

## Public data and exact preprocessing

WikiCS is loaded from the existing public `data/wiki_cs/data.json`: features, labels, published train/validation masks at index 0, shared published test mask, and undirected graph edges. Actor is loaded from the existing public node-feature/label and graph-edge text files and `film_split_0.6_0.2_0.npz`; its sparse binary node features and split masks match the earlier external study. Actor edges are coalesced and made undirected. No explicit self loops are added to either graph. Dataset file SHA-256 values, processed feature/label/edge/split hashes, source files, and library versions are recorded in each dataset's source manifest. The raw datasets are not redistributed in the public supplement.

The primary contrast is private-first minus private-last in test accuracy, paired by optimizer seed within each fixed dataset/split. Report every seed, mean, sample standard deviation, all four arms, parameter counts, selected epochs, mean member accuracy, and pooling gain. A result from three optimizer seeds on one split does not estimate across-split or across-graph variability. The two graphs were selected after seeing the Roman result, so the analysis is exploratory and cannot support a general rule by itself.

## Completion gate

The runner checks source/data hashes, initial member-function equality, equal partial-arm parameter counts, and immutable run directories. The independent verifier requires all twelve arms per dataset, all 300-epoch validation traces, earliest-tie checkpoint selection, artifact hashes, initial state/RNG/logit replay, selected checkpoint reload, and validation/test score and decision replay. `completion_audit.json` exists only after full verification. A failed or incomplete graph is disclosed rather than replaced or omitted.
