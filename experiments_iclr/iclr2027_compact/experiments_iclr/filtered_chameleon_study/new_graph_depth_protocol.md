# Official filtered-Chameleon depth extension

This separate stress test uses the public `chameleon_filtered.npz` from
`yandex-research/heterophilous-graphs`, commit
`a431395582e929d88271309716bea4fe24ce6318`. The raw archive SHA-256 is
`bf46f07e1fb5249280447e5fe3100f3e82fc4b93ad1e13ffcfdec924b6ac0bb5`.
The reference is Platonov et al., *A Critical Look at the Evaluation of GNNs
under Heterophily: Are We Really Making Progress?*, ICLR 2023. This graph was
selected because the legacy Geom-GCN Chameleon graph has known duplicate-node
concerns; the filtered study was fixed before its scores were read.

The loader uses node features and labels as published and mask 0 from the ten
published masks: 409 training, 287 validation, 194 test nodes. Original
directed edges are coalesced and symmetrized for consistency with the earlier
SAGE recipe. No self-loops are added; the raw filtered graph has none. This
symmetrization departs from experiments that retain the official directed
edge convention, so the numbers are not a direct leaderboard comparison.
All raw files, graph tensors, labels, and exact split IDs are fingerprinted
before training.

The complete design is depth 2 and 5, tied versus fully untied propagation,
and optimizer seeds 0, 1, 2: 12 mandatory cells. Every run uses the original
four-member BatchEnsemble input/output projectors and residual SAGE blocks,
width 128, LayerNorm, dropout 0.2, AdamW at learning rate 0.001 with zero
weight decay, and 300 full-batch epochs. Training minimizes mean member
cross-entropy. The checkpoint is selected by pooled validation accuracy at
every epoch, lower pooled validation cross-entropy on a tie, then earliest
epoch. Test labels are scored after checkpoint restoration only.

The paired arms share canonical initial weights, post-construction CPU/CUDA
RNG, and initial member logits to max absolute tolerance 1e-5. The complete
CUDA replay verifier checks all saved artifact hashes, 300 validation rows,
selected validation/test logits with exact member/pool decisions, and the
public data/split fingerprints. This is one graph and one fixed split; three
optimizer seeds do not sample independent graph realizations.
