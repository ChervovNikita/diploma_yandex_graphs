# Roman Empire post hoc bridge, fixed protocol v1

This study was chosen after the selected Roman and ogbn-arxiv results and
after the WikiCS/Actor external graphs had been chosen. It is a post hoc
configuration bridge, not prospective evidence of a graph moderator.
Training and checkpoint selection are frozen here before inspecting any
bridge test score.

Use the original Roman Empire NPZ bytes from the repository (`data/roman_empire.npz`,
SHA-256 `a58ba741d123bf892fe5c872138d07463d75a2e9012360b8dd78ac2d4766d428`).
Use official train/validation/test mask 0 and optimization seeds 0, 1, 2 on
that same mask. The observations are optimizer repetitions on one graph.
Convert raw edges to undirected with PyG `to_undirected`; do not add explicit
self loops. Node features and edges are transductive. Only training labels
enter the objective; validation labels select a checkpoint; test labels are
scored once after restoration.

The architecture, objective, optimizer, checkpoint rule, and training budget
are exactly the external WikiCS/Actor v3 and ogbn-arxiv 300-epoch SAGE
settings: four members, two residual layers, width 128, LayerNorm, dropout
0.2, AdamW at 0.001 with zero weight decay, 300 full-batch epochs, mean
member cross-entropy, and logit pooling. Evaluate pooled validation accuracy
every epoch, then cross-entropy and earliest epoch on ties. No early stop.
No parameter tuning uses validation or test scores in this bridge.

Three arms are fixed before outcomes: tied GNNM; initially identical untied
residual propagation stacks with shared boundary factors; and tied GNNM plus
identity-initialized BatchEnsemble factors around every SAGE neighbor/root
and FFN linear map. All share canonical initial parameters and the same
post-construction Python, NumPy, CPU Torch, and CUDA Torch RNG state.
Initial CPU tied/untied logits must agree exactly in an independent smoke;
initial CUDA member logits must agree within 1e-5 maximum absolute error,
above measured 2–3e-6 A100 scatter nondeterminism. The same bound applies
to the all-layer identity arm. Parameter counts are recorded.

The runner writes a source/data/protocol manifest before training, including
hashes for its source, the original model code, this document, the NPZ, the
parsed tensors, and the mask. It stages each arm atomically and records
validation traces, selected checkpoint, member logits, exact test IDs and
labels, initialization audit, and artifact hashes. The separate verifier
requires all nine arms and independently replays selected checkpoints before
any result is used. This new folder and manifest never mix with the
prospective WikiCS/Actor v3 results or the archived Roman controls.
