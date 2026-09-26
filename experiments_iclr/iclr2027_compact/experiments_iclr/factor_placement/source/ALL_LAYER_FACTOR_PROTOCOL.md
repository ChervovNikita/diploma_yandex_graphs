# Post hoc all-layer factor placement ablation

This 72-cell extension was chosen on 26 September 2026 after the primary
432-cell validation and test outcomes were read, to address a closest
factor-placement comparator gap. It is exploratory. It is neither an intact
holdout nor an exact reproduction of Kim's published GCN/GIN/GAT work: this
study uses residual SAGE, retains the existing boundary factors, and pools
raw logits. Its complete matrix, code, data fingerprints, and analysis rule
must be frozen before any all-layer training. Every cell, including a failed
one, is retained and reported. The operational target for a complete
independently audited extension is 2026-09-26 06:30 UTC.

## Model and comparison

Start from the exact four-member TIED model in the frozen primary study.
Keep its input and output BatchEnsemble factors. Wrap each residual SAGE
block's neighbor and root linear maps and each feed-forward block's two
linear maps with additional member-specific rank-one factors. Each new map
is `shared(x * R[m]) * S[m] + B[m]`, with every R/S initialized to one and
every B to zero. Under equal seeds, the candidate must have identical
initial member logits, shared-weight tensors, and post-construction RNG
state as the primary TIED model. The new factors receive gradients on the
first step; no graph-layer weight becomes private. This adds factor
parameters and computation, not a new ensemble member or propagation depth.

Four published split-0 graphs and all preprocessing are exactly those of
the primary freeze: Cora, WikiCS, Actor, and filtered Chameleon. Each graph
gets six learning-rate/AdamW decay candidates (0.0003, 0.001, 0.003 crossed
with 0, 0.01) and optimizer seeds 0, 1, 2: 72 cells. Each uses four members,
width 128, two residual SAGE blocks, dropout 0.2, mean member training CE,
and 1,000 epochs if finite. The training bundle contains training and
validation indices/labels only. At each epoch, choose a whole-model
checkpoint by highest pooled raw-logit validation accuracy, then lowest
pooled validation CE, then earliest epoch. A nonfinite seed invalidates its
three-seed candidate; all planned cells still run.

## Selection, scoring, and comparison firewall

All 72 all-layer cells use the same GPU77 software runtime as the original
TIED Actor and filtered Chameleon cells. Original Cora and WikiCS TIED cells
used a different host. A separately frozen 36-cell TIED replication repeats
the full original six-candidate grid for those graphs on GPU77. Both new
validation locks must exist before either new study reads test labels.

An independent verifier checks all 72 identities, frozen sources/data,
1,000-row validation traces or documented failures, checkpoint hashes and
state, selected epochs and validation metrics. Each selected validation
checkpoint stores pooled/member logits, node IDs, and labels in a separate
hashed artifact. Fresh replay permits at most `1e-4` absolute logit drift,
requires identical pooled/member class decisions, and checks accuracy and
cross-entropy. It writes an immutable
validation-only lock. For each graph, select one candidate using highest
three-seed mean validation accuracy, then lowest mean CE, then lower LR and
decay. Test labels may be read only after this 72-cell lock, the 36-cell
same-runtime TIED lock, and the primary 432-cell global lock exist and match
their freezes. Score only the selected
candidate and prespecified default (0.001, 0) for all three seeds; if they
coincide, score the unique configuration once. Independently replay every
allowed checkpoint, pooled/member logits, exact class decisions, and score.

Placement comparison uses the same-runtime TIED replication for Cora and
WikiCS and original GPU77 TIED for Actor and filtered Chameleon. The primary
source and scores never change. Comparisons with other arms retain their
runtime boundary where applicable.
The all-layer arm has the same six-candidate search size as each primary arm,
but was designed after primary outcomes; any reported difference is a local
post hoc factor-placement observation on one split per graph, not a
generalization estimate or evidence of a new optimizer. Store parameter
count and selected epochs; parameter storage is distinct from timing.
