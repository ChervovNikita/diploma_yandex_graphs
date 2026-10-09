# Compact private rotations: ancestry and placement decision

9 October 2026. One additional bounded primary method scope; no experiment or performance claim.

The BOFT paper already constructs a dense orthogonal weight adapter from sparse orthogonal butterfly factors. Each factor can use small Cayley-parametrized blocks, starts at identity, and permits multiplicative dropout by substituting identity blocks. Its base matrix is pretrained and frozen. The paper gives O(d log d) parameter structure for a full butterfly, with matrix-factor computation charged. Compact rotations, identity-preserving initialization and their spectral preservation are established ingredients.

A jointly trained graph ensemble with shared W and private R_m would change the training/application setting. It would not establish a new rotation primitive. The saved OMoE, HousE, ETHER+ and HTA conclusions already supply related expert-diversity, graph endpoint and neutral-adapter ancestry. No complete-rule novelty clearance follows from this one further paper scope.

There is a concrete placement trap. Applying the SAME orthogonal R_m to both query and key after their feature map leaves the attention numerator exactly unchanged: (Q R_m)(K R_m)^T = Q K^T. The denominator of normalized linear attention is unchanged as well. A penalty that spreads those rotated embeddings could change their displayed coordinates while leaving predictions identical. Do not spend a scientific fit on that exact cancellation.

Rotations before the nonlinearity or inside successive shared dense maps need not cancel. They could remove diagonal BatchEnsemble restrictions and improve individual members, but this is an attributed structured-adapter hypothesis. Their low parameter count does not remove each member's graph/feature computation. Shared singular values impose another restriction; learned shared W does not inherit frozen-base semantic-preservation or generalization guarantees from BOFT.

Decision: retain compact rotation adapters as a possible capacity comparator or later distinct hypothesis. Do not launch a generic rotation grid or describe rotations as methodological novelty. The active Q/K panel has a specified non-cancelling pre-sigmoid placement and a genuine full-Q/K comparator; the separate sheaf experiment concerns endpoint-conditioned relational transports. These should produce evidence before another architectural branch is added.

Read scope and exact downloaded bytes are in READ_SCOPE.json and RETRIEVAL.json. This was not a full-paper, proof or author-implementation audit. No reported BOFT numerical result is adopted.
