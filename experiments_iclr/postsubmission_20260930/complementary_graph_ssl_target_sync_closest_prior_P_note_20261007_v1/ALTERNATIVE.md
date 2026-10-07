# Persistent graph feature components as branch supervision

This prospective alternative assigns a different, persistent graph-conditioned reconstruction target to each private path. Every path still learns full node labels. It tests whether retaining feature signals at different graph frequencies helps repair shared errors; it does not treat spectral separation as evidence of useful classification diversity. This is separate history from the sealed quarter-order SSL proposal and is not launched.

## The graph mechanism

If a competent frozen GNN loses attribute cues that disagree with nearby nodes, its members can agree on wrong predictions near graph boundaries. Few TRAIN labels may provide little incentive to recover those cues even when they remain in raw features. Persistent low-to-high graph-frequency reconstruction could train private paths to retain different cues before full-label classification.

This is conditional on an actual information-loss/common-error mechanism. A native GAT need not be a simple low-pass filter. High-frequency attributes may instead be noise that cannot be predicted from masked context. Band disagreement is not calibrated uncertainty, and the sum of feature components says nothing by itself about probability averaging.

## Fixed complementary targets

Define the target graph prospectively from public topology: symmetrize A, add self-loops, and form the symmetric normalized Laplacian L. Let Q = L/2. Borrow BernNet's degree 3 Bernstein operators:

Bj = binomial(3,j) (I−Q)^(3−j) Q^j, for j=0,1,2,3.

They sum to I, so the four targets Tj = BjX sum exactly to X. They cover overlapping low-to-high spectral regions; they are not orthogonal or disjoint bands. Branch j always predicts Tj throughout SSL. The target assignments do not rotate across branches. The operators and degree are fixed, not learned from class labels.

Reuse the earlier common raw-input masking and 25 four-update blocks. Within a block, mask the full feature vectors of S, partition S into four equal node quarters, and use the same quarter Qt as the scored node set for every branch on update t. All branches receive only Xmask,Hmask,A. What differs is the feature target of their persistent task, not target-node order. Re-mask all S before the graph decoders.

Use normalized squared error, since residual feature components can have zero norm or signed entries. For task j, minimize the mean squared reconstruction error divided by sj², where sj² is the mean squared entry of Tj over the eligible public unlabeled nodes, with a floor of 10⁻⁶ times the raw-feature mean-square scale. Fix those scales before optimization. The raw-feature control uses its corresponding raw-feature scale. Equal normalization prevents target amplitude alone setting task strength; it can also emphasize noisy weak components, which is a limitation to test through served accuracy.

Original X and BjX appear only as loss targets. They must not enter the SSL encoder/path forward. All class fitting uses TRAIN labels. Public features of nodes outside TRAIN can supply transductive reconstruction targets; no VALID/TEST class labels or pseudo-labels enter the objective.

## One representative comparison

Use WikiCS split 0 and the fixed seeds 17, 29, 43 only in a separately adopted complete protocol. All conditions start from the same competent frozen backbone and private draws, use the same 100 SSL updates, scored nodes, optimizer, decoder dimensions and clean supervised update budgets.

| Condition | Four reconstruction targets | Classification predictor |
| --- | --- | --- |
| R | Every branch predicts full raw X | Four full-label members, fixed mean probabilities |
| G | Branch j persistently predicts Bj(L)X | Same probability bank |
| P | Branch j predicts Bj(ΠLΠᵀ)X, using one fixed label-free node permutation Π | Same probability bank |
| S | Exactly G's four targets and active graph paths | One jointly trained residual classifier on all four private representations |

This is one four-condition comparison across three seeds, or 12 future cells. The permutation preserves graph spectrum, polynomial order, component sum and sparse operator size while breaking alignment between target neighborhoods and the actual graph-feature context. It can also change reconstruction predictability; improved served accuracy is needed to connect that alignment to class utility.

The single has the same four useful graph feature paths and decoder heads as G. It pays four private paths at serving and predicts all labels through a joint class head. R is a repeated-task baseline; P distinguishes correctly aligned graph targets from generic four-component supervision; S tests whether the final bank is needed after matching signal/capacity. No no-SSL arm or loss-weight grid is added to this objective comparison, and sealed earlier arms are not silently treated as interchangeable results.

## Falsification and paid work

G must improve complete served accuracy over both R and P. Lower reconstruction loss or lower common wrong-node overlap alone is insufficient. If P matches G, graph alignment has not supplied the proposed benefit. If S matches or exceeds G, a joint single explains the useful reconstruction signal without retaining four probability members. Failure is not followed by a new basis, mask ratio or objective search.

For future diagnosis, define a feature-boundary stratum using public targets alone, for example the top quartile of (||T2,i||²+||T3,i||²)/(Σj||Tj,i||²+ε). After all fixed endpoints, report donor/member/served accuracy and common wrong-node overlap in that prebound stratum and its complement. This tests the graph-specific error hypothesis without choosing a stratum from held-out errors. Any predictive uncertainty claim additionally requires calibration evidence; disagreement is only a diagnostic.

Three sparse graph-feature propagation passes form QX,Q²X,Q³X; the four components follow by linear combinations. Charge this preprocessing separately for the true and permuted operators. Four11701×300 FP32 target tensors occupy56,164,800 bytes, about 56 MB, before other storage. They are feature targets, not teacher models.

With the same block cache as the original proposal, each condition adds 25 frozen backbone forwards,400 private graph path forward/backward passes and400 GCN decoder forward/backward passes. Four512-to-300 decoders have 615,600 real parameters. Target rows per branch and aggregate target appearances match the original quarter schedule. The new graph targets, loss normalization and preprocessing nevertheless make this a distinct study. A thawed common encoder would invalidate the four-update representation cache and require new forward/backward accounting.

## Closest prior and novelty boundary

BernNet directly supplies the filter basis and identity sum. ADaMoRE already combines low/high structural views, multi-hop foundational experts, residual experts, reconstruction, CKA diversity and cross-filter reconstruction of learned expert embeddings. TFE-GNN and the inspected MORGAN source already combine graph-frequency computations for prediction. ParetoGNN trains one shared graph encoder from distinct SSL tasks with gradient reconciliation.

The narrower candidate is fixed complementary raw-feature targets used persistently to train separately label-competent private paths, tested against repeated raw targets, graph-misaligned targets and an equally capable single. The inspected prior does not establish this exact contrast, and the search is not exhaustive. There is no claim of new spectral operators, a new ensemble principle, novelty clearance or predicted success.
