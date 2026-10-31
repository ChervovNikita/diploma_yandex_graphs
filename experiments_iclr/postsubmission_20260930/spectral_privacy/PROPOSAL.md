# Feature subspaces for private graph transformations

Status: bounded methodological proposal, 30 September 2026. No graph dataset was loaded or scored, no GPU job was run, and no submitted source or manuscript was edited. Public paper snapshots and the synthetic check are local to this folder.

## Decision

**Advance one small mechanism pilot: select a frozen output feature subspace for each of a fixed set of graph transforms using recurrent centered member gradients minus graph-unit sampling variability.** Train zero initialized private coefficients in those subspaces. Permit rank zero. Keep the target transforms fixed, so the question differs from choosing which complete layers to copy.

The adapter construction, gradient SVD, spectral denoising, and rank adaptation are established ideas. Their combination here is **not demonstrated methodological novelty**. September-2026 GDPS already combines within-task versus cross-task agreement, noise-sensitive sharing ratios, joint gradient SVD/CCA, and gradient-energy-weighted private initialization. The falsifiable addition is the exact matrix statistic and its ability to distinguish reproducible output feature directions of same-task ensemble disagreement from transient graph-unit variability, prospectively. A raw positive eigenspace of cross-fold gradients is too weak: it ignores the cost of noisy private directions and is particularly vulnerable to selecting noise in a high dimensional eigensystem. The stronger candidate below subtracts a specified noise covariance, separates basis discovery from screening, and requires controls against ordinary gradient initialization. Recommend a **mechanism pilot**, not a standalone new-method claim; a graph application alone does not establish novelty.

This is a parameterization of the submitted member transformations. It retains K full width member hidden states, their nonlinear forward/backward computations, and their propagation. It offers no activation-width or sparse propagation saving. It is deliberately distinct from the common/narrow-state architecture in `efficient_paths`.

## Fixed target and exact parameterization

For the first source SAGE experiment, target only the neighbor weight matrix `residual_modules[l].module.conv.lin_l.weight` in each block. Source `nn.Linear` weights have shape `[out,in]`; with width h these are h by h. Keep root weights, biases, normalization, FFNs, boundary BatchEnsemble factors, graph preprocessing, readout, and the member loss definition as in the matched source arms. The coordinator's new PPI recipe currently specifies three residual blocks, width 128, hidden multiplier one, and four members; these are pre-outcome compatibility choices rather than evidence for the proposed method.

For each target l, let U_l have r_l orthonormal columns in R^h, frozen after selection. Fix a deterministic Helmert matrix C in R^{K by (K-1)}, with C^T C=I and C^T 1=0. Introduce trainable coefficient matrices A_lq in R^{r_l by h} and define

    W_lm = W_l + Delta_lm
    Delta_lm = U_l sum_q C_mq A_lq .

Shared W_l remains trainable. Initialize all A_lq to zero, which preserves the common checkpoint's exact function. Coefficients are centered across members: sum_m Delta_lm=0. The correction of each individual weight matrix has feature rank at most r_l. The source row-feature forward is

    F.linear(z, W_l, shared_bias)
      + F.linear(F.linear(z, sum_q C_mq A_lq), U_l).

Use a member-aware wrapper around the same `lin_l` operation; preserve the SAGEConv bias placement. No modification of message aggregation is required. A frozen U and zero A also avoids the zero-times-zero gradient problem of initializing both factors of a bilinear LoRA correction to zero.

**Two different ranks:** K-1 bounds the member contrast axis of the stack of flattened corrections; r_l bounds the output feature subspace of an h by h transform. The former does not bound the latter. For K=2, centered gradients diag(1,2,3) and -diag(1,2,3) have flattened member contrast rank one but output covariance diag(1,4,9), of rank three. Calling this a rank-(K-1) feature adapter would be incorrect.

Added trainable scalars per target are (K-1)r_l h. Frozen U_l adds r_l h stored scalars, and shared C adds K(K-1) negligible scalars once. Thus a storage budget must charge approximately K r_l h, while optimizer state charges the trainable part. For K=4,h=128,r_l=16 and three targets, the adapter adds 18,432 trainable coefficients and 6,144 frozen basis scalars, plus C. The original shared W_l is still stored. Direct independent rank-r LoRA with two trainable factors is a different parameter and optimizer budget. A rank-h version spans all centered private neighbor-weight corrections, but its optimization trajectory is not that of copied complete graph blocks.

## Matrix statistic at a fixed probe state

Fit a tied probe using P training graphs only. Evaluate score graphs in eval mode, with autograd enabled, deterministic message preprocessing, and no dropout or running-stat updates. For whole graph unit i and each member m, collect the **unscaled own-member** gradient matrix G_lim of the graph's mean finite-observed-entry loss. Center across members before any covariance:

    D_lim = G_lim - mean_m G_lim .

Each graph receives equal weight, matching the new pilot's equal-graph mean-loss updates. Within-graph nodes, labels, or edges are not independent score units. Members use the same graphs and observed-label mask. Per-member centering is performed separately on each graph. Cross-member dependence within a graph is allowed in the conditional working model below.

For a group J of n>=2 score graphs, write Y_lm=mean_i D_lim. Define output covariance matrices

    Sigma_hat_l = 1/[K(n-1)] sum_(i,m) (D_lim-Y_lm)(D_lim-Y_lm)^T
    S_hat_l     = 1/K sum_m Y_lm Y_lm^T - Sigma_hat_l/n
    N_hat_l     = Sigma_hat_l/n0
    T_hat_l     = S_hat_l - N_hat_l .

Here n0 is an explicit, frozen **reference number of independent equal-weight graph units in one private update**. It is a runner field, not the number of score graphs or a validation-tuned threshold. The coordinator confirmed that the first PPI trainer updates once per complete graph, so **n0=1**. Do not transfer a rank rule between batch regimes silently. If a future runner accumulates all graph losses into one step, n0 is that graph count. A different n0 is a method amendment. This differs from the existing whole-layer selector's N, which describes noise in its discovery fold average and does not assert optimal shrinkage for actual graph-at-a-time updates.

S_hat is equivalently the symmetric off-diagonal graph-pair U-statistic

    1/[K n(n-1)] sum_m sum_(i != j) D_lim D_ljm^T .

All ordered pairs are present, so symmetry is exact apart from reduction error. Accumulate in float64 and explicitly symmetrize before `eigh`. This form generalizes sym(mean_m D_A,m D_B,m^T) to all distinct graph pairs without making multiple overlapping A/B partitions look like independent replications. Matrices act on **output** features. D_A^T D_B would instead select input features. Neither a member Gram matrix of size K by K nor the existing flattened block Gram summaries can reconstruct these h by h feature covariances. A new per-graph collector must expose each named matrix gradient; it can stream graph/member reductions and save source/state/unit identities. For Q10,K4,h128 and three target matrices, saving all float64 matrix gradients costs about 15.7 MB before metadata; retaining them is optional if exact covariance records can be replayed.

Do not take the SVD of T_hat, square it, or clip S_hat to positive semidefinite before subtracting N_hat. Signed eigenvalues encode negative recurrence; these operations would erase its sign or change the statistic. S_hat and T_hat may be indefinite on finite graphs even when their population signal covariance is positive semidefinite.

## One implementable selection rule

Use a single predeclared probe checkpoint, epoch 25, to avoid a search over probe times. Maintain the same source initialization and optimizer seed for all matched arms. Split the 20 PPI training graphs into P10 and Q10 using the existing P/Q seed 700. Shuffle Q with a new, frozen seed 704 into basis-discovery D4, screening E4, and local-audit F2. Publish graph IDs before fitting. These folds belong to a new spectral study; they do not overwrite the previous layer-selection protocol.

For each fixed target independently:

1. Compute T_D from D4 and eigendecompose it in descending signed eigenvalue order. Retain strictly positive eigenvectors as candidates.
2. For each fixed candidate u_j, compute t_Ej=u_j^T T_E u_j using E4. Keep it only if t_Ej>0 as well. An exact zero fails the gate.
3. Order eligible candidates by min(lambda_Dj,t_Ej), break exact ties by discovery index, and keep at most 16 per target. The static architectural ceiling is r_max=16, fixed before outcomes. Rank is the number retained; no eligible direction yields r_l=0.
4. Freeze U_l, rank, parameter/buffer counts, source/state/fold hashes, objective weighting, n0, and all scores before opening F2 diagnostics or final validation outcomes.

This uses at most the 24,576 additional stored float scalars calculated above. It does not force spending the budget, select complete layers, or sweep rank against validation. The min score and two-group gate are conservative engineering choices, not unbiased estimates or confidence intervals. Positive screening in two small groups can still admit false directions. A member-permutation negative control must quantify this failure, without interpreting its outcome as an exact randomization p-value unless the null exchangeability assumptions are separately justified.

E4 selects directions, so the selected E4 score is optimistic. F2 supplies a genuinely unused, training-only local check after the lock; with two graphs it supplies little statistical power. Three repeated partitions of the same graphs would not create more independent graphs. For MolHIV use the same procedure on independently grouped training scaffolds, with larger fixed D/E/F group counts recorded before outcomes. Treat each scaffold group as one equally weighted unit if that is the declared target objective; do not call molecules within a scaffold independent replications.

After selection all final arms continue from the same P-fitted epoch-25 common checkpoint with optimizer state reset in the same way, then train on all original training graphs under the same final budget. This warm-start choice gives the local basis a defined state and must apply to tied/random/raw-SVD controls too. It is a separate new study: it cannot borrow final cells trained from canonical initialization in the prior layout experiment. Charge the common probe and selection cost. Restarting all arms from canonical initialization could be an additional frozen variant later, because basis transfer across the intervening training trajectory is then a separate hypothesis.

## Conditional proof sketch and limits

Condition on a fixed probe state learned independently of the scoring graph units. Suppose scoring units are IID from a specified graph population, with finite second moments. Let

    D_im = delta_m + epsilon_im, E[epsilon_im]=0
    S = mean_m delta_m delta_m^T
    Sigma = mean_m E[epsilon_im epsilon_im^T] .

IID is needed across graph units, not across members. Within-graph member errors may be correlated. Then E[S_hat]=S and E[Sigma_hat]=Sigma. For a **fixed orthogonal projector** P independent of a fresh n0-unit mean Y, Frobenius denoising risk is

    R(P) = mean_m E ||P Y_m - delta_m||_F^2
         = tr(S) - tr(P[S-Sigma/n0]) .

The oracle rank-at-most-r projector therefore contains the leading positive eigenvectors of T=S-Sigma/n0, capped at r. This follows directly by diagonalizing T and maximizing tr(PT). It is a classical spectral risk argument applied to centered member demand. For each D-discovered fixed direction, E screening estimates its oracle risk advantage under independent E graphs. **It does not make the final adaptively screened subspace, its reported E score, or its selected rank unbiased.** Replacing population T by T_D is a statistical approximation; the two-group gate adds no finite-sample generalization theorem.

There is a separate local loss identity. At the common state with private coefficients zero, native SGD on the mean member A-fold objective updates W by -eta mean_m G_A,m and the centered adapter by

    Delta_m = -(eta/K) P D_A,m .

Holding all other updates equal, the mean member B-fold loss difference versus the tied step is

    -(eta/K) tr(P C_AB) + O(eta^2),
    C_AB = sym(mean_m D_A,m D_B,m^T) .

The 1/K factor comes from the actual mean member objective and the orthonormal contrast coefficients; silently multiplying private gradients by K would be an additional optimizer intervention. N_hat motivates denoising risk; it does not appear in this first-order loss identity. AdamW with coordinatewise moments is not rotationally invariant and need not follow this SGD identity. Fixed Helmert coordinates and coefficient optimizer conventions must therefore be recorded. Neither proof predicts final pooled micro-F1/AUC or calibrates the private SGD step size for Adam.

Whole PPI graphs may still share collection biology; scaffold groups are not proven IID either. On a single transductive graph, disjoint node labels share topology and neighborhoods, so the IID argument is especially inappropriate. If fold noises are correlated, cross products contain that covariance, and noise subtraction can miss it. The synthetic check gives the extreme counterexample: every score graph shares the same zero-mean noise realization, making estimated N almost zero and T positive while true member demand is zero. Conditioning on a probe fitted on all score labels also breaks the independent-state premise.

Graph-unit variability is broader than corrupted-label noise. Rare yet useful graph-specific adaptations can be filtered out as variance. The Frobenius metric is not invariant under arbitrary feature reparameterization or Hessian/Adam preconditioning. Bases may drift out of relevance as W, FFNs, and boundary factors train. A graph-specific full adapter can learn rotations that a frozen basis cannot. These are reasons for the pilot to fail, not omitted terms in a claimed theorem.

## Small PPI / MolHIV pilot and cheap falsifiers

This folder contains a specification and synthetic checks only. The proposal can be passed to the coordinator's frozen source-compatible runner after ordinary source/data/environment gates.

Use three paired seeds and four matched final arms per task: tied targets; selected spectral bases/ranks; random Haar output bases at exactly the selected per-layer ranks; and raw centered-gradient bases at those same ranks. The raw basis is the left SVD of the concatenated D-group mean matrices [Y_1 ... Y_K], equivalent to the eigenbasis of mean_m Y_m Y_m^T. It is a controlled adaptation of gradient-based LoRA initialization, not an exact reproduction of LLM fine-tuning. Thus the core pilot is 12 final cells per PPI mask regime, then 12 on MolHIV if PPI mechanism gates are useful. The coordinator's fixed 10%/100% observed-node regimes double the PPI core to 24 cells if both are included. An all-rank-16 raw-SVD arm adds three cells per regime and tests rank adaptivity; do not expand to a rank HPO grid. A full centered private-neighbor-map arm is a useful capacity reference: at h128/three maps it needs 147,456 private trainable coefficients, versus at most 18,432 here. It cannot honestly be called matched-cost at this fixed width and depth.

Use PPI official inductive split, whole graph folds, unmodified features/edges, multi-label BCE, source width128/three-block SAGE/LayerNorm/four members, source dropout and fixed optimizer recipe. Fix the same immutable 10%/100% observed-node masks in every matched arm; the complete graph topology/features remain available. Fit/final objectives must both average graph-specific mean observed-target BCE, matching the collector. Record masks and gradient counts before scoring and make missing targets explicit to the collector. Keep pooled validation micro-F1/BCE/earliest checkpoint and mean raw member logits at inference. Log primary mean member BCE and practical pooled outcomes separately. Pooled node metrics do not change the equal-graph training score definition.

For MolHIV use the official scaffold split and the coordinator's uniform atom/bond-aware encoder/readout wrapper in all arms. Do not claim the pure SAGE prototype has become chemically bond-aware by adding U A. If the matched molecular backbone exposes a square hidden transform after bond-aware messages, fix that transform for every arm; otherwise delay MolHIV until the source wrapper specifies a compatible fixed target. Scaffold grouping and loss weighting require their own frozen manifest. Validation ROC-AUC/BCE/earliest is common to all arms; test scoring occurs only after global locks. No existing submitted score is replaced.

Cheap controls share the same gradient collection, requiring no extra long training:

| Control | Specific question |
| --- | --- |
| S_hat without Sigma/n0 | Does noise subtraction reject transient high-energy directions? |
| Permute member labels independently by score graph | Is stable member correspondence needed, and how much null rank is selected? |
| Common-gradient basis from mean_m G_m | Is private demand different from general optimization demand? |
| Principal weight basis of W (PiSSA-style) at matched ranks | Does recurring demand add information beyond weight geometry? |
| n0 sensitivity reported at 1 and actual fixed batch count, without choosing the better result | Is the rule's noise interpretation consistent with the intended update regime? |
| SGD projected one-step audit on F2, eta and eta/2 | Does tr(P C_DF) predict the measured first-order mean member loss change with the native 1/K scale? |
| Local projector overlap and graph-by-graph score traces | Is rank concentrated in unstable directions or one graph? |
| GDPS self-task versus cross-task cosine score, computed on the same graph/member gradients | Does the matrix noise criterion add predictive information beyond the closest recent scalar sharing score? Report its relationship with F2 projected-step outcomes and selected rank; label this a same-task graph adaptation. |

The F2 one-step audit performs the same shared step in every arm, copies optimizer-free states, and changes only the centered projected correction. Its outcome is a training diagnostic, never checkpoint or rank selection. Report repeated-unit dependence and uncertainty without treating seeds, members, nodes, or eigenvectors as independent tasks.

Stop the method claim if (a) the local identity fails implementation/scaling checks; (b) null permutations regularly select similar rank/advantage; (c) matched-rank random or raw-gradient bases match the selected basis on unused training graphs; or (d) the basis becomes irrelevant before useful updates. Stop a rank-adaptivity claim if the fixed rank16 raw-gradient arm matches utility/cost. A successful small pilot supports only a mechanism hypothesis; prospective independent tasks and relevant graph/efficient-ensemble baselines are still required. Report storage, optimizer state, activations, latency, and total probe/final time separately; expect no propagation or activation improvement from this parameterization.

## Bounded closest-prior audit

Full-text or publisher-page snapshots were retrieved for the first seven entries below; the recent label-noise paper also has a retrieved full HTML. The audit is bounded to the mechanisms relevant to this candidate. No claim of first use or an exhaustive September-2026 literature review is justified.

| Prior | Inspected mechanism and consequence |
| --- | --- |
| [LoRA-GA](https://arxiv.org/abs/2407.05000), 2024 | Gradient SVD initializes low rank factors to approximate the first full fine-tuning step. Gradient-selected low rank bases are prior art; a raw-gradient basis is mandatory. |
| [PiSSA](https://arxiv.org/abs/2404.02948), retrieved v4 2025 | Trains principal weight components while freezing a residual. Weight spectral bases and an SVD adapter are prior art. |
| [GaLore](https://arxiv.org/abs/2403.03507), 2024 | Projects gradients into low rank SVD subspaces and can switch subspaces during full parameter training. Projected gradient geometry and memory claims are prior art. |
| [EVA](https://arxiv.org/abs/2410.07170), retrieved v5 2025 | Uses stable incremental activation SVD and redistributes ranks under a budget. Data-based stable bases and adaptive rank allocation are prior art. |
| [CorDA](https://arxiv.org/abs/2406.05223), 2024 | Decomposes pretrained weights using context activation covariance, including different adaptation/knowledge-preservation orientations. Data-aware decomposition is prior art. |
| [TAPS](https://openaccess.thecvf.com/content/CVPR2022/html/Wallingford_Task_Adaptive_Parameter_Sharing_for_Multi-Task_Learning_CVPR_2022_paper.html), 2022 | Jointly learns active private layers and their deltas with a sparsity penalty. Learned sharing and private deltas are prior art. |
| [Recon](https://arxiv.org/abs/2302.11289), 2023 | Uses layer task-gradient conflict scores to choose private layers. Gradient-based unsharing is prior art; the candidate changes the feature subspace and noise criterion. |
| [Sun & Nakamura, GDPS](https://arxiv.org/html/2603.25836v2), v2 17 September 2026, full text inspected | §3.2.2 uses within-language sample-gradient cosines versus cross-language cosines to distinguish task noise and shared signal, then maps their difference to sharing ratios 0.75/0.50/0.25. §3.2.3 additionally uses joint gradient SVD and ridge CCA for subspace alignment and energy proportions. §3.4 builds shared/private FFN capacity with weight factorization and gradient-energy-weighted residual SVD initialization. This is close prior art for **both** noise-sensitive sharing and spectral shared/private capacity. |
| [GraphLoRA, cross-graph transfer](https://arxiv.org/abs/2409.16670), 2024/2025 | Adds a small trainable GNN beside a pretrained GNN, with feature alignment and structure-aware regularization. A graph adapter is prior art; same-task member demand is a different experimental setting. |
| [Why LoRA Resists Label Noise](https://arxiv.org/abs/2602.00084), 2026 | An explicit low rank bias/variance and noise-dependent optimal-rank analysis with rank-aware curriculum. A rank/noise tradeoff is already claimed in close recent work; do not present the simple denoising proof as a new theory. |
| [NaRA](https://arxiv.org/abs/2605.29716), 2026, abstract inspected | Conditions a low rank core on the diffusion noise level. Its noise is generative diffusion state, unlike graph-unit sampling variation, but “noise-aware LoRA” is not a new label. |
| [Shared representations beyond well-specified regression](https://arxiv.org/abs/2501.18975), 2025, abstract inspected | Low rank/cluster shared structures under Hessian and noise concentration, with rank/nuclear-norm estimators. General noise-sensitive shared-subspace theory is prior art. |

The coordinator supplied the GDPS identifier after the bounded search failed to identify it; its full public source was then downloaded and inspected here. Google/DuckDuckGo retrievals were blocked; OpenAlex returned 429; Crossref partly returned 429. Targeted arXiv API queries recovered the recent noise/low rank and shared representation entries above. Search coverage remains incomplete. GDPS directly precludes a broad “noise-sensitive gradient sharing” or “spectral shared/private allocation” novelty claim. Its scalar self/cross cosine gate and joint task-subspace SVD/CCA differ from the candidate's centered output-matrix U-statistic, explicit Sigma/n0 subtraction, and independent basis screen. Those precise differences are hypotheses requiring evidence; they are not proof of originality. A prior with the same centered matrix risk rule would subsume the candidate. If the specific criterion adds no predictive value beyond GDPS/raw-gradient controls, retain it as a diagnostic and stop this method direction rather than rename a graph adapter as a new algorithm.

Retrieved sources distinguish the GNN transfer GraphLoRA above from the unrelated [2026 LLM recommender GraphLoRA](https://arxiv.org/abs/2606.07526). A guessed arXiv identifier was checked, found to be an unrelated algebraic-geometry paper, and excluded. Never cite it as GraphLoRA.

## Synthetic evidence and delivery boundary

`sanity_spectral.py` is a pure standard-library, synthetic small-matrix check. `SANITY_REPORT.json` reports PASS: native contrast-coordinate and exact quadratic identities agree to 1.73e-18; K=2 still permits output feature covariance eigenvalues 9,4,1; the fixed-projector IID Monte Carlo estimates T with maximum error 0.0114 over 4,000 draws and expected risk change -3 with observed -2.996; the dependent-graph example produces false signal 5.896 with essentially zero estimated noise. A two-unit example has positive recurrence eigenvalues 10,1,0 but corrected eigenvalues 1,0,-10.25, illustrating why positive recurrence alone can retain the largest noisy feature. Negative recurrence eigenvalues are preserved. These are algebra/simulation checks, with no learned-model performance implication.

Only the assigned proposal folder was written. Read-only local context was limited to `source_snapshot/models.py`, `sharing_policy/PROPOSAL.md`, `sharing_policy/recurrence_collector.py`, `efficient_paths/CANDIDATE.md`, and `efficient_paths/centered_paths.py`; coordinator messages supplied the current pre-outcome recipe amendment. No broader files, datasets, manuscript, or GPU environment were accessed.
