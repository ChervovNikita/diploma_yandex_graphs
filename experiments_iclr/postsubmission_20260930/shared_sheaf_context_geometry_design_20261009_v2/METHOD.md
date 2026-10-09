# Attributed context supervision for private sheaf geometry — V2

**Status:** inactive source/design packet. V1 is preserved. No novelty, performance, Bayesian-posterior or gauge-inequivalent-geometry claim is made. The independent assessment establishes an exact binary adaptive-CE equivalence; this revision accepts it and withdraws a distinct contrast-loss mechanism.

## 1. The narrow question

Can prospectively assigned, restricted-feature context supervision, with a cross-fitted feature-teacher offset and auxiliary gradients restricted to private incidence generators, improve a competent shared native predictor at its measured training and serving cost? Source dependence, transfer and restored full-input utility must be demonstrated separately. The ingredients are attributed to native NSD, BatchEnsemble ownership, shared/private sheaf ensembling, ordinary masked supervision, logit adjustment/adaptive example weighting and cross-fitting. Their combination receives no priority clearance.

Let shared θ contain every native stem, feature map, slow incidence weight, epsilon and head. Private η_m contains factors **only in the ordered incidence learners**, at every native layer. Each of four members keeps its own complete native state/operator trajectory. The primary loss and serving rule remain

`F = (1/M) sum_m (1/N_T) sum_(q in TRAIN) CE(z_m(G,X;q), y_q)`,

`p_pool = (1/M) sum_m softmax(z_m(G,X))`, with `M=4`.

No feature/head-private factors, member-vector repulsion or free auxiliary decoder are included in this design. The teacher and restricted views are training/assay aids; serving uses the original complete input and native probabilities.

## 2. Prospective queries and sources, covering every TRAIN row

Use canonical node IDs and SHA256 of the exact UTF-8 string `sheaf-context-v2|domain|node_id`. The domains below are distinct. Ties are broken by canonical node ID. Hashing uses no labels or feature values. The schedule is frozen before fitting and shared by every compared arm and seed.

* **Patches:** construct a deterministic BFS forest of the unlabelled undirected support. Choose each unvisited component root and order its neighbours by the `patch-order` hash. Divide the resulting node order into four consecutive bins whose sizes differ by at most one, allocating remainders to lower bin IDs. These bins are P_k. This specifies reproducible graph patches; it asserts neither connected bins nor optimal evidence regions. The native directed/reverse-paired support is unchanged.
* **Query slots:** sort all TRAIN node IDs by the `query-slot` hash. A row at zero-based rank i belongs to `Q_(i mod R)`, with `R=8`. Thus the Q_s are disjoint, cover all TRAIN and differ in size by at most one. They have the authentic class frequency; there is no 32/32 panel or label-stratified reweighting.
* **Evidence:** at post-warmup update t, starting with t=0, set `s=t mod R` and `A_(k,s) = P_k intersect (TRAIN minus Q_s)`. View X^(k,s) keeps raw feature rows only in A_(k,s); every other row is zero. Every queried row is removed from **all** source patches for that update. A TRAIN row can serve as evidence during the other seven slots. VALID and TEST feature rows are never auxiliary sources.

Every Q_s row enters the auxiliary loss for every member in that slot. Do not filter to nearby queries, either class, teacher mistakes or examples that later look useful. This gives complete prospective TRAIN coverage once one cycle is completed. Report distance-to-source strata and counts for interpretation; `distance <= L` is a conservative descriptive stratum, not an exact dependency theorem for state-conditioned degree normalization. Empty patches/source sets, empty slots or undefined teacher fit/selection sets fail qualification; do not search a replacement hash after observing scores. Neither mixed classes in each slot nor equal slot priors are imposed by design.

For exact equal per-row cycle weighting, define

`J_(m,t) = (R/N_T) sum_(q in Q_s) CE(z_(m,k(m),s)(q) + a_(m,s)(q), y_q)`.

The mean over a completed R-slot cycle is the full-TRAIN mean of the slot-specific losses. Averaging separately by slot size would give unequal per-row weights when N_T is not divisible by eight; use the displayed normalization. Losses evolve during training, so the cycle identity concerns exposure weights, not an assertion of a common fixed model across the cycle. Record actual row/slot exposure through the selected checkpoint. A selected checkpoint before any completed auxiliary cycle provides no result for this complete-coverage hypothesis.

## 3. Honest frozen teacher predictions

The ordinary full-TRAIN MLP remains a feature-only benchmark. Its in-sample TRAIN probabilities are **not** the auxiliary teacher.

Use five cross-fitted MLPs per replicate. The architecture is the frozen ordinary feature-only `10 -> 64 -> 64 -> 2` MLP; initialization/optimizer/dropout rules are inherited from the ordinary MLP source, with no strength search. A node's teacher fold j(v) is the first eight bytes of the `teacher-fold` SHA256 digest interpreted as an unsigned big-endian integer, modulo five. This same label-blind map applies to TRAIN and VALID IDs.

For teacher j, its outer fold `T_j={q in TRAIN:j(q)=j}` is excluded from all fitting, checkpoint selection and calibration. Inside `TRAIN minus T_j`, define an inner selection set using the first eight bytes of SHA256 of `sheaf-context-v2|teacher-inner|j|node_id`, modulo eight equal to zero. Fit on the remaining complement; select by inner-selection NLL only, with max500 epochs/patience200 and fresh selected-state reconstruction. Do not refit on the outer fold or calibrate using it. A fold model that fails competent fitting is a failed prerequisite, not permission to replace folds or adjust the teacher using outer outcomes.

For any TRAIN query q, cache `r_q = teacher_(j(q))(x_q)` only after that model is frozen/restored. Thus y_q is used by the sheaf losses and later teacher assessment, but never by the fitting/selection of q's teacher. Teacher labels are never graph feature inputs. Use the identical cache for all paired arms. Clip each class probability to `[10^-4,1-10^-4]` and renormalize. The clipping and all five teacher fits, inner evaluations, reconstruction and cached outputs are charged once per paired replicate; also report the unamortized cost if a method is run alone.

For the VALID assay, use the single frozen fold model assigned by the same j(v), rather than a five-model average with a different confidence distribution. Every VALID row is unseen by every teacher's fitting/selection. Report cross-fitted TRAIN teacher quality and fold/class/count summaries as qualification information; outer-fold/VALID outcomes cannot tune the teacher, schedule, offset strength or masks. Cross-fitting removes direct self-label fitting; it does not prove calibration, eliminate graph-correlated dependence or make ordinary Tolokers a fresh task.

## 4. Ordinary offset CE and its exact binary reduction

For an actual native context forward, capture its exact stem realization H_stem. The analytic zero-message reference from the original general NSD recurrence is

`H_null = (product_l diag(1+tanh epsilon_l)) H_stem`,

`z_null = native_shared_head(vec(H_null))`.

The stem must be captured from this forward, including its dropout; another stem draw is not the paired reference. z_null has no η dependence because private parameters are confined to incidence learners. Set

`a(q) = log r_q - z_null(q)`,

`J_q = CE(z_ctx(q) + a(q), y_q)`.

The offset is held constant in the auxiliary η derivative. This is conventional teacher-dependent logit adjustment. Algebraically `p_adjusted proportional to r * p_ctx / p_null`, and when `r=p_null` it is exactly ordinary context CE. With dropout off and all query raw rows zero, the shared pointwise stem/null is identical across queries and members; its subtraction is a common class offset. The frozen feature teacher supplies the query-dependent local offset.

On this binary task there is a stronger identity. Write `p_ctx=softmax(z_ctx)` and `p_adjusted=softmax(z_ctx+a)`. For finite logits,

`alpha_q = (1-p_adjusted(y_q)) / (1-p_ctx(y_q))`,

`grad_eta J_q = stopgrad(alpha_q) * grad_eta CE(z_ctx(q), y_q)`.

Both binary CE logit gradients are proportional to the same two-class direction. Multiplying by the complete native logit Jacobian gives the identity for η. A detached alpha-weighted CE therefore produces the same private update with the same starting state, optimizer, views and stochastic draws. Different scalar loss values do not create distinct scientific mechanisms. This identity is a static derivation here; numerical agreement is a future implementation qualification, using stable log-error probabilities and no denominator floor that changes the formula.

Consequently this packet contains **one** offset/adaptive-weight recipe, and no trained duplicate as a competing arm. The old `1-r(y)` fixed-hardness comparator is removed: its gradients are unmatched and cannot establish a new contrast. A plain unweighted context-CE arm remains a distinct comparison of weighting/offset policy within ordinary supervised augmentation. A win there would support that bounded policy choice, not a new evidence statistic or novel loss. Record private auxiliary gradient norms from the already required routed gradients to expose differing effective strength at the common lambda; do not claim gradient-magnitude matching merely from using the same lambda. The scalar identity is not asserted for arbitrary multiclass offsets.

Updates remain

`g_theta = grad_theta F`,

`g_eta_m = grad_eta_m F + (lambda/M) grad_eta_m J_(m,t)`, with `lambda=0.1`.

Holding θ parameters constant for J must retain the differentiable current-state, map, normalization and recurrent paths into earlier η. Detaching hidden states or only detaching the null does not implement the recipient rule. This is a restricted update rule; auxiliary changes can influence later θ updates through F. It is not joint optimization of F+lambda*J in all parameters, an ELBO or a separation guarantee.

## 5. Fixed ownership and a constant training-forward budget

After 100 F-only warmup epochs, use a prospectively fixed balanced context permutation, without a TRAIN-score assignment. For replicate seed b, sort context IDs by SHA256 of `sheaf-context-v2|ownership|b|context_id`; member m receives the m-th context in that ordering. Every context belongs to one member. The shared offset, unweighted-CE and zero-source arms branch from the same fresh-restored warmup model/optimizer/RNG state.

This revision withdraws learned-global-assignment claims and its learned-versus-random arm. Covering all rotating rows for a learned assignment would require additional complete context forwards or a partial scoring panel. A learned-assignment extension is a separate, prospectively budgeted question.

Each post-warmup update uses four original-input native member forwards plus **one** native context forward per member: eight complete graph forwards, as in V1. Rotation changes which already-produced rows enter the loss; it adds no graph forward. Static support/index tensors, patches and row membership are prepared once. Native state-dependent maps, block normalization/SVD and propagation still execute in every actual forward. Analytic null evaluation needs no second graph path or dense N-by-N diagnostic. The larger auxiliary query count is an intentional opportunity change from V1 and is identical across V2 arms. Account for its head/loss/backward work; constant forward count is not a claim of equal wall time, memory or total FLOPs.

## 6. Paired source test and prospective transfer assay

The analytic null removes the no-message residual reference. It does **not** remove all transformed stem biases, live diagonal/self terms or structural class priors. Zero raw features can have a nonzero biased stem and nonzero live propagation. Therefore `z_ctx-z_null` is called a prediction difference; it is not assumed to be source-feature evidence.

The required paired **all-zero-source arm** replaces X^(k,s) with zero raw features at every graph row. It retains Q_s, their labels, the same cross-fitted r_q, nominal source masks/ownership, live topology, native operations, geometry recipient, warmup/optimizer/RNG starting state, lambda, number of context forwards and checkpoint rule. Nulls are captured from each zero-view forward. Do not turn propagation off, delete edges, erase biases or presume this makes the difference/gradient zero. If this arm recovers the restored full-input and transferred improvement, informative source features are unnecessary for that effect.

For a minimal transfer assay after full-input checkpoint selection, fix schedule phase `s=0` prospectively. Run one assigned context view per member with the same `A_(k,0)` used in training. Read predictions for **every VALID row**; their raw feature rows are zero in those graph views. This costs four context graph forwards per selected bank, covers the entire VALID query population and performs no reachability/class/teacher-hardness selection. Report native context and teacher-adjusted pooled/member NLL, Brier and AUROC, teacher-only quality, natural-prior class counts and conservative distance/patch strata. Undefined strata are reported, not discarded or repaired. This assay covers one fixed evidence phase; it is not an all-eight-phase robustness claim. A separately authorized multi-phase assay must charge the extra forwards.

The zero-source bank receives the same all-VALID assay with all graph rows zero. Only selected checkpoints are reconstructed and assayed. No context metric chooses checkpoint, teacher, masks, phase or lambda. Full-input VALID AUROC already selects the models, so all VALID utility/transfer results remain exploratory and selection affected. TEST truth/metrics stay unopened. The TRAIN rotation prevents a privileged repeated panel; it still supervises TRAIN labels and cannot establish generalization by itself.

## 7. Claims that remain unavailable

Shared learned networks with private sampled sheaf geometry and predictive ensembling are established by the inspected BSNN §3. Its incidence-factorized conditional sampling limits joint stochastic uncertainty, not coherent conditional means. Global-latent/state-conditioned alternatives are existing extensions and can receive this same ordinary offset CE. Persistent factors are neither posterior draws nor automatically finite-KL Dirac approximations. BuNN, HetSheaf, MIMO graph channels, NSD and BatchEnsemble provide additional representation/ownership collisions.

Retain the original licensed general NSD: ordered endpoint learner, possibly singular tanh maps, reverse-incidence pairing, native augmented SVD degree normalization, jitter/clamping, sparse propagation, ELU and epsilon residuals. The detached unnormalized `.L` is not the deployed operator. General-map transpose is not inverse. Fixed ELU/head/normalization preclude an assumed arbitrary whole-network rotation equivariance. Node/frame and capacity replacements are necessary before any stronger private-geometry attribution.

If all native right feature matrices are zero, messages and the private auxiliary gradient vanish. This checks an unused propagation path; it does not prove informative source usage when messages survive. Neither raw matrix spread nor TRAIN query accuracy substitutes for restored full-input utility and VALID transfer.

The minimal paired design in `EXPERIMENT.json` can assess a bounded context/recipient/weighting/sharing hypothesis. It cannot clear novelty, broad superiority, irreducible connection geometry, Bayesian uncertainty or fresh-task confirmation. Ordinary Tolokers is an original paper benchmark. No numerical work or execution is authorized by this packet.
