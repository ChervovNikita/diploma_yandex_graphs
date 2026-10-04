# Prospective DDI integration protocol

## 1. Fixed architecture and native target path

Use the sealed four-member private-hop F4 adapter exactly: shared learned 512-dimensional node-ID embeddings; shared dense affine weight/bias with private input/output factors and bias offsets; four signed, unconstrained private KI coefficient rows; 15 shared augmented graph powers; four private native two-layer 512-wide Hadamard MLP heads. The constant channel preserves affine bias propagation. No persistent propagation cache, new head or trainable degree coefficient is added.

Preserve the author's commit `0855b0de74a8f0586b8cc203e9ba4dbbb57243f4`, Adam lr 0.001, dropout 0.3, clipping 2, batch 65,536, three global negatives per positive, complete 500 epochs and final partial batch. The epoch sampler is called once and its negatives are shared across members. There are no fabricated weights, features, pretrained/resource states, VALID insertion, target-path masking or learning-rate decay. Native clipping remains over aggregate encoder and predictor groups; learned embeddings stay in Adam outside those clipping groups.

The literal native loss for member m is `T_m = sum_(positive i,negative j=1..3) (1 - (score_m(i+) - score_m(i,j-)))^2`. Retain `T = mean_m(T_m)`. Missing TRAIN weights invoke `calculate_loss(..., margin=None)` and the native AUC fallback, not a hinge loss. Serving and VALID encoding use complete TRAIN and average raw member scores uniformly.

## 2. Auxiliary mask and complete supports

The native target forward is computed first on complete TRAIN. Its common dropped embedding input z is reused for a second, training-only encoder pass on `G_visible`. This prevents hidden incidence labels from remaining in the auxiliary encoder operator while keeping the same ordinary dropout context. Both graphs share the learned embedding and model parameters; only the target graph is served.

Construct Boolean complete TRAIN membership once from all TRAIN records. Per native update remove **every unique undirected edge identity in the entire current positive minibatch**, in both directions, plus every selected auxiliary query identity. Duplicate records cannot reintroduce a removed auxiliary incidence. This is an explicit new HL-GNN auxiliary mask, not an inherited native target mask or the NCNC record mask. The qualified DDI TRAIN has no duplicates or self-loops. Native sampled negatives are absent from full TRAIN; explicit query removal also covers a future assertion defect.

For selected query (u,v), exclude u and v as candidate identities, then use:

`C_L = (N_visible(u) \ N_visible(v)) \ {u,v}`;

`C_R = (N_visible(v) \ N_visible(u)) \ {u,v}`.

Left slots score candidate pairs (v,w); right slots score (u,w) through each route's existing native MLP on the masked representation. Candidate ordering is query order then ascending node ID, left then right. Keep every slot, including empty, zero-count and full-count supports. There is no neighbor sampling, count/degree cutoff, support truncation, top-k, approximate partition or dropping deterministic rows.

Detached teacher bit z_L(w) is membership of (v,w) in **complete TRAIN**, with the right side symmetric. A TRAIN zero means unobserved incidence, not a verified latent nonedge. The teacher/counts are loss-side data only.

Removing only query(u,v) would be vacuous here: after excluding endpoints, every visible residual candidate is absent from the opposite visible neighborhood and also absent from complete TRAIN if no other TRAIN incidence is masked. The whole-minibatch mask creates nonconstant hidden counterpart patterns. Its exposure must be measured on this actual stream; the NCNC DDI census cannot substitute for that measurement.

## 3. Predefined stratified selection and objective units

Choose up to 32 positive query positions uniformly without replacement from the actual native minibatch, and up to 32 positions uniformly from its already drawn 3B native negatives. An owned CPU generator is seeded by SHA256 of the fixed protocol name, seed, epoch, batch and purpose. The same prospective positions/seed rule is used across arms. This is **stratified selection by positive/negative origin**, independent of predictions, support sizes, teacher counts or fitted outcomes; it is not fully label-blind selection.

This 64-query schedule is a stochastic estimator of the declared equal-stratum auxiliary over native training minibatches. It is not a complete query likelihood, a census, or the old mandatory all-query NCNC protocol. Conditional on a fixed minibatch/mask, uniform selected positions estimate the full positive and native-negative population means. Retain every selected row, including deterministic supports. Averaging query-wise exact losses does not turn the graph into independent samples.

For member m, a_m and b_m are the unchanged exact conditional-Bernoulli side NLLs. For d=max(n_L+n_R,1):

`J = -log[(1/4) sum_m exp(-a_m-b_m)] / d`;

`S = {-log[(1/4) sum_m exp(-a_m)] -log[(1/4) sum_m exp(-b_m)]} / d`.

Joint uses one member for both sides. Separate mixes each side independently. At identical logits both laws have the same side marginals. Define A as half the selected-positive mean plus half the selected-negative mean, using J or S respectively. The proposed fixed coefficient is lambda=1 in **mean native squared-margin units**:

`L_joint = T + (3B) * lambda * A_J`;

`L_separate = T + (3B) * lambda * A_S`;

`L_target_only = T`.

Thus `L/(3B)` equals the mean native squared-margin term plus lambda*A. The native target sum is not silently divided or replaced; a raw coefficient 1 on the unscaled auxiliary would otherwise have different units. This scaling is a declared DDI adaptation, not the Collab coefficient rule or a measured optimal value. Root must review it before any scores; there is no coefficient grid in this packet.

## 4. Responsibilities and collapse

Joint responsibility is `rho_J(m)=softmax_m(-a_m-b_m)`; separate responsibilities are `rho_L=softmax(-a_m)` and `rho_R=softmax(-b_m)`. They weight auxiliary derivatives only. The target remains a uniform mean of native AUC sums. No responsibility, count, bit, subset score, entropy or degree-null output changes target sampling or served ranks.

At the same logits, `J-S = -log[4 dot(rho_L,rho_R)]/d`. The probe computes this identity in log space. There is no universal ordering. Identical component laws give uniform responsibilities and J=S. Either deterministic side eliminates the joint/separate distinction for that query. Component equality can persist under common informative reconstruction gradients, and collapsed optima are permitted. Joint training does not guarantee specialization, prevent dead members or identify a community model.

The saved initialization gives identical encoder routes (r=s=1, private offsets 0, equal KI rows) and fresh independently initialized native MLP heads. This breaks output symmetry without adding a diversity loss or changing initial state across arms. Monitor effective use and collapse descriptively; do not rerandomize a member, add entropy balancing, select a favorable seed or expand a grid after an outcome. Shared parameters and overlapping neighborhoods also limit a latent-class interpretation.

## 5. Matched family and selected serving comparison

Prospective family: seeds 0,1,2 x target_only,joint,separate, all fresh constructors/embeddings/Adam and all 500 epochs. The bridge has identical parameterization/initialization across arms. There is no donor state, warm start or state from a resource qualification. Reset the same native seed before each cell. Auxiliary scorer dropout uses an owned per-update seed inside `fork_rng`; fixed 8192-slot checkpoint chunks preserve scorer RNG during backward. Additional forward/recomputation must restore the native CPU/current-GPU RNG stream. Reuse z; draw no second input dropout.

Keep native negative draw, minibatch permutation, target dropout and head dropout aligned across arms; only learned parameters may diverge. Receipts include initial parameter hashes, epoch negative hashes, batch record IDs/RNG hashes, selected indices/query hashes, and joint/separate mask/support/teacher hashes. The complete-family comparator verifies these. A mismatch preserves all completed data but makes paired interpretation ineligible. No replacement cell or favorable subset is allowed.

Every 5 epochs traverse complete fixed official VALID through inherited OGB Hits@20/50/100. Select VALID Hits@20 by strict improvement, first exact tie. Run one complete replay of each own selected state after500 epochs; no diagnostic selects a state. There are 101 complete VALID traversals per cell and 909 for the full family. TEST remains inaccessible.

**Primary:** joint minus separate. The separate arm is the matched same-masked-context reconstruction reference; adding another reconstruction arm would expand this pilot. **Secondary:** joint minus target-only and separate minus target-only. These secondary contrasts include masked augmentation and extra compute. Same-state law switching isolates local responsibility coupling algebra; end-fit arm differences also include changed marginals/optimization and cannot be called a pure measured coupling effect.

Report every seed difference, selected epoch, mean, sample SD, range, descriptive t95 interval(df 2) and all 8 sign flips. Fractions convert to percentage points by multiplying differences by100; with 3 blocks the smallest nonzero two-sided sign-flip p is 0.25. These describe seed variation on one development graph, not graph-level confidence. Missing/failed cells leave the family incomplete; publish no success-only mean or best-seed summary.

If joint fails to improve on separate, simplify any cross-side-member claim even if reconstruction helps versus target-only. If both fail versus target-only, retain the failure and close this candidate; no post-outcome tuning grid. VALID development gains require separately fixed confirmation and competitive native/ordinary-ensemble baselines before a generalization claim.

## 6. Degree references and passive mechanism diagnostics

Count conditioning cancels a common side-logit offset. It does not cancel candidate-specific `log(1+degree(w))`, fix every candidate node's degree, remove graph density or prove degree invariance. The saved degree-prior conclusions are reused without claiming a new degree-corrected block model or novelty.

After selected models exist and a probe cost is admitted, prerelease a fixed TRAIN-only probe stream using this actual HL-GNN mask and selection seed; score every selected row, with positive/negative aggregates and informative denominators additionally reported. Use the same complete supports, labels and d for uniform `log binomial(n,k)` and visible-degree logits `log(1+degree_visible(w))`; no fitted coefficient or full-TRAIN degree shortcut. This new stochastic-probe description does **not** fulfill or silently replace the saved NCNC full-stream diagnostic plan; that plan remains retained.

Cross-score both J and S at each selected joint and separate parameter state (2x2 fitted-state/scored-law table). Report uniform/degree-relative NLL, entropy/log 4 of side/joint responsibilities, 4*endpoint overlap, same-state gap/identity residual, one/two-variable exposure and component-law spread. Extreme supports remain in population means. Actual target/auxiliary gradient norm/direction and member starvation probes require a separately admitted backward cost; they are not extra training regularizers. None rescues a failed served-prediction comparison or selects hyperparameters.

## 7. Dense-DDI cost and root budget review

The artifact has 1,067,911 records: native batch 65,536 gives 17 updates/epoch, including the 19,335 tail. The complete family is 76,500 native updates; six auxiliary fits add 51,000 masked 15-hop passes and at most 3,264,000 selected queries. Encoder propagation per auxiliary update doubles the native shared-power passes from 15 to 30; this is not free factorization. Four native target heads already score 4B pairs per member. Across a query, residual supports are disjoint, so at most N-2 candidate slots survive; the 64-query bound is 64*(N-2), still dense.

Exact conditional normalization costs O(M*n*r) with r=min(k,n-k); the copied bucket/complement/r1 methods retain the exact law. DDI can have larger n*r than Collab. Rolling ESP forward tensors are **not** an O(r) training-memory bound: autograd stores recurrence history. Fixed chunked checkpointing controls head activations, not ESP history or the two live encoder graphs. Do not infer memory/time from the Boolean census or the synthetic encoder equivalence. The completed Collab exact-CB diagnostic held both laws/four reverses and peaked 28.81 GB allocated/44.93 GB reserved; those figures are not this fit's memory.

Root review decisions: accept the unique-edge mask,64-query estimator,lambda scaling, identical joint/separate recipe and full native500-epoch budget; bind the qualified artifact path/report; measure/admit actual native/F4 and combined-update cost under existing server supervision before scheduling fits. This packet adds no launch client, server command, qualification ladder, GPU-hour estimate or implicit run authorization. If the proposed complete-support update exceeds the reviewed cap, record infeasibility and retain this packet; a smaller Q/chunk/precision/batch/mask/budget requires a prospectively revised source, not an automatic outcome-dependent fallback.
