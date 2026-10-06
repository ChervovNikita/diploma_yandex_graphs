# Compression initialization from independently learned GNNs

## Decision

One larger pipeline is worth a prospective test: first learn four genuinely independent competent GNNs, then compress their learned internal operators into a trainable shared body with private low-rank corrections, retaining each teacher's identity during joint continuation. This changes the source of competence and the internal architecture, rather than perturbing factors around one warm predictor. Its usefulness is unmeasured. Generic model merging, low-rank base estimation, graph-aware alignment and multi-teacher member distillation are already established. The complete recipe is not certified novel.

The important question is whether independently acquired useful functions fit inside the proposed shared capacity at an acceptable deployment cost. If they do not, initialization alone cannot remove that restriction. A private nonlinear block or wider backbone would then be a different architecture hypothesis, requiring a new prospective study.

This packet contains literature and design only. No models, datasets, checkpoint payloads, numerical experiments, GPU jobs, server commands or old sources/scores were accessed or changed.

## Closest primary evidence newly read

Six first scoped method reads were made beyond index72 and the checked earlier source packets; no full-paper or result certification is claimed. Two duplicate Git Re-Basin/OT PDFs were mechanically retrieved before the older packet exclusion was resolved; they were not substantively reread or counted. Failed metadata searches and incidental result-table exposure are recorded.

| Work and exact source | Operation supported by the scoped primary | Consequence and limit |
|---|---|---|
| TIES-Merging, arXiv2306.01708v1, PDF pp3–5 | Fine-tuned models derived from a common initialization yield task vectors. Trim small changes, elect a coordinate sign, average only agreeing changes, then scale/add to the base. | Discards conflicting directions to produce one multitask checkpoint. It does not retain four member functions, and independent-from-scratch GNNs need architecture-valid alignment first. Coordinate signs are not a coordinate-independent measure of semantic conflict. |
| DARE, arXiv2311.03099v1, pp4–5, with introduction scope pp2–3 | Drop/rescale fine-tuning deltas from the same pretrained backbone before ordinary merging. | Small SFT deltas and output expectations are not a GNN member-competence guarantee. Sparse masking need not yield low rank. Printed Eq1 uses Bernoulli(p), multiplies by that mask, then divides by1−p while calling p a drop rate: those literal conventions are inconsistent under the usual Bernoulli meaning. No code port is qualified here. |
| ZipIt!, arXiv2305.03053v1, pp3–5 | Match correlated features within and between differently initialized models; construct merge/unmerge maps and propagate them through layers; optionally leave later layers private. | Independent-teacher compression, partial sharing and preserved private upper branches are close ancestry. The paper explicitly writes approximate unmerging because its merge is not full rank. Feature correlation and arbitrary averaging do not imply exact nonlinear function preservation. |
| LoRE-Merging, ACL2025.findings-emnlp.1195, pp2–3 and p8 Algorithm1 | Estimate a common base and low-rank model deltas without an externally supplied base, then merge the deltas. | Directly closes a claim that estimating a shared base plus low-rank differences from pretrained models is new. Printed Eq1 uses squared nuclear norm, while the stated fixed-threshold SVT update is not generally that objective's proximal solution; Algorithm1 also has a self-referential TIES averaging line. Do not inherit an exact solver guarantee. Stored members versus one merged checkpoint remain distinct goals. |
| GFFMERGE/GNNMERGE, arXiv2606.03232v1, pp3–5; AppendixG selected pp25–26 | Match source node embeddings, relax dependence across layers by feeding teacher states into each local linear regression, solve linear blocks in closed form, then repair later nonlinear/task layers in force-field experiments. Generic GNN use is also described. | Graph-dependent pretrained alignment and initialization are direct prior. The layerwise relaxed solution is not an end-to-end nonlinear preservation theorem. Invertibility/ridge, attention/nonlinear blocks and actual native implementation still require qualification. The generic appendix includes WikiCS; its scores/timing do not qualify our backbone. |
| H-GRAMA, arXiv2602.19332v1, pp2–6 §3 | A label-free heterogeneous GNN operator basis; CKA depth matching, rectangular Procrustes transport, gate regression/convex mixing, then destination-conditioned message-moment calibration. | Graph operator alignment/merging and message calibration already exist. Arbitrary orthogonal transport does not commute with ReLU, sigmoid, products or LayerNorm affine parameters generally. Folded affine message calibration is an exact implementation identity for that correction, not preservation of every parent function or of useful joint ensemble errors. |

LoRE's symbolic inconsistency is specific: for the displayed objective `||D−E||F²+µ||D||*²`, singular values satisfy `s_i=(σ_i−µ sum_j s_j)+`. A constant thresholdµ is not the general solution. This does not refute reported empirical usefulness; it limits what can be imported from the printed derivation. No numerical reproduction was performed.

## Previously read work reused

BatchEnsemble already supplies `W_m=diag(s_m) W diag(r_m)` and full private nonlinear trajectories. MIMO learns several functions inside one network from independently paired training inputs; repeating one intact graph in every input slot throughout training is a multihead graph port. Packed-Ensembles keeps separate subnetworks and pays their computation. Hydra uses a common student body and persistent teacher-matched heads, so individual teacher targets and finite member association are prior. These are not new reads.

Git Re-Basin's exact architecture-valid permutations and OT-Fusion's layerwise transport were already read in saved packets. Exact permutation is different from approximate feature mixing. CKA is invariant to transformations that need not preserve native channel semantics; high CKA alone cannot justify copying coordinate-dependent graph operators. We reuse those conclusions without new primary credit.

Network bootstrap supplies dependent statistical multipliers, not compression of learned GNN functions. Virgo supplies graph-dependent cold variance initialization under approximate assumptions, not independent-teacher preservation. GradMax supplies zero-incoming/nonzero-outgoing growth with initial function/old-gradient inclusion and a live incoming derivative. PreGS already transfers a pretrained supervised graph expert to multiple routes. None turns low-rank teacher compression into an exact competence theorem. Their saved conclusions are reused.

TIES/DARE are conceptual comparison families here, not mandatory unqualified ports on unrelated random-start GNNs. If a later study uses teachers fine-tuned from one genuine common GNN base, their native task-vector operations become relevant controls. A freely chosen arithmetic mean is not retroactively the shared pretrained base required by those setups.

## What capacity permits

At a single linear map, BatchEnsemble diagonal modulation preserves every defined2×2 cross-ratio:

`(W_m[i,j] W_m[k,l])/(W_m[i,l] W_m[k,j])`

equals the shared W ratio when factors and denominator entries are nonzero. Generic independent matrices need not satisfy this. Hidden permutations and valid gauges must be considered, so this is a weight-family restriction, not a theorem that all deep functions differ or that a particular dataset needs an ensemble.

More generally, an exact centered four-model decomposition into a mean plus three unrestricted dense contrasts costs approximately four models. Calling the three contrasts a low-rank *member-axis* basis does not create compression. To reduce storage, impose low rank in input/output channel dimensions or share actual features/operators; either may discard useful functions.

For one map `W_m=W0+A_m B_m^T`, exact reconstruction requires `rank(W_m−W0)≤r`. Arbitrary full-rank differences do not satisfy that at r8. Layerwise small Frobenius or activation error also does not prove whole-GNN fidelity, because later inputs, nonlinear gates, normalization and attention change. A head-only common representation cannot recover teacher distinctions that have been discarded before that head.

The strongest exact construction is a block-diagonal stack of the four native learners. It preserves the bank but pays four models. Its role is an upper reference and a packing sanity check, not a compressed method. A shared bank with private corrections still executes four member states and dynamic graph trajectories after they diverge; one shared matrix in storage is not one encoder pass at serving.

## Concrete retained method

### 1. Acquire competence independently

Representative screen: full WikiCS, the existing pinned Polynormer-r512/head1/local7/global2 architecture, split0, TRAIN mask and author VALID union, seeds17/29/43. Each block has four genuine native learners with seeds `base+1009*m`, separate Adam and1100 ordinary100-local+1000-global epochs, unchanged features/support. No teachers start as copies of one warm model. All teacher work is paid.

For the compression donor bank use the prospectively fixed **terminal global1100 states**, ensuring one compatible forward graph. Keep the independently selected author-native bank from those same fits as an additional stronger quality reference, even if selected stages differ. This donor choice is fixed before outcomes; do not drop or replace a teacher to make it easier to compress. There is no warm-function equivalence claim for the low-rank bank.

### 2. Align only valid native symmetries

Teacher0 is the fixed coordinate reference. Use one prospectively specified global hidden-channel permutation per teacher, estimated from dropout-off TRAIN hidden trajectories over the same public graph, jointly across the interfaces that share a channel system. Propagate it through all dependent matrices, attention vectors, normalization affine parameters, products, gates, heads and cumulative/residual paths. Class labels and node IDs are never permuted.

For Polynormer the cumulative local sum and coordinate products prevent free independent rotations at every layer. A candidate permutation must first pass an actual native full-logit/gradient copy gate. If a correct native symmetry map cannot be implemented, stop that alignment claim. Procrustes or unrestricted rotations may be explored as approximate merging in another protocol, never called exact transport here.

### 3. Fit shared large maps and retain private internal corrections

All unique large native dense/attention-projection tensors receive a shared trainable base and member-private additive rank8 corrections. Preserve tied parameter aliases. Copy small member-specific attention vectors, all native normalization affine values, betas, biases and complete native output heads privately. Preserve native nonlinearities and graph operations. Every member carries its own full hidden trajectory; the shared body is learned throughout continuation.

For a particular aligned map, let teacher input rows on TRAIN be X_m, weight orientation be output×input, and `C_m=X_m^T X_m/N_T`. Use one fixed trace-scaled ridge `ρ=10^-3*trace(mean_m C_m)/d_in`; zero/nonfinite trace stops rather than triggers a redraw. Let Wbar be the arithmetic weight mean. The shared base is

`W0=(sum_m W_m C_m + Mρ Wbar)(sum_m C_m + Mρ I)^−1`.

This is a regularized teacher-input linear reconstruction fit, with direct RegMean/GNNMERGE ancestry. For each E_m=W_m−W0 and S_m=C_m+ρI, take the rank8 truncated SVD of `E_m S_m^(1/2)`, then multiply back by `S_m^(−1/2)`. If `U_r diag(σ_r) V_r^T` is that truncation, initialize

`A_m=U_r diag(sqrt σ_r)`, `B_m=S_m^(−1/2) V_r diag(sqrt σ_r)`.

Then `A_m B_m^T` is the best rank8 reconstruction under the stated positive-definite local input metric. That local statement supplies no nonlinear-GNN guarantee. The plain-SVD control uses the same aligned teachers, arithmetic W0 and unweighted residual SVD, with the same private architecture and rank.

Rank8 applies to every admitted large map; there is no rank/layer/ridge-strength sweep. Small maps and heads are copied privately by a shape rule fixed before outcomes. The exact parameter inventory must be frozen before science. Require total served parameter storage at most1.5 times one native model; failure closes this operating point rather than silently expanding rank or moving blocks private.

### 4. Gate actual function retention before continuation

Full-rank factor reconstruction first checks native source/alias/operator fidelity. Then measure each truncated bank member against its corresponding teacher on fixed TRAIN nodes, including logits, KL, top1 fidelity and margin violations. An exact finite-point certificate is: if `||z_student−z_teacher||∞ < teacher_top1_gap/2`, the teacher argmax is retained at that point. It does not certify correctness or unseen nodes.

Prospective admission for the candidate: at least99% TRAIN top1 fidelity and mean teacher-to-student KL at most.01 per member. Preserve every member and every block. Failed admission closes the claim of a competence-preserving initialization at this fixed capacity; no VALID-based reconstruction repair or extra rank is allowed. The control's failed initial fidelity is reported and does not by itself establish later quality superiority.

### 5. Continue all shared/private weights jointly

For100 fixed global-stage epochs, train the shared bank with the mean over members of `CE(y,p_m)+KL(p_teacher_m || p_m)` on TRAIN only, temperature1 and unit coefficients. Teachers are fixed; no teacher prediction targets from VALID/TEST enter this loss. Adam and dropout remain the existing native recipe, with independent member streams. All shared bases, private factors and small native parameters learn. No inference adaptation or learned router is added.

The teacher-specific KL preserves member association as in Hydra; it does not guarantee teacher accuracy. It cannot repair missing architecture capacity automatically. Record complete member/pooled metrics, teacher fidelity and inclusive cost. Selection is first strict maximum pooled VALID accuracy over100 post epochs. The true-independent continuation uses separate own losses/optimizers and individual selectors; that selector difference is disclosed.

## Strong controls and falsifiers

| Control | What it decides |
|---|---|
| Frozen terminal four-teacher bank, plus original independently selected native4 | Target competence/complementarity and strongest uncompressed quality/cost reference; no favorable teacher substitution. |
| Same all-depth rank8 architecture with plain weight-SVD initialization | Whether the graph-conditioned input metric/alignment recipe helps beyond generic low-rank pretrained compression. |
| Hydra-style shared-body, persistent teacher-matched full heads | Whether internal private operators are needed beyond known member-preserving distillation. Allocate its head width from the same prospective parameter budget; retain nonlinear heads and TRAIN teacher identity targets. |
| Strong complete native single, initialized by the aligned merged base and taught the mean teacher probabilities | Whether ensemble information can be consolidated into one competent nonlinear GNN. All native weights learn; no artificially weak frozen encoder. |
| Parameter-budget capable single | Use the smallest allowed hidden width whose native parameter count reaches the compressed bank budget, fixed analytically before outcomes. Give it the same public inputs and teacher-mean targets, adequate fixed supervised/continuation budgets, and charge all preparation. A weak unqualified fit cannot establish ensemble necessity. |
| Four true independently optimized native continuations | Each starts from its own independently learned terminal teacher, uses its own CE plus own teacher anchor, separate Adam/dropout/selector, and complete trajectory. No common-warm substitution or pooled-gradient coupling. Also keep the unchanged author-native independent4 reference. |

This is a fixed method/control study, not a hyperparameter grid. The exact Hydra/single architecture and source/operator gates must be bound before any implementation launch; they are not supplied as qualified runners by this idea note.

Retain the compression hypothesis only if the fixed storage gate and initial fidelity pass, final equal-seed mean member VALID accuracy is no lower than the donor bank, every member loses at most.5 percentage points, and pooled accuracy loses at most.5 points against both terminal and independently selected native4 references. Claim an improvement only if it exceeds both plain-SVD and Hydra controls at the paid operating point. A capable single matching closes demonstrated ensemble necessity. NLL, reconstruction error or disagreement alone is not success. An independent continuation dominates unless the compressed bank offers a measured acceptable storage/serving tradeoff. Three seeds on one split supply descriptive replication, not graph-level significance.

Freeze all decisions before one separate TEST evaluation of every retained arm/seed. Charge12 independent teacher fits, graph forwards/statistics/alignment/SVD, all teacher targets, student repair/continuation, validation opportunities, checkpoint/state storage, transient teacher+student peaks, and every member's serving work. Report training cost separately from deployment memory/latency and amortization; no training-efficiency claim follows from storage compression.

## Contrast with current growth source

Current growth v2 begins with one common warm NCN and adds4096 private parameters at one native preaggregation site. Zero incoming weights and nonzero outgoing bases preserve that baseline, while graph-error filters initialize output subspaces. The top8-partition control addresses initial subspace coverage. Its independent control consists of common-warm native copies and has an explicit joint-selector limitation.

This compression pipeline begins with four separately learned functions, aligns/reconstructs large internal maps across the entire nonlinear body, and tries to retain those acquired functions under a fixed storage budget. Its low-rank initialization is approximate and is not the current zero-growth copy gate. The independent reference is genuine independent pretraining. Current growth tests acquisition of new nonlinear features from one warm predictor; compression tests whether previously learned diversity can survive sharing. A positive result in either does not validate the other.

## What could be distinctive

The possible useful contribution is a concrete graph-operator compression/continuation protocol that retains individually acquired native competence with a measured shared-storage benefit. Its graph-dependent reconstruction metric, whole-body private corrections and per-teacher continuation targets must survive the matched controls. Common base estimation, SVD, graph alignment, low-rank adaptation and member distillation are attributed prior components. Existing EMR/Twin-style private-residual merging families were located in the consulted references but not method-read here; complete recipe overlap remains unresolved. This note establishes no originality or global absence claim.

If initial teacher fidelity fails at the fixed budget, the honest conclusion is that this compression family did not represent the teachers adequately. Stop this initialization proposal. A later private nonlinear message block, larger shared dictionary or partial unsharing would test capacity and must be declared as a different architecture study before outcomes.
