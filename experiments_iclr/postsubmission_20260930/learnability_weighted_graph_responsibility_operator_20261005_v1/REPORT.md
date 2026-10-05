# One finite learnability-weighted graph responsibility operator

5 October 2026. **Disposition: coherent as one disabled source/math candidate.** The finite assignment map preserves its constraints in real arithmetic, and its private/outer derivative ownership is explicit. Native Amazon partition, sparse higher-order derivatives, useful nonuniform responsibilities and predictive benefit remain unqualified. No Torch import, numerical/source execution, dataset/payload access, remote-host access or fit was performed for this packet. Existing private-transfer work is unchanged.

## Intended error intervention

The complete Amazon diagnosis found slightly weaker shared members and less probability-pooling benefit; the graph recurrence did not identify a causal graph defect. This candidate gives every member ordinary all-target CE, then allocates **extra** correct-versus-competitor pressure to members whose finite private own-CE step reduces that pressure. A fixed graph affinity smooths responsibility identity within a class pair. The shared core receives query credit through the actual finite private update and this assignment map. Serving is always the arithmetic mean of member probabilities.

This is a falsifiable training operation. It does not infer missing representation information from observed error agreement. A response measured on innerS can overfit that set, reflect a gradient-scale accident or sacrifice another competitor. Assignment variation is a mechanism diagnostic, not a utility certificate.

## 1. Fixed state and label ownership

Let shared trainable parameters be theta and the **complete** private block of member m be phi_m. The deterministic pure forward z_m(theta,phi_m) exposes every feature/context/skip that the actual private branch receives. It has no hidden parameter update, dropout advancement, mutable buffer, optimizer moment or stateful cache. This small source accepts dictionaries of tensors and a model callback; it binds no native Amazon architecture.

Use only permitted TRAIN labels. For an episode, innerS and outerR are nonempty and disjoint; every class occurs in innerS. Every i in innerS has an item t=(i,k) for **every** k != y_i. Pair p={y_i,k} is unordered. Its block contains the nodes of both classes, with the other class as competitor. No support cell or label direction is dropped. Item counts over all pairs total B=|S|(C-1).

The public graph affinity K on S is fixed, symmetric, nonnegative and has zero diagonal, constructed independently of labels from direct public graph adjacency. In pair p, restrict K to its nodes, set W_p=K_p/(1+max_i sum_j K_p[i,j]) and L_p=diag(W_p 1)-W_p. This retains same-label **and cross-label** edges. It gives ||L_p||_2 <= 2. Labels define permitted TRAIN task blocks; there is no homophily-only graph filter or use of current outerR/heldA labels in Q.

## 2. Preliminary response and cost normalization

At fixed theta and original phi_m, take exactly one virtual private own-CE SGD step:

    own_m = mean_{i in S} CE(z_m(i), y_i)
    phi_hat_m = phi_m - eta_probe * partial_phi_m own_m
    b_t,m(theta,phi) = softplus(z_m(i,k)-z_m(i,y_i))
    c_t,m = b_t,m(theta,phi_hat_m) - b_t,m(theta,phi_m).

Negative c means this particular private step reduces that correction loss. This is not a response to labels in R or A. To first order, c_t,m = -eta_probe <grad_phi b_t,m, grad_phi own_m> + O(eta_probe^2): it has ordinary gradient-alignment ancestry. The exact source uses the finite difference in losses, not this expansion.

For each item center across members, then use one smooth scale per pair:

    d_t,m = c_t,m - (1/M) sum_l c_t,l
    s_p = sqrt(epsilon^2 + (1/(n_p M)) sum_{t in p,m} d_t,m^2)
    a_t,m = d_t,m / s_p.

All these operations participate in the live outer chain. Centering discards common member learning; the allocation concerns relative response. Smooth RMS is not invariant to arbitrary scale when responses are below epsilon. If eta_probe makes costs tiny compared with epsilon, Q can remain nearly uniform. If all members have equal responses, Q is exactly uniform and the operation reduces to uniform extra margins. The declared defaults do not establish that the actual private factors produce measurable responses.

## 3. EXACT finite assignment map, approximate optimization

For each pair consider the strictly convex entropy/graph objective

    F_p(Q) = <a_p,Q> + tau sum_{t,m} Q_t,m log(M Q_t,m)
                         + (gamma/2) tr(Q^T L_p Q),
    sum_m Q_t,m = 1; sum_t Q_t,m = n_p/M; Q_t,m > 0.

The code does **not** return or differentiate the exact minimizer. It returns this one fixed differentiated algorithm. Begin Q^0_t,m=1/M and perform exactly T=8 iterations:

    G = a + tau[log(MQ)+1] + gamma LQ
    H = G - rowmean(G) - colmean(G) + grandmean(G)
    r = delta_r * logsumexp_{t,m}[(H_t,m/Q_t,m)/delta_r]
    v = delta_v * logsumexp_{t,m}[(1/Q_t,m)/delta_v]
    h = 1 / [eta_q^{-1} + 2r + 2 tau v + 2 gamma]
    Q <- Q - h H.

H has zero row and column sums. Thus each iteration preserves exact row sums and pair member masses in real arithmetic. Also r >= max H/Q >= 0 and h r < 1/2, so Q_new > Q/2 > 0. Every support cell stays positive. Along that segment the entropy curvature is bounded by 2 tau v and graph curvature by 2 gamma. Since H is an orthogonal tangent projection, G:H=||H||_F^2; this h gives F(Q_new) <= F(Q)-h||H||_F^2/2 when H != 0. This establishes feasible descent of the declared objective, **not** convergence to its optimum after eight steps.

No convergence stop, implicit optimizer derivative, warm start, clamp, marginal repair or tolerance-dependent fallback occurs. Floating arithmetic still requires residual/positivity qualification; a failed check cannot be hidden by changing the map. The smooth upper bounds are conservative and can reduce movement, especially for small Q. Actual entropy, cost scale and deviation from uniform must be retained, not inferred from the number of iterations.

## 4. Main response: private PARTIAL, outer TOTAL derivative

Compute Q(theta, all original phi) by the preliminary response and finite algorithm above. For each member define an independent-input scalar:

    ell_m(theta, phi_m; Q) = own_m(theta,phi_m)
                     + lambda * M/B * sum_t Q_t,m b_t,m(theta,phi_m)
    phi'_m = phi_m - eta_private * partial_phi_m ell_m(theta,phi_m;Q).

The M/B factor makes uniform Q exactly lambda times the ordinary mean competitor loss for each member. All members still receive their complete own CE. Balance concerns additional supervision mass, not competence or information.

**Q is independent while taking this private partial.** The implementation uses `torch.func.grad(_main_loss, argnums=1)` with Q as a separate argument, then calls the resulting gradient map with Q(theta,all phi). The returned partial consequently retains its dependence on Q for later outer differentiation. Detaching live Q would incorrectly discard that outer path; differentiating a scalar whose prebuilt Q depends on phi would incorrectly add the Q-to-phi chain to the private partial.

On R, compute

    J(theta,phi') = (1-rho) mean_m CE_R(z_m(theta,phi'_m))
                         + rho CE_R(mean_m softmax(z_m(theta,phi'_m))).
    theta+ = theta - eta_core * d_theta J(theta,phi'(theta)).

This is the total theta derivative through the preliminary private step, response-cost centering/RMS, all assignment iterations, independent-Q main partial and private response, plus the direct query path. Original phi are independent persistent start arguments: no derivative through earlier episodes/history is included. R labels enter J but never Q. An episode's S/R disjointness does not erase previous label exposure or create independent same-graph tasks.

Writing g_m=partial_phi ell_m with theta, phi and Q independent gives the exact engineering chain to check:

    D_theta phi'_m = -eta_private [partial_theta g_m + (partial_Q g_m) D_theta Q]
    D_theta J = partial_theta J + sum_m (partial_phi'_m J) D_theta phi'_m.

`stop_q` removes only the displayed Q-chain term, keeping the same forward g_m and the direct/mixed theta paths at an identical start. Differentiating through Q inside the private partial would be a different operator.

After updating theta, recompute the **entire** preliminary response, costs, Q and main response at theta+ from the original phi. Commit that recomputed phi and theta+. Discard both preliminary probes and theta-time virtual private states. The source detaches committed state between episodes. SGD has no Adam moments, decay, clipping or native optimizer equivalence claim.

## 5. One declared coefficient set and controls

G0 is fixed in the source: eta_probe=eta_private=0.01; eta_core=0.001; lambda=0.1; rho=0.5; epsilon=0.001; tau=gamma=1; T=8; eta_q=1; delta_r=delta_v=0.01. These are explicit finite-response gate conventions, without tuning or an assertion of suitable native scale. Changing them creates a different map and must precede label outcomes.

Required matched controls are implemented:

- `uniform`: identical probe/solver forward work, then Q=1/M; ordinary extra competitor margins.
- `graph_free`: gamma=0, same finite map/balance. This removes spatial affinity; column balance still couples items, so it is not a fully separable softmax.
- `margins`: use current b in place of response c, with the same centering/RMS and solver. The preliminary step is still paid.
- `stop_q`: exactly the same forward Q at an identical state, but detach Q only before the main partial. The private update/query values match the live arm at that state; its core derivative omits the Q chain. Later trajectories need not match.

M=1 gives Q=1 and the corresponding single-branch finite own/extra-margin response operation; a capable single should receive matched operation controls. More strongly, the complete M-branch bank and fixed probability-mean readout can be represented as **one nonlinear multibranch computational graph with the same function class**. Allowing that graph the same parameter grouping and weighting can reproduce this operator exactly. An ordinary one-head single is not automatically that control. The operation supplies no ensemble-necessity or novelty claim.

## 6. Minimal finite TRAIN-only response gate contract

A separate bounded protocol may bind one prospectively fixed stratified TRAIN panel A, B=TRAIN\A, fixed S/R subsets of B, a fresh B-only acquisition and H=16 matched episodes. **All A labels must be absent from pretraining, acquisition, probe/graph choices, selectors and all episode losses.** Existing all-TRAIN warm states do not satisfy this boundary. A is evaluation-only, never the episode's outerR. VALID/TEST remain untouched. Acquisition horizon/model/partition must be fixed and source-qualified before this gate; this operator cannot certify them.

Compare complete served probability-pool CE and accuracy on A at the fixed horizon, with complete per-member own CE/accuracy and all declared class-pair correction losses/support. Require complete member competence under a prospectively stated criterion; lower assigned TRAIN loss, disagreement or a selected best member is insufficient. Retain every supported task and failed cell; no A-based coefficient/affinity/horizon selection or new arm expansion. The separate protocol owns numerical acceptance criteria.

The operator reports per-pair both label counts, centered raw response RMS, smooth scale, epsilon/scale, normalized cost RMS, assignment relative RMS and maximum deviation from uniform, entropy, minimum Q and both marginal residuals, plus every member's inner probe CE change. These diagnose a vacuous or unsupported response without certifying transfer usefulness.

## 7. Source/resource limits and nearest ancestry

`response_operator.py` is release-disabled and model-agnostic. Its dense pair affinities are for small math qualification: affinity storage is sum_p n_p^2; item storage is |S|(C-1)M, and the differentiated eight-step graph map retains additional intermediate state. It is not a qualified large-Amazon sparse implementation.

Without recomputation/checkpoint optimization the source makes **9M model-callback forward calls per episode**: 5M in the virtual/query core calculation (before, probe-gradient, after, main-gradient, query), and 4M in the post-core recomputation. It constructs 4M private first derivatives across the two responses, differentiates the virtual ones plus assignments in the outer core derivative, and performs two complete assignment maps. Controls pay the same declared forward construction but generally different backward work; no equal wall-time claim follows. Actual graph/member activation and higher-order costs require separate native qualification.

Saved graph-balanced responsibility/MCL analysis already owns entropy/balance/graph responsibility ancestry. Saved MetaReg/MLDG owns train-only virtual updates and fixed deployment; saved DERTS owns adaptation-dependent gradient-based task weighting. The nearest-prior agent additionally reports primary Ren learning-to-reweight and Meta-Weight-Net method scopes; those independent receipts belong to that packet, not to a new primary read here. The finite margin response, entropy-balanced pairwise graph map and shared-core credit are one attributed composition requiring its own gate. No new primary paper was read or retrieved by this source packet.

Static syntax/structure checks only are recorded in `STATIC_CHECK.json`; they do not establish Torch, mixed derivatives, finite assignment invariants in floating arithmetic, native sparse AD or scientific utility. `RELEASE.json` and `MANIFEST.json` bind the disabled source and its exact specification.
