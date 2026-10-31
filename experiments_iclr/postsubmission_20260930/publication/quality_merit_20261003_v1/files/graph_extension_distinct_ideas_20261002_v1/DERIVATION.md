# Vector residual maintenance with learned channel mixing

This is a local mathematical derivation for a prospective algorithm. It is not a theorem attributed to arXiv:2609.32929v1, a numerical qualification, or a novelty certificate. The source explicitly leaves learned feature mixing open in saved paragraph block 19. Its scalar invariants and contraction arguments motivate the construction. The earlier saved IGNN/DEQ assessment already establishes learned equilibrium models, private recurrent factors, and residual bounds.

## Declared model and norm

Use an undirected, unweighted graph with permanent self loops, adjacency A, degrees d_i >= 1, and row mean P = D^{-1} A. This is the beta=1 case of the fresh source's normalization. Do not import this argument into attention, directed graphs, changing weights, batch normalization, or arbitrary modern graph transformers.

For each frozen member m define an N by h state Z_m, fixed forcing X_m, a frozen h by h matrix V_m, and the fixed point

    Z_m = sigma(alpha X_m + (1-alpha) P Z_m V_m).

Sigma is coordinatewise, 1-Lipschitz, and bounded in absolute value by B. Hard tanh supplies a concrete instance. V_m may be the existing two-sided factor map diag(r_m) W diag(s_m); this parameterization and its diagonal-adapter equivalence are established prior. The dynamic operation also applies to independently learned V_m.

For row vectors, ||v V_m||_infinity <= ||v||_infinity ||V_m||_1, where ||V_m||_1 is the maximum absolute column sum. Require a verified bound

    rho_m = (1-alpha) ||V_m||_1 <= rho < 1.

This is a sufficient restriction, not the most general IGNN well-posedness condition. A source implementation would need to establish that its fitted predictor remains competent under it. A global factor bound can use ||V_m||_1 <= ||r_m||_infinity ||W||_1 ||s_m||_infinity. A chosen norm constraint changes the training parameterization and must be declared; it is not an innocuous post-fit renormalization.

P is nonexpansive in the maximum element norm. The member map is therefore rho_m-contracting. It has a unique fixed point. Private factors or forcing make different maps; initial guesses or solver noise alone do not make new members.

## State, push, and error certificate

Maintain separate arrays for every member:

    Y_m = alpha X_m + (1-alpha) P Z_m V_m,
    R_m = sigma(Y_m) - Z_m.

A push of node i for member m sets delta = R_m[i], changes Z_m[i] by delta, computes the h-vector t = delta V_m, adds (1-alpha) P[j,i] t to Y_m[j] for every neighbor j including i, then recomputes affected residual rows. This preserves both invariants in real arithmetic. The dense transform costs O(h^2); visiting the neighbor rows costs O(d_i h). A cache of Z_m V_m trades another N h private array for some endpoint transforms.

Let e_m be the maximum element norm of R_m. The standard contraction inequality gives

    ||Z_m - Z_m_star||_max <= e_m / (1-rho_m).

Thus stopping when every residual row satisfies ||R_m[i]||_infinity <= (1-rho_m) epsilon certifies maximum state error at most epsilon. This certificate concerns the member's frozen fixed-point model. It neither proves correct labels nor makes the members posterior samples.

If a frozen head is L_m-Lipschitz from the state maximum norm to logit maximum norm, its logit error is at most L_m epsilon. A linear head permits a directly computed matrix norm; a nonlinear head requires the actual bound. Uniform raw-logit pooling has error at most mean_m L_m epsilon. Softmax CE has l1 gradient norm at most 2, so per-example CE distortion is at most twice the pooled logit maximum error. An approximate top-class margin greater than twice that error certifies agreement with the exact converged predictor. These are numerical approximation guarantees, not statistical uncertainty guarantees.

## Constant-size graph update repair

Insert one non-self undirected edge (u,v), giving d'_u=d_u+1 and d'_v=d_v+1. Only rows u and v of row-normalized P change. Keep every Z row fixed and set

    Y'_m[u] = alpha X_m[u]
               + (d_u/d'_u) (Y_m[u] - alpha X_m[u])
               + (1-alpha)/d'_u Z_m[v] V_m.

Apply the analogous formula at v and recompute R at those two endpoints. A deletion uses a minus sign and d'=d-1. Permanent self loops ensure d' >= 1. Every other Y and R row remains valid. Endpoint repair costs O(M h^2), or O(M h) if a valid transformed-state cache is maintained. The repair is exact for the declared graph/model in real arithmetic; numerical roundoff needs its own residual budget.

For the source's more general beta normalization, degree-rescaling Z at the endpoints maintains D^{beta-1} Z and extends the scalar invariant. The proposed first specification fixes beta=1, so no extra general-beta claim is needed.

## Potential and what it can pay for

For one member define

    Phi_m = sum_i d_i ||R_m[i]||_infinity / (1-rho_m).

After a push, the pushed row's residual norm is at most rho_m P[i,i] times its former norm. Other affected rows increase by at most rho_m P[j,i] times that norm. For an undirected graph, sum_j d_j P[j,i] = d_i. Consequently a push reduces Phi_m by at least d_i ||R_m[i]||_infinity. A threshold-crossing push therefore reduces it by more than d_i (1-rho_m) epsilon.

Since d_i >= 1, its O(h^2+d_i h) work is bounded conservatively by O(d_i(h^2+h)). This accounts for dense mixing; the scalar source's O(1/epsilon) cannot be quoted unchanged.

Assume the state is bounded by B at the cleaned pre-update state, as hard tanh gives when pushes start from a bounded initialization. Insertion changes each endpoint's Y by at most 2 rho_m B / d'_i. Its weighted residual potential increase is at most epsilon + 2 rho_m B/(1-rho_m). The two endpoints therefore inject at most 2 epsilon + 4 rho_m B/(1-rho_m) potential. Deletion has the same type of bound. Summing potential decreases and event injections gives a conservative per-member amortized cleanup cost of order

    (h^2+h) [1/(1-rho_m) + B/((1-rho_m)^2 epsilon)],

plus endpoint repair and the initial potential. Constants, floating-point safeguards and data structure costs remain unqualified. Near-unit contraction can destroy the useful bound and measured latency. Summing over M members gives an M factor. A union worklist does not remove it.

This argument is only a candidate proof outline. A later theory/source review should formalize the deletion case, bounded-state initialization, graph event model, termination, finite precision and head bounds. It supplies a specific proposed property to test before GPU work; it is not a completed publication theorem.

## What shared computation buys

One canonical adjacency/degree store can support all member pushes. A tagged union queue may fetch a common row once and apply only the active members' numerical deltas. Active residuals may be stacked for the shared W transform, with their private factors before and after it. These are established batching and incremental-serving operations from the retained Round02 assessment.

All learned member Z/Y/R arrays remain private. Numerical edge contributions still scale with active members, and common W does not make delta W identical across members. A competent independent vector-push baseline must use the same batching, graph index and precision. Any claim of shared-W benefit must survive that baseline. The proposed substantive delta is certified dynamic maintenance of learned channel-coupled graph equilibria, not a new queue or a state-sharing shortcut.

## Node forcing reveal

When a previously masked feature row becomes observable, compute its declared member forcing X_i exactly, change Y_i by alpha DeltaX_i, and recompute R_i. Other rows retain the invariant until residual pushes propagate the new state. The forcing encoder and injected potential are charged. This extends the stated algorithm beyond edge-only updates, and requires a separate feature-reveal/event contract; future features must not enter the existing prefix.
