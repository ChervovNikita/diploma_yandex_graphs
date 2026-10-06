# Exact factor gauge and optimizer-conditioned private learnability

6 October2026. **An exact real-arithmetic gauge exists in the pinned adapter. Euclidean-SGD response and first-order utility can change along it. Current trained-state sensitivity is unmeasured.** No source import/execution/edit, numerical/data/result/checkpoint/logit payload, SSH, training or scoring occurred. The fixed pilot is unchanged; literature conclusions were reused with zero new paper reads.

## Exact source, bias and nonlinear path

BoundaryProjector.forward (adapter line37) is

    A(x;r,s,B,W) = diag(s) W diag(r) x + B.

F.linear is called WITHOUT its bias argument. The author's native bias is copied into each private B row (lines26–29); even a bias-free author linear gets a trainable zero-initialized B. The native stem/local/global boundary owners become Identity (lines76–81). Thus there is no extra shared boundary bias inside the factor-scaled linear operation. All other core biases/weights remain unchanged.

For any nonzero scalar kappa, independently for each owner/member, define

    r' = kappa*r,   s' = s/kappa,   B' = B,   theta' = theta.

Then A(x;r',s',B',W)=A(x;r,s,B,W) for EVERY input and shared W. B is not rescaled. Private R/S are unconstrained Parameters; the stem's Rademacher rule is initialization, not a continuing sign/magnitude constraint. This transformation changes the declared parameter/initialization state and is not authorized inside the current pilot.

Apply this within a stem projector or an active/dormant head. The stem output is exactly identical, so the subsequent attention, ReLU, LayerNorm, channel products/gates, cumulative local states and core biases receive identical inputs. Head outputs are identical too. No inter-layer ReLU scaling or LayerNorm equivariance is assumed. The admitted callback is eval-only (port lines45–46), so dropout is off. Both stages' complete logits and fixed probability mean are preserved in real arithmetic. This is a within-projector factor redundancy, unlike the previously unproved generic inter-layer hidden-scaling construction.

This is a valid universal family, not a classification of all possible gauges. More general diagonal input/output scalings would need a_j*b_i=1 on each active nonzero effective W entry; actual zero/support degeneracies were not inspected. No floating-point/serialized equality or numerical result is claimed.

## Pullback of the declared Euclidean probe

Let T be the block-diagonal transformation above over the complete private bank. For own mean-CE f and a softplus competitor margin b, pointwise function equality gives f(theta,T phi)=f(theta,phi) and b(theta,T phi)=b(theta,phi). At differentiable points, or a compatible fixed selected-AD branch,

    g' = partial_phi' f = T^(-T) g,
    h' = partial_phi' b = T^(-T) h.

Thus g_R'=g_R/kappa, g_S'=kappa*g_S, g_B'=g_B. A same-rate Euclidean private step in primed coordinates pulls back to

    phi' - eta*g' = T[phi - eta*P*g],
    P = T^(-1)T^(-T)
      = diag(kappa^(-2) I_R, kappa^2 I_S, I_B).

The actual operator uses eta_probe=.01 and ordinary dictionary SGD (operator lines57–58,148–151). Therefore

    c_FIN' = b(theta,phi - eta*P*g) - b(theta,phi),
    c_FO'  = -eta*h^T P*g.

For one transformed owner, write A_R=<h_R,g_R>, A_S=<h_S,g_S>, with the bias/other-coordinate contributions unchanged. Then

    c_FO(kappa) = -eta*[kappa^(-2) A_R + kappa^2 A_S + A_B + A_rest].

The gauge-tangent identities r^T g_R=s^T g_S and r^T h_R=s^T h_S do NOT imply A_R=A_S or invariance of this quadratic metric. Non-unit magnitude changes effective block step sizes. Kappa=-1 (and+1) is orthogonal and gives P=I; it is a useful covariance negative check. Inactive local-head gradients are zero in the admitted global stage, so a gauge of ONLY that dormant head cannot create response sensitivity.

## Bias-aware checkable example

Use a five-class head with one active shared weight/input coordinate: z_j=s*r*x+B_j, all other class logits equal their B_l, with x=1 and r=s=1. An artificial all-five-class S can have only the chosen target-y example's active input nonzero; other examples have x=0. For its competitor j, let q=softplus'(z_j-z_y)>0 and d=partial_(sr) mean_CE_S>0. Every B_l remains a genuine private coordinate and can have nonzero own-CE gradient.

Here A_R=A_S=q*d. Rescaling with kappa=2 changes the R/S part from2*q*d to(1/4+4)*q*d=17*q*d/4, while the full B/other contribution stays identical. Thus the complete first-order utility changes by -9*eta*q*d/4; no bias coordinate was discarded.

After the Euclidean probe, pulled-back r=1-eta*d/kappa^2 and s=1-eta*d*kappa^2. The new gap is

    u_kappa = 1 - eta*d*(kappa^(-2)+kappa^2) + eta^2*d^2
              + B_j-B_y - eta*(g_Bj-g_By).

The bias update is common to both gauges, so u_2-u_1=-9*eta*d/4. Softplus is strictly increasing, giving different finite responses as well. This is an algebraic projector example, not an executed/current full-GNN fixture. In the complete model, an active-head gauge leaves other private gradients and their simultaneous updates unchanged; the exact effective-head matrix difference below acts on their common post-update hidden input and can vanish in special states.

For general r,s,W the probed effective matrix difference is exactly

    W_eff(kappa)-W_eff(1)
      = -eta*[(kappa^(-2)-1) diag(s) W diag(g_R)
              +(kappa^2-1) diag(g_S) W diag(r)].

The eta^2 cross term is invariant and cancels; the bias difference is zero. This supplies a direct numerical identity to check without assuming a scalar network.

## Implications and scope

An invertible T preserves the unconstrained private function class at fixed theta, hence intrinsic representational capacity. It does not preserve reachability under a fixed Euclidean update/time budget. Finite response is meaningfully a property of the declared learner—coordinates, step, optimizer, state and support—not an optimizer-independent member-capacity certificate. Different gauges can create response/cost heterogeneity before any prediction difference; whether this matters at the actual common state remains a future measured question.

Cost centering removes common member offsets, not general gauge-dependent terms. Pair RMS has fixed epsilon=.001, so even a common positive rescaling is not generally canceled exactly; Q may change, but need not. Current-margin/uniform raw assignments can remain equal while their main Euclidean private steps still change. This concern is not unique to response allocation: ordinary SGD and fixed-Q main-private updates are coordinate dependent too. Coupled Q/shared credit can amplify or suppress the effect, and no predictive sign follows.

For a pure covariance check, transporting the original metric to M'=T*T^T gives phi'-eta*M'*g'=T(phi-eta*g) and -eta*h'^T*M'*g'=-eta*h^T*g. This is an algebraic reference, not an admitted optimizer change. No private/Adam history, warm acquisition, rate, source, arm or budget is changed here. Existing nonsmooth/coarse-FD findings remain; a large effective P step can cross branches even when eta is fixed.

TEST_PLAN.json is disabled and requires a separately authorized small synthetic check before any study implication. It supplies no current sensitivity result, extra fit or novelty certification.
