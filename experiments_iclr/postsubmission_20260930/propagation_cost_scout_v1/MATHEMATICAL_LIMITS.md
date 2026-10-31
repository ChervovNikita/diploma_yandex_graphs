# Linear identities and their limits

This is hand-constructed mathematical analysis, not an implementation, dataset experiment, model replay, or latency measurement. `ARITHMETIC_CHECK.json` records rational arithmetic for the examples below.

## 1. Center the states that enter the operator

For M members, let `Z_m = LayerNorm(H_m)` be the current input to a fixed linear incoming-neighbor operator P. Define `Zbar = (1/M) sum_m Z_m` and `Delta_m = Z_m - Zbar`. Then

```
sum_m Delta_m = 0
P Z_m = P Zbar + P Delta_m.
```

A LayerNorm of a centered state is not generally centered. Compute the mean *after* memberwise LayerNorm. Shared LayerNorm affine parameters do not change this requirement.

## 2. A shared mask preserves the linear member mean

Define `G_m = P Zbar + Q Delta_m`, where Q is the same linear operator for every member. For each realized Q,

```
mean_m G_m = P Zbar + Q(mean_m Delta_m) = P Zbar.
e_m = G_m - P Z_m = (Q-P) Delta_m
sum_m e_m = 0.
```

If `E[Q | Z_1,...,Z_M] = P`, then `E[G_m | current states] = P Z_m`. Conditioning is essential: each layer samples independently of the states entering that layer. This does not imply unbiased states relative to a separately run exact network, because those states have already passed through previous approximate nonlinear layers.

The finite-precision checks required in a future implementation must record normalized `||sum_m Delta_m||` and `||mean_m G_m-P Zbar||` per layer, mask and seed. A suitable diagnostic scale is `max(1, sum_m ||Z_m||)`; record actual residuals and dtype rather than assume symbolic zero. Check zero-degree rows, degrees below four, duplicate entries, original self edges and all four masks. No tolerance is a predictive guarantee.

## 3. Unbiased four-color weighting

Partition the original edge entries with their original normalized weights into C_0,...,C_3, such that `sum_c C_c=P`. Uniformly choose c and set `Q=4 C_c`. Then

```
E[Q | current states] = (1/4) sum_c 4 C_c = P.
```

Receiver-balanced groups can contain different edge counts; uniform color selection still gives every edge inclusion probability 1/4. Do not replace original weights by normalization on the sampled degree.

Example: a degree-two mean row is `P=[1/2,1/2]`, with one edge in each of two colors and the other two colors empty. Corrected Q rows are `[2,0]`, `[0,2]`, `[0,0]`, `[0,0]`; their average is P. Sample-degree renormalization would instead give `[1,0]`, `[0,1]`, zero, zero; its average is `[1/4,1/4]`, which is biased. A degree-zero row remains zero. Fixed GCN weights may be retained similarly; sampled degree or attention-softmax normalization does not satisfy this derivation.

For the cost-matched naive control, `Q_naive=2(C_c+C_(c+1 mod 4))`. Each edge appears in two of the four choices, so `E[Q_naive]=P`. This control uses half the edges, while the candidate samples a quarter plus an exact common pass.

## 4. Limited same-mask linear MSE comparison

Let A=Q-P. For the naive use of the *same Q*, the errors are `A Z_m`. The identity `sum_m Delta_m=0` gives

```
(1/M) sum_m ||A Z_m||_F^2
  = ||A Zbar||_F^2 + (1/M) sum_m ||A Delta_m||_F^2.
```

The candidate's average linear aggregation MSE removes the first nonnegative term, for every realized shared Q. This is a standard orthogonal mean/deviation decomposition. It is not an improvement theorem against the proposed **half-edge**, cost-matched naive control, since that control uses a different operator. It is not a variance or quality guarantee after nonlinearities.

For a fixed realized Q,

```
(1/M) sum_m ||e_m||_F^2
  <= ||Q-P||_op^2 (1/M) sum_m ||Delta_m||_F^2.
```

This bound makes the tension explicit: a small contrast can make approximation cheap, but a collapsed contrast may provide no useful ensemble gain. Useful diversity must be established with deployed pooled predictions.

## 5. Complementary member masks can increase pooled error

Take two scalar-feature neighbors, `P=[1/2,1/2]`, residuals `Delta_1=[1,-1]` and `Delta_2=-Delta_1`. Their exact propagated residuals are both zero.

Use complementary edge samplers `Q_1=[1,0]`, `Q_2=[0,1]`. Then

```
Q_1 Delta_1 = 1
Q_2 Delta_2 = 1
mean approximate residual = 1, exact mean = 0.
```

Both samplers individually have the appropriate marginal distribution if a fair common coin swaps their assignment. The errors are perfectly positively correlated after applying the opposite residuals. Complementarity alone is therefore insufficient. Sharing `Q=[1,0]` instead gives residuals `1,-1`, with zero pooled residual for each realization.

This counterexample rejects the earlier complementary-member-edge idea. It is not retained as a second hypothesis.

## 6. Member-dependent Jacobians defeat first-order cancellation

Write the nonlinear update as `F_m(Z_m, G_m)`; the weights can be shared while its Jacobian with respect to G differs because its inputs differ. To first order, the change in the member mean is

```
(1/M) sum_m J_m e_m.
```

`sum_m e_m=0` does not imply that expression is zero. For two members it is `(J_1-J_2)e_1/2`.

A smooth scalar example uses the *same* nonlinear function `F(g)=g^2`, exact inputs `g_1=1`, `g_2=-1`, and errors `e_1=epsilon`, `e_2=-epsilon`. The exact mean output is 1. The approximate mean is `(1+epsilon)^2`, changing it by `2 epsilon + epsilon^2`. Jacobians `2` and `-2` produce a nonzero first-order term. At `epsilon=1/10`, the difference is exactly `21/100`.

Thus neither shared weights nor zero-sum aggregation error establish first-order cancellation after nonlinear updates. Final raw logits, pooled probabilities, cross-entropy, gradients and AdamW trajectories inherit no exactness guarantee. No zero-variance limit is implied: current member disagreement can persist even when weights are frozen, unlike a successfully refreshed temporal history.

## 7. The arithmetic budget is not the latency budget

Exact packed propagation has `M E D` edge-channel work. Candidate work is `E D + M E_c D` for the selected group c; its expected value is `(1+M/4) E D`. At M=4 this is `2 E D`, half of `4 E D`.

The complete model still keeps `MND` states and runs M dense nonlinear blocks. It adds an exact shared sparse pass, two sparse launches instead of one packed launch, mean/subtraction/broadcast kernels and their backward operations, plus adjacency storage/preparation. Sparse arithmetic, observed sparse time, end-to-end inference and total training wall time must be reported separately. No twofold speed or memory claim follows from this equation.
