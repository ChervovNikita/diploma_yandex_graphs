# Does arithmetic variance tying give a weak common-bank control?

**Yes, for the question of a common route-mixture normalization distribution.**
It omits a nonnegative between-route-mean component. A small fixed-checkpoint
follow-up is worthwhile. The existing arithmetic control remains a valid
intervention on saved BN state, and its report already discloses that it is not
a pooled-population variance. This follow-up strengthens interpretation; it
creates no training recipe and does not replace the current nine-fit study.

## What the current evidence establishes

The completed `bn_bank_assignment_v1` evaluation held learned weights and the
selected checkpoints fixed for private-BN seeds101/202/303. It found own-bank
AUC better than arithmetic tying and better than the mean score of the three
cyclic shifts in every seed. One individual shift improved seed101. BCE did
not uniformly favor own banks. Thus correspondence matters at these selected
weights; own correspondence is not proved uniquely optimal. This note reads
only those completed reports and source, not the ongoing study's partial scores.

Its tied rule sets, independently in each channel/layer,

```text
mu_tied = (1/M) sum_m mu_m
v_tied  = (1/M) sum_m v_m.
```

The source applies these buffers to all routes and calls the original frozen
`adapter.validate()` over the authentic full VALID view. Cyclic shifts move the
whole bank set without changing its moments. Neither intervention tests a
common bank that includes separation among route means.

## Derivation and assumptions

Choose route R uniformly among M routes, and let X be one BN channel's input.
If mu_m=E[X|R=m] and v_m=Var[X|R=m] describe the same route populations, then

```text
mu_mix = E[X] = mean_m(mu_m)
E[X^2] = mean_m(v_m + mu_m^2)
v_mix  = mean_m(v_m + mu_m^2) - mu_mix^2
       = mean_m(v_m) + mean_m((mu_m-mu_mix)^2).
```

This law of total variance needs finite second moments, a common channel
coordinate system and the stated route weights. It does not require independent
routes, Gaussian activations or independent BN channels. With nonuniform route
weights, replace every mean by the same weighted mean. Uniform weighting is
appropriate to this complete four-route, fixed-mean diagnostic; all routes see
the same graph batches and native input shapes.

The second expression is numerically preferable. The proposed implementation
computes variance in float64 before copying to the original buffer dtype. The
common mean uses the exact existing arithmetic-tie operation, so only variance
is changed relative to that control. It does not
add BN epsilon to the stored variance; native BN retains its existing epsilon.
For means {-1,+1} and variances {1,1}, arithmetic variance is1 and mixture
variance is2. When route means agree, the two formulas agree. Larger variance
changes activation scale, but nonlinear downstream computation means it need
not improve AUC, BCE or any other score.

### What this formula does not recover from EMA banks

PyTorch tracked BN running variances estimate within-batch variance (with its
unbiased batch correction), while running means average batch means. These
are not complete first/second-moment sufficient statistics for a pooled history.
The formula is exact for population moments; applied to saved banks, it is a
**plug-in mixture approximation**.

If mu_m,t=sum_s w_s mu_m,s and v_m,t=sum_s w_s v_m,s are idealized matching EMAs,
plugging them into the mixture formula still omits temporal variation of each
route's batch means. In general, (sum_s w_s mu_m,s)^2 differs from
sum_s w_s mu_m,s^2. Initialization contributes decaying residual mass, estimates
have sampling noise, and variable batch/node counts are not encoded by the
common counters. Equal counters support a common update clock, not exact
pooled-population estimates. Finite-sample unbiased pooling instead needs
sample sizes and within-group sums of squares, e.g.

```text
s_all^2 = [sum_m (n_m-1)s_m^2 + sum_m n_m(mu_m-mu_all)^2] / (sum_m n_m-1),
```

with size-weighted mu_all. Those ingredients cannot be reconstructed exactly
from these EMA buffers, and we do not invent effective sample sizes.

The original shared sequential BN also EMA-averages each route's within-batch
variance; it does not insert this between-route-mean term. It additionally has
four updates per graph batch and order-dependent weights. Therefore mixture
tying is not reconstruction of native shared BN, and arithmetic tying is not
an implementation error in that earlier diagnostic.

Finally, all layers are intervened on together. Earlier common banks change
inputs to later layers, whose saved moments came from the original training
trajectory with training BN and dropout. Neither common-bank construction is a
fresh measurement of those new eval activations. No recalibration is performed.

## How the additional result would change interpretation

| Complete three-seed result | Defensible update |
| --- | --- |
| Mixture substantially closes the own-minus-arithmetic gap | Some apparent harm from arithmetic tying was due to its missing between-mean scale. Personalized statistics are not the only plausible explanation of that contrast. Cyclic sensitivity remains observed. |
| Mixture remains materially worse than own | Dependence on route-specific statistics survives a coherent common-mixture approximation at these weights. Training-history and cross-layer coadaptation remain confounded. |
| Mixture matches or exceeds own | Own-bank assignment is not uniquely best even for the selected private-BN model. Treat the alternative as a development diagnostic, not a newly selected serving rule. |

Report every seed and AUC/BCE/Brier without a new significance claim. An AUC
repair need not repair calibration. The intervention cannot apportion the
training benefit of private BN, compare causally with another selected G0
checkpoint, or establish generalization. No outcome is needed to change or
pause the current nine-fit plan.

## Prepared follow-up

`PLAN.md` fixes own and mixture conditions for all three completed checkpoints.
The new script is a small explicit derivation of the authentic original bank
evaluator; `SOURCE_DIFF.patch` shows its changes. It retains the same loader,
callback/runtime checks, full VALID collector and mean-logit readout. Prior
completed tied/cyclic reports are pinned and reused, so they require no extra
forwards. The own condition checks reconstruction against the saved export.

Budget: three sequential CPU constructions, six full VALID calls,120 native
route forwards, two threads,300 seconds per seed, no retry. No training,
checkpoint selection, TEST, or normalization recalibration. Preparation checks
are static/source-only; native execution has not been performed or launched.
