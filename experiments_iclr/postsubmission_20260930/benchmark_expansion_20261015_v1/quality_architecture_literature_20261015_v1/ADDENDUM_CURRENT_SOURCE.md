# Authoritative current-source correction

15 October 2026. **This addendum supersedes REPORT.md's C6 row and its statements
that C6 lacks a complete objective/credit freeze, plus its claim that C1's V
“prevents within-class collapse.”** The frozen report and original manifest
remain unchanged. The three primary-reading scopes and zero-additional-mechanism
disposition are unchanged. No new literature search or model execution occurred.

## Current C6 is conditional_aggregation_v1

The original row inspected the older `link/adapters.py:make_late_gate`: hidden8,
live preheads/raw-logit deviations, and no complete training controller in that
builder. It does **not** describe the subsequently frozen C6 package. The
authoritative package manifest is
`edef4b1f552a495e09e64855afbd96c7eb14ce5ef02d26cc13ec530d80e9805f`.

Current source uses only **detached prediction descriptors** per route/query:
`c=1−H(p_m)/log(K)` and `d=JS(p_m,mean_j p_j)/log(2)`. Binary sigmoid output is
lifted to `[1−p,p]`. No preheads, labels, IDs or teacher enter the gate. Its
shared hidden4 GELU scorer and route intercepts give

```text
s_m = b_m + vᵀ GELU(W[c_m,d_m] + a)
alpha_m = .05 + .8 softmax_m(s),       sum_m alpha_m = 1
z_served = sum_m alpha_m z_m.
```

The 20-parameter conditional gate initializes at the exact uniform raw-logit
mean. Descriptor detachment removes that feature path's gradient to logits;
the direct served-logit path still uses learned alpha, and gate parameters
still learn. Every shared/native, private-boundary and aggregation parameter
receives the same declared autograd objective, **global J=.5F+.5L**, in one
backward and original optimizer transition. F is mean unchanged native own
risk plus the native regularizer; L uses the actual served output and the same
source reduction/regularizer. The regularizer occurs once in J. This is a
complete fixed objective/credit rule. Retaining F and the .05 route floor
does not guarantee member competence, calibration or ranking improvement.

The matched controls are already explicit:

| Method | Readout parameters | Training |
| --- | ---: | --- |
| `aggregation_mean_globalJ_v1` | 0 | Same all-block J |
| `aggregation_global_globalJ_v1` | 4 route intercepts | Same all-block J |
| `aggregation_conditional_globalJ_v1` | 20 | Same all-block J |

The primary training control is **mean-globalJ**, not the separate mean-own-F
baseline. A selected gated bank's same-state uniform mean diagnoses its readout
effect; separately trained mean-J/global-J/conditional-J banks compare the joint
pipelines. The wrapper returns both served and same-bank-mean VALID readouts
from the same four forwards. No VALID combiner fit, capacity-matching fit or
automatic C1/C4/C8 combination is part of this source.

## Qualification status

The separate existing MolBACE/GIN CPU receipt passes all three methods: six
actual optimizer transitions total, nine complete VALID calls of151 graphs,
and bit-exact restored served/mean VALID outputs with original Parameter
identity retained. This is the exact CPU engineering cell; it establishes no
convergence, predictive benefit, CUDA/resource admission or campaign GO.
The receipt does not qualify other molecular settings or native node/link
ports. Any newer bridge needs its own pinned receipt. Final selected-state
TEST reconstruction remains unsupported by this package.

## C1 variance wording correction

V **penalizes low within-class sample variance across distinct graphs**. It
does not guarantee avoiding or escaping collapse. For a coordinate with
`v=(1/n)sum_i(z_i−mean(z))²`,

```text
dv/dz_i = 2(z_i−mean(z))/n.
```

At exact within-class collapse this derivative is zero. The implemented
`sqrt(v+epsilon)` then also has zero derivative through v, even when the
hinge penalty is positive. Other task/alignment gradients may move the state;
V alone supplies no universal escape guarantee. It is still a sample-axis
regularizer, not cross-member output repulsion. Likewise the known zero-up
adapter's member-dependent derivative paths establish no new diversity theory.

Fresh source and qualification pins, plus unchanged frozen report/manifest
hashes, are recorded in `ADDENDUM_RECEIPT.json`.
