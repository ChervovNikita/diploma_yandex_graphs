# Corrected structured-responsibility control v2

15 October 2026. Separate proposal and standalone CPU helper. **Unintegrated;
no model fit, dataset record, server action, pilot change, or gain claim.**
The frozen `link/` v1 source remains historical evidence.

## 1. The v1 comparison has a raw-variance confound

Let c≥0 and let each latent variable below have the independent Exp(1) law.
For a nonself query e=(u,v,r), v1 assigns

```text
endpoint: w_e = c + (X_(r,u) + X_(r,v))/2
query:    q_e = c + Y_e.
```

Both means are μ=c+1. Their variances are respectively 1/2 and 1. The
nonself endpoint marginal is c+Gamma(shape=2, rate=2), while the v1 query
marginal is c+Gamma(shape=1, rate=1). Calling that query control “matched”
is incorrect. On a self-loop u=v, the endpoint rule is c+X_(r,u), with
variance 1; this special case happens to match v1's one-draw query marginal.
A mixed set of loops and nonself queries therefore needs row-specific laws.

This correction does not reinterpret any v1 fit or change its weights.
Any future comparison using v2 needs its own root-owned freeze and integration.

## 2. Persistent semantic identity

An identity is `(scope, directed, u, v, relation)`. `scope` identifies fixed
dataset/node-ID/relation-ID semantics. Keep it fixed across minibatches,
epochs, resume/replay and methods intended to share a weight table. Do not
put epoch, minibatch, row index, source label, backbone, or outcome in it.
The seed and member ID select a field within this scope.

- Undirected endpoints are canonically sorted. A reversed undirected query
  and any repeated native row reuse exactly the same query draw(s).
- A relation ID is part of both the node-field and query-field identity.
  Relations have separate ideal latent variables even when node IDs agree.
- Directed reversals have distinct query innovations. The endpoint rule is
  explicitly role agnostic: reversing a directed query leaves its endpoint
  weight unchanged. No sender/receiver-specific rule is silently introduced.
- The helper regards repeated `(u,v,relation)` rows as duplicate semantic
  queries. A source with genuinely distinct parallel-edge identities would
  need an explicitly extended identity schema before adoption.
- Sampled-positive/negative source labels authenticate alignment and choose
  the native loss group; they never enter the raw-weight hash. A sampled
  negative is not asserted to be a verified graph nonlink.

“Query IID” means independent innovations across **distinct canonical semantic
queries** in the ideal law. Duplicated rows are dependent by design. It also
does not mean identically distributed across loop and nonself types.

## 3. Endpoint covariance is half-incidence covariance

Fix one member. For each relation r, put a column at every `(r,node)` and set

```text
B[e,(r,i)] = (1[r=r_e,u_e=i] + 1[r=r_e,v_e=i])/2
w = c 1 + B X.
E[w] = (c+1) 1;                  K_endpoint = Cov(w) = B Bᵀ.
```

Equivalently one may form a separate B_r per relation and use block-diagonal
independent node fields. Rows remain in their original source order. Each row
sums to one; a loop has one coefficient 1, not two distinct coefficients 1/2.
Since `aᵀKa=||Bᵀa||²`, K is positive semidefinite without a graph-distance
kernel assumption. No full Q×Q covariance is needed for sampling.

| Pair of rows within one relation | Raw covariance |
| --- | ---: |
| Nonself row with itself | 1/2 |
| Distinct nonself queries sharing exactly one endpoint | 1/4 |
| Disjoint endpoint sets | 0 |
| Duplicate nonself rows; undirected reversal | 1/2 |
| Identical endpoint set under directed reversal | 1/2 |
| Loop `(u,u)` with itself or its duplicate | 1 |
| Loop `(u,u)` and nonself query incident to u | 1/2 |
| Loops at different nodes | 0 |
| Rows in different relations or different members | 0 in the ideal law |

Multiplicities matter: duplicate rows repeat the same B row and affect the
subsequent source-group denominator. A loop's two incidences must accumulate.

## 4. Corrected query control

For every distinct canonical nonself query draw two independent innovations:

```text
q_e = c + (Y_(e,a) + Y_(e,b))/2,             u_e != v_e.
q_e = c + Y_(e,loop),                       u_e == v_e.
```

This matches the entire ideal raw marginal of each endpoint row, including
its mean, variance and higher moments. Reuse these draws on duplicated rows.
Its ideal raw covariance is

```text
K_query[e,f] = 1[e and f have the same canonical query identity] * v_e,
v_e = 1/2 for a nonself query, 1 for a loop.
```

Thus the corrected comparison changes dependence across distinct queries
while preserving rowwise raw marginals. It cannot isolate raw covariance as
the only changed feature: higher-order joint dependence also differs.

The implementation uses BLAKE2b64 with namespace `link-resp-v2`, stable JSON
keys, and 52-bit midpoint uniforms transformed by `-log(U)`. No global RNG
is read or modified. Finite draws lie in `(0,53 log(2)]`; the ideal Exp law is
unbounded. Cryptographic hashing provides a deterministic pseudorandom table,
not a proof of stochastic independence over a finite set of seeds. Analytic
moments and independence statements above refer to the ideal latent law.
Endpoint and query namespaces are separate; no special cross-arm noise coupling
or same-seed equality of sampled marginal summaries is promised.

## 5. Native-group normalization changes the comparison

For each member and each complete native source reduction group g, the
existing proposed objective uses

```text
r_e = w_e / mean_(f in g) w_f,         sum_(e in g) r_e = |g|.
```

NCN and NCNC use **positive mean plus negative mean**, so the complete positive
and negative groups get separate denominators. BUDDY uses **one original
mixed batch mean**, so it gets one denominator. Do not rebalance BUDDY by
class, divide NCN by an extra two, normalize arbitrary chunks, drop rows, or
replace native sampling. Raw weights persist; normalized weights change with
minibatch membership and sampled duplicates.

The denominator changes all marginal distributions and induces dependence.
It can couple raw-independent queries and relations within one source group.
Matching raw marginals does **not** match normalized marginals: the denominator
has a different joint law under endpoint and query innovations. The proper
interpretation is a comparison of structured versus query innovations **with
the same declared native-group normalization**. A normalized-marginal-matched
dependence-only control is not supplied by this helper.

Mass conservation alone does not imply `E[r_e]=1`. With distinct nonself
query-IID rows in a group, exchangeability gives that equality; loops mixed
with nonself rows, repeated rows, and irregular endpoint incidence can remove
the symmetry. Here is an exact counterexample with duplicate nonself rows.
Let two duplicate rows have weight W and a disjoint row have independent
weight V, with W,V identically distributed and nonconstant. For T=V/W,

```text
E[r_duplicate] = 3 E[1/(2+T)]
 = (3/2) E[1/(2+T) + 1/(2+1/T)] < 1,
```

because the bracket is at most 2/3, with strict inequality when T≠1.
Conservation then gives `E[r_disjoint]>1`. This counterexample also applies
to the corrected query control when duplicate identities are present.

For a simple graph with three nonself queries `(a,b),(a,c),(d,e)`, endpoint
K has diagonal 1/2 and K_12=K_21=1/4. Let
`w(t)=μ1+t(w-μ1)`, so t=0 is constant weights. Taylor expansion within a
three-row group gives

```text
E[r_e(t)] = 1 + t² (Var(mean w) - Cov(w_e,mean w))/μ² + O(t³).
Var(mean w)=2/9;
Cov(w_1,mean w)=Cov(w_2,mean w)=1/4;  Cov(w_3,mean w)=1/6.
```

The leading biases are `-t²/(36μ²), -t²/(36μ²), +t²/(18μ²)`.
This expansion describes small noise; it is not an exact t=1 prediction.
The saved 30,000-seed synthetic hash check at c=.25 gives normalized means
`[.98756,.98500,1.02744]`, compared with query-control means
`[.99761,1.00143,1.00096]`. The endpoint disjoint-minus-incident-pair gap is
`.04116`, with descriptive simulation standard error `.00389`.

At a fixed parameter state, dividing raw weights by their known μ would give
an unbiased original-risk estimator in the ideal law, regardless of K. The
current ratio normalization generally does not. In training, persistent weights
also affect the learned losses and shared trajectory; even an identity that
holds for fixed losses would not establish an unbiased learned-risk claim.

## 6. Explicit private/shared criteria

The standalone helper returns

```text
F = mean_m native_scalar_loss_m,
W = mean_m native_group_reduce(r_m * per_record_loss_m).
shared criterion = F;                  private criterion = W.
```

It checks unweighted record reductions against all four supplied native live
scalar losses and requires ordered query signatures from each actual route
capture. Signatures detect mismatch only when the caller honestly computes
them from those captures; they cannot certify tensors' origins by themselves.
No auxiliary loss, backward pass, optimizer construction, or transition is
added. A future integration must gather both block gradients at one old state
and preserve the original optimizer ownership and clock. Such a mixed block
field generally has no common scalar potential.

## 7. Qualified local mechanism, not a neural-training theorem

Fix the shared parameters θ, a finite deterministic query table and the native
groups. Consider a private risk `R(ψ,r)=sum_e a_e r_e ℓ_e(ψ)+Ω(ψ)`.
The coefficients a_e are the actual native group denominators: for a single
NCN route, `1/n_positive` or `1/n_negative`; for BUDDY, `1/n_batch`.
Include any declared mean-over-members coefficient when analyzing a combined
private vector. Ω is independent of weights. Assume losses are twice
continuously differentiable near an interior stationary ψ0 at r=1 and its
Hessian H is positive definite. A locally unique differentiable stationary
branch then follows from the implicit-function theorem. This does not assert
that the training optimizer finds that branch.

Write G's column e as `a_e ∇ψ ℓ_e(ψ0)`, and let P be block diagonal over the
native source groups, with block `I - 11ᵀ/n_g`. For raw perturbation δw around
μ1, the derivative and the covariance of its linear proxy are

```text
δψ_linear = -H^-1 G P δw / μ,
Cov(δψ_linear) = H^-1 G P K Pᵀ Gᵀ H^-ᵀ / μ².
```

At noise scale t the proxy covariance has an additional t². K=B Bᵀ therefore
selects incidence-coherent combinations of task gradients; the query K selects
different combinations. The formula predicts neither a positive gain nor a
larger covariance in every direction. A covariance expansion for the actual
stationary solutions additionally needs an L2-controlled remainder; ordinary
pointwise differentiability alone does not justify exchanging the expansion
and expectation. Finite-noise ratio bias, higher-order remainders, finite
private capacity, nonconvexity, jointly evolving θ, dropout, minibatch sampling,
and Adam are outside this local calculation.

CPU checks verify the derivative in a fixed-θ, ridge-regularized two-dimensional
quadratic with unequal native group sizes, and verify the covariance matrix
algebra. The derivative errors at t=`1e-3,1e-4,1e-5` are
`3.49e-6,3.49e-7,3.49e-8`. This is a known influence/multiplier-bootstrap-style
local argument. It is not a theorem about the proposed neural training rule,
a posterior interpretation, an originality claim, or predictive evidence.

## 8. Retained ancestry and read scope

- Re-read the retained [10 October private-bootstrap triage](../../private_class_balanced_bayesian_bootstrap_triage_20261010_v1/REPORT.md):
  fixed positive supervised weights, private-only credit and mixed-block
  objectives already have close local and published ancestors, including
  Bootstrapped DQN and BatchEnsemble. No claim to a full new primary read.
- Re-read the retained [6 October graph-weight note](../../graph_error_guided_branch_growth_method_note_20261006_v1/REPORT.md)
  and its existing extracted primary text for Kojevnikov,
  *The Bootstrap for Network Dependent Processes*, arXiv:2101.12312v1,
  pp18–20. The dependent wild bootstrap specifies zero-mean multipliers with
  structured covariance; pp19–20 use a PSD overlap covariance and a Gaussian
  construction. That work studies conditional inference for network-dependent
  observations under explicit assumptions. It does not justify positive
  reweighted neural training, nonlinear shared/private optimization, or a gain.
- No fresh paper download, exhaustive priority search, dyadic-bootstrap primary
  read, or full-paper read credit is claimed here. The exact half-incidence
  covariance follows directly from the defined random variables.

## 9. Bounds, verification, and adoption boundary

`sampler.py` uses at most 262,144 existing query rows and 16 members. The
four-route objective never truncates; a larger native batch is rejected so
the bound must be explicitly revisited. Native NCN/NCNC's 65,536 positives plus
65,536 negatives fit together as 131,072 rows. The audited BUDDY port uses
1,024 mixed rows and keeps its source tail. Only the optional analytic audit
matrix has a 256-row cap; production sampling is sparse/O(MQ), with no dense
graph covariance. The upper row/member limits are engineering caps, not
scientific hyperparameters or runtime/memory qualification.

`checks.py` uses only synthetic identities, scalar losses and CPU tensors.
`CHECKS.json` records checks, numerical tolerances, versions and code hashes.
The suite covers persistence, orientation, duplicates, loops, relations,
RNG isolation, raw moments/covariance, native group laws, source parity,
explicit block gradients, large native-row acceptance, bounds, and the
qualified local quadratic calculation. It performs zero optimizer transitions.
`MANIFEST.json` records the frozen source/read hashes and delivery hashes.
Root retains ownership of integration, any real-data engineering qualification,
new study freezes, launches, and publication.
