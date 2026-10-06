# Disabled two-hop adapter: source preparation

**Ready for independent source review only.** `ADAPTER_RELEASED=False`. The module has not been imported or executed. No Torch, graph, CPU/GPU fixture, model, cost, derivative, fit or held-score execution occurred during preparation. AST checks establish source structure/syntax, not semantic or backend correctness. Existing native/private jobs retain priority; no native check or host-setting change was launched.

## Reuse and interface

The module supplies two tuples of the existing G0 `Pair` objects, differing only in their Laplacians. It calls the pinned native port's `_sparse_pairs(..., affinity_permutation=None)` for exact direct support and reuses its `SparseLaplacian` class in raw unit weights. It adds no assignment solver, callback, supervision loss, episode, trainer, model or acquisition path. The reviewed exact sequential VJP implementation consumes those banks through the unchanged G0 `_assignment_map`, independent-Q private partial, full Q/shared credit and original-phi recommit. The native port supplies the existing callback and pair interface.

Public `prepare_bridge_pair_banks` fails before backend/data work while disabled. The underscored factory is for a later separately authorized engineering caller. That caller supplies already admitted immutable public context, copied S roles, the one protocol permutation and independently verified source/protocol/census hashes. This adapter does not load files, authenticate physical dataset payloads, or replace the existing custodian gates. Native source flags/gates are not modified by this module.

## Fixed operator and stable degree construction

For each T=Sₐ∪Sᵦ and U=V\T, unit, reciprocal nonloop public edges define

\[
C=A_{TU}D_U^{-1}A_{UT},\quad W_B=\operatorname{offdiag}(C),\quad
L=(L_D+L_B)/(1+\max d_D+\max d_B).
\]

D_U contains **full public degrees**, including terminal neighbors; no interior label or orientation is used. The pinned connected graph must have 24,492 nodes and 93,050 undirected nonloop edges and no zero-degree vertex. Zero degrees fail this fixed-context guard; no ridge, teleportation, clamp or alternate graph is introduced. Classifier self-loops are retained outside this responsibility operator.

For an interior u with k_u distinct terminal neighbors,

\[
(d_B)_i=\sum_{u\sim i}(k_u-1)/d_u,\qquad
(L_BQ)_i=\sum_{u\sim i}[k_uQ_i-\sum_{j\sim u,\ j\in T}Q_j]/d_u.
\]

Degree is accumulated from **nonnegative integer k_u−1 contributions**, rather than subtracting diag(C) from C1. Action cancels the self term algebraically. Each k=1 star contributes zero coupling. Affinities have zero diagonal; Laplacian diagonals retain degrees. Zero-diagonal affinities need not be PSD. The resulting Laplacians are symmetric PSD, annihilate uniform fields in real arithmetic, and have spectral norm≤2 under the common sum-max-degree normalization. Numerical residuals must pass unchanged qualification tolerances; no repairs hide errors.

Two differentiable sparse `index_add` aggregations act on compact terminal–interior incidence. This is the same fixed-coefficient/dense-column pattern as the native port, with unused full-graph rows omitted algebraically. It refines the sketch's conservative full-graph product estimate without changing the operator. Work is O(|E_TU|M+|E_TT|M); intermediate storage is O(|U_used|M). Actual incidence counts, higher-order backend behavior and whole-map memory/time remain unknown. No C or dense terminal affinity is materialized, no inverse is solved, and Q is never detached/clamped/rebalanced by the adapter.

## One guarded bridge control

The supplied permutation must be a bijection preserving copied S classes **and equal the single protocol seed17 hash ordering**: increasing node-ID order in each class maps onto `sha256(amazon-response-G0|split=0|seed=17|perm|<class>|<node_id>)` order. The factory recomputes this rule solely to reject alternatives; it does not choose a permutation or sample RNG.

Existing native array convention is W_control[i,j]=W[p[i],p[j]]. Defining Pq=q[p⁻¹], the adapter applies PᵀL_BP as `(L_B @ q[p⁻¹])[p]`. Direct L_D is unchanged. Both banks share the same bridge factor and normalization; live uses identity gathers so the operation pattern is identical. Matrix-letter convention is explicit; this preserves the existing numeric p rule.

Both bank tuples must enter the reviewed exact sequential VJP path with G0 `control="live"`. Bridge permutation is represented in the pair bank, not by changing G0's control/γ. Thus γ and the **2γ denominator term** remain identical. No graph-free, target-oriented or coefficient/depth arm is added. Target orientation remains off.

## Static exact support accounting, not executed here

The factory encodes and checks the authorized census's ten direct row/edge/isolate/component totals. It computes integer partner-incidence counts and DSU components from direct edges and each interior's terminal star. This yields exact counts of bridge-supported rows, newly coupled direct isolates, remaining isolates and connected support for live and control, without emitting a clique matrix. A row acquiring a bridge is distinct from the support graph becoming connected.

`bridge_two_hop_path_instances=Σᵤ choose(k_u,2)` counts paths through interiors, **not distinct bridge edges**; repeated terminal endpoints may share several interiors. Distinct bridge-edge count is explicitly not reported. Pair membership totals are not unique S-node totals. `minimum_restoration_gate` requires at least one newly coupled former isolate in every live pair; it is a minimal premise, not sufficient coverage or quality evidence. Control nonidentity, native correctness and resources are unqualified.

## The one future paired no-commit sequential qualification

After independent source review, root may prepare and schedule a single separately released qualifier in an available admitted window. Existing native/private jobs take priority. No launcher or new admission/host-mode mechanism is provided here. `SOURCE_BINDINGS.json` fixes the reviewed disabled sequential implementation (`1f9e704a1b95a98cec688d695094058594758a99bac2f057e2fe90e7f9315167`), its source reviews and current flag-only released copy (`b9ac6babdee5d5ad2fb2f86eeabbb4738e1c15ae1603b8b3d67cd3629b717fb1`). It also fixes the common400 metadata receipt and origin RUN. The remote readonly common400 `initial.pt` descriptor is 109,671,738 bytes, SHA256 `2e0e9b44767abc12b3dc896986ea2d4faeaa667c8682b95929f927d10718154f`; its payload has not been opened here.

1. Authenticate this packet and the pinned G0/native/protocol/public-role/census bindings using existing custody/admission. Prepare both banks once, retain the exact static coverage report, and stop NO-GO if any live pair lacks a newly coupled former isolate. Establish that the frozen permutation actually changes the bridge operator; no replacement p is chosen if it is uninformative.
2. Reuse the existing native callback/pair interface and the **reviewed exact sequential VJP implementation**, which calls the unchanged G0 assignment map. Evaluate live first, then bridge-permuted, each starting from **the same immutable common400 θ/φ bank and public/S/R inputs**. Use all ten pairs, M4, exact default G0 Config and all eight assignment steps. Root must bind the bank and inputs before any call and verify unchanged caller state/native aliases, modes, attributes, buffers, inputs, RNG and false original gates on success or failure. Full monolithic native higher-order execution previously exhausted about 80 GB; it is not the qualification route.
3. Preserve complete derivative parity under the existing reviewed tolerances: every shared gradient coordinate, independent-Q private partial, full Q/raw-cost/probe mixed and Hessian chain, query logits/objective, raw/normalized costs, Q, private states and original diagnostics. Use the existing scoped derivative oracle/comparison strategy; no monolithic full-context OOM retry. Recommit must recompute Q and every private row at θ+ from **original φ**, then compare to a separate fresh sequential response at θ+ from that same original φ. Check public-loop handling, uniform-null action, PSD/norm construction, exact permutation convention, positive Q, row/member-column residuals and unused tangents without tolerance relaxation or numerical repair. Any failed invariant is NO-GO. Returned virtual θ+/φ states are discarded after engineering comparison; there is no persistent commit, acquisition, selector, scoring or A/VALID/TEST access.
4. Declare resource caps and the exact qualifier invocation matrix before the check. Measure whole-preparation/call wall time, peak process/device memory, and incremental direct/bridge aggregation, adjoint, recomputation and derivative work against the admitted direct-only sequential baseline. Retain attempted counters on failure and separately charge oracle work. The pinned live sequential episode accounts for 48 native forwards, 24 private-gradient constructions, 12 native member VJPs, 30 primal pair maps, 10 pair-map VJPs and one small query VJP; a separate original-φ response adds 12/8/0 native/private/VJP calls and 10 primal pair maps. These source counts do not measure bridge reverse work or resources. The prior strict original all-six certificate measured 505.280 s inner / 510.110 s wrapper, 37,422,669,312 bytes peak allocated and 39,864,762,368 bytes reserved. It is prior cold original-context evidence, **not a bridge or common400 budget**. Actual adapter coverage, incremental backend time/memory and full qualification overhead remain unknown.

Any backend, coverage, nonidentity or declared resource failure stops this candidate. No deeper walk, tuned weight, new tolerance or pilot is automatically authorized. Current seven-arm training, original scores and held gates stay fixed. This packet's result is disabled source preparation; all empirical usefulness, exact-native equivalence and feasibility remain unknown.
