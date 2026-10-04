# Shared propagation with HL-GNN member filters

Date: 2026-10-04. Bounded design/source packet. **No numerical imports, experiments, dataset/checkpoint access, training/scoring launch, server work or additional agent.**

`member_filter.py` implements a minimal encoder with fast affine member maps, optional free member-specific hop coefficients and two storage choices. `direct_reference` deliberately propagates every member and is the reference function for the planned equivalence check. `planned_equivalence.py` contains a prospective output and input/parameter/coefficient gradient comparison; it was parsed, not imported or executed. `FIT_CONTRACT.md` states the fitting and serving boundary. The native source/defaults are pinned at HL-GNN commit `0855b0de74a8f0586b8cc203e9ba4dbbb57243f4`.

## Exact factorization and native reduction

Let `Z=Dropout(X)` be one common dropped learned embedding matrix, `P` the native fixed normalized TRAIN adjacency, and `a_mk` learned scalar hop coefficients. With a node-independent member affine map `Z R_m + 1 b_m`, the direct encoder is

`H_m = sum_(k=0)^K a_mk P^k (Z R_m + 1 b_m)`.

Associativity and linearity give the exact real-arithmetic identity

`H_m = [sum_k a_mk P^k Z] R_m + [sum_k a_mk P^k 1] b_m`.

The actual fast map is `R_m = diag(r_m) W^T diag(s_m)` and `b_m = b * s_m + delta_m`: one shared dense weight/bias plus input/output feature factors and a member bias offset. The implementation propagates the augmented matrix `[Z, 1]` once through the shared K powers. Both feature and constant channels are mixed with the same member coefficients, and the affine map is then applied. This supports freely learned, signed, unnormalized private hop coefficients without propagating every route separately.

The bias term is necessary. Native HLGNN has an affine input map **before** propagation, and its symmetric normalized operator generally has `P 1 != 1`. Even if a different operator preserved constants, the output bias would be `(sum_k a_mk) b_m`, rather than just `b_m`, unless the coefficients summed to one. Native KI coefficients are not unit-sum constrained. Appending ones **after** input dropout preserves the undropped affine bias.

At M=1, the active forward uses the native sequence `dropout -> lin1 -> normalize P -> weighted propagation`, with K=15, alpha=0.5, dropout=0.3 and KI initialization. Its state dictionary has the native three parameter names and no extra adapters. Its reset preserves the author's omission of `lin1` reset, so independent fits must construct fresh objects. The `factored` method remains available at M=1 for the planned algebraic oracle, but the default M=1 forward retains native arithmetic operation order. Normalization in code is computed before the deterministic dense map; it uses the identical graph-only native call and does not alter RNG or parameters.

For tied parameters, fixed P and identical dropped input, direct and factored expressions define the same differentiable function, so input, dense-weight/bias, fast-factor/bias-offset and hop-coefficient gradients agree in exact arithmetic. Floating-point reassociation may change rounding; numerical tolerance and backend checks remain prospective. The source is not a completed equivalence certification.

## Dropout and representational boundaries

The source drops input embeddings before its linear map. Moving dropout after propagation generally changes the function: `P Dropout(X)` need not equal `Dropout(P X)`. Independent elementwise masks for each member also do not provide one common `P^k Z` basis. Repeated or packed M-mask propagation would still carry M distinct input signals and its arithmetic must be charged. This implementation samples one common input mask for M>1; that is an explicit ensemble adaptation, while M=1 follows native dropout.

Member feature factors alone change feature coordinates. They do not create different polynomial spectral filters when the hop coefficients are tied. Free private coefficient rows can create different graph-filter polynomials on the same operator; their ability to do so does not establish learned useful diversity. Every nonzero high-hop coefficient permits high-hop feature dependence; a largest coefficient at a shallow hop does not imply a smaller strict receptive radius.

The identity does not permit moving arbitrary nonlinear, node-dependent affine, attention, normalization or graph-edit operations across P. The module includes no inter-hop nonlinearities, member-specific operator, hidden-member averaging, auxiliary loss or custom gradient routing. Sharing input embeddings and dense W changes the parameterization relative to independent full models; equivalence is to the deliberately tied direct reference.

## Compute, storage and gradients

Let N be the nodes, E the nonzeros of normalized P (including its self-loops), D the input width, H the output width and M the members. The shared embedding is `N*D` parameters once. Dense affine storage is `D*H+H`, fast affine storage is `M*(D+2H)` for M>1, and coefficient storage is `K+1` when tied or `M*(K+1)` when private. M=1 removes all fast parameters and reduces to native storage. Coefficients initialize in float64 as in the pinned source; other parameters use the default PyTorch floating dtype. Predictor/optimizer/graph storage is additional.

| Computation | Sparse forward work | Dense/mixture work |
|---|---|---|
| Tied direct reference | M*K applications in H channels, order `M*K*E*H` | M dense projections, order `M*N*D*H`, plus weighted sums |
| Factored shared powers | K applications in D+1 channels, order `K*E*(D+1)` | The implemented affine maps still perform M dense projections; private coefficients mix at order `M*K*N*(D+1)` |
| Tied coefficients | Same common K applications | One shared feature/constant aggregate; M affine outputs |

Counting K rather than M*K sparse calls is exact for this implementation. It does not claim an M-fold measured speed gain: channels differ, dense projections remain, scalar mixtures and memory traffic can dominate, and stacking outputs has a cost. With D=H=512, the bias channel adds one propagation channel. No timing or peak measurement is supplied.

`storage="powers"` explicitly retains K+1 augmented `[N,D+1]` states and then forms one route aggregate at a time; outputs still occupy M `[N,H]` states. `storage="aggregates"` maintains one current augmented power plus M route aggregates when coefficients are private, or one aggregate when tied; it then forms all outputs. This is a **forward working-state tradeoff** between roughly `(K+1)*N*(D+1)` and `M*N*(D+1)` aggregate storage, with output storage and transient additions/projections/stacking charged in both cases.

Ordinary eager autograd may retain all propagated powers for coefficient gradients even in aggregate mode, plus projection inputs and intermediate operations. Streaming variables alone do not prove O(M) training activation storage, and this packet implements no recomputation or custom-memory backward. The shared graph path accumulates all member adjoints and applies K transposed sparse operations in D+1 channels; the direct reference has M paths in H channels. This is exact ordinary differentiation, including private coefficient gradients. Backends may retain additional graph buffers, so neither formula certifies peak memory.

Because DDI embeddings learn and dropout changes, powers are recomputed each training call. They cannot be treated as a fixed SIGN cache across optimizer steps or masks. A detached persistent cache would change the embedding/input gradient and violate the equivalence contract.

## Planned meaningful equivalence check

The callable plan uses an undirected weighted irregular graph plus an isolated node. It first verifies `P 1 != 1`, preventing a bias test from passing accidentally on a regular or row-stochastic graph. It uses nonzero bias offsets, signed arbitrary coefficient rows, nonidentity fast factors, K=0 and K=4, M=1 and M=3, tied/private coefficients, and both storage paths. A fixed dropout mask multiplies each separately cloned differentiable input.

The scalar probe combines an asymmetric output weighting, `tanh`, and a quadratic term. Each comparison checks the complete output and gradients for the input, W/b, every r/s/bias-offset parameter and every coefficient element via `torch.autograd.grad`. The native reduction plan separately copies a fresh pinned HLGNN state in memory, uses K=15/KI/dropout0.3/alpha0.5 and replays input-dropout RNG. It checks output, parameter names and all input/parameter gradients. Float64 tolerances are explicitly declared in the source; they are prospective criteria, not passed results. A later production check must also use the declared production dtype/backend and its documented tolerances.

## Prior work and novelty assessment

Existing conclusions were reused; no new primary method paper was acquired or read. The current `literature_memory/index_v47` supplies the scoped HL-GNN author-source record. Earlier retained reports supply GPR-GNN/SIGN/GAMLP conclusions; the SGC boundary below is the explicit SGC special case in the already inspected GPR-GNN passage, rather than a newly inspected SGC paper. `LITERATURE_REUSE.json` records exact source paths, hashes and scopes. A bounded index lookup did not locate separate GPR-GNN/SIGN/SGC/GAMLP records in v47; the saved earlier conclusions remain attributed at their original locations.

| Prior | Established operation / limit for this packet |
|---|---|
| HL-GNN, pinned author source | Learned affine input followed by weighted normalized powers, served by a Hadamard pair MLP. This packet supplies an exact tied-context implementation factorization and member parameterization; it does not invent that backbone or reproduce its empirical claims. |
| GPR-GNN | Free learned signed polynomial hop coefficients. Private rows instantiate this established filter mechanism per member. The saved member-subspace assessment also states that feature-diagonal maps commute with fixed linear P and do not alone create distinct spectral filters. |
| SIGN | Linear graph operators/powers before dense learning and retained multiscale features. A common power bank and dense heads are established. Learned DDI embeddings require recomputation here, making this an end-to-end dynamic input use rather than a fixed raw-feature cache. |
| SGC | Fixed single-hop-order coefficients and removal of nonlinearities are an established special case of the polynomial family, according to the reused GPR-GNN passage. No separate primary SGC method/source audit was performed in this packet. |
| GAMLP | Node-specific attention over propagated features and scalable decoupling are established. This packet's member-global coefficients are simpler; it includes neither GAMLP's node attention nor its complete label-propagation/reliable-label pipeline. |

Fast multiplicative member maps also have BatchEnsemble ancestry, retained in the earlier efficient-path conclusions. Algebraic reassociation, proper propagated-bias accounting, ordinary gradient equivalence and free per-member filters do **not** themselves establish methodological novelty. The new deliverable is a practical source realization tailored to the pinned affine HLGNN and learned DDI embedding recipe. No new scientific learning principle is established by this packet.

A plausible scientific question is whether learned graph-filter variation helps the shared-responsibility conditional auxiliary improve a fixed count-free native ranker. A useful comparison would distinguish a tied-filter member model from a private-filter member model under native supervision, then compare joint versus independent-side auxiliaries with identical private-filter parameterization, input/dropout context, candidate protocol and complete budget. Baseline quality and held-TEST ownership remain the existing experimental protocol. Coefficient differences or auxiliary likelihood improvement alone cannot establish predictive complementarity or specialization. The earlier structured Single control is still relevant to any necessity claim. No auxiliary or experiment is implemented here.

The root DDI geometry summary motivates this transfer test through TRAIN subset-choice opportunity. Its measurements concern Boolean support geometry under NCNC batch24,576, not HLGNN filter quality, encoder cost or conditional-likelihood throughput. It cannot certify the above prediction or compute claims.

`SOURCE_CHECKS.json` records the passed stdlib syntax/source checks. The remaining task is ordinary source integration and the planned equivalence check in the existing qualified numerical runtime; this packet adds no new resource gates or predictions of empirical success.
