# Private nonlinear input routes around one native backbone

This package prepares two fixed methods for the proposed cross-task study.
It extends the existing molecular private input residual to the same declared
logical input boundaries used by the native-preserving wrapper. It adds no
native model, internal-site search, hyperparameter grid or training authority.
Predictive benefit and novelty remain unestablished.

Each route changes an encoded input `h` to
`h + U_route GELU(V_route h)`. Both matrices learn from the native task loss.
`U` starts at zero. The rank is `min(8,width)`; the coefficient is1. No
orthogonality or diversity penalty is imposed after initialization.

## Exact source locations

| Native backbone | Residual location | Encoded width |
| --- | --- | --- |
| PolyFormer | Complete `lin1` output for every original polynomial channel | Native configured hidden width |
| Native OGB GIN/GCN, with or without virtual node | Complete nine-field `gnn_node.atom_encoder` output | 300 |
| NCN/NCNC | Declared first `encoder.convs.0.lin` output | 64 |
| BUDDY Collab SIGN-k0 | Both `label_lin_layer` and endpoint `lin_feat` outputs | 16 and1,024 |

The residual follows the existing BE input transform when BE is retained.
Categorical IDs, complete atom fields, native affines, message passing, graph
preprocessing, normalization, labels, negative sampling and task operators
remain unchanged. BUDDY retains both structural and feature branches. The
same feature residual applies independently to each endpoint before the
native feature product; it does not change endpoint or query identities.
All paths are explicit and source-bound; unsupported interfaces raise errors.

## Dimension-aware initialization

Where four rank-8 subspaces fit, one isolated CPU Gaussian QR frame provides
four disjoint orthonormal initial down-projections. This retains the existing
molecular candidate's float32 QR/sign convention and seed offset901003.

BUDDY's structural width is16. Four rank-8 disjoint subspaces cannot fit.
That branch instead initializes four separate orthonormal frames with
unavoidable overlap. The rank and actual frame policy are recorded per site.
No claim of four mutually orthogonal subspaces or maintained diversity is made.
Later branches use seed offset104729 times their declared branch index.

## Two methods and controls

`portable_input_residual_be_v1` retains original boundary GNNM factors and
adds the private residuals. Initial added residuals are exactly zero, so the
starting function equals the existing BE control.

`portable_input_residual_only_v1` uses the same four-route source orchestration
and one shared native backbone, with only the private residuals. Its conversion
removes the freshly constructed BE hooks and private bank registration and
optimizer ownership before any update. No native Parameter is removed or
replaced; removed bank tensors are not retained by the checkpoint closure.
The native output affine has no trainable private correction. This is a
four-route nonlinear adapter ensemble, not a factorized single-model control
or an independently trained teacher ensemble.

Residual-only routes initially implement the same native evaluation function.
Their distinct down-projections give different task-gradient pathways to the
zero-initialized up matrices; native route dropout remains unchanged. Whether
this creates useful complementary predictions is an empirical question. This
control isolates dependence on BE but is not a parameter-matched comparison.

Required comparators are native M1, original boundary GNNM and four genuinely
independent native fits. Use paired seeds/splits and the same prospectively
frozen matched stopping rule. The loss stays the mean of four native losses;
predictions stay mean raw logits. Neither candidate is a source-reproduction
baseline. There are no pretrained teachers or extra backbone copies.

## NCNC and source custody

The existing task/recursive-call scope remains active. In residual-only NCNC,
the output hook becomes an observer and leaves the native output unchanged.
The inherited observer's `applied/bypassed` counter names mean scope eligibility,
not factor application; candidate records rename the envelope accordingly.
Source recursive completion remains unchanged. Private native RNG streams and
one shared native BN state are retained, rather than copied per route.

Native input-group optimizer dictionaries/options remain intact. Residuals
join the group owning that input's original source weight. Checkpoints include
full native session state plus the exact method/source receipts. Restoration
rejects a different method, source generation or frame policy. Registration,
source parameter identity and complete optimizer ownership are checked.

## Runner integration

For node/OGB jobs, use the actual frozen native callback and
`base_procedure:"gnnm_boundary_native"`, plus the pinned
`extension.py:transform` session transform. Declare `method_settings` exactly
as in `METHODS.json`; before launch, pin the final actual package manifest with
`method_settings.manifest_sha256` and the session transform's source SHA256.

For link jobs, use the pinned **new** `link_callback.py` entrypoint and the
already explicit task-specific link policy. That entrypoint constructs the
frozen native integration's base four-route session, then applies this source
extension. Do not declare a generic link session transform: v2 deliberately
rejects that interface. The outer callback and inner native source authorities
are both retained. Source and current v1/v2 runtime files remain unchanged.

## Readiness

`METHODS.json` and `SOURCES.json` define methods and frozen source mapping.
Local tests cover residual tensors, zero-up/gradient semantics, orthonormal
frame limits, original molecular initialization equivalence and optimizer
ownership only. They provide no benchmark quality evidence. Actual native
node/OGB/link construction, full forward/backward, task-scope behavior and
selected-state checkpoint/resume must still be qualified on authorized data
before scientific fits. The full campaign still needs the user's GO.

The mechanism builds on adapter/low-rank residual ideas and the existing
molecular candidate. This implementation does not establish a new literature
gap, novelty, accuracy improvement or efficiency advantage.
