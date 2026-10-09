# Actual private sites and source-ablated dependence

This statement follows the pinned model's actual operations. It is not a numerical Jacobian qualification or a guarantee that every site supplies useful class evidence.

For one native grouped row, the adapter is, in real arithmetic,

`Y = ((X * r) W) * s + b`.

It executes the author forward `Y0=author(X*r)` and returns `Y0+(s−1)*(Y0−b)`. Bias remains outside s. r=s=1 gives the native function with native slow derivatives, while input/output-factor derivatives remain available. nn.Linear semantic sites use the same last-axis algebra. Dtype casts preserve native AMP activation/output dtype; actual mixed-precision materiality still needs whole-native qualification.

| Private site | Source-ablated Jacobian statement |
| --- | --- |
| `feature_projection.0.input_factor/output_factor`, removed family channel rows | If the qualified cached input channel is exactly zero, its embedding is bias-free, so the projection input is zero. The factor's probe Jacobian is zero: input scaling multiplies zero and output scaling excludes the native bias. That row's signed source update reduces to positive factual responsibility-weighted CE/BCE, or zero if factual evidence is also absent/classifier-invisible. |
| Same first projection, retained feature/label rows | Probe dependence can remain through native nonzero retained channels. All private rows of a member are eligible recipients; the update is not restricted to only its removed family's rows. A nonzero activation still does not guarantee a nonzero final-score derivative. |
| `feature_projection.4.input_factor/output_factor` | The second projection receives the output of first `LayerNorm([C,H])`, PReLU and dropout. That LayerNorm normalizes across **all** channel/hidden coordinates, has native affine Parameters and sees native biases. A removed raw channel can therefore have nonzero later representation from remaining channels and normalization. Its probe Jacobian is not structurally forced to vanish. Masking again after normalization would be a different operation and is not implemented. |
| `semantic_fusion.query/key/value.input_factor/output_factor` | Factors see all post-projection semantic channels. Native gamma starts exactly zero, so these sites have zero predictor derivative in **both** full and probe views at that initial state. Preserve this skip initialization. Once ordinary native training makes gamma nonzero, remaining channels/nonlinear semantic attention can retain probe derivatives; saturation, dropout, null directions or gamma remaining zero can still remove them. |
| `fc_after_concat.input_factor/output_factor` | This native semantic mixing layer receives `[H,C]` flattened features after residual semantic fusion. Retained channels, cross-channel LayerNorm and learned attention can keep probe dependence. Factors are broad semantic input/output corrections; native final task/head layers remain shared and untouched. If its entire unbiased input action vanishes, output-factor probe dependence also vanishes. |

Feature and TRAIN-label channels are both included. Identical metapath strings across their namespaces never identify the same row. A family removal must cover all derived feature and TRAIN-label channels containing that family while retaining the movie-own feature channel and native residual/self semantics.

The generic source rule is `rho_supply*grad CE/BCE(full) − rho_absent*grad CE/BCE(probe)` with nonnegative coefficients. Two executed forwards do not establish two nonzero terms. Real-native qualification must report factual/probe gradient activity by actual site and, for grouped sites, removed versus retained namespace rows. If probe dependence vanishes, name the resulting positive weighted factual-CE rule instead of claiming a new signed mechanism. A source-supply scalar/guard can still differ from an ordinary weighted scalar loss.

Any advantage can be ordinary feature/semantic adapter capacity, contextual example weighting, graph-view augmentation or useful sharing placement. SeHGNN learns semantic fusion over precomputed graph/label features; these private factors are **not** learned per-edge sheaf geometry. BatchEnsemble/Rank-1 factors, semantic/meta-path attention and heterogeneous ensembles are established ancestry. No universal expressivity, novelty, causal source utility or useful complementary-error claim follows from this placement.
