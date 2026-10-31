# Native source contract

## Source identity and architecture

Use `yandex-research/graphpfn` commit `3b9b115490249cc777227c846babfb55f35bd8c4`, specifically the paper `bin/base/gnn.py` Model, `lib/graph/deep.py`, `lib/graph/data.py`, and the native parameter-group factory. The saved Tolokers2 tuning TOML is a search specification. Model construction fixes the already documented binary task to two classes; it must not call the native all-label class-count method.

The native input is `Linear(F,512) → Dropout → GELU → Linear(512,512)`. Its body has three residual pre-LayerNorm GAT blocks. Each GAT uses `input_linear`, source/destination additive projections, LeakyReLU slope 0.2, destination edge softmax, source-value weighted sum, and a two-linear FFN with native dropout. Its exact value packing is **[N,128,4]**, rather than [N,4,128]. The output is `LayerNorm → Linear(512,512) → GELU → Linear(512,2)`.

GAT requires no supplied edge weights. The full graph is constructed with released edges and the public feature-row node count, then `remove_self_loop → to_simple → to_bidirected`. Preserve DGL endpoint/eid order when capturing COO endpoints for parity. The saved config's self-loop flag is ignored by this provider; no loops are added. A changed loop policy requires a separately frozen variant.

## Public preprocessing

| Native step | Frozen behavior |
|---|---|
| Input casts | Public features float32; categories cast int32; constant columns dropped with pandas `nunique != 1` |
| Type split | Preserve metadata feature order; remove dropped columns; binary fraction columns append to categorical columns in source order |
| Category cleanup | Native categorical columns with a single unique value are removed; source assertions remain qualification gates |
| Numerical transform | Quantile-normal; seed 0; n_quantiles=max(min(N//30,1000),10); subsample 1e9; fit using all public transductive features with Gaussian jitter 1e-5; transform original features |
| Imputation | Native most-frequent imputation fit on all public transductive features after transform |
| Categories | Ordinal vocabulary fit on all public features; dense float32 one-hot, drop=if_binary, handle_unknown=ignore |
| Fractions | Config none or quantile-normal, following the same native transform/imputation implementation; binary fractions already moved to categories |
| Model input order | Numerical, then categorical, then fraction |
| Targets | Two-logit CE; compact nonmissing train/validation labels only; no target transformation |

The future public preprocessor needs a closed-input implementation and parity receipt. The raw native `GraphDataset.from_dir` opens all labels, and `GraphTask` class/metric routines inspect all role labels. Neither is admitted in a source worker. Future row checks must bind string node IDs to exact feature row positions and verify edges index that axis. The draft custodian's missing-token vocabulary is intentionally explicit and narrower than a possible pandas vocabulary; source-role parsing parity is unresolved. Stop on mismatch rather than silently changing data semantics.

## Closed native loop

The native training script evaluates and prints test metrics each step. `closed_native_loop.py` removes that access: full-graph float32 two-logit forward, train CE, ordinary AdamW, validation class1 probability AP, strict improvement, max 1,000 updates/patience 100. It returns best checkpoint and a fixed-last step 50 model/optimizer/RNG donor. Native model construction and `delu.random.seed(seed)` occur in the separately reviewed caller. Preprocessing seed remains0.

The source-qualified loop must be checked against the native validation metric on compact validation inputs. It intentionally aborts on nonfinite source loss/probabilities; it cannot reproduce native all-role invalid-prediction checks while test labels are closed. The baseline selects only post-update checkpoints, following the source; the conditional extension additionally allows initialized epoch0. These differences must remain disclosed.

## Boundary family and AD candidate

`native_gat_adapter.py` replaces `input.0` and `output.3` with `S × Linear(R × x,W,None) + B`. W is shared; R/S/B are private. Bias sits outside output scaling. Every route traverses the full member-dependent native body; K4 does not imply one graph/body computation. Only `stem.S` and `head.R` enter the initializer (512 coordinates each). All factors begin as ones and biases clone the donor; all R/S/B and native W/body train during continuation.

DGL2.4.0 backend `GSpMM`, `GSDDMM`, and `EdgeSoftmax` use legacy autograd functions with `forward(ctx,...)` and no declared `setup_context`/`jvp`. This source finding does not certify `torch.func` support. The new Torch COO candidate substitutes only additive edge scores, destination softmax and aggregation, retaining parameters, FFN and channel layout. Its detached destination softmax shift is algebraically harmless; its numerical AD compatibility is **unverified**.

Required checks use the complete admitted graph and exact inputs: DGL/COO output and gradient parity, native/K1/K4 identity and private-bias checks, populated-history one-step AdamW checks, then three JVP/VJP/FD directions with two fixed epsilons. The borrowed qualifier permits one of the two predeclared epsilons per direction to meet both centered-logit and CE tolerances. It permits no adaptive epsilon search. Unsupported scatter AD, nondeterministic primitives or resource failure stop dependent phases.

## New AdamW transport

At identical routes, with mean member CE and **new R/S frozen during the equivalence audit**, shared parameter gradients equal native gradients and each private-bias row gradient is native/K. Copy shared moments, values and scalar steps. Replicate private B, first moment/K, second and max-second/K², and epsilon/K. Preserve learning rate, betas, all native group options and **decoupled weight decay unchanged**. New factors inherit associated boundary-weight options with empty moment state.

This differs from old R17 coupled Adam's decay/K rule. R17 is unchanged. The native grid has weight decay0 and its bias/norm group is also0; the new rule still needs populated-history parity. Compare named gradients, parameters, appropriately rescaled moments, steps, aliases/group coverage, and post-step logits on disposable full-graph copies. No equivalence is claimed after factor updates or divergent routes.

## Borrowed initializer and AP adaptation

The R17 graph bands, degree 3 Bernstein expansion, cap .5, relative radius .01, Armijo 1e-4, six backtracking attempts and fallback operations are copied unchanged. Metadata checks compare the five borrowed function bodies and constants byte-for-byte. Only the backbone binding, width 512/two-logit contract and accompanying documentation change. R17 optimizer/driver sources are not reused as a native runtime adapter.

The initializer accepts steps using each member train CE and **CE of mean raw logits**. The proposed pilot selects and reports **AP of mean member probabilities**; softmax of mean logits is secondary at the same selected checkpoint. This explicit adaptation carries no AP guarantee. Record accepted alpha, every branch/trial, finite/fallback reasons, full operation costs and midpoint absence. Separate branch backtracking means a graph-vs-common contrast is a complete-operation contrast; it is not a same-alpha causal alignment test.
