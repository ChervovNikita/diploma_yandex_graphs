# Native Citeseer nonlinear preaggregation growth source v2

## Prepared result

This packet implements the proposed nonlinear message growth at the actual one-layer native encoder site. It contains executable qualification, complete-cycle cost and fixed 60-cycle pilot source. All jobs remain disabled. No model, dataset, checkpoint, score or native runtime was opened locally; no server command, installation, training or old-source edit occurred. Runtime results and cost are pending. Root adoption and independent source review precede execution.

## Native insertion and packing

The pinned native `models.py` constructs `GCN(3703,256,256,1,.3,True,False,-1,"puregcn",True,0.,xdropout=.4,taildropout=0.,noinputlin=False)`. The vendor `PureConv.forward` applies `norm*x`, `spmm_add(A,x)+x`, then `norm*x`, where `norm=(1+degree)^(-1/2)`. The support is the provider's symmetric, unweighted TRAIN adjacency with target edges removed and no stored selfloops. The self term is added exactly once by PureConv. The source explicitly rejects any changed layer, residual, tail, dropout, projection or JK geometry.

`growth.py` executes the literal native xemb, PH, zero-dropout tail, stack and JK sum once. It adds the same linear tail/JK applied to `P tanh(H V_m) B_m`. This equals insertion before the native linear tail/JK in real arithmetic. All four incoming `d x 2` matrices are concatenated so a complete shared-bank call performs one extra sparse aggregation on eight columns. An own inner route uses two columns. Each member's complete nonlinear NCN head then receives its own corrected node representation. There is no frozen side encoder or postdivergence cached backbone.

Every growth arm adds4096 trainable parameters. Shared arms have four rank2 branches; the capable complete native single has one rank8 branch. V is exactly zero and B has orthonormal rows at initialization. B remains trainable without a later orthogonality constraint. The graph filters only initialize output subspaces; they do not impose persistent graph-frequency specialization. Existing shared encoder/dense/BE/private-normalization/bias/beta leaves remain trainable. Warm lift reuses the existing pinned neutral-factor/bias arithmetic rather than changing old files.

## Fixed TRAIN calibration and initializers

The calibration is the first episode of complete cycle0, using the prospective base seed and existing endpoint/random streams. The full draw is independent of model outputs. Both endpoint and matched-random positive targets are removed from support; the loss uses the endpoint outer TRAIN positives and native TRAIN negatives. No VALID or TEST values enter calibration. Dropout is off. H is the native preaggregation feature projection and G is the upstream derivative at the PureConv output before the tail/JK.

For q0..3, compute the degree-three Bernstein band `Fq=binom(3,q)(I+P)^(3-q)(I-P)^q/8` using the same native P on the same mask, then `Kq=H^T P^T Fq G`. Symmetric support is checked, so P^T=P. The top2 right singular vectors of each CPU float64 K define B; P actions remain native FP32. The repeated-top2 unfiltered shared control repeats the top2 basis of `H^T P^T G`. A second mandatory unfiltered control partitions the same top8 right basis into four disjoint2-row blocks. The capable single uses the complete same top8 basis. The partition control removes the repeated-top2 confound between graph-error filtering and initial projector diversity/coverage. The independent native4 control uses the four corresponding graph bases. Qualification computes all bases, while each cost/fit pays only its own required filter and SVD work.

The fixed usable-rank cutoff is max(1e-12,1e-6*smax). All four graph projectors must have normalized pairwise Frobenius distance greater than1e-4; basis sign changes alone do not count. Failure stops without redraw, strength search, rank substitution or fallback. Graph rank2 and capable-single rank8 must both pass in all-condition qualification.

## Gates and executable phases

`gates.py` compares actual copied-native models on the actual fixed TRAIN calibration support/queries. Dropout-off logits must equal the warm native single exactly and equal the no-growth copied model exactly. Every pre-growth leaf is compared between growth and no-growth references on the same loss and replayed dropout. FP32 gradients must pass rtol1e-6/atol1e-7; literal equality is reported separately. This does not certify bitwise old-gradient equivalence in advance. Incoming V gradients must be finite and nonzero for active members, and B gradients must be exactly zero at V0. The deterministic complete bank and every dropout-on route are checked. Independent members are checked separately with disjoint ownership. Qualification takes no optimizer step.

The cost phase performs one entire61-episode native cycle for the requested arm. The fit phase performs60 complete cycles and12 complete VALID evaluations. Both require explicit optimizer-work authorization. Fit additionally requires the same-source successful qualification for that seed and the same-arm complete-cycle cost receipt. Failed work is retained, never retried or shortened. Existing provider custody authenticates the host, physical GPU, runtime, input authorities and unchanged dependencies; this packet adds its own source manifest and hashes. No downloads, server allocation or runtime installation are implemented.

## Controls, optimization and selection

The no-growth reference is the existing same-seed warm_identity pilot; this runner admits its no-update qualification but rejects duplicate fit/cost jobs. Five new arms across seeds0/1/2 produce15 prospective fit cells. All start from the root-bound exact terminal20 TRAIN-only common warm checkpoint, with fresh postwarm optimizers/dropout streams.

Shared arms retain the existing three ordinary Adam passes per episode: mean own inner losses, the fixed half pooled/half own outer BCE, then repeated inner losses with replayed masks and one stream advance. The native single and each independent member concatenate all four inner lists, then use their own outer BCE. Independent4 has four full native encoders/heads, separate Adam states, and seed+5member query/dropout streams; every member pays a complete cycle. No independent loss or gradient is pooled. This is a declared matched episode recipe, not an unchanged author-native cold training recipe.

Serving uses mean raw native logits, matching the retained Citeseer reference. First strict maximum complete pooled VALID MRR rounded4 is selected at cycles5,10,...60. The independent control retains the existing synchronous pooled selector: it has common warm copies and independent optimization, with a joint-selection adaptation. It is not described as independently initialized or independently selected ordinary native4. TEST has no loader path and stays closed until root freezes every arm/seed decision.

## Cost and stop rules

The runner records inclusive time, calibration/filter/SVD time, full cycle times, actual native encoder P call counts and column widths, Adam/episode counts, parameter counts, optimizer ownership, initial/final/selected states, full draws and masks, peak CUDA memory, peak Linux RSS and output bytes. The complete-cycle receipt includes all four independent member cycles. Bound common warm20 work must be charged from its own receipt. Fixed caps are prior envelopes, not measured new costs. Source acquisition, checkpoint/selection I/O and all member serving work must also be charged when comparing operating points.

Stop before full fit on failed source/runtime or forward/gradient/rank/projector gates. Close graph-filtered initializer utility if either the repeated-top2 or disjoint-top8-partition unfiltered control matches. Close quality utility on member competence or paid cost-quality failure. A capable rank8 native single explaining the result removes demonstrated ensemble need. An independent4 explanation leaves only the measured sharing tradeoff. No raw-gradient-norm benefit follows under Adam; the filtered K is a proxy, and the selected B need not maximize the actual unfiltered incoming gradient.

GradMax, nonlinear message, graph-filter and low-rank ancestry remain explicit. This implementation supplies no novelty certificate, no accuracy result, and no information-recovery theorem. Three optimization seeds on one fixed graph/split are descriptive replications, not independent graph experiments.

## Before-outcome v2 amendment

Root requested the unfiltered TOP8-PARTITION control before any runtime or scientific outcome. V1 remains sealed and unchanged. Each member receives a disjoint pair of rows from the same unfiltered top8 Stiefel basis used by the capable single. The control retains4096 added parameters, identical TRAIN H/G, warm base, nonlinear units, optimizer and exposure. No strength sweep was added. Qualification includes it and cost/fit phases charge its own unfiltered initializer and full cycles.
