# Native attention versus compensating BE scales

8 October2026. Static path assessment only. Pinned Polynormer source SHA9b4e533f46ae7f91a23a552f88bbb47a01359e224b53996cf1a68c10a865f6a8 was verified against retained bindings. The portable BE adapter and prior attention plan were read. No model/library import, scientific array, current outcome, server, fixture, fit or source/gate edit occurred.

## Verdict

**Private attention is not automatically a strict capacity gain.** Existing BE output scales can reproduce certain private attention changes, and a later inverse input scale can remove their value change in simple homogeneous chains. The native Polynormer path does not supply that general cancellation: the parallel affine branch, shared normalization/mixing and accumulated states intervene. This failure of one construction is not proof that the whole BE network cannot represent the same predictor through another parameter setting.

Keep private attention only as a conditional utility/learning-policy hypothesis. Its plausible benefit is direct score adjustment without requiring coordinated value compensation across the native paths. Neither useful attention diversity, member competence nor better accuracy follows from static algebra. The already inactive comparison and its admission conditions remain unchanged.

## The absorption that really works

Use column-vector notation for a member's projected value u=D_s W D_r x. In the standard additive GAT score convention, sender and receiver vectors a_k,a_q score an edge through LeakyReLU(a_k^T u_sender+a_q^T u_receiver). For an invertible diagonal T, replacing s byT s gives u'=T u. If the desired private score vectors satisfy

```
a_k,m = T a_k,   a_q,m = T a_q,
```

the shared scorer on u' produces the same edge scores/attention as the private scorer on u. Aggregation then produces g'=T g. A following factorized linear map can cancel this by replacing its input factor r_next withr_next/T; its outside bias stays unchanged. If ReLU intervenes, require positive T so ReLU(Tz)=TReLU(z). Dropout with the same realized mask commutes with diagonal scaling. LeakyReLU/softmax are not barriers when their scores already match.

For both score roles, the same diagonal must work. At a coordinate with nonzero shared coefficients, private/shared ratios must match between the two vectors. A zero shared coefficient cannot be scaled into a nonzero one. General independent private sender/receiver vectors need not satisfy this construction. Matching edge scores is sufficient rather than necessary: attention is unchanged by receiving-row score offsets, and uniform LeakyReLU branches can make some receiver-score changes disappear in softmax. Degenerate input spans can also weaken the conditions. No general noncontainment follows from a failed ratio test.

## What happens in the actual native path

The inspected WikiCS recipe has7 local layers,1 head, pre_ln=false and learned sigmoid beta. Polynormer constructs bias-free local GAT maps but ordinary biased h_lins/lins; all dense maps receive BE factors. LayerNorm affine parameters and beta remain shared.

| Site | Exact obstruction or surviving special case |
| --- | --- |
| BE affine map, adapter lines24–26 | The map isD_s W D_r x+b, with shared b **outside** s. A private output rescaling does not scale b. Scaling a parallel branch byT therefore needsT b=b or an explicitly compatible bias change. The direct bias-free GAT projection avoids this problem. |
| Parallel branch, native160 | GAT(x)+lins(x) is formed before the next linear input factor. Scaling only GAT givesT g+r, notT(g+r). Positive rescaling of the full sum would require the parallel branch to co-transform; its shared outside bias generally obstructs a member-dependentT. |
| ReLU/dropout, native159/161–162 | Positive diagonal scaling commutes with ReLU; negative coordinates generally do not. Matched dropout masks commute. Neither operation alone defeats the positive homogeneous construction. |
| Product and shared LayerNorm, native158–167 | The state is(1-beta)*LN(h*x)+beta*x. The product alone can be preserved by reciprocal h/x scales in suitable bias-free positive cases. That leaves the normalized branch unchanged while the raw x branch is scaled; it does not produceT times the full mixture. Alternatively, LN(Tz) is not generallyT LN(z). Shared LN affine parameters cannot absorb arbitrary different route scales. This is the main generic barrier. |
| Learned beta, native163–167 | Sigmoid of finite learned beta keeps both branches active. It is a shared parameter, so one route-specific diagonal cannot generally be accommodated by independently changing the mixing coefficient. Special zero outputs, identity scales or limiting branch dominance can retain equivalences/approximations. |
| Accumulated local state, native168 | Even if individual blocks could produceT_i x_i, the sumΣ_i T_i x_i does not generally equal oneT timesΣ_i x_i. A final input factor cannot unwind arbitrary layer-specific scales. One commonT across all blocks or special data/readout subspaces can still permit cancellation. |
| Local head, native175 | A BE input factor can undo a single diagonal scale of the complete x_local. It cannot generally undo the differently scaled accumulated terms above. |
| Global boundary, native172 | Shared LN(x_local) occurs before any global dense input factor. Arbitrary channel scaling changes its centering/variance; the next BE input scale cannot simply recover the original input. Uniform positive scaling has special epsilon-zero invariance, which is not arbitrary diagonal equivariance. |
| Global path, native57–81 | Sigmoid query/key maps, normalized kernel attention, LN and multiplication by(h+beta) do not form a generic positively homogeneous chain. If their entire input is first restored exactly, they are unchanged; otherwise further coordinated identities would need separate proof. |

pre_ln is inactive in this recipe. If enabled, its LayerNorm would introduce another normalization before the candidate following inverse dense input factors. No claim about its unused configuration is needed for the current path.

The LayerNorm statement is elementary: it centers and normalizes a vector, so coordinate-dependent scaling changes the mean and variance. With uniform positive c and zero epsilon, LN(cx) has the same normalized coordinates asLN(x), rather thanc times those coordinates; affine parameters remain part of the rule. At nonzero epsilon, even that scale invariance is conditional/approximate. Compensation at one interface therefore does not automatically propagate through the whole native predictor.

## Scientific consequence

Reject a claim that shared attention forces every route to use the same graph operator, or that private attention necessarily expands the complete model's function class. Existing BE factors already alter hidden states and native attention. Keep the narrower question: does direct private score ownership make useful alternative rankings easier to learn under the fixed native training recipe?

The proposed intervention changes gradient aggregation and Adam state as well as the forward parameter partition. It can help through conditioning/optimization even when a predictor has another BE representation. Conversely, private scores can learn only temperature changes, nuisances or damaging neighborhoods. The planned COMMON/ROUTE interaction, whole-population repairs/harms, mean/minimum member competence, pooled risk and strong reference comparisons remain decisive. No new experiment, implementation permission or gate follows from this assessment.

## Scope limits

The score algebra uses the standard GAT operation already described in saved priors and the native constructor; the installed PyG implementation was not reopened or executed. This is not a private-attention hook qualification. Native operation/bias/normalization placement and adapter ownership are directly supported by the pinned source. Full-network representability, actual learned zero/sign patterns, equivalence under all possible parameter changes, optimizer trajectory equality, numerical effects and quality are unestablished. No new paper/full-paper/code-retrieval credit is claimed.
