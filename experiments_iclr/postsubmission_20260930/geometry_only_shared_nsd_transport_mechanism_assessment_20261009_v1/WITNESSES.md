# Exact conditional witnesses for the geometry only NSD bank

9 October 2026. Static real-arithmetic derivations for the exact bias-free incidence-factor sites. No model, numerical fixture, data or checkpoint was executed. These statements do not promise floating byte identity, useful training, novelty or whole-network frame equivariance.

## Discrete map output row signs leave deployed transport unchanged

For an ordered incidence (v,e), the prepared bank computes

`F_(v,e) = reshape[tanh(diag(s) W diag(r) concat(H_v,H_u))]`.

Choose any fixed diagonal sign matrix `S=diag(σ_1,...,σ_d)`, with each σ_a in {−1,+1}. Replace the output factor corresponding to matrix entry (a,b) by `s'_(a,b)=σ_a s_(a,b)`, leaving r and every slow parameter unchanged. The signs are uniform across incidences within that layer. Since tanh is odd, `F'_(v,e)=S F_(v,e)` for every input.

For every paired incidence and every node degree block,

`F'_(v,e)^T F'_(u,e)=F_(v,e)^T F_(u,e)`,

`F'_(v,e)^T F'_(v,e)=F_(v,e)^T F_(v,e)`.

The original raw Laplacian inputs are therefore identical. The native augmentation, SVD power/cutoff, clamping and sparse assembly receive the same inputs. Under the same realized normalization jitter they give the same deployed operator. At evaluation the native jitter is zero. Applying an independent such sign choice at each layer leaves all node states and final probabilities unchanged by induction, while the restriction maps can differ substantially.

This is an edge-stalk basis symmetry, not a node-frame change: node features never move. A raw map or output-factor repulsion can therefore reward predictor-identical states. The all-negative choice supplies the simplest example, `F'=-F`.

This is a discrete representable symmetry. It proves no continuous tangent gauge, no arbitrary rotation representability and no equivariance of ELU, shared feature transforms or the shared head.

## Native augmentation permits a genuine change of transport strength

Consider an admissible conditional two-node, one-edge evaluation state with d=4, no added LP/HP dimensions, and both endpoint maps `F_(v,e)=a I_4`, for 0<a<1. This can be produced by the bias-free learner when the common endpoint input contains a constant coordinate: set the four diagonal preactivations to `atanh(a)` and the other twelve to zero. Other hidden coordinates can carry a nonconstant message value. This is a representability construction, not a claim about actual Tolokers states.

The source has degree blocks `D_v=a² I_4`, paired off-diagonal block `−a² I_4` and evaluation normalization `(D_v+I_4)^(-1/2)`. Its entry clamp is inactive here. Thus

`L(a) = [a²/(1+a²)] [[I_4,−I_4],[−I_4,I_4]]`.

Taking a=1/2 and b=3/4 gives strengths 1/5 and 9/25 and nonzero normalized-Laplacian eigenvalues 2/5 and 18/25, respectively. The second map is realized from the first with the same W/r and diagonal-output factor multiplier `atanh(3/4)/atanh(1/2)=log(7)/log(3)`. Off-diagonal output rows remain zero.

The spectra differ, so these deployed operators cannot be related by orthogonal node-block conjugation. On any nonconstant transformed value H with `[[I,−I],[−I,I]]H≠0`, their native pre-ELU messages differ. With a common incoming state and residual parameters, coordinatewise ELU is strictly increasing in real arithmetic, so the next states differ as well. A shared classifier that reads a changed class-margin direction can expose that difference; a classifier or later layers can also suppress it.

The “+I” is essential: unaugmented homogeneous normalization would cancel this common map amplitude in this example. Tanh output scaling occurs before tanh, so a generic output factor is not a post-tanh map multiplier. The example establishes available transport/action change, not a whole-model expressivity separation or a trained utility claim.

## Exact fixed state action and classifier visibility conditions

For a common finite incoming state H, let `T_l(H)=A_l H B_l^T` be the shared native left/right feature transform and L_m the actual normalized, clamped operator. The complete pre-ELU message changes exactly when

`(L_m−L_n) T_l(H) ≠ 0`.

The neighbor contribution changes exactly when the same expression using only off-node blocks is nonzero. A different operator can act identically on the current value subspace; a zero right feature map makes every message zero regardless of the incidence maps.

With common incoming H and shared epsilon residual, different total messages give different next states in exact real arithmetic by ELU injectivity. This statement does not extend automatically through later layers. At the shared binary head, two final states give different probabilities exactly when

`(w_1−w_0)^T vec(H_m−H_n) ≠ 0`.

Different hidden states, operators or maps can therefore remain classifier invisible. For later layers the actual member states may already differ, so a comparison must distinguish private learner changes from earlier geometry-induced state changes.
