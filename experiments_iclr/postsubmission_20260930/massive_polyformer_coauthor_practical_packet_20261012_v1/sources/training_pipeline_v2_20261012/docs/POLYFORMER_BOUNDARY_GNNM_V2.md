# Standardized PolyFormer boundary GNNM v2

This disabled packet retains the standardized v1 learning roster and adds a
same-backbone, source-based original GNNM adaptation and its matched controls.
The unchanged author CLI packet and standardized v1 remain immutable.

## Construction and learning

`polyformer_original_boundary_own` has four routes in one shared body.
`polyformer_boundary_equipped_M1_own` has one route with the same equipment,
initializer and optimizer. Its genuine I4 uses four separately paid bodies,
private constructors/dropout streams and four own selectors. Its singleton
uses the prospectively fixed body0 checkpoint without further training.

Pinned original `BEBlock` replaces the existing `lin1` and `lin3` dense maps.
There are no added maps and no all-map FactorLinear installer. Native attention,
FFN and `lin2` remain plain shared modules. Source R/S/B/Xavier law is retained:
first input R is Rademacher; later R/all S are1; private B is0; shared W is a
bias-free Xavier map. Native body construction followed by lin1 then lin3
replacement occurs inside one private CPU constructor stream. CPU/CUDA
dropout streams are separately owned by each route and checkpointed, along
with the author's otherwise unregistered attention bias.

Each update performs one forward/member, separate CE/M backward calls at the
old parameter state, then one grouped AdamW update. All decay is0, eps1e-8,
betas(.9,.999). The attention/ordinary LR pair and grouping are the declared
PolyFormer adaptation. M4 and I4 serve softmax of mean raw logits; fixed scalar
CAL remains after this pool. Older procedures retain mean probability serving.

This is a new-backbone/optimizer-grouping/RNG/roles/horizon/selector adaptation,
not an unchanged published-task or score reproduction. The original five
GNNM datasets are excluded. All fits retain the standardized full2000 update
horizon and own first SELECT accuracy maximum checkpoint. No scientific fit,
pilot, benchmark score, GO or phase release is authorized by this packet.

## Effective finite domain and counts

Both new procedures use40 effective configurations: five joint LR pairs ×
two widths × two depths × two dropout values. Both decay fields are locked0;
the duplicated old decay cells are deduplicated. K/q/multi/heads/FFN/dprate
retain source-scoped dataset values. There is no factor-seed search axis.

Across four datasets and22 prepared splits, the addition is640 search and330
comparative fits,8 selectors and198 exports/CAL/evaluations each:970 paid fits,
1572 records,1940000 updates,4256000 TRAIN route forwards and4256000 validation
route forwards. It adds66 genuine I4/fixed body0 reuse banks and594 serving
route views. Combined totals are174458 fits,185492 records,144 selectors,
3630 exports/CAL/evaluations each,1254 I4/body0 reuse banks,10758 serving views,
348916000 updates,1587296000 TRAIN and1018048000 validation route forwards.

The fresh compiler corrects prior unprefixed comparison labels to actual
prefixed exported methods and binds boundary M4/M1/I4 at every split/seed.
Every previous learning contract is preserved except runtime relocation;
existing custody is rehashed without decoding heldout targets. Resources
remain provisional and require root qualification and whole-master gates.

## Engineering checks

Small synthetic CPU checks cover exact source constructor parameters and two
map replacement counts, plain interior and live gradients, source training
function parameter-update equality, one grouped optimizer with each active
parameter owned once, unscaled M1 CE, outer RNG isolation/private stream
replay, own checkpoint restoration, and target-absent M4/M1/I4 export with
bitwise fixed body0 reuse. Numerical tolerances and setup checks do not imply
benchmark accuracy or scientific acceptance.
