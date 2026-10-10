# TRAIN graph context generates private projector factors

## One attributed extension worth qualifying

**Learn each member's upstream projector modulation from local TRAIN-label
evidence along typed graph paths.** Keep the native graph predictor and ordinary
full-label supervision. Do not train independent teachers or add a voting rule.
The first screen is deterministic; stochastic latent inference is deferred.

The inspected IMDB evidence gives a concrete reason for this choice: the shared
bank has1059 wrong VALID decision events with no correct member, versus only3
lost correct alternatives. A learned selector has little opportunity there. The
independent reference retains3383 available correct events versus3051 for the
shared bank. This supports improving member decisions on the complete task; it
does not identify a hidden-collapse cause or prove a universal sharing limit.

The proposed increment is graph-specific **conditional parameter construction**:
a movie's labeled co-actor/co-director context, or analogous native typed paths,
changes how that member processes factual features **before** shared graph
propagation. It adds supervised relational evidence that the factual-feature
bank currently does not receive. It is not merely an embedding separation loss,
global class code, pooled-loss responsibility multiplier or post-training gate.

An ordinary same-information context model might explain every gain. That is a
decisive required control, not an objection to trying the extension. Novelty and
utility remain empirical questions; no superiority or acceptance is claimed.

## Published basis and limits

Rank-1 BNN already learns compact factors around one common weight bank without
training four teachers. Its experiments favor average per-component likelihood
over mixture likelihood; this supports retaining competent individual predictors
as an explicit comparison. It gives no theorem guaranteeing competent GNN members
or preservation of an independent bank's alternatives.

MPNP already builds predictors from partially labeled graph contexts. Its encoder
uses context labels and neighboring features; its decoder conditions on a global
latent. The extension below uses **query-local typed label fields to generate
output-side private factors inside an existing native projector**, rather than
one global context vector concatenated to decoder inputs. Context information,
conditional prediction and compact multiplicative factors are attributed prior
ingredients. No claim of inventing them is warranted.

The MPNP experiments also show that label propagation can dominate at high
context density and that ordinary GNNs can win on fixed-label tasks with little
functional variation. These are reasons to include strong context references.
Training subsets on one graph are not independent draws of labeling functions;
this proposal makes no Bayesian posterior/calibration or new-function guarantee.

## Exact proposed predictor

Nominate the **complete native IMDB task** that produced the audited failure,
subject to root's source and label-custody qualification. Labels/metrics/schema
keep their native definitions; their dimensions are not inferred from the event
counts. No movie, class, difficult-event subset or induced graph is selected.

1. Use public native edges and type identities. Enumerate all schema-compatible
   typed walks of length1 or2 that end at the supervised target type, for each
   source projector type. The root must freeze the exact list from the native
   schema. For an IMDB movie/actor/director schema these include actor→movie,
   director→movie, movie→actor→movie and movie→director→movie. These examples
   are not certification of the current loader. No favorable path is selected.
2. Root provides only TRAIN labels and a TRAIN identity mask. A context set C
   exposes those labels; every query label is absent from C. All graph features
   and edges remain factual, including unlabeled nodes.
3. For each pathρ, use its row-normalized typed transition operator Pρ through
   sequential sparse operations, without materializing a dense two-hop graph. Let
   M_C mark labeled context nodes and B_C contain their signed native label
   vectors (`2*y−1` for multilabel bits; one-hot-derived signed vectors for a
   categorical port). Compute labeled mass `nρ=Pρ M_C` and signed evidence
   `aρ=Pρ B_C`. Use `aρ/nρ` when nρ>0 and zero otherwise. Concatenate these
   class/label balances and nρ over the frozen paths into g_i(C). With no context
   evidence, g_i=0. Preserve observed-negative versus unknown distinctions via
   the explicit mass channel; missing labels are never encoded as negatives.
4. At an existing native type projector, keep the private input factor s_m,t and
   base output factor r_m,t. Generate only a hidden-width output modulation:

   `h_m,i = W_t(x_i * s_m,t) * [r_m,t + U_m,t tanh(H_m,t g_i(C))]`.

   This avoids generating a factor of raw feature-bank width. H has32 hidden
   coordinates, one prospectively fixed exploratory choice. H/U have no biases,
   so g_i=0 leaves the native factor path exactly unchanged. H is private and
   initialized independently by a fixed source-qualified linear initializer;
   U starts at zero. This preserves the factual point-factor predictor at step0
   while allowing distinct H-dependent U gradients. There is no guarantee those
   gradients or later decisions will differ usefully.
   The equation is schematic: native biases, activations and the exact factor
   placement must remain source-defined. Initialize the native body identically
   across controls before the extra modules; give H its own seeded initializer
   stream so module creation does not alter native weights or dropout streams.
5. Train W, existing factors and H/U with ordinary scalar supervised gradients.
   Shared banks average member losses; matched untied paths retain native
   unscaled per-path gradients. Do not detach a hidden branch or import pending
   initializer/attention outcomes. Do not add a latent KL, teacher or decoder.

The nρ channels are evidence mass, not independent sample counts or calibrated
confidence. Repeated anchors along typed walks remain dependent. The model can
learn different responses to same-type versus cross-type evidence; no homophily
or majority-label assumption is imposed.

## Target exclusion and matched serving

Freeze two complementary TRAIN halves C1/C2 prospectively, with every TRAIN node
in exactly one half and a custody audit of per-label support. In each epoch,
predict Q1=C1 using only C2 labels, and Q2=C2 using only C1 labels. Both complete
graph forwards contribute to one optimizer step; weight losses by query count
so every TRAIN target receives exactly one native label loss per epoch. Do not
score context nodes using their own exposed label. Small/rare-label context
support is an admission concern, not a reason to drop labels or tune the halves.

At validation and serving, evaluate both frozen context halves and average the
two context logits inside each member; then use the existing native bank's
member aggregation and decision rule. This keeps context density matched to
training and uses all TRAIN evidence across the two passes. It adds no fitted
pooling parameter. Root must freeze the exact native logit/probability convention
before implementation. Using all TRAIN labels in one serving context would be a
different, currently unadmitted predictor.

The ignored-context native controls use the same two query-half schedule,
optimizer-step count, full graphs, dropout policy, horizon and selector. This
controls the extra training passes. Serving cost is two factual passes per
member for the proposed predictor; count it explicitly. No speed or peak-memory
advantage is presumed from compact stored parameters.

## Decisive representative comparison

Use three prospectively fixed optimizer/context-role replicates on the whole
qualified development task. Exact seeds, protocol and meaningful quality floors
must be frozen by root before engineering or execution. All results, failures,
selection costs and harmed decisions remain in the family.

### First stage:12 bank records

| Condition | Purpose |
|---|---|
| Shared4, context ignored, matched paired-query schedule | Exact factual point-factor anchor |
| Shared4, local typed-context multiplicative factors | Proposed extension |
| Shared4, same local context through a private additive branch `h_base + V tanh(Hg)` | Same evidence, same generator parameter count and initialization; tests parameter construction versus ordinary context input |
| Genuine native independent4, matched paired-query schedule | Strong member competence and ensemble quality reference |

A capable ordinary native single may be the **predeclared member0** of the
independent bank only with exact constructor, optimizer, selector and schedule
identity; never select the best member.

Stop if the candidate fails whole-population native predictive quality and
likelihood, mean/worst member competence or useful correct-event coverage against
the frozen references. Root must specify the floors, uncertainty handling and
all-replicate rule. Improvement only over a weakened single, a setup check, a
hidden-distance statistic or selected old hard events is insufficient.

### Conditional second stage:9 bank records

- **Same-information capable single:** the local multiplicative context model
  with a128-coordinate generator, matching the total generator capacity of four
  32-coordinate members. Exact counts/source competence must be qualified.
- **Matched untied factor paths:** the same local conditional generators and
  label-context schedule, with genuine unscaled private learning.
- **Shared4 global context:** replace each type's local field with its
  predeclared global mean field, keeping generator dimensions/parameter counts.

If the additive bank matches, conditional factor generation adds no supported
benefit. If the capable context single matches, the gain is a graph-context
learner rather than an ensemble contribution. If global context matches, local
graph evidence is unnecessary. If untied paths win, the intended sharing benefit
is unsupported. These findings are attribution outcomes, not grounds to hide a
successful generic context result.

Assess whole native task metrics and proper label likelihood, member mean/worst
quality, any-correct event coverage, common all-wrong events, repairs and newly
introduced harms. For the measured IMDB setting, newly available competent
member decisions are the primary mechanism readout; lost-alternative pooling is
a secondary one. Node×label events and repeated roles are dependent and cannot
be treated as independent graph replicates. No old score is recalculated.

A complete positive development result warrants an independently frozen second
complete task/protocol and source-qualified stronger label-utilization/conditional
model comparisons. MPNP, GAMLP and existing graph-conditioned expert ancestry
must be credited; a weak Gaussian-one-hot MPNP port is not an adequate baseline.

## Go/no-go and cost

**Go for source and construction qualification; fits remain disabled.** There is
a specific input/parameter mechanism and a decisive same-information falsifier.
Qualification must check target exclusion, path schemas, zero-context identity,
gradient ownership, labels/metrics, initializer and exact parameter/runtime costs.
This is not a positive result or a cleared first-in-field claim.

Each complete bank epoch has two factual context passes per member and one
optimizer step. Typed label-field construction uses only public edges and TRAIN
targets; H/U add dense hidden-width work and temporary node-factor arrays.
Count both training passes, context preprocessing/storage, validation passes and
parameter memory. Native full horizons/clean validation selectors remain required.
No teacher acquisition is needed by the method. No additional GPU is requested
without measured source-qualified costs and a promising complete screen.
