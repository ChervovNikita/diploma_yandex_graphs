# Persistent masked contexts for a shared graph ensemble

The prototype tests whether different persistent missing-feature inputs teach
useful member decisions. Every member classifies every TRAIN label on the
original graph and, when applicable, on its assigned masked input. The auxiliary
predictor receives zero features for its quarter of eligible nodes. One common
linear decoder reconstructs those nodes' input features. Serving uses the
original factual input and mean class probabilities.

This is a source proposal. No model, data, checkpoint, GPU, fit, validation or
test score was executed or opened to prepare it. Scientific mode always refuses.
Root must freeze a population/protocol, review this source, qualify the complete
inputs/runtime and supply a separate engineering release. A successful1–3-step
qualification would establish engineering readiness only.

## Capable native provider and pending population amendment

The retained PolyFormer author source contains a PubMed loader, published launch
configuration and full training protocol. The monomial configuration is width256,
2 blocks, 3 polynomial tokens, 8 attention heads, FFN32, q1.6 and expansion2. It
uses native dropout.5 and attention/FFN dropout.8, LayerNorm, Adam base
lr.005/weight decay.001 and attention lr.0005/weight decay1e-8. The author
training inherits maximum2000 epochs and patience250.

The author PubMed protocol overwrites the official masks with class-balanced
60/20/20 random splits. Thus the original official-Planetoid-split proposal remains
inactive. A complete PubMed graph with the exact native split generation is a
concrete possible amendment. Both author monomial and Chebyshev launch rows are
documented in `NATIVE_CONFIGURATION.md`; this source implements monomial only,
chosen before new outcomes. It has not established capable learning under either
split. Root must choose and freeze the split seed, horizon/early-stop semantics,
selection, source/runtime and reference competence without outcome shopping.

The official loader applies PyG `NormalizeFeatures`. The prototype's decoder
targets that normalized input X, the raw input relative to hidden representations.
It does not target an unavailable unnormalized file. Root must freeze this target
meaning explicitly. All native polynomial tokens are recomputed from each masked
X. Reusing the factual polynomial feature bank would leak the node's missing
features through higher-hop tokens and is forbidden here.

## Exact method and gradient ownership

Four balanced quarters are assigned by a label-free SHA256 ordering of nonzero
feature node IDs with seed190101. Persistent ownership uses route m's quarter.
The shuffled control draws a uniform permutation of those four quarters at each
update from a separate stream. All nodes remain in the predictor, and all TRAIN
labels receive both factual and masked cross entropy. Zero-feature nodes remain
classification nodes but are not reconstruction anchors.

The fixed shared objective is mean factual CE +.5 mean masked CE +.1 mean CORE.
At temperature.2, each masked reconstruction z_i is paired with its frozen input
X_i and contrasted against32 OTHER reconstructions z_k from the same masked
quarter, drawn uniformly without replacement. The negative is reconstructed z_k,
not raw X_k. Both anchor and negative reconstructions receive gradients. Vectors
use norm floor1e-8. At least33 eligible anchors per quarter are mandatory.

Every native dense map uses the existing internal BE factor helper, with unit
factors installed after native reset. Shared W/b, shared LayerNorm/attention
parameters and private r/s remain trainable. The single linear decoder is live
and shared across auxiliary routes. CORE reaches upstream shared/private weights
and the decoder; it does not reach the classifier map after the captured hidden
representation. The classifier receives all factual and masked TRAIN CE terms.
There is no teacher, gradient filter, stop-gradient representation, projector
repulsion, target input skip, node-ID embedding or factual hidden cache reuse.

To bound graph tapes, route contributions to this ordinary scalar objective are
backpropagated at the same old parameter state, before any optimizer steps. This
is sum-rule gradient accumulation, not selective block routing. Shared loss is
averaged over routes. Untied losses are summed so each native body's gradient
retains ordinary unscaled supervision. In the auxiliary untied control, the
single shared decoder receives the mean reconstruction data gradient and one base Adam step;
this couples those bodies and is stated explicitly. Body gradients remain unscaled.
The decoder normalization is applied after old-state backwards and before its Adam step.

LayerNorm has no running population state; BatchNorm is rejected. Dropout is
native. Factual and masked streams are separate and persist by member (and by
quarter for the all-view single). Mask permutations, negative draws and native
construction have separate seeded streams. Source AST loading preserves exact
class bodies while omitting author imports, module-level seeding and main loops.
Only the target CUDA RNG state is set, and outer Python/NumPy/Torch RNG is restored
around constructor and dropout calls.

## Nine conditions and staged comparison

| Condition | Factual forwards/update | Masked forwards/update | Predictor bodies | Decoder |
|---|---:|---:|---:|---|
| shared4_own | 4 | 0 | one shared BE4 | none |
| shared4_masked_ce | 4 | 4 | one shared BE4 | none |
| shared4_core | 4 | 4 | one shared BE4 | one common |
| shared4_core_shuffled | 4 | 4 | one shared BE4 | one common |
| shared4_core_rewired | 4 | 4 | one shared BE4 | one common |
| single_native | 1 | 0 | one native | none |
| independent4_native | 4 | 0 | four genuine independent | none |
| single_four_view_core | 1 | 4 | one native | one common |
| untied4_shared_decoder_core | 4 | 4 | four untied, auxiliary-coupled | one common |

The all-view single receives one factual CE +.5 mean of all four masked CE terms
+.1 mean of all four CORE terms. One masked view of one independent member
does not provide the same exposure. The untied auxiliary condition receives the
same persistent quarter per member as the candidate, all TRAIN labels and the
one common decoder. Its name never implies independent auxiliary training.

Stage1 is shared4_own, shared4_core and independent4_native across seeds
9101/9203/9307, giving9 complete records. Root must freeze whole-population
quality/member safeguards, finish all9 and preserve every failure. Failed
quality or member competence stops the hypothesis honestly. Only a passing
predeclared screen enables the remaining6 conditions ×3 seeds, adding18 records.
The positive complete family is27 records, not9 survivors. Source construction
does not select any scores or fill any records.

The rewired control uses a separately frozen label-free double-edge-swap graph,
with exact swap count/rejection cap/seed/fingerprint. Source verifies each node's
degree, node/feature order, all original self edges and simple undirected
representation. It changes only the auxiliary polynomial bank. This source
generation does not invent the still-pending swap recipe or silently substitute
a failed graph. Conditional local-neighbor evidence requires this control.

Four private decoders with independent auxiliary bodies are a separately
disabled possible extra3-record confirmation if the core family passes and
shared-decoder attribution remains material. That variant is not implemented
or counted in the27 records. Importing independent native local-scorer starts
would likewise change initialization/source and needs a separate frozen
initializer × masked-context family. Neither ingredient is established positive
or novel by this source.

## Finite engineering interface

`python qualify.py --mode describe` and `python qualify.py --help` import only
stdlib. `--mode science` always exits before numerical imports. Engineering
requires an exact separate enabled release modeled on
`ENGINEERING_RELEASE_TEMPLATE_DISABLED.json`; it accepts one fixed condition,
one declared seed and1–3 updates on the full19717-node/500-feature graph.

The TRAIN NPZ must contain exactly X, edge_index, train_ids and train_y; no
validation/test masks, targets or other labels are loaded. The caller release
binds complete role custody, exact bytes, provider versions, chosen official/native
split, all graph dimensions, fresh resources and external hard supervision.
This file has no process owner, queue, data downloader, installer or retry.
Root reuses an existing ordinary runtime and owner. Qualifier outputs only
TRAIN losses, finite-gradient/state/serving checks, shape/call counts, inclusive
timings and observed CUDA peaks. It saves no predictions or checkpoints.

The first CORE engineering update separately audits whether its auxiliary
gradient reaches shared/upstream weights, live factors and decoder while the
classifier stays outside that reconstruction path. That audit does not supply
the optimizer gradients; the ordinary objective's backwards supply them.

## Resource forecast and limits

`RESOURCE_FORECAST.json` is symbolic. The native body has2,069,875 parameters;
shared BE4 adds55,772 factors; the common decoder adds128,500 parameters during
training and is discarded at serving. Five native3-token feature banks occupy
591,510,000 bytes before transient preprocessing, stacks and allocator costs.
The CORE chunk gathers512×32×500 entries, but autograd can retain all chunks;
the full route gather footprint, not only one chunk, is included in the ledger.

The prototype retains at most a route's factual and masked graph tapes before
that route backward. It still performs4 factual +4 masked fullgraph forwards
for shared/untied auxiliary methods. Parameter sharing does not eliminate those
passes. Inclusive timing must add preparation, validation/selection, serving,
snapshot storage and transfers. No wall-clock ETA, trained quality or qualified
memory cap is invented.

`NATIVE_CONFIGURATION.md` lists missing science bindings. Static checks verify
source structure and literal partition/negative-identity invariants without
Torch, a model or data. They do not establish numerical/native parity,
convergence, representation usefulness, novelty or acceptance.

## Corrected V2 source generation

V1 remains immutable. The fresh source audit identified a decoder-normalization
confound, serving-mode restoration and missing split/rewire custody checks.
V2 averages only the untied common decoder data gradient before its Adam step,
preserves untied native-body gradients, and restores every predictor module mode
in a finally block. Engineering reports the exact release digest/split identity/
seed and verifies all four TRAIN-only array fingerprints against a bound exporter
custody file. Rewired qualification requires legal frozen recipe values, bound
generator/custody and an exact auxiliary edge-array fingerprint. No held-out
labels are loaded to perform these checks. Numerical imports are included in the
worker timing, while admission/process startup stays in outer-owner accounting.
The masked rule, coefficients, seeds, predictor, original streams and9→27 roster
are unchanged. No new scientific mode, fit, dataset or numerical evidence is added.
