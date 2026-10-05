# Same-operation endpoint-frame controls

5 October 2026. Source-only successor. The sealed endpoint-adapter predecessor is unchanged. The root reports the private-frame CPU fixture passed on an authorized allocation with CUDA hidden (3.470s inclusive, zero fits/steps); the earlier transport failure is preserved separately. These controls have not executed. This packet performs no numerical imports or tests, data/checkpoint access, fitting, server access, runner creation or QA escalation.

## Three controls

`controls.py` imports only standard-library helpers. Its factory accepts the exact caller-loaded prototype, graph_ops and predecessor adapter modules, checks their paths/bytes, and returns three classes:

| Factory key | Operation |
| --- | --- |
| `shared_frame_f4` | Four existing F4 target/context trajectories use one learned H initialized at e0 at the identical outer endpoint site. All BE factors and private decoder paths remain. |
| `same_four_frames_native_single` | One native encoder/context/target trajectory consumes `[p0;p1;p2;p3]` in one wider xijlin, where `pk=(H(vk)h_u)⊙(H(vk)h_v)`, with its own four learned vectors initialized at e0,e1,e2,e3. There is one output logit, not four predictions. |
| `independent_native_single_one_frame` | One native single plus one identical H. The constructor requires keyword `axis_index`; later bank members must explicitly receive0,1,2,3. H initialization consumes no RNG. |

The single controls remove trainable BE r/s parameters from the portable one-member decoder and replace member LayerNorm wrappers with ordinary affine LayerNorm. Existing W/bias draws are retained without another random constructor. This is the native mathematical single operator through the portable source; historical author constructor RNG or exact GPU arithmetic is not claimed. All parent module ordering and unused fixed-pt state are retained.

The four-frame single uses a d×4d pair-map weight `[W0,W1,W2,W3]`. Its inherited recursive `depth_zero` still supplies d features and uses W0; its outer call supplies4d features and uses all blocks. The same W0/bias serves both paths. This dimension dispatch preserves the original recursion and prevents silently adding frames or private trajectories to detached completion. The inherited forward has one member, so context, completion calls and target decoding occur once.

`depth_zero`, `completion_scores`, `decode`, Twin `forward`, graph masking, native TRAIN loss and raw-logit `serve` remain inherited. The only inherited outer-forward hook is `endpoint_product`. Use the pinned parent's unchanged `training_batch_loss` and `native_optimizer`; for the single M=1 the loss is the native positive mean plus negative mean, and serving returns its one raw score. No objective, gate, normalization coefficient or clipping is added. The strict finite/nonzero norm guard and exact one-reflection formula come from the predecessor.

## Initialization and capacity limits

At axis initialization all four reflected products equal the original product. Thus the initial4d feature vector has four duplicate blocks. The wider single copies the native W into W0 and initializes W1–W3 to zero without RNG draws. This preserves the native initial function in exact algebra, but gives frames1–3 zero direct outer cotangent initially; their map columns must first learn before those vectors receive target gradients. The duplicate initial feature blocks also give equal initial column-gradient directions. These are disclosed optimization restrictions, not evidence that the single is competent or a reason to remove it. W0 is jointly used by the outer map and detached native recursion; extra columns do not change recursive features.

The controls use the same reflection family, frame order and initial axes. Independently fitted controls learn their own vector values; they are not coupled to candidate outcomes or checkpoint donors. They do not claim exact full-model capacity equality: the single has richer pair input and one context path, whereas F4 retains four private nonlinear/context paths.

At the same native width64 and128 input features:

| Model | Total parameters | Active parameters |
| --- | ---: | ---: |
| Native single |38,147|33,922|
| Native single + one H |38,211|33,986|
| Four independent native singles + one H each |152,844|135,944|
| F4 + one common H |43,854|38,857|
| Candidate F4 + four private H |44,046|39,049|
| Native single + same four H feature frames |50,691|46,466|

The four-frame single adds3d² target-map weights and4d frame floats: `10d²+(features+24)d+3` total, `9d²+(features+22)d+2` active. It therefore exceeds candidate F4 by6,645 total and7,417 active parameters at width64. No width reduction is substituted for the requested same-four-frames control. Count all stored parameters, unused ptlin, optimizer state, wider target work and four reflected products; retain its benefit from only one encoder/context trajectory. Common-H F4 currently computes its common product at each existing member call; no free caching or speedup is claimed.

## Import/use contract and checks

Load pinned `graph_ops.py` as `graph_ops` before its `prototype.py`, then load the sealed predecessor `adapter.py` under a distinct caller-selected module name. Pass those three module objects to `make_control_classes`. Returned class construction is deferred; this packet imports no model runtime itself. A supplied single recipe must have `member_count=1`; common-frame F4 must have4. An independent bank must preserve separate native losses, RNG streams and selection, and explicitly pass `axis_index=m`. The parent's `copy_native_unit_member` assumes the old one-member BE/norm state shapes and is not a copy helper for these ordinary single wrappers. No donor/loading helper is added here.

Static review checks source syntax, inherited hook placement, ordinary single affine operators, four-frame concatenation, first-block recursion dispatch and absence of random calls in this source. The parent and predecessor bytes are pinned in `SPECIFICATION.json` and `SOURCE_MANIFEST.json`. No new CPU/GPU check or fitting runner is provided. Numerical function/gradient correspondence and competent single fitting remain unverified; exact algebra does not guarantee exact GPU reductions with a wider matrix.

These controls implement the correction identified in `quality_method_synthesis_20261005_v1/REPORT.md`: multiple feature frames must be separated from multiple predictive routes. The earlier two-frame dense single is not treated as this control. Orthogonal contexts, HousE/GoldE endpoint transforms and bilinear compatibility remain attributed prior; this packet adds no novelty claim or new proposed study.
