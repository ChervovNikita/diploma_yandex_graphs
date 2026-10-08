# Disabled same-operation label-only controls

Source preparation only. No numerical dependency import, fixture, native capture,
training, runtime qualification or launch occurred. The reviewed candidate core
is unchanged. The one-path control's full native label-aware competence remains
to be established separately; this module does not certify it.

## Modes and ownership

| Mode | External native feature paths | Correctors and Adam | Final selection |
| --- | --- | --- | --- |
| `one_path` | One capable full-width native backbone | One ordinary full-map label embedding/Q/K/V/output, one private Adam | Its own corrected predictor selects one coherent native-plus-corrector state. |
| `multihead_single4` | One full-width native backbone | Four width64 label-attention heads, concatenation, one bias-free linear256-to-C joint readout; one Adam over all heads/readout | One corrected prediction selects a coherent complete native-plus-corrector state. |
| `shared_backbone_untied_correctors4` | One shared native trajectory and capture | Four fully untied correctors, four private Adam states | Mean-probability family selector; all four correctors and the one native model must share that selected epoch. This is a correction-map sharing control, not an independent GNN ensemble. |
| `ordinary_independent4` | Four separately initialized full-width native models with no learned parameter or optimizer sharing | One complete private corrector and Adam per native model | Each member independently selects its own corrected predictor and coherent native-plus-corrector state. Different selected epochs are allowed. No candidate-selected common epoch shortcut. |

Every correction path owns a class embedding `[C,64]`, bias-free Q/K `[64,F]`,
V `[64,64]` and output `[C,64]`. Its parameter count is
`128*F + 128*C + 4096`, excluding its external native model. All maps are ordinary
full matrices: there are no BE factors, shared learned corrector maps or shared
label embeddings. The multihead single omits individual head outputs and uses
one `[C,256]` joint matrix on concatenated label messages. Its rank is at most
`min(C,256)` and its total parameter count equals four ordinary full-map
correctors: `4*(128*F + 128*C + 4096)`. There is no joint nonlinear readout or
coefficient/width tuning. It produces one prediction rather than an ensemble of
four probability predictions. `one_path` is a width64 one-head matched operator;
it is not by itself a capable same-capacity ensemble falsifier. Full native
label-aware competence of both single controls still needs separate qualification.
Standard Linear/Embedding reset occurs under the reviewed
isolated CPU constructor RNG per declared route seed; output weight starts zero.
Each initial correction is therefore zero. No initialization novelty is claimed.

Corrector gradients stop at full external H and base logits. Native losses,
parameters, feature learning, native optimizer states and captures remain entirely
caller-owned. Four different capture IDs are a declared custody requirement;
this module cannot prove that an external trainer actually fitted independent
models. Passing four cloned captures from one model is not an ordinary4 control.

## Same label operation, explicit optimization difference

The reviewed core supplies common-mask generation, legal visible/target lookup,
fixed-role version guards, external capture validation and query-edge selection.
The operator uses feature-only Q/K scores, dot product divided by sqrt64, and a
stable denominator over **every incoming nonself edge record**, including nodes
with hidden or unavailable labels. Duplicate multiplicity/order is preserved.
Only `A minus Q` TRAIN labels supply embedding/V values; all other values are
zero. There are no raw-H values, biases, additional hops or post-aggregation
nonlinear maps. No visible-label neighborhood gives exactly zero correction.

For `n>=2`, the one common private mask draws `k=floor(n/2)` TRAIN queries before
the external mask-independent native forward. Every route excludes every Q label.
Training values use `(n-1)/(n-k)`. Heldout serving requires IDs disjoint from A,
uses all A labels at scale1 and accepts no heldout truths. Serving averages class
probabilities, including the one-path case. Pair comparison masks through the
same explicit mask seed, bound TRAIN order and draw schedule, not native RNG.
The conditional message first-moment identity does not make CE/probabilities or
gradients unbiased, and does not make shared queries independent observations.

`one_path`, `multihead_single4` and `ordinary_independent4` use **unscaled own CE** per optimizer,
lr .001, eps1e-8, weight decay0, default betas(.9,.999). The reported mean own CE
is not the backward scaling. `shared_backbone_untied_correctors4` uses **CE/4**
per private optimizer, retaining the candidate's scaling convention. All route
gradients are collected at unchanged old corrector parameters before any route
Adam step. The candidate uses one Adam and shared maps; the controls use one
Adam per full private map set (one Adam covers all four heads and their joint
readout for the multihead single). Finite Adam epsilon means unscaled CE and CE/4
are not byte-identical recipes. That full-reference confound is explicit;
no tiny numerical equality or accuracy claim is made.

## Factory and driver protocol

`make_controls` is keyword-only:

```python
make_controls(
    mode=..., nodes=N, feature_width=F, classes=C,
    edge_index=G, train_ids=A_ids, train_labels=A_labels,
    initializer_seeds=(...), mask_seed=...,
    context_identity={...}, backbone_ids=(...), selection_policy={...},
    device="cpu", core_root=None, later_execution_authorized=False,
)
```

The default refuses before torch or reviewed-core import. `initializer_seeds`
is a tuple of one or four distinct nonnegative integer seeds (four for either
four-route mode and the four-head single). `backbone_ids`
is a tuple of one nonempty ID for each single/shared-backbone mode, or four distinct IDs for
ordinary4. `context_identity` is a dictionary of lowercase64-character SHA256s:
`node_order_sha256`, `observed_edges_sha256`, `TRAIN_ids_sha256`,
`TRAIN_labels_sha256`, `context_policy_sha256`. Root must bind and verify their
actual identity against the candidate; this module validates the declarations,
not official data custody. The context policy includes the common exclusion,
denominator, value scale and paired mask schedule.

`selection_policy` requires a SHA256 `policy_sha256`, nonempty `metric_name`,
`role="VALID"`, `native_local_restore="each_own_native_local_selector"`,
`corrector_local_restore="same_own_native_selected_epoch"`, and
`end_local_streams="live_no_rewind"`. Its `final_selector` is
`mean_probability_family` for the shared-backbone four and
`each_own_corrected_predictor` for either single/full ordinary4. Root supplies the exact
external driver's metric/tie/cost policy in the bound policy, rather than this
source choosing it after outcomes. Label-context identity is common; ordinary4
selected native/corrector epochs are genuinely member-specific.

`make_shared_controls(**same_kwargs)` accepts one-path, four-head-single and shared-backbone modes
for a single-backbone driver. The result supports the reviewed protocol:
`draw_common_query_mask()`, `train_corrector_step(H, base_logits, mask)`,
`serve(H, base_logits, heldout_ids)`, `descriptor()`, `named_parameters()`,
`optimizer.state_dict()/load_state_dict()`, `_check_roles()`, `_finite_state()`,
and the live `mask_generator/steps/counters` fields. The optimizer facade stores
all private route Adam states and adds no optimizer over native parameters.
Current-view calls supply no selected-epoch proof. Native-own-local restoration
must copy all learned corrector parameters and all Adam states from that same
native epoch while leaving mask RNG/steps/counters live, as the external driver
requires. No new capture, restoration or native training loop is implemented.

The shared-backbone train/serve protocol additionally accepts keyword
`native_capture={backbone_id, source_parameter_epoch,
native_logical_steps_before_update, global_mode}`. Epoch and logical steps are
nonnegative integers and mode is boolean. It records caller-supplied provenance;
it never infers a source parameter epoch from `core.steps`. After native local
restoration, live logical steps may be100 while parameters/Adam came from epoch
e<=100; the first global capture can refer to that restored point e. The external
driver owns that fact and preserves the captured view-B forward's provenance.
The four-head single has `members=1`, `attention_heads=4`; its complete model
parameters and one Adam must be restored together by that external driver.

The general API accepts a list/tuple of capture dictionaries, each with `H`,
`base_logits`, `backbone_id`, and, for ordinary4, `native_epoch`. Train via
`train_corrector_step(captures, mask)` and serve via `serve(captures, heldout_ids)`.
Shared-backbone mode receives one capture; it reuses detached references for
four correctors rather than performing or claiming four native forwards.

For ordinary one/four-prediction routes, a separately bound selected-state
serving path uses `route_snapshot(member,
native_epoch=...)` returns that corrector's model/Adam state. `restore_route`
loads it with purpose `native_own_local_restore` or `final_selected_state`.
Final binding contains `backbone_id`, `native_epoch`, `selector_policy_sha256`,
and exact `native_checkpoint`/`corrector_checkpoint` bindings. Call
`serve(captures, ids, purpose="selected_final")` with each capture's matching
native checkpoint binding. It checks restored corrector parameter versions;
shared-backbone four additionally requires the same native state for all routes.
The module performs no checkpoint file loading or native restoration. The caller
must prove those captures came from the declared coherent native checkpoints.
Per-head snapshot/restore and selected-route serving are rejected for the
multihead single; its joint predictor needs one complete driver-owned snapshot.

Costs remain explicit: shared four pays one native path plus four correction
forward/backwards; ordinary4 pays four native paths plus four corrections.
Attempted/completed route, edge-score, label-presentation and Adam counters retain
partial failure work. Native preparation, capture, fitting, selection, snapshots
and memory costs remain outside this module. No selected-cell or exact-resume
readiness, runtime savings or scientific admission follows from these source checks.
