# Unapplied insertion plan

The reference sources are SHA-pinned in SOURCE_BINDINGS.json. The diff has
`P0_REFERENCE_ONLY` and `FUTURE_NEW_RELEASE_ONLY` paths and is not applied.
The helper is not imported by existing P0. No source outside this folder changes.

1. **C4 dense-message gate:** in
   `label_only_private_corrector_core_source_20261008_v1/core.py`,
   `LabelOnlyCorrectorCore._route_delta`, insert
   `message = self.query_value_gate(message, H[queries])` after original line240's
   `message.index_add_` and before line241's `self.output(message)`.
   This operates on 64D aggregated values, before the 10-class map. The existing
   post-readout finite and empty-context checks remain applicable.
2. **Control heads:** in
   `label_only_same_operation_controls_source_20261008_v1/controls.py`,
   `OrdinaryCorrectors._route_message`, insert after original line167's
   aggregation and before line168's finite/zero-message checks. Select the
   route's own `query_value_gate` in fully untied U4, otherwise the bank's shared
   gate. The existing `_route_delta` lines183–185 then reads out the gated 64D
   message. `_joint_delta` lines187–194 gates each head before concatenating
   to 256D, so P0's nonlinear joint readout in `posterior.py` lines10–32 receives
   the same conditional capability. There is no post-logit gate.
3. **Permanent erased C4 values:** replace only original core line235's embedding
   lookup with `lookup_permitted_label_embedding`. For this arm pass
   `visible_labels=None`, `visible_count=len(visible_positions)`,
   `erase_class_identity=True`, on every train/serve call. The helper returns
   `embedding.weight.mean(0,keepdim=True).expand(visible_count,-1)`.
   For identity-aware C4 pass the original visible TRAIN labels and False.
   The surrounding visible-ID `index_copy_`, source scale and all-neighbor
   aggregation remain unchanged. Do not mutate `train_labels`, targets,
   `_training_context`, TRAIN IDs or native training data.
4. **Registration before fresh optimization:** a future new stage must create
   the original fresh banks/readouts, then register all gate instances using
   `make_query_value_gate(torch=..., later_execution_authorized=True)` and move
   them to the bank's float32 device. C4/joint/one-path register
   `bank.query_value_gate`. U4 registers one independent module at
   `route['query_value_gate']` for every route. Set C4's
   `erase_value_class_identity` once from its arm specification; it is True only
   in C4_gate_identity_erased, with no epoch/train-eval switch. Rebuild C4/joint/
   one-path one-Adam owners over the final complete bank parameters. Rebuild U4's
   four Adams over their final respective route parameters. Preserve P0 Adam
   settings and optimizer snapshot facade. Constructors are fresh; do not import
   any old label-stage model/optimizer state.
5. **Keep the old-state gradient schedule:** C4's gate forward is recomputed for
   each route's separate CE/4 backward; never detach gamma or reuse a freed gate
   autograd graph. Joint single uses one backward over all four gated messages.
   U4 gates are private and each receives its route CE/4 gradient. Do not step
   any parameters until all required backwards are complete.
6. **Small future stage bookkeeping only:** declare five new arm names,
   expected learned counts and the existing respective Adam/backward counts
   `(1,1,4,1,1)` and `(4,1,4,1,4)`. Keep 1100 updates and the same ordered common-Q
   checks. Original `stage.py` lines261,308–340,398–450 contain the count/arm/work
   checks that a *new* release must update; none are patched here. Whole-bank
   parameter snapshots already enumerate all learned coordinates through
   `train.py` lines108–124 and P0 `stage.py` lines353–366. Gate parameters and
   their Adam moments must be included in coherent selected state. Do not use
   per-route native-local restore APIs for this staged experiment.

Before any later scientific release, perform only separately authorized
qualification appropriate to this new source: exact identity gate at zero init;
complete/no-duplicate optimizer ownership; no gradients into H/B; old-state
route accumulation; exact zero empty context/native fallback; identical Q and
support for all five arms; constant-token behavior in erased train and serving;
whole-bank gate/Adam snapshot completeness; and truthful added parameter/work
accounting. No such numerical qualification or launch occurs in this packet.

The helper and source diff are concrete preparation, not a complete runnable
trainer. Import paths, future release hashes and bank registration must be bound
in that later release; applying this diff alone is insufficient.
