# Final selected-state scoring is a remaining campaign implementation gap

## Conclusion and audit scope

The 6,460-fit proposal does **not yet implement a complete final TEST stage
for its 27 presets**. Its runners export selected VALID predictions; the
reporter explicitly rejects TEST rows. This can establish training and
validation comparisons, but cannot establish the paper's heldout metric
reproduction or the final native/GNNM/I4 comparison.

This note inspects local source and documents only. No dataset, checkpoint,
prediction tensor or TEST labels were opened, and no native code or remote
command was executed. Existing frozen sources/manifests remain unchanged.

The adopted inventory is eight PolyFormer presets, sixteen OGB molecular
presets and three Collab link presets. The older 31/37-cell source-audit roster
is not the adopted campaign. Original-paper five-dataset runs stay excluded.

## Reusable components and precise limits

| Component | Reuse | Missing for this campaign |
| --- | --- | --- |
| `runtime_v4/run_fit.py:run` and `runtime_link_v9/run_fit.py:run`, carried into `portable_dispatcher_v3` | Own `selected.pt`, exact job/runtime binding, selector epoch, full adapter/session state, RNG and retained costs. Finalization already restores the selected state. | Finalization calls `adapter.validate()` and exports `own-selected-VALID-logits-v1` only. It does not serve TEST. |
| `runtime_v4/adapters.py:BaseAdapter.load_state`, `NodeAdapter`, `OGBAdapter`; v9 link adapter/extension factories | Reconstruct and restore the actual native backbone, wrapper, extension and private state using the job's exact source chain. | There is no common TEST role interface preserving every declared extension's actual readout. A new scorer must use the exact job factory, not silently instantiate a base mean-only GNNM. |
| `reference_wrapper_redesign_20261013_v1/smoke_20261014/node_port.py` | Native normalized transductive graph, polynomial basis, literal source split, model and route forwards. | `validate()` indexes `val_mask`; no TEST evaluator. A successor must index that same split's `test_mask`, preserving graph/basis/features and the actual method readout. |
| `.../smoke_20261014/ogb_port/session.py` and `native_sources/ogb/main_pyg.py:eval` | Full categorical features, fixed GCN/GIN/virtual-node model, native batch32 evaluation/skip rule, observed-label semantics and OGB evaluator. | `_RoleView` explicitly allows TRAIN/VALID only; TEST indices/loaders and selected-state role serving are absent. The successor must use official scaffold TEST, stable graph IDs, the native singleton skip rule and complete native ROC-AUC evaluation. |
| `final_test_serving_v1/link/backend.py` | Concrete prepared NCN/NCNC/BUDDY role-specific inference; own selected state/I4 rules; complete-query OGB Hits@50; BUDDY fresh role cache and original author inference. | Source/static checks only, with actual VALID reconstruction and resource qualification pending. `_open_selected()` requires the old exact `link_native_integration_v1/session.py` callback. It does not dispatch the compiler4/v9 extension callback/source chain or guarantee each learned readout, context method and private-state wrapper. It is not ready for all current link methods unchanged. |
| `runtime_v2/aggregate_selected.py:aggregate`, carried into portable I4 | Exact body0..3 identities, alignment checks and mean of four own selected raw-logit exports. Matched M1 is exactly body0. | It accepts VALID exports only. A role-aware successor must aggregate the four already served TEST exports, without choosing a common epoch or fitting new weights. |
| `benchmark_reporting_implementation_v1/reporter.py` and integration collector/observer | Frozen IDs, source/matched separation, complete-cohort missingness, costs and prescribed descriptive paired summaries. | Input normalization and scalar sidecars are VALID-specific. TEST needs separately typed role/population/topology/selected-state evidence; existing outputs must not be relabeled. |

Older `training_pipeline_v2_20261012/adapters/*_endpoints.py:evaluate_bank`
contains metric/alignment examples, but is not a ready replacement. Node and
molecular paths require the superseded CAL/export contracts; the link path
declares `support_policy="full_TRAIN_only"` and
`author_protocol_replication=False`. Reusing it directly would change the
current source protocol.

## Source-specific serving requirements

- **PolyFormer:** retain each of the ten declared literal split seeds and its
  own mask; score full native TEST nodes on the same transductive graph and
  polynomial basis. No new split, reduced population or feature/edge change.
- **OGB molecules:** preserve official scaffold roles, full features, native
  model/BatchNorm and missing-label evaluator semantics. Restore all method
  state and serve the declared actual readout. Fixed dropout0/0.5 presets
  remain separate; TEST cannot choose which preset is called the paper result.
- **NCN/NCNC Collab:** VALID uses TRAIN adjacency; source TEST uses the
  unweighted TRAIN-plus-positive-VALID adjacency with the source coalesce and
  symmetrization order. Recompute role embeddings and preserve original query
  batching, NCNC recursive completion and task-level output-factor scope.
- **BUDDY Collab:** source TEST uses year>=2007 weighted TRAIN plus weighted
  VALID edges. Build the original role-specific HashDataset, hashes, SIGN
  features and structural/query features from that topology. TRAIN/VALID
  cache reuse is not a TEST substitute. Preserve author inference order and
  complete positive/negative queries, then native Hits@50.

The link topology/inference source audit is already concrete in
`final_test_serving_v1/link/SOURCE_AUDIT.md`; its `GAPS.json` accurately records
remaining execution/wiring work. It is reusable preparation, not numerical
qualification or a final score.

## One reviewed campaign package

Prepare one successor release containing **training plus deterministic frozen
final scoring**, before presenting the giant campaign for the user's GO:

1. Add a finite final-serving roster for all declared source-reference fits,
   matched native bodies and shared methods. Bind the source/data/evaluator,
   native TEST topology, actual readout, output population, checkpoint-selection
   rule, costs and missing/failure policy before results. Implement three
   role-serving backends using the existing native source chains; no new
   training framework is needed.
2. Keep the current TRAIN/VALID fit controllers and original paper scores
   unchanged. After fit termination, derive an immutable selection manifest
   from the recorded VALID selector and exact own checkpoint/runtime receipts.
   Restore all extensions/private state through the same factory and check
   selected VALID reconstruction before serving. This is an engineering
   consistency check, not a new scientific accuracy experiment.
3. Execute the predetermined final role once per eligible selected fit. No
   TRAIN updates, checkpoint/model/seed/readout choice, temperature fit,
   test-driven preset choice or retry search. Preserve failures, capped or
   no-selected-state cases explicitly; never substitute final/recovery weights.
   Retain prior TEST exposure history rather than claiming all tests are new.
4. Serve the four independent native bodies once each; reuse those exports
   for I4. This adds **zero training fits**: at most6,460 own-state serving jobs
   and270 I4 aggregation jobs. The same6,190 reporting identities can receive
   final-role rows when eligible, including270 clearly labeled same-bank mean
   diagnostics obtained from existing served bank outputs. No diagnostic may
   reselect a bank or replace its actual learned readout.
5. Extend scalar/export reporting with explicit TEST role, complete population,
   topology, request, selected checkpoint/epoch, evaluator and method readout
   receipts. Retain source and matched tracks, all prescribed paired contrasts,
   missing outcomes and actual inference/cache/export costs. VALID and TEST
   must remain distinct tables/evidence namespaces.

The existing link prototype expects a separate exact-request authorization
envelope. For a single reviewed campaign, adapt this in a successor to verify
the real campaign GO and a deterministic derived selection/request binding.
Do not invent a new human approval or claim a machine-derived artifact is a
fresh human action. **This note adds no new permission requirement:** the
giant campaign still awaits its already required user verification and GO;
the proposed review should cover both training and its frozen final scoring.
Existing host/source/resource checks still apply.

## Limits on reproduction claims

Final TEST scoring is necessary, but does not erase the source caveats already
recorded in `source_protocols/PAPER_SETTINGS.md` and the final reporting
protocol. PolyFormer's advancing source RNG and independent per-fit seed law
are not identical; the OGB paper's winning dropout is unspecified and its
repeat seed law is not published; omitted source evaluation/constructor RNG
effects and the BUDDY inference audit correction remain disclosed. Keep the
270 fixed-horizon source-reference results separate from the matched extended
controllers. Report paper-metric comparison under the declared adaptation;
claim exact reproduction only where the complete source protocol actually
matches. VALID scores alone must never be advertised as reproduced TEST scores.

**Readiness:** current TRAIN/VALID preparation can be reviewed as such. A
complete paper-metric campaign is not implementation-ready until this finite
final-serving stage and its evidence wiring are included. This note proposes
the missing work; it does not claim that work has been implemented or run.
