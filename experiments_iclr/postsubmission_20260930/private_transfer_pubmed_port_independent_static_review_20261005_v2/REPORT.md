# Independent static re-review of the disabled Pubmed port, v2

5 October 2026. This is a focused source and protocol audit of the separately prepared v2 packet. The four P2 findings in the immutable v1 independent review are resolved at the source level in the bytes reviewed here. I found no additional concrete source defect in this focused re-review. Every real execution, numerical qualification and scientific admission remains absent or disabled.

The reviewed packet is `private_transfer_pubmed_disabled_port_preparation_20261005_v2`. Its MANIFEST.json SHA256 is `b1359b85bb3a9fa59f9377e304121d663a3808dc5d7959eca15ee7dc623dda9c`; its SOURCE_MANIFEST.json SHA256 is `ba897c6588c0ac5d7d2e8473e1582b34c67986c23f265f0588f61c55c747aa6f`. The prior immutable review REPORT.md SHA256 is `ac01e32b57d06a8ba805c42c176feff653bcd6937c14d8b4e0b187d28abc1684`.

I independently authenticated those supplied pins, all 39 v2 packet-manifest entries, all 22 v2 source-manifest entries, all 44 saved input bindings and all 15 declared exact copies. I also authenticated all 36 predecessor packet entries, its 21 source entries, all four prior-review manifest entries, their applicable seal pins, and six supplemental original30/companion9 protocol inputs. No mismatch was found. INPUT_MANIFEST.json records exact paths, roles, byte counts and hashes; SOURCE_MANIFEST.json is an exact copy of the reviewed v2 source manifest. CHECKS.json records independent static comparisons, including AST fingerprints.

The audit used local saved source/protocol and compact design-adoption text, stdlib JSON and hashes, AST parsing/comparison, and explicit source arithmetic. No target or numerical code was imported or run. No fixtures, sampler, numerical tests, feature/data/history/checkpoint/prediction/outcome payloads, network, server, MacLink or 18.77 access occurred. No agents were spawned. Only this new review directory was written; the preparation packets, prior review, bound sources, frozen protocols and canonical artifacts were preserved. This report is neither an acceptance verdict nor an execution release.

All source references below are relative to the reviewed v2 packet unless another directory is named. The report assesses the implementation itself rather than adopting the author's repair or verification declarations.

## Disposition of the four prior findings

| Prior finding | Source disposition | Implemented basis |
| --- | --- | --- |
| F1, P2: candidate FP32 receipt input/runtime binding | Resolved at source level | Receipt runtime and exact TRAIN/feature identities are compared with actual loader/runtime returns before model construction or updates. |
| F2, P2: native cohort cost/budget admission | Resolved at source level | Candidate and native fits call the same complete-comparison cost, selection-budget and horizon gate. |
| F3, P2: complete-four custody before selected payload access | Resolved at source level | An all-four metadata/hash pass verifies jobs, sources, inputs, outputs, final artifact inventories and successful owned terminals before any selected freeze is decoded. |
| F4, P2: original30 promotion versus descriptive39 analysis | Resolved at source level | Separate authority fields and protocol wording preserve the original30 decision and complete39 descriptive analysis without a companion/F1 threshold. |

### F1: the receipt gate now compares actual identities

`run.py:29–57` takes `versions` and `identities` explicitly. At lines45–49 it requires the receipt's `runtime` to equal both the actual runtime return and `job.runtime_versions`, and its `inputs` to equal the actual `train_pos.txt` and `gnn_feature` SHA256/byte identities. `custody.runtime`, lines168–186, independently compares the measured six numerical version fields with the job and requires a single visible CUDA device. `custody.load_inputs`, lines189–243, authenticates member bytes against the pinned acquisition manifest and retains the exact adopted TRAIN pin.

The ordering in `run.main`, lines108–116, is actual runtime, authenticated input loading, `admissible(job, versions, identities)`, then source/model construction. Thus the v1 counterexample of a receipt with the same source/sizes but different feature bytes or runtime no longer satisfies this implemented gate. No future receipt was opened by this audit. The gate does not itself create feature provenance, runtime qualification or the root approvals.

`qualify_training_step.py` remains byte-exact. Its source records the actual runtime/input identities in the receipt, and its discarded qualification still uses TRAIN plus features only. Restricting the receipt comparison to those two members accurately preserves that scope; it does not invent a VALID/pool qualification or seed grid. For an authorized future fit, input loading includes separately authorized VALID before this receipt comparison; the implemented guarantee here is the gate before candidate model construction and updates, not a claim that it precedes every input read.

### F2: both fit paths enforce the complete cohort admission

`custody.validate_cohort_admission`, lines33–48, requires the cohort's `fits_authorized`, `root_adopted_after_TRAIN_cost` and `TEST_closed` flags, approved selection-budget fairness and a paid-budget description, nonempty authenticated complete-cycle evidence, and nonempty byte-authenticated evidence in exactly eight comparison categories: candidate live/detached/ordinary complete cycles, native complete epoch, candidate/native complete VALID, native selected replay, and native four-member pooling. It additionally requires approved byte-authenticated horizon-practicality evidence.

`run.freeze_schedule`, line72, and `native_ncnc.validate_fit_plan`, line52, call that same helper before fitting. Candidate schedule admission occurs before numerical setup, as does native fit-plan admission. The v1 counterexample with root adoption enabled but empty costs, false budget fairness and a disabled plan now fails this shared helper. Native member identity, the fixed seed policy, the exact native selector/schedule, and the separate native job's fit/VALID scope checks remain intact.

These checks require exact evidence and explicit root approval; they do not interpret measurement contents, establish that categories are truthfully measured, equalize the two methods' exposure budgets, or infer a practical limit. Those responsibilities still require reviewed real evidence. That limitation is an outstanding execution prerequisite rather than the prior missing native admission check. The current proposal's cost lists are empty, horizon approval and budget fairness are false, and root adoption and fits remain false.

### F3: all-four semantic custody precedes selected quality decoding

`native_family_custody.py:19–99` implements a separate metadata/hash pass with no numerical imports. It authenticates the exact family inventory and requires it to be present in the root-approved evidence. It checks the complete four-member ordering and common source/input/runtime/base-seed identity, then checks every member before returning:

- Exact member/seed/fit/model identity and supplied freeze/state/score artifact records.
- Existing byte-authenticated artifacts in distinct exact member outputs, with the expected selected role names.
- Distinct byte-authenticated member jobs; exact source-manifest/program/runtime/input-manifest/output identities; native fit/VALID, reviewed source, bound and no-retry job scope.
- A COMPLETE metadata-only final custody record with exact source/input/runtime/member/job relationships, all six required fit output records, and the selected roles bound to that inventory.
- An authenticated metadata-only owned supervisor terminal with COMPLETE status, integer exit code0, no timeout, and exact fit/job/output/result/final-custody relationships.

The family, member, final-custody and terminal projections use exact allowed field sets and exclude selected quality fields. Selected artifact bytes are hashed for authentication but are not decoded in this pass. The audit itself opened no such artifacts. This is a substantive change from the v1 approved-reference check and member-at-a-time replay.

`native_pool_replay.py:25` calls the complete-four pass before the first selected freeze read at line28. Lines27–35 then check all four COMPLETE freeze identities and their selected-file bindings before numerical runtime/data setup at line38. The actual loader/runtime identities must match the preflight metadata at lines39–40. The first selected state/score deserialization is at lines54–55, after these two full-family checks. Per-member rehashes remain in the replay loop.

The native fitter now writes a metadata-only `FIT_ARTIFACT_CUSTODY.json` after its normal freeze (`native_ncnc.py:138–144`). That fitter-generated record supplies artifact custody; successful owned completion still comes from a separate supervisor terminal. `OWNED_NATIVE_FAMILY_CUSTODY_CONTRACT.json` explicitly marks actual evidence absent and the supervisor implementation unqualified, and defines the projection a future reviewed supervisor must truthfully produce. The recipe has false family approval, empty members/evidence and null family inventory. The source now enforces the prospective all-four relationships; this review supplies no real owned terminal/family evidence and no proof that an absent supervisor produces them correctly.

### F4: promotion and descriptive analysis have distinct roles

`COHORT_PROPOSAL_DISABLED.json` now contains `original30_unchanged_promotion_gate_required=true`, `complete39_descriptive_mechanism_analysis_required=true` and `companion9_acceptance_gate_added=false`. Its prerequisite description explicitly preserves the original quality/promotion and attribution decision, descriptive companion contrasts, unavailable diagnostics and the absence of a companion threshold. The old combined `original30plus9_quality_gate_required` key is absent from the active proposal.

The fit authorizer requires two separate evidence references, `original30_promotion` and `complete39_descriptive_analysis` (`custody.py:131–134`). Their job scopes make the same distinction. Neither reference is currently approved. Requiring completion/custody of descriptive analysis is not a numerical companion acceptance condition; no F1 result threshold or new quality criterion is implemented.

I checked this against the bound supplemental prospective mechanism plan at `shared_private_transfer_fixed30_plus9_mechanism_analysis_plan_20261005_v1/REPORT.md:80,84,114,120`, the original release's unchanged quality/attribution policy, and the companion plan's explicit absence of a new acceptance gate. The original release retains strict positive mean and at least two higher blocks for its original required quality contrasts, with separate attribution restrictions. The v2 port introduces no numeric replacement for that policy, no F1 promotion gate, and no diagnostics-based rescue.

## Numerical source and schedules preserved

All 22 v2 source files parsed as AST data. Exactly four predecessor source files changed: `custody.py`, `run.py`, `native_ncnc.py` and `native_pool_replay.py`. The new source is `native_family_custody.py`; the other 17 predecessor sources are byte-identical. All 15 declared saved-source exact copies also match independently.

The independent 29 AST comparisons match. They cover the candidate metric and full numerical/control body from `x.to(device)` through final output, the candidate failure handler, unchanged custody helpers including actual runtime/input loading and geometry/adjoint helpers, native setup/serve, the entire original native fit try-body preceding the new custody writer, native failure handling, the native replay tail from the first state load through restoration/replay/metrics/timeout, all pooling/output operations following that loop, six native state/RNG helpers, and the candidate/reference NCNC train functions after normalizing only the function name. CHECKS.json records both AST fingerprints for each comparison.

`models.py` is byte-identical to the Pubmed v1 model. Independently comparing it with `shared_backbone_private_transfer_training_source_20261005_v2/models.py` gives exactly the prescribed width3703→500 and the two outer-count assertions1409026→589058 changes. The candidate remains nonrecursive NCN, width256, one puregcn layer, input projection, JK=True, .4 input dropout, .3 encoder/head dropout, no edge dropout and the same unframed four private rows. The Adam, virtual/shared/recomputed-private operations, dropout-stream replay, objectives, geometry and committed raw-logit serving remain in the unchanged numerical source.

The native module and vendor bodies are byte-exact to their bound sources. The independent native member fit preserves complete source-native NCNC, its own normalized loss/optimizer, batch1024, 36 shuffled dropped-tail commits per epoch, maximum9999 epochs, VALID every5, eleven misses, no initial VALID serve and the first strict maximum of rounded4 complete MRR. Replay preserves the inherited128×float32-epsilon logit tolerance, exact per-query/rounded metrics, selected metric identity, full pre/post state and RNG checks, and mean raw-logit pooling with zero updates or ensemble reselection.

The v1 and v2 cohort `cells`, `native_fits`, native selection, candidate selection, served conditions/counts and pool-selection definitions compare equal. Candidate schedules remain60 complete cycles/every5/11 misses. The fixed seeds/factor seeds remain unchanged; native seeds remain `s,s+5,s+10,s+15`. Member0 is reused for the single. The proposal remains nine candidate fits plus twelve native fits, yielding21 unique fits and15 served comparisons if later adopted. Every job difference is confined to custody/documentary fields; numerical recipe fields are unchanged.

Source arithmetic remains consistent: encoder128,257; dense head bases/biases460,801; F4 shared589,058; one private row6,402; F4 private25,608; untied4 outer2,356,232. Native TRAIN drops812 rows per epoch. Candidate full coverage is589 episodes including the44-row tail. These are source calculations, not constructed-model measurements or runtime qualification.

## Present execution blockers and limits

These are intentionally missing future prerequisites, not new defects in the four repaired safeguards:

| Requirement | Present status |
| --- | --- |
| Real allocation input manifest and feature/VALID/pool authority | Available-manifest hashes remain null and feature/pool authority is unapproved. No data payload or provenance proof was inspected or acquired. |
| Geometry beyond compact adoption | Only saved seed0/cycle0 TRAIN geometry adoption is bound. It does not admit model training or establish other seed/cycle feasibility. The unchanged runner fails without redraw/skips/substitution. |
| Exact runtime/model and candidate FP32 qualification | Runtime and source-review approvals are false; a fresh exact-source/input/runtime FP32 receipt remains absent. Static equality cannot supply derivative/Adam/committed-state qualification. |
| Native factory/train parity and evaluator/selected replay/pooling integration | Pending on the allocation. The discarded native two-epoch recipe is accounting/cost engineering, not reference numerical parity or complete VALID/replay admission. |
| Complete costs, practical horizons and resource bounds | All required cost categories are empty, horizon/budget approvals are false, soft bounds are null and external hard bounds are unconfirmed. No time, RSS, CUDA/output or horizon result was inferred. |
| Reviewed owned supervisor and actual terminal/family evidence | No new supervisor/launch adapter is supplied or qualified. Real truthful metadata projections, completed owned jobs and enforceable custody/bounds remain absent. |
| Original scientific prerequisites and prospective release | Original30 promotion/attribution and complete39 descriptive-analysis references remain unapproved. The new comparison has no fit or scientific release. |

All six supplied recipes independently retain false source-review, fit and VALID flags, null available-manifest hashes and soft bounds, false external-hard-bound confirmation and unapproved authorities apart from the limited geometry adoption. Their program/source pins match the actual v2 bytes. The cohort itself remains disabled. The preparation seal's explicit no-execution/admission declarations are consistent with these source and recipe checks.

The source repair dispositions establish the specified guard/control-flow and protocol corrections only. They do not establish executable integration, numerical parity, evaluator correctness on actual values, practical cost, successful ownership, predictive performance, novelty, acceptance or fitting permission. No runnable invocation, data acquisition, canonical edit, publication or target execution was attempted.
