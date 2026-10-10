# Disabled finite IMDB factual-mask readout recommendation

This is a source/custody recommendation, not an enabled release, numerical reader or scientific launch. Root may prepare a finite CPU-only reader on the authorized original allocation after reviewing this catalogue. No resident archive was opened during this audit. No number of available/lost alternatives is invented.

## Exact population and inputs

Study binding: `d0b30e8297bbedc6063a0bf0c553869157165759d8cd2a41e93e4fe4324d0ea4`. Pilot source seal: `4350d456c4f646738f0fc71268e7641c13dd4a97ef36c841262fbf2a872cd9d0`. Paired roles/base seeds `(1,1),(2,2),(3,3)`, original member order `[0,1,2,3]`, fixed label axis `[0,1,2,3,4]`. VALID has 274 rows per role, 1,370 Bernoulli events; roles may overlap and are not independent graph replicates. The actual node-ID arrays are not copied here.

Use all 24 completed factual banks: six shared arms (`shared_own_only`, `native_pool_credit`, `source_view_supervision`, `uncoupled_source_contrast`, `COMMON_cycle`, `assigned_source_supply`) and two genuine four-body references (`plain_native`, `untied_same_six_factors`), each on all three paired roles. This preserves the already completed comparison denominator. The main explanatory contrasts are candidate versus own-only and each of the two independent references. The single native models have no available fresh per-label prediction diagnostic; retain that gap, and do not add a forward to fill it.

`IMDB_MASK_READOUT_BINDINGS.json` contains every exact resident `SELECTED_VALID_DIAGNOSTICS.pt` path, bytes and SHA256 from the unchanged complete quality readout's selected origins, plus matching closed-origin inventory and selected-checkpoint metadata descriptors. Those are declared/previously verified source bindings; current resident bytes must be rechecked by the future admitted reader. It must bind original shared/reference fit releases, family reports, complete-cell inventory, original launch/terminal/absence custody and source/protocol. Do not load model checkpoints: the checkpoint descriptors establish origin, not reader inputs.

Role identity is fixed by the original bound role JSON digests:

| Role | Role descriptor SHA256 | Bytes |
|---|---|---:|
| 1 | `36c11db54f9b99198bfc682c62f288344f0c9858877839b9eb11eb5127c9ffa9` | 14081 |
| 2 | `1c2e76861504a4990cb56fb0e497ba1f304dab34ece648ed09113b14b3a7be9d` | 14081 |
| 3 | `4edc400cf72baa0e9d5ff6c5a4c5974d9c3dd4086b89f2a302f22908cc58645b` | 14081 |

Within each role, require exact ordered equality of stored `valid_ids` across all eight banks, and exact `paired_identity` agreement for study, role/base seed, role binding, full-view binding and input descriptors. Hash the ordered IDs in a stated format for compact output; export no IDs. Do not load dataset files, TRAIN/TEST arrays or new label files. Cross-role overlap may be reported from already stored IDs if separately authorized, but is not needed for this readout and must not inflate the replication count.

## Exact stored masks and semantics

The source requires complete finite four factual plus twelve ablated VALID outputs, then writes these **factual** fields (`selected_diagnostics.py:123–134`):

| Field | Expected dtype/shape | Meaning |
|---|---|---|
| `member_correct` | bool `[4,274,5]` | Each member's strict `(sigmoid(factual_logit)>0.5)` prediction equals stored Bernoulli truth. |
| `full_pool_correct` | bool `[274,5]` | The FP32 arithmetic mean of four full-input probabilities, thresholded strictly `>0.5`, equals truth. Exactly0.5 predicts negative. |
| `common_wrong_events` | bool `[274,5]` | Original `(~member_correct).all(dim=0)`. |
| `any_correct_member_events` | bool `[274,5]` | Original `member_correct.any(dim=0)`. |
| `valid_ids` | original ordered list, length274 | Factual cohort identity only; no transfer of IDs to Mac. |
| `member_order`, `paired_identity` | saved literal metadata | Member ordering and original source/role/full-view/input custody. |

These names and shapes are supported by source and the root-reported274-row schema plus complete report event totals. They have not been freshly checked against resident archives in this audit. A future reader must fail rather than silently reshape/reorder or drop a missing bank. Torch archive loading can materialize other saved fields; only these masks/identity metadata may participate in the calculation. Do not use saved probability arrays, `targets`, ablation masks, U/D arrays, logits or model weights to calculate new metrics.

Check that the two saved coverage masks agree exactly with their declared boolean reductions. This checks archive coherence, not source metric reproduction. Preserve source-selected states and native thresholds. No new scorer, calibrator, convex-mixture weights, threshold, selected epoch or oracle prediction is constructed.

## Only permitted new aggregates

For each original bank, role and label, export integer counts and corresponding 274-row/1,370-event denominators:

- Member-correct and member-wrong events; histogram of how many of the four members are correct (0–4).
- Stored common-wrong and any-correct event counts.
- Correct alternatives lost by the existing factual pool: `any_correct_member_events & ~full_pool_correct`.
- Pool correct despite all members wrong: `common_wrong_events & full_pool_correct`, retaining any nonzero value for investigation rather than repairing it.
- Unique correct events per member: `member_correct[m] & (member_correct.sum(dim=0)==1)`.
- Pool rescues/harms relative to each existing member: `full_pool_correct & ~member_correct[m]` and its converse.
- Paired candidate/reference factual pool repairs/harms on matching ordered cohort IDs, using only their stored `full_pool_correct` masks. These are event counts, not recomputed micro/macro-F1, BCE, accuracy, NLL or calibration scores.

Report all24 banks, all three roles, all five labels and the declared explanatory contrasts. Mask sums and repeated events are descriptive counts; members/labels/roles are not independent repetitions. A binary all-member-wrong event supplies no correct decision within unchanged convex probability mixing. The count of lost alternatives measures available oracle opportunity, not deployable routing performance or a causal effect of sharing. Do not combine label/role-specific counts into an invented new scientific quality criterion.

## Original route, closure and finite ownership

Resident artifacts are inside `/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930`, on literal authorized `anogena-2` allocation route, hostname `anogena-2-0`, sole UUID `GPU-44039938-fd82-41d2-fefd-de71514e2fac`. Follow the project's original route/custody checks before project operations; run the numerical reader with CUDA hidden and no scientific model/provider imports beyond the minimal archive/boolean processing runtime. Use an existing ordinary runtime and a reviewed finite CPU owner, fresh output and explicit external bound; no new installation, existing-job action or retry. Root determines the finite cap after source/runtime review; this note does not invent an ETA or measured peak.

The saved complete readout custody at 2026-10-09T19:02:09.522239Z cross-links the original shared18, reference24 and native-five launch/release/terminal identities and records their exact handles/groups/CUDA absent. It is historical evidence, not a fresh host observation. The new reader must require the approved complete original source/custody linkage; it should not reopen old jobs or claim a missing OS terminal on their behalf. Raw masks, labels, probability tensors and checkpoints remain on the resident host. Return only compact scalar-count JSON/CSV, source bindings and finite execution/closure/cost receipt to the Mac.

No original source, frozen score, checkpoint, canonical ledger/status or published comparison is modified. Whether the counted alternatives justify a later deployment-aligned risk study is a subsequent root decision; this plan admits no IMDB training cycle.
