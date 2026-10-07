# Independent static roles and runtime review

Reviewed immutable source packet `learnable_internal_be_contrastive_multitask_suite_20261007_v1`, MANIFEST SHA256 `3c0c88031dfc30178f0942794e64f4ae3677d3a2def1f2035adaa9e745474211`. Scope is source preparation for the fixed pilot; this review gives no scientific acceptance, learning result, or whole-goal completion verdict. The packet remains unchanged.

Recommendation: retain v1 as disabled preparation and require a separately sealed, independently reviewed successor before any official fit release. Concrete source defects remain in the coupled untied WikiCS transition, finite evaluation checks, and launch cost/failure boundary. Official role exports and full resource/supervisor bindings are separately missing release work. The implemented adapters and source wrappers have substantial engineering content; their existence does not certify official data custody or runnable resource readiness. `REVIEW.json` deliberately has `approved: false` for the reviewed manifest.

## Evidence and review boundaries

I read the actual protocol, all nine Python files, fixed configs/disabled jobs, native dependency contract, runtime pin, and CPU/static/staging/handoff receipts. Local stdlib hashing confirms all 19 source files have their sealed hashes and sizes, and all nine Python files parse. The three named native source files also match their dependency hashes. I inspected native Polynormer, NCN GCN/DropAdj/CNLinkPredictor, and sparse-overlap source. The evaluator/negative-sampling provider paths in RUNTIME_PIN are remote runtime paths unavailable locally; I assessed their source pin contract and the actual calls without accessing the remote allocation.

I imported no frameworks or models, opened no dataset/label/checkpoint/logit payload, used no GPU, made no remote calls, and reran no numerical tests. This review writes only its own report directory. BINDINGS.json records exact identities; READ_SCOPES.json records scope.

The existing CPU_CHECK is tied to the exact reviewed manifest. Its 28 rows are 24 tiny arm/task checks plus four unit/native mode cases. `check_cpu.py:25–49` uses tiny dimensions, float64, zero WikiCS/molecule dropout, and `.eval()`; it exercises differentiable operators and factor updates, not the stochastic training runner. Its collab fixture deliberately includes duplicates and a reverse orientation in support, then verifies target absence (`30–35`). The loader safety fixture only invokes `require_keys` on an already materialized dictionary (`113–117`). No optimizer-selector transition, complete official loader, actual negative sampling generator, task evaluator, resource supervisor, or full-size epoch is exercised by these rows. The labels on CPU report fields should be read at this narrower scope. No extra bitwise FP32 identity gate is requested.

## Concrete findings

### R1 — Packed untied local selection contradicts its declared pooled contract

`models.py:169–172` sets `model.independent` for both `independent4` and `independent4_contrastive` (and singles). In `run.py:109–119`, every such arm restores four `own_local_m.pt` checkpoints and optimizer states at the WikiCS transition. Those files are independently selected by per-member metrics at `145–152`. The packed untied contrastive comparator therefore continues from a bank assembled from separate selected epochs. PROTOCOL.md:27 declares that this jointly regularized comparator has a pooled selector. Its final `selected.pt` is pooled (`141–142`), so the defect is specifically its local-to-global training transition.

The per-member transition branch must use the ordinary independently trained bank policy. The coupled untied arm must restore the synchronized pooled `selected_local.pt` model and its optimizer states. Live end-local dropout/data RNG should remain preserved as declared. For a single model, own and pooled local selection coincide, so that does not create the same scientific mismatch.

### R2 — Full official role custody is not established by the present guard

`load_projection` implements narrow, hash-bound TRAIN and VALID payload loading and refuses provider dataset construction. `check_projection` verifies some substantial facts: fixed public feature shapes for WikiCS/collab, separate exact role key sets, TRAIN/VALID ID disjointness for WikiCS/molecules, canonical collab pair dtype/shape/bounds, molecular pointer coverage, and finite labels/unique role IDs.

Nevertheless, every supervised count is compared to a value declared in the supplied manifest (`data.py:58–63,68–73,90`). There is no supplied official role export or independent source/split custody receipt. WikiCS's fixed 580 TRAIN labels, 5,274 VALID labels, and 442,907 prepared graph entries are not enforced. Full public feature shape alone is compatible with a sampled supervised subset or altered support. The loader does not require the prospective split/source hash fields described in PROTOCOL.md:15. Collab temporal role identity and molecular scaffold disjointness are assertions in the authority, rather than facts established from independently checked split IDs.

Schema gaps also matter before a role is called safe: WikiCS lacks exact edge_index dtype/rank/bounds and ID dtype/rank/domain, y class-domain, and ID/y length checks. Molecules lack ID/y equality and domain checks, pointer dtype checks, edge_index dtype/rank consistency, edge_attr row-count versus edge-count checks, and categorical domain checks. Per-graph local edge range is checked when batching (`132–143`), which is useful but later than projection admission. Negative WikiCS IDs could silently select from the end of the public node tensor. Collab already has stronger pair bounds and an actual unordered TRAIN/VALID-positive overlap guard.

Required release work is independently reviewed official source/split/count custody plus TEST-free exports and complete schema/domain guards. These are absent official acquisition/hydration steps, distinct from the already implemented adapter functions. This review does not require real hydration during source review.

### R3 — Extra heldout fields are rejected after their values have loaded

`data.py:46–49` calls `torch.load(weights_only=True)` on each entire role dictionary before `require_keys` executes. `weights_only` constrains deserialization classes; it does not make extra tensor fields unread. If a malformed export contained `test_y`, that tensor would already be materialized before exact-field rejection. The CPU test at `check_cpu.py:113–117` checks a preconstructed dictionary and cannot establish the comment that rejection happens before values load.

For correctly audited TEST-free exports, the intended no-TEST custody can hold, and there is no TEST reader in the runner. The present loader alone does not prove that custody. Use a format with a narrowly checked field inventory before loading values, or make the independent TEST-free export audit explicit and remove the stronger pre-deserialization test claim.

### R4 — Finite selector metrics can conceal nonfinite predictors

`run.py:88` checks only pooled/member scalar metrics. It never checks all member logits or pooled prediction tensors before evaluator calls. WikiCS NaN probabilities can still give a finite accuracy after `argmax`; link comparisons/ranking can yield a finite Hits value despite a nonfinite input. Thus metric finiteness does not prove model finiteness. Finite TRAIN loss and gradients checked before `opt.step()` (`134–138`) do not certify post-update parameters or every VALID output.

Reject nonfinite member logits and pooled predictions before selection, and bind the declared finite model/state failure contract. This is a concrete failure-handling defect rather than a request for numerical identity or extra optional tests.

### R5 — Launch preparation and startup failures are outside recorded cost

`run.py:27–57` performs runtime imports, full data loading, native-source import, CUDA model materialization, optimizer construction, and RNG-stream setup before `started` and the `try` boundary at `101–105`. Data/schema/source failure or startup OOM has no `FAILURE.json`, while these costs are omitted from trace/freeze elapsed seconds. PROTOCOL.md:45 promises real cost accounting including graph preparation and optimizer states.

Per-cell durable timing and failure custody must begin before preparation and include startup plus externally killed cells. Within the present loop, full member/view paths, full validation, checkpoints, and the extra assembled independent-bank evaluation are timed. A timeout or other caught training exception preserves completed epochs, steps, elapsed time, and no-retry status, but this does not cover hard kills or startup.

### R6 — Resource and hard deadline assertions are not actual supervision

`runtime.admit:91–103` requires resource receipt schema/task/source identity and flags for one complete TRAIN update, complete VALID evaluation, and closed scores. It does not bind the exact data manifest/payloads, config/arm workload, runtime identities, measured memory/time envelope, or hard supervisor. A receipt for a cheaper arm or different projection can satisfy these minimal fields. `external_hard_bound_confirmed` is only a Boolean. No qualification runner or actual external supervisor is included, as the handoff correctly admits.

The process checks its soft clock only at epoch entry (`run.py:106–108`). It can spend arbitrarily long in a sparse operation, validation, or last epoch, and does not enforce `hard_seconds` itself. A separately bound launch-to-exit supervisor must enforce the hard cap and emit durable failed-cell/cost records. Resource admission must bind the actual full task/data/config/member/view/evaluation/checkpoint workload, with representative memory/time evidence that keeps scores closed. Do not shrink backbones, drop tails, reduce labels, or replace seeds to rescue resource failure.

## Implemented training and data behavior

| Family | Actual training/batching/selection behavior | Qualification still absent |
|---|---|---|
| WikiCS | One update per epoch over all provided TRAIN IDs using the entire provided public graph/features. Fixed 100 local + 1,000 global epochs, width512 native Polynormer, no schedule/early stop. Full VALID ID forward each epoch; mean member probabilities, split0 accuracy. | Official split0 IDs/labels and complete topology identity, full graph gradient/resource viability. |
| Collab | All provided TRAIN records are shuffled; batches65,536 include tail. Equal-count TRAIN-only sampled negatives are generated per epoch from TRAIN support, with no heldout-positive filtering. Every member/view sees every target. Full complete VALID positives/shared negatives, query batches131,072, mean raw logits, named OGB Hits@50. | Official complete temporal records/negative population, exact provider sampling contract and full-size cost feasibility. |
| Molhiv | Shuffled graph-local concatenated TRAIN molecules, batches128 include tail. Original nine atom and three bond fields are retained; PyG batching offsets local edges. Both views use model dropout, without chemistry corruption. Fixed100 epochs width256/layers5; full VALID, mean raw logits and named OGB ROC AUC. | Official scaffold IDs/source/finite labels, schema/domain certification, full-size batching/learning/resource feasibility. |

All full provided targets receive two own-supervision forwards per update. Only the auxiliary contrastive object set is bounded to at most512 deterministic positions (`run.py:125–133`); this is not a reduced CE population or shortened horizon. The class centers and pair targets use only that TRAIN batch's labels, not VALID labels. Native/molecular LayerNorm is stateless, with no member/view running moments. Ordinary independent4 has four full parameter sets and four Adam instances; own loss gradients are unscaled. Packed untied contrastive members are intentionally jointly regularized.

Collab target-support protection is real: `pair_ids` canonicalizes both orientations, and `support_graph:117–129` removes every duplicate record of each positive minibatch target before constructing/coalescing BOTH adjacency orientations. It then checks the constructed support. `batches:165–167` passes that same target-masked adjacency to both forwards. `CollabBackbone:103–104` sends it to both encoder and decoder. Native encoder DropAdj can only remove entries; native decoder edrop0 returns unchanged support. Native CN `cndeg=-1` and `adjoverlap` disable neighbor subsampling, preserving complete common-neighbor evidence on the masked graph. Validation support uses TRAIN only and actual canonical VALID-positive overlap is rejected before training.

Support deduplication does not deduplicate the supervised target records. Canonical repeated/reversed targets may still cooccur in a contrastive sample; the method reviewer separately audits this instance-positive identity issue. TRAIN-only negative sampling deliberately can include a future positive because no future labels are consulted; that is not heldout leakage. The exact installed negative-sampling function hash is guarded, but this review did not independently execute or inspect that remote provider.

No explicit TEST tensors or labels are read by the runner, and native modules are imported without their dataset/main scripts. Public WikiCS/collab features/structure are transductive inputs, permitted by the declared task. VALID labels are loaded into their separate role for evaluation and checkpoint selection and are not supplied to the training objective. This statement is conditional on authentic, correctly exported roles; the absent official exporter cannot be replaced by trusting the manifest's Boolean claims.

## Selector, mode, and RNG custody

For ordinary independent4, each member has its own strict `>` first-maximum local and overall selector. Its local optimizer state restores at the WikiCS transition; it retains live end-local RNG rather than rewinding to checkpoint RNG. Final serving loads each own best body, sets each saved local/global flag, evaluates the assembled bank, and saves `body_global`, `evaluation_only: true`, and `individual_best_bank_only`. Pooled metrics never select that bank. Individual checkpoints retain their optimizer states. These are implemented facts, not interchangeable native historical trajectories.

Shared BE uses pooled local state and optimizer state, then global mode. Joint generic checkpoints save model/optimizers, mode, persistent dropout streams and global Python/NumPy/torch RNG snapshots. Persistent per-member torch CPU/CUDA streams run inside `fork_rng`, so ordinary member dropout streams do not depend on the number of other paths. Batch shuffles use a dedicated epoch/base generator common across arms, allowing object alignment. The packed untied transition defect R1 is the exception to declared selector behavior.

`FREEZE` records the final selected checkpoint hash and complete horizon, but a whole-family closure ledger, per-selected independent horizon/hash summary, common-error report, and review of TEST serving/restoration are separate later steps. A closed-score flag and silent metric stdout are procedural custody markers; this release does not implement a separate permission barrier against opening files.

## Runtime admissions, disabled jobs, and budgets

The three shipped files are templates, not72 concrete runnable jobs. Each template has root authorization/source approval/fixed adoption false, config/data/source hash placeholders, missing output/evidence, TEST false, and retry false. `admit` rejects these defaults before tensor imports. This correctly keeps the source packet disabled. The protocol/config fix eight arms and three paired seeds per family, giving72 bank cells. No active queue integration or retry loop is present.

Admission pins hostname `anogena-2-0`, exact singleton GPU UUID, existing interpreter path, exact repository cwd/phase, exact CUDA_VISIBLE_DEVICES, immutable source/config hashes, and exact pilot arm/seed. `runtime_versions` enforces declared torch/PyG/NumPy/OGB/sparse/scatter/CUDA versions and full path/hash identities for the named graph operators, molecular encoders, negative sampler, and OGB evaluators. It disables TF32/benchmark and limits CPU threads. Those checks identify the admitted direct providers; they do not establish bitwise repeatability of sparse CUDA execution or real data/learning competence.

The task pooling/evaluation calls match their definitions: WikiCS averages probabilities and counts argmax correctness; collab averages raw logits with the shared negative pool and sets K50; molhiv averages raw logits and invokes the named ROC AUC evaluator. The R4 output-finiteness hole remains. No TEST scoring source exists, and collab's later VALID-at-TEST support convention is deliberately unbound.

Parameter bank/path arithmetic is consistent:14 parameter banks per task/seed,126 total;26 member paths per task/seed,234 total, with two stochastic views per update. These are accounting identities, not memory or efficiency measurements. Nominal soft ceilings are24*(8+12+8)=672 hours; hard ceilings24*(9+13+9)=744 hours. Caps are unmeasured admission ceilings and are not forecasts. Per-task max_cells/max_pilot_epochs are declared but no whole-suite scheduling/cost ledger enforces them; root must bind the finite72-cell release and all failed/missing outcomes before opening comparisons.

## Release conditions and remaining future work

A successor must repair R1 and R4 and make R5 accounting concrete. R2/R3 require separately reviewed, authentic TEST-free official role exports and complete schema guards. R6 requires actual full representative resource evidence and a bound hard supervisor; receipt labels alone are insufficient. All jobs remain disabled until the successor's source/config/data/runtime/resource identities are independently reviewed and root adopts/releases the exact fixed protocol.

Official acquisition/hydration, a TEST reader/scoring release, full task resource qualification, the training-only sensitivity/common-error panel, full family summary/freeze, and capacity/compute matching are distinct absent or unverified steps. Their absence must not be presented as a method failure or as already implemented successful execution. The source release provides no empirical improvement, efficiency, novel-method, or confirmation claim.
