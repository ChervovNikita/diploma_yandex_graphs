# Technical successor review of roles and runtime

Reviewed immutable `learnable_internal_be_contrastive_multitask_suite_20261007_v2`, MANIFEST SHA256 `fab86d30e67d8773031adc8dcfc636a516eba656a60f9794d033029812cd7af4`. This is an independent technical successor review using the previous v1 audit context, with no requested favorable verdict. V1 and v2 remain unchanged. This report concerns source preparation for allocation pilots; it grants neither an execution release nor scientific acceptance or whole-goal completion.

V2 concretely repairs the selector transition, numeric field boundary, prediction finiteness, and ML startup accounting. It implements substantially stronger official exporters/schema guards and a real live-parent supervisor. Remaining source issues are narrow: resource measurement types/finiteness, hard deadline grace accounting, discriminating optimizer test evidence, and the exact molecule export boundary claim. The method reviewer separately reports a conflicting family-closure policy that must be consolidated before adoption. `REVIEW.json` has `approved: false` for this manifest.

## Bindings and permitted work

All27 sealed source files match their manifest hashes and byte sizes. All12 Python files parse. All three native dependency files match their hashes. The unchanged model, native, staging and runtime-provider contracts were reused after hash verification from v1 context; new/modified role, execution, selection and test source was read directly. The existing30-row CPU report and stage receipt bind to this exact manifest; numerical checks were not rerun.

No framework/model imports, dataset/label/checkpoint/logit reads, GPU work, remote access, or packet edits occurred. A stdlib-only reproduction of the resource measurement Boolean expression confirmed its NaN/type gap. I read one named Collab transfer metadata receipt and verified its plan hash. The named WikiCS available manifest and molecular frozen authority are not available locally; their actual remote contents/raw-file custody were not independently verified in this review. No missing-local-file conclusion is made about their remote existence. BINDINGS.json and READ_SCOPES.json record these limits.

## Prior six findings: actual successor status

| V1 finding | V2 source assessment |
|---|---|
| R1 coupled untied WikiCS transition | Fixed. `run.py:113–114` calls the shared helper with `ordinary_independent`, which is true only for `independent4`. `selection.py:11–25` restores own body/Adam states only for that arm and a synchronized pooled model/all Adam states otherwise, then enables global mode without rewinding live RNG. |
| R2 official role completeness/schema/export | Implemented source repair, with actual export/custody still unverified. Fixed official counts and full dtype/rank/domain/local-edge checks replace self-declared sample counts. Concrete disabled exporters and an exact independent export-review requirement now exist. |
| R3 rejection only after tensor materialization | Fixed for new role NPZ loading. `inspect_npz` checks exact ZIP names and each NPY header before `np.load`; rejects extra/missing/duplicate entries, object dtype, Fortran arrays and unsupported headers. `allow_pickle=False` is used for numeric loads. The new CPU spy tests the actual helper. |
| R4 finite metric conceals bad predictor | Fixed in the actual runner. TRAIN both views and every VALID chunk check member logits and pooled predictions before loss/metric/selection. TRAIN representations and post-step model parameters/Adam tensors also undergo finite checks. |
| R5 preparation cost/startup failure omission | Fixed for supervised cell accounting. Worker timer starts before admission and the try includes runtime/data/native/model/optimizer/stream preparation. Parent timer starts before launch and writes a terminal receipt for failed startup or hard kill even without child FAILURE.json. |
| R6 resource/supervisor flags only | Substantially repaired. Exact workload identity and measured fields are required; a live parent PID/start-time/job/source/hash-bound supervisor launches and terminates its owned process group. Remaining R7/R8 below and actual qualification evidence prevent an execution-ready conclusion. |

## Remaining concrete findings

### V2-R7 — Malformed resource measurements pass admission

`runtime.py:103–110` requires a passed exact-identity resource receipt and full-work flags. But its numeric check is only `isinstance(peak_GPU_bytes,int)`, positive bytes, and `inclusive_seconds <= 0`. Python bool is an int, so `true` passes as peak bytes; a JSON NaN or Infinity time passes the comparison, as does Boolean true. Python `json.loads` accepts these nonstandard NaN/Infinity constants by default.

A stdlib-only reproduction of the exact expression confirms `{peak_GPU_bytes: true, inclusive_seconds: NaN}` is not rejected. Require genuine non-Boolean positive integer bytes and genuine finite positive numeric seconds, or reject nonstandard constants at parse plus validate types. Resource qualification must still establish actual full-work feasibility; these source guards cannot turn unexecuted evidence into a measured result.

### V2-R8 — Termination grace lies outside the declared hard ceiling

`supervise.py:38–48` begins SIGTERM after `hard_seconds` expires and then allows up to10 additional seconds before SIGKILL, with250ms polling. The supervisor is real, but the process lifetime can exceed the declared9h/13h cap by this grace and polling interval. PROTOCOL.md:47's744h sum omits it. The parent terminal receipt correctly records actual inclusive elapsed time.

Put TERM/KILL grace inside the admitted hard deadline, or explicitly bind and disclose the extra finite ceiling in per-cell and whole-pilot accounting. This is a deadline/accounting precision issue, not a request for bitwise numerical gates or additional scientific fits.

### V2-R9 — Adam restoration evidence is not discriminating

The shared helper does call `load_state_dict` for every appropriate optimizer, so source custody is implemented. In `check_cpu.py:135–164`, however, only model parameters are perturbed after saving. Live Adam states remain equal to saved states. The joint branch compares equal states but would still pass if the optimizer restore calls were omitted; the ordinary branch does not compare optimizer states at all. Yet both rows declare `optimizer_custody_checked: true`.

Make live Adam moments/steps differ from the saved candidates, then compare both ordinary and synchronized joint restorations. Until then, label this evidence as model-branch restoration/live CPU RNG/global-mode testing with optimizer restore verified statically. This finding concerns what the CPU result proves, not an observed incorrect optimizer in the helper.

### V2-R10 — Molecule TEST boundary receipt is broader than its actual behavior

`export_roles.py:36–39` parses all41127 rows of node/edge-count metadata into numeric arrays, including heldout graph-size counts. It then parses only selected official TRAIN/VALID label, atom, edge and bond rows. Counts are used to compute row offsets and do not enter model training/centering. There is no evident TEST target leakage in this source path.

Nevertheless, `TEST_numeric_values_parsed: false` at145 is too broad. Scope the receipt and description to the actual boundary: TEST target/atom/bond/edge values are not parsed, while complete public count metadata is parsed for offsets. If the required boundary literally forbids TEST numeric metadata too, revise that path before a separate data-only release. Global gzip bytes necessarily traverse unselected labels; the source correctly discloses that this is a selected-value boundary, not OS isolation.

The method reviewer additionally identifies PROTOCOL.md:41's all-three-task closure requirement conflicting with75's first-family effects review before later adoption. Resolve one prospective rule for complete24-cell adopted families. Do not count this as a newly found executable role/selector defect or duplicate the method review.

## Official exporters and role custody

All exporters are disabled by `data_export_authorized: false` and source-review false defaults. Source review, exact source seal, current allocation checks and a fresh phase output are required before framework/data imports. No provider dataset constructor, processed molecule tensor, TEST split loader or predictor is invoked.

WikiCS uses an exact hash-bound preexisting authority and its authenticated safe six-field TRAIN/VALID tensor. It verifies TEST labels unavailable to trainer and exact keys, then emits numeric NPZ roles. This first input still uses torch loading; its safety relies on the independently authenticated existing TEST-free projection, not on NPZ preinspection of that old input. The newly emitted training loader has the stronger NPZ boundary. Actual authority/payload inspection and export output receipt remain absent from this review.

Collab binds the original authorized singleton archive and its transfer receipt, reads only four exact allowlisted/hash-bound members, and never reads the TEST split member. Exact official TRAIN/VALID split pickles are trusted by member hashes; their dictionary keys, temporal years, duplicate-preserving raw TRAIN correspondence and independently expected public array digests are checked. It preserves every provided TRAIN record. The transfer receipt hash matches locally, but archive/member values were not opened here. An existing byte transfer receipt does not certify this new export output.

Molhiv binds the raw/scaffold authority plus explicit eight raw/TRAIN/VALID file hashes. It avoids processed tensors and TEST split files. Original categorical/local edge data is reconstructed with reciprocal edge interleaving and bond repeats; official TRAIN/VALID IDs must have complete counts and be disjoint. Node/edge-count metadata supports offsets; only selected target/features/edges are numerically parsed, with the R10 wording qualification. Scaffold grouping is not recomputed; authentic pinned official split files establish the role, and a later independent export audit must verify it.

The training loader fixes counts to WikiCS580/5274, Collab1,179,052/60,084 plus100,000 VALID negatives, and Molhiv32,901/4,113. WikiCS fixes full public features and442,907 prepared topology entries. Every family checks finite/domain-correct labels, IDs/pairs/pointers/categories and complete local-edge boundaries. Counts/shapes alone still cannot prove official role authenticity. `runtime.admit:97–99` now requires an independently approved exact data manifest for the reviewed source; the manifest embeds its concrete source custody and payload hashes. No real export/hydration was executed or approved by this packet.

## Full populations, support masking, sampling, leakage and selectors

Full populations/horizons and batches remain fixed. WikiCS has one full provided TRAIN update per epoch,100 local plus1,000 global. Collab100 epochs/batches65,536 and complete tails; Molhiv100 epochs/batches128 and complete tails. Every member receives two full own-supervision views for every TRAIN target. Only auxiliary contrastive objects are bounded512; this does not subsample CE, the graph or the horizon.

Collab support protection is unchanged and valid: unordered pair masking removes all duplicate and reverse records of every positive minibatch target before BOTH adjacency orientations are constructed; it checks their absence afterwards. That same masked support feeds encoder and NCN decoder in both views. Native encoder DropAdj only deletes support; decoder edrop0 preserves it, and cndeg-1 uses full common-neighbor overlap. VALID positives overlapping canonical TRAIN support are rejected.

TRAIN negative sampling still uses the pinned PyG function on TRAIN edges only, requests one sampled negative per raw TRAIN record, and fails on an incomplete count. No heldout-positive filtering occurs. The sampler can represent both orientations of one undirected negative; v2 canonical identities make repeated/reversed targets positives across views and reject contradictory labels for one canonical edge. The method reviewer independently verifies that objective repair. This review did not inspect or execute the remote negative-sampling provider; its exact path/hash is enforced at runtime.

VALID labels are used only in complete no-grad evaluation and checkpoint selection. Losses/contrastive centers receive TRAIN labels; all normalization is stateless LayerNorm. Public WikiCS/collab structure/features are allowed transductive inputs. No TEST scoring wrapper exists. Authentic export custody remains a condition for these leakage conclusions.

Ordinary independent4 retains four bodies/optimizers, own unscaled gradients, first-exact-tie own local/overall selectors, live end-local RNG, and an evaluation-only bank of separately selected bodies with their individual serving modes. Packed untied contrastive now restores synchronized pooled local state and all Adam histories. Shared/single selectors, checkpoint RNG snapshots, persistent per-member CPU/CUDA torch dropout streams and common dedicated batch-order generator are preserved. The actual helper repair is stronger evidence than the previous labels, with R9's remaining test limitation.

## Supervisor, resource identity and allocation readiness

The supervisor actually launches `run.py` in the exact repository with the existing interpreter, a new child session/process group, and a parent-created receipt hash passed through environment. The worker requires matching job/source/deadline/workload identity, parent PID and `/proc` start ticks. At timeout the parent verifies its owned child identity before TERM/KILL. A finally block writes terminal inclusive time/status/exit code/no retry. This is implemented process supervision, not just a Boolean promise, subject to R8's grace accounting.

Resource identity now binds task, arm, fixed config/hash, data manifest/hash, source manifest, runtime-pin hash, hostname/GPU UUID, member count and two own views. Work requires two-view backward/Adam, complete VALID, checkpoint serialization and closed scores; WikiCS must cover both full local/global modes. Peak memory and elapsed time are required but need R7's numeric fix. A complete real task resource qualification runner/receipt, measured feasibility for every admitted workload, and a whole-family cost/missing-cell ledger still are not provided. Do not confuse guard implementation with those measured results.

The three fit and three export files are disabled templates, not72 concrete runnable jobs. The configs keep all eight arms, three paired seeds, full family24-cell populations/horizons and no automatic next tranche. Source admission still requires independent source/export/resource evidence and root adoption/release. Shared resource and full family costs must include all members/views, gradients/Adam, validation, checkpoint copies/serialization, independent selector/assembled-bank evaluation and preparation. The nominal672h/744h ceilings are unmeasured caps, with R8 correction required; they are not runtime forecasts or evidence of efficiency.

Current guards are intentionally specific to `anogena-2-0`, GPU UUID `GPU-44039938-fd82-41d2-fefd-de71514e2fac`, the exact repository/phase/current cwd, existing `native_ncn_runtime_20261005_v1` interpreter, CUDA-visible UUID, package versions and named provider source paths/hashes. Supervision uses the same paths and native dependencies. Stage metadata confirms only this source stage and three native dependency matches, not data hydration or GPU competence. These guards do not admit a future77 port. Such a port needs its own pinned/reviewed host/device/interpreter/provider paths/source seal and full workload qualification; no77 access or readiness claim is made here.

## Recommendation

Preserve v2 as a substantive repaired source successor, correct R7/R8 and the R9/R10 evidence/boundary descriptions in a separately sealed successor, and consolidate the prospective full-family closure policy. Review that exact successor before any execution release.

Then separately authorize/audit authentic official role exports, bind independently reviewed output payloads, qualify full representative task/member/view/backward/Adam/checkpoint/VALID work with scores closed, and adopt/release the precise allocation family with a real supervisor and complete cost/failure ledger. TEST scoring, confirmation, evidence-response/common-error panels and capacity/compute matching remain distinct future work. This source review supplies no learning, improvement, efficiency, novelty or unused-population confirmation result.
