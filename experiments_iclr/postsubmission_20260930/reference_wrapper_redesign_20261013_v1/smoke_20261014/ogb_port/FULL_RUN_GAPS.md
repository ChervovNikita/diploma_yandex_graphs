# OGB smoke port: concrete full-run gaps

14 October 2026. Source analysis and documentation only. Authorities are the
pinned `native_sources/ogb/main_pyg.py`, `ROSTER.json` training contracts,
`build_design.py`, this folder's `session.py`, and
`smoke_20261014/independent_bank.py`. No full runner or runtime qualification is
provided by this note.

## Bind the two training tracks

| Campaign track | Required behavior | Missing binding |
| --- | --- | --- |
| Source-horizon100 (`native_replica`) | Native M1 only; 100 complete epochs; no early stop; whole VALID ROC-AUC after every epoch; first maximum over epochs 1–100; no checkpoint0 candidate. | `create_session()` constructs the model but consumes neither horizon nor selection contracts. The source saves scalar results, not a selected model. A controller must retain the actual first-best state. |
| Extended1000/min100/patience50 (`long_convergence`) | Native bodies and the four molecular wrapper procedures; cap 1,000 complete epochs, minimum 100, then 50 nonimproving VALID checks; strict improvement resets staleness. Source Adam remains constant at LR .001; no warmup exists here. | Implement and checkpoint the epoch counter, post-minimum stale counter, best score/epoch/state and stop reason. Epoch 100 must finish before staleness begins; without later improvement the 50 later checks finish at epoch 150. Selection still considers epochs before the stopping gate. |

Bind the plan's placeholder `runtime.source_native_adapter:*` and
`runtime.native_boundary_adapter:*` factories to a full controller around the
actual callback. Explicitly translate roster `layers/hidden/dropout` into native
`num_layer/emb_dim/drop_ratio`, and validate the fixed batch32/full-feature,
worker0, Adam/no-decay contract; `_configuration()` currently checks only some
aliases. Preserve the 16 separate OGB presets and ten declared seeds. A cap
without an adequate plateau remains an inconclusive stopping outcome in the
campaign record.

## A complete epoch cannot be a fixed number of `train_update()` calls

Native `train()` consumes one entire shuffled TRAIN loader, including its short
final batch. It skips `batch.x.shape[0] == 1` or `batch.batch[-1] == 0` without a
forward or optimizer step; it does not globally drop the tail. `_TrainBatcher`
stores a complete permutation and cursor, but automatically opens another
permutation when exhausted. `_next_train_batch()` keeps seeking an eligible
batch and can cross that boundary when the tail is skipped. Therefore update
counts, or `ceil(TRAIN/32)` successful calls, do not define a source epoch.

Expose a finite traversal boundary before another permutation starts. Record
consumed positions, skipped batches and actual optimizer transitions separately;
run epoch evaluation after the final consumed position, even if that batch was
skipped. Native `main_pyg.py:136` also evaluates a second **complete shuffled
TRAIN traversal** in eval mode before VALID. This traversal is absent from the
smoke API and must retain its distinct eval skip rule (`x.shape[0] == 1` only).
It must not consume the next training epoch's update cursor.

The source reuses its shuffled TRAIN DataLoader for both traversals, advancing
sampler/DataLoader RNG. The smoke port instead persists a private training
generator and a separate VALID generator. Bind and record the full-run RNG law,
including TRAIN evaluation and loader iterator creation; the current paired
smoke adaptation does not establish historical advancing-RNG equality.

## Select with the source evaluator; give I4 four selectors

`validate()` already traverses whole official VALID and calls OGB
`Evaluator(dataset)` on native labels and raw logits. Preserve its missing-label
handling and source eval skip rule. Use `metrics['rocauc']` for selection, rather
than its diagnostic BCE. OGB returns a fraction; convert to the roster's
`rocauc_percent` only when reporting. Pin the installed evaluator/package and
processed provider along with source and split receipts.

The smoke `IndependentBank.validate()` returns current mean logits, mean member
loss and `metrics={}`. `train_update()` advances all four members in lockstep;
neither method supplies a full-run selector or independent stopping. Retain
each native body's own whole-VALID ROC-AUC curve, first-best checkpoint,
patience state and completion reason. After a body stops, other bodies must
continue their own native epochs without further updating it. Preserve the
member RNG streams already isolated by `_own()`.

Serve I4 from those four independently selected states, which may have different
epochs; align graph IDs and average raw logits, then call the same OGB evaluator.
Do not select a common epoch from the pooled score. The extended M1 is exactly
selected body zero, with the same checkpoint identity, rather than an extra
fit. GNNM needs one selector for its complete shared bank, including every
private factor and native BN buffer, using the served mean-logit VALID score.

## Final TEST needs a separate finalized-state path

`_train_valid_indices()` opens only TRAIN/VALID and `_RoleView` rejects TEST.
There is no TEST loader, role custody, evaluator call or selected prediction
export in this callback. Add a finalization path that authenticates the original
scaffold TEST IDs and their disjointness, locks selected checkpoint hashes and
epochs first, then evaluates whole TEST with source batching/skip semantics.
Export graph IDs, labels, raw logits and source evaluator results with those
checkpoint identities. I4's export uses its four own selected states; GNNM uses
its one selected bank. The plan's statistical `evaluate-roster` job consumes
authenticated exports and cannot perform model serving itself.

The native script evaluates TEST every epoch (`main_pyg.py:138`), while the
prepared contract says to quarantine native TEST diagnostics and retain source
RNG semantics. Reconcile that explicitly with final TEST separation: any source
diagnostic compatibility path must be isolated from selection, stopping and
decisions; a final-only path must disclose and account for the removed TEST
loader's RNG consumption. The smoke port's lack of TEST access resolves neither
full-run binding. Current engineering authorization covers no TEST scoring.

## Resume must restore the epoch phase and selected state as well as tensors

`state()/load_state()` already cover model/private factors, native BN buffers,
Adam, training permutation/cursor/generator, VALID generator, fixed TRAIN
responsibilities, role IDs, module modes and update count. Optimizer restoration
now deep-copies its payload. The controller separately owns global
Python/NumPy/Torch CPU/CUDA RNG; I4 additionally owns four member RNG states.

A full checkpoint must also preserve the completed/in-progress epoch, whether
TRAIN updates or epoch evaluation are next, all added evaluation RNG, best and
latest state identities, curves, patience and per-body stop state. Resume must
consume the remaining permutation once, perform pending evaluation once, and
retain the same native buffers: one TRAIN forward for M1, sequential routes
0–3 through shared native BN for GNNM. Each I4 body retains its own BN.
Responsibility weights remain the same one-time full-TRAIN table.

`PREPARATION.json` records unresolved native virtual-node replay: exact immediate
restore passed, but subsequent output/state replay failed. The ownership
hardening is not a new replay pass. Full-run resume qualification still requires
that failure to be resolved and epoch/tail/evaluation boundaries to be checked
under the unchanged native operators and declared tolerances.
