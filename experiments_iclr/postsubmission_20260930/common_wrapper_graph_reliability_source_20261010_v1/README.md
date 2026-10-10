# Fixed graph-context reliability source and support

Status: source only; no fit/export/runtime admission. Root reviews this package and owns execution, publication, interpretation and confirmation. No scientific model, dataset, checkpoint, label, logit or prediction payload was opened during preparation. No numerical source was imported or run.

## One mechanism and its ancestry

The fixed56-parameter shared5→8→1 tanh scorer maps normalized entropy, margin, own-node consensus, own-member neighbor agreement and peer-neighbor agreement to a softmax over members. It serves a convex mixture of the existing probabilities. Both class permutation equivariance and member permutation symmetry follow from the scalar features, common scorer and symmetric leave-one-out means. Every member executes. Confidence and agreement are supervised reliability cues, not correctness guarantees. A strict common wrong rival in every member cannot be repaired by this mixture.

The5October graph-conditioned frozen stacker already proposed the same broad operation. This is an attributed, more restricted reliability question. Amazon99 failed every practical screen against capable processed controls and remains NO_GO. Direct12 failed its exact regularized frozen TRAIN-classifier refit; it did not fit a frozen probability mixer on fusion-development labels. Those closures were read before implementing this package. A retained correct member in the new closed117 diagnosis motivates serving work but does not predict a successful selector. No novelty or generalization claim follows.

## Exact selected support

`FROZEN_SUPPORT.json` identifies45closed banks: SAGE/GCN/GAT × seeds7301/7403/7507 × ordinary_M1, ordinary_genuine_I4, factorized_allmap_M1, factorized_allmap_genuine_I4 and shared4_unchanged. Both capable singles and genuine independent references receive all nonduplicate fusion rules. Separable/exchange, coherent24, current24 and full9on77 are excluded.

Each allocation bank lives at:

`experiments_iclr/postsubmission_20260930/common_wrapper_{SAGE|GCN|GAT}_root_20261010_v1/actual_family_v1/{arm}_seed{seed}/`

Its original `selected_VALID.npz` has exactly IDs, y, raw_logits, probability_mean, member_errors and pooled_errors. Raw logits are float32 `[M,5274,10]`; VALID IDs/y are int64 `[5274]`, original probabilities float32 `[5274,10]` and errors Boolean. The bound archive owns every own-node prediction/label. Its original scores and selector are preserved.

Selected native checkpoints are `member0/selected.pt` for each M1/shared4 bank, or `member0` through `member3` for a genuine I4. The complete descriptor binds their exact paths and selected steps. Each state has model, optimizer, streams, step and selection. Historical checkpoint hashes were not available in the local descriptor; do not invent them. The future one-pass exporter captures/stability-checks current hashes and checks the selected step and complete closure. Original archives, source/config, TRAIN/VALID containers and owner/complete receipts already have bound digests.

Reuse the original `Family.make` and `Family.logits` from SAGE `run_family.py` or the common native-backbone `run_family.py`. They construct the same native `models.Model`, factor banks and shared routes and restore `state['model']` strictly and `state['streams']`. Their existing full-graph forward is followed by output row selection. The helper returns an unused optimizer during construction, which is released; it never takes a step. There is no new backbone trainer, selector, owner, retry or acquisition framework.

The prepared TRAIN container already carries float32 `[11701,300]` features, int64 `[2,442907]` graph edges and only580TRAIN IDs/labels. VALID contains only5274VALID IDs/labels. Reuse these safe containers. Never reopen official/raw WikiCS data, load full-node ground-truth labels or export TEST quality.

## Minimal neighbor export

For the selected states, compute full-node logits/probabilities **in memory**, without full-node labels. The graph support is the same authorized full11701-node feature/edge support used by the original trainer. Distinct incoming nonself neighbors define the row mean P; isolates use their own prediction. Add no reverse edges; the already supplied graph is authoritative. No neighbor label, label residual, true homophily or correctness flag enters P or the features.

Persist only VALID IDs and float64 neighbor means `[M,5274,10]`. Existing archived VALID logits supply the own-node probabilities throughout fitting/assessment. TRAIN and other visible nodes contribute label-free prediction support to neighbor means; their logits/probabilities are not persisted, and no TEST IDs/labels/accuracy/quality export is created. No extra TRAIN label export is needed. This requires99selected checkpoint forwards/126native member trajectories, not another base fit.

One deterministic export compares its VALID slice with stored raw logits using fixed `atol=1e-6, rtol=1e-5`. Record maximum absolute and tolerance-scaled logit differences and member argmax disagreements without truth labels. Bitwise equality is not required. Preserve discrepancies and costs; if beyond the fixed tolerance, return to root with the artifact and stop. No numerical investigation or replay loop is included. This is an engineering check, never a utility threshold. Exported own-node logits never replace the archive.

The raw uniform pool remains the archived float32 decision/probability anchor. Stable NLL follows the original float64 log-softmax/log-sum-exp convention. A separate unfitted FP64 uniform mixture is also reported, so any reconstruction difference stays visible. Gate initial weights are uniform over the mathematically recomputed frozen probabilities; that statement does not claim bitwise identity with the native float32 pool.

## Fixed fusion controls and supervision

`operators.py` implements one candidate and the essential two baseline families:

1. Positive global and per-member temperature calibration, then uniform probability pooling; log-T starts at0, fit pooled NLL.
2. The exact same56-parameter scorer with P=I, plus an ordinary multinomial linear stacker on concatenated own-node probabilities. P=I replaces the last two inputs by own-vector norm squared and own-node consensus; the repeated coordinate is disclosed. Linear stacking is a class-specific decoder and can leave the convex hull.

The gate initializes W uniform[-.1,.1] from seed11709, b/a zero. Both graph/self gates use mixture NLL plus0.01 mean KL(member weights || uniform). Linear stacking starts at zero and adds0.01 mean-square penalty over all coefficients/biases. Every nonduplicate operator uses CPU float64, full-batch Adam lr.01, betas.9/.999, epsilon1e-8, no weight decay and exactly500updates. The last update is the endpoint. There is no head, penalty, seed, temperature or epoch search and no final refit. M1 gating has no mathematical effect and is not fitted; its per-member temperature duplicates the global temperature prospectively.

All45banks use the identical VALID row order and same label-free fixed five-fold partition from CPU torch randperm seed11709, position modulo5. For each fold, its labels are excluded from the aggregator loss and never choose settings/checkpoints. This is **encountered development**: every base bank already selected on all VALID labels, and the proposal used development diagnostics. It is not whole-pipeline cross-fitting or independent confirmation. Allowed transductive prediction context does not make connected nodes independent observations.

The callable validates all45inputs before its first fit and retains failed endpoints while attempting the entire fixed fusion roster. It reports855nonduplicate fit calls, at most427500small-head updates, every fixed endpoint, served accuracy/NLL/Brier, native repairs/harms and repaired/remaining native pooling losses. No automatic GO or outcome-selected continuation is implemented. OOF predictions/IDs/folds are VALID-only artifacts. Root must freeze any numerical worthwhile-effect/uncertainty rule before fitting.

Both callables atomically update their output `PROGRESS.json` after each completed bank. It contains only completed-bank count, current key, elapsed time and the fit-call or selected-forward counter. No partial quality is included; root retains process ownership and polling.

## Decisive falsification and compute

The practical mechanism fails in this scope if served quality does not improve beyond the strongest temperature/self/linear reference with acceptable proper risk. Better NLL without recovered accuracy supports calibration utility only. P=I matching or beating the graph gate rejects the contribution of neighboring predictions in this recipe. Equal fusion opportunity on singles/genuine I4 prevents claiming a special shared benefit from an averaging-only comparison. A positive development result needs a frozen whole pipeline and separately authorized unused confirmation before stronger claims.

The56-parameter candidate adds one cached sparse propagation over126member probability fields across the full roster: O(|E|MC) per bank, then O(NM(C+48)) head/features plus8tanh operations per member/node. VALID-only neighbor arrays total53,161,920uncompressed bytes across45banks, excluding NPZ metadata and native archives. The linear reference has C(MC+1) coefficients and O(NMC²) forward work. All native restoration/forward, CPU transfer, graph cache, head fit, I/O/storage and previous acquisition costs must be charged. Timings and peak memory remain unmeasured; no route-compute savings follow.

## Root-called interface

There is no launcher. In a fresh authorized root-owned worker, insert this directory at the front of `sys.path`, then call:

`export_neighbors.run(FROZEN_SUPPORT_path, fresh_export_output)`

After root qualifies/reviews that complete export, in the existing CPU execution policy call:

`study.run(FROZEN_SUPPORT_path, export_output, fresh_study_output)`

Both callables verify the literal allocation, sole GPU identity and repository before input reads. Output paths must be fresh and within the existing postsubmission phase. Root supplies process limits/owner/custody and publishes the source before execution. `BUILD_SUPPORT.py` and `check_static.py` are local standard-library bookkeeping only. Static syntax/roster/hash checks do not verify numerical restoration, gradients, convergence or scientific utility; those remain unexecuted here.
