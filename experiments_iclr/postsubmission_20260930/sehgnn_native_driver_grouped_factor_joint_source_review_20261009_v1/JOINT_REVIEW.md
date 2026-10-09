# Joint source review: native SeHGNN driver and grouped factors

## Decision

**Accepted in the focused source scope after minimal cost/cleanup fixes.** Native runtime and candidate integration are not qualified by this audit. No data, labels, arrays, model checkpoints/outcomes, numerical providers or server actions were accessed. No floating-tolerance loop, new orchestration or paper verdict was introduced.

## Native driver

- `model.py` is byte-identical to the pinned author source (`0948239c4c4d06dd258cd106fd2fa7b6aaa2a62688fe70dd662413fa1eae532a`). All six required helper bodies are AST-identical. The public entry constructs the native class and installs no adapter.
- The isolated loader opens only node.dat/link.dat/label.dat. TEST-opening stock constructors, check_acc, TEST scoring/export and unassigned-row inference are absent. Targets outside explicit TRAIN/VALID are NaN. The complete transductive graph/features remain native; unavailable labels are not treated as supervised negatives.
- Destination-row/source-column support, dense keyword identity, all25 feature and12 label channels are retained. Label propagation receives only TRAIN labels, with native row-normalized products and diagonal removal without subsequent renormalization.
- The seeded global NumPy shuffle, generator=None TRAIN loader, native AMP/Adam and single master GradScaler across the source seed loop are preserved. Evaluation uses prebuilt known-role batches, FP32 native eval and captured/restored streams. Selected reconstruction restores model/buffers, Adam, scaler and safe primitive RNG state while preserving the live master/caller state.
- Selection retains the literal200 zero-based epochs, strict complete VALID BCE improvement/earliest ties and `epoch-best_epoch>50`. The native task residuals, dropout, semantic gamma and nonaffine five-logit LayerNorm remain.

### Minimal fixes resolved before the native seal

1. CUDA timing finalization now preserves the original error and always records the failed cost scope (`engine.py:35–69`).
2. The cohort reads final per-fit RESULT/COMPLETE after cleanup, retains partial failures and rejects failed fit/cohort cleanup (`runner.py:237–288`).
3. Per-fit CPU user/system values are deltas from captured initial_usage; RSS remains explicitly cumulative (`engine.py:302,386–388`).

No source-level blocking native-fidelity issue remains in the reviewed scope. Actual DGL/provider/model qualification is separately root-owned.

## Grouped-factor interface

The six placements are the two grouped feature projections, semantic Q/K/V and fc_after_concat. The wrapper calls the actual native affine operation once, keeps bias outside the output factor, and preserves the native unit operation in real arithmetic. Grouped W orientation, separate sorted feature/label namespaces (25+12), H-major concat flattening and effective-factor AMP dtypes are correct. Embeddings, task/head layers, normalizations, activations, dropout and gamma are unchanged.

Every native slow Parameter is genuinely shared by object identity through the deepcopy memo. Each member has separate modules and registered buffers, including its three task BatchNorm running states/counters: **this adapter shares no registered buffer**. Private factors are disjoint; installation adds no optimizer and performs no native reset or slow RNG draw. Install after final native placement and before the deduplicated optimizer.

The source/Jacobian limits are stated correctly. Zero-initialized gamma initially nulls Q/K/V predictor gradients. A zero removed input row can null its first-projection probe derivatives; later cross-channel LayerNorm/semantic mixing can retain dependence. Two forwards do not prove two nonzero private gradient terms.

## Interface limits and required native checks

This adapter provides **no source-view builder, state transaction callback or candidate bank fit driver**. The native semantic_fusion ignores the external mask, so a family cannot be removed merely by passing mask. Qualified views must remove every dependent feature and TRAIN-label path, preserve all keys/order/shapes and movie-own input, and correctly rebuild the native support/normalization context. This is a precomputed semantic-channel interface, not learned per-edge geometry.

Real integration must qualify all shared/member buffers (especially BatchNorm), native caches/normalizers, modes and exact per-view dropout/RNG restoration before/after every source reference, replay, trial and eval guard. Extra source calls must not train normalization statistics. M1 unit/native materiality, M4 streamed updates, actual site/row Jacobians, atomic private rollback, selected slow-alias preflight, optimizer/scaler/state restoration and full-scope time/memory also remain required. Small risk counts do not certify a cheap large-vector cone. These source checks establish neither predictor competence nor useful source evidence.

## Exact bindings

Native driver manifest: `fd5dc35d2a5712a1b0a267fbeff27d50cf352fe6957bdd1bb6dcfb40c1c71a88`.

Native seal file: `ad0dcbbca1b87e2759806e22eb4a45dc8fae82314febc119e4a840cad71d3d72`.

Grouped adapter source: `7d5f899645361212a64fe70ab0b06fd74cc5cfcf8d2ca361e833b884cadd5e47` (unsealed at review).

See `JOINT_REVIEW.json` and `INPUT_BINDINGS.json` for reviewed inputs, finding disposition and activity limits.
