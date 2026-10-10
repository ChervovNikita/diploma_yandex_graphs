# Small SAGE family entry

`run_family.py --config /absolute/config.json --output /absolute/fresh-output`

This source reuses the original `Model` and `run_base.train_step`, the immutable common-wrapper v1 and the established factor module. It has no launcher, owner, retry, resume or external-host operation. The source was checked with the standard-library AST parser only; root owns numerical qualification and the final frozen configuration.

## Configuration

Required keys:

| Key | Value |
|---|---|
| `native_repo` | Directory containing the authorized original `models.py`, `run_base.py`, `run_common.py` and their normal import dependencies. |
| `common_routes` | Absolute path to unchanged `shared_fast_graph_model_interface_20261010_v1/common_routes.py`. |
| `factors` | Absolute path to existing `portable_internal_be_public_interface_20261007_v2/core/factors.py`. |
| `train_npz` | Safe archive with exactly `x`, `edge_index`, `ids`, `y`. |
| `valid_npz` | Safe archive with exactly `ids`, `y`. |
| `device` | Root's authorized numerical device, e.g. `cuda:0`. |
| `seeds` | Three distinct paired integer seeds; the proposal uses `[0,1,2]`. |

Optional defaults are `hidden=128`, `depth=2`, `dropout=0.2`, `rank=16`, `learning_rate=0.001`, `max_updates=1000`, `patience_updates=300`. FFN multiplier is 1, normalization is LayerNorm and AdamW weight decay is 0. Root must freeze the values it intends to run. A smaller horizon is an engineering qualification, without quality evidence.

The entry consumes the safe graph exactly as supplied, with no second graph preparation or feature normalization. It requires float32 features of shape `[11701,300]`, int64 graph/roles, TRAIN580 and VALID5274, with distinct, disjoint role IDs. The trainer's full-node label vector contains only projected TRAIN labels at TRAIN IDs; all other entries are neutral storage and are never selected for its loss. No original dataset loader is called. Importing `run_base.train_step` loads that module's normal dependencies but never invokes its main program, loader or TEST evaluation.

## Fixed semantics and pairing

The complete roster is seven arms × three seeds: ordinary M1/genuine I4, all-map factorized M1/genuine I4, unchanged shared4, equal-size separable and exchange. Genuine I4 acquires each fresh body sequentially with separate parameters, optimizer and persistent RNG stream, selects each body by its own accuracy, then pools its selected outputs. All singles call the native function and native `train_step`, including the factorized M1 body. Shared arms acquire by the mean of the four own CE losses with one backward and one AdamW step.

Serving and selection use the arithmetic mean of member **probabilities** for every bank. This explicitly supersedes the proposal JSON's raw-logit-mean wording under root's latest instruction. No pooled-target supervision, teacher or additional diversity loss is used.

Accuracy and serving retain native float32 softmax probabilities. Reported member and pooled NLL use float64 `log_softmax` and `logsumexp` from the original float32 logits, without clipping probabilities. Shared-route graph objects contain only the factual edge index; labels and training masks stay in the separate native training object.

For family seed `s` and member `m`, the defaults are:

- native initialization: `s + 1000003*m`;
- input-factor Rademacher row: native seed `+ 2000003`;
- dropout stream: native seed `+ 3000007`;
- connector initialization: `s + 4000037`.

Offsets have corresponding config keys `member_seed_stride`, `factor_seed_offset`, `dropout_seed_offset`, `block_seed_offset`. Factor rows are initialized separately with shape `[1,300]`, so each shared-bank row matches the corresponding factorized I4 input row. Every other r/s factor remains one. Ordinary/factorized/native member 0 and all three shared arms share the same native start. The four genuine I4 bodies use distinct native starts. The caller's existing member RNG scope preserves each route's stream; checkpoint state includes those streams.

VALID is evaluated after every update. Every first strict accuracy maximum is saved immediately; an exact accuracy tie retains the earliest checkpoint regardless of NLL. Patience resets on strict improvement and increments otherwise. No scheduler is used. After training, selected model state is strictly restored, optimizer state restored, and one selected readout is recorded.

## Outputs and cost scope

Each acquisition unit saves original `trace.jsonl`, overwritten-on-improvement `selected.pt` (model/optimizer/RNG/selection), and `RESULT.json`. Each arm/seed saves `selected_VALID.npz` with ordered VALID IDs/labels, raw member logits, probability mean and member/pooled error masks, plus its `RESULT.json`.

`PROGRESS.json` contains only current arm/seed/member/update and completed acquisition units. JSON updates are atomic. `SOURCE_HASHES.json` records the exact config-byte hash and the entry/native/wrapper/factor source hashes; root owns external source binding and qualification.

`COMPLETE_FAMILY.json` is written only after all 21 groups and 39 acquisition units finish. It declares completion and no TEST access, and records actual update/backward/Adam counts, TRAIN/VALID/selected-readout full-graph native trajectory counts, and native graph-block calls. It contains per-seed pooled/member accuracy and NLL, correct-alternative coverage, losses during pooling, and repairs/harms versus each of the six reference arms including separable. It makes no automatic quality decision. A failure preserves the incomplete output and exits; there is no automatic continuation.

Costs include sequential acquisition wall time (construction, TRAIN/VALID passes and checkpoint saves), selected full-graph forward/metrics/CPU transfer and group probability-pooling time, parameter counts/bytes and selected checkpoint bytes. GPU peaks are reset for each acquisition and include resident safe graph/data plus model/optimizer. Genuine I4 sums acquisition, parameter/storage and selected readout costs, while taking the largest observed acquisition GPU peak. It does not claim four-body simultaneous serving memory. RSS is the process lifetime peak. Root records concurrent allocation overlap separately; these readout costs are not isolated serving benchmarks or evidence of published competence.
