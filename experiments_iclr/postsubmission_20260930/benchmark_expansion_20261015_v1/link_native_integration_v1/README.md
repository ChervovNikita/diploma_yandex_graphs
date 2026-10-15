# Actual native link session integration v1

Status: source prepared and locally checked. Actual native numerical qualification,
resource measurements and link full-run policy/dispatch wiring remain pending.
No native model, tensor library, optimizer update, VALID scoring, server operation
or fit was executed to prepare this directory.

## Source boundary

`session.py` composes the frozen
`reference_wrapper_redesign_20261013_v1/smoke_20261014/link_port/callbacks.py`
session and the finalized `../link/` query objectives. The actual frozen entrypoint
is `callbacks.py:create_session`; this native port has no `session.py`.

The native model classes, original Parameter objects, optimizer groups, negative
sampler, graph inputs, native losses and source data order are retained. Capture
hooks observe the existing task prehead. TRAIN orchestration supplies the existing
four task routes to the finalized query objective and makes one transition through
the original optimizer. NCNC uses the existing explicit top-level task scope;
recursive completion does not receive task output factors. No frozen source file
or original paper score was modified.

The frozen BUDDY port hydrates original subgraph features on demand rather than
retaining the original persistent feature cache. Original feature operators and
values are intended to be preserved; persistent storage and timing differ. This
integration inherits that adaptation and has no new numerical evidence for it.

## Entry points

```python
session = create_session(recipe, procedure, data_root, device)
session.train_update()       # one original source minibatch optimizer map
session.train_epoch()        # remaining maps to the next completed native epoch
session.validate()           # all 60,084 positive and 100,000 negative VALID rows
state = session.state()
session.load_state(state)
session.close()
```

`procedure` accepts the frozen native M1/base/global-half/private-half aliases,
the eight finalized query objective candidate IDs in `../link/CANDIDATES.json`,
or `independent4`. Optional nonlinear hidden adapters and late logit gates are
explicitly unsupported in this integration.

`transform(native_session, job)` (also `extend_session`) wraps an already-created
four-route native base session. Its job must declare `family="link"`,
`TEST_access=false`, `recipe`, `data_root`, `procedure`, and optional finalized
`method_settings.candidate_id/kwargs`. The supplied job callback refers to the
original native callback for this transform API.

`create_adapter(torch, job, policy, root)` supplies the one-fit full-epoch
interface. Here the supplied job callback refers to this integration's
`session.py`. It rejects the common generic `Policy` before constructing a
model: callers must supply an explicitly link-specific policy with
`family="link"`, `metric="hits@50"`, `direction="max"`, `ties="first"`.
This module does not choose a source horizon, cap, patience or warmup. A
source-reference policy additionally permits only unextended native M1.

The shared `benchmark_expansion_20261015_v1/runtime` still dispatches only node
and molecular sessions. Link-specific source horizon/selection evidence, policy
validation, runner dispatch and a frozen run specification must be completed
before any full link fit. A generic source-reference Policy has an OGB fallback
and must never be used as a link source protocol.

## Paths and supplied hashes

The recipe may declare `_repository_root`; direct `create_session` defaults to
the parent of the local phase directory. `create_adapter` always binds it to its
provided `root`. Relative `_redesign_root`, `_native_callback`, `data_root`, and
every `data_files` value resolve against that repository root. Absolute paths,
`..` traversal and existing symlinks must also resolve inside the same root.

`_native_callback` must be the frozen `smoke_20261014/link_port/callbacks.py`
under `_redesign_root`. Before callback loading, its three Python source files
and declared external dependencies are checked against the native manifest.
The recipe requires exactly `data_files.graph`, `data_files.train`, and
`data_files.valid`; provider discovery and TEST roles are excluded. Missing
files fail when the original native loader reads them. No local data copies
were available or read during preparation.

Recipe `callback_sha256`, `_native_callback_sha256`, and
`native_callback_sha256`, when supplied, bind the frozen native callback.
The adapter/transform job's `callback_sha256` binds its `callback_module` using
the API-specific meaning above. Top-level job `_native_callback_sha256` and
`native_callback_sha256` bind the native callback separately; conflicting job
and recipe declarations fail. Actual receipts and supplied declarations remain
in metadata and checkpoints. File SHA256 is streamed in 1 MiB chunks.

## Epochs, checkpoints and VALID

NCN/NCNC retain source batches of 65,536 positive records with matching sampled
negative rows and drop the source positive tail. BUDDY retains shuffled mixed
positive/negative batches of 1,024 and includes the actual partial tail.
`epoch.py` distinguishes a started epoch, partial progress and a completed epoch.
A failed map that advances the sampler cannot be counted as trained or saved.

`train_epoch()` can finish a partial resumed epoch and returns the maps performed
by this invocation separately from the resumed prefix. Its source loss accumulator
survives checkpoints. NCN/NCNC report the mean complete-minibatch own-task loss;
BUDDY retains the source reporting law `sum(batch_mean * 1024)/query_count`,
including the final partial batch. Native source loss is reported separately from
the private extension criterion.

State includes model/optimizer/sampler/negative populations, route RNG and NCNC
scope counters, global Python/NumPy/Torch CPU/CUDA RNG, modes, gradient flags,
epoch progress/accumulator, candidate settings, integration/helper receipts,
native source metadata and byte/SHA receipts for the actual GRAPH/TRAIN/VALID
files. Reload requires matching source/data/candidate bindings and preserves
the original Parameter objects. Numerical save/reload/replay is still pending.

VALID follows the frozen native call order and returns aligned raw member and
mean-logit outputs, query endpoints, positive-first labels and stable row IDs.
The native Hits@50 evaluator and full-bank competence diagnostics use the entire
official VALID split. They do not establish accuracy gains before execution.

## Genuine independent four-model bank

`independent4` requires four distinct integer `native_seed_offsets`, first zero.
Each native M1 has separate weights, optimizer, sampler, RNG, selected checkpoint
and stopping state. No original Parameter is shared across bodies. Construction
and each member operation preserve the caller's global RNG.

Use `train_member_epoch(i)`, `validate_member(i)`,
`observe_member_validation(i, result)` and `stop_member(i, reason)` for each own
lifecycle. Selection requires a positive completed native epoch and complete
VALID, maximizes Hits@50 and keeps the first tie. `validate_selected()` averages
each member's own selected raw logits and restores live training state. For full
runs, use four separately owned native fit adapters/selectors; the one-fit
adapter rejects an I4 job. Joint `train_update()` exists for short qualification.

## Checks and remaining execution

Run the local source-only checks from the phase directory:

```sh
python3 -B benchmark_expansion_20261015_v1/link_native_integration_v1/check_static.py
```

`LOCAL_CHECKS.json` preserves the successful checks and an earlier temporary-path
fixture failure. These checks import no Torch and execute no model or tensor
math. `QUALIFICATION.json` lists actual native work still required. The existing
smoke controller can call this module's `create_session`, but its four charged
training callbacks (two maps, uninterrupted third map, replayed third map) and
full VALID passes have not run for this integration. Full-epoch and fresh-process
resume qualification require additional root-owned checks. All costs, failures
and source hashes must be retained in new output directories.
