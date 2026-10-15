# Own selected-VALID source reconstruction v1

Reconstruct an actual `run_fit` own selected node/OGB state through its recorded
versioned runtime, callback, extension and native source dependencies. Invoke
complete native VALID once and compare to that fit's saved selected_VALID
export. No training, optimizer transitions, selector updates/reselection, TEST
query, TEST scoring or new human approval flow is provided. Existing campaign
GO and closed-TEST requirements remain in force. CPU v1 requires the literal
guarded allocation route, CUDA hidden and two threads; it starts no GPU worker.

## Fixed supported source mappings

`REGISTRY.json` lists exact hashes for runtime-v1's four producer files and
runtime-v2's six producer files, the original NodeSession/OGB callbacks,
native/base procedures, portable input v2 BE/residual-only, structural C2
molecular/v2-node, C6 mean/global/conditional molecular/node, BNv3 molecular,
and the already frozen three molecular pilot extensions. No future source is
certified by a family name. Unknown runtime, callback, extension, package,
binding, method setting or state generation is refused. In particular C2
node v1's preserved failed guard, portable v1, BNv1/v2 and links are not
substituted with a base model. Native I4 bodies are four separate own replays;
this package does not create or reselect an I4 bank.

Every replay requires producer-colocated `JOB.json`, `selected.pt`, and
`selected_VALID.pt` with explicit path/byte-count/SHA256 descriptors. The
original job digest, body/cohort/seed/split/policy, selected epoch, export's
checkpoint receipt and complete VALID role must match. Reconstruction selects
the recorded runtime by its exact hash map, imports that runtime in an isolated
module namespace, builds its actual adapter on CPU, checks the reconstructed
runtime binding, and loads the entire adapter state. Wrapped method states
are never stripped. The job remains the original authority; the construction
copy changes only the device to CPU. Source global Python/NumPy/Torch CPU RNG
is restored; CUDA RNG remains archived and is not applied on CPU. This is
inference replay, not training resume or whole-paper RNG equivalence.

All model/checkpoint/data/source paths must remain in the own repository.
Data declarations pin exactly the native processed node data and cached basis,
or molecular processed store plus official TRAIN/VALID scaffold role files.
No TEST role file is accepted or requested. The original native provider may
load its standard shared processed store; TRAIN/VALID views and native serving
remain unchanged. Cached node bases must already exist and be pinned.

## Actual serving and C6 diagnostics

The recorded adapter's actual `validate()` normalizes the source output, labels,
IDs and metric. A temporary observer captures the SAME native session VALID
return before the frozen adapter drops extra fields. C6 same-bank mean raw
logits, native loss and metrics are saved alongside actual served outputs.
There are no extra backbone forwards, changes to the selector, gate refit or
VALID-fitted combiner. Every optimizer's step is blocked during replay.

## Practical parity rule

IDs/order, labels/NaN support, tensor shapes/dtypes, selected epoch and all
identity/source receipts match exactly. Raw logits use fixed atol1e-4/rtol1e-4;
native loss atol1e-4/rtol1e-4; each native metric absolute tolerance.002. These
are modest engineering replay tolerances, fixed before qualification. They
are unrelated to scientific effect sizes or acceptance criteria. No adaptive
tolerance escalation, exact-tiny replay requirement or performance claim is
introduced. The output records actual differences and fixed tolerances.

## Request / CLI

Use schema `own-selected-VALID-replay-v1`, `role=VALID`, `TEST_access=false`,
`device=cpu`, `replayer_manifest_sha256=<exact package manifest>`, and the
three explicit file descriptors under `job`, `checkpoint`, `selected_VALID`.
`data_files` is `{processed,basis}` for nodes or
`{processed,train_split,valid_split}` for molecules, with identical descriptor
shape `{path,bytes,sha256}`. No caller-selected tolerances are accepted.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
<allocation-python> -B replay.py --project-root <own-repository> \
  --request <own-request.json> --output <fresh-own-output-directory>
```

The CLI performs route admission before project reads. Successful output is
`reconstructed_VALID.pt` plus a compact `RESULT.json`. Refusals/failures are
preserved in a fresh output, never silently retried. A selected replay must
use actual producer artifacts; engineering checkpoints are not relabeled as
scientifically selected states. Separate bounded qualification may reuse such
existing checkpoints to test native reconstruction with zero optimizer steps,
clearly recording that a saved selected-export gate was not exercised there.

Sources are prepared here; exact-method native replay qualification is a
separate evidence layer. No GPU/resource admission, final TEST reconstruction,
convergence or predictive-benefit claim is implied by source registration.
