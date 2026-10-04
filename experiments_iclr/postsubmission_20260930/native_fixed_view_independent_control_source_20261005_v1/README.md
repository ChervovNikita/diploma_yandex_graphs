# Ordinary native independent members with their assigned fixed views

**Separately proposed source control; no execution admission or empirical result.**
`control.py` supplies one-member training and a fixed four-member probability pool.
The existing 30-fit protocol and sources are unchanged. This control closes the
specific gap identified in the learnable-view assessment: conventional independent
full native networks must receive the same augmentation opportunity before a
sharing or utility interpretation can be made.

## Bound behavior

- **Acquisition:** reuse the byte-bound native reference `build` function. Each
  member calls the ordinary native `Polynormer` constructor, transfers to the
  selected device, resets parameters and creates its own ordinary Adam. Independent
  seeds come directly from the existing reference `fit_spec` (the three existing
  split/block seeds and member offsets); no initialized bank is copied.
- **Views:** reuse the byte-bound bank `schedule.assignment("tied_persistent", …)`:
  members 0–1 receive `equal`, members 2–3 receive `different`. Use the existing
  TRAIN-defined fixed views, coverage, split/view seeds and input identities.
- **Objective for member m:**
  `CE(native logits_m, complete TRAIN) + CE(assigned fixed-view logits_m, complete TRAIN)`.
  Both coefficients are 1. Each pass is backpropagated immediately; one Adam step
  follows both passes, so both forwards see the same pre-step parameters. There is
  no ensemble/pool loss. The bank's `/4` is its average over four jointly optimized
  members; the ordinary member objective retains its native CE coefficient of 1.
- **Schedule/selection:** reuse the native reference's 200 local then 2500 global
  updates, own strict native-graph validation correct-count selector and earliest
  ties. At the actual 200-update boundary, restore that member's selected-local
  model/Adam, preserve live RNG and enable global stage. Restore its overall selected
  state at completion, including a local-stage winner, and require exact selected
  native-logit restoration. There is no epoch-zero selection or additional selector.
- **Serving:** `pool_members` requires all four declared completed members and returns
  the arithmetic mean of their own selected native-graph class-softmax probabilities.
  It checks common source/input/device context, unique physical checkpoint paths and
  each in-memory selected-logit identity. It supplies no best-member choice or fusion.

`source_functions()` verifies 15 existing source/input descriptor bindings and
loads only stdlib source. It compiles verified bytes with local import mappings,
avoiding changes or bytecode caches in the bound directories. It deliberately
does not traverse manifest payloads containing saved test receipts.
`load_runtime(numerical=True)` additionally loads the unchanged native neural source.
Default numerical entry fails; the CLI prints source-only status.

## Reusable entry points and costs

`build_member(rt, split, member, device)` performs fresh native acquisition.
`assert_independent_ownership(rt, [(model, optimizer), …])` can check all four
live models for overlapping Parameter objects or underlying parameter storage and
checks each optimizer's exact ownership. Constructor independence is also directly
visible in the reused build source; no boundary-factor wrapper is involved.

`run_member(…, execute=False, …)` requires explicit execution plus a **separate**
caller context with schema `native-fixed-view-independent-control-context-v1`,
`execution_authorized=True`, the exact `control_source_sha256`, existing reference
manifest/protocol hashes, `paired_input_protocol`, `coverage`, live `features`,
`validation_labels_sha256` and `selected_device`. The original paired protocol
binds inputs and provenance; its disabled execution status does not authorize this
new control. The supplied context is a caller assertion, not a replacement for
root review, numerical qualification or resource admission. No admission document,
caller harness or new manifest is produced here.

One completed physical member incurs **5400 training graph forwards, 5400
backwards, 2700 own native selection forwards and 1 native restoration-verification
forward**, plus its independent acquisition, graph preparation, checkpointing and
pool/storage work. Four members therefore incur all four acquisitions and sums of
these costs. No existing native-only member or copied-bank member is reused as a
fit or free donor.

Future execution writes per-update spent-pass traces, construction/local/final
images, result metadata and a `COST.json` receipt. Attempted operations are charged
even on failure; failed writes contribute their actual checkpoint/output bytes.
The receipt records wall time, parameter/storage counts, process-lifetime peak RSS
and selected-device CUDA peaks where applicable. The returned cost descriptor's
bytes must be added to `output_bytes_before_cost_receipt` for total output storage.
Runtime-loading time is marked as shared overhead to charge once per actual load.
Pooling reports its own time and output tensor storage. The Python `finally` receipt
cannot guarantee a write after an OS kill, power loss or disk failure; already
flushed traces and partial files must remain chargeable. Failure propagates; there
is no retry, member replacement or partial pool.

## Verification and readiness limits

Local stdlib verification passed: source parse/import; all 15 exact bindings;
all 12 inherited member seeds; each member's persistent assignment across all
2700 updates; inherited strict selector/earliest ties and 200/201 stage boundary;
default-disabled numerical/scientific entry; no numerical imports. These are
source checks, not native-gradient or runtime qualification.

`verify_cpu_components(rt)` is the single callable numerical source check. It uses
the exact already-bound native reference `test_numerical.py` fixture recipe, graph
and TRAIN labels. It freshly acquires four native models under the inherited seeds,
audits disjoint parameter storage and, for each assigned view in both native stages,
compares the actual production `accumulate_member_gradients` helper with separate
per-pass `autograd.grad` gradients under replayed RNG. The oracle directly uses
native and assigned-view complete-TRAIN CE, rather than the production pass planner.
It checks inactive heads, unchanged model/buffer values and absence of Adam moments.
Its existing-fixture tolerances are numerical verification tolerances, not research
settings. It reports all component forwards/backwards, four fresh acquisitions and
eight identical-state oracle copies. It performs **zero optimizer steps, zero fits**
and no file writes or saved outcome/checkpoint/data reads. Invocation after admitted
numerical loading is simply `verify_cpu_components(load_runtime(numerical=True))`.

The default local interpreter is
`/Applications/Xcode.app/Contents/Developer/usr/bin/python3`; it has no PyTorch,
NumPy or PyG. **No numerical gradient/ownership, full-shape execution, memory/time
qualification or predictive test was run by this source agent.** Root may run the
callable component check in its separately authorized CPU environment. Native optimizer/head gradient checks,
state restoration and transition functions are reused unchanged, but their reuse
does not qualify this new objective numerically. The provided CPU check covers the
objective and parameter independence, and does not qualify selected-state replay
under the new objective or representative full-shape resources. Common-runtime deterministic CUDA
qualification and whole-control resource admission remain necessary before a run.

No data, saved checkpoints or saved result contents were read; no remote call,
training, launch, extra tuning constant, architecture, selection policy, original
score edit or novelty claim was made. The required native/source/view bindings
are present. The missing item is runtime qualification and separate prospective
admission, not a scientific recipe to invent.
