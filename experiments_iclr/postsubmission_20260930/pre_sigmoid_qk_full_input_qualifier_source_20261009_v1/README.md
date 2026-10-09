# Full-input Q/K engineering qualification source

This immutable source packet prepares one normal-host invocation of the pinned
Q/K adapter. It executes no scientific pilot or reviewer decision. Preparation
performed text/hash/AST checks only: no server, data payload, torch execution,
existing-source edit or launch.

`qualify.py` imports only stdlib until the separately authorized callable runs.
Both the callable's `root_execution_authorized=False` default and the disabled
release reject before numerical imports or input loading. The seal stays
disabled. Root must inspect/commit this source, copy the release template into
a fresh execution-release packet, fill the manifest/program hashes from this
seal and set its explicit engineering authorization/review flags. No scientific
launch permission is supplied by that release.

Callable:

```python
result = qualify.run_qualification(
    release=reviewed_release,
    root_execution_authorized=True,
)
```

CLI on the already pinned allocation runtime:

```text
<RUNTIME.json python> qualify.py --release <separate enabled release.json> --release-sha256 <exact release hash>
```

The immutable adapter manifest is
`0e82d0c94d0f7bb5c2ca849aad0c5b37a7dac26c8d3eafa203ea4a8d6230187c`;
its program is
`be9974399e090b424c57fe4f77ba10126f575d8e89454945ce51f0f7c61b9631`.
The existing adapter checks the normal hostname, interpreter, PYTHONPATH,
GPU visibility/name and torch/numpy versions against the pinned RUNTIME.json.
This qualifier adds no mount inspection or other-process handling.

## Fixed engineering work

Engineering seed **20261091** is outside the pinned pilot and confirmation seed
lists. Twelve fresh cells are four operators crossed with single, independent4
and be_init. Each gets one full580-TRAIN streamed F update locally, followed by
one global update. This disposable two-update transition is not the scientific
100-local/1100-total schedule; it uses no VALID selector or scientific checkpoint.

The native_tied cells run first. For each kind, dropout-off full-TRAIN-query
logits before its local update and before its global update become the in-memory
references. The latter follows the same single local engineering update as
each other operator. Before each update, its same-kind/seed/stage logits must
differ by at most **2e-5 absolute**, with no post-update equality demanded.
Native-tied references use the unchanged native Q/K path. Full Q/K is value-copy
matched; delta and gamma start zero. Reference tensors are kept only in memory,
never saved as predictions or checkpoints. A failed native kind reference
blocks unmatched dependent updates; all cells/errors remain in the receipt.

The qualifier checks:

- Exact gated TRAIN/VALID bindings and complete native array fields. VALID is
  loaded only for schema/count/disjointness validation; zero VALID predictions,
  metrics or selection occur. No TEST role is available.
- CPU/CUDA/Python/NumPy RNG and persistent streams unchanged by installation;
  dropout-off identity passes preserve the streams as well.
- Original key weight/bias objects retained, unit global r/s, exact zero added
  scales, independently owned copy-matched query fields and exact added counts.
  The outside-bias formula is bound by immutable adapter/FactorLinear source;
  no synthetic affine fixture or extra projection oracle is introduced.
- Every actual parameter has one native Adam owner; independent bodies share
  no parameter. All global key/query/scale gradients, updates and Adam state
  are absent locally. Globally each such tensor has finite nonzero gradients,
  at least one changed coordinate and finite Adam state with its first active
  step. The full gradient-field counts and per-Q/K-coordinate fields are saved.
- Actual full graph/query/output/representation shapes, forward/projection
  events, source backward counters and every Adam call after all member/view
  backwards. Only one native large projection per layer/member is used except
  full_qk, which uses two. Hooked map/pair events are counted using that exact
  source multiplicity.
- Finite parameters/Adam, synchronized elapsed time, this process's RSS high
  water/CPU time, and reset stage CUDA allocated/reserved peaks. Construction,
  loading, failed/partial work and all engineering outputs are retained.

Success totals are 12 cells, **24 logical updates, 144 streamed member/view
forwards, 144 backwards, 48 Adam steps, 72 identity member forwards**, hence
216 complete full-node member forwards. Global Q/K event totals are 270 large
projection calls and 108 patched pair calls. Identity forwards carry no loss or
quality measurement. TRAIN CE is only the required update/finite diagnostic.

## Outputs and limits

One fixed fresh directory inside the phase/repository is used:
`pre_sigmoid_qk_full_input_qualification_output_20261009_v1`. It contains twelve
cell JSON receipts and REPORT.json, including errors and partial work. There
are no torch/NumPy payload files, checkpoints, scores, subprocesses, owner
queues, retries or resumes. Clearing this process's own CUDA allocator after a
discarded cell does not touch other processes.

This qualifier is engineering evidence for the fixed inputs, stages and source.
It does not qualify a scientific full driver, checkpoint selection/restoration,
serving selected scientific states, a resource envelope or operator accuracy.
Source seals and all existing study files remain unchanged. Root separately
owns execution and any subsequent scientific release.
