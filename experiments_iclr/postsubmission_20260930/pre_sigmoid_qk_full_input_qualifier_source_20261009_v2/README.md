# Same-state full-input Q/K qualification — V2

V1 and its execution artifacts remain immutable. Its non-native global
comparisons followed separate stochastic local updates, so they compared
different current parameter states. Those failures do not diagnose an operator
error. V2 corrects that reference construction, retaining **2e-5** absolute
tolerance, seed20261091, the exact scientific adapter, recipe, inputs, twelve
operator/kind cells, and one local plus one global F update per cell.

## Exact state matching

Before each update, this cell's one current model bank is put in evaluation mode.
V2 records all parameter object IDs/versions, batch tensor IDs/versions, Python,
NumPy, CPU/CUDA RNG and persistent member streams. Extra delta/gamma scales must
still be zero, and full-query maps must still equal their copied key maps; all
global coordinates have been unused during the local stage.

For the reference pass, every current GlobalAttn instance temporarily receives
`type(attn).forward` bound to that same instance and `qk_shared=True`. This is
the original hash-pinned native class forward. It uses the same current key
weights/factors/outside bias, values, local graph computation, norms, head,
inputs and member rows. It neither copies nor restores weights from another
cell. Independent-query copies remain installed and owned, but the native
reference does not use them.

The candidate's original instance-forward presence/value and qk_shared flag are
restored in `finally`, including on reference failure. Restoration is checked
before the candidate pass. Parameter and batch IDs/versions plus RNG/streams
must match before and after both passes. Both passes use the identical full
TRAIN-query batch; there is no backward, target metric, validation selection,
synthetic fixture or parameter perturbation in this identity check. Then the
unchanged streamed F update runs once. There is no equality requirement after
updates or between different operators' trajectories.

## Inherited checks and revised work

The full-input loader, unique optimizer ownership, zero installation RNG draw,
initial coordinate counts, local None gradients/unchanged coordinates/no Adam
state, global finite nonzero gradients/changed coordinates/first Adam step,
finite parameters/state, error retention and own RSS/CPU/CUDA accounting remain
the V1 checks. VALID is used only for role schema/count/disjointness validation,
with zero VALID forwards or metrics; TEST, scientific checkpoints and old
scientific scores are absent.

Native key projection hooks now cover patched operators' native reference
passes too. Candidate patched pairs still bypass those key-module forwards and
count their single direct projection. Each stage has **two identity member
forwards plus two TRAIN-view forwards per member**. Full success requires:

- 12 cells, 24 logical updates, 144 TRAIN forwards, 144 backwards and 48 Adam
  calls, as before.
- 144 identity member forwards and **288 total full-node member forwards**.
- **342 large Q/K projections**: 270 candidate/TRAIN events plus72 native
  reference events. Patched pair calls remain108.

V1's compact OBSERVATION01.json is bound as provenance only. Its observer
terminated cleanly, while the qualifier worker exited1 with report_complete=false;
the recorded inclusive worker cost43.72s and errors remain V1 evidence. V2 does
not reopen its server-only full report or introduce another audit family.

## Disabled callable and CLI

`qualify.py` imports only stdlib at module load. The callable's default
`root_execution_authorized=False`, the disabled release and seal reject before
numerical imports/input loading. Root must inspect/commit this packet and create
a separate enabled engineering release using its new program/manifest hashes.
Adapter/owner/source seals and scientific recipe are unchanged.

```python
result = qualify.run_qualification(
    release=reviewed_v2_release,
    root_execution_authorized=True,
)
```

```text
<pinned RUNTIME.json python> qualify.py --release <separate V2 release.json> --release-sha256 <exact release hash>
```

The only output is a fresh repository directory
`pre_sigmoid_qk_full_input_qualification_output_20261009_v2`, with cell JSONs
and REPORT.json. Reference arrays stay in memory. No source/data/torch/server
execution or launch occurred during V2 preparation; only text/hash/AST checks
were performed. No tolerance change, model retuning, retry/resume, old-file
replacement, mount inspection or other-process cleanup is supplied.
