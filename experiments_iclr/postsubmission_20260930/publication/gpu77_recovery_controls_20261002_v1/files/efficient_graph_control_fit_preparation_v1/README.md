# Efficient graph controls: runnable fitting preparation

**Source-only, sealed and unexecuted.** This packet adds a fitting API around the unchanged native/port sources. No model, scientific import, dataset, label, checkpoint, GPU, SSH or experiment launch was used in preparation. It changes no live cohort or scientific decision.

## Fixed controls

Seeds are **17, 29, 43**, paired with source splits **0, 1, 2**. Contexts remain SquirrelFiltered/PolyFormer-Mono and AmazonPhoto/Polynormer-r. Every control is cfg0, with no tuning grid.

`cfg0_independent_m4` is an aligned independent-parameter control, **not an exact native independent-ensemble reproduction**. It selects all four members jointly at one common epoch using pooled validation NLL, rather than selecting four checkpoints separately. Mean-member CE scales each private member's gradient by **1/4** relative to standalone CE training. Adam epsilon means exact native optimizer equivalence is not implied by that scaling. These choices are fixed; no separately selectable variant is exposed.

| Arm | Squirrel width / complete parameters | Photo total width / complete parameters |
|---|---:|---:|
| Competent native single | 256 / 4,419,437 | 512 / 7,762,960 |
| Same-width independent M4 | 256 / 17,677,748 | 512 / 31,051,840 |
| Fixed-cap untied M4 port | 116 / 4,177,828 | 248 / 7,707,904 |
| Cached-token MIMO M4 | 208 / 4,308,428 | Not prepared |

The two fixed-cap targets are 4,430,644 and 7,773,732 complete parameters. The next legal widths exceed those caps. Actual constructed counts are checked before role validation or selection.

The fixed-cap arm requires `backend="vmap"` or `backend="sequential"` explicitly. Root must choose one before fitting. The latter is an untied sequential reference and carries no efficient-packing claim. Unsupported vmap raises a visible failure; there is no backend retry or fallback. One declared backend gives **21 cells**. `CELL_PLAN_DRAFT.json` illustrates vmap and remains unadmitted.

## Schedule, selection and handoff

- Squirrel: at most **2,000 training updates**, patience **250**, plus an eligible epoch-0 evaluation.
- Photo: exactly **200 local + 1,000 global training updates**, plus an eligible epoch-0 evaluation in each stage. Before global activation, restore the selected local **model AND Adam state**, verify exact tensor/metadata restoration, and check its validation NLL. The global selector starts fresh; the final checkpoint must be global.
- Primary selection is **validation NLL from softmax(mean raw member logits)**. Improvements use strict `<`; exact ties keep the earliest checkpoint. All validation NLL and logits must be finite.
- Uniform epoch-0 eligibility is a **declared control-protocol adaptation**, not native reproduction. It can select the unupdated local or global stage; the training schedule still runs as declared, subject only to Squirrel patience.
- Independent and fixed-cap M4 loss is mean member cross entropy, with the private-gradient scaling disclosed above. MIMO retains **mean tuples, sum matching heads**, hence four times the mean-head loss; this is disclosed and is not silently normalized. Native Adam settings and attention-specific PolyFormer LR/decay are retained.

Published probability pooling is adapted to raw-logit pooling for the common primary selector. Mean member probabilities remain a secondary reducer from the same selected logits and cannot change selection. Singleton pooling coincides. The prepared cached-token MIMO is a graph control adaptation: complete cached polynomial rows act as examples, tuple slots independently permute compact TRAIN positions, and each position appears once per slot per update. Labels come from those same compact positions. Inference repeats the complete target token row in all slots. No node tuples are applied to unchanged graph topology.

## Runtime API

Add this packet directory first on `sys.path` in the future root-owned runtime, then import `control_protocol` and `fit_controls`. Those two imports use only the standard library. Source sealing and admission checks precede scientific imports in `fit_control`.

```python
from control_protocol import CompactRole, PreparedGraph, specification
from fit_controls import fit_control

# Repackage ONLY fields from the caller's existing prepared graph and role packs.
graph = PreparedGraph(existing.teacher_backbone,
                      existing.teacher_input, existing.teacher_edge_index)
train = CompactRole(existing_train.nodes, existing_train.labels)
validation = CompactRole(existing_validation.nodes, existing_validation.labels)
spec = specification("polyformer_mono", "fixed_cap_packed_m4", 17, backend="vmap")
selection = fit_control(spec, graph, train, validation,
    admission=root_verified_admission, mode="full", out=fresh_root_attempt_directory)
```

No loader or preprocessing function exists here. Only the exact prepared-graph type and two compact role types are admitted. TRAIN and validation are nonempty, unique, disjoint, int64 packs with matching labels on the input device. The graph has the retained full-context FP32 dimensions. The API has no final-label argument and saves no final labels.

Input identities bind tensor shape/dtype and contiguous CPU bytes. For each tensor, `H = SHA256(canonical_json({shape,dtype}) + NUL + tensor_bytes)`. Graph identity is `digest({backbone, inputs:H, edges:H_or_null})`; compact role identity is `digest({nodes:H, labels:H})`. `digest` is canonical JSON SHA256 as defined in `control_protocol.py`. PolyFormer ignores its graph edge field because inference consumes the existing complete cached rows; Photo binds coherent edges. Root verifies role/split provenance as well as these content identities.

`mode="qualify"` uses three Squirrel updates or two local plus two global Photo updates, and never issues a qualification pass or report admission. This is a full-context fit path and is gated by graph-init closure. It does not replace the independent root qualification checks.

## Root prerequisites

1. Close the current graph-init study. Bind a genuine closure receipt. No full-context qualification or fitting before closure.
2. Freeze the scientific control decision, one packed backend, fixed role/split bindings, and this manifest/protocol/cell identity. The draft template is deliberately inadmissible.
3. Bind the actual Torch/PyG versions, device and environment fingerprint: Python/CUDA/cuDNN, GAT source hash, deterministic/TF32/thread/CUBLAS flags and CUDA device properties. `_runtime_binding` defines the exact fingerprint; preparation has not collected or qualified it. Retained expected versions are Torch2.1.2+cu118 and PyG2.7.0, with compatibility unresolved.
4. Qualify actual parameter counts, finite training/derivatives, selection/ties, selected checkpoint restoration and input/role custody. Photo requires both stages and selected local model/Adam handoff. vmap additionally requires output/gradient/two-step Adam equivalence, member isolation, distinct training dropout and forward/backward support. MIMO requires complete tuple/token/compact-label correspondence, per-slot exposure, inference repetition and summed-head loss/gradient checks. `qualifications.py` and `test_ports.py` are unchanged prepared hooks; unsupported-test skips are not passes.
5. Root verifies genuine receipts and issues full-fit admission for each exact cell/runtime. This adapter validates assertions but cannot authenticate them or make the scientific decision. Run each attempt under root supervision and an outer wall cap; preserve every failed attempt and all outer costs. SIGKILL/process loss requires a root terminal receipt because an in-process handler cannot run.
6. After all control attempts close, admit any analysis/report separately. Every adapter output remains `report_eligible=false`.

## Saved outputs and costs

Each fresh attempt writes `START.json`, runtime/input binding receipts, `TRACE.jsonl`, a durable selected state for each stage, `SELECTED_CHECKPOINT.pt`, `SELECTED_MEMBER_LOGITS.pt`, `SELECTION.json`, `COSTS.json`, and a terminal binding their hashes. Selected states contain both model and Adam state plus RNG custody metadata. Photo retains `SELECTED_local.pt`. Training RNG continues across the local handoff, matching the retained fitter; saved RNG is not silently replayed at that transition.

Costs cover construction, input binding, epoch-0/selection validation, training, MIMO tuple validation, state snapshots, Photo transition, final restoration and logits serialization. GPU intervals synchronize the one input device; CUDA peaks and process CPU/RSS are recorded. RSS is the process lifetime high-water mark. Root separately charges existing preprocessing/cache materialization, process startup, source/admission checks, final receipt hashing/serialization, qualification and all failed attempts. No speed or memory advantage is established here.

Catchable failures write `FAILURE.json`, costs and a failed terminal, preserving prior selected checkpoints and partial trace. They are re-raised with no retry/fallback. Storage failures or process termination require root supervision to supply closure. Source checking uses `python3 source_checks.py` only; it does not import or run the scientific files. `SOURCE_CHECKS.json` reports the precise source-only evidence and limitations.
