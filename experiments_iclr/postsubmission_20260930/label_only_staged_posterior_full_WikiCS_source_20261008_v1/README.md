# Disabled staged label-posterior source for full WikiCS

Method frozen 8 October 2026; source completed 9 October 2026. This packet prepares one fixed prospective adaptation. It is
source only: no numerical import, dataset/checkpoint loading, fixture, native
construction, CUDA call, training, current outcome-file opening, server operation or launch
was performed during preparation. The final seal verifies payload bytes; it does
not qualify execution. Root must review the source, qualify the complete input
on the declared GPU and separately decide a scientific release.

## Fixed native state and label training

For each native seed 6101, 6203 and 6307, restore the authentic **own-selected**
native state identified in root's pinned metadata. Its selected epochs are 143,
135 and 154, all in global mode. These are native parameter epochs, distinct from
the new label-stage epochs. The acquisitions came from root's screen v2 source,
whose exact run identity is checked against the metadata. They are not compared
to the old screen v1 source identity.

Restore native model, Adam and private streams; freeze all native parameters.
One eval forward captures the full active-head H `[11701,512]` and native logits
`[11701,10]`. The hooks are removed immediately. Save `FIXED_CAPTURE.pt`, with
native/context identity and tensor digests. Every later label update and VALID
selection uses this same H/logit pair. Native training and further native forwards
are disabled on the resulting stage. Selected reconstruction loads this cache
and performs no new native capture.

The existing sealed label operator and its training executor are reused with a
**zero** `[11701,10]` base-logit tensor. Thus the old residual executor computes
own masked label CE without a native-logit offset. It is not a newly implemented
attention operator. H remains detached and the message values contain permitted
TRAIN label embeddings only. All incoming nonself records stay in the attention
denominator, including nodes with zero label values; edge order and multiplicity
are preserved.

Each update draws a common 290-of-580 Q, from the isolated CPU mask stream seeded
by `native_seed+1900001`. Every bank receives the same ordered Q and literal
visible TRAIN complement. All Q labels are absent from every label value field.
The original value scale `579/290` is applied once during training; serving uses
scale 1. Each bank completes 1100 updates. Every update serves all 5274 development
nodes and its entire bank. There is no local-stage reset during label training.

| Arm | Predictions | 64D heads | Backwards/update | Adam calls/update | Learned parameters |
|---|---:|---:|---:|---:|---:|
| C4 | 4 | 4 | 4 CE/4 | 1 | 76328 |
| S_joint4head | 1 | 4 | 1 CE | 1 | 349184 |
| U4_sharedB | 4 | 4 | 4 CE/4 | 4 | 283648 |
| S_one_path | 1 | 1 | 1 CE | 1 | 70912 |

The capable joint single concatenates four untied 64D label messages, then uses a
bias-free `256→256→ReLU→10` readout. Its final weight is zero initially. The fresh
instance's original linear readout is replaced and its full Adam owner rebuilt;
no original source or graph/TRAIN buffer is modified. C4 uses the native seed;
control heads use `native_seed+[0,1009,2018,3027]`. The nonlinear readout constructor
uses isolated seed `native_seed+700001`.

## Actual serving and selection

If a node has no permitted TRAIN nonself neighbor, every route and its pool return
the native probability exactly. Otherwise each route serves

`p_route = (1/5) p_native + (4/5) softmax(label_logits_route)`.

The family serves the arithmetic mean of these actual route mixtures. The joint
single receives the same 4/5 label-family weight despite having one posterior.
The support boolean is structural and identical across arms; there is no learned
confidence gate or coefficient/horizon grid. NLL uses log-softmax, log-add-exp and
log-mean-exp without clipping. In exact arithmetic the native floor gives
`NLL_served ≤ NLL_native + log(5)`; this supplies no accuracy or calibration guarantee.
Member metrics describe the **actual bounded mixtures**, not raw label posteriors.

Each arm selects its strict first maximum complete VALID correctcount over all
1100 label updates. Its checkpoint contains the coherent entire arm parameter
bank and Adam state, exact native/capture/source binding and selected label epoch.
Native own-selected states remain fixed throughout this selection opportunity.

## Complete-family gate

Mechanical COMPLETE receipts contain source/work/state/capture hashes, without
selected score values. `verify_closed_family` requires all three seed blocks and
all twelve bank endpoints, 1100 complete updates/VALID events, declared actual
head/backward/Adam counts, one native capture, no native update or learned restore,
and exact selected/cache/private-metric file hashes. The collector reads numerical
metrics only after this full barrier. No seed, arm, epoch or subgroup can rescue
failure; one-path is secondary.

C4 must pass **both** S_joint4head and U4_sharedB contrasts:

- Accuracy gain is positive in each seed; its equal-weighted three-seed mean is
  at least 0.2 percentage points.
- Mean served NLL delta is at most 0.
- Mean of the three seed-level mean-member accuracy deltas is at least −0.1 pp.
- Mean of the three seed-level **min-member** accuracy deltas is at least −0.2 pp.
- Mean of the three seed-level mean-member NLL deltas is at most 0.01.
- Mean of the three seed-level **max-member** NLL deltas is at most 0.02.

The worst-member safeguards compare each seed's minimum/maximum first, then average
its contrast over three seeds. They do not flatten all seed/member observations.
A single prediction defines both its mean-member and worst-member metrics.

## APIs and restoration custody

All numerical entry points default to refusal before dependency, data or checkpoint
loading. Importing `stage.py`, `posterior.py` and `collect.py` loads stdlib only.
Own MANIFEST/SEAL and every bound predecessor payload are checked before later
execution. `route` requires the pinned normal repository, host, Python executable,
PYTHONPATH, GPU UUID/name, sole visible GPU and Torch/NumPy versions.

- `load_roles`: exact role/vendor files and the original public TRAIN/VALID loader.
- `make_stage`: complete fresh banks for engineering or scientific purpose; only
  selected serving may reconstruct one arm, and it must supply the fixed cache.
- Stage `draw_common_masks`, `train_step`, `serve_ids`, `capture_state`, `snapshot`:
  qualification and coherent label-stage primitives.
- Stage `evaluate`: allowed only for separately admitted scientific purpose.
- Stage `restore_learned`: engineering snapshot restore, or selected serving only.
  It copies learned parameters/Adam while preserving live mask RNG, bank logical
  steps, counters and native RNG. `parameter_label_epoch` records the restored
  coordinates separately from live `label_epoch`. Scientific training rejects
  restoration; exact scientific resume is not provided.
- `run_complete`: one fresh, separately admitted seed block; no retry/resume.
- `reconstruct_selected`: trusted in-memory state/cache trees only. Caller owns
  their prior exact file-hash custody; tensor/native/context digests are rechecked.
- `reconstruct_selected_files`: recommended file adapter; checks all twelve
  mechanical closures and exact receipt-pinned selected/cache bytes **before**
  trusted deserialization, then restores selected serving without a native forward.
- `collect_family`: separately admitted full-family metric opening and frozen gate.

Engineering snapshots are unscored (`metrics=None`, `selector_performed=False`).
The role loader deserializes and validates VALID y; engineering stage evaluation
is refused. Serving can use development IDs without scoring their truth. The source
does not itself provide an owned worker/watchdog or an automatic root release.

## Cost records and limits

Historical native acquisition already spent the original complete 1100-update
schedule and native selection work. The restored snapshot's logical steps describe
its selected epoch, not the full acquisition. Root metadata attests completion;
the historical elapsed-time and memory join remains pending. Native work is neither
free nor a fresh same-hardware comparison.

The new driver times role hash/validation/loading, source verification/runtime
checks, native checkpoint hashing, native construction, trusted deserialization,
custody checks/restore, fixed capture/cache transfer and tensor hashing, fresh label
banks, capture export/save/hash, label training, complete VALID scoring, selected
parameter/Adam copying and serialization, and private trace/receipt I/O. Total new
elapsed time begins before role loading and excludes only the final COMPLETE write.
CUDA allocation/reservation and process peaks are recorded without resetting them;
their scope is explicitly this process since its last external peak reset. These
are future instrumented measurements, not observed resource bounds or results.

Root must establish actual complete-input CUDA feasibility, nonlinear Adam ownership,
mask/exclusion/RNG/gradient guards, live-counter learned restore, cache-based serving
reconstruction and owned resource limits before a scientific launch. Full 1100-update
quality and runtime remain unqualified by source parsing.

## Structural and interpretation limits

Root supplied a structural diagnostic: 2512 of 5274 development nodes have zero
visible TRAIN-label classes and 1733 have exactly one. In C4 own-native errors of
the one-class cohort, the true class was present in only 49/276, 62/270 and 49/266
cases across the three seeds. These are root-provided observations, not new file
reads or outcomes produced by this packet.

For a fixed linear route and a one-visible-class context, its label logit vector
is a positive attention-mass scalar times one fixed class-logit vector in exact
arithmetic. Attention alone cannot change that route's preferred class before
native mixing. The route pool and native mixture can still change final preference;
the joint nonlinear single can combine differing head strengths. This limitation
remains in the declared one-hop source. No gated-value amendment was made.

The test concerns the **entire staged adaptation**, including frozen native state,
posterior CE and bounded serving. It does not isolate loss versus aggregation
causality. H already encodes native TRAIN supervision; literal Q masking does not
create out-of-sample native errors. The 1/5–4/5 map does not establish independent
votes, Bayesian precision, novelty, global superiority or unused confirmation.

U4_sharedB is a same-native-state mechanism control. It is not an ordinary
independently acquired four-native ensemble, and three seed blocks cannot supply
four members. Genuine I4 availability and custody audit remain pending. Existing
failed C4, original CS06 and plain-native/paper endpoints are preserved. This packet
changes no sealed predecessor, canonical memory, ledger, publication or running job.

Source ancestry is pinned to the existing native integration, C4 core, same-operation
controls and public interface. The sealed scout and thin posterior/reference
assessment informed the fixed method; no new primary literature was read.
