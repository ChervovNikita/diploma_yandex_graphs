# Prospective checkpoint lifetimes

This proposal applies only to the future approved study. It does not delete or
rewrite any existing research output. The source module is a prototype and is
not installed into the frozen controller or scientific packets.

## Authorities and preserved history

Every fit keeps its exact run configuration, source/data custody, all TRAIN and
SELECT curves, selectors and prefix certificates, operation counts, timings,
resource observations, convergence summaries, failures, owner/retry history,
selected checkpoint receipt, and declared outcomes. The selected inference
state and member prediction bank remain available. A failed hypothesis is never
erased or represented as an unexecuted condition.

A complete resumable checkpoint and a serving checkpoint have distinct schemas,
paths and receipts. The serving checkpoint preserves exact model parameters,
all persistent buffers, mixed module modes, requires-grad flags, validation and
selection metadata, and the complete required adapter state. It omits the
optimizer and recovery snapshot copies. It cannot be used to resume training or
to initialise a paid continuation. The export reads a bound complete selected
checkpoint, writes a new authority, verifies exact serialization, and leaves the
source unchanged. The restore checks model/adapter equality and leaves the
fresh, unrelated optimizer unchanged. Per-family adapter state can contain large
tensors; its actual size must be measured rather than assumed negligible.

If a virtual prefix supplies a real serving bank, its distinct selected state
gets its own serving authority. Diagnostic-only prefixes keep their full curves,
selected metrics, certificates and original checkpoint receipts. Their optimizer
states do not remain indefinitely after their trajectory/dependency closure.
For the SeHGNN packet, the 200/500 prefixes are explicitly diagnostic-only and
never create additional heldout banks. This distinction is declared before fits.

## Full-state lifetime

| State | Retain full optimizer/recovery authority until |
|---|---|
| Unfinished or failed fit with an open retry | Its terminal ownership and retry disposition are closed. |
| HPO selected state | The entire applicable HPO cohort closes and all required SELECT histories/selection receipts are saved. |
| Comparative selected state | Its selected-serving equality proof and label-free prediction export close. |
| Warm/teacher/pretraining source | Every registered consumer that can require that exact state closes, including retry admissions. |
| Terminal source for NSD, growth, curvature or staged views | Every paid dependent closes; selected state must never substitute for the declared terminal authority. |
| Latest trajectory recovery | The fit is complete and cannot resume its trajectory. |
| Nonserved diagnostic prefix | Its source trajectory is complete and its diagnostic receipt/curve is closed. |

The lease calculation comes from the frozen dependency graph and each source
consumer's named checkpoint authority. A failed consumer with an open retry
keeps its parent lease. Unknown consumers, open admissions, or missing receipts
block release. Histories and the selected-serving authority are never release
candidates. Releasing a completed search loser changes its future ability to
resume the exact old optimizer trajectory; that limitation is explicitly
recorded. Reproducing its selected inference state and declared SELECT history
remains possible.

`release_plan` is an executable pure validator that emits an immutable draft,
with every complete-state candidate individually bound and every blocker named.
It performs no deletion. A future controller integration must derive its input
from the registry/DAG, bind this policy to the new source manifest, and add a
separate release receipt when the prospective operation occurs. It must not
replace an already bound complete-checkpoint receipt with a serving receipt or
claim that omitted optimizer bytes remain available.

## Label-free export timing

The current global comparison barrier unnecessarily holds label-free bank
exports until the entire master finishes. A prospective source successor can
allow an export after its own fit parents and selected authorities close.
Exports may read only their registered node/edge IDs and graph features; they
cannot read CAL/EVALUATE target files or emit predictive scores. CAL, EVALUATE,
their target custody, and any score-based decisions keep the whole-roster gate.
Saving label-free predictions does not authorise opening those predictions for
scientific selection or changing the frozen study. This is a new explicit gate
law, not an exception to an existing frozen plan.

The prototype `label_free_export_ready` encodes the narrow gate. Integration
requires auditing each endpoint loader to ensure it loads IDs separately from
targets. The current SeHGNN export already obtains selected logits without
target-file reads, but this proof does not automatically transfer to every
node/link/molecular/custom endpoint.

## Bounded storage admission

Before a fit starts, reserve its host's retained serving states, unresolved HPO
full-state cohort, leased paid-source states, live recovery/prefix state and
atomic write transient, prepared data, member banks, histories, and failed
attempts. Account for two simultaneous fits on each GPU and all GPUs on the same
host. A free-space snapshot is not a filesystem quota guarantee. Admission must
retain a declared free-space margin and refuse another fit when this bound no
longer fits. A source snapshot cannot be discarded to make room while its lease
is open. Offloading can preserve exact bytes inside the authorised repositories
when that route is separately available; it is not assumed here.

The old native SeHGNN tensor-only duplicated retention proxy is 12.224 TB. Its
single complete selected-checkpoint roster is 1.528 TB. Its exact selected-model
tensor proxy after full-state lifetimes close is 0.509 TB, before BN buffers,
adapter state, traces and other studies. Keeping every selected resumable state
forever is itself too large for the allocation's observed approximately 1.174 TB
free space. The storage improvement therefore depends on explicit full-state
lifetimes, not just removing identical recovery copies. The largest complete
grouped checkpoint proxy is 4.020 GB; the current atomic recovery write can need
an additional 16.081 GB per active grouped fit. Those live transients remain in
the reservation until a source successor changes the recovery format and proves
its recovery behavior.

A native HPO cohort has a 0.471 TB single-selected-complete tensor proxy. Its
completed nonselected terminal/recovery/diagnostic-only optimizer copies can
close independently, while all selected resumable HPO authorities remain until
the whole HPO cohort closes. After that closure, exact selected-serving states
remain and optimizer copies become release candidates. No whole-run storage
admission is claimed: most other families are still unpriced, and source parent
leases plus custom adapter state must be included.

## Proof and remaining integration

Five pure lifecycle/no-unblinding tests passed. A tiny CPU fixture preserved a
tied module, BN buffers, mixed modes, frozen flag, custom nonpersistent adapter
state and selector/private state exactly; it left the optimizer and full source
checkpoint unchanged. Wrong-fit/hash authorities were rejected, and the current
training restore refused the serving schema. No optimizer kernel, fit, score,
CUDA operation or deletion occurred. This qualifies the generic representation
mechanism only; it does not qualify every source family's constructor/restore,
input custody, resource envelope or paid warm-state reconstruction.

Minimal integration order: declare and freeze this prospective policy; add
registry-derived full-state leases and distinct serving receipts; audit and
enable label-free exports only; teach the endpoint loader to use its serving
authority while training/continuation loaders require full authorities; run
source-family restore/negative checks; integrate storage reservations; then
include the changed policy and exact source in the user's GO review. Existing
scientific evidence remains untouched throughout preparation.
