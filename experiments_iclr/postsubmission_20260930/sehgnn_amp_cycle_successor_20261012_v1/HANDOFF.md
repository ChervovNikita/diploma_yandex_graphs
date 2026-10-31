# Native SeHGNN AMP accounting successor

This is an immutable disabled replacement for the predecessor native five-label
SeHGNN packet. It corrects optimizer accounting and restores native AMP overflow
behavior in the ordinary/native/original-GNNM wrapper. It does not change the
finite dataset, masks, HPO domain, comparisons, source operators or 1000 complete
epoch horizon. It is source preparation, with no scientific GO.

## Exact replacement

Replace the predecessor covered entry from
`sehgnn_typed_newdata_bridge_batch_20261012_v1/MASTER_INPUT_ENTRY_WITH_COVERAGE.json`
with this successor's `MASTER_INPUT_ENTRY_WITH_COVERAGE.json`. Both declare the
same scientific study ID; including both is invalid and double-counts all fits.
Preserve the predecessor packet and its proofs as historical preparation.
Use `massive_native_SeHGNN7_AMP_packet_20261012_v2` prospectively.

## Corrected native law

The original native source uses a CUDA GradScaler. A complete supervised epoch
can have finite output and BCE while scaled gradients overflow. GradScaler then
skips the optimizer map and reduces its scale. Complete epochs, owner requests,
and actual committed maps are distinct quantities.

The BasicSession wrapper previously rejected nonfinite unscaled gradients before
calling GradScaler. Its guard now runs only inside the optimizer step that
GradScaler actually dispatches. This restores the native legal skip path. The
existing source BankSession already followed that law. Joint-source SeHGNN is
explicit FP32 and keeps direct finite optimizer steps. Its learning law is
unchanged. No blanket claim that the previous wrapper was unchanged is made.

The adapter records every owner request, complete epoch, committed step and AMP
skip. The generic engine permits zero committed maps only for the explicit
native SeHGNN AMP contract: its known source factory, width 512, five labels,
25 feature channels, 12 TRAIN-label channels, one optimizer owner, the native
arm set, fresh state, and exactly 1000 complete epochs. Ordinary adapters still
require positive updates. The controller records bounds rather than an exact
AMP map count. Curvature trial bounds retain separate lists and counters.

On a due source-supply opportunity after a skipped own map, private after−before
displacements are exactly zero. The original helper scales correction magnitude
by their norm. It therefore returns a zero correction, while still charging
every due reference/replay/cotangent operation. No source correction work is
silently suppressed. An accepted nonzero correction on a skipped map is
inconsistent with that pinned source law and is rejected.

## Finite roster and accounting

The packet has 20 source arms, 9 LR/dropout points, 2 fresh HPO seeds, 3 fresh
comparative seeds and 10 custom development masks. It remains 1230 fit groups,
1518 optimizer owners, 3320 job records and 690 exported banks, scalar calibration
jobs and evaluations each. Genuine separately selected independent-four banks
remain 90; grouped source learners are not relabelled as those banks. Original
GNNM contributes 48 new-dataset fit records.

The fixed optimizer-request total is 1518000. BASIC/SUPPLY have 612 fit groups and
612000 AMP requests, with 0..612000 committed maps. FP32 JOINT has 618 fit groups
and 906000 direct maps on successful complete fits. Consequently the successful
complete source roster has 906000..1518000 committed maps. Failed/incomplete runs
retain their actual counts and are not claimed to attain those complete bounds.
The old 1518000 exact-map assertion is superseded.

TRAIN forwards before private trials remain 17253120; reverse calls 10074336;
SELECT forwards 3390000; leaf reverses 570240; private model VJPs 1374336;
private opportunities 47520; trial attempt upper 190080; trial forward
upper 2090880. These are registered source work and bounds, not observed scores
or a wall-time estimate. Scalar calibration has 345000 registered updates.

## Engineering evidence and limits

The successor passed 7 standard-library roster/controller tests and 5 narrow AMP
engine gate tests. Root independently scheduled the exact BasicSession epoch
method on a synthetic 1-to-5 projector with the native final LayerNorm and BCE,
using real CUDA autocast and GradScaler. All optimizer kernels were denied.
Its 16 complete cycles produced 6 genuine gradient-overflow skips/backoffs and 10
finite dispatches denied before any optimizer map. It read no graph or target
authority, created no scientific fit and emitted no predictive score. The root
closure proves the child ended and the existing Physics owner was unchanged.

The predecessor's full-width constructor/gradient/serving/state, source policy,
typed propagation, grouped-ownership and Bernoulli endpoint proofs are retained
with their original scope and hashes. They qualify those unchanged components;
they are not relabelled as evidence of newly executed full-width AMP training.
No real-input native resource bound, nonzero accepted-correction trajectory,
predictive quality or publication result is established here.

Data remain HGB native IMDB with five Bernoulli labels and no official TEST
access. The ten masks use TRAIN/SELECT/CAL/EVALUATE development roles. The native
200 epoch original-source reference remains separately disabled under its literal
80/20 recipe; it is not directly comparable to this primary four-role protocol.
Original paper experiments and scores are not rerun or changed.

Actual-input resource admission, checkpoint lifetime/storage policy, global
source/data freeze, three-GPU placement and explicit human GO remain gates.
No new scientific run may start from this handoff alone.
