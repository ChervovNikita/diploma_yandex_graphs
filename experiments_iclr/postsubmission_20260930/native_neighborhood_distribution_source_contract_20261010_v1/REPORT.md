# Precise neighborhood-distribution source contract

10 October 2026. Semantics are ready for the requested inactive implementation;
numerical qualification and scientific admission remain root-owned. The complete
30-fit decoder reader reports failure of both accuracy and NLL criteria. That
exact recurrence remains closed without tuning or a calibration rescue.

## Definitive operation

CONTRACT.json is the implementation authority. Read current native logits Z and
the exact live tensor H entering the original output_linear after native output
normalization. Add no hidden/descriptor normalization, dropout, label-derived
scaling or second native forward. Use float32 native/projection/descriptor values.

For each candidate route learn one raw direction, normalize it with a 1e-8 norm
floor, project every H, and aggregate its complete factual incoming nonself record
multiset. Retain duplicates; remove every self record; add no identity record;
leave native graph support unchanged. Stable value sorting preserves original
record order at ties. At q=.25/.50/.75, interpolate at (d−1)q and subtract the
population neighbor mean. Append unscaled log1p(d). Empty descriptors are all
zero; a one-record row has centered quartiles zero and log-degree log2. Ties use
the stated ordinary stable-sort subgradient, with duplicate-source gradients
accumulated naturally. No neighborhood can be sampled, capped or dropped.

The candidate has four private direction vectors and four private no-bias 4×C
score maps, all maps zero at construction. Add descriptor times map to native
logits. Train .5 native CE+.5 corrected CE averaged over actual native routes;
serve mean corrected probabilities. Every direction/message/hidden/map/body
path stays live. The zero maps give native scores and unit initial native-gradient
coefficient, with initially zero correction-path direction/hidden gradients.
This is initialization plumbing, not a competence or utility guarantee.

## Exact five fresh arms

1. Shared4, one private direction and quartile map per route: candidate.
2. Shared4, private directions and [mean,population std,min,max,log-degree] maps.
   Population variance is mean((a−mean)^2); std=sqrt(variance+1e-8).
   Empty fields zero; nonempty constant/singleton std=1e-4. Five-input maps zero.
3. Shared4, one common direction with four private quartile maps. Native H_m
   differs, so sharing the direction does not imply identical signatures.
4. Factorized M1, four directions/twelve centered quartiles plus degree and H,
   jointly read by Linear(D+13,128,bias)→ReLU→Linear(128,C,bias). No added norm
   or dropout; first layer uses standard Linear initialization, output weight/bias
   zero. One corrected class vector, not four pseudo-members.
5. Genuine factorized I4: four separately fitted/selected copies of arm4, each
   with its own four directions and nonlinear residual. Root strengthened this
   reference from the predecessor's narrower one-direction members. It receives
   the same observed graph/features/labels with richer per-body learned views and
   readout capacity; there is no equal-parameter or equal-sorting-work claim.

Counts remain15 banks/24 optimizer and native-body fits over three paired seeds.
Closed plain native banks are contextual references, not a sixth fresh arm.
Actual-member loss reduction and selector remain native: whole raw corrected pool
for shared banks, own raw corrected accuracy for M1/each independent fit. Preserve
native and corrected raw readouts at the corrected-selected checkpoint. Strict
restoration includes directions/maps/residuals, optimizer and RNG states.

Projection draws use isolated local CPU float32 normal generators:
base+5000011+1000003*(4*body_index+direction_index). Candidate/moment directions
use body0/direction0–3; common direction uses0/0; M1 uses0/0–3; I4 uses body0–3
and direction0–3. The all-signature head uses isolated seed
base+6000017+1000003*body_index. Preserve native/factor/dropout streams exactly;
no orthogonalization, redraw, initializer selection or auxiliary seed grid.

## One prospective calibration policy

CALIBRATION_POLICY.json fixes the existing global-positive-temperature operation
as secondary evidence. Apply exactly the same one-scalar policy separately to
native and corrected selected readouts for all new banks. Use the prior fixed
five label-free folds (CPU permutation seed11709), fit each fold complement for
exactly500 float64 CPU Adam updates at lr.01, and score only its held fold.
Initialize logT=0; use the final update, with no candidate, epoch or rate search.
Scale each member's log probabilities by exp(logT)>0 and renormalize before
probability pooling. Preserve all raw metrics and source authorities first.

Calibration does not affect TRAIN, native acquisition, selectors, model weights
or TEST. No final all-VALID temperature refit is proposed. These are encountered
post-selection OOF calibration diagnostics because base selectors used allVALID;
they provide no fresh or whole-pipeline validation claim. Positive temperature
preserves each member's ranking/coverage and can change pooling ranks. Therefore
confidence gains remain separate from acquisition of new correct alternatives.
The old decoder stays failed. Root fixes the new primary utility gates before
qualification or training; no secondary readout may silently replace them.

## Attribution and limits

PNA establishes learned-message multi-statistic aggregation and degree scalers.
The matched moment arm is a scalar-projection adapter, not a full PNA reproduction.
FSW establishes projected-distribution/quantile-function embedding and discusses
cardinality/isolates; direct interpolated quartiles inherit no Fourier-embedding,
injectivity or Wasserstein guarantee. The October5 proposal is reused openly.
A positive small screen still needs competent source-native PNA/FSW references,
broader backbones and frozen unused whole-pipeline confirmation before broader
claims. Neighborhood statistics and quantiles are not new primitives.

The combination researcher owns the requested thin source using the pinned
native Family. This contract creates no new owner, launcher, retry lifecycle,
model/data computation, remote operation, TEST access or canonical-state change.
