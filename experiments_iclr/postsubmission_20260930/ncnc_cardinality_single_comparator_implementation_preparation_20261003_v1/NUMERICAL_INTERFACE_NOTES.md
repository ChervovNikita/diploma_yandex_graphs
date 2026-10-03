# Source arguments and remaining numerical checks

These are source-level arguments, not executed numerical receipts.

## Centered auxiliary derivative

The model computes u=t−mean(t) outside `LogESPAtObservedCount` and uses the
same u in the observation numerator. The custom function differentiates its
normalizer with respect to independent u: its reverse seed is one-hot K,
yielding the conditional inclusion vector μ given K. The centering chain
subtracts K/R from that derivative. The numerator's centering derivative is
Z−K/R, so the full fixed-context NLL derivative is (μ given K−Z)/R. The
count-head q-context path remains live and adds its permitted total gradient.

The custom backward stores u, offsets and K; it reconstructs and discards one
full query prefix at a time. It does not retain every query's DP table or
backpropagate through an impossible logaddexp(−∞,−∞) cell. Higher-order
derivatives are not claimed. First-order gradcheck remains pending.

## Actual marginal readout

The mixture reverse pass seeds the terminal count cells with detached pi and
treats u as independent DP inputs. It therefore obtains actual mixture
inclusion probabilities, without the erroneous centering subtraction.

For finite branches the logaddexp derivative is a two-term softmax; the source
computes its inclusion weight as sigmoid(include_branch−exclude_branch), with
the exclusion weight as its complement. For a probability readout, included
and excluded reverse mass are summed at every slot, and the result is
included/(included+excluded). With a unit-mass count law this is algebraically
the same marginal derivative. This log-law mass normalization keeps the
probability interval without clipping or a probability epsilon. The custom
auxiliary derivative uses the unnormalized reverse derivative, as autograd
requires. The prospective enumeration, marginal/count identity and gradient
gates must verify both paths against the declared law.

## Open uniforms and count CDF

The SHA key maps to an integer in0..2⁵³−1. The conceptual uniform is
(integer+.5)/2⁵³. Direct conversion of the upper midpoint can round to1 in
FP64. The source instead computes log U with ordinary log in the lower half
and log1p of the negative distance from1 in the upper half. No epsilon or
endpoint clamp changes the integer mapping. Endpoint and frequency fixtures
are still pending.

The count sampler uses a full log-CDF minus its terminal log mass, then closes
the terminal class at log1=0. This is exact normalization of the same pi law.
All feasible classes remain present. Each conditional slot key is assigned
before forced-count shortcuts. The only positive-log-inclusion correction is
the frozen≤1e−10 FP64 rounding guard; larger violations fail.

## Network and data boundary

Native neural maps and the added head are FP32. Candidate score outputs,
affinities, count log-softmax, likelihood and sampling arithmetic are FP64;
q context summaries are cast to FP32 at the head interface without detaching
their auxiliary gradient. Torch's scatter-reduce amax supplies the declared
equal max-tie subgradient. The exact-runtime fabricated gates must verify it.

Project teacher construction requires the literal authenticated complete TRAIN
tensor hash and full dimensions. Model/context/sampler interfaces accept no
teacher or labels. Both positive and negative target forwards precede the
teacher lookup. The teacher receives only fixed counterpart coordinates and
cannot change support. A forbidden-split teacher is rejected by the oracle;
the model has no oracle argument at all.

The stateless evaluation key depends on canonical target/counterpart pairs,
seed/version, draw and purpose. D4 and M4 share the slot key domain. Runtime
batch invariance and source-equivalent teacher mutations still require the
planned fabricated tests.
