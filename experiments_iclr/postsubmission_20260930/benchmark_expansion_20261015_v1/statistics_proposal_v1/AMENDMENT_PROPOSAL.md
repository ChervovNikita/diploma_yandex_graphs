# Discrete sign-flip constraint and proposed analysis amendment

15 October 2026. **Amended design proposal pending user review.** This audit
changes no frozen README, roster, source, analysis policy, fit count or pilot.
The 270 source-reference and 2,480 cold-fit design, and the running 24-fit pilot,
remain unchanged. No scientific outcomes, model/data records, TEST values or
server were accessed. Expanded-catalogue confirmation families are not yet fixed.

## What is actually declared

The current `reference_wrapper_redesign_20261013_v1` README, ROSTER and PLAN agree:
each declared GNNM procedure is compared with matched native M1 and genuine I4,
within each fixed preset, using ten paired trials. The declared primary metric
is that recipe's source task metric. Historical paper values and source-horizon
checks are not paired observations in these contrasts.

| Task | Fixed presets | GNNM procedures per preset | Versus M1 and I4 |
| --- | ---: | ---: | ---: |
| Node classification | 8 | 5 | 80 |
| Molecular graph classification | 16 | 4 | 128 |
| Link prediction | 3 | 3 | 18 |
| **Declared family** | **27** | **113 procedure/preset cells** | **226 contrasts** |

Equivalently, boundary GNNM, global half-pool and private half-pool each contribute
54 contrasts; fixed responsibility contributes 48; node SupCon contributes 16.
The audit reconstructs every contrast and verifies all ten actual PLAN bank
bindings, including M1/body0 reuse and I4's four own selected states. See
[REGISTERED_FAMILY.json](REGISTERED_FAMILY.json). “Declared” refers to these local
design artifacts, not a claim of external preregistration.

The reporting policy specifies complete two-sided sign-flip mean-difference
tests and Holm over all 226; it labels the resulting evidence exploratory.
It specifies 95% marginal paired percentile-bootstrap intervals, 10,000
resamples and fixed analysis seed130102026. It does not explicitly serialize a
test alpha. **This audit uses α=.05 as supplied in the audit request; an amended
protocol must state alpha explicitly.**

## Exact resolution prevents any Holm rejection

For n independently sign-flippable nonzero paired differences, the complete
two-sided tail includes an observed sign pattern and its global reversal.
Therefore `p_min = 2 / 2^n`. Zeros/ties can make the floor larger, not smaller.
Holm first compares the smallest p-value to `alpha/m`; if it fails, it stops.

At n=10 and m=226:

```
p_min                 = 2/1024 = 0.001953125
first Holm threshold  = .05/226 = 0.000221238938...
best first adjusted p = 226*(2/1024) = 0.44140625
```

Thus **no cell can be rejected by the stated Holm procedure**, even if all 226
p-values attain their best possible value. This is exact arithmetic independent
of effect size, effect variance or a favorable observed standard deviation.
More Monte Carlo permutations or bootstrap resamples do not create more trials.

| Nonzero sign-flippable trials | Smallest two-sided p | Best first Holm-adjusted p, m226 | First rejection possible? |
| ---: | ---: | ---: | --- |
| 10 | .001953125 | .44140625 | No |
| 12 | .00048828125 | .1103515625 | No |
| 13 | .000244140625 | .05517578125 | No |
| 14 | .0001220703125 | .027587890625 | Yes |
| 16 | .000030517578125 | .00689697265625 | Yes |

The necessary resolution condition is `2^n >= 2m/alpha`, giving n>=14 here.
This is **not** a sample-size or power recommendation. At 14, the first cutoff
allows only two of 16,384 sign patterns, so a nonzero mixed-sign cell cannot meet
that initial cutoff. Under an illustrative independent-sign model with .9
probability of a positive difference, unanimity occurs with probability about
.2288. That is the chance for one cell to meet this initial cutoff under that
toy model, not full Holm power or a fitted description of these algorithms.
One zero difference at n=14 reduces the effective resolution to 13.

## Honest options for a prospective amendment

**A. Freeze a smaller primary question family.** Preserve every planned fit,
comparison, failure and descriptive result, while choosing a scientifically
justified primary subset before confirmation readout. At n10 a family of at most
25 can make a first Holm rejection possible; it still may have poor power.
For illustration, 12 or24 prechosen questions can meet that resolution limit;
neither is selected by this proposal. Merely choosing original boundary GNNM
versus both controls across all 27 presets still leaves 54 tests; even one
comparator across all 27 leaves 27, so neither resolves the n10 limit.
Publish the full matrix as exploratory, with per-trial differences and **marginal**
95% intervals clearly labelled as not simultaneous familywise coverage.
Neither method, comparator nor preset may be selected to rescue observed p-values.
A disclosed VALID utility screen can select candidates for a later untouched-TEST
confirmation; it cannot retroactively shrink the family for the same development
outcomes being tested.

**B. Plan a new confirmation with more independent paired trials.** Keeping 226
Holm tests requires at least 14 effective nonzero sign-flippable units merely
for attainability. Select a scientifically meaningful effect target and assess
operating characteristics under plausible heterogeneity, ties, symmetry and
failure rates before choosing n. Use new prospectively fixed seeds/splits
as appropriate. More seeds on one fixed graph estimate algorithmic randomness,
not a new graph/scaffold population. Additional trials would need an explicit
resource/design amendment; no new count, fit or authorization is created here.

**C. Use a genuinely different hierarchical question only with explicit
assumptions.** A small dataset-level omnibus/gatekeeping family could answer a
different question, but presets sharing a dataset cannot become independent
replicates. Predefine dataset weights, metric scales/estimands, dependence and
valid gatekeeping/error allocation. Seven current datasets are a selected
benchmark, not a random sample of all graph tasks. Testing seven dataset-level
units would have a two-sided sign-flip floor of 2/128; even that limits how many
global tests are attainable. Splitting 226 tests into nominal-.05 families or
opening uncorrected cell tests after an omnibus pass does not preserve the
original global FWER claim. A dependence-aware joint test or parametric model
also needs a new justified model and multiplicity proof; it is not an automatic
shortcut or a switch motivated by tiny observed SDs.

None of these options is selected. The current inability to reject is not a
reason to reinterpret nonrejection as equivalence, nor to relabel old exploratory
results as confirmation. Statistical changes and their timing must be documented.

## VALID screening and untouched TEST confirmation

1. Keep the running pilot as VALID-based exploration. Its candidate comparison,
   epoch selection and mechanism readouts are not untouched confirmation.
2. Before accessing confirmation TEST outcomes, freeze methods, primary subset
   or hierarchy, estimands, alpha/multiplicity, trial units, metric directions,
   convergence/selection, missing-run handling and analysis code. Document which
   choices used VALID screening and the prior 226-test resolution constraint.
3. Train on original TRAIN and choose checkpoints with original whole VALID:
   one complete shared GNNM bank, four own native I4 states, exact M1/body0 reuse.
   Freeze checkpoint identities before the final authorized TEST evaluation.
4. Verify that TEST has not informed method, gate, coefficient, epoch or subset
   choices. Evaluate each frozen selected predictor on full official TEST once
   under the source evaluator; retain every result and failure. Repeated seed
   scores on the same TEST set do not create independent test populations.
5. A negative TEST result stays negative. No outcome-driven extra seeds, subset
   changes, test choice, selective suppression or retrospective “confirmation”
   label follows. Current work authorizes no TEST access or scoring.

See [ASSUMPTIONS.md](ASSUMPTIONS.md) for the test's actual assumptions and limits.

## Reproduce the numerical demonstration

From the project workspace, run with an ordinary Python3 interpreter:

```sh
python3 postsubmission_research_20260930/benchmark_expansion_20261015_v1/statistics_proposal_v1/verify_discrete_constraint.py
```

The stdlib-only script checks the declared banks and enumerates synthetic integer
vectors using exact fractions. [NUMERICAL_DEMONSTRATION.json](NUMERICAL_DEMONSTRATION.json)
contains the tail counts, Holm calculation, zero/tie example and trial table.
It reads design metadata only and writes only into this new proposal folder.

