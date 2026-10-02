# Photo17 CE finite-difference failure: independent focused analysis

## Assessment

**FP32 scalar-loss cancellation is quantitatively plausible, not proven.** The useful next step is a paired precision diagnostic at the same cold point, directions, epsilons and tolerances. The original failed receipt and registry terminal remain failed.

Read exact sealed `qualify_gradient_interface` source and Photo17 JSON/text receipts only. No arrays, scientific imports/execution, SSH/GPU work, source edits or gate reversal occurred. Initializer SHA remains `1a8036c7bbf9f2b831747f99d3aa2dfdabc31cb636cd41ccad9457206a70cbcf`.

## What the evidence establishes

The helper uses a FP32 softmax CE residual, its VJP, a double-precision final `g·d` contraction, and two central differences of **already reduced FP32 CE scalars**. Direction 2 fails both CE checks; directions 0/1 pass. Maximum JVP/VJP dual error is `6.5862e-7`, maximum centered-logit FD error is `0.0011444`, and the retained Adam-equivalence receipt reports passed. These other checks do not prove the small CE projection is accurate.

| Direction 2 | ε = 0.001 | ε = 0.0003 |
|---|---:|---:|
| Analytic CE derivative | −0.001110623789 | −0.001110623789 |
| Recorded FP32 CE FD | −0.001251697424 | −0.001390774967 |
| Recorded relative error | 11.27% | 20.14% |
| Expected loss difference `2εa` | −2.22125e−6 | −6.66374e−7 |
| Implied recorded loss difference | −2.50339e−6 | −8.34465e−7 |
| Loss-difference discrepancy | −2.82147e−7 | −1.68091e−7 |

All six recorded CE FD values exactly reconstruct from integer multiples of `q=2^-23≈1.19209e-7`, followed by FP32 reciprocal multiplication. Direction 2 corresponds to −21q and −7q. This is a scalar arithmetic identity; baseline/endpoint losses were not recorded, so it does not prove their binade or the cause. At these epsilons, one q becomes derivative steps `5.96046e-5` and `1.98682e-4`: 5.37% and 17.89% of the small analytic derivative. The observed discrepancies are only about 2.37q and 1.41q in loss units. Comparable absolute scalar errors are harmless relative to the much larger derivatives in directions 0/1.

The worsening at smaller ε and this scale strongly support loss-evaluation/reduction noise as a hypothesis. Under an otherwise exact FD/analytic reference, pure final rounding of two correctly rounded losses in one `q` binade alone would bound difference error by q, below the observed gaps; preceding FP32 CE/reduction error, forward rounding, nonlinearities or derivative error may contribute. For close FP32 scalars, subtraction itself can be exact by Sterbenz's lemma: the problem is the accuracy already lost before subtracting. **Casting only the two final FP32 scalar losses to double cannot recover it.**

## Minimal principled diagnostic

1. Evaluate the same FP32 model/state and the same plus/minus logits once. Keep original results beside diagnostic results for every existing direction/ε; change no seed, step, tolerance or selection rule.
2. For a literal **reduction/difference-only** change, compute `F.cross_entropy(logits, labels, reduction='none')` in FP32, then convert that per-example loss vector to FP64 before its mean and scalar difference. This removes FP32 global-mean quantization while preserving FP32 per-example CE. It may leave per-example/forward noise.
3. `F.cross_entropy(logits.double(), labels)` is a useful stronger comparison, but also promotes log-softmax/per-example CE arithmetic. Describe it as **FP64 CE evaluation on FP32 forward logits**, not only a reduction change. Leave the model forward, perturbations and training loss unchanged.

Analytic consistency is necessary. Retain the original `a_old=g32·d`. Also record the loss-specific JVP contraction `a32_JVP=<r32,Jd>` with a double final sum; random-cotangent duality does not guarantee that particular near-canceling projection. For each diagnostic loss definition, use its own matching logit gradient and `Jd` to obtain `a_matched`. For full FP64 CE this is `r64=(softmax64(z)-one_hot)/n` and `a_matched=<r64,Jd32>` in FP64. Report old-versus-matched analytic differences and FD-versus-each reference. Agreement only with a newly changed analytic target does not establish agreement with the original derivative.

## What a new pass means

If higher-precision aggregation fixes FD while the original and matching analytic references agree, that supports a measurement-precision explanation for this failed local check. If it does not, inspect the loss-specific JVP/VJP projection and remaining forward/per-example noise; do not loosen tolerances or search directions/epsilons to obtain a pass.

A pass would validate the modified diagnostic at this cold point in three directions. It would not retroactively pass the immutable original FP32 check, prove exact derivatives of the rounded FP32 training program, qualify actual-warm states or all parameter directions, or establish training/resource/utility performance. The original failed claim stays preserved. Any later qualification-method amendment must state the changed numerical measurement and its bridge to the unchanged FP32 training gradient.

`RECEIPTS.json` binds source/failure/trace metadata and scalar calculations. `REPORT.json` contains the structured assessment; `OUTPUT_MANIFEST.json` binds this analysis.
