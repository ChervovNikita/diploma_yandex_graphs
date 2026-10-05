# Complete Citeseer quality comparison and method decision

## What finished

All 36 prospectively fixed physical fits completed. The reviewed analysis authenticated their data, source, complete histories, native checkpoint selections and saved prediction identities. It reconstructed all seven model families from their fixed selected states. No model, seed, ensemble epoch or averaging coefficient was selected from this comparison. This is new TRAIN/VALID development evidence; original manuscript scores remain unchanged and Citeseer TEST stays unopened.

The setting is the complete released Citeseer-HeaRT validation benchmark: 227 positive links with 500 fixed hard negatives per positive. Each family has three paired seed blocks. NCN supplies the native encoder and common-neighbor predictor; four-member families average raw logits.

| Model family | Mean VALID MRR (%) | Block 0 | Block 1 | Block 2 |
| --- | ---: | ---: | ---: | ---: |
| Native single | 26.7811 | 28.2107 | 25.7340 | 26.3986 |
| Ordinary independent four | 27.9204 | 27.9145 | 28.8375 | 27.0093 |
| Shared four, no endpoint frames | 28.4115 | 29.6747 | 28.9351 | 26.6249 |
| Shared four, one common frame | 28.0053 | 28.3656 | 28.5359 | 27.1144 |
| Shared four, private frames | 28.1795 | 28.9888 | 28.5297 | 27.0200 |
| Single receiving all four frame features | 26.9768 | 29.1551 | 25.2954 | 26.4800 |
| Independent four, each with a frame | 28.0116 | 28.9130 | 28.1507 | 26.9710 |

Rounded values above are for reading. Full unrounded scores, histories and paired uncertainty are in the fixed analysis RESULTS.json. Three seeds on one validation-selected split do not establish statistical significance or graph generality. Every primary descriptive seed interval includes zero; all four Holm-adjusted sign-flip p-values are 1.0. The exact two-sided test has minimum p=0.25 at this sample size.

## Frozen candidate decision

The private-frame candidate clears the directional criterion against each of the four primary controls: positive mean and at least two of three higher paired blocks. Its mean improvements are +1.3984 percentage points versus native single, +0.2591 versus ordinary independent four, +1.2027 versus the four-frame single, and +0.1680 versus independent frame four.

It **fails the separate privacy/adapter requirement**. The mean difference versus the common frame is +0.1742 points, but versus unchanged shared four it is -0.2320 points. The latter is negative in two of three blocks. The prospective rule required a positive mean against both. Therefore this exact private-frame recipe is not promoted to confirmation as a supported accuracy extension. Neither favorable primary contrasts nor structural subgroups can override that decision.

## What can be learned

Unmodified shared four has the highest observed mean. Its +1.6304-point mean difference versus native single and +0.4911 versus ordinary independent four are encouraging descriptive results for the existing shared-ensemble recipe. The contrast with independent four is positive in two of three blocks. This is useful new task evidence, but it does not establish the new endpoint-frame method or turn the existing construction into a novel principle. It cannot be presented as held-out confirmation.

The result argues for retaining an unframed competent shared-ensemble control when testing the separate backbone-training hypothesis. Adding a compact interaction operation has not shown benefit here. No tuning, additional fitted arm, TEST evaluator or new confirmation run is authorized by this adoption.

The separate native higher-order qualification subsequently found absent dense-adjoint second derivatives on nonempty sparse paths. That is an implementation limitation of the current runtime, not predictive evidence against the backbone-training hypothesis. An explicitly declared exact constant-adjacency adjoint wrapper is the next source repair. Train-only adaptation and gradient-transfer interpretations remain attributed to prior meta-learning work.

## Provenance and interpretation

- Complete cohort terminal SHA256: `120f57e53377494a055a994405581fd9fb1fcfe0c5636bdf4446dd8735bd94f8`.
- Fixed analysis RESULTS SHA256: `d675f6ccd3d4d3eef1da88c445c82bc28c4ba2b664710b057e6630a491127ec6`.
- The 36 fits retain one prospectively reused original native baseline and 35 remaining unique fits.
- Summed inclusive physical-fit time is 4,654.69 seconds. Runs coexisted with Amazon training; these measurements are not dedicated throughput comparisons.
- Structural and frame-map descriptions remain the prospectively fixed diagnostics. They cannot establish prediction diversity or rescue a failed aggregate comparison.
- No novelty clearance, confirmed superiority, manuscript acceptance, original-score change or TEST access is claimed.
