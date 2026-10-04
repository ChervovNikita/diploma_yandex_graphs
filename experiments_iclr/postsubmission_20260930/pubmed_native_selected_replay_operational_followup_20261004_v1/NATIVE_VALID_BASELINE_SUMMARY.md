Native Pubmed TRAIN/VALID baselines — six cells, 2026-10-04

All six selected scientific states passed serialized replay; its supervisor completed with zero optimizer updates and no cap breach. These are native baseline results.

| Native model | Seed 0 VALID MRR | Seed 1 VALID MRR | Seed 2 VALID MRR |
|---|---:|---:|---:|
| SAGE | 0.0903 | 0.0734 | 0.0776 |
| NCNC | 0.0881 | 0.0789 | 0.0821 |

MRR is the native rounded four-decimal metric over 2,216 VALID queries with 500 ordered candidates per query. Each cell is the first rounded maximum under the fixed five-epoch VALID cadence.

| Fit | Selected epoch | Final native epoch | Serialized replay | Max absolute score difference |
|---|---:|---:|---|---:|
| SAGE_seed0 | 370 | 425 | PASS | 0 |
| SAGE_seed1 | 45 | 100 | PASS | 0 |
| SAGE_seed2 | 40 | 95 | PASS | 0 |
| NCNC_seed0 | 285 | 340 | PASS | 0 |
| NCNC_seed1 | 125 | 180 | PASS | 0 |
| NCNC_seed2 | 155 | 210 | PASS | 0 |

The native fits incurred 49,220 Adam calls and 270 VALID serves. Selected replay incurred zero optimizer updates and completed in 14.93 inclusive supervisor seconds. It checked exact restored/postserve state, RNG and per-query/rounded metrics under the prospectively fixed score tolerance.

The feature values are the released raw Planetoid/PyG values without NormalizeFeatures. The native negative candidate pool preserves replacement duplicates and other-VALID positives. This is one TRAIN/VALID baseline cohort: it establishes no TEST performance, transfer result, new GNNM result or novelty. Original-paper scores were not amended.

Launch metadata advisory: the suspected literal backslash-n suffix was unconfirmed. The preserved raw descriptor is valid JSON and ends in an actual newline; a separate normalized copy was reconstructed from authenticated stdout. The original was not edited, and launch was not repeated.
