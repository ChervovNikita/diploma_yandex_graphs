# Closed WikiCS private-correction comparison

All 21 endpoints are now complete: seven prescribed methods and seeds 17, 29, 43. The final four continuation/private-path endpoints completed on 18.77 with clean exits, no retries, and original native prefixes reused. Source model, recipe, and original paper scores were not changed.

The primary candidate E_stage adds four sequential private graph corrections to one frozen native Polynormer. It was required to beat every quality reference at its complete prescribed endpoint, including continued single and ordinary independent ensemble. This is selected development evidence on the 5,274-node validation union stopping population, one WikiCS graph/split.

| Method | Seed 17 | Seed 29 | Seed 43 | Mean accuracy (%) | Mean NLL |
|---|---:|---:|---:|---:|---:|
| E_stage | 80.3565 | 79.4463 | 80.9253 | 80.2427 | 1.7055 |
| E_joint | 80.4323 | 79.5791 | 80.8874 | 80.2996 | 1.6612 |
| E_own | 80.5461 | 79.5411 | 80.6788 | 80.2553 | 1.6884 |
| S_continue | 81.7975 | 81.4941 | 81.5700 | 81.6205 | 1.1663 |
| S_paths | 80.5082 | 79.7876 | 80.1858 | 80.1605 | 1.5263 |
| I_native | 82.1767 | 82.1578 | 82.2146 | 82.1830 | 0.9575 |
| U_stage | 81.4562 | 80.9632 | 81.4372 | 81.2856 | 1.4231 |

## Decision

The candidate loses to the continued single in every paired seed and does not meet the primary quality requirement. This same-host comparison is sufficient to stop promotion of this recipe. It also loses to the independently acquired ensemble in every seed, and neither sequential freezing nor the served-residual objective shows its required all-seed advantage over E_joint/E_own. Preserve the result and stop this exact recipe; do not add tuning or select a favorable reference.

The candidate-versus-I_native/U_stage comparisons for seeds 29/43 cross runtimes: common arms were trained on the allocation, while the independent prefixes and their continuations/private corrections were trained on 18.77. They are descriptive and do not isolate parameter sharing. I_native versus U_stage is on the same host within each seed. The identical scientific run.py bytes are necessary provenance but do not erase runtime differences.

Native continuation keeps its prescribed first-best selector across acquisition and continuation. Changed residual predictors use their fixed final endpoint. These are complete methods with different selection rules, not a causal intervention on just one architectural part. Three seeds on one selected development population provide no held-out confirmation, graph-level replication, or acceptance claim. Interval summaries are exploratory seed-level df2 t intervals; no nodes or members are treated as independent fitted-model replicates.

## Provenance and cost

ALLOCATION_OBSERVATION.json contains the 17 original FREEZE records and whole owner closures after observed owner termination; recorded probability files were hashed without model forwards. COMPLETED_READOUT_OBSERVATION.json in the successor release contains the remaining four original records and clean complete closure. Their checkpoints/probabilities remain server-side. This is metadata readout, not independent checkpoint replay. Exact historical start ticks are not invented where unavailable.

The two 18.77 lanes each charged about 165.5 seconds for I_native and 45.1 seconds for U_stage after previously paid four-model 1,100-epoch acquisition. These marginal times exclude original prefix training, and do not imply a total-cost advantage. Earlier costs, failures, acquisition selections and model scopes remain in their original records.
