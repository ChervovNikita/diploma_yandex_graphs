# Complete native PyG GNN-FiLM WikiCS comparator

The complete fixed9-bank/18-fit/45-calibration pipeline closed successfully. It used the same complete11,701-node WikiCS graph,580TRAIN and5,274development nodes, seeds7301/7403/7507, width128/depth2/dropout.2/AdamW.001/zero decay, ceiling1000/patience300, own-CE and probability averaging. All45five-fold temperature attempts preceded interpretation. No TEST was scored.

| Procedure | Raw accuracy % | Raw NLL | Calibrated accuracy % | Calibrated NLL | Stored parameters per bank |
| --- | ---: | ---: | ---: | ---: | ---: |
| FiLM_M1 | 77.2153 | 2.009551 | 77.2153 | 0.885407 | 340928 |
| FiLM_shared4 | 79.9836 | 1.345641 | 79.9520 | 0.731659 | 354146 |
| FiLM_I4 | 78.4161 | 1.253124 | 78.4035 | 0.774740 | 1363712 |

## Comparison with the complete E result

Existing E scores81.1971%. E minus shared FiLM is1.213500points, with paired seed differences.948047/1.687524/1.004930; E minus genuine FiLM I4 is2.780938points, differences2.483883/2.938946/2.919985. Every contrast is positive at allthree seeds. Raw/calibrated NLL and Brier also favor E over each tested FiLM procedure; all vectors and descriptive95%paired intervals are in E_COMPARISONS.csv. These are exploratory same-development comparisons, not unused confirmation or multiplicity-adjusted superiority tests.

The full native PyG2.7FiLM operator includes receiver-produced gamma/beta, messagewise ReLU before averaging, factual self records and native FiLM skip. It is richer than the earlier receiver-only N control and has different parameterization/geometry/capacity from E. E stores237,474parameters versus354,146shared FiLM; no capacity-matched or published-score reproduction claim follows. Different sampled functions, stopped checkpoints and selection histories remain.

Shared FiLM itself gains1.567438points over its genuine I4 under this fixed recipe. This is another observed shared-ensemble benefit, without establishing that FiLM is a state-of-the-art WikiCS reference or that E universally beats conditional filtering. The tested single/independent FiLM scores are lower than the capable E/N/SAGE references. A richer operator need not be a better predictor under one fixed setup.

## Retention and next use

The12compact evidence partitions preserve all9raw/calibrated banks, allclasses/error counts, each contrast, all45calibration details, checkpoint/archive/native-body/owner receipts and cost accounting. The full2.53MBreport remains on the server, with its bound hash. No checkpoint or logit arrays were copied to this Mac. Serving/acquisition costs overlap Physics and do not supply an isolated efficiency ranking.

Retain this complete comparison in the next manuscript revision together with Penn and eventual whole Physics outcomes. Cite GNN-FiLM as prior conditional-filter work; do not use these adaptation results to claim a new primitive, published-score superiority or a clear acceptance verdict. Do not tune its weak cells against these development outcomes inside this frozen study or silently rescue a failed criterion. No manuscript reviewer follows this individual closure.
