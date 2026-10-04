# Actual DDI TRAIN runtime result

Purpose: measure the complete native-sized training workload and the two exact conditional-pattern losses on the actual DDI graph before scheduling a predictive comparison.

All three fresh seed-0 arms completed one full epoch: 17 updates, 1,067,911 positive records and the 19,335-record final batch. Physical exit 0, owned-session closure and reap passed in each case. Initial states, all native sampling/RNG receipts and joint/separate support/teacher receipts agree across all 17 updates.

| Arm | Epoch seconds | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|
| target_only | 12.38 | 11.43 | 11.89 |
| joint | 226.07 | 12.36 | 13.01 |
| separate | 212.41 | 12.36 | 13.01 |

Linear TRAIN-only scaling of these co-resident measurements gives 187.9 GPU-hours for the preserved nine-fit 500-epoch proposal, or 37.6 GPU-hours for nine F4 fits at 100 epochs. Native M1 fits, VALID, replay and later-epoch uncertainty are additional. These are budget scales, not promised completion times or standalone benchmarks.

The auxiliary arms contain 544 selected positive and 544 native-negative queries per epoch, with complete supports and deterministic rows retained. Their actual population denominators are stored in ROOT_ADOPTION.json. This is exposure to informative reconstruction targets; it establishes no served ranking improvement.

No VALID scores, TEST data, donor state or checkpoint were produced. Runtime states were discarded. The scientific 500-epoch source remains disabled and preserved. A separate prospectively fixed 100-epoch development comparison is being prepared from these TRAIN costs before any DDI predictive score. It must report the shorter budget and use separately fixed competitive confirmation if promising.

The first bootstrap used a system Python without hashlib.file_digest and stopped before any remote launch. Its failed transport is preserved; the explicit recovery used the pinned Python 3.12 and the same reviewed release/workload. The original technical review does not cover that bootstrap adapter. No host settings or existing jobs were changed.
