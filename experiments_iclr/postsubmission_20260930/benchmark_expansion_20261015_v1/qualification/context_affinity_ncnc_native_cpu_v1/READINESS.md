# NCNC common/split context: bounded CPU lifecycle passed

Both frozen methods passed their own native Collab/NCNC two-update checks. Source manifest `1748b647dee6de992261dfc96b8d1b0b210307d786ce61b7da4f7d05a1b0b526` remains unchanged.

| Method | Updates | Full VALID | Constructions | AUX probes | Seconds | Peak RSS GiB |
|---|---:|---:|---:|---:|---:|---:|
| context_affinity_ncnc_common_v1 | 2 | 3 | 2 | 1 | 363.96 | 12.89 |
| context_affinity_ncnc_split_v1 | 2 | 3 | 2 | 1 | 414.80 | 12.77 |

Actual totals: 4 Adam transitions, 6 full VALID calls, 4 constructions, 4 native reverse collections plus 2 extra AUX probes; 780.73 supervisor case seconds. Each source update kept 65,536 positive and 65,536 negative records. Each VALID used all 60,084 positive and 100,000 negative records in native order. No memory-event increase occurred.

The original TaskCallScope remained intact. Every update recorded 8 top-level and 16 recursive predictor calls; recursive calls bypassed the output bank and contributed no AUX query objects. Checkpoints restored native model/Adam/sampler/TaskCallScope/global-private RNG/target state exactly, with identical served and same-bank mean VALID outputs. AUX reached shared and private input factors, with zero direct output-factor gradient.

These are two-update CPU engineering results. CUDA/resources, complete native epochs, production selected-mean export and final TEST reconstruction remain unqualified. No TEST scoring, scientific fit, accuracy conclusion or campaign GO is represented. Raw terminal results, logs, checkpoints' remote byte receipts and earlier running snapshots are preserved.
