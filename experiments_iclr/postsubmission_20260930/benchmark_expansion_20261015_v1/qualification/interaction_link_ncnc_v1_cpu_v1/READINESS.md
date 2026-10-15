# NCNC interactions: bounded CPU lifecycle passed

Frozen source manifest `e7fb265ea3c3650d7ba4c24157adda04b20cdba88ad46ebc54a498b9b892b0c1`; all method and driver bytes rechecked after termination.

| Method | Status | Updates | Full VALID | Constructions | Reverse collections | Seconds | Peak RSS GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| interaction_geometry_input | passed | 2 | 3 | 2 | 4 | 410.46 | 12.69 |
| interaction_responsibility_private_pool | passed | 2 | 3 | 2 | 4 | 366.72 | 13.55 |
| interaction_input_meanJ | passed | 2 | 3 | 2 | 2 | 381.94 | 12.65 |
| interaction_input_conditionalJ | passed | 2 | 3 | 2 | 2 | 408.68 | 12.68 |

Actual totals: 8 Adam transitions, 12 full VALID calls, 8 constructions, 12 native reverse collections and 0 extra reverse collections; 1571.54 supervisor case seconds. The full source minibatch remains 65,536 positive plus 65,536 negative queries; each VALID contains all 60,084 positive and 100,000 negative queries.

The two block-credit recipes used the actual declared objectives and ordered parameter blocks; assigned gradients exactly matched the tensors returned by those same native calls before one original Adam step. The two scalar-J recipes used one audited joint scalar with active finite gradients. No repeated backward comparison or tolerance search occurred. Original depth-one TaskCallScope/recursive completion stayed intact: every update recorded 8 task and 16 recursive calls, with recursive outputs bypassing the private output bank and no recursive AUX objects.

Fresh construction and checkpoint restore preserved model/Adam/sampler/TaskCallScope/private-global RNG/modes/epoch state exactly. Served and same-bank mean VALID outputs were bit-exact after restore; the same-call mean sidecar added no forward and was invalidated on TRAIN/restore.

This qualifies only the bounded native CPU lifecycle. Full epochs, CUDA/resources, production selected-mean export and final TEST reconstruction remain pending. No TEST scoring, scientific fit, accuracy conclusion or full campaign GO is represented.
