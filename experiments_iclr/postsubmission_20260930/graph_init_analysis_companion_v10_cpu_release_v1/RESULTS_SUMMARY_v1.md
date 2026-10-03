# Complete initialization comparison

The saved-array audit completed all 30 cases and 72 phase bindings. It reproduces saved selected-state validation NLL within the original 1e-6 tolerance. The separate supervisor reaped the child with exit 0.

| Graph | Initialization | Mean VALID NLL (nats) | Mean VALID accuracy (%) |
|---|---|---:|---:|
| Photo | graph | 0.350472 | 94.684 |
| Photo | common_only | 0.341297 | 94.161 |
| Photo | random_tangent | 0.355773 | 94.466 |
| Photo | topology_permuted | 0.363985 | 94.858 |
| Photo | warm_copy | 0.321730 | 94.684 |
| Squirrel | graph | 1.364723 | 40.841 |
| Squirrel | common_only | 1.364560 | 40.841 |
| Squirrel | random_tangent | 1.364605 | 40.841 |
| Squirrel | topology_permuted | 1.364748 | 40.841 |
| Squirrel | warm_copy | 1.367596 | 40.691 |

Graph initialization has no consistent advantage over the matched alternatives. On Photo its mean NLL is worse than unchanged warm copying. On Squirrel its contrasts with random or permuted initialization are very small. This variant is closed without further tuning or heldout promotion.

These are development results. VALID selected checkpoints, there are three overlapping split blocks, and the configuration was explored after earlier outcomes. Descriptive paired intervals and all case bindings are retained in NUMERICAL_AUDIT_ADOPTION_v1.json. Reopening serialized checkpoints for independent inference remains outstanding. Original paper scores are unchanged.
