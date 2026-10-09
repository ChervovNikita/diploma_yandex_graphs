# Complete all21 exploratory comparison

All four predeclared contrasts are reported; intervals describe three optimizer seeds on the already-used split. No confirmation or mechanism claim follows.

| Contrast | VALID AUROC paired differences | Mean | Sample SD | Descriptive df2 95% interval | Mean VALID NLL delta |
|---|---|---:|---:|---|---:|
| vanilla_pool_minus_independent_pool | -0.012885111, -0.029475539, -0.012588441 | -0.018316364 | 0.009665268 | [-0.04232622029614301, 0.005693493020124344] | 0.018505861 |
| vanilla_pool_minus_independent_member0 | -0.0033753831, -0.01967661, -0.0091662152 | -0.010739403 | 0.0082636968 | [-0.03126756347781193, 0.009788758285569382] | 0.013490081 |
| centered_pool_minus_independent_pool | -0.0086312591, -0.0058885906, -0.012540919 | -0.0090202564 | 0.0033431809 | [-0.017325178161894474, -0.0007153345813305702] | 0.0065680345 |
| centered_pool_minus_independent_member0 | 0.00087846856, 0.0039103392, -0.0091186937 | -0.0014432953 | 0.006817761 | [-0.018379552587580797, 0.015492961928131871] | 0.001552254 |

## Unchanged pilot gates

Each of the three AUROC gains must be strictly positive, mean gain at least0.003, and mean VALID NLL delta at most0 for both contrasts of each predeclared recipe. Numerical gate passage still requires root member-competence/cost review and the frozen additional controls.

| Recipe | Both frozen numerical contrasts pass | Root competence/cost review |
|---|---|---|
| vanilla | False | required |
| centered | False | required |

## Member learning and complementarity

Every member, its mean/worst quality and complete TRAIN eval learning curves are retained. Full-population errors report common failures, unique correct coverage, pool rescue/harm and confidence on each error set. All positive-negative rank pairs, including ties, are processed in bounded blocks.

Higher shared TRAIN eval NLL alongside lower VALID member quality supports a learning deficit. Low common-error intersection or distinct correct coverage supports complementary predictions; coverage lost by the pool and rescue/harm show how much the fixed probability mean converts it. These descriptive signals cannot assign cause to capacity, sharing, factor priors or geometry without the frozen additional controls.

| Recipe/seed | Mean TRAIN NLL minus independent mean | Mean VALID AUROC minus independent mean | All4 wrong fraction | Any member correct fraction | Correct coverage lost by pool rows | Pool AUROC minus own mean |
|---|---:|---:|---:|---:|---:|---:|
| vanilla/7409 | 0.01065813 | -0.0053108652 | 0.17965294 | 0.82034706 | 20 | 0.00079377849 |
| vanilla/8501 | 0.045901552 | -0.020718009 | 0.20551208 | 0.79448792 | 13 | 0.00026221675 |
| vanilla/9607 | 0.0055096075 | -0.0075725483 | 0.15685607 | 0.84314393 | 96 | 0.001154772 |
| centered/7409 | -0.019320883 | -0.00031364179 | 0.17727118 | 0.82272882 | 10 | 5.0406716e-05 |
| centered/8501 | -0.00057446212 | 0.003042563 | 0.17761143 | 0.82238857 | 18 | 8.8593622e-05 |
| centered/9607 | 0.011000924 | -0.0064009741 | 0.18713848 | 0.81286152 | 8 | 3.0719244e-05 |

Per-member rescue/harm, confidence, unique coverage, all rank pairs and complete comparable eval-TRAIN curves are in ANALYSIS.json; every member and selected epoch is in EVERY_MEMBER_QUALITY.csv. Cost records retain all original independent fitting costs and the differing serving residency.

Shared paths use a pooled selected epoch; independent paths use their own selected epochs. Learned factor tensors and actual operator responses are absent from serving references, so this reader makes no causal factor/geometry or operator-response claim.
