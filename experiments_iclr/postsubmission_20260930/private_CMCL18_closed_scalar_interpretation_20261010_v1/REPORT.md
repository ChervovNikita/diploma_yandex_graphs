# Complete specialist-credit results

All declared fits closed before interpretation. These are development results on an encountered graph. No original paper score changed. TEST remains closed.

| Predictor | accuracy | NLL |
|---|---:|---:|
| own_floor | 90.977506 | 0.281966 |
| private_cmcl | 90.309489 | 0.345505 |
| all_block_cmcl | 90.656181 | 0.344376 |
| vanilla_cmcl | 90.749196 | 0.368313 |
| private_uniform | 89.641468 | 0.423597 |
| private_constant_credit | 90.165738 | 0.353734 |
| contextual_factor1_native | 90.816844 | 0.322804 |
| contextual_factor1_mean4_dropout | 90.783020 | 0.282255 |
| contextual_independent4_own | 90.216472 | 0.354645 |
| contextual_single_native | 89.937426 | 0.511273 |
| contextual_shared4_own | 90.901404 | 0.287697 |

## Primary paired comparisons

Accuracy and F1 deltas below are percentage points. NLL and BCE use natural-log units. The exploratory intervals describe variation among three repeats after validation selection. They do not account for selecting checkpoints, trying several methods, or changing graphs.

### private_cmcl__minus__own_floor

- accuracy: mean -0.668017, paired deltas -0.786400, -0.634193, -0.583458, interval [-0.930378, -0.405657].
- NLL: mean +0.063539, paired deltas +0.062911, +0.074377, +0.053330, interval [+0.037364, +0.089715].

### private_cmcl__minus__all_block_cmcl

- accuracy: mean -0.346692, paired deltas -0.304413, -0.304413, -0.431252, interval [-0.528607, -0.164778].
- NLL: mean +0.001129, paired deltas -0.001131, -0.002168, +0.006686, interval [-0.010895, +0.013153].

### private_cmcl__minus__vanilla_cmcl

- accuracy: mean -0.439707, paired deltas -0.279045, -0.304413, -0.735664, interval [-1.077187, +0.197772].
- NLL: mean -0.022808, paired deltas -0.026629, -0.030591, -0.011204, interval [-0.048252, +0.002636].

### private_cmcl__minus__private_uniform

- accuracy: mean +0.668021, paired deltas +0.279045, +1.166922, +0.558096, interval [-0.459854, +1.795896].
- NLL: mean -0.078092, paired deltas -0.063168, -0.098631, -0.072475, interval [-0.123765, -0.032418].

### private_cmcl__minus__private_constant_credit

- accuracy: mean +0.143751, paired deltas -0.076103, +0.279045, +0.228310, interval [-0.333406, +0.620907].
- NLL: mean -0.008229, paired deltas -0.000274, -0.013813, -0.010600, interval [-0.025802, +0.009344].

### private_cmcl__minus__contextual_factor1_native

- accuracy: mean -0.507355, paired deltas -0.583459, -0.279045, -0.659562, interval [-1.007539, -0.007172].
- NLL: mean +0.022701, paired deltas +0.003712, +0.018449, +0.045942, interval [-0.030543, +0.075945].

### private_cmcl__minus__contextual_factor1_mean4_dropout

- accuracy: mean -0.473532, paired deltas -0.532723, -0.329781, -0.558091, interval [-0.784388, -0.162675].
- NLL: mean +0.063250, paired deltas +0.060204, +0.061692, +0.067854, interval [+0.053174, +0.073325].

### private_cmcl__minus__contextual_independent4_own

- accuracy: mean +0.093017, paired deltas +0.050737, +0.050737, +0.177576, interval [-0.088899, +0.274932].
- NLL: mean -0.009140, paired deltas +0.015467, -0.011485, -0.031400, interval [-0.067570, +0.049291].

### private_cmcl__minus__contextual_single_native

- accuracy: mean +0.372063, paired deltas +0.329783, +0.481990, +0.304415, interval [+0.133483, +0.610642].
- NLL: mean -0.165768, paired deltas -0.281047, -0.244714, +0.028457, interval [-0.586039, +0.254503].

### private_cmcl__minus__contextual_shared4_own

- accuracy: mean -0.591915, paired deltas -0.837137, -0.481988, -0.456620, interval [-1.120409, -0.063421].
- NLL: mean +0.057808, paired deltas +0.071914, +0.049927, +0.051582, interval [+0.027390, +0.088225].

### all_block_cmcl__minus__own_floor

- accuracy: mean -0.321325, paired deltas -0.481987, -0.329781, -0.152206, interval [-0.731339, +0.088689].
- NLL: mean +0.062411, paired deltas +0.064042, +0.076545, +0.046645, interval [+0.025107, +0.099714].

### vanilla_cmcl__minus__own_floor

- accuracy: mean -0.228310, paired deltas -0.507355, -0.329781, +0.152206, interval [-1.076116, +0.619497].
- NLL: mean +0.086347, paired deltas +0.089540, +0.104967, +0.064534, interval [+0.035659, +0.137035].

### private_uniform__minus__own_floor

- accuracy: mean -1.336038, paired deltas -1.065445, -1.801115, -1.141554, interval [-2.341027, -0.331049].
- NLL: mean +0.141631, paired deltas +0.126080, +0.173007, +0.125806, interval [+0.074129, +0.209133].

### private_constant_credit__minus__own_floor

- accuracy: mean -0.811768, paired deltas -0.710297, -0.913239, -0.811768, interval [-1.063835, -0.559700].
- NLL: mean +0.071768, paired deltas +0.063185, +0.088190, +0.063930, interval [+0.036429, +0.107108].

## Interpretation limits

Every fixed arm is retained in SUMMARY.json. Positive member diversity, a positive interaction, or a small validation delta alone does not establish a useful ensemble mechanism. Strongest capable references, exact error changes and unused confirmation govern any next claim. Concurrent fits prevent an isolated speed interpretation.
