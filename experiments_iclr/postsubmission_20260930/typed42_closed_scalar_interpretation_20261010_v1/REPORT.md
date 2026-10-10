# Complete heterogeneous context results

All declared fits closed before interpretation. These are development results on an encountered graph. No original paper score changed. TEST remains closed.

| Predictor | micro_F1 | macro_F1 | BCE |
|---|---:|---:|---:|
| shared_own_pair4 | 68.437870 | 66.098536 | 0.537806 |
| shared_local_mul4 | 68.279561 | 65.744586 | 0.537691 |
| shared_local_add4 | 68.165610 | 65.489306 | 0.537849 |
| shared_global_mul4 | 68.178121 | 65.843846 | 0.537767 |
| single_local_mul1 | 68.102364 | 65.543119 | 0.535916 |
| untied_local_mul4 | 68.404462 | 66.013789 | 0.535225 |
| ordinary_native_independent4 | 69.081120 | 66.553687 | 0.531218 |
| contextual_native_independent4 | 69.044373 | 66.515332 | 0.531930 |

## Primary paired comparisons

Accuracy and F1 deltas below are percentage points. NLL and BCE use natural-log units. The exploratory intervals describe variation among three repeats after validation selection. They do not account for selecting checkpoints, trying several methods, or changing graphs.

### shared_local_mul4__minus__shared_own_pair4

- micro_F1: mean -0.158309, paired deltas +0.375049, -0.251631, -0.598345, interval [-1.383886, +1.067269].
- macro_F1: mean -0.353950, paired deltas +0.292119, -0.718257, -0.635711, interval [-1.747631, +1.039732].
- BCE: mean -0.000115, paired deltas +0.000615, -0.001097, +0.000136, interval [-0.002309, +0.002078].

### shared_local_mul4__minus__shared_local_add4

- micro_F1: mean +0.113951, paired deltas +1.322184, -0.818563, -0.161767, interval [-2.610361, +2.838264].
- macro_F1: mean +0.255281, paired deltas +1.624045, -0.842117, -0.016087, interval [-2.862999, +3.373560].
- BCE: mean -0.000158, paired deltas -0.000644, +0.000137, +0.000032, interval [-0.001212, +0.000896].

### shared_local_mul4__minus__shared_global_mul4

- micro_F1: mean +0.101440, paired deltas +0.792000, -0.189036, -0.298645, interval [-1.390406, +1.593286].
- macro_F1: mean -0.099260, paired deltas +0.734502, -0.721616, -0.310665, interval [-1.964166, +1.765646].
- BCE: mean -0.000076, paired deltas +0.000674, -0.000930, +0.000029, interval [-0.002081, +0.001929].

### shared_local_mul4__minus__single_local_mul1

- micro_F1: mean +0.177197, paired deltas +0.372651, -0.566389, +0.725330, interval [-1.481392, +1.835787].
- macro_F1: mean +0.201467, paired deltas +0.337063, -0.322122, +0.589459, interval [-0.967754, +1.370687].
- BCE: mean +0.001775, paired deltas +0.001010, +0.003686, +0.000629, interval [-0.002363, +0.005913].

### shared_local_mul4__minus__untied_local_mul4

- micro_F1: mean -0.124901, paired deltas +0.634058, -0.948071, -0.060688, interval [-2.094863, +1.845062].
- macro_F1: mean -0.269203, paired deltas +0.615807, -0.765959, -0.657457, interval [-2.177911, +1.639505].
- BCE: mean +0.002466, paired deltas +0.001069, +0.006020, +0.000308, interval [-0.005239, +0.010171].

### shared_local_mul4__minus__ordinary_native_independent4

- micro_F1: mean -0.801559, paired deltas -0.804621, -0.946617, -0.653439, interval [-1.165767, -0.437351].
- macro_F1: mean -0.809101, paired deltas -1.008062, -0.885934, -0.533306, interval [-1.421509, -0.196693].
- BCE: mean +0.006474, paired deltas +0.004649, +0.009949, +0.004823, interval [-0.001006, +0.013953].

### shared_local_mul4__minus__contextual_native_independent4

- micro_F1: mean -0.764812, paired deltas -0.937303, -0.883065, -0.474066, interval [-1.393917, -0.135706].
- macro_F1: mean -0.770746, paired deltas -1.005706, -0.986630, -0.319902, interval [-1.740949, +0.199457].
- BCE: mean +0.005761, paired deltas +0.004240, +0.008852, +0.004191, interval [-0.000889, +0.012411].

### shared_local_add4__minus__shared_own_pair4

- micro_F1: mean -0.272260, paired deltas -0.947135, +0.566932, -0.436578, interval [-2.185772, +1.641252].
- macro_F1: mean -0.609230, paired deltas -1.331926, +0.123859, -0.619624, interval [-2.417554, +1.199093].
- BCE: mean +0.000043, paired deltas +0.001260, -0.001234, +0.000103, interval [-0.003057, +0.003143].

### shared_global_mul4__minus__shared_own_pair4

- micro_F1: mean -0.259748, paired deltas -0.416951, -0.062595, -0.299699, interval [-0.708197, +0.188700].
- macro_F1: mean -0.254690, paired deltas -0.442383, +0.003359, -0.325045, interval [-0.828648, +0.319268].
- BCE: mean -0.000039, paired deltas -0.000058, -0.000166, +0.000107, interval [-0.000381, +0.000302].

### single_local_mul1__minus__shared_own_pair4

- micro_F1: mean -0.335506, paired deltas +0.002398, +0.314758, -1.323675, interval [-2.496492, +1.825480].
- macro_F1: mean -0.555416, paired deltas -0.044944, -0.396136, -1.225169, interval [-2.060855, +0.950022].
- BCE: mean -0.001890, paired deltas -0.000394, -0.004782, -0.000494, interval [-0.008114, +0.004333].

### untied_local_mul4__minus__shared_own_pair4

- micro_F1: mean -0.033408, paired deltas -0.259009, +0.696441, -0.537656, interval [-1.641243, +1.574427].
- macro_F1: mean -0.084746, paired deltas -0.323687, +0.047702, +0.021746, interval [-0.599796, +0.430303].
- BCE: mean -0.002581, paired deltas -0.000454, -0.007117, -0.000172, interval [-0.012346, +0.007183].

### ordinary_native_independent4__minus__shared_own_pair4

- micro_F1: mean +0.643250, paired deltas +1.179670, +0.694987, +0.055094, interval [-0.757978, +2.044479].
- macro_F1: mean +0.455151, paired deltas +1.300182, +0.167677, -0.102404, interval [-1.393476, +2.303779].
- BCE: mean -0.006589, paired deltas -0.004034, -0.011046, -0.004687, interval [-0.016211, +0.003033].

## Interpretation limits

Every fixed arm is retained in SUMMARY.json. Positive member diversity, a positive interaction, or a small validation delta alone does not establish a useful ensemble mechanism. Strongest capable references, exact error changes and unused confirmation govern any next claim. Concurrent fits prevent an isolated speed interpretation.
