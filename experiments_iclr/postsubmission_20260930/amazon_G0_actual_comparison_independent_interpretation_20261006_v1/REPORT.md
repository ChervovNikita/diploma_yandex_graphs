# Independent interpretation of the actual Amazon comparison

The completed comparison does not establish an accuracy improvement for G0. Live correctly classifies 1,028 of the same 2,449 A nodes, 106 fewer than ENS4, 83 fewer than SINGLE, and 82 fewer than the ordinary own plus pool reference. The original seven state gate is false. Live has much smaller probability losses than these references, but its lower NLL does not establish overall quality superiority or rescue the preset gate.

## Frozen accuracy and probability scores

| State | Correct nodes | Accuracy | Live minus state in correct nodes | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| Live G0 | 1028 | 41.9763% | 0 | 1.380533 | 0.698119 |
| ENS4 | 1134 | 46.3046% | -106 | 4.288698 | 0.893033 |
| SINGLE | 1111 | 45.3655% | -83 | 6.290451 | 1.027852 |
| Ordinary own plus pool | 1110 | 45.3246% | -82 | 8.027279 | 1.023097 |
| Ordinary own | 1106 | 45.1613% | -78 | 7.322676 | 1.020664 |
| Initial common state | 1107 | 45.2021% | -79 | 2.041272 | 0.788921 |
| CMCL H16 | 1018 | 41.5680% | +10 | 1.369322 | 0.703068 |

The integer counts are derived from the reported unrounded accuracy and n, with rounding residuals checked. They are paired net differences on identical nodes. The frozen error flows provide discordant transition counts only for initial versus each of the six endpoints; reference versus live discordant counts are not reported, so no such counts or significance test are inferred. SINGLE is the native member 0 alias, not another independent fit.

Live and graph free have identical 1,028 correct nodes and identical reported initial to endpoint transition counts. Live NLL is higher by 0.00000240149, while its Brier is lower by 0.00000170941. Permuted and stop q also have 1,028 correct nodes. Uniform has 1,030 and margins has 1,029. Their small probability differences do not meet the unchanged gate. Equal accuracy and aggregate flow counts do not prove equality of individual predictions or probabilities.

Against CMCL, live gains ten correct nodes and lowers Brier, but has worse NLL by 0.0112111. This is a mixed comparison rather than a clean win. All four live members lose accuracy against their initial members: 87, 68, 88, and 114 fewer correct nodes respectively. Passing the member comparisons against uniform does not restore competence relative to the initial state.

## What the error flow supports

Live versus initial repairs 224 wrong predictions and changes 303 correct predictions to wrong, for a net loss of 79 correct nodes. Another 1,118 remain wrong, and 804 remain correct. Within the initially wrong 1,342 nodes, NLL falls by 1.668677, Brier falls by 0.429645, confidence falls by 0.230330, and true class probability rises by 0.095404. Across all nodes, confidence falls by 0.254849 and true class probability falls by 0.091119. These observations support a substantial change toward softer probabilities, including relief on initially confident errors, alongside a loss of correct decisions. NLL and Brier are proper probability losses; no calibration curve or ECE was reported, so formal calibration improvement is not independently established.

Among the 970 initial pool errors where every member was wrong, live creates at least one correct member on 315 nodes and repairs 102 pool errors; 868 remain wrong. The field `common_wrong_member_created` counts creation of a correct member on this cohort, not creation of new unanimous errors. Of the 932 baseline pool errors with a shared strictly stronger non-target class, 96 are repaired and 836 remain wrong. One of those repairs has no correct endpoint member under the evaluator's native logit member correctness definition. The saved served probability and FP32 definitions must be retained when interpreting this exception. Graph free has the same reported cohort counts, so these repairs do not identify a graph responsibility benefit in this run.

The class tradeoff is pronounced: relative to initial, class 0 loses 109 correct nodes, class 1 gains 103, and classes 2, 3, and 4 lose 44, 15, and 14. The largest degree band of 3 to 5 loses 75, while degree 11 to 20 gains 16. These are descriptive partitions on one scored graph, not independent mechanism tests.

The references' high pool NLL of 4.29 to 8.03 and Brier of 0.89 to 1.03 coexist with higher accuracy. The reported probability metrics contain no FP32 target zeros or argmax ties. The data are consistent with severe confidence errors in the references; they do not identify why those errors arose. The candidate's lower probability losses do not explain away its lower accuracy, and poor reference probability losses do not invalidate the observed accuracy comparison.

## Scope and completion limits

This is one experimental seed configuration and split on one graph. ENS4's internal four member seeds are ensemble diversity, not four paired benchmark replications. Nodes share graph structure and training data, so an iid node significance test would be inappropriate. No confirmation or population superiority claim follows.

G0 and CMCL have H16 continuations; native references have independent W400 acquisitions followed by 2,300 S/R updates, and ordinary references have 2,300 S/R updates from the same common W400 state. CMCL uses simultaneous SGD with core/private rates 0.001/0.01; ordinary and native controls use Adam. The G0 sequential learning rule, optimizer history, supervision geometry, and update horizon differ. This is an actual reference comparison, not matched optimization or compute.

The scope discloses PCIe A100 training with installed torch scatter and torch sparse for ordinary controls, and allocation SXM4 training without those optional extensions for candidate/native/CMCL. All states were served on the common allocation runtime. Common serving removes a serving runtime discrepancy but cannot undo the distinct training histories. The result does not attribute the accuracy losses to the optimizer, horizon, runtime, graph erasure, or another training cause.

The owner reports an exited child with code zero, all 44 serving member forwards and 12 prediction artifacts complete, zero fits or persistent updates, resource closure passed, and no restoration or publication errors. The local score, error flow, and result hashes match the owner bindings. This independent interpretation was prepared by an agent who did not author the evaluator. It reads frozen result JSON and bound configuration/source metadata only; no prediction or model payload, held labels, server, or new evaluation was accessed.

The original gate remains false. This particular G0 realization failed its preset requirements; that does not reject all GNNM variants. The separate 77 ranking objective control concerns a different loss hypothesis.

## Smallest actual TRAIN audit

The next useful audit is scalar training history and the actual run configuration for these completed fits. Source code proves that native members write `member0_W.trace.jsonl` through `member3_W.trace.jsonl` with 400 W CE entries each and `member0_SR.trace.jsonl` through `member3_SR.trace.jsonl` with 2,300 S/R CE entries each. Ordinary runs write `own.trace.jsonl` and `own_pool.trace.jsonl`, each with 2,300 entries containing own CE, probability pool NLL, and optimized loss. These ten small text traces cover all six reference trajectories; native member 0 alone is the minimum initial check of SINGLE, but cannot diagnose the whole ensemble. These traces were not available among the local result copies inspected here and were not fetched.

Bind the traces to their cohort RUN and RESULT, both ordinary RUN and RESULT files, and the common origin RUN. Verify contiguous update counts, finite losses, the local to global switch after W update 200, the W400 boundary, carried Adam history, and the last fixed S/R horizon. Summarize the complete curves and fixed early/late windows without selecting a new checkpoint. The recorded losses are train mode objectives with dropout, not deterministic served TRAIN losses; TRAIN accuracy and confidence were not logged.

The configuration audit should check the actual optimizer defaults, objective reductions, active train parameter names, architecture and dropout, global stage, seed and construction order, preprocessing and graph hashes, label role hashes and target assembly, W population 4,898 and S/R population 4,899, and absence of W/A supervision in S/R continuation. Native source fixes Adam rate 0.001, zero weight decay, width 256, ten local layers, one global layer, input dropout 0.2, and model/global dropout 0.3. These facts nominate a configuration audit; they are not diagnosed mistakes.

Persistently high or unstable TRAIN CE would support an optimization or configuration concern. Low final TRAIN CE with poor A accuracy and high A NLL would support a failure to transfer confident training fit. Late TRAIN CE reduction alone cannot establish overfitting because no held trajectory was recorded. Neither pattern identifies a specific cause without further evidence. This audit requires no new forward, fit, held scoring, or checkpoint payload read, and supplies no authorization to retune the original gate. `TRAIN_AUDIT_SPEC.json` records exact prospective text paths and bindings.
