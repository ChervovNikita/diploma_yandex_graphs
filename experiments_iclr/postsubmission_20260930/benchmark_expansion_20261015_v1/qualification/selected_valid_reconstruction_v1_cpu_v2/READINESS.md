# Selected VALID reconstruction readiness

The source package passed its bounded CPU qualification. Source manifest:
`e397d3f1738e763daf8ecfe614833ce64464b974f978f23a67c4eb874c7af27c`.

## Actual selected checkpoints

Four existing MolBACE seed101 run_fit checkpoints were reconstructed through their own recorded runtime, callback and extension. Each complete VALID export (151 graphs) matched saved IDs/order/labels/shape/dtype and the frozen practical logit, loss and metric tolerances. ROC-AUC differences were zero.

| Procedure | Maximum absolute logit difference | Loss difference |
| --- | ---: | ---: |
| native_single | 2.44379044e-06 | 3.78943437e-08 |
| gnnm_boundary_native | 7.74860382e-07 | 4.42100676e-08 |
| gnnm_graph_alignment_variance | 5.36441803e-07 | 2.52628958e-08 |
| gnnm_private_input_residual | 1.54972076e-06 | 3.15786197e-08 |

## Separate engineering state checks

All fourteen existing states passed two fresh restores and complete VALID calls: C6 mean/global/conditional, portablev2 BE/residual-only, and C2 structural/permuted on MolBACE and CiteSeer. Model parameters/buffers remained unchanged during VALID. These states have no selected run_fit export authority; no selected-export claim is made for them.

All six C6 states retain both actual serving and same-bank mean logits/loss/metrics for each restore (12 diagnostic entries). The actual serving result remains the selector authority. Tensor paths/hashes and metrics are in C6_DIAGNOSTIC_INDEX.json. These two-update engineering states are not accuracy evidence.

## Cost and preservation

32 native constructions; 32 complete VALID calls; 0 optimizer attempts/steps, TRAIN forwards, new fits, TEST accesses or GPU worker changes. Loop 47.279988s; full launcher-to-log interval 50.236–51.236s (one-second launch timestamp); CPU user/system 60.845708/15.108207s; peak RSS 1,780,332 KiB.

Initial v1 driver preflight was refused before any model construction because CUDA hiding was applied too late. Its receipt is preserved. Immutable qualification v2 fixes the launcher environment; the replayer, body, inputs, tolerances and work bounds did not change. All seven frozen source packages match before/after checks locally and remotely.

## Coverage limits

BNv3 mapping is prepared but no persisted existing BNv3 state qualified here. GPU execution, link, TEST/final serving, and unknown runtime/source remain unsupported. The post-freeze C1node and alignment-only molecular controls are outside this registry. Native I4 members can be replayed individually; this does not add a bank selector.

Logit/loss tolerances (atol/rtol 1e-4) and absolute metric tolerance .002 serve engineering replay only. They do not set scientific effect sizes or validate improvements.
