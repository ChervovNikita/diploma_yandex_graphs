# Named Adam and private-bias coordinate transport

The native warm predictor is a `TeacherFamily(single_author,cfg0)` whose canonical names start `models.0.`. The K1 AD and K4 continuation wrappers are raw boundary families. All native alias names are enumerated with `remove_duplicate=False`; names sharing one Parameter must still share one corresponding raw Parameter. Distinct native Parameters may not merge. Optimizer groups must cover every live parameter once, including inactive local-head parameters. The source verifies copied parameter values and all mapped alias sets.

| Native name after models.0 prefix | Raw name |
|---|---|
| PolyFormer lin1.weight/bias | stem.weight/B |
| PolyFormer lin3.weight/bias | head.weight/B |
| Polynormer lin_in.weight/bias | stem.weight/B |
| Polynormer pred_local.weight/bias | local_head.weight/B |
| Polynormer pred_global.weight/bias | global_head.weight/B |
| Every other native name | core.native_name |

The checkpoint stores parameter-name groups, exact native group options, all scalar steps, first/second/max-second moments, complete aliases, model tensors, primitive metadata and RNG arrays. It does not rely on optimizer integer IDs or model parameter order. Scalar Adam step stays on CPU for ordinary noncapturable/nonfused Adam; moments move to the live parameter dtype/device. Python/NumPy RNG state is saved as primitives/Tensors so weights-only load does not require arbitrary NumPy objects.

## Derivation at identical routes

Let K=4 and native gradient (including coupled decay) be q=g+wB. Mean member CE gives each private bias row gradient g/K. Choose its coupled decay w/K so q_b=q/K. If native Adam moments are m,v, copy m_b=m/K and v_b=v/K², preserving the scalar step, betas, learning rate and bias correction. With epsilon_b=epsilon/K, the private update is

`lr (mhat/K) / (sqrt(vhat/K²)+epsilon/K) = lr mhat/(sqrt(vhat)+epsilon)`.

AMSGrad maximum second moment, when present, receives the same1/K² scaling. Native W/body gradients are the average sum over identical trajectories and need no coordinate scaling. New R/S have no warm history and begin with empty moments; their optimizer options are inherited from their associated native boundary-weight group. This integration admits ordinary coupled Adam only; decoupled weight decay would require a distinct derivation/version.

Thus the4bias rows initially equal the native bias, their Adam steps equal the native bias update, and all shared native updates equal the native updates in real arithmetic. Literal replication of unscaled m/v with unchanged epsilon/decay does not satisfy this equivalence. This transport is a **new explicit prospective optimizer amendment** relative to round15's underspecified bias-moment-copy phrase. It is applied identically to all five controls and does not alter the external native baseline.

## Mandatory numerical gate

Future root qualification uses complete admitted topology/features and exactly compact train labels. It populates real nonzero native Adam history with disposable dropout-off updates; Photo exercises local then global state. It deep-copies the donor, restores native named state, creates identity K4, transports optimizer state and freezes every R/S. It checks native logits versus K1 and each K4 member, native shared gradients, K×each private bias row gradient, one actual Adam update, every mapped parameter, every moment/step and final member logits. It also checks inactive local-head state. The identical gate runs again on the actual50/250warm checkpoint for each arm.

Modern frozen identity/gradient tolerances remain atol1e-6/rtol1e-5 for logits and atol1e-6/rtol1e-4 for gradients. Parameters/moments use atol1e-6/rtol1e-5. These are prospective FP32 comparison tolerances, not a bitwise claim. They cannot be widened after a failed cell. R/S are frozen only during the disposable one-step check; new-state emptiness and exact warm bias copying are checked before the seeding slice is installed.

No code in this packet was run. This derivation does not assert that its future numerical gate has passed, that states stay equivalent after routes diverge, or that the resulting ensemble improves utility.
