# Manual static review v1

Review scope: authored source text, manual reads/searches and read-only SHA-256 file hashing. No module was imported or compiled; no backend/model/parameter object was constructed; no generation, density calculation, gradient, fit, data draw, RNG, GPU/server access or installation occurred. This packet is source-only and unqualified.

## Inspected source relationships

- The encoder wraps one authorized ordinary source `Model` forward and exposes H to decoder methods with no encoder-forward handle. The known source fingerprint is recorded. Dropout must be zero; exposed source normalization must be node-wise LayerNorm or Identity. Caller provenance and graph binding still need independent audit.
- Factored branches share explicit input/recurrent/output matrices and have private diagonal factors, gate/output biases, initial hidden vectors and per-trajectory states. Single and untied arms use conventional matrices through the same decoder methods. Every full trajectory advances at all N nodes.
- Generation and teacher forcing call the same state/mask/log-probability step. Graph edges/order are supplied immutable contracts. All feasible assignments have common positive support in exact arithmetic; infeasible full densities are negative infinity. Future-lock conditional generation is absent.
- The objective receives full all-member complete densities, including each generating branch. Mixture rewards include every density. Mean-branch rewards substitute only the generating density while retaining the full density/reduction workload. No leave-one-out density or own-sample-including baseline is provided.
- The mixture reward and baseline subtraction are detached; only generating complete scores retain gradients. The explicit log-density derivative integrates to zero under normalized common theta-independent support and fixed pi/gamma. Iid versus stratified weights are explicit caller choices.
- Cost counters distinguish one source encoding, full decoder generation/density trajectories, forward-only cross densities, generating-score gradient trajectories and actual objective reductions. Mask traversal is labeled an upper bound because `any()` can terminate early. Generation self-density value/gradient reuse is not implemented.

## Source limitations left explicit

Python syntax and backend calls are not executable-qualified. Finite-value checks do not establish stable probabilities, valid random draws, exact floating-point sampling or unbiased Monte Carlo behavior. Free-step sampling raises if conversion reaches probability zero or one; this is a numerical failure to resolve, not a conditional rejection sampler. Supplied branch allocation, uniform independence and baseline independence require external qualification.

The sampling/scoring batch must keep graph, order, parameters and H fixed. Public injected objects can violate that contract, parameter aliasing, provenance or the generating-score gradient contract; text checks cannot establish those properties. Cross matrices supplied from outside the provided method must contain true complete branch densities and correct score gradients.

Parameter/initialization/resource matching for single/untied controls is not implemented. The head-only shared-transition density-reuse variant is documented but unimplemented. Full task/outcome/control/configuration freeze, exact-reference and numerical-gradient checks, required controls and representative training-only resource qualification remain root responsibilities. Existing caps and the one-A100 limit persist; no execution or training admission is granted.

The prior sealed research and fixed-cache packets are preserved. This packet's manifest/seal records source/document integrity and scope only; it does not certify numerical correctness or novelty.
