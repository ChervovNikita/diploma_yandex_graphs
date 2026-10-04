# What the joint endpoint auxiliary actually encourages

This is an algebraic interpretation of the implemented loss, not a new theorem or a predictive result. Source: immutable `exact_cb_support_bucket_paired_predictive_preparation_20261004_v1/conditional_loss.py`, `training_pattern_losses`, lines 222–252.

For a query, let `a_m` and `b_m` be the conditional subset negative log-likelihoods from member `m` at the two endpoints, and let there be `M` members. Define likelihoods `u_m = exp(-a_m)`, `v_m = exp(-b_m)`, and normalized endpoint responsibilities `q_m = u_m / sum(u)`, `r_m = v_m / sum(v)`.

The joint mixture has likelihood `sum(u_m v_m) / M`. Independently mixing the endpoints has likelihood `sum(u) sum(v) / M^2`. Therefore their unnormalized negative log-likelihood difference is

`joint - separate = -log(M * sum_m q_m r_m)`.

The implementation divides this quantity by the common support denominator `max(n_left + n_right, 1)`. Both objectives then use the same all-query reduction. Degenerate deterministic supports contribute zero; the identity applies to the remaining likelihood terms. Computation uses log-space arithmetic, so the exponential notation is mathematical shorthand.

The contrast rewards agreement between endpoint responsibilities relative to the uniform reference. It can be negative when both endpoints favour the same member, positive when they favour different members, or zero when the overlap is `1/M`. If every member predicts identically, both responsibilities are uniform and the contrast is zero. Uniform responsibility on either endpoint also makes the contrast zero.

Consequently, sharing endpoint responsibility does not directly guarantee diverse members, prevent collapse or force different reasoning routes. At exact member symmetry, the joint-versus-separate contrast also has no first-order incentive to break that symmetry: the overlap derivative through a normalized uniform responsibility vanishes. The complete joint auxiliary still has its reconstruction gradient, and unequal initialized members can receive different gradients. These facts do not imply stable specialization or predictive gains after training.

This standard finite-mixture algebra supplies a diagnostic interpretation, not a novelty claim. If the fitted joint arm helps, report whether responsibility concentration and endpoint agreement change alongside served ranking quality. Such diagnostics remain secondary and cannot replace the prospectively fixed predictive comparisons. The frozen coefficient, normalization and comparisons are unchanged by this note.
