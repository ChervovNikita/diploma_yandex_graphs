"""Separately disabled, read-only unchanged-common-state support diagnostic.

No loader, checkpoint reader, parameter commit, core update, R/A labels, scores
or scientific acceptance rule. Root must separately authorize a reviewed future
caller; the fixed numerical nonvacuity convention is intentionally unresolved.
"""

from __future__ import annotations

SOURCE_RELEASED = False
G0V2_SHA256 = "fba3ca3d4bb35da0438923941d97bc4833004dd9fd385735c2352f343693b3e3"


def inspect_common_response(operator, source_sha256, theta_warm, phis_warm,
                            forward, pairs, inner_indices, inner_labels, *,
                            common_state_sha256, warm_metadata):
    """Measure exact live response/Q at theta_warm, then discard virtual phi.

    Caller supplies the fixed S and complete native callback/sparse pairs bound
    to the authenticated fresh400 common state, not an old trained embedding.
    The exact V2 private-response routine is allowed only for this distinct
    source-planned diagnostic. It is never used to bypass the gated learner.
    """
    if not SOURCE_RELEASED:
        raise RuntimeError("Disabled source-only unchanged-warm response diagnostic")
    if (source_sha256 != G0V2_SHA256 or operator.SOURCE_RELEASED is not False
            or len(phis_warm) != 4 or len(pairs) != 10
            or warm_metadata != {"id": "initial", "warm_updates": 400, "episodes": 0,
                                 "global_stage": True, "eval_mode": True}
            or len(common_state_sha256) != 64
            or any(c not in "0123456789abcdef" for c in common_state_sha256)):
        raise ValueError("Exact repairedV2 and authenticated unchanged common state required")
    values = tuple(theta_warm.values()) + tuple(
        value for private in phis_warm for value in private.values())
    versions = tuple(value._version for value in values)
    virtual_private, diagnostics = operator._private_response(
        theta_warm, phis_warm, forward, inner_indices, inner_labels, pairs, 5,
        operator.Config(), "live", collect_diagnostics=True)
    del virtual_private  # Neither preliminary probe nor main response is committed.
    if versions != tuple(value._version for value in values):
        raise ValueError("Unexpected mutation during unchanged-warm diagnostic")
    return {"at": "unchanged_common_theta_and_original_phi", "common_state_sha256": common_state_sha256,
            "diagnostics": diagnostics, "core_updates": 0, "committed_private_updates": 0,
            "complete_member_callback_forwards": 16, "private_gradient_constructions": 8,
            "virtual_probe_private_steps": 4, "virtual_main_private_steps": 4,
            "assignment_maps": 1, "pair_solver_iterations": 80,
            "numeric_nonvacuity_convention_fixed": False, "A_scoring_performed": False}
