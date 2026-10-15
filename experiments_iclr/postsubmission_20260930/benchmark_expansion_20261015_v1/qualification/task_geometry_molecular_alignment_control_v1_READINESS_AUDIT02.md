# Molecular alignment-only qualification: active-support limitation

This audit narrows the earlier readiness wording. Both actual CPU updates, fresh-session restore and all three whole VALID passes succeeded, but the reused first 32 source TRAIN graphs were all class 1. The helper correctly returned zero alignment and zero eligible alignment anchors on both updates. The original result and costs remain preserved.

The evidence qualifies lifecycle and the one-class gate only. It does not qualify backward through an active alignment term. Two requested optimizer transitions have already been consumed. No further update is launched without the parent granting a separately frozen two-update check on 32 real native TRAIN graphs containing both classes. This would be an explicitly balanced engineering batch, not a change to the scientific source sampler or native BCE.

The frozen new callback, exact molecular helper and unchanged parent alignment+variance mapping remain intact.
