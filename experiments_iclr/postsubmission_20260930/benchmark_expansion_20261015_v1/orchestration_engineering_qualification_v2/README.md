# Engineering driver successor after attempt01

Attempt01 passed the actual allocation false-GO refusal, then stopped before
the disabled77 admission assertion. Its local disabled77 CLI inherited the
allocation config SHA, causing `Host configuration hash changed` before it could
return the required `Host disabled before probes` message. The original result
and refusal are preserved verbatim in `PARENT_FAILURE.json`. No model, optimizer
update, TEST access or fixture supervisor was reached in attempt01.

The only behavioral repair sets each negative CLI's bundled config path and
matching SHA in its own copied child environment. It preserves every other
environment value, including the allocation CUDA mask. Production dispatcher,
combined orchestration, integration, scorer, source chain and fixture remain
unchanged and pinned in `SOURCE_BINDINGS.json`. The disabled77 assertion is
unchanged; no host enablement or admission rule is relaxed.

The driver continues to use the original eight fixed cases, exactly two
negative CLIs and three stdlib metadata-only fixture supervisors. It requires
a fresh output directory and retains earlier attempts. It creates no campaign
GO and performs no scientific/model, TEST or network work. Actual native
runtime readiness remains pending.

`check_environment_fix.py` verifies the repair using bundled config JSON only:
the inherited-hash failure reproduces, matching child overrides load each
proper config, and disabled77 remains disabled. It also verifies source pins
and the retained bounded-suite assertions. It does not execute the driver,
CLI, guards, host probes or parent suite.

Root may execute this reviewed successor once on the authorized allocation,
using a new output directory ending in `final_orchestration_attempt02`. This
source preparation has not executed that suite. The giant campaign still
awaits the user's explicit verification and GO.
