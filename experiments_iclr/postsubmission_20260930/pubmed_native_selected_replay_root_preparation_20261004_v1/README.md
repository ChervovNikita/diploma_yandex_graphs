Operational selected-state replay preparation — 2026-10-04

This packet follows the native scientific fit launch. It is operational readiness work, not a new independent source review or a scientific replay result.

The local `ROOT_RELEASE_replay_selected.json` is deliberately disabled: status `DISABLED_TEMPLATE`, no authorized stages, no authorization reference, and no scientific fit admission. It prospectively preserves the reviewed zero-update replay invocation, 1800-second wall cap, 32-GiB host RSS cap, 70-GiB allocated/75-GiB reserved CUDA caps, exact source-v3 manifest, cohort and original root-verified feature/negative-pool/prerequisite bindings. It cannot pass the release gate in this state.

`prepare_disabled_release.py` consumes only project-wrapper scalar output. Complete cohort freeze, final custody and supervisor terminal are fetched and copied only after all six original fits close normally. Their exact hashes are then bound into the disabled candidate. Selected checkpoint/score hashes are checked against scalar inventory entries without opening or deserializing those files.

The original fit uses source manifest `cd360a31fc6feead174f1da3ea9478655aa3bd78f68aab548601cc9fb37090e5` and release `b1c8a054a8ae1acd4bf75fa0fb3d75d52b513474f028e943adb5a53ee2ca4a6e`. Only supervisor PID 3198977/start-time ticks 1723410112 and its recorded owned child are observed. The MacLink wrapper is used solely to reach shmelev@192.168.18.77 through the authorized forwarding route; secret values are not copied or recorded.

Root must separately admit and launch replay after inspecting the completed preparation. This packet executes no optimizer updates, replay/scoring or TEST, changes no attempt, and modifies neither reviewed source nor original fit outputs. Scientific serialized replay remains unexecuted even when this preparation is ready.
