# Independent v4 startup source review — PASS

Candidate manifest SHA-256: `3fd8e1899132a0a0298f743e7a9179bc302aee24b5178e74ec20caeece7abb23`

The preserved v3 terminal, failure and stderr hashes pass. They show `reset_peak_memory_stats(0)` failed at worker line74 before module/data/model loading, with no diagnostic adoption or automatic retry.

The exact v3-to-v4 source diff moves that single reset after authenticated `ordinary_runtime`. Its unchanged, PLAN-pinned source calls `set_device(0)` and `synchronize(0)` before returning. V4 additionally requires CUDA initialization and current device0 before resetting peak statistics. Profile/RNG capture and scientific work remain after the reset; the inclusive wall start remains before admission.

`PLAN.json` changes only the execution directory from failed `root_20261004_v1` to fresh `root_20261004_v2`. All caps, workload, scientific input pins, objective/gradient rules and runtime bindings remain identical. Reversing both repairs in memory restores exact v3 bytes. Common, metrics, supervisor and retained contracts/provenance are byte-identical. F01 and F02 remain closed from the sealed previous reviews.

All22 candidate payloads and seal binding pass. Binding and static identity checks: 76 PASS. No blocking source finding remains in this scoped review.

This source PASS does not authorize execution or qualify CUDA success, numerical output, predictive results or resource suitability. No target source modification, import/execution, numeric library, arrays/states/scores, server access, staging or launch occurred.
