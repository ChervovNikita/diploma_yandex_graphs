# V3 narrow terminal custody row repair

V1, V2 and both BLOCKED reviews remain preserved. The V2 review closes F01 and identifies only F02: the final terminal rewrite updated the top-level custody hash while leaving its existing nested `supervisor_output_files` terminal row stale.

After the final `TERMINAL.json` write, V3 computes its current hash and byte count once. It refreshes only an existing `TERMINAL.json` row in the prior supervisor inventory and uses the same hash for the top-level `terminal_sha256`. Failure custody without that inventory stays a failure. There is no recursive re-inventory, new cap loop, worker/metric/objective/data/runtime/cap change or numerical run.

The exact reverse text patch restores V2 supervisor bytes. Worker, common, metrics, PLAN, disabled release, prerequisites and staging bytes are identical. Existing V2 after-collection/publication budget checks and explicitly disclosed final status/fsync/exception/stdio tails and sampled-RSS limits stay unchanged. V3 remains disabled pending independent source review and root release.
