# Narrow restart repair from combined orchestration v1

Parent remains unchanged:

`combined_campaign_orchestration_v1/MANIFEST.json`
SHA256 `33d044c4270366760cee21949e08a3f38fa21800f18d5ade7a66b9d175aa883c`.

Root review found a real interruption point. V1 created both the host shard
and LOCAL_READY only under `not shard_path.exists()`. A crash after durable
shard creation but before readiness creation left a valid shard that caused
the restart branch to skip LOCAL_READY forever.

V2 keeps the same run/template/GO/roster/seal APIs and source-only pending
scorer/runtime bindings. The primary always calls `ensure_local_ready` after
all local worker acknowledgments. That helper:

1. Reuses an existing shard without calling its builder, or creates it once.
2. Authenticates release/roster/owner, exact task/state/terminal/result scalar
   chains, original selected-state metadata and frozen role semantics.
3. Independently ensures the exact LOCAL_READY receipt after the shard.

Both local workers wait for shard **and** readiness. The helper runs once at
this stage boundary, outside polling; it does not rebuild the shard or scan
native descriptors on every wait. No checkpoint/logit tensor is opened.

The focused repo-local JSON/filesystem check commits a shard with no
LOCAL_READY, then exercises recovery. It verifies zero builder calls,
readiness restoration, byte-identical shard reuse, idempotent second
recovery and refusal of a changed terminal outcome. No supervisor, native
data/model work, GPU probe, remote operation or GO is involved. Fixture files
are generated under this new package and removed after the check; test
results remain in LOCAL_CHECKS.

All other original v1 source files/history remain preserved. Imports and
source-manifest schema name move to the v2 package; native/GO APIs do not.
