# All-fit smoke tests followed by authorized training

The user's 16 October message explicitly authorizes two real optimizer updates
for every one of the 6,460 fits, bug fixes, then the unchanged full training
campaign in tmux. `USER_GO.json` records the instruction. Earlier no-GO prose is
superseded; do not ask for authorization again after the checks pass.

Use only the guarded allocation and server51 routes described in AGENTS.md.
Both live HEADs remain 48329b19941ea869c439e5a220a3df0e30aa1237 while the remaining
allocation PubMed pilot completes. Server51 pilots are complete (8 fits and four
collections each); the allocation pilot was 6/8 complete at 23:19 UTC. Preserve
the separately owned allocation GRU and its 32 GiB GPU / 16 GiB host envelope.

The new packages are being finalized:

- `qualification/all_fit_smoke_20261016_v1`: every exact deployed fit ID, two
  updates, complete native VALID, own smoke-state reload, no TEST. Successful
  temporary smoke checkpoints may be removed after verification. Never use
  them as scientific warm starts.
- `qualification/all_fit_resources_20261016_v1`: frozen resource catalogue and
  static pairing API; manifest e41cb2ca51ded8793b83a68b5a1dc1575491813033392e14c711a364544b2148.
- `campaign_execution_20261016_v1`: tmux worker owners, per-attempt custody,
  shared host resource accounting, global smoke barrier, then fresh full fits.

Root started allocation tmux `gnnm-allfit-prerequisites-20261016`. It waits for
the original pilot to complete and exit, then runs the already reviewed new
MolBACE reporter lifecycle (38 updates) and BUDDY saved-state replay (zero
updates). Its `ALLOCATION_PREREQUISITES.json` must be PASS before the allocation
all-fit worker starts GPU children. Source repair package and failure history
remain intact; do not repeat BUDDY's 1,585-update acquisition.

Resource package plus USER_GO are staged on both hosts. AGENTS.md updated on
both hosts, with previous bytes preserved under this planning folder.
No massive scientific fit has started. Smoke/worker deployment is still in
progress; use actual launch and terminal receipts rather than this initial note
for current execution status.

The old preparation automation no longer existed. The new active 30-minute
heartbeat is `run-gnnm-smoke-tests-and-training`. It should fix actual failures,
verify all-fit coverage, activate the full queues without another approval,
and monitor progress. Notify meaningful changes only.
