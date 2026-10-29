# Four concurrent tasks per GPU

On 29 October 2026 the user requested four concurrent training jobs on each of
the three assigned GPUs. This package supersedes the old two-slot scheduling
limit. It retains the frozen 6,460 fits, 270 ensemble exports, GPU assignments,
native training commands, seeds, model settings, stopping rules and reporting
repair. The existing resource catalogue remains historical source; its
`maximum_owned_children_per_GPU: 2` metadata is overridden explicitly by this
scheduler's recorded four-slot policy.

The ceiling is four resource-compatible tasks, including any ensemble export
occupying a slot. VRAM growth is summed across this GPU. RAM, CPU and disk
reservations are summed across the host, including both server51 queues.
The original pair exclusions and the 20 GiB launch disk floor remain active.
Large jobs can run with fewer companions. Four BUDDY reservations alone require
80 GiB plus the safety margin, so the scheduler may use a smaller pending task
for the fourth slot on an 80 GiB GPU.

The previous queue owner has no live concurrency configuration and retains
ownership locks in its training children. `handover.py schedule` starts a tmux
watcher that requests the existing graceful drain. Current fits finish normally.
The watcher then verifies the old owner is gone, all inherited leases are free,
and the original tmux pane is dead. It archives its own exact STOP marker and
respawns that pane with the four-slot worker. Each GPU transitions independently.
No running training process is signalled, and no active checkpoint is restarted.

The four-slot worker reuses the exact successful smoke records and global
release. It collects existing completed results and continues pending IDs under
the original exclusive locks. It does not retry new failures automatically.
If a current fit fails during the drain, the watcher preserves the failure and
stops before replacement; inspect the handover journal.

`preflight` is read-only apart from normal interpreter environment settings.
`schedule` and `watch` deliberately write only inside the project repository,
and use ordinary tmux sessions. All commands are readable plaintext.

Test coverage is recorded in `test_concurrency.py`. Handover journals are stored
under `planning/concurrency4_20261029_v1/<worker>/` on each host. The package
manifest records the exact source used. Training invocations retain the
unchanged reporting-repair source; scheduler provenance records this new package.
