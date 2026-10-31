# Live three-GPU continuation

Published source48329b19941ea869c439e5a220a3df0e30aa1237 is on both hosts.
All three PubMed pilot queues are running and have advancing full-epoch traces.
Read planning/three_gpu_sync_20261015_v1/LIVE_THREE_GPU_STATE.json and the two
RUNNING_*_02.json records for exact owners, start ticks and observations.
Do not launch duplicate studies or score a partial cohort.

Keep both live checkouts at their pinned commit while these queues run: the
owner validates exact HEAD for every child. Future source/docs publication
should use a separate checkout until the cohorts finish, or a separately
reviewed successor after completion; changing live HEAD would stop next jobs.
Do not edit the actual map or frozen source.

Map SHA31cdc3346597c1d622a90ef02e48d21360a0ca0bd4cc43369c4437b40a4f5898.
Workers allocation/gpu0=101,server51/gpu0=202,server51/gpu1=303, each8fits.
OriginalGRU continues on allocation. All24fits+12collectors+18banks are required
before joint analysis. Collection bundles stay on their training hosts.
Both51queues survive MacLink/SSH disconnect. Restore the other Mac for monitoring
and collection. CurrentMac controller remains in exec session29289;
authenticated51 SSH master is in MacLink shell gnnm-sync-20261015 (session50527).
No password is saved in files. Use the existing live control socket via
planning/three_gpu_sync_20261015_v1/remote51.py.

The full6460-fit proposal has dormant balanced queues on both servers;
2154/2153/2153fits plus90I4exports perdevice. These are not a launch-ready
campaign release and fullcampaignGO is still absent. BUDDY serving source
repair is published but zero-training replay is pending. Never redo its1585
paid updates.
