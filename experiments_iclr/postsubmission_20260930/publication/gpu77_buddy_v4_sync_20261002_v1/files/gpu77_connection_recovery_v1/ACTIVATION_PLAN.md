# Use the additional two GPUs

The current obstacle is the network route from this Mac to `192.168.18.77`, before SSH authentication. No password has been tried. The existing MacLink relay uses the account that reaches the prohibited seven-GPU allocation. A replacement controller through the permitted `anogena-2` account is running with separate private state in `.ml77`.

The other Mac must pair with that replacement, or this Mac needs a direct VPN route. The replacement settings are `relay-onegpu.json`; the other Mac needs its own SSH authentication to the permitted relay. Use a separate client config so the existing pairing is preserved. Pairing codes last five minutes and are never committed.

Once the connection is available, root will do the following:

1. Reach `shmelev@192.168.18.77` through the other Mac, using the supplied password only at the SSH prompt.
2. Verify that the actual Git root is `/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning`, record the hostname and both physical GPU UUIDs, and inspect the available runtime. A saved address is not proof of server identity or GPU count.
3. Fetch the published `codex/postsubmission-research-20260930` branch and check out an exact commit in the existing authorized repository. Preserve local changes; do not force a reset. Keep caches, environments, temporary files, logs and outputs inside that repository.
4. Qualify the repaired BUDDY code with all seven CPU checks in the actual runtime. Stage the full official collab graph, preserving unopened test pairs until the complete family lock. Use the same common cache for all predictors.
5. Measure a complete resource epoch for all five BUDDY arms before admitting the fixed 15-cell development comparison. Queue independent jobs across the two verified GPU UUIDs. Retain all results and account for common cache work, fitting, validation and storage.

The full-data BUDDY comparison tests whether private rank-one factors retain an ensemble benefit with compact predictor storage when independent members also share the deterministic graph cache. It is a practical test of an extension, not an established new learner. The separate literature task investigates graph-informed uncertainty over fast factors and will preserve prior-work conflicts. The active one-GPU graph-initialization study continues under its original registry; its receipts will not be relabeled as 18.77 results.
