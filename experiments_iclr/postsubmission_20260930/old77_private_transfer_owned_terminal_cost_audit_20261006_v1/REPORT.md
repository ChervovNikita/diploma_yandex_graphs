# Original18.77 private-transfer queue: terminal and cost audit

Metadata observation 2026-10-06T04:40:15.064953+00:00. The two exact originally registered queue handles are absent. Each registered b1/b2 ten-cell block has BLOCK_FREEZE selected_blocks_complete=true and ten independent operational EXIT receipts: exit0, terminal wait observed, one attempt, no reason, signals or retries. The block-level complete=false correctly refers to the original full30-cell cohort, not failure of the selected ten-cell block. No QUEUE_FAILURE exists.

- b1: ten complete fits; sum of per-fit elapsed 21438.611199483275s; inclusive queue 21440.541244570166s; last exit 2026-10-05T12:43:36.563234+00:00.
- b2: ten complete fits; sum of per-fit elapsed 20926.903280753642s; inclusive queue 20927.991358924657s; last exit 2026-10-05T12:35:05.015093+00:00.

Old77 has20 completed physical fit attempts. Total per-fit elapsed 42365.51448023692s; sum of inclusive queues 42368.53260349482s. The two queue times overlap, so their sum is paid process elapsed, not elapsed wall duration or GPU energy. RSS/GPU peaks are sampled supervision metadata, not fresh allocator/kernel cost. All original starts, cell IDs, EXIT descriptors and terminal metadata remain in OBSERVATION.json.

This supersedes the unknown-status/lower-bound-six evidence only as a new audit; ATTEMPT_HISTORY.json is unchanged. Root owns any global cost-accounting successor. No scores, training histories, training FREEZE contents, checkpoints, predictions or labels were opened. Original donor exclusion and the chosen new26 physical fits remain unchanged. Operational completion neither admits these donors nor certifies their predictions/training correctness.
