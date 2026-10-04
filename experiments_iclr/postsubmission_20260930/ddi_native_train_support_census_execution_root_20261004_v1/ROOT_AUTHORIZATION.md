# One DDI TRAIN geometry census

Root reviewed the sealed source manifest `989f10f25f6b1fd2affa58609c24bd957f2ac1755c6040aa0fdf139aa5546758`, native mask semantics, negative sampler, exact counting reductions and declared limits. This is a root technical review, not independent manuscript review.

Authorize one seed-0 pass of all 43 native full masks, batch 24,576, with tail 11,143 discarded and counting chunks 1,024. Use authorized 18.77 GPU0 only, after checking available memory; this is GPU geometry work, correcting the earlier CPU-only planning description. No model, likelihood, optimizer, score, VALID or TEST access is authorized. Do not reuse this state for training or retry automatically.

The worker checks declared sampled memory limits and an 1,800-second workload cap. The owning supervisor enforces an 1,860-second child wall cap and can signal only the child's exact original dedicated process group. It records the child's physical exit. No server configuration, repository checkout or unrelated process is changed.

This identifies structural opportunity and a dynamic-programming cost proxy. It does not establish predictive superiority, novelty, runtime of a trained model, or numerical equivalence to a separate sparse implementation.
