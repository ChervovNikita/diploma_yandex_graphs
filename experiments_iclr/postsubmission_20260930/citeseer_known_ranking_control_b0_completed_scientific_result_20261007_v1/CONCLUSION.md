# Complete one-seed ranking-loss control

All three fixed60-cycle fits completed with clean exit0 and retained checkpoint/logit/history hashes. Existing stored validation histories were checked for the first maximum at the fixed5-cycle cadence. No new training, forward or TEST scoring was performed by the audit.

| Model | Selected validation MRR | Hits@10 | Selected cycle |
|---|---:|---:|---:|
| Single |0.2723|0.5154|60|
| Jointly trained untied four |0.2889|0.4978|40|
| Shared four |0.2869|0.5022|10|

Shared four improves MRR over single by0.0146 and is0.0020 below jointly trained untied four. Hits@10 is below single. This is one seed on a development validation split, with no uncertainty estimate or TEST confirmation. The untied arm uses joint Adam and a pooled-plus-member objective, and is not an ordinary independently trained ensemble. BPR is existing prior work and this control is not a methodological novelty or a new-method superiority claim. Cross-runtime comparisons to older pilot values are not treated as paired effects. Preserve this result and do not expand a ranking-loss grid on its basis.
