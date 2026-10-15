# Portable private input routes: actual CPU qualification

V2 source manifest is `21ad8f2c644a1833f0015dfc4c0c0193bcfc9598b6f84d98b9d686330bed8115`.
V1 and frozen runtime/pilot sources remain unchanged. The two method laws and
IDs remain unchanged; v2 repairs repository binding and callback provenance.

| Actual source case | Scope per method | Outcome |
| --- | --- | --- |
| Native GIN MolBACE, dropout 0.5 | Two optimizer transitions on the same 32 real TRAIN graphs, three complete 151-graph VALID passes, serialized checkpoint and fresh-session resumed update | Both methods passed |
| Native PolyFormer CiteSeer-Cheb | Full graph of 3,327 nodes, two native updates using all 1,929 TRAIN nodes, three complete 665-node VALID passes, checkpoint and fresh-session resumed update | Both methods passed |

Both methods preserved initial reference outputs exactly and restored VALID
outputs with maximum absolute difference 0 on these CPU checks. Original native
Parameter identities were retained. CiteSeer used the source's nonreentrant
activation checkpointing. Across both datasets, actual work was eight native
constructions and eight optimizer attempts/completions. Torch was 2.1.2+cu118,
execution used two CPU threads and CUDA was hidden. No GPU job or TEST scoring
was added. These are engineering checks, not scientific fits or accuracy evidence.

The node check was admitted after inspecting its existing 147.8 MB cached basis
and available host memory. No heavy Physics or full-native link CPU job was
started. A remote dispatch token-substitution SyntaxError caused zero execution
and is preserved separately; all actual native CPU cases passed.

The launcher must explicitly provide `recipe._repository_root`, plus the method
manifest and transform/callback hashes. The actual phase/source must be inside
that root. The link bridge injects the runner root and rejects a conflict.
Outer scientific and inner frozen-native callback receipts are distinct.

The method definition can enter the proposed manifest. Remaining qualifications
are the other node/molecular source cells, actual NCN/NCNC/BUDDY integration and
task scope, CUDA/two-process resources, full-epoch worker/selector integration,
and final-serving reconstruction for the new callback/state identities. Current
frozen serving rejects these new identities; it must not fall back to base GNNM.
No all-cell execution admission, convergence, novelty, accuracy gain or giant
campaign GO follows from these checks.

Machine-readable details and exact evidence hashes are in the adjacent
`portable_private_input_v2_READINESS.json` and preserved observation receipts.
