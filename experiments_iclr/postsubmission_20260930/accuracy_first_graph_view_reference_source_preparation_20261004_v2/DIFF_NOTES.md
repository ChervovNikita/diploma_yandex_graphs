# V2 exact repair notes

Predecessor: `accuracy_first_graph_view_reference_source_preparation_20261004_v1`. Preserved predecessor manifest SHA-256: `54695f4ed086ed843835a7b040e8ac82a7e211cf17e39ae06b226a23c60b5c05`.

- Relative output directories become absolute at creation. Ordinary image writing resolves checkpoint paths; native GNNM uses a thin absolute-path adapter over the byte-identical bank image helper.
- CPU/CUDA admission runs before fitting and ordinary native construction. Unsupported backends are rejected without seeding, constructing or writing outputs. CUDA aliases still delegate to the unchanged selected-device resolver.
- Both context validators require all four actual input identities in `input_bindings_by_split[str(role.split)]`. Missing or conflicting entries fail; applicable self/bank source-manifest checks remain exact.
- Objectives, training steps, evaluation, restoration, transition, pooling, fixed seeds and clocks are unchanged. All numerical test files are byte-identical to v1. Native GNNM retains all five byte-identical bank-helper copies.
- `9` stdlib checks passed; `test_admission_stdlib.py` exercises the repaired public branches with project-local synthetic fixtures. Exact v2 numerical checks remain unrun. Root's v1 CPU reports and default-CUDA exact-replay failure do not qualify v2 GPU execution; deterministic-runtime qualification remains pending.

`SOURCE_DIFF.patch` records exact changed/new Python source versus v1. `PRESERVATION_CHECKS.json` records preserved source/review custody. No actual paired inputs, predictive results, experiment freeze, extra bank condition or neural method is introduced.
