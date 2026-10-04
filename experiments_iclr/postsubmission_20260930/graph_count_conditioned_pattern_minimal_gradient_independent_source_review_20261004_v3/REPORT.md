# Independent v3 source review — PASS

Candidate: `graph_count_conditioned_pattern_minimal_gradient_preparation_20261004_v3`

Candidate manifest SHA-256: `a3e4fe67e9da71903edcd5bde23a7736e31cb53e8e5dece2f24eda6ae011fa77`

## Narrow result

F02 is closed. After the final durable terminal write, v3 computes the current terminal hash and byte count, refreshes the existing `TERMINAL.json` inventory row, and uses the same hash for the top-level custody link before republishing custody. Failure custody without an inventory remains failure. The only source delta is this repair in `supervise.py` lines369–378.

F01 remains closed from the sealed v2 review. The accepted sampled/event cap checks and disclosed finite publication, fsync, exception and stdio tails are unchanged. No recursive inventory or cap-check/write loop is added.

## Verified bindings and reused scope

All20 candidate manifest payloads and the seal binding pass. External repair pins and preserved v1/v2 review packets pass. Eight worker/science/plan/release/prerequisite/staging/template files and four retained provenance payloads are byte-identical to v2. All12 supervisor functions outside `main`, and the complete module AST outside `main`, are unchanged. The computed unified diff exactly matches `V2_TO_V3.diff`; reversing the single repair block in memory restores v2 supervisor bytes.

The completed v1 scientific/source review and v2 cap-closure review are reused. No broader scientific review, source modification, target import/execution, numeric library, arrays/states/scores, server access, staging or launch occurred. This PASS is an independent source review; `execution_authorized` remains false and it makes no numerical, predictive or resource-execution qualification.

Binding and identity checks: 79 PASS; blocking findings: none.
