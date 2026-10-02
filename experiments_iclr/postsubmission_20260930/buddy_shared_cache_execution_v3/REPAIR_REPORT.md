# BUDDY shared-cache source v3: dynamic module registration

Root's authorized v2 CPU receipt (`buddy_cpu_numerical_qualification_v2/REMOTE_CPU_RUN_RECEIPT.json`) reports six passed tests and one hash initialization error: PyG2.7 Inspector could not resolve `sys.modules['pinned_buddy_hashing']`. No certificate was emitted. V3 fixes that loader integration error without changing hashing mathematics.

## Exact delta from sealed v2

- `cache_builder.py` adds `load_registered_module`: registers the freshly created module in `sys.modules` before `exec_module`, then attaches its resolved source path/SHA identity. `native_hashes` calls this helper after its existing pin verification. An existing module is reused only when source identity, file path and spec origin match; differing/absent identity is refused without touching that entry. Any new-load failure restores the previously absent registry entry, including a loader that replaced its own entry.
- `test_guards.py` adds three stdlib checks covering registration before execution and reuse, changed-source/unrelated-module collision refusal, and cleanup after ordinary/replacement loader failures.
- README and this report describe the failure/repair; fresh local evidence and v3 source manifest/seal bind the new sources.

All other executable sources, including `models.py`, `run.py`, `guards.py`, `checkpoint_io.py`, `check_source.py`, `launch_family.py` and **all seven numerical tests in `test_cpu.py`**, are byte-identical to v2. CONFIG.json, SOURCE_PINS.json, dependency extras and every vendor snapshot remain identical to both sealed predecessors. Arms/seeds/cache semantics/objectives/tolerances are unchanged. Artifact format labels inherited from v2 are retained; source identity requires fresh v3 artifacts/certificate.

## Verification and limits

Stdlib source checks and all **11 synthetic guard checks pass locally**. Synthetic Python fixture modules alone were executed for loader lifecycle checks. No Torch/native hashing/model source, real data/model/checkpoint, GPU, remote action or network was executed/accessed by this v3 repair. No papers were acquired or read. V1/v2 seals and every manifest entry were reverified unchanged.

Root must rerun all seven numerical CPU tests without skips or tolerance changes:

```sh
python check_source.py
python test_guards.py
python test_cpu.py --output CPU_QUALIFICATION.json
```

Independent source recheck, fresh numerical qualification, official complete-data/cache correctness and all-arm resource qualification/admission remain pending. V2's six successful tests do not qualify v3. The v2 robustness/history/checkpoint/source guards and their documented limits remain intact.
