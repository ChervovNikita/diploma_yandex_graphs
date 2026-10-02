# CPU native import exception diagnostic v2

**Prepared and unexecuted.** This is a diagnostic revision of the failed v1 import qualification. It does not widen restrictions or library/public grants, patch packages, load datasets or models, or request tensors/fits/GPU compute. Failed v1 and its fetched receipt/audit remain unchanged.

## What the v1 evidence establishes

Root's `cpu_native_imports_root01` installed Landlock ABI1 and the retained **78-instruction seccomp filter** before scientific imports. `packaging` completed; `numpy` failed with the generic source-directory `ImportError`; worker stderr was empty. Three explicit syscall probes still returned EPERM. The audit has 40 rows and ends at NumPy's `_core/overrides.py` and `_utils/_inspect.py` source reads.

The likely failure class is NumPy core extension/dependency loading or initialization under the restricted host-path runtime. The current evidence **does not identify the underlying library, dependency, errno or syscall**. The outer message is not proof of execution from a NumPy source checkout: the worker explicitly changes directory to the fresh outputs tree, and imports from the pinned image's site-packages. No original cause/context or traceback was captured. The Python audit omits native C syscall results and cannot prove a specific shared-library access denial. No dependency or permission change is proposed here.

## Exactly what changes

- A fresh v2 run container and receipt schema identities.
- The worker reports both `__cause__` and `__context__`, retaining context even when suppressed by `raise ... from None`. It stores exception type/message/repr, errno/name/path/filename attributes and traceback filename/function/line frames.
- Useful traceback text is emitted on existing stderr and included in the JSON receipt and existing audit output. It reads no source lines and captures no frame locals/globals. Chain/message/frame/text limits are explicit; cycles are identified and diagnostic failures retain the original outer error.
- Python audit additionally records the existing `import` event, including its resolved-filename argument when present; the event cap remains 2048. This can expose a C-extension import filename, but is still not a complete native syscall trace. Failure receipts now retain the audit count.

`proven_cpu_policy.py` is byte-identical. `restrict_native` is AST-identical. All capability-removal, FD-close/pin, Landlock and seccomp calls/arguments are unchanged. Contract fields are identical except `run_container`; readonly image/library files/directories, public file grants and READ_DIR-only public parents remain exact. Public contents are not parsed. Writable trees have the same rights at the fresh v2 outputs/cache/tmp identities. The outer and launcher change only receipt schema identity; interpreter, environment, namespace setup, inventory-only preflight, pins, probes, 14 imports, caps and no-retry behavior remain as v1.

## Exact root launcher

After root source review, execute this once with its fresh identity:

```sh
python3 '/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/industrial_cpu_native_import_qualification_v2/root_run_native_imports.py' --run-id cpu_native_imports_root02_exception_chain
```

The launcher retains local `root_runs/cpu_native_imports_root02_exception_chain/ROOT_LAUNCH.json` and `NATIVE_IMPORT_CAPABILITY.json`. Inspect `worker_result.exception_diagnostics` and the outer receipt's `stderr`; the original traceback is also in remote `outputs/python_audit.jsonl`. The fresh remote run root is under the same phase, in `industrial_cpu_native_import_qualification_v2/root_runs/cpu_native_imports_root02_exception_chain`. No SSH or root launcher has been run in this preparation.

This remains CPU import compatibility only. Success would not establish numerical parity, data/model execution, GPU compatibility or chroot/mount equivalence. `SOURCE_CHECKS.json` verifies policy/grant identity and isolated stdlib synthetic exception fixtures; it does not verify the real NumPy failure cause. A subsequent root receipt must supply that evidence.
