# Readonly-stdin wrapper revision v2

## Status and predecessor

Prepared, **not executed**. This sibling folder contains a revised wrapper and an exact copy of the sealed v1 CPU worker. All existing v1 sources, seals, review records and `cpu_content_root01` evidence are preserved.

Root's v1 run on the authorized UUID route refused with `stdin must be readonly`. Its receipt reports `restriction_applied: false`, no fixture tests, unchanged worker source and unchanged host namespaces/mountinfo. The source checks stdin before applying Landlock or seccomp, so those restrictions remained untested. The failure is bound to the reviewed v1 wrapper SHA-256 `5c0913b402bb0f0ff67b112c8e174b41b06a79b3404d007438c13916a56a2253` and worker SHA-256 `2d0c1c18ae17432c20ded3cc509a90db5c5d3de9471b43f6a766e16e6b1973c4`.

The v1 wrapper used `Popen(stdin=subprocess.DEVNULL)`. CPython opens that endpoint read/write, conflicting with the worker's explicit readonly contract. The old receipt records the refusal but does not record the measured flags; the explanation is root's diagnosis, consistent with the launcher and CPython behavior. Its capability receipt SHA-256 is `e9efb5638a3afe560201d31f6d8c01384968142eccce2a344615bd7576e03339`.

## Exact wrapper correction

Immediately before child launch, the remote outer process explicitly opens `/dev/null` with `O_RDONLY|O_CLOEXEC|O_NOFOLLOW`, verifies `fstat` character-device type and Linux major 1/minor 3, and verifies `F_GETFL & O_ACCMODE == O_RDONLY`. It passes this integer FD as `Popen(stdin=stdin_fd)` and closes the outer copy in `finally`. The worker independently rechecks its resulting FD 0. The receipt records the outer measurement; stdout/stderr remain write-only wrapper pipes.

The readonly contract applies to the namespace worker's stdin. Trusted inventory and SSH transport still use `DEVNULL`. The worker bytes, Landlock rules, seccomp filter, static `/dev/null` ioctl pin, descriptor closure, synthetic fixture tests and capability/environment requirements remain identical. The wrapper schema and default fresh identity prefix identify this new prospective source revision. There is no automatic retry.

## Deliberate remote artifact layout

The immutable worker requires both its run root and retained source under the fixed remote path:

```text
.../postsubmission_20260930/industrial_content_sandbox_prototype_v1/root_runs/<fresh-identity>
```

Root explicitly authorized a **new** identity in that remote layout to retain worker byte identity. Existing remote v1 sources and failed-run files must remain unchanged. Revised wrapper source, preparation, seals and local launch/receipt files belong to this new sibling v2 folder. The wrapper refuses reuse of an existing remote run directory. Do not interpret the remote folder name as reuse of the old wrapper source.

After reviewing this new wrapper/source seal, root may invoke it from this folder with a fresh identity, for example:

```sh
python3 root_run_cpu_prototype.py --run-id cpu_content_root02
```

The fixed SSH route and 10-second inventory preflight are unchanged. Before any remote fixture writes, inventory must contain exactly `GPU-44039938-fd82-41d2-fefd-de71514e2fac`. This is public GPU inventory only. The worker remains CPU-only, the child wall cap is 30 seconds, and the transport cap is 60 seconds.

## Acceptance limits

Source and embedded-outer AST parsing plus byte/hash comparisons are the only author verification. No source was executed, no SSH/inventory/GPU action was performed by the author, and no real data/labels/models were read. Root's previous refusal supplies no Landlock/seccomp success claim. The new wrapper still requires root's bounded diagnostic to obtain any operational evidence. A CPU fixture pass would qualify only the prospective content-access mechanism; metadata visibility, inherited proc and the absence of GPU/library compatibility remain disclosed in the v1 protocol. Sealed native v2 and the live precision study are untouched.
