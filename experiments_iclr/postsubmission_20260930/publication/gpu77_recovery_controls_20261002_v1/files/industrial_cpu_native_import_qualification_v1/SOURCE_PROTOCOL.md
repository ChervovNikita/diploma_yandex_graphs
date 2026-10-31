# CPU native imports under the content policy

## Status and goal

Prepared, **not executed**. The next diagnostic qualifies scientific/native imports after restriction. It requests no CSV/array loading, tensors, graph construction, model construction, fits, gradients or CUDA initialization. Import success would establish library compatibility with this CPU content policy; numerical/native parity and GPU behavior remain separate. No v2 chroot/mount-equivalence claim is made.

The immutable CPU fixture receipt is interpreted in `CPU_RECEIPT_INTERPRETATION.md`. `proven_cpu_policy.py` is a mechanically extracted source prefix of that exact sealed worker. All 11 extracted function/structure definitions are AST-identical, including capability removal, stdin/pin validation, close_range closure and the 78-instruction seccomp builder. No prior source, failed receipt, native v2 or precision-study file was edited.

## Active files

| File | Role |
| --- | --- |
| `proven_cpu_policy.py` | Exact extracted standard-library policy core. Its old fixture path constants and directory-only helper are retained as lineage; the new worker uses its own fixed run contract and extended rule builder. |
| `native_import_worker.py` | Trusted stdlib bootstrap, Landlock/file-rule extension, unchanged seccomp, explicit content/syscall probes, then 14 imports and selected distribution metadata. |
| `root_outer.py` | Fixed public UUID inventory, runtime/source integrity checks, public-only metadata binding, readonly stdin, fresh project artifacts and bounded child supervision. |
| `root_run_native_imports.py` | Fixed authorized SSH route; validates its manifest/seal, sends pinned sources and retains local receipts. |
| `IMPORT_CONTRACT.json` | Explicit interpreter, library roots/files, source/manifests, public paths, imports, environment paths and caps. |

After source review, root may invoke from this folder:

```sh
python3 root_run_native_imports.py --run-id CPU_native_imports_root01
```

Use a fresh identity. There is no retry. Local/remote runs belong to this new qualification folder. Before remote artifact writes, a 10-second inventory query must return exactly `GPU-44039938-fd82-41d2-fefd-de71514e2fac`; it is public inventory only. Author verification is local source/text/AST/scalar metadata work, with no SSH, inventory, scientific import or GPU action.

## Interpreter and readonly boundary

The executable is the pinned regular image file `opt/python/bin/python3.12`, launched with `-I -S -B`. The image's `opt/native/bin/python` points absolutely to `/opt/python/...` and is not used in this host-path route. Scientific site setup and `.pth` execution are skipped; `sys.path` is explicitly replaced with image stdlib/lib-dynload, image scientific site-packages and sealed native source.

The contract grants readonly file/directory content to these three dedicated image subtrees:

- `runtime_image_v1/opt/python`.
- `runtime_image_v1/opt/native/lib/python3.12/site-packages`.
- `runtime_image_v1/usr/lib/x86_64-linux-gnu`.

It also grants the sealed native source packet and 11 individually pinned host loader/system-library files. Host file descriptors come from image-manifest copy hashes, resolved to canonical regular files during preflight. No broad host `/usr/lib`, home, repo, proc or dev grant is supplied. Explicit image `LD_LIBRARY_PATH` entries cover Python, selected system/driver libraries, Torch and the image's NVIDIA wheel library directories. Reading CUDA shared-library bytes is not GPU context/device access.

Trusted preflight checks the pinned image manifest, enumerates exactly the exposed image subtree entries, verifies regular-file bytes and symlink targets, rejects unexpected/special entries, and checks all sealed native source bytes against its manifest/seal. Library symlinks must resolve within allowed image roots. Dedicated-prefix custody and concurrent trusted-host modifications remain root obligations. It reads/hashes library/source bytes for integrity and imports only stdlib; it does not import scientific packages.

## Public inputs and writable boundary

The existing public bundle is the fixed `public_probe_run_v1/public_inputs` root. Its only readable files are `ROLE_FREEZE.json` and the four public `tolokers-2` members: info, features, edge list and official masks. Parent directories receive READ_DIR alone. Private label files added later would gain no READ_FILE permission. Existing labels, raw archive or unrelated files cause the public-name preflight to refuse.

A local ROLE_FREEZE copy/hash was unavailable. Trusted outer preflight therefore measures its existing SHA, validates CLOSED/source-archive/role-policy metadata, records it in `RUN_BINDINGS.json`, and the restricted worker rechecks that hash. Declared public-member SHA/size metadata is logged; CSV/member contents are not loaded, decoded or rehashed. This diagnostic establishes an import boundary, not public-data integrity/numerical acceptance.

Only fresh `outputs`, `cache` and `tmp` trees receive ordinary read/write/create/remove rights. The child has sanitized HOME/cache/tmp variables, empty CUDA_VISIBLE_DEVICES, DGLBACKEND=pytorch and one-thread library settings. Inherited proc remains the original proc mount; no proc contents are granted. The worker creates no mount/chroot aliases.

## Restriction and descriptor sequence

The fresh image interpreter initializes trusted stdlib and reads its retained pinned worker/core/contract and run-binding metadata. It verifies six new namespace identities, mapped-root PID1, single-threaded startup and absence of scientific packages. Capabilities are removed, no_new_privs observed, stdio reviewed, and all nonstdio/nonpin FDs closed through UINT_MAX.

The new Landlock builder retains the complete ABI1 handled mask and proven directory/output rights. Its only extension is READ_FILE rules for explicit regular files plus READ_DIR-only public parents. Temporary rule descriptors are closed, close_range runs again, and the exact extracted seccomp builder is installed. Exec, pathname truncate, readonly O_TRUNC/openat2, metadata/xattr mutation, cross-process FD/memory, io_uring, namespace/mount APIs, AF_UNIX/foreign-FD receipt, alternate ABIs and FD-pin replacement stay protected. ftruncate retains the closed-FD/new-writable-FD argument.

The only ioctl-allowed pin is validated `/dev/null`, major1/minor3. Native imports do not receive GPU FDs or a GPU pin policy. Character-device/context compatibility is not inferred from this CPU pin. The new rule builder is source-prepared and unexecuted; the passed fixture receipt alone cannot certify its broader readonly paths.

## Imports and observable boundaries

The worker checks excluded/readonly opens without reading or writing their bytes, hashes only public metadata, samples truncation/open syscall denials on a synthetic fixture, and then imports packaging, NumPy, SciPy, pandas, YAML, delu, Torch, torchdata, DGL, sklearn.metrics and four native definition modules. It records each start/completion and resolved origin, enforces approved module roots and the existing top-version specifiers, and checks Torch CUDA remains uninitialized. It calls no Model constructor, native qualification routine or fit. Third-party import initializers execute inside the restrictions; their compatibility remains to be observed.

Receipts log trusted-startup, Landlock, seccomp and import-completion boundaries, source/binding hashes, namespaces, permissions and explicit syscall return/errno observations. `outputs/python_audit.jsonl` immediately logs selected Python open/dlopen/socket/process events and import boundaries, capped at 2048 audit events. This is **not a complete native syscall trace**; native C operations can bypass Python audit hooks. Postimport observations check the retained pin still reaches `/dev/null` ENOTTY and an ordinary output FD still gets ioctl EPERM.

Trusted integrity preflight has a 300-second cap; worker imports have 120 seconds; a 430-second outer alarm terminates an active child on outer failure; transport has 450 seconds. Every failure is retained, including stage, error and worker stdout/stderr. A missing library or denied required operation is a compatibility finding for source review; restrictions are not weakened automatically. Root may inspect the recorded paths/errors before choosing any source revision.
