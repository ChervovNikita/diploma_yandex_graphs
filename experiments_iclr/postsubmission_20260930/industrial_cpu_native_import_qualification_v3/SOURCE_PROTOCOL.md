# CPU native imports: one pinned project-local zlib file

**Prepared and unexecuted.** Failed v1/v2 source and receipts remain preserved. This route adds exactly the zlib dependency identified by root's v2 diagnostic. It acquires no other dependency and patches no package or runtime image.

## Evidence and binding

The retained `cpu_native_imports_root02_exception_chain` exception graph identifies `_multiarray_umath` and its inner loader error: **`libz.so.1: cannot open shared object file: No such file or directory`**. The existing image manifest has no exact `libz.so.1` payload. This is a verified loader diagnostic, not a complete native syscall/path trace.

Root acquired the canonical system file `/usr/lib/x86_64-linux-gnu/libz.so.1.2.11` and copied its bytes into the fresh project-local regular file:

`/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/industrial_runtime_zlib_dependency_v1/libz.so.1`

Bound size: **108,936 bytes**. Bound SHA256: **`64c206f0146cc58bbddc4f22054436f4ff278f5a554aa3ce6921ddf7e9133370`**. `ZLIB_ACQUISITION_RECEIPT_BOUND.json` is an unchanged copy of root's local acquisition receipt. Root's acquisition was metadata/library-copy work; it did not qualify library execution. This source preparation has not accessed the remote library bytes or executed a library/package.

## Exact scope of the change

The contract appends this **one file descriptor** to its existing individually pinned library list (`host_readonly_library_files` is the inherited field name; the added path is project-local). Existing outer preflight checks canonical regular-file size/SHA before worker start. The worker's unchanged file-rule builder grants that sole file READ_FILE. The containing directory is prepended once to `LD_LIBRARY_PATH`.

**No directory grant is added.** The colocated remote `ACQUISITION_RECEIPT.json` gains no worker content permission. No host library-directory, proc/dev, dataset, label or archive grant changes. All prior public files and READ_DIR-only public parents remain exact. The loader's additional search path does not grant any other file content.

The proven policy is byte-identical, and worker/outer/launcher are identical to v2 except receipt schema identity. Exception-chain diagnostics remain unchanged. All other contract fields—including interpreter, image/native manifests, source/import list, sanitized environment settings, namespaces, pinned device/FD handling, syscall probes, caps and no retry/fallback—remain exact. New writable paths are only the equivalent fresh v3 outputs/cache/tmp run identities.

## Exact root launcher

After root source review, run once:

```sh
python3 '/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/industrial_cpu_native_import_qualification_v3/root_run_native_imports.py' --run-id cpu_native_imports_root03_repo_zlib
```

Inspect the retained `NATIVE_IMPORT_CAPABILITY.json`, its `worker_result.exception_diagnostics` and stderr. Any next dependency/import failure remains visible and does not trigger acquisition, fallback or grant widening. This CPU import diagnostic requests no dataset loading, model construction, tensors, fitting, gradients or GPU initialization/compute.

`SOURCE_CHECKS.json` verifies exact source/policy/contract differences, acquisition receipt/hash bindings and preserved predecessors. Import compatibility remains unqualified until root runs this route; a successful import would not establish numerical parity, GPU support or chroot/mount equivalence.
