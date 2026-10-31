# CPU content sandbox prototype v1

## Status and scope

Prepared, **not executed**. Two standard-library Python modules implement a root-owned OS diagnostic with fixed synthetic fixtures. The outer wrapper first performs a bounded public GPU-UUID inventory check for the authorized host; the worker is CPU-only. There is no industrial v3 native packet, numerical gate, real input, archive, label, model or GPU scientific computation. Sealed industrial v2 and the precision study remain unchanged.

Root adopted a prospective content-access contract: deny covered file contents and writes outside explicit paths, while disclosing pathname/metadata visibility. This prototype does not certify v2's absent-host-path, chroot or fresh mount contract. Before restriction, a trusted isolated system interpreter imports standard-library code, reads its own retained source, namespace/capability metadata and synthetic fixtures, and validates stdio plus `/dev/null`. No scientific imports occur at this stage or later.

## Modules and protocol

| File | Purpose |
| --- | --- |
| `cpu_content_diagnostic.py` | Namespace checks, capability removal, descriptor closure/pin validation, Landlock ABI1, focused x86_64 seccomp and bounded fixture tests. |
| `root_run_cpu_prototype.py` | Fixed authorized SSH route, source pin check, fresh repo-owned run, isolated system Python and bounded receipt collection. Its embedded `OUTER` controls the remote diagnostic. |

After source review, root may invoke the wrapper with a fresh identity:

```sh
python3 root_run_cpu_prototype.py --run-id cpu_content_v1
```

The wrapper transmits the pinned worker source as a quoted argument. Before any remote directory or fixture write, the outer process runs `/usr/bin/nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits` with a 10-second cap and requires exactly `GPU-44039938-fd82-41d2-fefd-de71514e2fac`. Failure stops before fixture creation. This is public inventory, with no GPU computation; the author has not invoked it.

Remote system Python verifies the worker hash, retains it under `root_runs/<identity>/trusted_source`, creates fixed synthetic fixtures, and launches `/usr/bin/unshare` with distinct user/mount/PID/net/IPC/UTS namespaces and `--propagation unchanged`. No actual mount is requested; after restriction, a null-argument syscall fixture checks the mount denial. `/usr/bin/python3 -I -S -B` starts mapped-root PID1, with sanitized environment, `/dev/null` stdin and wrapper-pipe stdout/stderr. The worker refuses a nonfixed repo root, source pin mismatch, shared startup threads, unexpected stdio, missing capability drop or failed required syscall. Its core-dump limit is zero.

Remote child wall cap is 30 seconds; local SSH transport cap is 60 seconds. There is no retry. Every failure is retained; use a new identity for any subsequent run. Local and remote fresh run directories contain `ROOT_LAUNCH.json` or `CONTENT_SANDBOX_CAPABILITY.json` and stdout/stderr as applicable. The outer process verifies its original namespace identities and mountinfo remain unchanged. A successful CPU receipt is only `PASSED_CPU_DIAGNOSTIC_ONLY`; it does not qualify libraries, GPU execution or representative model behavior.

## Rules

The worker handles all ABI1 filesystem rights, granting only:

- Read file/directory content under the fresh `public_inputs` fixture directory.
- Read/write, ordinary file/directory creation and removal under fresh `outputs`, `cache` and `tmp` directories.

It grants no broad home, repo, proc, dev, interpreter/library tree or special-file creation access. All contents are constant synthetic bytes. Imports are completed before restrictions, so this prototype does not establish postrestriction library availability. A future native design would require explicit reviewed library/source/input grants and new path metadata; the v2 chroot image is not automatically reusable at its host prefix.

Capabilities are removed and `no_new_privs` is observed. Temporary Landlock descriptors are closed. All nonstdio/nonpin inherited FDs are closed with `close_range` through `UINT_MAX`, rather than a finite cutoff; failure refuses the run. The final seccomp filter is installed while single-threaded and inherited by the bounded test child. Existing mappings are not retroactively revoked; the fresh trusted interpreter does not map real data before restriction.

## Seccomp and ABI1 limits

The fixed x86_64 classic-BPF filter rejects other audit architectures and kills x32 entries. It denies pathname truncation; readonly `O_TRUNC` in both `open` and `openat`; all `openat2`; chmod/chown/time/xattr mutation variants; ptrace/process_vm/pidfd_getfd; io_uring; mount/chroot/pivot_root/setns/unshare and the newer mount API; execve/execveat; AF_UNIX socket/socketpair creation; recvmsg/recvmmsg; and close_range. Syscall numbers >=453 are refused, covering later unreviewed additions such as new xattr-at calls. Denials return `EPERM`; Landlock path denials are expected to return `EACCES`.

Normal writable output `O_TRUNC` remains allowed. `ftruncate` is allowed because newly opened writable regular-file FDs must derive from allowed write paths, or newly created anonymous objects such as memfd; existing foreign FDs were closed and reception/acquisition paths are blocked. Readonly newly opened FDs cannot be `ftruncate`d under ordinary FD access-mode checks. Reviewed stdio and the character-device pin are explicit descriptor exceptions. Neither Landlock nor this filter conceals file metadata or promises a general information-flow sandbox. ABI1's ptrace-domain protection exists in addition to the conservative syscall denials. Cross-directory rename/link restrictions from missing ABI2 REFER remain.

The inherited proc mount belongs to its original PID namespace. The worker grants **no proc contents**; the negative own-status open and positive namespace `readlink` demonstrate the intended distinction. This is neither own-proc visibility nor a CUDA proc policy.

## Static FD pin argument

The only ioctl-allowed FD is a fresh `O_RDWR|O_CLOEXEC|O_NOFOLLOW` open of `/dev/null`, validated by `fstat` as a character device with the expected device identity. Its integer is recorded and kept occupied until process termination.

1. `close(pin)` and every `fcntl(pin, ...)` are denied. `close_range` is denied globally.
2. `dup2`/`dup3` targeting the pin are denied; ordinary dup/F_DUPFD cannot replace an already occupied descriptor.
3. A duplicated pin has a different integer and receives no ioctl permission.
4. Exec is denied, so CLOEXEC cannot release the pin in a continuing replacement program. Process teardown closes it only as that process ends.
5. There are no unconfined preexisting threads or inherited sockets. New threads/children inherit the restrictions; foreign-FD reception/acquisition and io_uring are denied. A new unrestricted file-table mutator is therefore absent.

For this CPU `/dev/null` pin, no descriptor-replacement hole is identified: `/dev/null` implements no successful ioctl that could install/replace a file descriptor. The filter proves the chosen FD remains a validated object under the covered userspace file-table operations. This is stronger than a request-family guess or a seccomp user-notification check that races against FD replacement.

It does **not** prove that arbitrary character-device ioctls are safe. A future GPU bootstrap would require a disclosed trusted public-only initialization stage, approved device/context FDs established before final lockdown, and review of driver operations that create, release or select resources. NVIDIA ioctls may create additional FDs after initialization; those FDs would not receive ioctl permission under this static policy. Late opening/context setup or required fcntl/close behavior can make the policy incompatible. Prevalidated character-device type alone does not establish one-GPU context scope. No GPU bootstrap, pin set or compatibility claim is supplied here.

## Bounded fixture observations

The worker checks positive readonly input reads and output write/ftruncate/writable truncating opens; negative excluded reads/writes/creates and readonly writes; path and readonly-open truncation denials; metadata/xattr, memory/FD, async and namespace/exec denials; no new proc/device-content grant; and disclosed metadata visibility. Pin tests cover close, close_range, dup2/dup3 replacement, mutating/duplicating fcntl, unpinned regular-file ioctl, duplicated-pin ioctl and retained pin identity. A single forked child attempts x32 getpid and must die with SIGSYS. These are synthetic OS diagnostics only.

Runtime results are unknown. Source/AST review, payload pinning and syscall-policy inspection cannot substitute for root's bounded run. Failed Landlock, seccomp or FD-invariant evidence must be preserved; the policy must not be weakened to obtain a pass.
