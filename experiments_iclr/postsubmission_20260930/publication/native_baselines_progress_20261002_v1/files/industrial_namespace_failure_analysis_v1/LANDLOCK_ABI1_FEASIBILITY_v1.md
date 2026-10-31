# Landlock ABI1: bounded feasibility assessment

## Verdict

The read-only root query proves that the x86_64 kernel exposes Landlock ABI1 (`landlock_create_ruleset`, syscall 444, version flag 1, return 1, errno 0). No ruleset, restriction, seccomp filter, library import or GPU action was exercised. Availability permits considering a distinct v3; it does not establish an operational boundary.

Landlock ABI1 can mediate covered path-based file opens and filesystem operations without sudo, after `no_new_privs`. A focused seccomp complement could address several ABI1 gaps for a content-read/write contract. That is conditional feasibility, with GPU ioctl and proc compatibility unresolved. It cannot reproduce v2's contract requiring absent host/archive paths, fresh proc/tmp/dev mounts and a chroot root. If those visibility properties remain required, this route is unsuitable.

## Contract differences

| Property | Consequence for a possible v3 |
| --- | --- |
| Covered reads outside an allowlist | Denied, generally with `EACCES`; denial is not path removal. |
| Metadata and known pathname visibility | `stat`, `readlink` and other uncovered metadata operations can expose names, ownership, sizes and targets. Denying directory reads reduces enumeration but does not hide known names. |
| Root and mount view | Existing host root/mounts remain. Working user/PID/net/IPC/UTS namespaces can be retained, but no new proc mount was available. |
| Existing v2 guard | `namespace_guard.py` explicitly rejects permission-only denial and requires the exact mounted root. It cannot truthfully certify this mechanism unchanged. |
| Existing v2 paths/runtime | `/sources`, `/inputs`, `/outputs` and `/opt/native` aliases are not created by Landlock. The v2 image rewrites interpreter symlinks/configuration to absolute `/opt/python` paths. New explicit paths and launch metadata are necessary; the chroot image is not automatically runnable from its host prefix. |
| Physical GPU nodes | A ruleset can limit direct opens to the approved `/dev/nvidiaN` and required ctl/UVM nodes. Other nodes remain visible by metadata. `CUDA_VISIBLE_DEVICES` does not enforce kernel isolation, and path rights alone do not constrain driver resources selected through an allowed ioctl. |

## Read allowlist and descriptor boundary

Use reviewed dedicated interpreter/environment trees, the necessary explicit system/driver library paths, sealed scientific source, and the approved compact/public input bundles. Grant execute only where needed. Grant write/create/remove rights only to fresh reviewed output/cache/tmp trees. Do not grant broad repo, home, `/usr`, `/proc` or `/dev` roots to make imports work: a broad repo/home grant would expose the raw archive, unrelated evidence or credentials. Actual resolved library and symlink targets must fit the chosen rules. Import/loader compatibility is unproved.

Install restrictions in a single-threaded trusted bootstrap before scientific imports, data mapping or worker threads. Set `no_new_privs`, remove capabilities and clear the environment. Landlock restriction and seccomp filters must apply to every subsequent thread/descendant; an already running unconfined thread cannot be left behind.

Landlock does not revoke already opened descriptors or existing writable mappings. Close all unneeded FDs, including directory/O_PATH handles, proc/mem/data handles, sockets and device handles; do not rely on a finite FD-number cutoff. Keep only reviewed stdin/stdout/stderr endpoints, with no readback from sensitive host files and no writable sink outside the approved outputs. Close ruleset/path handles after installation. Prevent later foreign-FD acquisition as well: ABI1 does not scope pathname UNIX sockets or SCM_RIGHTS reception. A small filter may deny AF_UNIX creation and/or `recvmsg`/`recvmmsg`, subject to NVIDIA runtime compatibility. Existing net namespaces alone should not be treated as proof against every pathname UNIX socket.

## ABI1 gaps and a focused seccomp complement

| Gap | Prospective complement and limit |
| --- | --- |
| Truncation | ABI1 lacks `LANDLOCK_ACCESS_FS_TRUNCATE` (added ABI3). Deny `truncate` and its architecture aliases. Reject `O_TRUNC` combined with read-only access in `open`/`openat`; an ABI1 read grant alone can otherwise permit such truncation. Deny `openat2` unless a separate mechanism safely handles its pointed-to flags: classic seccomp cannot inspect that structure. Normal writable output opens may retain `O_TRUNC` because Landlock checks their write access. |
| `ftruncate` | Existing writable FDs are not retroactively restricted. A blanket denial is simple but may break outputs/shared-memory/cache use. Allowing it can be justified only after descriptor closure and prevention of foreign FD acquisition, so every writable file descriptor derives from an allowed write path or a new approved anonymous object. Ordinary readonly FDs cannot be `ftruncate`d by their access mode. Neither option is operationally checked here. |
| Metadata mutation | ABI1 does not mediate chmod/chown, timestamp or xattr changes. Deny their pathname and FD syscall variants; these calls cannot be filtered by pathname using classic seccomp. This also forbids such mutations on output files, so library behavior must fit that restriction. |
| Cross-process access | ABI1 already checks ptrace domain relationships: a sandboxed tracer cannot ptrace a target outside its permitted subdomain relationship. Conservative denial of `ptrace`, `process_vm_readv/writev` and `pidfd_getfd` can simplify the boundary when unused. These are complements, not a claim that Landlock has no ptrace protection. |
| Async I/O | Deny `io_uring_setup/enter/register` instead of assuming ABI1 coverage of async operations and their submitting credentials. Compatibility is unproved. |
| Alternate syscall entries | Verify the x86_64 audit architecture and reject x32/other syscall-ABI entries so number/argument filters cannot be bypassed. Include relevant legacy variants in a real implementation. |
| Device/file ioctls | ABI1 has no ioctl access right. Blanket denial breaks CUDA. Classic seccomp sees the FD number and request value, not the FD's inode type/path; it cannot express "allow NVIDIA ioctls, deny all regular-file mutation" by file type. A finite reviewed ioctl request policy might work on the actual filesystems/driver, but merely blocking a few known file-mutation requests does not prove an exhaustive boundary. This is the main unresolved GPU write-boundary issue. |

ABI1 also predates `REFER` (ABI2): cross-directory rename/hard-link behavior is restrictive and may affect atomic output/cache operations. ABI1 has no Landlock TCP mediation; network isolation would come from the existing separate net namespace and any explicit syscall policy. These limits belong in a v3 contract, not in an equivalence claim.

## Own-proc requirement

The inherited proc mount remains associated with its original PID namespace even when the worker enters a new PID namespace. Landlock cannot make other PID directories or their metadata disappear. Denying their covered content opens is a weaker property than "only own proc visible".

A prospective own-content policy would enumerate only necessary actual `/proc/self/...` file handles in the installing process, avoid whole `/proc` or `/proc/self` directory grants, and exclude `/proc/self/root`, `fd` and memory escape handles. Rules attach to resolved filesystem objects, not a textual wildcard that automatically follows every future child/thread. The proc object's rule support, inode lifetime and descendant behavior have not been demonstrated by the ABI query. Broad directory grants are not a substitute for this check.

The existing GPU UUID binding reads `/proc/driver/nvidia/gpus/*/information`; it is outside own-proc. CUDA may require additional proc/sys metadata. Literal own-proc-only therefore conflicts with the current v2 identity method. A distinct v3 must either authorize narrow public driver metadata explicitly or adopt a separately justified identity method. No such replacement is supplied or accepted here.

## Bounded next decision

First decide whether the changed content-access contract is acceptable. If it is, a short root-owned, CPU-only ruleset/seccomp capability probe in fresh scratch could determine whether restrictions install and close the identified file mutation/FD paths, without archive/label/model access. Library and GPU compatibility would remain a separate existing public operational check. No full v3 packet, restriction implementation or extra scientific qualification collection is prepared in this assessment.

Evidence: `LANDLOCK_ABI_LAUNCH_v1.json` SHA-256 `796b288b84d13cbb6ac32f5a6df839212616975bc08112e4830f0c666f5ac597`; completed mount diagnostic as cited in `MOUNT_FAILURE_CONCLUSION_v1.md`; focused reads of sealed v2 bootstrap/guard/runtime-launch sources. This is local feasibility analysis only. Sealed v2, the live precision study and root-owned run evidence remain untouched.
