# Namespace write guard for 18.77 — bubblewrap source preparation v7

Actual repository:
/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git

Target: Linux 5.4 x86_64. Parent reports installed bubblewrap 0.6.1.
V1 had a source-manifest hash defect caught before remote deployment. V2 copied
the reviewed payloads exactly and repaired its seal; actual qualification then
failed before mounts because inherited mountpoints were stacked. V1 and V2 are
preserved. V3 changed the mount backend to the existing purpose-built bubblewrap.
Root's direct capability probe then showed bwrap --remount-ro denied /dev/null
I/O (EACCES) while allowed/denied repo writes behaved correctly. V4 changed ONLY
the device-tree readonly operation to preserve existing device mount flags.
Actual v4 qualification then rejected /sys/kernel/debug/tracing with EACCES.
V5 adds only an empty read-only private tmpfs at /sys/kernel/debug; the strict
visible-mount verifier continues to reject EACCES rather than skip it.
Actual v5 qualification next rejected shared/slave mount propagation. V6 adds
only an explicit recursive-private propagation mount inside the new namespace
before device-tree remounts; host mount propagation is unchanged.
V6 filesystem qualification passed on 18.77. Its data-free CUDA test failed OS
error 304. Parent reports strace found successful device opens/ioctls and root-style
NVIDIA chmod/provisioning attempts under namespace UID0. V7 changes ONLY numeric
identity mapping: caller host UID/GID remain the same numeric values inside the
user namespace. This is an unqualified diagnostic; readonly mounts stay literal.

## Readiness

Source preparation only. No shell, SSH, namespace/mount operation, scientific
runtime, Torch/CUDA import, or capability fixture was executed in preparing v7.
The source manifest is generated and read-back verified from actual file bytes.
No syntax/test success or current execution enforcement is claimed. Root must
source-review and qualify this package on 18.77.

Stage the four sealed payloads and SOURCE_MANIFEST.json unchanged into a fresh
repo-local folder, conventionally phase/namespace_write_guard77_v7.
DEFECT_NOTE.json is a separate preparation record outside the four-payload seal.
Preserve existing scientific source and the exact v3 launcher argv.

## Mount and credential policy

Outer uses /usr/bin/bwrap with new user, mount, and PID namespaces, the ordinary
caller's same numeric host UID/GID mapped identically, --as-pid-1,
--die-with-parent, and --new-session. Inside maps must be exactly HOST_UID HOST_UID 1
and HOST_GID HOST_GID 1, with ordinary UID>0.
It constructs a read-only root using --ro-bind / /. Immediately after inner
libc initialization, existing namespace CAP_SYS_ADMIN applies
mount(NULL, /, NULL, MS_REC|MS_PRIVATE, NULL) before device seals. This changes
propagation only in the new mount namespace. The final strict propagation
certificate is unchanged and must succeed after capability cleanup.

--dev-bind /dev /dev preserves host CUDA character devices. Inner reads ONLY
visible device-subtree mount targets (including nested pts/shm) via fdinfo mnt_id
and remounts them readonly, deepest first. The mount call preserves actual VFS
flags, including existing NODEV where appropriate, and never adds NODEV to the
device-bearing bind. Bwrap --remount-ro is omitted for /dev because the direct
root probe showed that it blocked character-device I/O. Covered host mounts are
never remounted. Fresh proc still uses bwrap's --remount-ro; NODEV there is
appropriate. ONLY /sys/kernel/debug is masked with a private empty tmpfs and
remounted read-only before inner. Its obscured tracing descendants become
nonexistent in this namespace; host mounts/files remain unchanged. Inner requires
the debug mount to be read-only tmpfs and retains the existing EACCES rejection.
The sole writable bind is actual repo or the qualification allowed root.

After remounting only the visible device subtree, inner certifies visible mount entries
through mountinfo plus fdinfo mnt_id. Every visible mount except the permitted
bind must be read-only, propagation private, and proc private to the new PID
namespace. Raw mountinfo is retained separately for audit. Hidden/covered mounts
are not reachable through ordinary paths and are not promoted to visible evidence.
A writable nested mount beneath the permitted bind is rejected.

Bubblewrap drops all capabilities and adds only namespace CAP_SYS_ADMIN for
device-subtree remounts and CAP_SETPCAP for final bounding-set cleanup. After
those bounded remounts, inner sets no_new_privs, clears capabilities, and
verifies CapInh/CapPrm/CapEff/CapBnd/CapAmb all zero. It closes inherited FDs >=3;
writable regular-file stdio must resolve to matching inodes within writable_root.
Pipes, sockets, and character-device stdio remain allowed.

Guard remains PID 1 to forward signals/reap; scientific argv is executed unchanged
as an ordinary child. No sudo or fallback is attempted. A nonzero mapped UID may lose bwrap's retained
mount capabilities across exec; if so the unchanged mount operation fails closed.
Root must qualify the mapping before runtime use. A source, capability, or
certification failure exits 125 before scientific launch.

## Root qualification and execution interface

Use the intended installed Python with -B. These are argument vectors for root:

1. Capability fixture:

   [PYTHON, "-B", GUARD, "qualify", "--fixture-root", NEW_FIXTURE_ROOT]

   Fixture root is a new canonical repo-local directory with existing parent.
   Allowed/denied siblings and all ordinary-file write probes are inside that
   directory. stdout/stderr must be terminal/pipe or a file inside allowed.
   Tests: allowed create/write/read; denied create/append/mkdir must yield EROFS;
   installed-interpreter read; /dev/null character-device I/O; private proc.
   No scientific data, Torch, or GPU experiment is included.

   Success artifacts:
   - NEW_FIXTURE_ROOT/allowed/mount_proof.json
   - NEW_FIXTURE_ROOT/allowed/qualification_result.json

2. Root reviews and admission-binds exact source manifest, MOUNT_POLICY.json,
   successful qualification, and installed-bwrap evidence. Failure means no
   unguarded scientific launch.

3. Actual execution:

   [PYTHON, "-B", GUARD, "run", "--proof", FRESH_REPO_LOCAL_PROOF,
    "--qualification", QUALIFICATION_RESULT, "--", ...UNCHANGED_V3_ARGV]

   Proof must be a fresh canonical repo-local path with existing parent. It is
   O_EXCL-created after mount and credential certification immediately before
   child launch. Its SHA is dynamic: v3 records it at entry; admission prebinds
   source/policy/qualification rather than a future proof SHA.

   Root may use the same interface for a data-free runtime/import qualification
   before authorizing scientific argv. CUDA compatibility is a root operational
   qualification, not a claim made by this source package.

## Buddy integration (unchanged schemas)

Exports canonical BUDDY_NAMESPACE_POLICY_PATH, BUDDY_NAMESPACE_PROOF_PATH,
BUDDY_NAMESPACE_SOURCE_MANIFEST_PATH, and BUDDY_NAMESPACE_QUALIFICATION_PATH.

Policy: namespace-write-guard77-policy-v1.
Manifest: namespace-write-guard77-source-manifest-v1; files is a name→SHA object.
Qualification: namespace-write-guard77-qualification-v1.
Proof: namespace-write-guard77-execution-proof-v1.

Actual proof mode=execute and actual_repo=write_root=actual repo.
Proof binds policy, source-manifest, qualification, exact argv, namespace IDs,
zero credentials, stdio, visible mounts, raw_mountinfo, device_tree_remounts, private_debug_mask,
recursive_private_propagation_applied, and installed bwrap hash.
All seven certification fields must be true:
outside_mounts_read_only, write_root_bind_writable, private_mount_propagation,
private_pid_proc, inherited_extra_fds_closed, capabilities_zero, no_new_privs.

argv_sha256 is SHA256 of
json.dumps(argv, ensure_ascii=True, separators=(",", ":")).encode("utf-8").

V3 validates/records external guard evidence and retains its own independent
hardware-enforcement flag as false. Guard certification covers its bounded visible
mount policy for the current process, not v3's independent implementation.

## Limits

Character-device, network, socket, and pipe I/O remain allowed. Trusted host
processes can modify shared files concurrently. This is not a hostile-host/code
containment framework. Read-only /dev structurally preserves device I/O but does
not prove CUDA/NVML compatibility. Cache/temp environment belongs to buddy v3.

No old guard, scientific source, data, or output is changed by this preparation.
Unsupported capabilities, writable visible mounts, external writable stdio,
source drift, stale proof paths, or incomplete credential certification fail closed.
