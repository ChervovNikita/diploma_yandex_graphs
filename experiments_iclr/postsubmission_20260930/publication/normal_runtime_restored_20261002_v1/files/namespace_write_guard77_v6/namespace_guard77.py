#!/usr/bin/env python3
"""Source-only prepared Linux 5.4 mount guard; run qualification on 18.77 first."""
import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys

REPO = Path("/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git")
HERE = Path(__file__).resolve().parent
POLICY = HERE / "MOUNT_POLICY.json"
MANIFEST = HERE / "SOURCE_MANIFEST.json"
BWRAP = "/usr/bin/bwrap"
MS_RDONLY, MS_REMOUNT, MS_BIND = 1, 32, 4096
MS_REC, MS_PRIVATE = 16384, 1 << 18
PRESERVED = {"nosuid": 2, "nodev": 4, "noexec": 8, "noatime": 1024,
             "nodiratime": 2048, "relatime": 1 << 21, "strictatime": 1 << 24}
CERT_KEYS = ("outside_mounts_read_only", "write_root_bind_writable",
             "private_mount_propagation", "private_pid_proc",
             "inherited_extra_fds_closed", "capabilities_zero", "no_new_privs")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def argv_sha(argv):
    return hashlib.sha256(json.dumps(argv, ensure_ascii=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def within(path, root):
    return path == root or root in path.parents


def canonical(path, root=REPO, exists=True):
    supplied = Path(path)
    require(supplied.is_absolute(), "absolute path required")
    resolved = supplied.resolve(strict=exists)
    require(str(supplied) == str(resolved), "symlinks/noncanonical paths rejected")
    require(within(resolved, root), "path escapes permitted repository root")
    return resolved


def machine():
    u = os.uname()
    return {"sysname": u.sysname, "release": u.release, "machine": u.machine}


def load_source():
    u = machine()
    require(u["sysname"] == "Linux" and u["machine"] == "x86_64"
            and u["release"].split(".")[:2] == ["5", "4"],
            "profile requires Linux 5.4 x86_64")
    require(REPO.resolve(strict=True) == REPO and REPO.is_dir(), "actual repo mismatch")
    canonical(HERE)
    policy = json.loads(POLICY.read_text())
    require(policy["schema"] == "namespace-write-guard77-policy-v1"
            and policy["actual_repo"] == str(REPO), "policy identity mismatch")
    manifest = json.loads(MANIFEST.read_text())
    require(manifest["schema"] == "namespace-write-guard77-source-manifest-v1",
            "source seal schema mismatch")
    expected = {"namespace_guard77.py", "qualification_fixture.py",
                "MOUNT_POLICY.json", "README.md"}
    require(set(manifest["files"]) == expected, "source seal payload mismatch")
    for name, digest in manifest["files"].items():
        require(sha(HERE / name) == digest, "source seal mismatch: " + name)
    return policy


def decode_mount_path(value):
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), value)


def mounts():
    result = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        a, b = left.split(), right.split()
        require(len(a) >= 6 and len(b) >= 3, "malformed mountinfo")
        result.append({"id": int(a[0]), "parent": int(a[1]), "device": a[2],
                       "root": decode_mount_path(a[3]),
                       "target": decode_mount_path(a[4]),
                       "options": a[5].split(","), "optional": a[6:],
                       "filesystem": b[0], "source": decode_mount_path(b[1]),
                       "super_options": b[2].split(",")})
    require(result and any(m["target"] == "/" for m in result), "no root mount")
    return result


def close_extra_fds():
    for value in os.listdir("/proc/self/fd"):
        if value.isdecimal() and int(value) >= 3:
            try:
                os.close(int(value))
            except OSError as exc:
                require(exc.errno == errno.EBADF, "failed to close inherited fd")


def check_stdio(write_root):
    checked = []
    for fd in range(3):
        try:
            metadata = os.fstat(fd)
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        except OSError as exc:
            require(exc.errno == errno.EBADF, "could not inspect stdio")
            checked.append({"fd": fd, "kind": "closed"})
            continue
        writable = (flags & os.O_ACCMODE) != os.O_RDONLY
        mode = metadata.st_mode
        if stat.S_ISREG(mode):
            target = Path(os.readlink("/proc/self/fd/" + str(fd)))
            require(target.is_absolute() and target.resolve(strict=True) == target,
                    "stdio regular-file path cannot be certified")
            opened = target.stat()
            require((opened.st_dev, opened.st_ino) == (metadata.st_dev, metadata.st_ino),
                    "stdio regular-file inode mismatch")
            require(not writable or within(target, write_root),
                    "writable stdio regular file outside write root")
            kind = "regular"
        elif stat.S_ISFIFO(mode):
            kind = "pipe"
        elif stat.S_ISSOCK(mode):
            kind = "socket"
        elif stat.S_ISCHR(mode):
            kind = "character-device"
        else:
            require(not writable, "writable unsupported stdio descriptor")
            kind = "other-read-only"
        checked.append({"fd": fd, "kind": kind, "writable": writable})
    return checked


def libc_api():
    lib = ctypes.CDLL(None, use_errno=True)
    lib.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                          ctypes.c_ulong, ctypes.c_void_p]
    lib.mount.restype = ctypes.c_int
    lib.prctl.restype = ctypes.c_int
    return lib


def visible_mounts(entries, subtree=None):
    # fdinfo mnt_id identifies the top visible mount at a target. This reads
    # mount metadata only; covered inherited mounts are never remounted.
    by_id = {m["id"]: m for m in entries}
    visible = {}
    targets = {m["target"] for m in entries
               if subtree is None or within(Path(m["target"]), subtree)}
    for target in sorted(targets):
        try:
            fd = os.open(target, os.O_PATH | os.O_CLOEXEC)
        except OSError as exc:
            require(exc.errno in (errno.ENOENT, errno.ENOTDIR),
                    "visible mount target cannot be inspected: " + target)
            continue
        try:
            info = Path("/proc/self/fdinfo/" + str(fd)).read_text()
            values = dict(line.split(":", 1) for line in info.splitlines() if ":" in line)
            mount_id = int(values["mnt_id"].strip())
        finally:
            os.close(fd)
        require(mount_id in by_id, "visible mount id absent from mountinfo")
        entry = by_id[mount_id]
        if entry["target"] == target:
            visible[mount_id] = entry
    require(visible, "no visible filesystem mounts found")
    return list(visible.values())


def seal_visible_device_tree(lib):
    # Only the already established bwrap /dev tree is changed. Preserve its
    # actual VFS flags: never introduce NODEV on the device-bearing mount.
    entries = visible_mounts(mounts(), subtree=Path("/dev"))
    require(any(m["target"] == "/dev" for m in entries), "visible /dev bind missing")
    recorded = []
    for entry in sorted(entries,
                        key=lambda m: (m["target"].count("/"), len(m["target"])),
                        reverse=True):
        flags = MS_REMOUNT | MS_BIND | MS_RDONLY
        for name, bit in PRESERVED.items():
            if name in entry["options"]:
                flags |= bit
        if lib.mount(None, os.fsencode(entry["target"]), None, flags, None) != 0:
            raise OSError(ctypes.get_errno(),
                          "device-tree read-only remount failed: " + entry["target"])
        recorded.append({"mount_id": entry["id"], "target": entry["target"],
                         "remount_flags": flags, "prior_options": entry["options"]})
    return recorded


def certify_mounts(write_root):
    raw = mounts()
    final = visible_mounts(raw)
    require(all(not any(v.startswith(("shared:", "master:", "propagate_from:"))
                        for v in m["optional"]) for m in raw),
            "mount propagation remains shared/slave")
    permitted = [m for m in final if m["target"] == str(write_root)]
    require(len(permitted) == 1 and "rw" in permitted[0]["options"]
            and "ro" not in permitted[0]["options"], "write-root bind not writable")
    require(all("ro" in m["options"] and "rw" not in m["options"]
                for m in final if m["id"] != permitted[0]["id"]),
            "a visible mount outside permitted bind is writable")
    require(not any(m["id"] != permitted[0]["id"] and
                    within(Path(m["target"]), write_root) for m in final),
            "unqualified nested mounts beneath writable root")
    require(os.getpid() == 1, "guard is not init of a private PID namespace")
    require(os.readlink("/proc/1/ns/pid") == os.readlink("/proc/self/ns/pid"),
            "proc exposes a different PID namespace")
    proc = [m for m in final if m["target"] == "/proc"]
    require(len(proc) == 1 and proc[0]["filesystem"] == "proc"
            and "ro" in proc[0]["options"], "private read-only proc not certified")
    dev = [m for m in final if m["target"] == "/dev"]
    require(len(dev) == 1 and "nodev" not in dev[0]["options"],
            "device bind does not permit character-device I/O")
    debug = [m for m in final if m["target"] == "/sys/kernel/debug"]
    require(len(debug) == 1 and debug[0]["filesystem"] == "tmpfs"
            and "ro" in debug[0]["options"], "private read-only debug mask absent")
    return final, permitted[0], raw


def drop_privileges(lib):
    def prctl(option, arg=0):
        if lib.prctl(ctypes.c_int(option), ctypes.c_ulong(arg),
                     ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0)) != 0:
            code = ctypes.get_errno()
            raise OSError(code, "prctl failed: " + str(option))
    prctl(38, 1)  # PR_SET_NO_NEW_PRIVS
    prctl(47, 4)  # PR_CAP_AMBIENT / PR_CAP_AMBIENT_CLEAR_ALL
    last = int(Path("/proc/sys/kernel/cap_last_cap").read_text())
    require(0 <= last <= 63, "unsupported capability range")
    current = dict(line.split(":", 1) for line in
                   Path("/proc/self/status").read_text().splitlines() if ":" in line)
    if int(current["CapBnd"].strip(), 16):
        for cap in range(last + 1):
            prctl(24, cap)  # PR_CAPBSET_DROP; final cleanup after device remounts
    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]
    class Data(ctypes.Structure):
        _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32),
                    ("inheritable", ctypes.c_uint32)]
    header, data = Header(0x20080522, 0), (Data * 2)()
    lib.capset.argtypes = [ctypes.POINTER(Header), ctypes.POINTER(Data)]
    lib.capset.restype = ctypes.c_int
    if lib.capset(ctypes.byref(header), data) != 0:
        raise OSError(ctypes.get_errno(), "capset failed")
    status = dict(line.split(":", 1) for line in
                  Path("/proc/self/status").read_text().splitlines() if ":" in line)
    for field in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"):
        require(int(status[field].strip(), 16) == 0, field + " remains nonzero")
    require(status["NoNewPrivs"].strip() == "1", "no_new_privs absent")
    return {field: status[field].strip() for field in
            ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb", "NoNewPrivs")}


def read_qualification(path):
    qpath = canonical(path)
    qualification = json.loads(qpath.read_text())
    require(qualification["schema"] == "namespace-write-guard77-qualification-v1"
            and qualification["qualification_passed"] is True
            and qualification["actual_repo"] == str(REPO)
            and qualification["policy_sha256"] == sha(POLICY)
            and qualification["source_manifest_sha256"] == sha(MANIFEST)
            and qualification["linux"] == machine(),
            "matching successful capability qualification required")
    return qpath


def write_json_exclusive(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def supervise(argv, environment):
    # Guard remains PID 1, forwards signals, and reaps children. Scientific argv
    # runs as an ordinary PID, so default SIGTERM semantics remain available.
    child = os.fork()
    if child == 0:
        os.setsid()
        try:
            os.execvpe(argv[0], argv, environment)
        except BaseException as exc:
            print("namespace guard child exec failed: " + str(exc), file=sys.stderr)
            os._exit(125)
    def forward(signum, _frame):
        try:
            os.killpg(child, signum)
        except ProcessLookupError:
            pass
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
        signal.signal(signum, forward)
    while True:
        reaped, status_code = os.waitpid(-1, 0)
        if reaped == child:
            return (os.WEXITSTATUS(status_code) if os.WIFEXITED(status_code)
                    else 128 + os.WTERMSIG(status_code))


def inside(args):
    load_source()
    require(os.getuid() == 0 and os.geteuid() == 0, "mapped-root user namespace required")
    uid_map = Path("/proc/self/uid_map").read_text().split()
    gid_map = Path("/proc/self/gid_map").read_text().split()
    require(len(uid_map) == len(gid_map) == 3 and uid_map[0] == gid_map[0] == "0"
            and uid_map[2] == gid_map[2] == "1"
            and int(uid_map[1]) > 0, "ordinary host-user single-id map required")
    root = canonical(args.write_root)
    require(args.mode == "qualify" or root == REPO, "execute write root must be actual repo")
    proof = canonical(args.proof, root=root, exists=False)
    require(proof.parent.is_dir() and not proof.exists(), "fresh proof path required")
    qualification = read_qualification(args.qualification) if args.mode == "execute" else None
    argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    require(argv and argv[0], "unchanged launch argv missing")
    stdio = check_stdio(root)
    lib = libc_api()
    # Apply only inside the new bwrap mount namespace; no host propagation change.
    if lib.mount(None, b"/", None, MS_REC | MS_PRIVATE, None) != 0:
        raise OSError(ctypes.get_errno(), "recursive-private namespace propagation failed")
    close_extra_fds()
    device_remounts = seal_visible_device_tree(lib)
    final, writable, raw = certify_mounts(root)
    credentials = drop_privileges(lib)
    # Repeat mount certification after losing authority to alter it.
    final, writable, raw = certify_mounts(root)
    close_extra_fds()
    os.chdir(REPO)
    proof_value = {
        "schema": "namespace-write-guard77-execution-proof-v1",
        "mode": args.mode, "actual_repo": str(REPO), "write_root": str(root),
        "policy_path": str(POLICY), "policy_sha256": sha(POLICY),
        "source_manifest_path": str(MANIFEST), "source_manifest_sha256": sha(MANIFEST),
        "qualification_path": str(qualification) if qualification else None,
        "qualification_sha256": sha(qualification) if qualification else None,
        "proof_path": str(proof), "argv": argv, "argv_sha256": argv_sha(argv),
        "cwd": str(REPO), "linux": machine(), "uid_map": uid_map, "gid_map": gid_map,
        "namespaces": {name: os.readlink("/proc/self/ns/" + name)
                       for name in ("user", "mnt", "pid")},
        "certification": {name: True for name in CERT_KEYS},
        "credential_status": credentials, "stdio": stdio,
        "write_root_mount_id": writable["id"], "mounts": final,
        "raw_mountinfo": raw, "guard_backend": "bubblewrap",
        "device_tree_remounts": device_remounts,
        "private_debug_mask": "/sys/kernel/debug",
        "recursive_private_propagation_applied": True,
        "outer_program_path": BWRAP, "outer_program_sha256": sha(BWRAP),
        "scope": "ordinary path filesystem writes; character-device I/O and network allowed",
    }
    write_json_exclusive(proof, proof_value)
    environment = dict(os.environ)
    environment["BUDDY_NAMESPACE_POLICY_PATH"] = str(POLICY)
    environment["BUDDY_NAMESPACE_PROOF_PATH"] = str(proof)
    environment["BUDDY_NAMESPACE_SOURCE_MANIFEST_PATH"] = str(MANIFEST)
    environment["BUDDY_NAMESPACE_QUALIFICATION_PATH"] = str(qualification) if qualification else ""
    print("namespace guard certified; proof: " + str(proof), file=sys.stderr)
    return supervise(argv, environment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    qualify = commands.add_parser("qualify")
    qualify.add_argument("--fixture-root", required=True)
    run = commands.add_parser("run")
    run.add_argument("--proof", required=True)
    run.add_argument("--qualification", required=True)
    run.add_argument("argv", nargs=argparse.REMAINDER)
    inner = commands.add_parser("_inside")
    inner.add_argument("--mode", choices=("qualify", "execute"), required=True)
    inner.add_argument("--write-root", required=True)
    inner.add_argument("--proof", required=True)
    inner.add_argument("--qualification")
    inner.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "_inside":
        return inside(args)
    load_source()
    require(os.getuid() > 0 and os.geteuid() == os.getuid(), "no host-root/sudo launch")
    require(os.access(BWRAP, os.X_OK), "existing /usr/bin/bwrap required")
    if args.command == "qualify":
        fixture = canonical(args.fixture_root, exists=False)
        require(not fixture.exists() and fixture.parent.is_dir(), "new fixture root required")
        fixture.mkdir(mode=0o700)
        allowed, denied = fixture / "allowed", fixture / "denied"
        allowed.mkdir(mode=0o700)
        denied.mkdir(mode=0o700)
        (denied / "existing.txt").write_text("repo-contained denied fixture\n")
        write_root, proof, qualification = allowed, allowed / "mount_proof.json", None
        argv = [sys.executable, "-B", str(HERE / "qualification_fixture.py"),
                "--allowed", str(allowed), "--denied", str(denied),
                "--report", str(allowed / "qualification_result.json")]
    else:
        write_root = REPO
        proof = canonical(args.proof, exists=False)
        require(proof.parent.is_dir() and not proof.exists(), "fresh proof path required")
        qualification = read_qualification(args.qualification)
        argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
        require(argv and argv[0], "launch argv missing after --")
    check_stdio(write_root)
    close_extra_fds()
    inner_argv = [sys.executable, "-B", str(HERE / "namespace_guard77.py"), "_inside",
                  "--mode", "qualify" if args.command == "qualify" else "execute",
                  "--write-root", str(write_root), "--proof", str(proof)]
    if qualification:
        inner_argv += ["--qualification", str(qualification)]
    inner_argv += ["--"] + argv
    bwrap_argv = [BWRAP, "--unshare-user", "--uid", "0", "--gid", "0",
                  "--unshare-pid", "--as-pid-1", "--die-with-parent", "--new-session",
                  "--ro-bind", "/", "/", "--dev-bind", "/dev", "/dev"]
    bwrap_argv += ["--tmpfs", "/sys/kernel/debug",
                   "--remount-ro", "/sys/kernel/debug",
                   "--proc", "/proc", "--remount-ro", "/proc",
                   "--bind", str(write_root), str(write_root), "--chdir", str(REPO),
                   "--cap-drop", "ALL", "--cap-add", "CAP_SYS_ADMIN",
                   "--cap-add", "CAP_SETPCAP", "--"] + inner_argv
    os.execv(BWRAP, bwrap_argv)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        print("namespace guard FAILED CLOSED: " + str(exc), file=sys.stderr)
        sys.exit(125)
