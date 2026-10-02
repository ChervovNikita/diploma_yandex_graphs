#!/usr/bin/env python3
"""Capability fixture. All ordinary-file write attempts stay inside actual repo."""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

REPO = Path("/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("allowed", "denied", "report"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    allowed, denied, report = (Path(getattr(args, name)).resolve(strict=False)
                               for name in ("allowed", "denied", "report"))
    require(REPO in allowed.parents and REPO in denied.parents
            and allowed.parent == denied.parent and report.parent == allowed,
            "fixture paths must be nested sibling roots inside actual repo")
    proof_path = Path(os.environ["BUDDY_NAMESPACE_PROOF_PATH"])
    proof = json.loads(proof_path.read_text())
    require(proof["mode"] == "qualify" and proof["write_root"] == str(allowed)
            and all(value is True for value in proof["certification"].values()),
            "qualified namespace proof missing")
    suffix = str(uuid.uuid4())
    probe = allowed / ("allowed-probe-" + suffix)
    probe.write_bytes(b"namespace capability fixture\n")
    require(probe.read_bytes() == b"namespace capability fixture\n", "allowed write/read failed")
    results = {"allowed_create_write_read": True}
    attempts = {
        "denied_create": lambda: os.open(denied / ("denied-probe-" + suffix),
                                        os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
        "denied_append": lambda: os.open(denied / "existing.txt", os.O_WRONLY | os.O_APPEND),
        "denied_mkdir": lambda: os.mkdir(denied / ("denied-directory-" + suffix)),
    }
    for name, operation in attempts.items():
        try:
            unexpected = operation()
        except OSError as exc:
            require(exc.errno == errno.EROFS, name + " did not fail with EROFS")
            results[name] = {"denied": True, "errno": exc.errno}
        else:
            if isinstance(unexpected, int):
                os.close(unexpected)
            raise RuntimeError(name + " unexpectedly succeeded; qualification FAILED")
    with open(sys.executable, "rb") as stream:
        require(bool(stream.read(1)), "installed interpreter is not readable")
    results["installed_runtime_readable"] = True
    fd = os.open("/dev/null", os.O_WRONLY)
    try:
        require(os.write(fd, b"character-device capability fixture\n") > 0,
                "character-device I/O failed")
    finally:
        os.close(fd)
    results["character_device_io_dev_null"] = True
    require(os.readlink("/proc/1/ns/pid") == os.readlink("/proc/self/ns/pid"),
            "host PID proc escape remains")
    results["private_proc_namespace"] = True
    value = {
        "schema": "namespace-write-guard77-qualification-v1",
        "qualification_passed": True,
        "actual_repo": str(REPO), "write_root": str(allowed),
        "policy_path": proof["policy_path"], "policy_sha256": proof["policy_sha256"],
        "source_manifest_path": proof["source_manifest_path"],
        "source_manifest_sha256": proof["source_manifest_sha256"],
        "proof_path": str(proof_path), "proof_sha256": sha(proof_path),
        "linux": proof["linux"], "tests": results,
        "scope": "repo-contained capability fixture; no scientific data, Torch, or CUDA test",
    }
    fd = os.open(report, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print("namespace guard qualification passed: " + str(report))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        print("namespace guard qualification FAILED: " + str(exc), file=sys.stderr)
        sys.exit(125)

