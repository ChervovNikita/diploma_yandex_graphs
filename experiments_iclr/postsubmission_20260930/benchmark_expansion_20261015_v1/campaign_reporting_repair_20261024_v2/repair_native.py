"""Apply the reporting repair before the unchanged named native entrypoint."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def guard():
    expected = {
        "anogena-2-0": ["GPU-44039938-fd82-41d2-fefd-de71514e2fac"],
        "olig": ["GPU-9c6c20fa-a620-479d-f72f-163ac74deef2", "GPU-30283973-140d-9e7b-e780-0fcc487dec54"],
    }
    require(socket.gethostname() in expected, "Unrecognized scientific host")
    uuids = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()
    require(uuids == expected[socket.gethostname()], "Wrong physical GPU identity table")


def pin(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def source(expected):
    manifest = pin(HERE / "MANIFEST.json")
    require(manifest["sha256"] == expected, "Repair manifest changed")
    for row in json.loads((HERE / "MANIFEST.json").read_text())["files"]:
        path = (HERE / row["path"]).resolve()
        require(path.is_relative_to(HERE), "Repair source escapes package")
        actual = pin(path)
        require(all(actual[key] == row[key] for key in ("bytes", "sha256")), "Repair source changed")
    return manifest


def preserve_recovery(native, config_path, config_sha, record_dir):
    require(pin(config_path)["sha256"] == config_sha, "Recovery authority changed")
    config = json.loads(Path(config_path).read_text())
    fit_id = native[native.index("--task-id") + 1]
    require(fit_id in config["repairs"], "No explicit reporting recovery for this fit")
    row = config["repairs"][fit_id]
    output = Path(native[native.index("--output") + 1]).resolve()
    require(str(output) == row["output"], "Recovery output changed")
    require(native[native.index("--resume") + 1] == str(output / "recovery.pt"), "Own recovery checkpoint required")
    record_dir = Path(record_dir).resolve()
    require(record_dir.is_relative_to(output.parent.parent), "Recovery journal escapes own task")
    target = record_dir / "preserved_before_reporting_resume"
    target.mkdir()
    preserved = {}
    for name, receipt in row["artifacts"].items():
        path = output / name
        require(pin(path) == receipt, "A failed-run artifact changed before reporting recovery: " + name)
        # The native resume already preserves RESULT.json. Preserve all mutable
        # logs and the previously written export before repeating finalization.
        if name not in ("selected.pt", "recovery.pt"):
            shutil.copy2(path, target / name)
            preserved[name] = pin(target / name)
    journal = {"schema": "preserved-before-explicit-reporting-resume-v1",
               "job_id": fit_id, "recovery_authority": pin(config_path),
               "original_artifacts": row["artifacts"], "preserved_copies": preserved,
               "required_new_training_epochs": 0, "TEST_accessed": False}
    (record_dir / "REPORTING_RECOVERY_INPUTS.json").write_text(json.dumps(journal, indent=2, sort_keys=True) + "\n")


def main():
    guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependency-dir")
    parser.add_argument("--repair-source-sha256", required=True)
    parser.add_argument("--recovery-config")
    parser.add_argument("--recovery-config-sha256")
    parser.add_argument("--record-dir")
    parser.add_argument("native_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    native = args.native_argv[1:] if args.native_argv[:1] == ["--"] else args.native_argv
    manifest = source(args.repair_source_sha256)
    if args.recovery_config:
        require(args.recovery_config_sha256 and args.record_dir, "Explicit recovery pin and journal required")
        preserve_recovery(native, args.recovery_config, args.recovery_config_sha256, args.record_dir)
    from reporting_patch import install
    repair = install()
    if args.record_dir:
        (Path(args.record_dir) / "REPORTING_REPAIR_PRODUCER.json").write_text(
            json.dumps({"source": manifest, "installed": repair, "native_argv": native}, indent=2, sort_keys=True) + "\n")
    sys.path.insert(0, str(HERE.parent / "campaign_named_processes_20261021_v1"))
    from launch_named import run_native
    run_native(native, args.dependency_dir)


if __name__ == "__main__":
    main()
