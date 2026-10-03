"""Stdlib admission, source custody and durable private metadata."""
from hashlib import sha256
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import tempfile

HERE = Path(__file__).resolve().parent
PLAN_SHA = "e97aeb2655d54f261acca1874c8e03047608625aac1b5e1e980853f7d44e0004"
MODEL_SHA = "a99b0e3b0e8ec597b03e2c390ac176597b5b6422d365fc20624ab98a113d42f9"
RESOURCE_SHA = "031c6a1fc36514a7dd1b7b520e5ec251b2c3f385bcab917ca101f6184712790a"
DATA_AUTHORITY_SHA = "df6980064b3ee31377e6d96215146e1d954abf5464182464582801eddf17077d"
DATA_FILES = ("raw/node-feat.csv.gz", "raw/edge.csv.gz", "split/time/train.pt", "split/time/valid.pt")
UNITS = ("native_bank4", "factor_private4", "factor_pooled4", "native70")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_sha(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_bound_json(path, expected_sha):
    path = Path(path).resolve()
    require(file_sha(path) == expected_sha, "Bound JSON custody mismatch: " + str(path))
    return json.loads(path.read_text())


def verify_manifest(root, expected_sha):
    root = Path(root).resolve()
    manifest = read_bound_json(root / "MANIFEST.json", expected_sha)
    for row in manifest["files"]:
        path = (root / row["path"]).resolve()
        require(path.is_relative_to(root), "Manifest path escaped packet")
        require(path.stat().st_size == row.get("bytes", row.get("size")) and file_sha(path) == row["sha256"],
                "Sealed payload mismatch: " + row["path"])
    return manifest


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def utc():
    return datetime.now(timezone.utc).isoformat()


def preflight(release_path, stage):
    release_path = Path(release_path).resolve()
    release = json.loads(release_path.read_text())
    require(release.get("schema") == "ncnc-collab-predictive-driver-root-release-v1", "Wrong root release schema")
    require(stage in release.get("authorized_stages", []) and release.get("root_authorization_reference"), "Stage not root released")
    driver_sha = file_sha(HERE / "MANIFEST.json")
    require(release.get("driver_manifest_sha256") == driver_sha, "Root release targets another driver")
    verify_manifest(HERE, driver_sha)
    paths = {key: Path(release[key]).resolve() for key in ("design_root", "prototype_root", "resource_root")}
    verify_manifest(paths["design_root"], PLAN_SHA)
    verify_manifest(paths["prototype_root"], MODEL_SHA)
    verify_manifest(paths["resource_root"], RESOURCE_SHA)
    plan = json.loads((paths["design_root"] / "PILOT_PLAN.json").read_text())
    authority = read_bound_json(release["data_authority_file"], release["data_authority_sha256"])
    require(authority.get("schema") == "ncnc-collab-TRAIN-VALID-data-authority-v1", "Wrong data authority schema")
    require(set(authority["files"]) == set(DATA_FILES), "Authority must contain exactly TRAIN/raw/VALID files and no TEST")
    require(release["data_authority_sha256"] == DATA_AUTHORITY_SHA, "Different TRAIN/VALID authority")
    require(authority.get("test_file_opened") is False, "This stage cannot adopt TEST authority")
    runtime = read_bound_json(release["runtime_authority_file"], release["runtime_authority_sha256"])
    require(runtime.get("schema") == "ncnc-collab-predictive-runtime-authority-v1" and runtime.get("ordinary_host_execution") is True and
            runtime.get("TF32") is False and runtime.get("mixed_precision") is False and runtime.get("deterministic_algorithms") is False,
            "Different qualified normal-runtime profile")
    return {"release": release, "release_path": release_path, "release_sha256": file_sha(release_path),
            "paths": paths, "plan": plan, "authority": authority, "runtime": runtime,
            "identity": {"driver_manifest_sha256": driver_sha, "design_manifest_sha256": PLAN_SHA,
                         "prototype_manifest_sha256": MODEL_SHA, "resource_manifest_sha256": RESOURCE_SHA,
                         "data_authority_sha256": release["data_authority_sha256"],
            "runtime_authority_sha256": release["runtime_authority_sha256"]}}


def admit_invocation(context, stage, unit, seed, output, resume):
    require(stage in ("synthetic", "valid_engineering", "resource_engineering", "fit", "family_lock"), "Unknown stage")
    require(unit in UNITS and seed in range(5), "Outside frozen unit/seed family")
    release = context["release"]
    require(isinstance(release.get("family_id"), str) and bool(release["family_id"]), "Missing root family identity")
    invocation = {"stage": stage, "unit": unit, "base_seed": seed,
                  "output_directory": str(Path(output).expanduser().resolve())}
    require(invocation in release.get("authorized_invocations", []), "Invocation not explicitly root released")
    family_lock_output = Path(release["family_lock_output_directory"]).expanduser().resolve()
    if stage == "family_lock":
        require(invocation["output_directory"] == str(family_lock_output), "Family closure destination differs")
    if stage == "fit":
        require(not (family_lock_output / "FAMILY_LOCK.json").exists(), "Family is already immutable; fit/resume/replacement is closed")
    require(not resume or stage == "fit", "Only fit stage can resume its own epoch journal")
    context["identity"]["family_id"] = release["family_id"]
    context["identity"]["family_lock_output_directory"] = str(family_lock_output)
    return invocation


def lock_output(path):
    import fcntl
    descriptor = os.open(Path(path) / ".RUN_LOCK", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(descriptor)
        raise RuntimeError("Execution output already has an active worker")
    return descriptor


def runtime_stdlib(context):
    import importlib.metadata
    import sys
    authority = context["runtime"]
    executable = Path(sys.executable).resolve()
    require(executable == Path(authority["interpreter_path"]).resolve() and
            file_sha(executable) == authority["interpreter_sha256"], "Root runtime interpreter differs")
    versions = {name: importlib.metadata.version(name) for name in authority["distribution_versions"]}
    require(versions == authority["distribution_versions"], "Runtime distribution versions differ")
    return versions


def fit_qualification(context):
    gate = context["release"]["fit_qualification"]
    synthetic = read_bound_json(gate["synthetic"]["path"], gate["synthetic"]["sha256"])
    require(synthetic["schema"] == "ncnc-pilot-synthetic-qualification-v1" and synthetic["status"] == "PASS" and synthetic["identity"] == context["identity"], "Synthetic driver qualification is incomplete/different")
    receipts = gate["resource_engineering"]
    require(len(receipts) == 4 and {r["unit"] for r in receipts} == set(UNITS), "All four complete TRAIN/VALID engineering units required")
    admitted = {}
    for pin in receipts:
        receipt = read_bound_json(pin["path"], pin["sha256"])
        require(receipt["schema"] == "ncnc-pilot-complete-engineering-v1" and receipt["identity"] == context["identity"] and receipt["unit"] == pin["unit"] and
                receipt["stage"] == "resource_engineering" and receipt["status"] == "COMPLETE_FINITE", "Resource/VALID qualification is incomplete/different")
        count = 4 if pin["unit"] == "native_bank4" else 1
        require(receipt["full_TRAIN_epochs"] == receipt["full_official_VALID_evaluations"] == count and receipt["metric_computed"] is False and receipt["resource_state_donor"] is False, "Engineering work/donor scope differs")
        require(len(receipt["records"]) == count and all(r["TRAIN"]["full_batches"] == r["TRAIN"]["optimizer_steps"] == 17 and r["VALID"]["positive_queries"] == 60084 and r["VALID"]["negative_queries"] == 100000 for r in receipt["records"]), "Engineering finite/count coverage differs")
        admitted[pin["unit"]] = receipt
    p, q = (admitted[u]["records"][0] for u in ("factor_private4", "factor_pooled4"))
    require(p["initial_state_sha256"] == q["initial_state_sha256"] and p["initial_rng_sha256"] == q["initial_rng_sha256"] and p["TRAIN"]["stream"] == q["TRAIN"]["stream"] and p["final_rng_sha256"] == q["final_rng_sha256"], "Engineering F4 initial/negative/permutation/dropout schedules differ")


def fresh_output(path, *, resume=False):
    path = Path(path).expanduser().resolve()
    require(path.is_absolute() and path != HERE and not path.is_relative_to(HERE), "Execution output must be external to sealed driver")
    require(not any((ancestor / "MANIFEST.json").exists() for ancestor in (path, *path.parents)), "Output lies inside a sealed packet")
    if resume:
        require(path.is_dir() and (path / "JOURNAL.json").is_file(), "Resume output lacks own journal")
        require(not (path / "FAMILY_CLOSURE.json").exists(), "Family-closed unit is immutable and cannot resume")
    else:
        require(not path.exists(), "Fresh output already exists")
        path.mkdir(parents=True, mode=0o700)
    return path
