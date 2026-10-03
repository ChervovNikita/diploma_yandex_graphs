"""Stdlib custody and narrow root-released J/F invocation contracts."""
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
ASSESSMENT_SHA = "ef83d71ad431d2efb84f4a2d6907547165443ecf70cd6d2b5448565b0ceded26"
BASE_DRIVER_SHA = "a59c669356e1a1437ce90107a7c76eb8f0b5579dc48d67b4378a7e237e956b7e"
DATA_AUTHORITY_SHA = "df6980064b3ee31377e6d96215146e1d954abf5464182464582801eddf17077d"
RUNTIME_AUTHORITY_SHA = "e63f602baa8a91e129fb4a0c0debd6ec282e7cd132be87eb66cd8c5cf0b7c882"
DATA_FILES = ("raw/node-feat.csv.gz", "raw/edge.csv.gz", "split/time/train.pt", "split/time/valid.pt")
ARMS = ("J", "F")
STAGES = ("numerical", "full_graph", "fit", "diagnostics", "close")
ENGINEERING_SEED = 20261003
ENGINEERING_SIGN_SEED = 2026100301
DIAGNOSTIC_SEED = 2026100307


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
    require(file_sha(path) == expected_sha, "Bound JSON differs: " + str(path))
    return json.loads(path.read_text())


def verify_manifest(root, expected_sha):
    root = Path(root).resolve()
    manifest = read_bound_json(root / "MANIFEST.json", expected_sha)
    for row in manifest["files"]:
        path = (root / row["path"]).resolve()
        require(path.is_relative_to(root), "Manifest path escaped packet")
        require(path.stat().st_size == row.get("bytes", row.get("size")) and file_sha(path) == row["sha256"], "Sealed payload differs: " + row["path"])
    return manifest


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path); temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def preflight(release_path, stage):
    release_path = Path(release_path).resolve()
    release = json.loads(release_path.read_text())
    require(release.get("schema") == "ncnc-structural-pattern-root-release-v1", "Wrong release schema")
    require(stage in release.get("authorized_stages", []) and release.get("root_authorization_reference"), "Stage not root released")
    driver_sha = file_sha(HERE / "MANIFEST.json")
    require(release.get("driver_manifest_sha256") == driver_sha, "Release targets another preparation")
    verify_manifest(HERE, driver_sha)
    expected = {"design_root": PLAN_SHA, "prototype_root": MODEL_SHA, "resource_root": RESOURCE_SHA,
                "base_driver_root": BASE_DRIVER_SHA, "assessment_root": ASSESSMENT_SHA}
    paths = {key: Path(release[key]).resolve() for key in expected}
    for key, pin in expected.items():
        verify_manifest(paths[key], pin)
    require(release["data_authority_sha256"] == DATA_AUTHORITY_SHA and release["runtime_authority_sha256"] == RUNTIME_AUTHORITY_SHA, "Authority adaptation is outside this preparation")
    authority = read_bound_json(release["data_authority_file"], DATA_AUTHORITY_SHA)
    require(authority.get("schema") == "ncnc-collab-TRAIN-VALID-data-authority-v1" and set(authority["files"]) == set(DATA_FILES) and authority.get("test_file_opened") is False, "Exactly TRAIN/raw/VALID authority required")
    runtime = read_bound_json(release["runtime_authority_file"], RUNTIME_AUTHORITY_SHA)
    require(runtime.get("schema") == "ncnc-collab-predictive-runtime-authority-v1" and runtime.get("ordinary_host_execution") is True and runtime.get("TF32") is False and runtime.get("mixed_precision") is False and runtime.get("deterministic_algorithms") is False, "Qualified normal runtime profile differs")
    plan = json.loads((HERE / "PILOT_PLAN.json").read_text())
    require(plan["base_seed"] == 0 and plan["epochs"] == 100 and plan["lambda"] == 1 and plan["arms"] == list(ARMS), "Fixed representative pair differs")
    require(release.get("family_id") == "ncnc-structural-pattern-pair-20261003-v1", "Wrong scientific family identity")
    return {"release": release, "release_path": release_path, "release_sha256": file_sha(release_path),
            "paths": paths, "plan": plan, "authority": authority, "runtime": runtime,
            "identity": {"driver_manifest_sha256": driver_sha, "prototype_manifest_sha256": MODEL_SHA,
                         "base_driver_manifest_sha256": BASE_DRIVER_SHA, "assessment_manifest_sha256": ASSESSMENT_SHA,
                         "data_authority_sha256": DATA_AUTHORITY_SHA, "runtime_authority_sha256": RUNTIME_AUTHORITY_SHA,
                         "family_id": release["family_id"]}}


def admit_invocation(context, stage, unit, seed, output, resume):
    require(stage in STAGES and seed == 0, "Only fixed seed0 stages supported")
    require(unit in ARMS if stage == "fit" else unit == "pair", "Stage/unit differs")
    invocation = {"stage": stage, "unit": unit, "base_seed": seed, "output_directory": str(Path(output).expanduser().resolve())}
    require(invocation in context["release"].get("authorized_invocations", []), "Invocation not root released")
    require(not resume or stage == "fit", "Only own fit journals may resume")
    require(not (Path(context["release"]["closure_output_directory"]) / "CLOSURE.json").exists(), "Pair is closed")
    return invocation


def runtime_stdlib(context):
    import importlib.metadata
    import sys
    authority = context["runtime"]
    executable = Path(sys.executable).resolve()
    require(executable == Path(authority["interpreter_path"]).resolve() and file_sha(executable) == authority["interpreter_sha256"], "Runtime interpreter differs")
    versions = {name: importlib.metadata.version(name) for name in authority["distribution_versions"]}
    require(versions == authority["distribution_versions"], "Runtime distribution versions differ")
    return versions


def qualification_required(context, stage):
    pins = context["release"].get("qualification", {})
    names = ("numerical",) if stage == "full_graph" else ("numerical", "full_graph")
    for name in names:
        pin = pins[name]
        r = read_bound_json(pin["path"], pin["sha256"])
        require(r["schema"] == "ncnc-pattern-qualification-v1" and r["identity"] == context["identity"] and r["stage"] == name and r["status"] == "PASS", "Missing/different " + name + " qualification")
        require(r["state_donor"] is False and r["test_file_opened"] is False, "Qualification state must not be a scientific donor")
        if name == "full_graph":
            require(r["full_TRAIN_epochs"] == 2 and r["full_VALID_evaluations"] == 2 and r["project_metric_computed"] is False and r["full_graph_numerical_and_serialized_replay"] is True, "Full J/F numerical/resource coverage incomplete")


def fresh_output(path, *, resume=False):
    path = Path(path).expanduser().resolve()
    require(path != HERE and not path.is_relative_to(HERE), "Output must be outside source preparation")
    require(not any((a / "MANIFEST.json").exists() for a in (path, *path.parents)), "Output is inside a sealed packet")
    if resume:
        require(path.is_dir() and (path / "JOURNAL.json").is_file(), "Own resume journal missing")
    else:
        require(not path.exists(), "Fresh output exists")
        path.mkdir(parents=True, mode=0o700)
    return path


def lock_output(path):
    import fcntl
    descriptor = os.open(Path(path) / ".RUN_LOCK", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(descriptor); raise RuntimeError("Output already has active worker")
    return descriptor
