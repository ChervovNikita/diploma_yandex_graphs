"""Stdlib custody and one root-released prospective arm/seed invocation."""
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
ARMS = ("target_only", "joint", "separate")
STAGES = ("fit",)
ENGINEERING_SEED = 20261003
ENGINEERING_SIGN_SEED = 2026100301
DIAGNOSTIC_SEED = 2026100307
PREDECESSOR_SHA = "9021a598c7642bf428124de1230a078dfddbf3968095432354ee18a14541104c"
RUNTIME_PROFILE_ID = "ncnc-pattern-process-deterministic-20261004-v1"
RUNTIME_PROFILE_TRANSITION = {"deterministic_algorithms_before": False, "deterministic_algorithms_after": True,
                              "warn_only_after": False, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}


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
    import sys
    require("torch" not in sys.modules, "Stdlib admission must precede numerical imports")
    phase = HERE.parent
    plan = json.loads((HERE / "PILOT_PLAN.json").read_text())
    execution = phase / plan["execution_directory"]
    release_path = Path(release_path)
    require(not release_path.is_symlink() and release_path.resolve().is_relative_to(execution), "External own release required")
    require(sys.platform == "linux" and Path.cwd() == Path(plan["repository"]) and os.uname().nodename == "peptide", "Exact ordinary host/repository required")
    release = json.loads(release_path.read_text())
    require(release.get("schema") == "ncnc-structural-pattern-root-release-v1" and release.get("execution_enabled") is True
            and release.get("root_authorization_reference") and release.get("authorized_stages") == ["fit"], "Fixed fit not root released")
    driver_sha = file_sha(HERE / "MANIFEST.json")
    require(release.get("driver_manifest_sha256") == driver_sha and release.get("fit_supervision_manifest_sha256") == driver_sha,
            "Release targets another source packet")
    verify_manifest(HERE, driver_sha)
    for row in plan["source_pins"]:
        path = phase / row["path"]
        require(path.resolve().is_relative_to(phase) and not path.is_symlink() and path.stat().st_size == row["bytes"]
                and file_sha(path) == row["sha256"], "Dependency source/receipt pin differs: " + row["path"])
    review_pin = release["independent_source_review"]
    review_path = Path(review_pin["path"])
    require(review_path.resolve().is_relative_to(phase) and not review_path.is_symlink()
            and review_path.stat().st_size == review_pin["bytes"] and file_sha(review_path) == review_pin["sha256"], "Fresh independent review custody differs")
    review = json.loads(review_path.read_text())
    require(review.get("status") == "PASS" and not review.get("blocking_findings")
            and review.get("candidate_manifest_sha256") == driver_sha and review.get("execution_authorized") is False,
            "Fresh exact predictive source PASS required")
    require(plan["base_seeds"] == [0, 1, 2] and plan["epochs"] == 100 and plan["lambda"] == 1 and plan["arms"] == list(ARMS), "Prospective family changed")
    require(release.get("family_id") == plan["family_id"] and release.get("fit_caps") == plan["caps"]
            and release.get("cuda_visible_devices") == plan["GPU_UUID"] and release.get("runtime_profile_transition") == RUNTIME_PROFILE_TRANSITION
            and release.get("automatic_retry_or_resume") is False and release.get("TEST_supported") is False, "Bound family/profile/resources differ")
    require(release["data_authority_sha256"] == DATA_AUTHORITY_SHA and release["runtime_authority_sha256"] == RUNTIME_AUTHORITY_SHA, "Unchanged native authorities required")
    authority = read_bound_json(release["data_authority_file"], DATA_AUTHORITY_SHA)
    require(authority.get("schema") == "ncnc-collab-TRAIN-VALID-data-authority-v1" and set(authority["files"]) == set(DATA_FILES)
            and authority.get("test_file_opened") is False, "Exactly TRAIN/raw/VALID authority required")
    runtime = read_bound_json(release["runtime_authority_file"], RUNTIME_AUTHORITY_SHA)
    require(runtime.get("schema") == "ncnc-collab-predictive-runtime-authority-v1" and runtime.get("ordinary_host_execution") is True
            and runtime.get("TF32") is False and runtime.get("mixed_precision") is False and runtime.get("deterministic_algorithms") is False,
            "Existing ordinary runtime authority required")
    paths = {key: (phase / value).resolve() for key, value in plan["native_source_roots"].items()}
    require(Path(release["prototype_root"]).resolve() == paths["prototype_root"] and Path(release["design_root"]).resolve() == paths["design_root"], "Native source roots differ")
    context = {"release": release, "release_path": release_path.resolve(), "release_sha256": file_sha(release_path),
        "paths": paths, "plan": plan, "authority": authority, "runtime": runtime,
        "identity": {"driver_manifest_sha256": driver_sha, "prototype_manifest_sha256": MODEL_SHA,
            "base_driver_manifest_sha256": BASE_DRIVER_SHA, "data_authority_sha256": DATA_AUTHORITY_SHA,
            "runtime_authority_sha256": RUNTIME_AUTHORITY_SHA, "runtime_profile_id": RUNTIME_PROFILE_ID,
            "runtime_profile_transition": RUNTIME_PROFILE_TRANSITION, "family_id": plan["family_id"],
            "conditional_manifest_sha256": plan["conditional_manifest_sha256"],
            "native_bucket_result_sha256": plan["native_bucket_prerequisites"]["diagnostic"]["sha256"]}}
    qualification_required(context, "fit")
    return context




def driver_module_custody(context):
    """Read-only source/module binding check; no scientific imports or profile writes."""
    import sys
    require(file_sha(context["release_path"]) == context["release_sha256"], "Admitted root release bytes changed")
    manifest = verify_manifest(HERE, context["identity"]["driver_manifest_sha256"])
    for row in context["plan"]["source_pins"]:
        path = HERE.parent / row["path"]
        require(path.stat().st_size == row["bytes"] and file_sha(path) == row["sha256"], "Dependency source/receipt changed")
    for name, relative in (("graph_ops", "graph_ops.py"), ("prototype", "prototype.py")):
        module = sys.modules.get(name)
        if module is not None:
            require(Path(module.__file__).resolve() == context["paths"]["prototype_root"] / relative, "Native module alias shadowed")
    for row in manifest["files"]:
        if row["path"].endswith(".py") and Path(row["path"]).name == row["path"]:
            name = Path(row["path"]).stem
            module = sys.modules.get(name)
            if module is not None:
                require(Path(module.__file__).resolve() == HERE / row["path"], "V5 helper module shadowed: " + name)


def require_profile_receipt(receipt):
    """Only actual V5 transition/final-profile receipts can admit later stages."""
    transition = receipt["runtime_profile_transition"]
    before, after = transition["actual_profile_before"], transition["actual_profile_after"]
    require(transition["status"] == "TRANSITION_VERIFIED"
            and transition["declared_transition"] == RUNTIME_PROFILE_TRANSITION
            and transition["ordinary_runtime_authority_sha256"] == RUNTIME_AUTHORITY_SHA
            and transition["ordinary_runtime_source_binary_admission_completed"] is True
            and transition["preimport_environment"]["Torch_absent_before_configuration"] is True
            and transition["preimport_environment"]["Torch_absent_after_configuration"] is True
            and transition["preimport_environment"]["CUBLAS_WORKSPACE_CONFIG_after"] == ":4096:8"
            and before["deterministic_algorithms"] is False and before["deterministic_warn_only"] is False
            and before["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
            and after == dict(before, deterministic_algorithms=True, deterministic_warn_only=False)
            and transition["RNG_before_sha256"] == transition["RNG_after_sha256"]
            and set(transition["RNG_components_before_sha256"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
            and transition["RNG_components_before_sha256"] == transition["RNG_components_after_sha256"]
            and transition["actual_profile_matches_exact_transition"] is True
            and transition["RNG_exactly_unchanged"] is True, "V5 actual deterministic transition incomplete")
    final = receipt["runtime_profile_final"]
    require(final["actual_runtime_profile"] == after and final["runtime_profile_exact"] is True,
            "V5 final deterministic runtime profile differs")


def admit_invocation(context, stage, unit, seed, output, resume):
    require(stage == "fit" and unit in ARMS and seed in (0, 1, 2) and not resume, "Only fixed fresh nine-cell fits supported")
    invocation = {"stage": stage, "unit": unit, "base_seed": seed, "output_directory": str(Path(output).expanduser().resolve())}
    expected = str(HERE.parent / context["plan"]["execution_directory"] / ("fit_" + unit + "_seed" + str(seed)) / "run01")
    require(invocation["output_directory"] == expected and context["release"]["authorized_invocations"] == [invocation], "Exact cell not root released")
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
    plan = context["plan"]
    def read_pin(pin):
        path = Path(pin["path"])
        require(path.resolve().is_relative_to(HERE.parent) and not path.is_symlink()
                and path.stat().st_size == pin["bytes"] and file_sha(path) == pin["sha256"], "Actual prerequisite custody differs")
        return json.loads(path.read_text())
    for name, pin in plan["native_qualification"].items():
        value = read_pin(pin)
        require(value.get("status") == "PASS" and value["identity"]["driver_manifest_sha256"] == plan["native_driver_manifest_sha256"]
                and value["identity"]["prototype_manifest_sha256"] == MODEL_SHA and value["state_donor"] is False
                and value["test_file_opened"] is False, "Existing qualified native source required")
        require_profile_receipt(value)
        terminal = read_pin(plan["native_qualification_terminals"][name])
        require(terminal["status"] == "COMPLETE" and terminal["child_exit_code"] == 0 and terminal["cap_violation"] is None
                and terminal["qualification_receipt"]["sha256"] == pin["sha256"], "Existing native physical completion required")
    pins = plan["native_bucket_prerequisites"]
    diagnostic, terminal, final, custody = (read_pin(pins[k]) for k in ("diagnostic", "terminal", "final", "supervisor_custody"))
    require(diagnostic["schema"] == "native-exact-CB-bucket-equivalence-resource-diagnostic-v1"
            and diagnostic["status"] == "COMPLETE_DIAGNOSTIC_ONLY" and diagnostic["equivalence_verdict"] == "PASS_FIXED_ORIGINAL_RULE"
            and diagnostic["source_manifest_sha256"] == plan["native_bucket_source_manifest_sha256"]
            and diagnostic["candidate_manifest_sha256"] == plan["conditional_manifest_sha256"]
            and diagnostic["core_manifest_sha256"] == plan["core_manifest_sha256"]
            and diagnostic["optimizer_updates"] == 0 and diagnostic["VALID_TEST_reads"] is False, "Exact completed native bucket equivalence required")
    require(terminal["status"] == "COMPLETE_DIAGNOSTIC_ONLY" and terminal["physical_session_closed"] is True
            and terminal["direct_child_reaped"] is True and terminal["physical_exit_code"] == 0 and terminal["stop"] is None
            and terminal["linked"]["status"] == "COLLECTED_COMPLETE_DIAGNOSTIC_ONLY"
            and terminal["linked"]["result_sha256"] == pins["diagnostic"]["sha256"]
            and terminal["linked"]["final_custody_sha256"] == pins["final"]["sha256"]
            and custody["terminal_sha256"] == pins["terminal"]["sha256"] and final["completed"] is True,
            "Exact owned native bucket completion/custody required")




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
