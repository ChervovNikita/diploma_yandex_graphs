"""Stdlib admission and custody for the bounded engineering candidate."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import importlib.util
import json
import os
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ARMS = ("P0", "J_P", "F_P", "C_mu")
STAGES = ("native", "batch")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def file_sha(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def read_pin(path, pin):
    path = Path(path).resolve()
    require(path.stat().st_size == pin["bytes"] and file_sha(path) == pin["sha256"],
            "Custody mismatch: " + str(path))
    return path


def bound_json(path, pin):
    return json.loads(read_pin(path, pin).read_text())


def verify_manifest(root, expected):
    root = Path(root).resolve()
    require(file_sha(root / "MANIFEST.json") == expected, "Manifest identity differs: " + str(root))
    m = json.loads((root / "MANIFEST.json").read_text())
    for row in m["files"]:
        p = (root / row["path"]).resolve()
        require(p.is_relative_to(root), "Manifest path escaped")
        read_pin(p, dict(row, bytes=row.get("bytes", row.get("size"))))
    return m


def load_module(name, path):
    path = Path(path).resolve()
    prior = sys.modules.get(name)
    if prior is not None:
        require(Path(prior.__file__).resolve() == path, "Module shadowed: " + name)
        return prior
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def source_custody(research):
    research = Path(research).resolve()
    bindings = json.loads((HERE / "SOURCE_BINDINGS.json").read_text())
    for row in bindings["manifests"]:
        p = (research / row["packet"]).resolve()
        require(p.is_relative_to(research), "Packet path escaped")
        verify_manifest(p, row["sha256"])
    paths = {}
    for row in bindings["files"]:
        p = (research / row["relative_path"]).resolve()
        require(p.is_relative_to(research), "Source path escaped")
        paths[row["key"]] = read_pin(p, row)
    # Validate actual prior execution, while preserving its CPU-only scope.
    driver = json.loads(paths["cpu_driver_receipt"].read_text())
    qa = json.loads(paths["cpu_qa_receipt"].read_text())
    require(driver["status"] == "PROCESS_COMPLETE" and driver["exit_code"] == 0
            and driver["timeout"] is False and driver["source_files_verified"] == 53
            and driver["GPU_use"] is False and driver["scientific_data_or_fit"] is False
            and qa["status"] == "PASS_FABRICATED_CPU_ONLY"
            and all(qa["results"][k]["status"] == "PASS" for k in ("legacy", "supplement", "vector")),
            "Actual bound CPU QA did not pass")
    return bindings, paths


def preflight(release_path, stage, output, *, worker=False):
    require("torch" not in sys.modules, "Fresh stdlib preflight must precede Torch")
    release_path = Path(release_path).resolve()
    release = json.loads(release_path.read_text())
    plan = json.loads((HERE / "PLAN.json").read_text())
    manifest_sha = file_sha(HERE / "MANIFEST.json")
    verify_manifest(HERE, manifest_sha)
    require(release.get("schema") == "pooled-native-batch-root-release-v1"
            and release.get("execution_enabled") is True and release.get("root_authorization_reference"),
            "Separate explicit root execution release required")
    require(release.get("preparation_manifest_sha256") == manifest_sha
            and release.get("resource_limits") == plan["resource_limits"], "Different source or bounds")
    require(release.get("fits_authorized") is False and release.get("state_donor") is False
            and release.get("VALID_or_TEST_authorized") is False, "Engineering scope required")
    require(stage in STAGES, "Only two engineering stages are supported")
    research = Path(release["research_root"]).resolve()
    output = Path(output).resolve()
    require(output.is_relative_to(research) and not output.is_relative_to(HERE)
            and not any((p / "MANIFEST.json").exists() for p in (output, *output.parents)),
            "Fresh project output outside every sealed packet required")
    invocation = {"stage": stage, "output_directory": str(output), "arms": list(ARMS),
                  "model_seed": 610041, "master_seed": 2026100401, "epoch": 1, "batch": 0}
    require(invocation in release.get("authorized_invocations", []), "Exact invocation not released")
    require(not output.exists() if not worker else (output / "SUPERVISOR_STARTED.json").is_file(),
            "Output is not this invocation's fresh worker directory")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == release["cuda_visible_devices"]
            and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", "Root-bound device/no bytecode required")
    require(release.get("runtime_profile") == "ordinary-authenticated-then-V5-deterministic",
            "Fixed native profile required")
    bindings, paths = source_custody(research)
    authority = json.loads(paths["data_authority"].read_text())
    runtime = json.loads(paths["runtime_authority"].read_text())
    require(sys.platform == "linux" and research == Path(runtime["research_root"]).resolve(),
            "Bound ordinary Linux research root required")
    require(release["cuda_visible_devices"] in runtime["allowed_cuda_visible_devices"], "Unadmitted GPU")
    require(Path(sys.executable).resolve() == Path(runtime["interpreter_path"]).resolve()
            and file_sha(sys.executable) == runtime["interpreter_sha256"], "Pinned interpreter required")
    identity = {"preparation_manifest_sha256": manifest_sha,
                "source_bindings_sha256": file_sha(HERE / "SOURCE_BINDINGS.json"),
                "frozen_spec_sha256": file_sha(paths["frozen_spec"]),
                "CPU_QA_receipt_sha256": file_sha(paths["cpu_qa_receipt"]),
                "data_authority_sha256": file_sha(paths["data_authority"]),
                "runtime_authority_sha256": file_sha(paths["runtime_authority"])}
    if stage == "batch":
        prior = bound_json(release["native_qualification"]["path"], release["native_qualification"])
        require(prior["schema"] == "pooled-native-batch-qualification-v1"
                and prior["identity"] == identity and prior["stage"] == "native"
                and prior["status"] == "PASS" and prior["state_donor"] is False
                and prior["data_files_opened"] == [] and prior["fits"] == 0,
                "Actual passed native qualification required")
        supervised = bound_json(release["native_supervisor_qualification"]["path"], release["native_supervisor_qualification"])
        require(supervised["schema"] == "pooled-native-batch-bounded-supervisor-v1"
                and supervised["status"] == "PASS" and supervised["identity"] == identity
                and supervised["child_exit_code"] == 0 and supervised["state_donor"] is False,
                "Passed native bounded supervisor receipt required")
        require(prior["runtime_identity"]["profile"]["CUDA_VISIBLE_DEVICES"] == release["cuda_visible_devices"],
                "Native qualification used another visible device")
        # Metadata/receipt authentication before Torch; arrays remain deferred.
        epoch = bound_json(Path(release["epoch_root"]) / "EPOCH.json", release["epoch_receipt"])
        require(epoch["master_seed"] == 2026100401 and epoch["epoch"] == 1
                and epoch["batch_size"] == 65536 and epoch["full_batches"] == 17
                and epoch["dropped_tail_records"] == 64940, "Fixed first complete epoch/batch required")
        require(release.get("TRAIN_raw_feature_read_authorized") is True, "Separate data admission required")
    else:
        require(release.get("TRAIN_raw_feature_read_authorized") is False,
                "Native stage is fabricated-only")
    return {"release": release, "release_path": release_path, "release_sha256": file_sha(release_path),
            "stage": stage, "output": output, "research": research, "paths": paths,
            "bindings": bindings, "plan": plan, "authority": authority, "runtime": runtime,
            "identity": identity, "invocation": invocation}


def assemble(context):
    """Called only inside the bounded admitted worker, after stdlib checks."""
    p = context["paths"]
    replay_common = load_module("replay_common", p["replay_common"])
    replay_runtime = load_module("replay_runtime", p["replay_runtime"])
    original = json.loads(p["provider_source_bindings"].read_text())
    provider_sources = {row["key"]: context["research"] / row["relative_path"] for row in original["sources"]}
    for row in original["sources"]:
        read_pin(provider_sources[row["key"]], row)
    rt = replay_runtime.runtime({"sources": provider_sources, "bindings": original,
                "release": context["release"], "provider_manifest_sha256": file_sha(p["provider_manifest"])})
    torch = rt["torch"]
    torch.cuda.reset_peak_memory_stats(0)
    mods = {}
    for name in ("prototype", "pattern_model", "pattern_objective", "pattern_teacher",
                 "count_density", "pooled_model", "count_model", "vector_density", "vector_head",
                 "ragged_density", "vector_count_model", "learning", "replay_provider"):
        mods[name] = load_module(name, p[name])
    reference = sys.modules["native_reference"]
    native, native_utils = reference.native_modules()
    mods.update(native=native, native_utils=native_utils,
                design=load_module("_pooled_qualification_design", p["design_spec"]))
    return rt, mods


def final_custody(context):
    require(file_sha(context["release_path"]) == context["release_sha256"], "Release changed during work")
    verify_manifest(HERE, context["identity"]["preparation_manifest_sha256"])
    source_custody(context["research"])
    runtime = context["runtime"]
    require(file_sha(runtime["interpreter_path"]) == runtime["interpreter_sha256"], "Interpreter bytes changed")
    for pin in runtime["runtime_source_pins"]:
        require(file_sha(pin["path"]) == pin["sha256"], "Runtime source bytes changed")
    for pin in runtime["runtime_binary_files"]:
        read_pin(pin["path"], pin)
    require(file_sha(runtime["negative_sampler"]["path"]) == runtime["negative_sampler"]["sha256"],
            "Native sampler bytes changed")
    for row in context["bindings"]["module_paths"]:
        module = sys.modules.get(row["module"])
        require(module is not None and Path(module.__file__).resolve() == context["paths"][row["key"]],
                "Imported source closure differs: " + row["module"])
