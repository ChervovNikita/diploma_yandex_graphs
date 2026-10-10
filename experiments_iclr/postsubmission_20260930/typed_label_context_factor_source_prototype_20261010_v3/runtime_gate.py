"""Inactive authority and pinned-module boundary; no numerical imports at import."""
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

from .caps import CLOSED

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PROVIDERS = ("torch", "numpy", "dgl", "torch_sparse", "sklearn")
CONDITIONS = ("shared_own_pair4", "shared_local_mul4", "shared_local_add4",
              "shared_global_mul4", "single_local_mul1", "untied_local_mul4")
SEEDS = (
    {"role_seed": 1, "optimizer_seed": 1, "context_seed": 104729, "generator_seed": 190001,
     "member_rng_seeds": [210001, 210002, 210003, 210004]},
    {"role_seed": 2, "optimizer_seed": 2, "context_seed": 130363, "generator_seed": 190002,
     "member_rng_seeds": [220001, 220002, 220003, 220004]},
    {"role_seed": 3, "optimizer_seed": 3, "context_seed": 155921, "generator_seed": 190003,
     "member_rng_seeds": [230001, 230002, 230003, 230004]},
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def inside(path, exists=True):
    raw = Path(path)
    require(raw.is_absolute() and not raw.is_symlink(), "Explicit non-alias project path")
    value = raw.resolve(strict=exists)
    require(value.is_relative_to(PROJECT), "No reads or outputs outside the project research repository")
    return value


def binding(path):
    path = inside(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def bound(row):
    require(isinstance(row, dict) and set(row) == {"path", "bytes", "sha256"}, "Complete bound project file")
    path = inside(row["path"])
    require(binding(path) == row, "Exact root-bound project file bytes")
    return path


def source_gate():
    seal, manifest = read(HERE / "SEAL.json"), read(HERE / "MANIFEST.json")
    require(seal["source_only"] is True and seal["execution_enabled"] is False
            and sha(HERE / "MANIFEST.json") == seal["manifest_sha256"], "Immutable inactive source seal")
    for row in manifest["files"]:
        path = (HERE / row["path"]).resolve(strict=True)
        require(path.is_relative_to(HERE) and path.stat().st_size == row["bytes"]
                and sha(path) == row["sha256"], "Unchanged successor source")
    sources = read(HERE / "SOURCE_BINDINGS.json")
    for row in sources["files"]:
        path = (PROJECT / row["path"]).resolve(strict=True)
        require(path.is_relative_to(PROJECT) and path.stat().st_size == row["bytes"]
                and sha(path) == row["sha256"], "Unchanged pinned dependency source")
    return sources


def release_gate(release_path, caps=CLOSED):
    """Read bound JSON authority only, before any provider/model/data import."""
    caps.require("source_bound", "model", "data", "runtime")
    sources = source_gate()
    release_path = inside(release_path)
    release = read(release_path)
    action = release["action"]
    require(release["enabled"] is True and release["root_source_review_approved"] is True
            and release["root_review"]["sha256"] == caps.root_review_sha256
            and release["source_seal_sha256"] == sha(HERE / "SEAL.json")
            and release["protocol_sha256"] == sha(HERE / "PROTOCOL.json"), "Explicit root-reviewed successor authority")
    bound(release["root_review"])
    require(action in ("qualify_complete_native_context_pair", "fit_complete_native_context_comparison"),
            "Separate full-input qualification or scientific comparison action")
    scientific = action == "fit_complete_native_context_comparison"
    if scientific:
        caps.require("scientific")
    require(release["scientific_execution_approved"] is scientific
            and release["data_scope_approved"] is True and release["provider_runtime_approved"] is True
            and release["conditions"] == list(CONDITIONS)
            and release["seed_specs"] == list(SEEDS if scientific else SEEDS[:1]),
            "Complete predeclared six-condition/three-role recipe; no chosen favorable seeds")
    require(release["capabilities"] == {name: getattr(caps, name) for name in
            ("source_bound", "model", "data", "runtime", "scientific")}, "Capability/release agreement")
    require(set(release["development_files"]) == {"node.dat", "link.dat", "label.dat"},
            "Only exact development input names; no heldout member interface")
    input_root, output = inside(release["input_root"]), inside(release["output_directory"], exists=False)
    require(input_root.is_dir() and not output.exists() and output != PROJECT,
            "Exact development directory and fresh project output")
    require(set(release["roles"]) == {"1", "2", "3"}, "Three separately bound frozen native roles")
    roles = {}
    for seed, row in release["roles"].items():
        path = bound(row)
        require(path.name == "seed" + seed + ".json", "Named frozen role descriptor only")
        role = read(path)
        require(role["seed"] == int(seed) and role["input_files"] == release["development_files"], "Exact role/input identity")
        roles[int(seed)] = role
    schema = read(bound(release["schema_receipt"]))
    require(schema["status"] == "complete" and schema["schema_qualified"] is True
            and schema["all_ten_roles_qualified"] is True
            and schema["TEST_member_open_stat_hash_or_parse"] is False
            and schema["TEST_membership_known"] is False
            and schema["actual_input_schema"]["input_files"] == release["development_files"], "Full schema custody without heldout access")
    qualified = read(bound(release["native_qualification_receipt"]))
    require(qualified["complete"] is True and qualified["native_backbone_numerically_qualified"] is True
            and qualified["source_seal_sha256"] == sha(PROJECT / sources["runtime_sources"]["native_seal"])
            and qualified["input_files"] == release["development_files"]
            and qualified["schema_receipt"]["sha256"] == release["schema_receipt"]["sha256"]
            and qualified["runtime"]["versions"] == release["expected_runtime_versions"]
            and qualified["runtime"]["math_flags"] == release["expected_math_flags"]
            and qualified["runtime"]["cuda_device_uuid"] == release["cuda_device_uuid"]
            and qualified["runtime"]["python_executable"] == release["python_executable"]
            and qualified["runtime"]["environment"] == release["runtime_environment"], "Native full-input numerical qualification is mandatory")
    require(set(release["expected_runtime_versions"]) == set(PROVIDERS)
            and all(isinstance(v, str) and v for v in release["expected_runtime_versions"].values()), "Exact prequalified provider versions")
    require(release["runtime_environment"]["CUDA_VISIBLE_DEVICES"] == release["cuda_device_uuid"]
            and release["cuda_device_uuid"].startswith("GPU-") and release["device"] == "cuda:0"
            and set(release["runtime_environment"]) == {"CUDA_VISIBLE_DEVICES", "PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS"}
            and all(os.environ.get(k) == v for k, v in release["runtime_environment"].items()), "Root-owned one-device environment before provider imports")
    require(str(Path(sys.executable).absolute()) == release["python_executable"]
            and platform.python_version() == release["python_version"]
            and Path(sys.prefix).resolve().is_relative_to(PROJECT), "Exact project-local runtime environment")
    require(release["resource_budget"]["host_RSS_bytes"] >= 32 * 1024**3
            and release["resource_budget"]["device_bytes"] >= 24 * 1024**3
            and release["resource_budget"]["root_owns_external_resource_monitor"] is True, "Complete native resource budget; no shrink fallback")
    if scientific:
        candidate = read(bound(release["candidate_qualification_receipt"]))
        require(candidate["status"] == "complete" and candidate["complete"] is True
                and candidate["mode"] == "qualification" and candidate["qualification_passed"] is True
                and candidate["source_seal_sha256"] == release["source_seal_sha256"]
                and candidate["protocol_sha256"] == release["protocol_sha256"]
                and candidate["roles"] == release["roles"] and candidate["input_files"] == release["development_files"]
                and candidate["conditions"] == release["conditions"]
                and candidate["runtime"]["versions"] == release["expected_runtime_versions"]
                and candidate["runtime"]["math_flags"] == release["expected_math_flags"]
                and candidate["runtime"]["cuda_device_uuid"] == release["cuda_device_uuid"]
                and candidate["runtime"]["python_executable"] == release["python_executable"]
                and candidate["runtime"]["environment"] == release["runtime_environment"]
                and len(candidate["runs"]) == len(CONDITIONS)
                and all(r["complete"] and r["qualification_passed"] for r in candidate["runs"]),
                "All six complete native paired paths separately qualified before scientific fits")
        reference_policy = read(bound(release["independent_reference_protocol"]))
        require(reference_policy["candidate_protocol_sha256"] == release["protocol_sha256"]
                and reference_policy["root_source_review_approved"] is True
                and reference_policy["ordinary_independent_four_body_required"] is True
                and reference_policy["same_information_independently_selected_four_body_required"] is True
                and reference_policy["completed_pilot_scores_are_not_matched_references"] is True,
                "Root binds the mandatory individually selected ordinary and same-information references")
    return release, roles, sources, binding(release_path)


def load_module(name, path):
    path = inside(path)
    if name in sys.modules:
        require(sha(sys.modules[name].__file__) == sha(path), "Existing module must be the exact owned pinned bytes")
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def runtime(release, sources, caps=CLOSED):
    caps.require("source_bound", "model", "data", "runtime")
    providers = {name: importlib.import_module(name) for name in PROVIDERS}
    versions = {name: str(value.__version__) for name, value in providers.items()}
    require(versions == release["expected_runtime_versions"], "Exact qualified providers")
    torch = providers["torch"]
    require(torch.get_default_dtype() == torch.float32 and torch.cuda.is_available()
            and torch.cuda.device_count() == 1, "Native FP32 default and one visible CUDA device")
    torch.cuda.init()
    device = torch.device(release["device"])
    torch.cuda.set_device(device)
    prop = torch.cuda.get_device_properties(device)
    require(prop.total_memory >= release["resource_budget"]["device_bytes"], "Full native device budget")
    flags = {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
             "cudnn_deterministic": torch.backends.cudnn.deterministic,
             "cudnn_benchmark": torch.backends.cudnn.benchmark,
             "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
             "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
             "default_dtype": str(torch.get_default_dtype())}
    require(flags == release["expected_math_flags"], "Adopt exact native math flags without rewriting them")
    paths = sources["runtime_sources"]
    modules = {}
    for key in ("state", "model", "native", "engine", "loader"):
        name = "state_helpers" if key == "state" else "_typed_context_pinned_" + key
        modules[key] = load_module(name, PROJECT / paths[key])
    modules["loader"].source_gate()
    rt = dict(providers, device=device, model_module=modules["model"], model_class=modules["model"].SeHGNN,
              native=modules["native"], SparseTensor=providers["torch_sparse"].SparseTensor,
              remove_diag=providers["torch_sparse"].remove_diag,
              protocol=read(PROJECT / paths["native_protocol"]))
    record = {"versions": versions, "math_flags": flags, "cuda_device_uuid": release["cuda_device_uuid"],
              "device": str(device), "device_name": prop.name, "device_total_bytes": prop.total_memory,
              "python_version": platform.python_version(), "python_executable": str(Path(sys.executable).absolute()),
              "python_prefix": str(Path(sys.prefix).resolve()), "environment": dict(release["runtime_environment"]),
              "modules": {key: binding(value.__file__) for key, value in modules.items()}}
    return rt, modules["engine"], modules["loader"], record
