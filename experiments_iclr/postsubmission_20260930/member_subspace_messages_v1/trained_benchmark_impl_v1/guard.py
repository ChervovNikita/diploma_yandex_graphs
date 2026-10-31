"""Separate sealed caller admission; inherited scientific policy is immutable."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
SCIENTIFIC = "shared_message_predictive_v3"
SCIENTIFIC_SHA = "e4d1b4b7d96ae07e7588a854cb8a0d70e9af6612ac72d1df583c3c637c376c9b"


def confined(path):
    path = Path(path).absolute()
    if not path.is_relative_to(PHASE):
        raise ValueError("Outside phase")
    cursor = PHASE
    for part in path.relative_to(PHASE).parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("Symlink evidence/source refused")
    resolved = path.resolve()
    if not resolved.is_relative_to(PHASE):
        raise ValueError("Resolved path outside phase")
    return resolved


def sha(path):
    return hashlib.sha256(confined(path).read_bytes()).hexdigest()


def ref(path):
    path = confined(path)
    return {"path": str(path.relative_to(PHASE)), "sha256": sha(path)}


def checked(reference):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("Exact path/hash reference required")
    path = confined(PHASE / reference["path"])
    if str(path.relative_to(PHASE)) != reference["path"] or sha(path) != reference["sha256"]:
        raise ValueError("Changed/noncanonical reference")
    return path


def verify(manifest_sha):
    manifest_path = HERE / "SOURCE_MANIFEST_v1.json"
    if sha(manifest_path) != manifest_sha:
        raise ValueError("Companion source manifest differs from approved request")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "gnnm-trained-inference-source-manifest-v1" or manifest.get("execution_authorized") is not False:
        raise ValueError("Exact source-only companion seal required")
    seen = set()
    for row in manifest["files"]:
        path = confined(HERE / row["relative_path"])
        if path.parent != HERE or row["relative_path"] in seen or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("Companion sealed file changed")
        seen.add(row["relative_path"])
    for reference in manifest["inherited_and_protocol_refs"].values():
        checked(reference)
    bindings_path = HERE / "SOURCE_BINDINGS.json"
    if sha(bindings_path) != manifest["source_bindings_sha256"]:
        raise ValueError("Companion closure link changed")
    bindings = json.loads(bindings_path.read_text())
    if bindings.get("schema") != "gnnm-trained-inference-separate-source-bindings-v1" or bindings.get("execution_authorized") is not False:
        raise ValueError("Separate companion closure required")
    for relative, wanted in bindings["sha256"].items():
        checked({"path": relative, "sha256": wanted})
    scientific_manifest = PHASE / SCIENTIFIC / "SOURCE_MANIFEST_v1.json"
    if sha(scientific_manifest) != SCIENTIFIC_SHA:
        raise ValueError("Frozen scientific generation changed")
    old_manifest = json.loads(scientific_manifest.read_text())
    row = next(r for r in old_manifest["files"] if r["relative_path"] == "import_guard.py")
    source = PHASE / SCIENTIFIC / "import_guard.py"
    if sha(source) != row["sha256"]:
        raise ValueError("Inherited guard changed before bootstrap")
    name = "_trained_benchmark_scientific_guard_v3"
    spec = importlib.util.spec_from_file_location(name, source)
    inherited = importlib.util.module_from_spec(spec)
    sys.modules[name] = inherited
    exec(compile(source.read_bytes(), str(source), "exec", dont_inherit=True), inherited.__dict__)
    base, tree = inherited.verify_packet(SCIENTIFIC_SHA)
    policy = json.loads((HERE / "CALLER_IMPORT_POLICY_v1.json").read_text())
    expected = copy.deepcopy(base)
    expected.pop("generated_source_policy")
    own = {str((HERE / n).relative_to(PHASE)): sha(HERE / n) for n in seen if n.endswith(".py")}
    expected["importable_sha256"].update(own)
    expected.update(caller_policy_schema="gnnm-trained-inference-augmented-caller-policy-v1",
                    caller_sources=own, inherited_scientific_manifest=ref(scientific_manifest))
    if policy != expected:
        raise ValueError("Caller policy must be the exact inherited policy plus sealed caller sources")
    caller_tree = inherited.inspect_tree(HERE, list(own), policy["package_directories"])
    policy["generated_source_policy"] = base["generated_source_policy"]
    return inherited, policy, {**tree, "explicit_sealed_caller_files": caller_tree}


def install(manifest_sha):
    inherited, policy, tree = verify(manifest_sha)
    inherited.installed_path_only()
    inherited.verify_generator_installation(policy)
    inherited.imported_modules(policy)
    # Own caller files are explicitly loaded by their exact source paths.
    # The PhaseFinder continues to expose only the original scientific roots.
    sys.meta_path.insert(0, inherited.PhaseFinder(policy))
    return inherited, policy, tree


def receipt(state, *, require_generated=True):
    inherited, policy, tree = state
    rows = inherited.imported_modules(policy)
    inherited.verify_imported_receipts(rows, policy, require_generated=require_generated)
    return dict(caller_manifest=ref(HERE / "SOURCE_MANIFEST_v1.json"),
                caller_policy=ref(HERE / "CALLER_IMPORT_POLICY_v1.json"),
                inherited_scientific_manifest=policy["inherited_scientific_manifest"],
                runtime_import_tree=tree, imported_phase_modules=rows,
                generated_module_policy_sha256=policy["generated_module_policy_sha256"],
                installed_generated_code_sources=inherited.verify_generator_installation(policy))


def audit_receipt(saved, state):
    inherited, policy, tree = state
    if (saved.get("caller_manifest") != ref(HERE / "SOURCE_MANIFEST_v1.json")
            or saved.get("caller_policy") != ref(HERE / "CALLER_IMPORT_POLICY_v1.json")
            or saved.get("inherited_scientific_manifest") != policy["inherited_scientific_manifest"]
            or saved.get("runtime_import_tree") != tree
            or saved.get("generated_module_policy_sha256") != policy["generated_module_policy_sha256"]
            or saved.get("installed_generated_code_sources") != inherited.verify_generator_installation(policy)):
        raise ValueError("Saved companion import/source receipt differs")
    inherited.verify_imported_receipts(saved["imported_phase_modules"], policy, require_generated=True)
    own_required = {"__main__": str((HERE / "runner.py").relative_to(PHASE)),
                    "_trained_benchmark_caller_guard_v1": str((HERE / "guard.py").relative_to(PHASE))}
    source_by_module = {row["module"]: row["relative_to_phase"] for row in saved["imported_phase_modules"]}
    if any(source_by_module.get(name) != path for name, path in own_required.items()):
        raise ValueError("Actual sealed worker and caller guard source receipts required")
