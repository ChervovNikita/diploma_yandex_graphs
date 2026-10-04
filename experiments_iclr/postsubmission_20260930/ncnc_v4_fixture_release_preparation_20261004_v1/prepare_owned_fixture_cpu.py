"""Authenticate the complete existing fabricated fixture before CPU custody rebinding.

CONFIG is supplied by the separately pinned reviewed transport command. No fit,
scorer, model construction, study input, CUDA initialization or retry is used.
"""
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from hashlib import sha256

REPO = Path(CONFIG["repo"])
PHASE = REPO / "experiments_iclr/postsubmission_20260930"
OUT = PHASE / "ncnc_v4_fixture_release_preparation_20261004_v1"
ORIGINAL = Path(CONFIG["original_fixture"]["root"])
SOURCE = PHASE / "ncnc_selected_checkpoint_metric_audit_v4_source_20261004"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pin(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def verify(value):
    path = Path(value["path"])
    require(path.is_absolute() and path.resolve() == path and path.is_file(), "Noncanonical custody path")
    require(("bytes" not in value or path.stat().st_size == value["bytes"]) and sha(path) == value["sha256"], "Byte custody changed: " + str(path))
    return path


def write(name, value):
    with (OUT / name).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def fixture_inventory():
    rows = []
    for path in sorted(ORIGINAL.rglob("*")):
        require(path.resolve() == path, "Borrowed fixture path")
        if path.is_file():
            rows.append({**pin(path), "path": str(path.relative_to(ORIGINAL))})
    require(rows == CONFIG["original_fixture"]["files"], "Complete original fixture differs from root inventory")
    return rows


require(Path.cwd().resolve() == REPO and not OUT.exists(), "Wrong repository or preparation already exists")
require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only custody preparation required")
source_packets = []
for spec in CONFIG["packets"]:
    root = PHASE / spec["root"]
    manifest_path = root / "MANIFEST.json"
    require(sha(manifest_path) == spec["manifest_sha256"], "Manifest changed")
    manifest = json.loads(manifest_path.read_text())
    require(manifest["files"] and len({r["path"] for r in manifest["files"]}) == len(manifest["files"]), "Empty/duplicate packet")
    for row in manifest["files"]:
        path = root / row["path"]
        require(path.resolve() == path and path.is_relative_to(root), "Packet path escaped")
        verify({"path": str(path), "bytes": row.get("bytes", row.get("size")), "sha256": row["sha256"]})
    if spec.get("seal"):
        verify(spec["seal"])
    source_packets.append({"root": str(root), "manifest": pin(manifest_path), "payload_count": len(manifest["files"])})
runtime_path = verify(CONFIG["candidate"]["runtime_authority"])
runtime = json.loads(runtime_path.read_text())
require(Path(sys.executable).resolve() == Path(runtime["interpreter_path"]).resolve(), "Wrong preparation interpreter")
verify({"path": str(Path(sys.executable).resolve()), "sha256": runtime["interpreter_sha256"]})
require({name: importlib.metadata.version(name) for name in runtime["distribution_versions"]} == runtime["distribution_versions"], "Runtime distributions changed")
runtime_pins = [*runtime["runtime_source_pins"], *runtime["runtime_binary_files"], runtime["negative_sampler"], CONFIG["evaluator"]]
for value in runtime_pins:
    verify(value)
review_path = verify(CONFIG["candidate"]["independent_source_review"])
review = json.loads(review_path.read_text())
require(review["status"] == "PASS" and review["sidecar_manifest_sha256"] == CONFIG["candidate"]["sidecar_manifest_sha256"] and review["identity"] == CONFIG["candidate"]["original_identity"], "Exact fresh source review changed")
original_rows = fixture_inventory()
require(len(original_rows) == 172 and sum(r["bytes"] for r in original_rows) == 46954391, "Original fixture denominator differs")
OUT.mkdir(mode=0o700)
write("ORIGINAL_FIXTURE_BYTE_AUTHENTICATION.json", {"schema": "ncnc-v4-original-fabricated-fixture-byte-authentication-v1", "UTC": datetime.now(timezone.utc).isoformat(), "status": "PASS_BEFORE_ANY_DESERIALIZATION", "fixture_root": str(ORIGINAL), "identity": CONFIG["original_fixture"]["identity"], "files": original_rows, "file_count": len(original_rows), "bytes": sum(r["bytes"] for r in original_rows), "payload_deserialization_before_authentication": False, "original_tree_modified": False})
write("SOURCE_RUNTIME_REVIEW_AUTHENTICATION.json", {"status": "PASS", "UTC": datetime.now(timezone.utc).isoformat(), "source_packets": source_packets, "runtime_authority": pin(runtime_path), "runtime_pins": runtime_pins, "distribution_versions": runtime["distribution_versions"], "interpreter": pin(Path(sys.executable).resolve()), "review": pin(review_path), "study_inputs_accessed": False})

# Authenticated source definitions only; these modules have no numerical import
# or GPU initialization at module scope. The helper body was reviewed locally.
sys.path.insert(0, str(SOURCE))
from replay_gate import family_gate, descriptor, verify_manifest
from replay_numeric import trusted_load
from replay_synthetic_cases import clone_fixture
driver = Path(CONFIG["candidate"]["driver_root"])
sys.path.insert(0, str(driver))
api = {name: importlib.import_module(name) for name in ("pilot_state", "pilot_data")}
for name, module in api.items():
    require(Path(module.__file__).resolve() == driver / (name + ".py"), "Driver module shadowed")
require(not any(name in sys.modules for name in ("pilot_model", "pilot_evaluate")), "Numerical driver imported")
lock_path = ORIGINAL / "lock/FAMILY_LOCK.json"
lock = json.loads(lock_path.read_text())
identity = CONFIG["original_fixture"]["identity"]
require(lock["identity"] == identity, "Original fabricated identity differs")
def local(path):
    return {**pin(path), "path": path.name}
unit_pins = []
for binding in lock["inputs"]:
    unit, seed = binding["unit"], binding["base_seed"]
    root = ORIGINAL / (unit + "_" + str(seed))
    require(binding["output_directory"] == str(root), "Original unit path differs")
    unit_pins.append({"unit": unit, "base_seed": seed, "output_directory": str(root), "disposition": "COMPLETE", "terminal": local(root / "COMPLETE.json"), "physical_terminal": pin(ORIGINAL / (root.name + "_PHYSICAL.json")), "attempts": local(root / "ATTEMPTS.json"), "journal": local(root / "JOURNAL.json"), "closure": local(root / "FAMILY_CLOSURE.json")})
context = {"identity": identity, "release": {"family_lock": pin(lock_path), "unit_custody": unit_pins}, "output": OUT / "metadata_validation_only_not_execution"}
family_gate(context)
import torch
require(not torch.cuda.is_initialized(), "CUDA was initialized")
tensor_pin = pin(ORIGINAL / "FABRICATED_TENSORS.pt")
payload = trusted_load(torch, ORIGINAL, {**tensor_pin, "path": "FABRICATED_TENSORS.pt"})
require(payload["schema"] == "ncnc-replay-owned-fabricated-tensors-v1" and payload["fabricated_inputs_only"] is True and set(payload["tensors"]) == {"x", "pairs", "raw_edge_index", "valid_positive", "valid_negative"}, "Fabricated tensor contract differs")
def require_cpu(value):
    if torch.is_tensor(value):
        require(value.device.type == "cpu", "CUDA tensor entered preparation")
    elif isinstance(value, dict):
        for item in value.values():
            require_cpu(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            require_cpu(item)
require_cpu(payload)
tensor_digests = {k: api["pilot_data"].tensor_sha(v) for k, v in payload["tensors"].items()}
context.update(synthetic_tensor_custody=tensor_pin, synthetic_tensor_digests=tensor_digests)
owned_root = OUT / "owned_fabricated_reference"
owned = clone_fixture({"fixture_root": ORIGINAL, "context": context}, owned_root, api, torch)
require(owned["identity"] == {**identity, "family_lock_output_directory": str(owned_root / "lock")}, "Only owned identity path may change")
family_gate(owned)
value_rows = []
for row in original_rows:
    if not row["path"].endswith(".pt"):
        continue
    original_path = ORIGINAL / row["path"]
    owned_path = owned_root / row["path"]
    before = trusted_load(torch, original_path.parent, {**row, "path": original_path.name})
    after = trusted_load(torch, owned_path.parent, local(owned_path))
    require_cpu(before); require_cpu(after)
    if row["path"] != "FABRICATED_TENSORS.pt":
        require(before["identity"] == identity and after["identity"] == owned["identity"], "Payload identity changed beyond own path")
        before = {k: v for k, v in before.items() if k != "identity"}
        after = {k: v for k, v in after.items() if k != "identity"}
    before_sha, after_sha = api["pilot_state"].state_digest(before), api["pilot_state"].state_digest(after)
    require(before_sha == after_sha, "Model/Adam/RNG/flags/tensor/metadata values changed")
    value_rows.append({"path": row["path"], "original_values_sha256": before_sha, "owned_values_sha256": after_sha, "identity_only_excluded": row["path"] != "FABRICATED_TENSORS.pt", "all_tensors_on_CPU": True})
require(len(value_rows) == 46 and sha(owned_root / "FABRICATED_TENSORS.pt") == tensor_pin["sha256"], "Fabricated payload denominator/bytes changed")
require(fixture_inventory() == original_rows and not torch.cuda.is_initialized(), "Original changed or CUDA initialized")
owned_rows = [{**pin(path), "path": str(path.relative_to(owned_root))} for path in sorted(owned_root.rglob("*")) if path.is_file()]
require(len(owned_rows) == len(original_rows), "Owned file denominator differs")
with (owned_root / "MANIFEST.json").open("x") as stream:
    json.dump({"schema": "ncnc-owned-existing-fabricated-full25-fixture-manifest-v1", "fabricated_inputs_only": True, "new_training_updates": 0, "files": owned_rows}, stream, indent=2, allow_nan=False)
    stream.write("\n")
manifest_sha = sha(owned_root / "MANIFEST.json")
verify_manifest(owned_root, manifest_sha)
fixture_spec = {"schema": "ncnc-owned-existing-fabricated-full25-fixture-descriptor-v3", "fabricated_inputs_only": True, "new_training_updates": 0, "fixture_root": str(owned_root), "fixture_manifest_sha256": manifest_sha, "identity": owned["identity"], "family_lock": owned["release"]["family_lock"], "unit_custody": owned["release"]["unit_custody"], "tensor_custody": owned["synthetic_tensor_custody"], "tensor_digests": tensor_digests}
write("EXISTING_FABRICATED_FIXTURE_DESCRIPTOR.json", fixture_spec)
write("CPU_OWNED_COPY_VALUE_EQUIVALENCE.json", {"status": "PASS", "CPU_payloads_compared": len(value_rows), "payloads": value_rows, "model_Adam_RNG_flags_and_nonidentity_values_identical": True, "original_fixture_file_count_after": len(fixture_inventory()), "original_tree_modified": False, "CUDA_initialized": torch.cuda.is_initialized(), "new_training_updates": 0, "scorer_calls": 0, "study_inputs_accessed": False, "tensor_digests": tensor_digests, "owned_family_metadata_gate": "PASS"})
candidate = CONFIG["candidate"]
candidate["existing_fabricated_fixture"] = pin(OUT / "EXISTING_FABRICATED_FIXTURE_DESCRIPTOR.json")
candidate["candidate_invocation_not_authorized"]["output_directory"] = str(OUT / "qualification/run01")
candidate["unresolved_before_root_enabled_release"] = ["root review of authenticated owned fixture/CPU equivalence and supervisor", "separate enabled root synthetic-only release", "dispatch recheck physical GPU0 >=16384MiB and MemAvailable >=17179869184 bytes", "fresh output/run receipts admission"]
write("ROOT_SYNTHETIC_RELEASE_DISABLED_CANDIDATE.json", candidate)
write("PREPARATION_RESULT.json", {"status": "PASS_AUTHENTICATED_OWNED_FIXTURE_READY_FOR_ROOT_REVIEW", "original_fixture": {"root": str(ORIGINAL), "files": 172, "bytes": 46954391, "unchanged": True}, "fixture_descriptor": pin(OUT / "EXISTING_FABRICATED_FIXTURE_DESCRIPTOR.json"), "owned_fixture_manifest": pin(owned_root / "MANIFEST.json"), "disabled_release_candidate": pin(OUT / "ROOT_SYNTHETIC_RELEASE_DISABLED_CANDIDATE.json"), "CPU_equivalence": pin(OUT / "CPU_OWNED_COPY_VALUE_EQUIVALENCE.json"), "qualification_started": False, "execution_enabled": False})
print(json.dumps({name: json.loads((OUT / name).read_text()) for name in ("PREPARATION_RESULT.json", "EXISTING_FABRICATED_FIXTURE_DESCRIPTOR.json", "ROOT_SYNTHETIC_RELEASE_DISABLED_CANDIDATE.json", "CPU_OWNED_COPY_VALUE_EQUIVALENCE.json", "SOURCE_RUNTIME_REVIEW_AUTHENTICATION.json")}, allow_nan=False))
