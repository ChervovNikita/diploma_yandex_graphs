# SPDX-License-Identifier: Apache-2.0
"""Source bytes, license and Python syntax only; never import model code."""
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def verify():
    packet = Path(__file__).resolve().parent
    native = packet / "source_custody" / "nsd"
    pins = json.loads((packet / "PINNED_CORE_SHA256.json").read_text())
    tree = json.loads((packet / "source_metadata" / "nsd_tree.json").read_text())
    tree_blobs = {item["path"]: item["sha"] for item in tree["tree"]
                  if item["type"] == "blob"}
    retrievals = json.loads((packet / "SOURCE_RETRIEVAL.json").read_text())
    retained = [item for item in retrievals if item.get("payload_retained")]
    source_checks = []
    for item in retained:
        if item["repo"] != "twitter-research/neural-sheaf-diffusion":
            raise AssertionError("Unexpected retained source repository")
        path = native / item["path"]
        payload = path.read_bytes()
        sha256 = hashlib.sha256(payload).hexdigest()
        blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        assert sha256 == item["sha256"], path
        assert blob == item["git_blob_sha1"] == tree_blobs[item["path"]], path
        source_checks.append(item["path"])
    for relative, digest in pins.items():
        assert hashlib.sha256((native / relative).read_bytes()).hexdigest() == digest
    license_text = (native / "LICENSE").read_text()
    assert "Apache License" in license_text and "Version 2.0" in license_text
    assert not (packet / "source_custody" / "dnsd").exists()
    syntax_paths = [packet / "adapter.py", packet / "static_verify.py"]
    syntax_paths += sorted(native.rglob("*.py"))
    for path in syntax_paths:
        source = path.read_text()
        ast.parse(source, filename=str(path), feature_version=(3, 9))
        compile(source, str(path), "exec", dont_inherit=True)
    result = {
        "UTC": datetime.now(timezone.utc).isoformat(),
        "status": "static_pass_only",
        "retained_source_files_sha256_and_git_blob_pass": source_checks,
        "pinned_core_files": len(pins),
        "apache_2_0_license_verified": True,
        "dnsd_source_payload_absent": True,
        "python_3_9_syntax_and_compile_only_pass": [path.relative_to(packet).as_posix()
                                                   for path in syntax_paths],
        "native_or_adapter_imported": False,
        "tensor_or_model_operations_executed": False,
        "dataset_labels_checkpoints_or_training_accessed": False,
        "numerical_qualification": "not_run",
        "competence": "unestablished",
        "launch_status": "inactive"
    }
    (packet / "VERIFICATION.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "source_files": len(source_checks),
                      "python_files": len(syntax_paths), "launch_status": "inactive"}))


if __name__ == "__main__":
    verify()
