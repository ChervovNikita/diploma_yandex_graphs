# SPDX-License-Identifier: Apache-2.0
"""Verify bytes/JSON/Python AST only; do not import a packet/numerical module."""
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

NUMERICAL = {"torch", "numpy", "sklearn", "torch_geometric", "torch_sparse", "torch_scatter", "torch_householder"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    packet = Path(__file__).resolve().parent
    seal = json.loads((packet / "SOURCE_SEAL.json").read_text())
    for relative, expected in seal["sha256"].items():
        assert sha256(packet / relative) == expected, relative
    native = packet.parent / "private_sheaf_native_source_design_20261009_v1"
    retrievals = json.loads((native / "SOURCE_RETRIEVAL.json").read_text())
    retained = [r for r in retrievals if r.get("payload_retained")]
    for item in retained:
        path = native / "source_custody" / "nsd" / item["path"]
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
        assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == item["git_blob_sha1"]
    syntax = []
    for path in sorted(packet.glob("*.py")):
        source = path.read_text()
        tree = ast.parse(source, filename=str(path), feature_version=(3, 9))
        compile(source, str(path), "exec", dont_inherit=True)
        top_imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_imports.extend(item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                top_imports.append(node.module)
        assert not any(name.split(".")[0] in NUMERICAL for name in top_imports)
        syntax.append({"file": path.name, "syntax": "Python3.9 AST/compile pass", "top_imports": top_imports})
    for path in packet.glob("*.json"):
        json.loads(path.read_text())
    protocol = json.loads((packet / "RUNNER_PROTOCOL.json").read_text())
    prior = json.loads((native / "PROTOCOL.json").read_text())
    assert protocol["seeds"] == prior["competence_screen"]["seeds"]
    assert protocol["optimizer"] == prior["competence_screen"]["optimizer"]
    assert protocol["configs"] == [{"id": c["id"], "native_args": c["native_args"]}
                                   for c in prior["competence_screen"]["configs"]]
    assert not json.loads((packet / "RELEASE_TEMPLATE_DISABLED.json").read_text())["enabled"]
    result = {"UTC": datetime.now(timezone.utc).isoformat(), "status": "static_pass_only",
              "sealed_paths": len(seal["sha256"]), "retained_native_source_files_unchanged": len(retained),
              "syntax_checks": syntax, "fixed_presets_seeds_optimizer_match_prior_protocol": True,
              "public_release_template_disabled": True, "exposure_correction_saved_separately": True,
              "packet_or_numerical_modules_imported": False, "runner_or_exporter_executed": False,
              "graphs_labels_checkpoints_accessed": False, "servers_contacted": False,
              "numerical_and_placement_qualification": "not_run", "competence": "unestablished",
              "source_seal_sha256": sha256(packet / "SOURCE_SEAL.json")}
    (packet / "VERIFICATION.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "python_files": len(syntax),
                      "native_files_unchanged": len(retained), "release": "disabled"}))


if __name__ == "__main__":
    main()
