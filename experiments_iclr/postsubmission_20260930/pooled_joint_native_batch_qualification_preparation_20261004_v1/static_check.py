#!/usr/bin/env python3
"""Stdlib AST/compile/JSON/pin checks only; never imports project/Torch code."""
import ast
from hashlib import sha256
import json
from pathlib import Path


def run():
    here = Path(__file__).resolve().parent
    research = here.parent
    parsed = []
    for path in sorted(here.glob("*.py")):
        source = path.read_text()
        ast.parse(source, filename=str(path))
        compile(source, str(path), "exec")
        parsed.append(path.name)
    for path in sorted(here.glob("*.json")):
        json.loads(path.read_text())
    bindings = json.loads((here / "SOURCE_BINDINGS.json").read_text())
    verified = []
    def pin(path, row):
        assert path.stat().st_size == row.get("bytes", row.get("size")), str(path)
        assert sha256(path.read_bytes()).hexdigest() == row["sha256"], str(path)
    for row in bindings["files"]:
        path = research / row["relative_path"]
        pin(path, row); verified.append(row["relative_path"])
    payloads = 0
    for row in bindings["manifests"]:
        root = research / row["packet"]
        manifest_path = root / "MANIFEST.json"
        assert sha256(manifest_path.read_bytes()).hexdigest() == row["sha256"]
        for entry in json.loads(manifest_path.read_text())["files"]:
            pin(root / entry["path"], entry); payloads += 1
    return {"status": "PASS_STATIC_ONLY", "AST_and_compile_files": parsed,
            "source_pins_verified": len(verified), "dependency_manifests_verified": len(bindings["manifests"]),
            "dependency_manifest_payload_entries_verified": payloads, "project_or_Torch_imported": False,
            "native_or_GPU_executed": False, "dataset_arrays_opened": False,
            "old_checkpoints_or_science_outcomes_opened": False,
            "numerical_replay_or_full_batch_feasibility_qualified": False}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
