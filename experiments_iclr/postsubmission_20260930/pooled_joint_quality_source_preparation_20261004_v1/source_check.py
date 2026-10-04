"""Stdlib-only static audit. Does not import scientific implementation.

This command is source parsing/compilation/hash verification, not numerical
execution, admission, qualification, resource testing or predictive evidence.
"""
import ast
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path


def check():
    root = Path(__file__).resolve().parent
    sources, parsed_json = [], []
    for path in sorted(root.rglob("*.py")):
        source = path.read_text()
        ast.parse(source, filename=str(path))
        compile(source, str(path), "exec")  # compile only; no code evaluation
        sources.append(str(path.relative_to(root)))
    for path in sorted(root.rglob("*.json")):
        json.loads(path.read_text())
        parsed_json.append(str(path.relative_to(root)))
    binding = json.loads((root / "SOURCE_BINDINGS.json").read_text())
    for pin in binding["external_sources"]:
        path = root.parent / pin["path"]
        data = path.read_bytes()
        if len(data) != pin["bytes"] or sha256(data).hexdigest() != pin["sha256"]:
            raise RuntimeError("Source dependency changed: " + pin["path"])
    spec = json.loads((root / "FROZEN_SPEC.json").read_text())
    assert spec["arms"] == ["P0", "J_P", "F_P", "C_mu"]
    assert spec["paired_model_seed"] == 610041 and spec["replay_master_seed"] == 2026100401
    assert spec["epochs"] == 100 and spec["optimizer_updates_per_arm"] == 1700
    assert spec["execution_authorized"] is False and spec["selected_state_donors"] is False
    # Local source dimension formulas; no instantiated model or imports.
    head = (195 * 64 + 64) + (64 * 64 + 64) + (530 * 64 + 64) + (64 + 1)
    assert head == spec["C_mu"]["head_parameters_formula"] == 50753
    assert head + 38147 == spec["C_mu"]["total_parameters_formula"] == 88900
    return {"schema": "pooled-joint-quality-static-verification-v1", "status": "PASS_STATIC_ONLY",
            "UTC": datetime.now(timezone.utc).isoformat(), "parsed_compiled_python": sources,
            "parsed_json": parsed_json, "external_source_pins_verified": len(binding["external_sources"]),
            "model_or_runtime_imports": False, "numerical_checks_executed": False,
            "data_checkpoint_score_reads": False, "GPU_or_remote_execution": False,
            "runtime_feasibility_or_fit_ready": False, "execution_authorized": False}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
