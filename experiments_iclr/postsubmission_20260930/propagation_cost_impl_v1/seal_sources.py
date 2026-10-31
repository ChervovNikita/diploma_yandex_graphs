"""Seal this packet and its honest evidence boundary without editing inputs."""

from __future__ import annotations

from datetime import datetime, timezone
import json

from .contract import ROOT, SCHEMA, check_pins, sha256


def main():
    output = ROOT / "SOURCE_MANIFEST_local_run01.json"
    if output.exists():
        raise FileExistsError("Do not overwrite a source seal")
    check_pins()
    scout = ROOT.parent / "propagation_cost_scout_v1"
    context_files = [scout / name for name in ("PILOT_SPEC.json", "REPORT.md", "MATHEMATICAL_LIMITS.md")]
    gate = json.loads((ROOT / "STDLIB_GATE_local_run04.json").read_text())
    if gate.get("status") != "passed" or len(gate["checks"]) != 30:
        raise RuntimeError("Expected final stdlib verification is missing")
    files = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("Source packet cannot contain symlinks")
        if path.is_file():
            if path.suffix not in (".py", ".md", ".json"):
                raise RuntimeError("Unexpected output in unlaunched source packet")
            files.append({"relative_path": str(path.relative_to(ROOT)),
                          "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "schema": SCHEMA, "status": "source_only_sealed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "context_bindings": [{"relative_to_project": str(path.relative_to(ROOT.parent)),
                              "bytes": path.stat().st_size, "sha256": sha256(path)}
                             for path in context_files],
        "files": files, "file_count": len(files),
        "stdlib_checks": 30, "torch_runtime_executed_by_author": False,
        "GPU_runtime_executed_by_author": False, "dataset_accessed": False,
        "remote_accessed": False, "real_pilot_closed": True,
        "complete_graph_feasibility_measured": False,
        "predictive_result_count": 0, "latency_result_count": 0,
        "scientific_adoption": False, "novelty_claim": False,
    }
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(output), "sha256": sha256(output),
                      "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
