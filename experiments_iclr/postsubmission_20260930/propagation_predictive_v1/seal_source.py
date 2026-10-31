"""Seal source/dependency bytes after a stdlib audit. Authorizes no execution."""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def file_sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def main():
    binding_path, manifest_path = HERE / "SOURCE_BINDINGS.json", HERE / "SOURCE_MANIFEST_v1.json"
    if binding_path.exists() or manifest_path.exists():
        raise ValueError("Preserve previous source seal; create a new explicit packet generation")
    static = json.loads((HERE / "STATIC_SOURCE_CHECK_v1.json").read_text())
    if static.get("status") != "source_checks_passed_no_numerical_execution":
        raise ValueError("Missing static source audit")
    for name, expected in static["python_source_sha256"].items():
        if file_sha(HERE / name) != expected:
            raise ValueError("Scientific source changed after static check")
    closure = {}
    for packet in ("propagation_cost_engineering_v1", "propagation_cost_impl_v1"):
        upstream = PHASE / packet / "SOURCE_MANIFEST_local_run01.json"
        manifest = json.loads(upstream.read_text())
        closure[str(upstream.relative_to(PHASE))] = file_sha(upstream)
        for row in manifest["files"]:
            path = upstream.parent / row["relative_path"]
            if path.stat().st_size != row["bytes"] or file_sha(path) != row["sha256"]:
                raise ValueError("Sealed upstream dependency changed")
            closure[str(path.relative_to(PHASE))] = row["sha256"]
    for relative in ("protocols/run_authorized_v2.py", "protocols/run_logged.py", "protocols/repo_env.sh",
                     "protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json",
                     "propagation_cost_scout_v1/PILOT_SPEC.json",
                     "propagation_cost_scout_v1/PILOT_AMENDMENT_ROOT_before_fits_v2.json"):
        closure[relative] = file_sha(PHASE / relative)
    for path in sorted(HERE.iterdir()):
        if path.is_file() and path.name not in (binding_path.name, manifest_path.name):
            if path.is_symlink():
                raise ValueError("Symlink in prospective scientific sources")
            closure[str(path.relative_to(PHASE))] = file_sha(path)
    with binding_path.open("x") as stream:
        json.dump({"schema": "gnnm-propagation-predictive-source-bindings-v1", "sha256": closure,
            "actual_data_runtime_or_execution_policy_bound": False,
            "scientific_training_authorized": False}, stream, indent=2, sort_keys=True)
        stream.write("\n")
    files = [{"relative_path": str(path.relative_to(HERE)), "bytes": path.stat().st_size,
              "sha256": file_sha(path)} for path in sorted(HERE.iterdir()) if path.is_file() and path != manifest_path]
    with manifest_path.open("x") as stream:
        json.dump({"schema": "gnnm-propagation-fixed30-predictive-source-manifest-v1",
            "status": "source_only_sealed_scientific_launch_closed", "files": files,
            "source_bindings_sha256": file_sha(binding_path), "static_check_sha256": file_sha(HERE / "STATIC_SOURCE_CHECK_v1.json"),
            "Torch_CUDA_remote_data_or_predictive_results_accessed_by_author": False,
            "existing_scores_recalculated": False, "scientific_fits": 0,
            "all_complete_graph_admissions_and_expanded_resource_gates_required": True,
            "novelty_acceptance_or_pilot_success_established": False}, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"manifest": str(manifest_path), "manifest_sha256": file_sha(manifest_path),
                      "source_bindings_sha256": file_sha(binding_path), "scientific_launch_closed": True}))


if __name__ == "__main__":
    main()
