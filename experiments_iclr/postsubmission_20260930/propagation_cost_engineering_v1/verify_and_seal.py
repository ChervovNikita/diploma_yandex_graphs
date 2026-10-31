"""Stdlib source/forecast boundary checks and immutable source seal."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from propagation_cost_engineering_v1.contract import (
    ROOT, PROJECT, SCHEMA, TASKS, FORECAST_GPU_SECONDS, AMENDMENT_SHA256,
    sha256, verify_impl)


def main():
    manifest_path = ROOT / "SOURCE_MANIFEST_local_run01.json"
    if manifest_path.exists():
        raise FileExistsError("Preserve sealed source generation")
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append({"name": name, "passed": True})

    for path in sorted(ROOT.glob("*.py")):
        ast.parse(path.read_text(), filename=str(path))
    check("all engineering sources parse", True)
    check("sealed mechanics unchanged", bool(verify_impl()))
    amendment = PROJECT / "propagation_cost_scout_v1/PILOT_AMENDMENT_ROOT_before_fits_v2.json"
    check("prospective five-arm amendment pinned", sha256(amendment) == AMENDMENT_SHA256)
    model_source = (ROOT / "complete_models.py").read_text()
    tree = ast.parse(model_source)
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    check("ensemble computation inherited unchanged", {n.name for n in classes["CompleteEnsemble"].body
          if isinstance(n, ast.FunctionDef)} == {"_check"})
    check("no fixture guard monkeypatch", "fixture_shape" not in model_source)
    check("BASE arm metadata explicit", 'arm = "single_base"' in model_source)
    check("fresh prospective source seeds", "torch.manual_seed(optimizer_seed)" in model_source)
    check("own-member serial CE one AdamW update", model_source.count("optimizer.step()") == 1)
    check("detached same mean in both branches", "mean = normalized.mean(dim=1).detach()" in model_source
          and "deviation = normalized - mean[:, None, :]" in model_source and "common = family.full(mean)" in model_source)
    check("detached full operator refuses gradients", "if features.requires_grad" in model_source)
    check("serial inference streams pooling", "result.add_(logits)" in model_source)
    data_source = (ROOT / "data_contract.py").read_text()
    check("training-only export keys", '"training_targets"' in data_source and '"y"' not in data_source)
    check("prepared receipt lineage equality", 'metadata["data_binding"] != data_binding' in (ROOT / "prepare.py").read_text())
    profile_tree = ast.parse((ROOT / "profile.py").read_text())
    defs = [n for n in profile_tree.body if isinstance(n, ast.FunctionDef)
            and n.name in {"variants", "variant_name", "choose_paths", "forecast_family"}]
    namespace = {"SCHEMA": SCHEMA, "TASKS": TASKS, "FORECAST_GPU_SECONDS": FORECAST_GPU_SECONDS}
    exec(compile(ast.Module(body=defs, type_ignores=[]), "pure_forecast_fixture", "exec"), namespace)
    check("twenty engineering variants with native controls", len(namespace["variants"]()) == 20)
    reports = []
    for task in TASKS:
        rows = []
        for arm in ("candidate", "detached_current_member_mean", "exact_member", "single_base", "naive_shared_sample"):
            rows.append({"arm": arm, "name": arm, "status": "passed", "inference_status": "passed",
                         "train_step_wall_seconds": [1.0] * 4,
                         "inference_validation_proxy_wall_seconds": [0.5] * 3,
                         "final_test_metric_proxy_wall_seconds": 2.0,
                         "operator_model_optimizer_H2D_setup_seconds": 10.0,
                         "peak_reserved_bytes": 10 * 2**30, "inference_peak_reserved_bytes": 10 * 2**30})
        rows.append({"arm": "exact_member", "name": "INVALID_MEMORY_FAST_PATH", "status": "resource_failed_memory_cap",
                     "inference_status": "passed", "inference_validation_proxy_wall_seconds": [0.0001],
                     "inference_peak_reserved_bytes": 61 * 2**30})
        reports.append({"task": task, "variants": rows, "complete_input_H2D_seconds": 2.0,
                        "actual_engineering_GPU_phase_wall_seconds": 100.0,
                        "preparation_wall_seconds": 100.0, "input_audit_CPU_wall_seconds": 10.0,
                        "prior_cpu_seconds": 100.0, "prior_disk_bytes": 1000,
                        "actual_prepared_file_bytes": 2000})
    choice = namespace["choose_paths"](reports[0])
    check("memory-failed inference cannot win", choice["choices"]["exact_member"]["fastest_inference_path"] != "INVALID_MEMORY_FAST_PATH")
    forecast = namespace["forecast_family"](reports, 20.0)
    check("30fit forecast includes setup H2D and prior", forecast["fit_count"] == 30
          and abs(forecast["total_GPU_seconds_forecast"] - 7495.0) < 1e-9)
    check("forecast never self-admits fits", forecast["scientific_training_admitted"] is False)
    reports[0]["prior_disk_bytes"] = 201 * 2**30
    check("disk cap rejects otherwise cheap family", namespace["forecast_family"](reports, 20.0)["status"] == "resource_forecast_fail")
    result = {"schema": SCHEMA, "status": "passed", "kind": "stdlib_source_and_artificial_forecast_fixture",
              "checks": checks, "artificial_timing_values_are_not_measurements": True,
              "Torch_or_CUDA_executed": False, "data_opened": False,
              "remote_accessed": False, "scientific_fits": 0}
    (ROOT / "STDLIB_GATE_local_run01.json").write_text(json.dumps(result, indent=2) + "\n")
    files = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("No symlinks in source packet")
        if path.is_file():
            files.append({"relative_path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                          "sha256": sha256(path)})
    manifest = {"schema": SCHEMA, "status": "source_only_sealed", "UTC": datetime.now(timezone.utc).isoformat(),
                "files": files, "file_count": len(files), "sealed_mechanics_manifest_sha256": verify_impl(),
                "prospective_amendment_sha256": AMENDMENT_SHA256,
                "Torch_or_CUDA_executed_by_author": False, "data_opened_by_author": False,
                "remote_accessed_by_author": False, "scientific_fits": 0,
                "complete_graph_resource_feasibility_measured_by_author": False,
                "scientific_training_authorized": False}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"checks": len(checks), "files": len(files),
                      "manifest_sha256": sha256(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
