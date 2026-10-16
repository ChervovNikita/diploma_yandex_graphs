"""Collect actual compiler-bound RESULT and scalar sidecars; JSON metadata only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys


def native_history_cost_row(fit_id, past, source_pin, observer_cost_seen):
    attempt = hashlib.sha256(json.dumps(source_pin, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"attempt_id": attempt, "fit_id": fit_id,
        "worker_wall_seconds": None if observer_cost_seen else past.get("invocation_wall_seconds"),
        "recorded_native_invocation_wall_seconds": past.get("invocation_wall_seconds"),
        "cost_attribution": "possible observer overlap; uncovered historical cost unresolved" if observer_cost_seen else "native RESULT invocation observation",
        "status": past.get("status"), "optimizer_calls_this_invocation": past.get("optimizer_calls_this_invocation"),
        "resources": past.get("resources"), "source": source_pin,
        "unrecorded_crashed_work": "unknown; absent or uncovered attempts are never inferred as zero"}


def main():
    from composed_authority import runtime_witness
    import importlib.util
    host_path=Path(__file__).resolve().parent.parent/'host_binding.py'
    hs=importlib.util.spec_from_file_location('_server51_collect_host_guard',host_path)
    host=importlib.util.module_from_spec(hs);hs.loader.exec_module(host)
    config=host.guard()
    here = Path(__file__).resolve().parent
    expansion = here.parent.parent
    root = Path(config["project_root"]).resolve()
    sys.path.insert(0, str(expansion / "benchmark_reporting_implementation_v1"))
    import reporter as r
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-index", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--campaign-output-root")
    args = parser.parse_args()
    index_path, output = Path(args.registry_index).resolve(), Path(args.output).resolve()
    if not index_path.is_relative_to(root) or not output.is_relative_to(root) or output.exists():
        raise ValueError("Own pinned metadata input and fresh report output required")
    index = json.loads(index_path.read_text())
    source_binding = index.get('host_projection') or {}
    for key, filename in [('observer_source', 'observe_exports.py'), ('collector_source', 'collect_campaign.py'),
                          ('runtime_entry_source', 'runtime_entry.py'), ('composed_authority_source', 'composed_authority.py')]:
        if source_binding.get(key) != r.pin(here / filename):
            raise ValueError('Frozen normalization reporting source differs: ' + key)
    if r.pin(expansion / "benchmark_reporting_implementation_v1/reporter.py") != index["parent_reporter_source"]:
        raise ValueError("Registry's accepted reporter source changed")
    campaign = Path(args.campaign_output_root or index["campaign_output_root"]).resolve()
    if not campaign.is_relative_to(root):
        raise ValueError("Campaign output escapes repository")
    output.mkdir()
    observer_pin = r.pin(here / "observe_exports.py")
    entry_pin = r.pin(here / "runtime_entry.py")
    total = {"presets": 0, "expected_predictors": 0, "observed_predictors": 0,
             "complete_preset_contrasts": 0, "withheld_intervals": 0, "physical_RESULTS": 0, "I4_RESULTS": 0}
    reports = []

    def read(path):
        path = Path(path)
        if path.suffix != ".json" or not path.resolve().is_relative_to(root):
            raise ValueError("Own JSON metadata only")
        return json.loads(path.read_text())

    for preset_pin in index["preset_files"]:
        if r.pin(preset_pin["path"]) != preset_pin:
            raise ValueError("Frozen preset registry changed")
        preset = read(preset_pin["path"])
        if preset.get('selected_serving_source') != index.get('selected_serving_source'):
            raise ValueError('Profile selected-serving source differs from the normalization index')
        observations, normalized_fits, ledger = [], {}, []
        for fit_id, spec in preset["fit_specs"].items():
            folder = campaign / spec["output_relative"]
            result_path = folder / "RESULT.json"
            if not result_path.is_file():
                continue
            if r.pin(spec["descriptor"]["path"]) != spec["descriptor"]:
                raise ValueError("Original compiler descriptor changed")
            job, raw = read(spec["descriptor"]["path"]), read(result_path)
            evidence = None
            selected_sidecar = None
            observer_cost_seen = False
            reporting = campaign / "reporting" / r.digest(fit_id)
            for reporting_index in sorted(reporting.glob("*/INDEX.json")):
                observed_index = read(reporting_index)
                if observed_index.get("job_hash") != spec["job_hash"] or observed_index.get("producer_receipt") != observer_pin:
                    raise ValueError("Observed reporting source/job authority mismatch")
                verified = {p["path"]: p for p in observed_index.get("verified_input_files", [])}
                expected_files = {p["path"]: p for p in spec["observed_input_files"]}
                attached = observed_index.get("binding_spec_attached") is True and verified == expected_files
                invocation_wall = 0.0
                for invocation_path in sorted(reporting_index.parent.glob("INVOCATION_*.json")):
                    inv = read(invocation_path)
                    if inv["job_hash"] != spec["job_hash"] or inv["producer_receipt"] != observer_pin:
                        raise ValueError("Invocation reporting authority mismatch")
                    costs = inv["actual_cost"]
                    ledger.append({"attempt_id": r.digest(r.pin(invocation_path)), "fit_id": fit_id,
                        "worker_wall_seconds": costs["worker_wall_seconds"], "status": costs["status"],
                        "optimizer_calls_this_invocation": costs["optimizer_calls_this_invocation"],
                        "resources": costs["resources"], "source": r.pin(invocation_path)})
                    observer_cost_seen = True
                    if r.numeric(costs["worker_wall_seconds"]):
                        invocation_wall += costs["worker_wall_seconds"]
                    if inv.get("selected_export_sha256") == (raw.get("selected_VALID_export") or {}).get("sha256"):
                        sidecar = inv.get("selected_sidecar")
                        if sidecar is not None:
                            if sidecar["actual_validation"] != raw.get("selected_VALID"):
                                raise ValueError("Actual scalar sidecar differs from the same selected RESULT")
                            selected_sidecar = sidecar
                        if attached and not inv.get("binding_unresolved_reasons"):
                            if inv["bindings"] != spec["bindings"]:
                                raise ValueError("Observed common/source/readout fingerprint differs from the frozen registry")
                            if 'composed_runtime_authority' in spec:
                                witness = runtime_witness(job, spec, raw.get('runtime_binding'))
                                if inv.get('composed_runtime_witness') != witness or sidecar is None \
                                        or sidecar.get('composed_runtime_witness') != witness:
                                    raise ValueError('Actual composed runtime/export witness differs from the frozen profile')
                            evidence = inv
                elapsed = observed_index.get("observer_elapsed_seconds")
                if r.numeric(elapsed):
                    ledger.append({"attempt_id": r.digest({"overhead": r.pin(reporting_index)}), "fit_id": fit_id,
                        "worker_wall_seconds": max(0.0, elapsed - invocation_wall),
                        "cost_scope": "observed context overhead including verification/serialization/owner gaps; not pure preprocessing",
                        "status": "observed", "source": r.pin(reporting_index)})
            # Always preserve every retained native RESULT. An observer may have
            # begun after older failed/crashed/resumed attempts. Current scalar
            # observer rows lack a guaranteed UTC attempt identity, so overlapping
            # history is retained with unresolved attribution rather than dropped
            # or silently added a second time to the numeric wall sum.
            histories = sorted((folder / "history").glob("before_resume_*_RESULT.json")) + [result_path]
            for historical in histories:
                past = read(historical)
                if past.get("job_hash") != spec["job_hash"]:
                    raise ValueError("Retained attempt belongs to another fit")
                ledger.append(native_history_cost_row(fit_id, past, r.pin(historical), observer_cost_seen))
            row = r.extract_runtime(job, raw, spec["identity"], evidence)
            normalized_fits[fit_id] = row
            if spec["identity"]["scientific_method_id"] == "native_M1" or job["procedure"] != "native_single":
                observations.append(row)
            if "same_bank_mean_identity" in spec and selected_sidecar is not None and "same_selected_bank_mean" in selected_sidecar:
                mean = selected_sidecar["same_selected_bank_mean"]
                if mean["runtime_binding"] != raw.get("runtime_binding"):
                    raise ValueError("Mean sidecar runtime authority differs from its actual selected bank")
                diagnostic = copy.deepcopy(row)
                diagnostic.update(spec["same_bank_mean_identity"])
                diagnostic["bindings"] = copy.deepcopy(row["bindings"])
                if evidence is not None:
                    diagnostic["bindings"]["readout"] = spec["same_bank_mean_identity"]["bindings"]["readout"]
                diagnostic.update(readout="same_selected_bank_mean",
                    selected_served_metric=mean["validation"]["metrics"].get(row["endpoint"]["name"]),
                    native_proper_loss=mean["validation"]["loss"],
                    costs={"physical_fit_count": 0, "same_physical_fit_cost_reference": fit_id},
                    same_call_authority=mean["same_call_authority"], diagnostic_only=True)
                observations.append(diagnostic)
            total["physical_RESULTS"] += 1
        for task_id, spec in preset["I4_specs"].items():
            result_path = campaign / spec["output_relative"] / "RESULT.json"
            if not result_path.is_file():
                continue
            raw = read(result_path)
            components = [normalized_fits[key] for key in spec["component_fit_ids"] if key in normalized_fits]
            if len(components) != 4:
                raise ValueError("I4 RESULT exists without its four exact component RESULT authorities")
            evidence = None
            reporting = campaign / "reporting" / r.digest(task_id)
            for sidecar_path in sorted(reporting.glob("*/AGGREGATE_SCALARS.json")):
                sidecar = read(sidecar_path)
                if sidecar.get("bank_receipt") != raw.get("bank_receipt"):
                    continue
                if sidecar["producer_receipt"] != entry_pin or sidecar["aggregator_source"] != spec["aggregator_source"] \
                        or sidecar["members"] != raw["members"] or sidecar["metrics"] != raw["metrics"] or sidecar["loss"] != raw["loss"]:
                    raise ValueError("I4 scalar execution/source/member authority mismatch")
                common = {key: components[0].get("bindings", {}).get(key) for key in ("native_source", "data", "controller", "evaluator")}
                if all(value is not None and all(p.get("bindings", {}).get(key) == value for p in components) for key, value in common.items()):
                    evidence = {"aggregate_bank_sha256": raw["bank_receipt"]["sha256"], "producer_receipt": entry_pin,
                        "bindings": {**common, "readout": spec["identity"]["bindings"]["readout"]}}
                cost_index = read(sidecar_path.parent / "INDEX.json")
                ledger.append({"attempt_id": r.digest(r.pin(sidecar_path.parent / "INDEX.json")), "fit_id": task_id,
                    "physical_fit_count": 0, "status": "aggregate_export", "worker_wall_seconds": cost_index["worker_wall_seconds"]})
            observations.append(r.extract_i4(raw, spec["identity"], components, evidence))
            total["I4_RESULTS"] += 1
        result = r.report(preset["registry"], observations, ledger, preset["contrast_plan"])
        result['normalization_control_attribution'] = preset.get('normalization_control_attribution')
        result['selected_serving_source'] = preset.get('selected_serving_source')
        result["campaign_outputs_exist"] = campaign.exists()
        result["source_preset_registry"] = preset_pin
        preset_id = preset["registry"]["presets"][0]["id"]
        path = output / (preset_id + ".json")
        path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        reports.append(r.pin(path))
        total["presets"] += 1
        total["expected_predictors"] += len(preset["registry"]["expected_predictors"])
        total["observed_predictors"] += len(observations)
        total["complete_preset_contrasts"] += sum(c["complete_preset_summary"] for c in result["contrasts"])
        total["withheld_intervals"] += sum(not c["complete_preset_summary"] for c in result["contrasts"])
    summary = {"schema": "real-campaign-metadata-report-index-v2", "registry_index": r.pin(index_path),
        "campaign_output_root": str(campaign), "counts": total, "reports": reports,
        "no_tensor_or_model_or_TEST_work": True, "full_campaign_GO": False}
    (output / "INDEX.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"counts": total, "index": r.pin(output / "INDEX.json")}))


if __name__ == "__main__":
    main()
