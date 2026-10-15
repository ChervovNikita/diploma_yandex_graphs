"""Compile exact reporting identities/profiles from pinned compiler4 metadata.

Allocation-only stdlib metadata build. Does not deserialize data or import models.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys

INDEX_SHA = "485be18f27a0cdd17e3d6a923e1dc1a46deefe580cf0563373070c4b9fe4bdcb"
PLAN_SHA = "42973f8c9573b3d782879eff0da7aee3546d1416dccce31f6f3ab0c120059351"
CANDIDATES_SHA = "474110d38bb4f07527f59028a26f8fb8377e19dc492cf62ea4fe6c057582179e"


def main():
    if socket.gethostname() != "anogena-2-0" or subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines() != [
                "GPU-44039938-fd82-41d2-fefd-de71514e2fac"]:
        raise ValueError("Exact allocation guard required before project metadata reads")
    here = Path(__file__).resolve().parent
    expansion, phase = here.parent, here.parent.parent
    root = phase.parent.parent
    sys.path.insert(0, str(expansion / "benchmark_reporting_implementation_v1"))
    import reporter as r
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(here) or output.exists():
        raise ValueError("Fresh own generated metadata output required")
    output.mkdir()
    source_pins = []

    def read(path, expected=None):
        path = Path(path)
        if not path.resolve().is_relative_to(root):
            raise ValueError("Metadata source escapes repository")
        pin = r.pin(path)
        if expected is not None and pin["sha256"] != expected:
            raise ValueError("Frozen metadata changed: " + str(path))
        source_pins.append(pin)
        return json.loads(path.read_text())

    index = read(expansion / "expanded_job_binding_v1/generated_v4/DESCRIPTOR_INDEX.json", INDEX_SHA)
    plan = read(expansion / "expanded_dispatcher_v2/generated_v4/PLAN.json", PLAN_SHA)
    candidates = read(expansion / "expanded_dispatcher_v2/generated_v4/EXECUTION_CANDIDATES.json", CANDIDATES_SHA)
    facts_path = expansion / "benchmark_review_protocol_proposal_v1/FACTS.json"
    facts = read(facts_path, "d50e88abafbebe9b5baec96c79b97cc6692ff8ad7a859b9749b8c08ed5d9c3b7")
    contrasts = r.compile_contrast_plan(facts)
    cells = read(expansion / "job_binding_preparation_v1/CELL_BINDINGS.json")["cells"]
    authors = read(phase / "reference_wrapper_redesign_20261013_v1/native_sources/MANIFEST.json")["files"]
    task_by_id = {t["id"]: t for t in plan["tasks"]}
    candidate_by_id = {c["id"]: c for c in candidates["candidates"]}
    if len(index["fits"]) != 6460 or len(task_by_id) != 6730:
        raise ValueError("Campaign inventory changed")
    presets = {p["id"]: p for p in facts["presets"]}
    group = {key: {"expected_predictors": [], "fit_specs": {}, "I4_specs": {}, "cohorts": {}} for key in presets}
    profile_inputs = {}

    def normalized(pin):
        return {"bytes": pin["bytes"], "sha256": pin["sha256"]}

    def filepin(pin):
        return {key: pin[key] for key in ("path", "bytes", "sha256")}

    def endpoint(family, job):
        return {"name": job["policy"]["metric"], "units": "fraction", "direction": job["policy"]["direction"]}

    for record in index["fits"]:
        desc_pin = record["descriptor"]
        job = read(desc_pin["path"], desc_pin["sha256"])
        if r.digest(job) != record["runtime_job_hash"]:
            raise ValueError("Descriptor job hash mismatch")
        task, candidate = task_by_id[job["id"]], candidate_by_id[job["id"]]
        if candidate["execution"]["job_descriptor"] != desc_pin:
            raise ValueError("Dispatcher/source descriptor mismatch")
        preset_id = record["cell_id"]
        target, preset, cell = group[preset_id], presets[preset_id], cells[preset_id]
        family, seed, split = record["family"], record["cohort_seed"], record["split"]
        literal = preset["source_literal_split_seeds"][str(split)]
        track = "source_reference" if record["track"] == "native_replica" else "matched_extended"
        body = record["body"]
        native_family = "polyformer" if family == "node" else "ogb" if family == "ogb" else "buddy" if preset_id == "collab-buddy" else "ncn"
        native_files = [a for a in authors if a["family"] == native_family]
        native_closure = [{"logical_path": a["snapshot_path"].split("/native_sources/", 1)[-1], **normalized(a)} for a in native_files]
        native_inputs = [{"path": str(phase / a["snapshot_path"]), **normalized(a)} for a in native_files]
        data_files = [filepin(v) for v in cell["data_role_receipts"].values()]
        data_closure = {key: normalized(value) for key, value in cell["data_role_receipts"].items()}
        eval_path = root / (".venv/lib/python3.11/site-packages/ogb/graphproppred/evaluate.py" if family == "ogb"
                           else ".venv/lib/python3.11/site-packages/ogb/linkproppred/evaluate.py") if family != "node" \
            else Path(cell["smoke_callback"]["path"])
        if str(eval_path) not in profile_inputs:
            profile_inputs[str(eval_path)] = r.pin(eval_path)
        eval_pin = profile_inputs[str(eval_path)]
        method = job.get("scientific_method_id", record.get("method_id", job["procedure"]))
        if job["procedure"] == "native_single":
            method = "native_M1" if body == 0 else "native_body_" + str(body)
        bindings = {
            "native_source": r.digest({"native_files": native_closure, "native_config": cell["native_fixed_config"]}),
            "data": r.digest({"dataset": cell["dataset"], "roles": data_closure, "split_index": split, "literal_split_seed": literal}),
            "controller": r.digest(job["policy"]),
            "evaluator": r.digest({"source": normalized(eval_pin), "endpoint": endpoint(family, job)}),
            "readout": r.digest({"method": method, "runtime_procedure": job["procedure"], "callback_sha256": job["callback_sha256"],
                                  "extension": job.get("session_transform"), "readout": "actual"}),
        }
        inputs = {p["path"]: p for p in native_inputs + data_files + [filepin(eval_pin)]}
        callback = {"path": job["callback_module"], "sha256": job["callback_sha256"]}
        callback["bytes"] = Path(callback["path"]).stat().st_size
        inputs[callback["path"]] = callback
        extension = job.get("session_transform")
        if extension:
            inputs[extension["path"]] = {"path": extension["path"], "bytes": Path(extension["path"]).stat().st_size, "sha256": extension["sha256"]}
        identity = {"dataset": cell["dataset"], "preset_id": preset_id, "track": track, "cohort_id": job["cohort_id"],
            "cohort_seed": seed, "split_index": split, "literal_split_seed": literal,
            "scientific_method_id": method, "runtime_procedure_id": job["procedure"], "readout": "actual",
            "fit_id": job["id"], "actual_seed": job["seed"], "endpoint": endpoint(family, job), "bindings": bindings,
            "native_proper_loss_definition": "native complete-VALID callback loss/reduction; exact source/evaluator authority retained"}
        spec = {"descriptor": desc_pin, "job_hash": record["runtime_job_hash"], "identity": identity,
            "runtime_entrypoint": candidate["execution"]["runtime_entrypoint"], "output_relative": task["output_relative"],
            "bindings": bindings, "observed_input_files": list(inputs.values()),
            "wrapper_authority": {"callback": callback, "extension": extension, "descriptor": desc_pin},
            "native_comparison_fingerprint_excludes_wrapper_and_host_paths": True}
        target["fit_specs"][job["id"]] = spec
        if track == "source_reference" or job["procedure"] != "native_single" or body == 0:
            target["expected_predictors"].append(identity)
        if track == "matched_extended":
            cohort = {"cohort_id": job["cohort_id"], "seed": seed, "split_index": split, "literal_split_seed": literal}
            old = target["cohorts"].get(seed)
            if old is not None and old != cohort:
                raise ValueError("Actual compiler method/control cohort IDs differ")
            target["cohorts"][seed] = cohort
        needs_mean = any(c["rhs"] == {"method_id": method, "readout": "same_selected_bank_mean"} for c in contrasts["presets"][preset_id])
        if needs_mean and track == "matched_extended":
            mean_identity = json.loads(json.dumps(identity))
            mean_identity["readout"] = "same_selected_bank_mean"
            mean_identity["bindings"]["readout"] = r.digest({"actual_readout_hash": bindings["readout"], "readout": "same_selected_bank_mean"})
            target["expected_predictors"].append(mean_identity)
            spec["same_bank_mean_identity"] = mean_identity

    for task in plan["tasks"]:
        if task["kind"] == "fit":
            continue
        target = group[task["cell_id"]]
        members = [target["fit_specs"][key] for key in task["dependencies"]]
        m1 = members[0]["identity"]
        if task["M1_exact_reuse"] != m1["fit_id"] or len(members) != 4:
            raise ValueError("I4/M1 compiler dependency identity mismatch")
        identity = json.loads(json.dumps(m1))
        identity.update(scientific_method_id="native_I4", runtime_procedure_id="aggregate_I4_exports", fit_id=task["id"], actual_seed=None,
                        component_fit_ids=task["dependencies"])
        aggregate_pin = candidate_by_id[task["id"]]["execution"]["runtime_entrypoint"]
        identity["bindings"]["readout"] = r.digest({"source": normalized(aggregate_pin), "aggregation": "mean_raw_logits_after_four_own_selectors"})
        target["expected_predictors"].append(identity)
        target["I4_specs"][task["id"]] = {"identity": identity, "output_relative": task["output_relative"],
            "component_fit_ids": task["dependencies"], "aggregator_source": aggregate_pin}

    registry_files = []
    counts = {"physical_fits": 0, "I4_exports": 0, "expected_predictor_rows": 0, "source_reference_rows": 0,
              "same_bank_mean_rows": 0}
    for preset_id, data in sorted(group.items()):
        registry = {"schema": "expanded-ten-cohort-report-registry-v1", "presets": [{"id": preset_id,
            "matched_cohorts": [data["cohorts"][seed] for seed in range(10)], "contrasts": contrasts["presets"][preset_id]}],
            "expected_predictors": data["expected_predictors"]}
        payload = {"schema": "real-compiler-reporting-preset-v2", "registry": registry,
            "fit_specs": data["fit_specs"], "I4_specs": data["I4_specs"],
            "contrast_plan": {**contrasts, "presets": {preset_id: contrasts["presets"][preset_id]},
                "cohort_keys": {preset_id: contrasts["cohort_keys"][preset_id]}}}
        path = output / (preset_id + ".json")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        registry_files.append(r.pin(path))
        counts["physical_fits"] += len(data["fit_specs"])
        counts["I4_exports"] += len(data["I4_specs"])
        counts["expected_predictor_rows"] += len(data["expected_predictors"])
        counts["source_reference_rows"] += sum(x["track"] == "source_reference" for x in data["expected_predictors"])
        counts["same_bank_mean_rows"] += sum(x["readout"] == "same_selected_bank_mean" for x in data["expected_predictors"])
    if counts["physical_fits"] != 6460 or counts["I4_exports"] != 270:
        raise ValueError("Reporting registry lost a physical fit or I4 export")
    summary = {"schema": "real-compiler-reporting-registry-index-v2", "compiler_generation": 4, "dispatcher_generation": 4,
        "counts": counts, "preset_files": registry_files, "source_inputs": source_pins[:5],
        "all_descriptor_files_SHA_verified": 6460, "campaign_output_root": plan["output_root"],
        "no_native_tensor_or_TEST_work": True, "no_launch_permission": True,
        "parent_reporter_source": r.pin(expansion / "benchmark_reporting_implementation_v1/reporter.py")}
    (output / "INDEX.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
