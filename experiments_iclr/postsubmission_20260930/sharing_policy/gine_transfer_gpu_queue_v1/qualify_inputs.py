"""Qualify only new corrected-allocation metadata; no target deserialization."""
import argparse
import json
from pathlib import Path

from queue_common import HERE, PHASE, REPO, PARENT, authorization, check_parent_sources, confined, idle_guard, phase_path, plan, sha, source_hashes, write_json


def request_binding(request_path):
    request_path = confined(request_path)
    ready_path = HERE / "QUALIFICATION_REQUEST_authorized1gpu_run01.json"
    if request_path != ready_path:
        raise ValueError("Use the sealed exact newly authorized qualification request")
    request = json.loads(request_path.read_text())
    value = plan()
    if request.get("schema") != "gnnm-authorized-GPU-qualification-request-v1":
        raise ValueError("Unknown qualification request schema")
    for name in ("bundle", "protocol"):
        if (request[name]["path"] != value["new_authorized_" + name + "_phase_relative"]
                or request[name]["sha256"] != value["new_authorized_" + name + "_sha256"]):
            raise ValueError("Input path/digest differs from exact fresh authorized PLAN binding")
    return request


def command_value(argv, flag):
    if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv):
        raise ValueError("Required prerequisite input flag not unambiguous:" + flag)
    return argv[argv.index(flag) + 1]


def command_input_path(argv, flag, expected):
    supplied = Path(command_value(argv, flag))
    supplied = supplied if supplied.is_absolute() else REPO / supplied
    if supplied.resolve() != Path(expected).resolve():
        raise ValueError("Prerequisite input path differs:" + flag)


def expected_prerequisite_argv(role, request, bundle, protocol_path):
    python = ".venv/bin/python"
    report = phase_path(request["reports"][role]) if role in request["reports"] else None
    if role in ("chemistry", "mapping", "synthetic", "real128"):
        script = {"chemistry": "verify_chemical_groups.py", "mapping": "verify_mapping_identity.py",
                  "synthetic": "verify_cpu.py", "real128": "verify_cpu.py"}[role]
        argv = [python, str(PARENT / script)]
        if role == "real128":
            argv += ["--train-bundle", str(bundle)]
        return argv + ["--report", str(report)]
    if role == "freeze":
        return [python, str(PARENT / "probe.py"), "freeze", "--train-bundle", str(bundle),
                "--output", str(protocol_path)]
    if role == "cuda":
        return [python, str(PHASE / plan()["CUDA_gate_phase_relative"]), "--train-bundle", str(bundle),
                "--expected-bundle-sha256", request["bundle"]["sha256"], "--export-supervisor",
                str(phase_path(request["supervisors"]["export"])), "--output", str(report)]
    if role == "export":
        provider = PHASE / "new_tasks/data/molhiv/provider/ogbg_molhiv"
        return [python, str(PARENT / "make_train_bundle.py"), "--data-manifest",
                str(PHASE / "new_tasks/data/molhiv/frozen_shareable_manifest_v2.json"), "--processed",
                str(provider / "processed/geometric_data_processed.pt"), "--split-train",
                str(provider / "split/scaffold/train.csv.gz"), "--molecule-map",
                str(provider / "mapping/mol.csv.gz"), "--output", str(bundle)]
    raise ValueError("Unknown prerequisite role")


def verify_prerequisite_argv(role, argv, request, bundle, protocol_path):
    if argv != expected_prerequisite_argv(role, request, bundle, protocol_path):
        raise ValueError("Prerequisite argv differs from its exact fresh authorized role:" + role)


def qualification_value(request_path, initial_idle):
    request_path = confined(request_path)
    request = request_binding(request_path)
    bundle, protocol_path = (phase_path(request[name]["path"]) for name in ("bundle", "protocol"))
    if PARENT not in bundle.parents or PARENT not in protocol_path.parents:
        raise ValueError("Qualified bundle/protocol must remain in parent v6")
    for name, path in (("bundle", bundle), ("protocol", protocol_path)):
        if sha(path) != request[name]["sha256"]:
            raise ValueError("New input differs from coordinator's exact digest")
    protocol = json.loads(protocol_path.read_text())
    if (protocol["train_bundle_sha256"] != sha(bundle) or protocol["validation_loaded_or_scored"]
            or protocol["test_loaded_or_scored_by_probe"] or protocol["predictive_fit_performed"]
            or protocol["recipe"]["optimizer_seeds"] != [0, 1, 2]
            or protocol["recipe"]["label_fractions"] != [0.1, 1.0]
            or protocol["recipe"]["probe_epochs"] != [0, 10, 25]):
        raise ValueError("Require unchanged complete train-only parent protocol")
    check_parent_sources(protocol)
    export_path = phase_path(request["export_provenance"])
    if export_path != bundle.with_suffix(".provenance.json"):
        raise ValueError("Export provenance must be derived from the exact authorized exporter output")
    export = json.loads(export_path.read_text())
    grouping = export["grouping"]
    if (export["status"] != "training_only_bundle_exported" or export["bundle_sha256"] != sha(bundle)
            or export["source_sha256"] != protocol["source_sha256"]
            or not grouping["all_train_graphs_retained"] or not grouping["all_train_input_atom_connectivity_checks_passed"]
            or not grouping["original_x_edge_attr_and_edge_index_unchanged"]
            or grouping["molecules_dropped_or_exception_singletons"] != 0 or export["export_scope"]["scores_read"]):
        raise ValueError("Fresh complete unchanged-topology TRAIN export required")
    evidence, uuid = {}, initial_idle["GPU"]["uuid"]
    expected_scripts = {"chemistry": "verify_chemical_groups.py", "mapping": "verify_mapping_identity.py",
                        "synthetic": "verify_cpu.py", "real128": "verify_cpu.py",
                        "export": "make_train_bundle.py", "freeze": "probe.py"}
    for name in ("chemistry", "mapping", "synthetic", "real128", "export", "freeze", "cuda"):
        supervisor = phase_path(request["supervisors"][name])
        report_path = phase_path(request["reports"][name]) if name in request["reports"] else None
        target = report_path if report_path else bundle if name == "export" else protocol_path
        flag = "--report" if report_path and name != "cuda" else "--output"
        record = authorization(supervisor, uuid, flag, target)
        command = json.loads((supervisor / "command.json").read_text())["argv"]
        verify_prerequisite_argv(name, command, request, bundle, protocol_path)
        expected_script = PHASE / plan()["CUDA_gate_phase_relative"] if name == "cuda" else PARENT / expected_scripts[name]
        if Path(command[1]).resolve() != expected_script:
            raise ValueError("Prerequisite did not execute the required parent source")
        if name == "freeze" and command[2] != "freeze":
            raise ValueError("Required parent scientific freeze was not executed")
        if name in ("freeze", "real128", "cuda"):
            command_input_path(command, "--train-bundle", bundle)
        if name == "synthetic" and "--train-bundle" in command:
            raise ValueError("Synthetic-only prerequisite supplied a data bundle")
        if name == "cuda":
            if command_value(command, "--expected-bundle-sha256") != request["bundle"]["sha256"]:
                raise ValueError("CUDA gate expected bundle digest differs")
            command_input_path(command, "--export-supervisor", phase_path(request["supervisors"]["export"]))
        uuid = record["gpu_uuid"]
        if report_path:
            report = json.loads(report_path.read_text())
            if report["status"] != "passed":
                raise ValueError("Failed prerequisite:" + name)
            if name in ("synthetic", "real128"):
                if report["source_sha256"] != protocol["source_sha256"] or report["optimizer_steps"] != 0:
                    raise ValueError("Autograd prerequisite source/scope mismatch")
            if name == "real128":
                real = report["real_training_batch"]
                if real["status"] != "passed" or real["train_bundle_sha256"] != sha(bundle) or real["heldout_data_accessed"]:
                    raise ValueError("Real TRAIN batch prerequisite differs")
            if name == "cuda":
                if (report["bundle_sha256"] != sha(bundle) or report["graphs"] != 128
                        or report["optimizer_steps"] != 1 or report["validation_or_test_scored"]
                        or report["predictive_improvement_claimed"]
                        or report["source_sha256"] != plan()["CUDA_gate_source_sha256"]
                        or report["authorized_exporter_record_sha256"] != evidence["export"]["authorization_sha256"]
                        or sha(report_path) != plan()["CUDA_gate_report_sha256"]):
                    raise ValueError("Authorized real128 CUDA/CPU-oracle/AdamW gate differs")
        evidence[name] = {"supervisor_phase_relative": str(supervisor.relative_to(PHASE)), **record,
                          "report_phase_relative": str(report_path.relative_to(PHASE)) if report_path else None,
                          "report_sha256": sha(report_path) if report_path else None}
    idle = idle_guard(uuid)
    value = {"schema": "gnnm-correct-allocation-GPU-input-qualification-v1", "status": "qualified",
             "source_sha256": source_hashes(), "request_sha256": sha(request_path),
             "qualifier_source_sha256": sha(HERE / "qualify_inputs.py"),
             "bundle_phase_relative": str(bundle.relative_to(PHASE)), "bundle_sha256": sha(bundle),
             "protocol_phase_relative": str(protocol_path.relative_to(PHASE)), "protocol_sha256": sha(protocol_path),
             "parent_source_sha256": protocol["source_sha256"], "qualified_gpu_uuid": uuid,
             "export_provenance_phase_relative": str(export_path.relative_to(PHASE)), "export_provenance_sha256": sha(export_path),
             "prerequisites": evidence, "idle_no_context_gate": idle,
             "initial_scope_idle_gate_before_data_digest_read": initial_idle,
             "training_graphs": export["training_graphs"], "structural_groups": grouping["groups"],
             "target_tensor_deserialized_by_qualifier": False, "wrong_allocation_evidence_accepted": False}
    return value


def qualify(args):
    initial_idle = idle_guard()
    value = qualification_value(args.request, initial_idle)
    write_json(args.output, value)
    print(json.dumps({"status": "qualified", "qualification_sha256": sha(args.output), "GPU_uuid": value["qualified_gpu_uuid"],
                      "target_tensor_deserialized": False}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    qualify(parser.parse_args())


if __name__ == "__main__":
    main()
