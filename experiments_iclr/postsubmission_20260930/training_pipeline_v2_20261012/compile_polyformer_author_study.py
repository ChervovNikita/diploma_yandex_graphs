"""Compile eight unchanged author CLI groups into a disabled scheduler packet."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shlex

from author_cli_spec import (SPLIT_SEEDS, author_counts, receipt, validate_author_spec,
                             verify_receipt)
from run_author_cli import verify_allocation
from spec import canonical, confined, fingerprint

RUNTIME_FILES = ["author_cli_spec.py", "run_author_cli.py", "compile_polyformer_author_study.py",
                 "registry.py", "spec.py", "run_study.py"]


def build_author_study(root, packet, runtime_prefix, resource_profiles):
    root=Path(root).resolve();packet=confined(root,packet)
    roster=json.loads((packet/"AUTHOR_ROSTER.json").read_text())
    qualified=json.loads((packet/"PREPARATION_SUMMARY.json").read_text())
    final=json.loads((packet/"FINAL_VERIFICATION.json").read_text())
    if (not qualified["dependency_and_unchanged_cli_help_qualified"] or qualified["scientific_training_processes_started"]
            or qualified["GO_created"] or final["scientific_fits"] or final["scores"]
            or not final["quarantine_empty"] or qualified["contract_sha256"]!=receipt(root,packet.relative_to(root)/"AUTHOR_ROSTER.json")["sha256"]):
        raise ValueError("Only the unchanged, qualified, disabled author preparation may enter this compiler")
    versions=json.loads(json.loads((packet/"AUTHOR_IMPORT_AND_RUNTIME_CHECK.json").read_text())["stdout"])
    inventory=json.loads((packet/"SOURCE_INVENTORY.json").read_text())
    dependencies=json.loads((packet/"PRIVATE_DEPENDENCY_INVENTORY.json").read_text())
    if len(roster["roster"])!=8 or roster["planned_model_fits"]!=80:
        raise ValueError("Prospectively retain the complete declared eight-group author roster")
    repository=root.parents[1]
    libdirs=[repository/".venv/lib/python3.11/site-packages/torch/lib",repository/".venv/lib/python3.11/site-packages/nvidia/cusparse/lib"]
    environment={"PYTHONPATH":str(packet/"dependencies/site"),"LD_LIBRARY_PATH":":".join(map(str,libdirs)),"DGLBACKEND":"pytorch"}
    bindings=[]
    for row in inventory:
        bound={"path":str((packet/"source"/row["path"]).relative_to(root)),"bytes":row["bytes"],"sha256":row["sha256"]}
        verify_receipt(root,bound);bindings.append(bound)
    official=[]
    for line in (packet/"source/node_classification.sh").read_text().splitlines():
        if not line.startswith("python "):
            continue
        argv=shlex.split(line)
        if argv[argv.index("--dataset")+1] in {"citeseer","pubmed","cs","physics"}:
            official.append(argv)
    if official != [job["original_script_argv"] for job in roster["roster"]] or any(job["python_argv"]!=argv[1:] for job,argv in zip(roster["roster"],official)):
        raise ValueError("Author roster arguments/order differ from the unchanged official script")
    for name in RUNTIME_FILES:
        bindings.append(receipt(root,Path(runtime_prefix)/name))
    for name in ["AUTHOR_ROSTER.json","SOURCE_INVENTORY.json","RAW_CACHE_ANCESTRY.json","PREPARATION_SUMMARY.json","FINAL_VERIFICATION.json","AUTHOR_IMPORT_AND_RUNTIME_CHECK.json","PRIVATE_DEPENDENCY_INVENTORY.json","SEABORN_WHEEL_INVENTORY.json"]:
        bindings.append(receipt(root,packet.relative_to(root)/name))
    for row in dependencies:
        bound=dict(row,path=str(packet.relative_to(root)/row["path"]))
        verify_receipt(root,bound);bindings.append(bound)
    runs=[]
    for job in roster["roster"]:
        profile=resource_profiles["jobs"].get(job["job_id"])
        if not profile or not profile.get("evidence"):
            raise ValueError("An explicit per-CLI-group resource profile and scope are required")
        work=packet/job["cwd"]
        source_files=[]
        for row in inventory:
            bound={"path":str((work/row["path"]).relative_to(root)),"bytes":row["bytes"],"sha256":row["sha256"]}
            verify_receipt(root,bound);source_files.append(bound);bindings.append(bound)
        raw_files=[]
        for row in roster["raw_custody"][job["canonical_dataset"]]["raw_files"]:
            bound={"path":str((work/job["provider_raw_dir"]/Path(row["path"]).name).relative_to(root)),"bytes":row["bytes"],"sha256":row["sha256"]}
            verify_receipt(root,bound);raw_files.append(bound)
        provider_dir=work/Path(job["provider_raw_dir"]).parent
        runs.append({"run_id":job["job_id"],"kind":"author_cli_group","method":"PolyFormer-unchanged-author-"+job["base"],
            "split":"native_source_ten_seed_sequence","seed":42,"body":0,"parents":[],
            "config":{"stage":"author_cli","dataset":job["canonical_dataset"],"source_dataset_argument":job["dataset"],
                      "base":job["base"],"source_order":job["source_order"],"reservation_mib":profile["reservation_mib"],
                      "wall_seconds":profile["wall_seconds"],"estimated_seconds":profile["estimated_seconds"],
                      "resource_evidence":profile["evidence"],"source_based_code_path_replication":True,
                      "published_task_and_environment_exact_match_claim":False},
            "budget":{"unit":"optimizer_updates_per_repeat","policy":"author_native_validation_patience","maximum":2000,"patience":250,"repeats":10,"resume":False},
            "selection":{"metric":"accuracy","direction":"max","ties":"first_strict_above_initial_zero","evaluation_interval":1,"checkpoint0":False,"artifact":"source_in_memory_no_checkpoint"},
            "cli":{"cwd":str(work.relative_to(root)),"argv":job["python_argv"],
                   "python_executable":str(repository/".venv/bin/python"),"execution_repository":str(repository),
                   "source_files":source_files,"raw_files":raw_files,"environment":environment,"runtime_versions":versions,
                   "private_dependency_dir":str((packet/"dependencies/site").relative_to(root)),
                   "quarantine":{name:str((packet/path).relative_to(root)) for name,path in job["output"].items()},
                   "basis_dir":str((work/"bases").relative_to(root)),"processed_dir":str((provider_dir/"processed").relative_to(root)),
                   "protocol":{"split_seeds":SPLIT_SEEDS,"repeat_grouping":"ten_sequential_repeats_one_process",
                               "constructor_before_split":True,"native_global_and_advancing_RNG":True,"per_epoch_TEST":"quarantined_unchanged_diagnostics",
                               "TRAIN_SELECT_progress":"rounded_source_diagnostics_only; never claim exact selected epoch from rounded logs"}}})
    unique={row["path"]:row for row in bindings}
    if len(unique)!=len(bindings):
        for row in bindings:
            if unique[row["path"]]!=row:
                raise ValueError("Conflicting source binding")
    value={"schema_version":1,"study_id":"polyformer-unchanged-author-cli-20261012-v1","stage":"development",
        "hypothesis":"Reproduce the pinned original PolyFormer code paths and reporting policy on the four source-scoped datasets under the explicitly declared allocation environment.",
        "dataset":"polyformer-four-source-dataset-author-roster",
        "data":{"task":"node_classification","preprocessing":"unchanged native DataLoader NormalizeFeatures/undirected/native basis from private raw caches",
                "role_protocol":"author_native_train_select_test_no_cal",
                "roles":{"train":"native60% class-balanced split rule","select":"native20% random remainder VAL","evaluate":"native remaining TEST; per-epoch diagnostics quarantined"},
                "prior_exposure":"development; whole-project unused-evidence audit incomplete; no protected-confirmation claim",
                "evidence":[receipt(root,packet.relative_to(root)/"RAW_CACHE_ANCESTRY.json")],"heldout_raw_tensors_decoded_by_compiler":False},
        "source":{"git_commit":roster["source_commit"],"repository":roster["source_repository"],"files":list(unique.values()),
                  "unchanged_author_roster":receipt(root,packet.relative_to(root)/"AUTHOR_ROSTER.json"),
                  "worker_entrypoint":str(Path(runtime_prefix)/"run_author_cli.py"),"environment_scope":"declared differing environment; source code-path replication, not bitwise or score-match guarantee"},
        "runs":runs,"comparisons":[{"reporting":"original ten-repeat descriptive reports, preserve every failure/cost",
                                      "adaptation_rule":"no outcome-driven settings, repeats, exclusions or reruns","manuscript_acceptance_guarantee":False}],
        "release":{"policy":"whole_roster","primary_endpoint":"quarantined source-native ten-repeat report after whole-author terminal closure",
                   "global_plan_gates_required":True,"required_before_launch":["whole_search"],"author_output_gate":"whole_author",
                   "release_failure_rule":"all groups terminal, failures retained; closure does not assert every group succeeded"},
        "resources":{"wall_seconds":sum(r["config"]["wall_seconds"] for r in runs),"profiles":resource_profiles,
                     "qualification":"provisional unless explicitly root-qualified; CLI import-only qualification is not model resource qualification"}}
    return validate_author_spec(value)


def write_packet(root, output, value):
    root=Path(root).resolve();output=confined(root,output)
    if output.exists():
        raise ValueError("Use a fresh disabled author scheduler packet")
    output.mkdir();(output/"studies").mkdir()
    path=output/"studies"/(value["study_id"]+".json.gz")
    path.write_bytes(gzip.compress((canonical(value)+"\n").encode(),mtime=0))
    keys=[value["study_id"]+"/"+r["run_id"] for r in value["runs"]]
    plan={"schema_version":1,"slots_per_gpu":2,"studies":[{"path":str(path.relative_to(root)),"spec_hash":fingerprint(value)}],
          "groups":[{"id":value["study_id"],"gpu_assignment":"allocation/gpu0","run_keys":keys}],"barriers":{},
          "global_gates":{"whole_search":{"path":str(output.relative_to(root)/"gate_receipts/whole_search.json"),
                                           "source_stages":["search","search_selection"],"required_for_stages":["author_cli"],
                                           "required_hosts":[],"all_packet_source_run_keys":[],"external_complete_master_HPO_cohort_required":True},
                          "whole_author":{"path":str(output.relative_to(root)/"gate_receipts/whole_author.json"),
                                           "source_stages":["author_cli"],"required_for_stages":[],"required_hosts":["allocation"],
                                           "all_packet_source_run_keys":sorted(keys),"terminal_failures_preserved":True}},
          "output_root":str(output.relative_to(root)/"execution"),"runtime_author_worker":value["source"]["worker_entrypoint"],
          "scientific_launch_enabled":False,"human_GO_required":True,"approval_receipt_created":False,
          "master_integration_required":True}
    summary={"schema":"disabled-polyformer-author-scheduler-summary-v1","plan_hash":fingerprint(plan),"counts":author_counts(value),
             "author_replication_contract_prepared":True,"source_based_code_path_replication":True,
             "published_task_environment_or_score_match_claim":False,"scientific_training_started":False,"GO_created":False,
             "scientific_launch_enabled":False,"global_master_gate_integration_required":True,
             "standalone_plan_cannot_satisfy_whole_master_HPO_gate":True,"actual_author_fits":0}
    (output/"PLAN.json").write_text(json.dumps(plan,indent=2)+"\n")
    (output/"SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
    return plan,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["project-root","author-packet","runtime-prefix","resources","output"]:
        parser.add_argument("--"+name,required=True)
    args=parser.parse_args();verify_allocation()
    root=Path(args.project_root).resolve()
    profiles=json.loads(confined(root,args.resources).read_text())
    value=build_author_study(root,args.author_packet,args.runtime_prefix,profiles)
    value["source"]["files"].append(receipt(root,args.resources))
    plan,summary=write_packet(root,args.output,value)
    print(canonical({"plan_hash":summary["plan_hash"],"counts":summary["counts"],"scientific_launch_enabled":False,"GO_created":False}))


if __name__=="__main__":
    main()
