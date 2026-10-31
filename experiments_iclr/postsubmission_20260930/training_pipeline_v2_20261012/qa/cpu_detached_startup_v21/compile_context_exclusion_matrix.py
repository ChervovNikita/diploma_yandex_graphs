"""Prepare source-exact context mechanism targets and a disabled finite study.

Both CLI modes verify allocation hostname/sole GPU before project file reads.
--prepare-targets opens public GRAPH and TRAIN only, without a model/optimizer.
Normal compilation uses stdlib receipt metadata only and emits no GO or fits.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

from adapters.context_exclusion import (SOURCE_ARMS, SOURCE_FILES, SOURCE_CONTRACT,
                                       objective_module, read_bound, receipt, source_paths, target_payload)
from compile_native_matrix import bind_custody, file_binding, RUNTIME_FILES, resource_for
from prepare_node_data import verify_allocation
from spec import canonical, confined, fingerprint, validate

SEARCH_SEEDS = (72101, 72103)
COMPARATIVE_SEEDS = (72201, 72203, 72207)
I4 = {"ordinary_i4": "ordinary_m1", "factorized_i4": "factorized_m1",
      "genuine_factorized_i4_context_mixture": "factorized_m1_context_mixture"}
METHODS = {arm: "context_positive_exclusion_" + arm for arm in SOURCE_ARMS if "context" in arm}
METHODS.update(ordinary_m1="ordinary_native_single", ordinary_i4="ordinary_genuine_independent4",
               factorized_m1="allmap_factor_single", factorized_i4="allmap_factor_genuine_independent4",
               original_boundary="original_boundary_GNNM",
               genuine_factorized_i4_context_mixture="context_positive_exclusion_genuine_factorized_i4_context_mixture")
DOMAIN = [{"learning_rate": lr, "dropout": dropout}
          for lr in (.0005, .001, .002) for dropout in (.1, .2, .4)]


def validate_custody(custody):
    if custody["dataset_id"] != "wikics" or sorted(r["split_id"] for r in custody["splits"]) != list(range(20)):
        raise ValueError("This first mechanism packet requires the complete20-split WikiCS custody")
    if custody["status"] != "prepared" or custody["benchmark_fits"] != 0 or custody["predictive_scores"] != 0:
        raise ValueError("Actual no-fit/no-score provider preparation required")
    for row in custody["splits"]:
        if set(row["role_class_counts"]["train"]) != {str(i) for i in range(custody["model_graph"]["classes"])}:
            raise ValueError("The source comparison requires TRAIN support for every declared class")
        if any(count < 2 for count in row["role_class_counts"]["train"].values()):
            raise ValueError("Every actual TRAIN class needs a nonself same-class positive")
        if row["evaluate_purpose"] != "development":raise ValueError("No unused confirmation is claimed")


def prepare_target_custody(root, custody_path, source_bindings, output):
    """Called only after main's allocation guard; no heldout tensor decoder."""
    import numpy as np
    import torch
    root = Path(root).resolve();output = Path(output)
    if output.exists():raise ValueError("Preserve previous target custody; fresh output required")
    custody = json.loads(Path(custody_path).read_text());validate_custody(custody)
    paths = source_paths(root, source_bindings);helper = objective_module(paths["objective"])
    directory = Path(custody_path).parent
    file_rows = {r["path"]: r for r in custody["files"]}
    def binding(relative):
        actual = receipt(root, str((directory / relative).resolve().relative_to(root)))
        if {k: actual[k] for k in ("bytes", "sha256")} != {k:file_rows[relative][k] for k in ("bytes", "sha256")}:
            raise ValueError("GRAPH/TRAIN changed after provider custody")
        return actual
    output.mkdir(parents=True)
    entries = []
    for row in custody["splits"]:
        graph_binding, train_binding = (binding(row["node_inputs"][k]) for k in ("graph", "train"))
        graph = torch.load(read_bound(root, graph_binding), map_location="cpu", weights_only=True)
        train = torch.load(read_bound(root, train_binding), map_location="cpu", weights_only=True)
        if set(graph) != {"x", "edge_index"} or set(train) != {"ids", "y"}:
            raise ValueError("Target custodian opens exactly target-free GRAPH and TRAIN")
        ancestry = {"graph": graph_binding, "train": train_binding,
                    "objective_sha256": SOURCE_FILES["objective"][1]}
        pack = target_payload(torch, np, helper, graph, train, ancestry)
        path = output / ("split_" + str(row["split_id"]) + ".pt")
        torch.save(pack, path)
        entries.append({"split_id": row["split_id"], "target": receipt(root, str(path.relative_to(root))),
                        "preparation": pack["preparation"], "tensor_hashes": pack["tensor_hashes"]})
    value = {"schema": "context-exclusion-target-custody-v1", "dataset_id": "wikics",
             "provider_custody": receipt(root, str(Path(custody_path).relative_to(root))),
             "request_hash": custody["request_hash"], "source_bindings": source_bindings,
             "source_contract": SOURCE_CONTRACT, "splits": entries, "model_constructed": False,
             "optimizer_created": False, "SELECT_CAL_TEST_tensors_opened": False,
             "scientific_fits": 0, "GO_created": False}
    (output / "CUSTODY.json").write_text(canonical(value) + "\n")
    return value


def make_study(root, custody, data_dir, targets, sources, source_bindings, profiles, runtime, validator=validate):
    validate_custody(custody);root = Path(root).resolve();data_dir = Path(data_dir).resolve()
    split_rows = {row["split_id"]: row for row in custody["splits"]}
    target_rows = {row["split_id"]: row for row in targets["splits"]}
    if (len(targets["splits"]) != len(split_rows) or set(target_rows) != set(split_rows)
            or targets["source_contract"] != SOURCE_CONTRACT
            or targets["request_hash"] != custody["request_hash"] or targets["source_bindings"] != source_bindings
            or targets["SELECT_CAL_TEST_tensors_opened"] is not False
            or targets["model_constructed"] is not False or targets["optimizer_created"] is not False):
        raise ValueError("Exact shared TRAIN-only target custody required")
    known = {row["path"]: row for row in custody["files"]}
    def binding(relative):
        row = known[relative]
        return {"path": str((data_dir / relative).relative_to(root)), "bytes": row["bytes"], "sha256":row["sha256"]}
    def inputs(split, field):return {k:binding(v) for k, v in split_rows[split][field].items()}
    def endpoint_inputs(split):return {k:v["path"] for k,v in inputs(split,"endpoint_inputs").items()}
    runs = [];selections = {};banks = []
    def add(identifier, kind, arm, split, seed, body, config, parents=()):
        adapter = (runtime + "/adapters/context_exclusion.py:make_adapter" if kind == "model_fit" else
                   runtime + "/adapters/search.py:select_configuration" if kind == "selection" else
                   runtime + "/adapters/node_endpoints.py:" + {"ensemble_assembly":"export_bank", "calibration":"scalar_calibration", "evaluation":"evaluate_bank"}[kind])
        row = {"run_id":identifier,"kind":kind,"method":METHODS[arm],"split":split,"seed":seed,"body":body,
               "adapter":adapter,"config":config,"parents":list(parents),
               "budget":{"unit":"optimizer_updates","policy":"full_budget","maximum":5000} if kind=="model_fit" else {},
               "selection":{"metric":"accuracy","direction":"max","ties":"first","evaluation_interval":1,"checkpoint0":False} if kind=="model_fit" else {}}
        runs.append(row);return row
    def common(split, arm, stage):
        architecture,members,mode=SOURCE_ARMS[arm]
        profile=resource_for(profiles,"wikics","context-exclusion-"+arm)
        return {"context_arm":arm,"architecture":architecture,"members":members,"context_mode":mode,
                "backbone":"SAGE","width":128,"depth":2,"classes":custody["model_graph"]["classes"],
                "weight_decay":0.,"context_source":source_bindings,"node_inputs":inputs(split,"node_inputs"),
                "context_targets":target_rows[split]["target"] if mode!="none" else None,
                "fusion":"mean_logits" if architecture=="original_gnnm" else "mean_probabilities",
                "stage":stage,"is_original_gnnm":architecture=="original_gnnm","source_optimizer_count":1,
                "reservation_mib":profile["reservation_mib"],"estimated_seconds":profile["estimated_seconds_per_update"]*5000,
                "recipe_scope":"exact_context_mechanism_standardized_roles_full5000"}
    for arm in SOURCE_ARMS:
        candidates=[]
        for point in DOMAIN:
            config_id=fingerprint(point)[:20]
            ids=[add("search-"+arm+"-"+config_id+"-s"+str(seed),"model_fit",arm,0,seed,0,
                     {**common(0,arm,"search"),**point})["run_id"] for seed in SEARCH_SEEDS]
            candidates.append({"config_id":config_id,"parameters":point,"run_ids":ids})
        selections[arm]=add("select-"+arm,"selection",arm,0,0,0,
            {"stage":"search_selection","configurations":candidates,"ranking_metric":"accuracy",
             "ranking_direction":"max","reservation_mib":1,"estimated_seconds":1},
            [i for c in candidates for i in c["run_ids"]])["run_id"]
    for split in sorted(split_rows):
        for seed in COMPARATIVE_SEEDS:
            fits={}
            for arm in SOURCE_ARMS:
                first=add("compare-sp"+str(split)+"-"+arm+"-s"+str(seed),"model_fit",arm,split,seed,0,
                    {**common(split,arm,"comparative"),**DOMAIN[0],"selected_config_from":selections[arm]},[selections[arm]])
                fits[arm]=[first["run_id"]]
            for arm,member in I4.items():
                first=next(r for r in runs if r["run_id"]==fits[member][0]);fits[arm]=[first["run_id"]]
                for body in (1,2,3):
                    identifier=first["run_id"]+"-i4-b"+str(body)
                    fits[arm].append(add(identifier,"model_fit",member,split,seed,body,copy.deepcopy(first["config"]),first["parents"])["run_id"])
            for arm,parents in fits.items():
                member=I4.get(arm,arm);fusion=common(split,member,"comparative")["fusion"]
                prefix="bank-sp"+str(split)+"-"+arm+"-s"+str(seed)
                endpoints=endpoint_inputs(split)
                profile=resource_for(profiles,"wikics","context-exclusion-"+member)
                export=add(prefix,"ensemble_assembly",arm,split,seed,0,{"stage":"bank_export","fit_parents":parents,
                    "endpoint_inputs":endpoints,"fusion":fusion,"reservation_mib":profile["reservation_mib"],"estimated_seconds":1},parents)["run_id"]
                cal=add(prefix+"-cal","calibration",arm,split,seed,0,{"stage":"calibration","export_parent":export,
                    "endpoint_inputs":endpoints,"fusion":fusion,"placement":"after_serving_pool","steps":500,
                    "learning_rate":.01,"reservation_mib":512,"estimated_seconds":5},[export])["run_id"]
                evaluation=add(prefix+"-evaluate","evaluation",arm,split,seed,0,{"stage":"evaluation","export_parent":export,
                    "calibration_parent":cal,"endpoint_inputs":endpoints,"fusion":fusion,"reservation_mib":512,"estimated_seconds":5},[export,cal])["run_id"]
                banks.append({"arm":arm,"split":split,"seed":seed,"fit_parents":parents,"export":export,"calibration":cal,
                              "evaluation":evaluation,"genuine_I4":arm in I4,"body0_reused":arm in I4})
    roles={role:{str(split):inputs(split,"node_inputs" if role in {"train","select"} else "endpoint_inputs")[role if role in {"train","select"} else role+"_targets"]
                 for split in split_rows} for role in ("train","select","cal","evaluate")}
    study={"schema_version":1,"study_id":"context-exclusion-v2-wikics","stage":"development","dataset":"wikics",
        "hypothesis":"Exact fixed context target/support allocation on source SAGE with competent ordinary/equipped/independent and original GNNM references.",
        "data":{"task":"node_classification","preprocessing":custody["request"]["graph_policy"],"roles":roles,
                "prior_exposure":custody["request"]["evaluation_exposure"],"evidence":{"custody_request_hash":custody["request_hash"],
                "model_graph":custody["model_graph"],"source_contract":SOURCE_CONTRACT,"source_exact_training_recipe":False}},
        "source":sources,"runs":runs,"comparisons":[{"references":["original_boundary","ordinary_m1","ordinary_i4","factorized_m1","factorized_i4"],
            "candidates":[a for a in METHODS if "context" in a],"scope":"all20 prepared development splits; three fixed comparative seeds"}],
        "release":{"policy":"whole_roster","primary_endpoint":"closed raw and scalar-calibrated banks","global_plan_gates_required":True},
        "resources":{"wall_seconds":max(p["wall_seconds"] for p in profiles.values()),"status":"PROVISIONAL_DISABLED_REQUIRES_FULL_INPUT_RESOURCE_QA_AND_GO"}}
    validator(study)
    fit_rows=[r for r in runs if r["kind"]=="model_fit"]
    starts=Counter(r["config"]["stage"] for r in fit_rows)
    counts={"optimizer_starts_by_stage":dict(starts),"atomic_fit_groups":len(fit_rows),"paid_optimizer_objects":len(fit_rows),
            "physical_body_instances":len(fit_rows),"full_budget_optimizer_updates":5000*len(fit_rows),
            "source_target_preparations":len(split_rows),"search_selection_jobs":len(selections),"comparative_banks":len(banks),
            "bank_exports":len(banks),"scalar_CAL_fits":len(banks),"evaluation_jobs":len(banks),
            "genuine_I4_banks":sum(b["genuine_I4"] for b in banks),"reused_I4_body0":sum(b["body0_reused"] for b in banks),
            "source_acquisition_fits":0,"continuations":0,"extensions":0,"all_run_records":len(runs),
            "actual_optimizer_updates":0,"actual_predictive_scores":0}
    return study,counts,banks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ("project-root","custody","source-bindings","output"):parser.add_argument("--"+field,required=True)
    parser.add_argument("--prepare-targets",action="store_true")
    for field in ("targets-custody","runtime-package","resources","source-commit"):parser.add_argument("--"+field)
    parser.add_argument("--fixture-receipts",nargs="+")
    args=parser.parse_args();verify_allocation()  # Before any project file read.
    root=Path(args.project_root).resolve()
    def path(value):
        v=Path(value)
        return confined(root,str(v.resolve().relative_to(root)) if v.is_absolute() else str(v))
    custody_path,bindings_path,output=map(path,(args.custody,args.source_bindings,args.output))
    bindings=json.loads(bindings_path.read_text());source_paths(root,bindings)
    if args.prepare_targets:
        result=prepare_target_custody(root,custody_path,bindings,output)
        print(canonical({"prepared_target_splits":len(result["splits"]),"models":0,"fits":0,"GO":False}));return
    if not all((args.targets_custody,args.runtime_package,args.resources,args.fixture_receipts)):
        raise ValueError("Compilation requires prepared targets, frozen runtime, explicit resources and QA receipts")
    if output.exists():raise ValueError("Fresh disabled packet directory required")
    runtime=path(args.runtime_package)
    if runtime.is_relative_to(root/"training_pipeline_v2_20261012"):
        raise ValueError("Use a frozen runtime outside the live package")
    resource_path=path(args.resources);resources=json.loads(resource_path.read_text())
    capacities=resources["worker_capacities_cuda_bytes"]
    if set(capacities)!={"allocation/gpu0"} or type(capacities["allocation/gpu0"]) is not int or capacities["allocation/gpu0"]<1:
        raise ValueError("This source packet is allocation-only; no77 assignment or connection")
    custody,data_dir,data_bindings=bind_custody(root,custody_path);validate_custody(custody)
    targets_path=path(args.targets_custody);targets=json.loads(targets_path.read_text())
    if targets["provider_custody"]!=file_binding(root,custody_path):raise ValueError("Target/provider custody ancestry differs")
    target_bindings=[file_binding(root,targets_path)]
    for row in targets["splits"]:
        if file_binding(root,path(row["target"]["path"]))!=row["target"]:raise ValueError("Target pack bytes changed")
        target_bindings.append(row["target"])
    runtime_files=list(RUNTIME_FILES)+["adapters/context_exclusion.py","compile_context_exclusion_matrix.py",
                                      "compile_native_matrix.py","prepare_node_data.py"]
    source_files=[file_binding(root,runtime/name) for name in runtime_files]
    source_files += list(bindings.values())+data_bindings+target_bindings+[file_binding(root,bindings_path),file_binding(root,resource_path)]
    source_files += [file_binding(root,path(f)) for f in args.fixture_receipts]
    source_files=list({r["path"]:r for r in source_files}.values())
    commit=args.source_commit or subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    sources={"git_commit":commit,"files":source_files,"working_tree_fingerprints_authoritative":True,
             "frozen_runtime_package":str(runtime.relative_to(root)),"official_author_protocol_replication":False}
    schema_spec=importlib.util.spec_from_file_location("context_frozen_schema",runtime/"spec.py")
    schema=importlib.util.module_from_spec(schema_spec);schema_spec.loader.exec_module(schema)
    study,counts,banks=make_study(root,custody,data_dir,targets,sources,bindings,resources["profiles"],str(runtime.relative_to(root)),schema.validate)
    if any(r["config"]["reservation_mib"]*1024**2>capacities["allocation/gpu0"] for r in study["runs"]):
        raise ValueError("A declared fit/endpoint reservation exceeds allocation capacity")
    prefix=str(output.relative_to(root));spec_path=prefix+"/studies/"+study["study_id"]+".json.gz"
    groups=[{"id":study["study_id"],"gpu_assignment":"allocation/gpu0","allowed_workers":["allocation/gpu0"],
             "maximum_cuda_bytes":max(r["config"]["reservation_mib"] for r in study["runs"])*1024**2,
             "estimated_seconds":sum(r["config"]["estimated_seconds"] for r in study["runs"]),
             "run_keys":[study["study_id"]+"/"+r["run_id"] for r in study["runs"]]}]
    plan={"schema_version":1,"slots_per_gpu":2,"torch_threads_per_fit":2,"output_root":prefix+"/execution",
          "runtime_launcher":str((runtime/"launcher.py").relative_to(root)),"runtime_package":str(runtime.relative_to(root)),
          "worker_capacities_cuda_bytes":capacities,"studies":[{"path":spec_path,"spec_hash":fingerprint(study)}],
          "groups":groups,"global_gates":{},"barriers":{},"scientific_launch_enabled":False,"human_GO_required":True,
          "host_bindings":{"allocation":{"status":"verified_at_compilation","hostname":"anogena-2-0",
                            "gpu_uuids":["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]}}}
    for name,stages,required in [("whole_search",["search","search_selection"],["comparative"]),
                                ("whole_comparative",["comparative"],["bank_export","calibration","evaluation"]),
                                ("whole_calibration",["calibration"],["evaluation"])]:
        plan["global_gates"][name]={"path":prefix+"/gate_receipts/"+name+".json","source_stages":stages,
                                  "required_for_stages":required,"required_hosts":["allocation"]}
    output.mkdir(parents=True);(output/"studies").mkdir()
    (root/spec_path).write_bytes(gzip.compress((canonical(study)+"\n").encode(),mtime=0))
    summary={"schema":"context-exclusion-disabled-summary-v1","plan_hash":fingerprint(plan),"counts":counts,
             "banks":banks,"source_contract":SOURCE_CONTRACT,"domain":DOMAIN,"source_canonical_methods":7,
             "source_mechanism_port":True,"source_exact_recipe":False,"full_249_coverage":False,
             "scientific_training_started":False,"GO_created":False,"status":"DISABLED_PROVISIONAL_REQUIRES_ROOT_QA_MASTER_AND_GO"}
    (output/"PLAN.json").write_text(canonical(plan)+"\n");(output/"SUMMARY.json").write_text(canonical(summary)+"\n")
    print(canonical({"status":summary["status"],"counts":counts,"fits_executed":0,"GO":False}))


if __name__=="__main__":main()
