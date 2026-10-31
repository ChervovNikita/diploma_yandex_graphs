"""Disabled finite native Polynormer Q/K source studies on new graph providers.

No numerical imports, model/data tensors, fits, remote access or GO actions.
The source recipe/operators and original boundary control stay separately typed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gzip
import json
from pathlib import Path
import subprocess

from adapters.native_polynormer_qk import ARMS, I4, SOURCES, CONTRACT, bound
from compile_native_matrix import bind_custody, file_binding, RUNTIME_FILES
from prepare_node_data import verify_allocation
from spec import canonical, fingerprint, validate, confined

DROPOUT_TRIPLES = ((.5, .5, .5), (.5, .3, .5), (.5, .5, .3))
DOMAIN = [{"learning_rate": lr, "in_dropout": a, "dropout": b, "global_dropout": g}
          for lr in (.0005, .001, .002) for a,b,g in DROPOUT_TRIPLES]
SEARCH_SEEDS = (78101, 78103)
COMPARATIVE_SEEDS = (78201, 78203, 78207)
METHODS = {arm: "native_polynormer_qk_" + arm for arm in (*ARMS, *I4)}
METHODS["original_gnnm"] = "original_RSB_GNNM_on_native_Polynormer"


def make_study(root, custody, data_dir, source, sources, runtime):
    expected = {"wikics": 20, "amazon-photo": 10};dataset = custody["dataset_id"]
    if dataset not in expected or sorted(row["split_id"] for row in custody["splits"]) != list(range(expected[dataset])):
        raise ValueError("Complete new-data WikiCS20/Photo10 masks required")
    splits = {row["split_id"]: row for row in custody["splits"]};files = {row["path"]: row for row in custody["files"]}
    def binding(relative):
        row = files[relative]
        return {"path": str((data_dir/relative).relative_to(root)), "bytes": row["bytes"], "sha256": row["sha256"]}
    def inputs(split, field): return {key: binding(relative) for key, relative in splits[split][field].items()}
    runs = [];selections = {};banks = []
    def common(arm, split, stage):
        operator, kind, members = ARMS[arm]
        return {"polynormer_qk_arm": arm, "polynormer_qk_operator": operator, "source_ownership_kind": kind,
                "architecture": "original_gnnm" if kind == "boundary4" else "internal_be" if members == 4 else "single",
                "backbone": "native_Polynormer", "width": 512, "depth": 7, "global_layers": 2, "heads": 1,
                "local_transition_updates": 100, "members": members, "physical_slow_bodies": 1, "source_optimizer_count": 1,
                "source_optimizer_calls_per_cycle": 1, "source_TRAIN_forwards_per_update": 2*members,
                "source_TRAIN_backwards_per_update": 2*members, "is_original_gnnm": kind == "boundary4",
                "classes": custody["model_graph"]["classes"], "weight_decay": 0., "polynormer_qk_sources": sources,
                "node_inputs": inputs(split,"node_inputs"), "fusion": "mean_logits" if kind == "boundary4" else "mean_probabilities",
                "stage": stage, "source_contract": CONTRACT, "reservation_mib": 32768, "estimated_seconds": 5000000,
                "resource_evidence": "unmeasured positive scheduling placeholder; no admission or ETA",
                "recipe_scope": "exact QK math/streamed F;new providers/domain/seeds/full5000/100local/commonCAL;not author CLI"}
    def add(identifier, kind, arm, split, seed, body, config, parents=()):
        if kind == "model_fit": config["initialization_seed"] = seed+1009*body
        filename = ("native_polynormer_qk.py:make_adapter" if kind == "model_fit" else "search.py:select_configuration" if kind == "selection"
                    else "node_endpoints.py:" + {"ensemble_assembly":"export_bank","calibration":"scalar_calibration","evaluation":"evaluate_bank"}[kind])
        run = {"run_id": identifier, "kind": kind, "method": METHODS[arm], "split": split, "seed": seed, "body": body,
               "adapter": runtime+"/adapters/"+filename, "config": config, "parents": list(parents),
               "budget": {"unit":"optimizer_updates","policy":"full_budget","maximum":5000} if kind == "model_fit" else {},
               "selection": {"metric":"accuracy","direction":"max","ties":"first","evaluation_interval":1,"checkpoint0":False,"patience_steps":5000} if kind == "model_fit" else {}}
        runs.append(run);return run
    for arm in ARMS:
        candidates = []
        for point in DOMAIN:
            key = fingerprint(point)[:20]
            ids = [add("search-"+arm+"-"+key+"-s"+str(seed),"model_fit",arm,0,seed,0,{**common(arm,0,"search"),**point})["run_id"] for seed in SEARCH_SEEDS]
            candidates.append({"config_id":key,"parameters":point,"run_ids":ids})
        selections[arm] = add("select-"+arm,"selection",arm,0,0,0,
                              {"stage":"search_selection","configurations":candidates,"ranking_metric":"accuracy","ranking_direction":"max","reservation_mib":1,"estimated_seconds":1},
                              [identifier for candidate in candidates for identifier in candidate["run_ids"]])["run_id"]
    for split in sorted(splits):
        for seed in COMPARATIVE_SEEDS:
            fits = {}
            for arm in ARMS:
                run = add("compare-sp"+str(split)+"-"+arm+"-s"+str(seed),"model_fit",arm,split,seed,0,
                          {**common(arm,split,"comparative"),**DOMAIN[0],"selected_config_from":selections[arm]},[selections[arm]])
                fits[arm] = [run["run_id"]]
            for arm, single in I4.items():
                first = next(run for run in runs if run["run_id"] == fits[single][0]);fits[arm] = [first["run_id"]]
                for body in (1,2,3):
                    fits[arm].append(add(first["run_id"]+"-i4-b"+str(body),"model_fit",single,split,seed,body,copy.deepcopy(first["config"]),first["parents"])["run_id"])
            for arm, parents in fits.items():
                base = I4.get(arm,arm);prefix = "bank-sp"+str(split)+"-"+arm+"-s"+str(seed)
                endpoints = {key:value["path"] for key,value in inputs(split,"endpoint_inputs").items()};fusion = common(base,split,"comparative")["fusion"]
                export = add(prefix,"ensemble_assembly",arm,split,seed,0,
                             {"stage":"bank_export","fit_parents":parents,"endpoint_inputs":endpoints,"fusion":fusion,"reservation_mib":32768,"estimated_seconds":10000},parents)["run_id"]
                cal = add(prefix+"-cal","calibration",arm,split,seed,0,
                          {"stage":"calibration","export_parent":export,"endpoint_inputs":endpoints,"fusion":fusion,"placement":"after_serving_pool","steps":500,"learning_rate":.01,"reservation_mib":512,"estimated_seconds":1000},[export])["run_id"]
                evaluation = add(prefix+"-evaluate","evaluation",arm,split,seed,0,
                                 {"stage":"evaluation","export_parent":export,"calibration_parent":cal,"endpoint_inputs":endpoints,"fusion":fusion,"reservation_mib":512,"estimated_seconds":1000},[export,cal])["run_id"]
                banks.append({"arm":arm,"split":split,"seed":seed,"fit_parents":parents,"export":export,"calibration":cal,"evaluation":evaluation,
                              "genuine_I4":arm in I4,"body0_reused":arm in I4,"served_views":4 if arm in I4 else ARMS[arm][2]})
    roles = {role:{str(split):inputs(split,"node_inputs" if role in ("train","select") else "endpoint_inputs")[role if role in ("train","select") else role+"_targets"]
                  for split in splits} for role in ("train","select","cal","evaluate")}
    study = {"schema_version":1,"study_id":"native-polynormer-qk-standardized-v1-"+dataset,"stage":"development","dataset":dataset,
             "hypothesis":"Native Polynormer exact pre-sigmoid/kernel QK operators with source single/independent4/shared ownership;equipped,bare and actual original RSB controls.",
             "data":{"task":"node_classification","preprocessing":custody["request"]["graph_policy"],"roles":roles,"prior_exposure":custody["request"]["evaluation_exposure"],
                     "evidence":{"custody_request_hash":custody["request_hash"],"model_graph":custody["model_graph"],"author_protocol_replication":False}},
             "source":source,"runs":runs,"comparisons":[{"references":[METHODS[arm] for arm in ("native_tied_single","native_tied_independent4","native_tied_be_init","bare_native_single","bare_native_independent4","original_gnnm")],
                                                         "candidates":[METHODS[arm] for arm in (*ARMS,*I4) if not arm.startswith("native_tied") and arm not in ("bare_native_single","bare_native_independent4","original_gnnm")],
                                                         "scope":"three exact operator x three ownership kinds;paired all-mask comparative cohorts;no source outcome reuse"}],
             "release":{"policy":"whole_roster","primary_endpoint":"all raw/common-scalar-CAL banks","global_plan_gates_required":True},
             "resources":{"wall_seconds":1000000000,"status":"PROVISIONAL_UNMEASURED_DISABLED_SOURCE_CANDIDATE"}}
    validate(study)
    fits = [run for run in runs if run["kind"]=="model_fit"]
    counts = {"optimizer_starts_by_stage":dict(Counter(run["config"]["stage"] for run in fits)),"atomic_fit_groups":len(fits),
              "active_Adam_owners":len(fits),"physical_slow_bodies":len(fits),"full_budget_optimizer_updates":5000*len(fits),
              "source_TRAIN_forwards":sum(5000*run["config"]["source_TRAIN_forwards_per_update"] for run in fits),
              "source_TRAIN_backwards":sum(5000*run["config"]["source_TRAIN_backwards_per_update"] for run in fits),
              "unstepped_pre_install_Adam_constructor_objects":sum(run["config"]["polynormer_qk_operator"] not in ("bare","original_boundary") for run in fits),
              "selection_jobs":len(selections),"comparative_banks":len(banks),"bank_exports":len(banks),"scalar_CAL_jobs":len(banks),"scalar_CAL_updates":500*len(banks),
              "evaluation_jobs":len(banks),"genuine_I4_banks":sum(bank["genuine_I4"] for bank in banks),"all_job_records":len(runs),
              "original_GNNM_fit_records":sum(run["config"]["is_original_gnnm"] for run in fits),"actual_optimizer_updates":0,"actual_predictive_scores":0}
    return study,counts,banks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root","source-bindings","runtime-package","output"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--custody",required=True,nargs="+")
    args = parser.parse_args();verify_allocation();root = Path(args.project_root).resolve()
    def path(value):
        p = Path(value)
        return confined(root, str(p.resolve().relative_to(root)) if p.is_absolute() else str(p))
    bindings = json.loads(path(args.source_bindings).read_text())
    if set(bindings) != set(SOURCES) or any(bindings[key]["sha256"] != value[1] for key,value in SOURCES.items()):
        raise ValueError("Complete exact source bindings required")
    for row in bindings.values(): bound(root,row)
    runtime = path(args.runtime_package);output = path(args.output)
    if (not runtime.resolve().is_relative_to(root) or runtime.is_relative_to(root/"training_pipeline_v2_20261012")
            or output.exists() or not output.resolve().is_relative_to(root)):
        raise ValueError("Exact frozen runtime and fresh project packet required")
    files = [file_binding(root,runtime/name) for name in (*RUNTIME_FILES,"adapters/native_polynormer_qk.py","compile_native_polynormer_qk_matrix.py","compile_native_matrix.py","prepare_node_data.py")]
    files += list(bindings.values()) + [file_binding(root,path(args.source_bindings))]
    commit = subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    output.mkdir();(output/"studies").mkdir();studies = [];counts = Counter();stages = Counter();groups = []
    for custody_path in map(path,args.custody):
        custody,data_dir,data_bindings = bind_custody(root,custody_path)
        source = {"git_commit":commit,"files":files+data_bindings,"working_tree_fingerprints_authoritative":True,
                  "frozen_runtime_package":str(runtime.relative_to(root)),"author_protocol_replication":False}
        study,work,banks = make_study(root,custody,data_dir,source,bindings,str(runtime.relative_to(root)))
        relative = str((output/"studies"/(study["study_id"]+".json.gz")).relative_to(root))
        (root/relative).write_bytes(gzip.compress((canonical(study)+"\n").encode(),mtime=0));studies.append({"path":relative,"spec_hash":fingerprint(study)})
        stages.update(work["optimizer_starts_by_stage"]);counts.update({key:value for key,value in work.items() if type(value) is int})
        groups.append({"id":study["study_id"],"gpu_assignment":"allocation/gpu0","allowed_workers":["allocation/gpu0"],"maximum_cuda_bytes":32768*1024**2,
                       "estimated_seconds":sum(run["config"]["estimated_seconds"] for run in study["runs"]),"wall_seconds":1000000000,"job_records":len(study["runs"]),
                       "required_study_hash":fingerprint(study),"study_id":study["study_id"]})
    if len(studies)!=2 or counts["atomic_fit_groups"]!=2610 or counts["all_job_records"]!=6680:
        raise ValueError("Exact finite two-dataset original-GNNM-containing roster required")
    summary = {"schema":"native-polynormer-qk-disabled-candidate-v1","counts":{**dict(counts),"optimizer_starts_by_stage":dict(stages)},"source_contract":CONTRACT,
               "numerical_or_resource_qualified":False,"scientific_fits":0,"scientific_GO":False,"source_candidate_only":True}
    plan = {"schema_version":1,"studies":studies,"fit_groups":groups,"worker_capacities_cuda_bytes":{"allocation/gpu0":80*1024**3,"server77/gpu0":24*1024**3,"server77/gpu1":24*1024**3},
            "concurrency_per_gpu":2,"scientific_launch_enabled":False,"human_GO_required":True,"resources_are_provisional":True,
            "plan_scope":"source-candidate finite QK port;not new accuracy evidence or comprehensive-ready"}
    for name,value in (("PLAN.json",plan),("SUMMARY.json",summary)):
        (output/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"scientific_launch_enabled":False,"model_fits":2610,"planned_optimizer_updates":13050000,"numeric_qualification":False}))


if __name__=="__main__": main()
