"""Fresh molecular v4 from immutable v3 metadata plus a boundary GNNM port."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gzip
import json
import math
from pathlib import Path

from adapters.molecular import validate_config,validate_role_authorities,validate_independent_roster
from adapters.molecular_boundary import ARM,validate_boundary_config
from author_cli_spec import receipt,verify_receipt
from compile_molecular_matrix import make_plan,SEARCH_SEEDS,COMPARATIVE_SEEDS
from prepare_node_data import verify_allocation
from spec import canonical,confined,fingerprint,validate

BOUNDARY_PIN=(10977,"07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303")
NATIVE_TRAIN_PIN=(11240,"b1d748e50c854802b93f4d4fa7485f096daa1cd36d3dd3bd1c89fbfa63cbd124")


def cost_contract(config,run,graphs):
    batch=config["training"]["batch_size"];batches=math.ceil(graphs/batch)
    return {"train_graphs":graphs,"batch_size":batch,"drop_last":False,"last_batch_graphs":graphs%batch or batch,
            "minibatches_per_epoch":batches,"optimizer_calls_per_minibatch":1,
            "maximum_optimizer_step_calls":run["budget"]["maximum"]*batches,
            "train_pack_receipt":config["molecular_inputs"]["train"],
            "law":"full_TRAIN_shuffle_tensor_split_include_tail_one_optimizer_step_per_minibatch"}


def append_boundary_reference(original,runtime_prefix,boundary_source,source_files):
    value=copy.deepcopy(original);value["study_id"]=original["study_id"].replace("molecular-v2-","molecular-v4-",1)
    old_prefix=value["runs"][0]["adapter"].split("/adapters/")[0]
    for row in value["runs"]:
        if row["adapter"].startswith(old_prefix+"/"):
            row["adapter"]=runtime_prefix+row["adapter"][len(old_prefix):]
    source=value["source"]
    source["v3_ancestry"]={"study_id":original["study_id"],"spec_hash":fingerprint(original),"existing_model_learning_contracts_unchanged":True}
    source["files"]=[r for r in source["files"] if not r["path"].startswith(old_prefix+"/")]
    source["files"]+=source_files+[boundary_source]
    unique={r["path"]:r for r in source["files"]}
    for row in source["files"]:
        if unique[row["path"]]!=row:raise ValueError("Conflicting molecular v4 immutable source binding")
    source["files"]=list(unique.values())
    graphs=value["data"]["evidence"]["pack_descriptives"]["train"]["graphs"]
    for row in value["runs"]:
        if row["kind"]=="model_fit":
            row["config"]["molecular_cost"]=cost_contract(row["config"],row,graphs)
            row["config"]["train_minibatches_per_epoch"]=row["config"]["molecular_cost"]["minibatches_per_epoch"]
    template=next(r for r in value["runs"] if r["kind"]=="model_fit" and r["config"]["stage"]=="search" and r["config"]["arm"]=="single")
    selection=next(r for r in value["runs"] if r["kind"]=="selection" and r["method"]=="molecular_single")
    epochs=template["budget"]["maximum"];profile=value["resources"]["profile"]
    runs=value["runs"]
    def add(identifier,kind,seed,config,parents=()):
        row={"run_id":identifier,"kind":kind,"method":"molecular_"+ARM,"split":0,"seed":seed,"body":0,
             "adapter":runtime_prefix+("/adapters/molecular_boundary.py:make_adapter" if kind=="model_fit" else "/adapters/search.py:select_configuration"),
             "config":config,"budget":copy.deepcopy(template["budget"]) if kind=="model_fit" else {},
             "selection":copy.deepcopy(template["selection"]) if kind=="model_fit" else {},"parents":list(parents)}
        runs.append(row);return row
    def fit_config(seed,stage,parameter):
        config=copy.deepcopy(template["config"]);config.update(copy.deepcopy(parameter))
        config.update(arm=ARM,members=4,is_original_gnnm=True,boundary_source=boundary_source,
                      head_policy="source_boundary_scalar_graph_adaptation",stage=stage,
                      initialization_seed=seed*10,dropout_seed=seed*10+300001,shuffle_seed=seed*10+600001,
                      training={"batch_size":128,"eval_batch_size":128,"views_per_update":1},
                      estimated_seconds=profile["estimated_seconds_per_epoch"]*epochs*4)
        config["molecular_cost"]=cost_contract(config,template,graphs)
        config["train_minibatches_per_epoch"]=config["molecular_cost"]["minibatches_per_epoch"]
        return config
    candidates=[]
    for old in selection["config"]["configurations"]:
        parameter=copy.deepcopy(old["parameters"]);parameter["optimizer"].update(name="AdamW",weight_decay=0)
        parameter["contrastive"].update(alignment_weight=0,residual_weight=0,temperature=.2)
        identifier=fingerprint(parameter)[:20];fits=[]
        for seed in SEARCH_SEEDS:
            row=add("search-"+ARM+"-"+identifier+"-s"+str(seed)+"-b0","model_fit",seed,fit_config(seed,"search",parameter))
            fits.append(row["run_id"])
        candidates.append({"config_id":identifier,"parameters":parameter,"run_ids":fits})
    if len(candidates)!=40 or len({row["config_id"] for row in candidates})!=40:
        raise ValueError("Original boundary graph port retains40 effective native-shape/LR configurations")
    selector=add("select-"+ARM,"selection",0,{"stage":"search_selection","ranking_metric":"rocauc","ranking_direction":"max",
        "configurations":candidates,"reservation_mib":256,"estimated_seconds":1},[i for c in candidates for i in c["run_ids"]])
    banks=[]
    for seed in COMPARATIVE_SEEDS:
        config=fit_config(seed,"comparative",candidates[0]["parameters"]);config["selected_config_from"]=selector["run_id"]
        fit=add("compare-"+ARM+"-s"+str(seed)+"-b0","model_fit",seed,config,[selector["run_id"]])
        endpoint={"endpoint_inputs":value["data"]["evidence"]["endpoint_inputs"],"fusion":"mean_logits","target_policy":"complete_binary"}
        bank_id="bank-"+ARM+"-s"+str(seed)
        export=add(bank_id,"ensemble_assembly",seed,{**endpoint,"stage":"bank_export","fit_parents":[fit["run_id"]],
            "reservation_mib":profile["reservation_mib"],"estimated_seconds":profile["estimated_export_seconds"]*4},[fit["run_id"]])
        export["adapter"]=runtime_prefix+"/adapters/molecular_endpoints.py:export_bank"
        cal=add(bank_id+"-cal","calibration",seed,{**endpoint,"stage":"calibration","export_parent":bank_id,
            "placement":"after_serving_pool","steps":500,"learning_rate":.01,"reservation_mib":512,"estimated_seconds":profile["estimated_CAL_seconds"]},[bank_id])
        cal["adapter"]=runtime_prefix+"/adapters/molecular_endpoints.py:scalar_calibration"
        evaluation=add(bank_id+"-evaluate","evaluation",seed,{**endpoint,"stage":"evaluation","export_parent":bank_id,
            "calibration_parent":cal["run_id"],"reservation_mib":512,"estimated_seconds":profile["estimated_evaluation_seconds"]},[bank_id,cal["run_id"]])
        evaluation["adapter"]=runtime_prefix+"/adapters/molecular_endpoints.py:evaluate_bank"
        banks.append({"arm":ARM,"seed":seed,"fit_parents":[fit["run_id"]],"export":bank_id,"calibration":cal["run_id"],
                      "evaluation":evaluation["run_id"],"routes":4,"independently_paid_fits":0,"own_selected_checkpoints":1,"body0_reused":False})
    for comparison in value["comparisons"]:
        for key in ["references","candidates"]:
            comparison[key]=[name if name.startswith("molecular_") else "molecular_"+name for name in comparison.get(key,[])]
        comparison["references"].append("molecular_"+ARM)
        comparison["original_GNNM_reference"]="source-bound original boundary blocks/native BCE/4 AdamW0/mean logits; declared new binary graph-task adaptation"
    value["hypothesis"]+=" Original boundary GNNM is included as an explicitly declared new graph-task reference."
    source["graph_pooling_correction"]="actual source serves global_mean_pool; global_add_pool is only in virtual-node updates; catalogue add-pool wording is superseded by bound source"
    source["molecular_step_cost_law"]={"train_graphs":graphs,"batch_size":128,"minibatches_per_epoch":math.ceil(graphs/128),
        "last_batch_graphs":graphs%128 or 128,"drop_last":False,"optimizer_calls_per_minibatch":1,"scalar_CAL_calls_excluded":True}
    for row in runs:
        if row["kind"]=="model_fit":
            (validate_boundary_config if row["config"]["arm"]==ARM else validate_config)(row["config"],row)
            validate_role_authorities(value,row["config"]);validate_independent_roster(value,row)
    validate(value)
    fits=[r for r in runs if r["kind"]=="model_fit"];counts=Counter(r["kind"] for r in runs)
    detail={"optimizer_starts_by_stage":dict(Counter(r["config"]["stage"] for r in fits)),
            "initial_training_segments":len(fits),"continuation_segments":0,"extension_segments":0,
            "full_budget_epochs":len(fits)*epochs,"known_optimizer_step_calls":sum(r["config"]["molecular_cost"]["maximum_optimizer_step_calls"] for r in fits),
            "source_stage_fits":0,"author_replication_fits":0,"original_boundary_graph_fit_count":83,
            "search_selection_jobs":counts["selection"],"bank_exports":counts["ensemble_assembly"],"scalar_CAL_fits":counts["calibration"],
            "evaluation_jobs":counts["evaluation"],"all_run_records":len(runs),"train_cost_law":source["molecular_step_cost_law"]}
    return value,detail,banks,{ARM:{"configuration_count":40,"search_seed_ids":list(SEARCH_SEEDS),"source_one_own_view_per_member":True}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["project-root","v3-packet","runtime-prefix","boundary-source","output"]:parser.add_argument("--"+name,required=True)
    args=parser.parse_args();verify_allocation();root=Path(args.project_root).resolve()
    old=confined(root,args.v3_packet);old_plan=json.loads((old/"matrix/PLAN.json").read_text());old_summary=json.loads((old/"matrix/SUMMARY.json").read_text())
    if fingerprint(old_plan)!=old_summary["plan_hash"] or old_summary.get("GO_created") or old_summary.get("scientific_training_started"):
        raise ValueError("Only the disabled frozen v3 plan may be expanded")
    runtime=confined(root,args.runtime_prefix)
    source_files=[receipt(root,path.relative_to(root)) for path in sorted(runtime.rglob('*')) if path.is_file() and path.suffix in {".py",".json",".md"}]
    source_files += [receipt(root,(old/"matrix/PLAN.json").relative_to(root)),receipt(root,(old/"matrix/SUMMARY.json").relative_to(root))]
    boundary=receipt(root,args.boundary_source)
    if (boundary["bytes"],boundary["sha256"])!=BOUNDARY_PIN:raise ValueError("Original BEBlock source pin changed")
    native_train=receipt(root,"source_snapshot/run_tabm.py")
    if (native_train["bytes"],native_train["sha256"])!=NATIVE_TRAIN_PIN:raise ValueError("Original member-own training source pin changed")
    source_files.append(native_train)
    prepared=[]
    for row in old_plan["studies"]:
        path=confined(root,row["path"]);original=json.loads(gzip.decompress(path.read_bytes()))
        if fingerprint(original)!=row["spec_hash"]:raise ValueError("Frozen v3 study changed")
        # Rehash every actual custody/source receipt; no tensor is decoded.
        for bound in original["source"]["files"]:verify_receipt(root,bound)
        for bound in original["data"]["roles"].values():verify_receipt(root,bound)
        source_files_for_study=source_files+[receipt(root,path.relative_to(root))]
        prepared.append(append_boundary_reference(original,args.runtime_prefix,boundary,source_files_for_study))
    output=confined(root,args.output)
    if output.exists():raise ValueError("Preserve immutable v3 and use fresh v4 output")
    output.mkdir();(output/"studies").mkdir()
    plan=make_plan(prepared,args.output,old_plan["host_bindings"])
    for value,*_ in prepared:
        path=output/"studies"/(value["study_id"]+".json.gz");path.write_bytes(gzip.compress((canonical(value)+"\n").encode(),mtime=0))
        plan["studies"].append({"path":str(path.relative_to(root)),"spec_hash":fingerprint(value)})
    summary={"schema":"molecular-boundary-v4-disabled-preparation-v1","plan_hash":fingerprint(plan),"per_study":[{"study_id":s["study_id"],"counts":c,"original_banks":b,"new_domains":d} for s,c,b,d in prepared],
        "total_model_fits":sum(c["initial_training_segments"] for _,c,_,_ in prepared),"total_run_records":sum(c["all_run_records"] for _,c,_,_ in prepared),
        "full_budget_epochs":sum(c["full_budget_epochs"] for _,c,_,_ in prepared),"known_optimizer_step_calls":sum(c["known_optimizer_step_calls"] for _,c,_,_ in prepared),
        "CAL_optimizer_calls_excluded_from_backbone_total":True,"source_based_original_GNNM_graph_adaptation":True,"published_author_protocol_replication":False,
        "v3_preserved_immutable":True,"v3_data_reused_by_exact_hash":True,"heldout_target_tensors_decoded":False,
        "scientific_training_started":False,"GO_created":False,"scientific_launch_enabled":False,"resource_status":"PROVISIONAL_REQUIRES_ROOT_QUALIFICATION"}
    (output/"PLAN.json").write_text(json.dumps(plan,indent=2)+"\n");(output/"SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(canonical({k:v for k,v in summary.items() if k!="per_study"}))


if __name__=="__main__":main()
