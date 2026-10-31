"""Separate disabled CMCL/private-Adam/CORE packets; exact source ownership."""
from collections import Counter
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import random
import time

from adapters import private_cmcl as cmcl, private_adam as adam, masked_core as core, private_polyformer_controls as controls
from adapters import private_polyformer_source as p, polyformer
from compile_native_matrix import bind_custody,file_binding,resource_for
from prepare_node_data import verify_allocation
from spec import canonical,confined,fingerprint,validate

SEARCH_SEEDS=(75101,75103)
COMPARATIVE_SEEDS=(75201,75203,75207)
REWIRE_SOURCE={"path":"combination_masked_context_pubmed_stage2_source_20261010_v1/rewire.py","bytes":5363,
    "sha256":"7a96eabf7de1ee58da063a1c3dad5ca6712aa2a2063ac41b7316ca0f34be59c8"}
REWIRE={"double_edge_swap_seed":190121,"accepted_swaps_per_nonself_undirected_edge":10,"maximum_proposals_per_requested_swap":100}
FAMILIES=("private_cmcl","private_adam","masked_core")
CANONICAL={"private_cmcl":{"cmcl-"+name:"private_cmcl_"+name for name in cmcl.CONDITIONS},
    "private_adam":{"normalized":"private_adam_normalization_own4_m"},
    "masked_core":{"core-"+name:"masked_context_reconstruction_"+name for name in core.CONDITIONS if name not in ("shared4_own","single_native")}}


def source_roster():
    return list({r["path"]:r for r in [*cmcl.SOURCE.values(),*core.SOURCE.values(),adam.SOURCE,
        *polyformer.SOURCE_FILES.values(),controls.boundary.BOUNDARY_SOURCE,REWIRE_SOURCE]}.values())


def family_arms(family):
    if family=="private_cmcl":return ["cmcl-"+name for name in cmcl.CONDITIONS]+["ordinary_m1","factorized_m1","original_boundary"]
    if family=="private_adam":return ["normalized",*controls.CONDITIONS]
    return ["core-"+name for name in core.CONDITIONS]+["factorized_m1","original_boundary"]


def prepare_core_graph(root,custody_path,output):
    """Decode exactly GRAPH; no TRAIN/SELECT/CAL/evaluate labels or model."""
    import torch
    root=Path(root).resolve();output=Path(output)
    if output.exists():raise FileExistsError("Preserve prior graph-generation attempt")
    custody=json.loads(Path(custody_path).read_text());directory=Path(custody_path).parent
    graphs={row["node_inputs"]["graph"] for row in custody["splits"]}
    if len(graphs)!=1:raise ValueError("One common provider public graph required")
    relative=next(iter(graphs));declared=next(r for r in custody["files"] if r["path"]==relative)
    binding=p.receipt(root,str((directory/relative).relative_to(root)))
    if {k:binding[k] for k in ("bytes","sha256")}!={k:declared[k] for k in ("bytes","sha256")}:
        raise ValueError("Provider GRAPH changed")
    graph=torch.load(p.bound(root,binding),map_location="cpu",weights_only=True)
    if set(graph)!={"x","edge_index"}:raise ValueError("Target-free exact GRAPH authority required")
    env={"random":random,"time":time,"REWIRE":REWIRE}
    p.definitions(root,REWIRE_SOURCE,("graph_sets","swap_graph"),env)
    output.mkdir(parents=True)
    started=time.monotonic()
    try:
        ordered,metadata=env["swap_graph"](graph["edge_index"].t().tolist(),n=len(graph["x"]),deadline=started+600)
        edges=torch.tensor(ordered,dtype=torch.long).t().contiguous()
        check={"torch":torch}
        p.definitions(root,core.SOURCE["method"],("verify_auxiliary_edges",),check)
        check["verify_auxiliary_edges"](graph["edge_index"],edges,len(graph["x"]))
        path=output/"rewired.pt";torch.save({"edge_index":edges},path);path.chmod(0o444)
        eligible=int((graph["x"].norm(dim=-1)>0).sum())
        value={"schema":"masked-core-new-provider-graph-custody-v1","dataset_id":custody["dataset_id"],
            "GRAPH":binding,"generator_source":REWIRE_SOURCE,"auxiliary_graph":p.receipt(root,str(path.relative_to(root))),
            "generation_seconds":time.monotonic()-started,"algorithm":"exact_pinned_double_edge_swap","metadata":metadata,
            "eligible_nonzero_feature_nodes":eligible,"all_four_32_negative_populations_supported":eligible>=132,
            "one_common_graph_for_all_optimizer_seeds":True,"TRAIN_SELECT_CAL_evaluate_targets_opened":False,
            "models_constructed":0,"optimizers_created":0,"scientific_fits":0,"GO_created":False}
        (output/"CUSTODY.json").write_text(canonical(value)+"\n");(output/"CUSTODY.json").chmod(0o444)
        return value
    except BaseException as error:
        (output/"FAILURE.json").write_text(canonical({"error_type":type(error).__name__,"error":str(error),
            "generation_seconds":time.monotonic()-started,"alternate_seed_or_graph":False,"scientific_fits":0})+"\n")
        raise


def make_study(root,custody,data_dir,sources,profiles,runtime,family,graph_custody=None,policy=None,validator=validate):
    dataset=custody["dataset_id"]
    expected={"wikics":20,"amazon-photo":10}
    if family not in FAMILIES or dataset not in expected or sorted(r["split_id"] for r in custody["splits"])!=list(range(expected[dataset])):
        raise ValueError("Exact family and complete WikiCS20/Photo10 split roster required")
    root,data_dir=Path(root).resolve(),Path(data_dir).resolve();splits={r["split_id"]:r for r in custody["splits"]};known={r["path"]:r for r in custody["files"]}
    def inputs(split,field):
        return {key:{**known[relative],"path":str((data_dir/relative).relative_to(root))} for key,relative in splits[split][field].items()}
    arms=family_arms(family)
    ordinary="core-single_native" if family=="masked_core" else "ordinary_m1"
    independent={"ordinary_i4":ordinary,"factorized_i4":"factorized_m1"}
    methods={arm:CANONICAL[family].get(arm,{"core-single_native":"source_polyformer_native_single","core-shared4_own":"source_polyformer_shared4_own",
        "ordinary_m1":"source_polyformer_native_single","factorized_m1":"source_polyformer_factorized_single","original_boundary":"source_PolyFormer_original_boundary_GNNM","shared4_own":"source_polyformer_shared4_own"}.get(arm,arm)) for arm in arms}
    methods.update(ordinary_i4="source_polyformer_genuine_own_selected_native_I4",factorized_i4="source_polyformer_genuine_own_selected_factorized_I4")
    runs=[];selections={};banks=[]
    def config(arm,split,stage,point):
        profile=resource_for(profiles,dataset,family+"-"+arm)
        value={"classes":custody["model_graph"]["classes"],"learning_rate":point["learning_rate"],"dropout":point["dropout"],
            "native":{**p.NATIVE,"dropout":point["dropout"]},"node_inputs":inputs(split,"node_inputs"),"fusion":"mean_probabilities",
            "stage":stage,"is_original_gnnm":False,"physical_bank_reference":False,"genuine_own_selected_I4":False,
            "source_optimizer_count":1,"physical_slow_bodies":1,"source_optimizer_calls_per_cycle":1,
            "reservation_mib":profile["reservation_mib"],"estimated_seconds":profile["estimated_seconds_per_update"]*5000,
            "provider_role_horizon_adaptation":"new_roles_full5000_first_SELECT_accuracy;source_horizon2000_patience250_not_replicated"}
        if arm.startswith("cmcl-"):
            value.update(architecture="private_cmcl",cmcl_condition=arm[5:],cmcl_source=cmcl.SOURCE,members=4,
                identity={"split":split,"input_custody_sha256":value["node_inputs"]["graph"]["sha256"],
                    "TRAIN_custody_sha256":value["node_inputs"]["train"]["sha256"],"VALID_custody_sha256":value["node_inputs"]["select"]["sha256"]})
        elif arm.startswith("core-"):
            name=arm[5:];bodies,members,owners,_,_=core.LAW[name]
            value.pop("source_optimizer_calls_per_cycle")
            value.update(architecture="masked_core",core_condition=name,core_source=core.SOURCE,members=members,physical_slow_bodies=bodies,
                source_optimizer_count=owners,optimizer_calls_per_cycle=owners,macro_cycle_law="one_complete_old_state_all_owner_cycle",
                source_independent4_selector="one_common_bank_SELECT" if name=="independent4_native" else None,
                historical_stage2_activation="not_activated;new_unconditional_development_comparison" if name in core.PENDING_STAGE2 or name=="single_native" else "historical_stage1_recipe_new_data_port",
                new_comparison_policy=policy,auxiliary_graph=graph_custody["auxiliary_graph"] if name=="shared4_core_rewired" else None,
                auxiliary_graph_custody=graph_custody if name=="shared4_core_rewired" else None)
        elif arm=="normalized":
            value.update(architecture="private_adam",core_condition="shared4_own",core_source=core.SOURCE,private_adam_source=adam.SOURCE,
                private_adam_normalization="private_R_S_epsilon_and_weight_decay_divided_by_M",members=4)
        else:
            value.update(control_condition=arm,core_condition="shared4_own" if arm=="shared4_own" else "single_native",core_source=core.SOURCE,
                architecture={"ordinary_m1":"single","factorized_m1":"factorized_single","original_boundary":"original_gnnm","shared4_own":"internal_be"}[arm],
                members=4 if arm in ("original_boundary","shared4_own") else 1)
            if arm=="original_boundary":value.update(fusion="mean_logits",is_original_gnnm=True,boundary_source=controls.boundary.BOUNDARY_SOURCE,
                source_boundary_sites=["lin1","lin3"],optimizer="source_boundary_grouped_AdamW_zero_decay")
        return value
    def add(identifier,kind,arm,split,seed,body,cfg,parents=()):
        if kind=="model_fit":module="private_cmcl" if arm.startswith("cmcl-") else "masked_core" if arm.startswith("core-") else "private_adam" if arm=="normalized" else "private_polyformer_controls"
        entry="adapters/"+module+".py:make_adapter" if kind=="model_fit" else "adapters/search.py:select_configuration" if kind=="selection" else "adapters/node_endpoints.py:"+{"ensemble_assembly":"export_bank","calibration":"scalar_calibration","evaluation":"evaluate_bank"}[kind]
        row={"run_id":identifier,"kind":kind,"method":methods[arm],"split":split,"seed":seed,"body":body,"adapter":runtime+"/"+entry,
            "config":cfg,"parents":list(parents),"budget":{"unit":"epochs" if arm.startswith("core-") else "optimizer_updates","policy":"full_budget","maximum":5000} if kind=="model_fit" else {},
            "selection":{"metric":"accuracy","direction":"max","ties":"first","evaluation_interval":1,"checkpoint0":False} if kind=="model_fit" else {}}
        runs.append(row);return identifier
    for arm in arms:
        candidates=[]
        for point in p.DOMAIN:
            identity=fingerprint(point)[:20]
            ids=[add("search-"+arm+"-"+identity+"-s"+str(seed),"model_fit",arm,0,seed,0,config(arm,0,"search",point)) for seed in SEARCH_SEEDS]
            candidates.append({"config_id":identity,"parameters":point,"run_ids":ids,"contains_source_anchor":point==p.SOURCE_ANCHOR})
        parents=[i for candidate in candidates for i in candidate["run_ids"]]
        selections[arm]=add("select-"+arm,"selection",arm,0,0,0,{"stage":"search_selection","configurations":candidates,"ranking_metric":"accuracy",
            "ranking_direction":"max","ranking_scope":"mean_own_selected_seed_accuracy","reservation_mib":256,"estimated_seconds":60},parents)
    for split in sorted(splits):
        for seed in COMPARATIVE_SEEDS:
            fitted={}
            for arm in arms:
                cfg=config(arm,split,"comparative",p.DOMAIN[0]);cfg["selected_config_from"]=selections[arm]
                fitted[arm]=[add("compare-sp"+str(split)+"-"+arm+"-s"+str(seed)+"-b0","model_fit",arm,split,seed,0,cfg,[selections[arm]])]
            for bank,arm in independent.items():
                fitted[bank]=list(fitted[arm])
                for body in (1,2,3):
                    cfg=config(arm,split,"comparative",p.DOMAIN[0]);cfg["selected_config_from"]=selections[arm]
                    fitted[bank].append(add("compare-sp"+str(split)+"-"+bank+"-s"+str(seed)+"-b"+str(body),"model_fit",arm,split,seed,body,cfg,[selections[arm]]))
            endpoints={key:row["path"] for key,row in inputs(split,"endpoint_inputs").items()}
            for arm,parents in fitted.items():
                member=independent.get(arm,arm);profile=resource_for(profiles,dataset,family+"-"+member);fusion=config(member,split,"comparative",p.DOMAIN[0])["fusion"]
                prefix="bank-sp"+str(split)+"-"+arm+"-s"+str(seed)
                genuine=arm in independent
                export=add(prefix,"ensemble_assembly",arm,split,seed,0,{"stage":"bank_export","fit_parents":parents,"endpoint_inputs":endpoints,"fusion":fusion,
                    "physical_bank_reference":genuine,"genuine_own_selected_I4":genuine,"physical_slow_bodies":4 if genuine else config(member,split,"comparative",p.DOMAIN[0])["physical_slow_bodies"],
                    "reservation_mib":profile["reservation_mib"],"estimated_seconds":120},parents)
                cal=add(prefix+"-cal","calibration",arm,split,seed,0,{"stage":"calibration","export_parent":export,"endpoint_inputs":endpoints,"fusion":fusion,
                    "placement":"after_serving_pool","steps":500,"learning_rate":.01,"reservation_mib":512,"estimated_seconds":10},[export])
                evaluate=add(prefix+"-evaluate","evaluation",arm,split,seed,0,{"stage":"evaluation","export_parent":export,"calibration_parent":cal,"endpoint_inputs":endpoints,"fusion":fusion,
                    "reservation_mib":512,"estimated_seconds":10},[export,cal])
                banks.append({"arm":arm,"split":split,"seed":seed,"fit_parents":parents,"export":export,"calibration":cal,"evaluation":evaluate,
                    "genuine_I4":genuine,"body0_reused":genuine,"source_common_selected_independent4":arm=="core-independent4_native"})
    roles={role:{str(split):inputs(split,"node_inputs" if role in ("train","select") else "endpoint_inputs")[role if role in ("train","select") else role+"_targets"] for split in splits} for role in ("train","select","cal","evaluate")}
    history={"private_cmcl":"Closed PubMed CMCL failed attribution/served-NLL protections; preserved and not erased by new-data migration",
        "private_adam":"Closed private Adam convention failed own-floor mechanism/served-NLL protection; preserved",
        "masked_core":"Historical source stage2 remains not root adopted and pending; new-policy prospective comparisons do not activate it"}[family]
    study={"schema_version":1,"study_id":family+"-standardized-v1-"+dataset,"stage":"development","dataset":dataset,
        "hypothesis":"Pinned "+family+" source mechanism on new complete providers with native/equipped M1, own-selected genuine I4 and actual boundary original GNNM.",
        "data":{"task":"node_classification","preprocessing":custody["request"]["graph_policy"],"roles":roles,
            "prior_exposure":custody["request"]["evaluation_exposure"],"evidence":{"custody_request_hash":custody["request_hash"],"model_graph":custody["model_graph"],
                "prepared_provider_splits":sorted(splits),"source_exact_training_recipe":False,"source_history":history,"protected_confirmation":False,
                "new_CORE_comparison_policy":policy if family=="masked_core" else None}},"source":sources,"runs":runs,
        "comparisons":[{"references":[methods[a] for a in arms if a not in CANONICAL[family]]+[methods[a] for a in independent],
            "candidates":list(CANONICAL[family].values()),"split_ids":sorted(splits),"seed_ids":list(COMPARATIVE_SEEDS),"all_canonical_records_preserved":True,
            "source_common_selected_independent4_not_own_selected":family=="masked_core"}],
        "release":{"policy":"whole_roster","primary_endpoint":"closed_raw_and_scalar_CAL_banks","global_plan_gates_required":True},
        "resources":{"wall_seconds":max(row["wall_seconds"] for row in profiles.values()),"status":"PROVISIONAL_DISABLED_REQUIRES_RESOURCE_QA_MASTER_AND_HUMAN_GO"}}
    validator(study)
    from compile_master_plan import fit_semantics
    fits=[r for r in runs if r["kind"]=="model_fit"];semantics=[fit_semantics(r) for r in fits]
    counts={"atomic_fit_groups":len(fits),"paid_optimizer_objects":sum(s["optimizer_objects"] for s in semantics),
        "physical_slow_body_instances":sum(s["physical_slow_bodies"] for s in semantics),"full_budget_optimizer_updates":sum(s["known_optimizer_step_calls"] for s in semantics),
        "registered_full5000_budget_units":5000*len(fits),"optimizer_starts_by_stage":dict(Counter(r["config"]["stage"] for r in fits)),
        "source_scoped_TRAIN_forward_calls":sum(s.get("known_source_TRAIN_forward_calls",0) for s in semantics),
        "source_scoped_TRAIN_backward_calls":sum(s.get("known_source_TRAIN_backward_calls",0) for s in semantics),
        "search_selection_jobs":len(selections),"comparative_banks":len(banks),"bank_exports":len(banks),"scalar_CAL_fits":len(banks),
        "registered_scalar_CAL_updates":500*len(banks),"evaluation_jobs":len(banks),"genuine_I4_banks":sum(b["genuine_I4"] for b in banks),
        "reused_exact_I4_body0":sum(b["body0_reused"] for b in banks),"source_common_selected_independent4_banks":sum(b["source_common_selected_independent4"] for b in banks),
        "all_run_records":len(runs),"actual_model_fits":0,"actual_optimizer_updates":0,"actual_predictive_scores":0}
    return study,counts,banks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("project-root","output"):parser.add_argument("--"+name,required=True)
    parser.add_argument("--custody",nargs="+",required=True);parser.add_argument("--prepare-core-graph",action="store_true")
    parser.add_argument("--family",choices=FAMILIES);parser.add_argument("--runtime-package");parser.add_argument("--resources")
    parser.add_argument("--graph-custody",nargs="+");parser.add_argument("--fixture-receipts",nargs="+")
    parser.add_argument("--policy");parser.add_argument("--source-commit")
    args=parser.parse_args();verify_allocation()
    root=Path(args.project_root).resolve()
    def path(value):
        actual=Path(value);return confined(root,str(actual.resolve().relative_to(root)) if actual.is_absolute() else str(actual))
    output=path(args.output)
    for row in source_roster():p.bound(root,row)
    if args.prepare_core_graph:
        if len(args.custody)!=1:raise ValueError("One provider graph per invocation")
        value=prepare_core_graph(root,path(args.custody[0]),output)
        print(canonical({"dataset":value["dataset_id"],"generation_seconds":value["generation_seconds"],"accepted_swaps":value["metadata"]["accepted_swaps"],"scientific_fits":0,"GO_created":False}));return
    if not all((args.family,args.runtime_package,args.resources,args.fixture_receipts,args.policy)) or output.exists():
        raise ValueError("Fresh output, exact family/runtime/resources/policy and CPU receipts required")
    runtime=path(args.runtime_package);resources=json.loads(path(args.resources).read_text());policy=json.loads(path(args.policy).read_text())
    if policy["schema"]!="new-private-source-comparison-policy-v1" or policy["historical_CORE_stage2_activated"] is not False:
        raise ValueError("Explicit new-study policy preserving old stage2 required")
    runtime_rows=[file_binding(root,file) for file in sorted(runtime.rglob("*")) if file.is_file() and "__pycache__" not in file.parts]
    history=json.loads((runtime/"docs/PRIVATE_SOURCE_HISTORY_AND_DIAGNOSTIC_BINDINGS.json").read_text())
    for row in history["files"]:p.bound(root,row)
    fixed=[*source_roster(),*history["files"],file_binding(root,path(args.resources)),file_binding(root,path(args.policy)),*[file_binding(root,path(value)) for value in args.fixture_receipts]]
    prepared=[]
    for index,custody_path in enumerate(args.custody):
        custody,data_dir,bindings=bind_custody(root,path(custody_path));gc=None
        graph_rows=[]
        if args.family=="masked_core":
            if not args.graph_custody or len(args.graph_custody)!=len(args.custody):raise ValueError("Each CORE provider needs its exact GRAPH-only rewire custody")
            gc=json.loads(path(args.graph_custody[index]).read_text())
            if gc["dataset_id"]!=custody["dataset_id"] or gc["TRAIN_SELECT_CAL_evaluate_targets_opened"] is not False or gc["generator_source"]!=REWIRE_SOURCE:
                raise ValueError("Exact provider graph-only source construction required")
            graph_rows=[file_binding(root,path(args.graph_custody[index])),gc["auxiliary_graph"]]
            p.bound(root,gc["auxiliary_graph"])
        sources={"git_commit":args.source_commit or "working_tree_fingerprints_authoritative","files":list({r["path"]:r for r in [*runtime_rows,*fixed,*bindings,*graph_rows]}.values()),
            "frozen_runtime_package":str(runtime.relative_to(root)),"working_tree_fingerprints_authoritative":True,"source_archived_recipe_reproduction":False}
        prepared.append(make_study(root,custody,data_dir,sources,resources["profiles"],str(runtime.relative_to(root)),args.family,gc,policy))
    if {study["dataset"] for study,_,_ in prepared}!={"wikics","amazon-photo"}:raise ValueError("Complete two-provider packet required")
    output.mkdir(parents=True);(output/"studies").mkdir();prefix=str(output.relative_to(root))
    capacities=resources["worker_capacities_cuda_bytes"]
    plan={"schema_version":1,"slots_per_gpu":2,"torch_threads_per_fit":2,"output_root":prefix+"/execution","runtime_package":str(runtime.relative_to(root)),
        "runtime_launcher":str((runtime/"launcher.py").relative_to(root)),"worker_capacities_cuda_bytes":capacities,"studies":[],"groups":[],"barriers":{},"global_gates":{},
        "scientific_launch_enabled":False,"human_GO_required":True,"host_bindings":{"allocation":{"status":"verified_at_compilation","hostname":"anogena-2-0","gpu_uuids":["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]}}}
    summaries={};combined=Counter()
    for study,counts,banks in prepared:
        relative=prefix+"/studies/"+study["study_id"]+".json.gz";(root/relative).write_bytes(gzip.compress((canonical(study)+"\n").encode(),mtime=0));(root/relative).chmod(0o444)
        plan["studies"].append({"path":relative,"spec_hash":fingerprint(study)})
        maximum=max(r["config"]["reservation_mib"] for r in study["runs"])*1024**2
        if maximum>capacities["allocation/gpu0"]:raise ValueError("Declared reservation exceeds allocation capacity")
        plan["groups"].append({"id":study["study_id"],"gpu_assignment":"allocation/gpu0","allowed_workers":["allocation/gpu0"],"maximum_cuda_bytes":maximum,
            "estimated_seconds":sum(r["config"]["estimated_seconds"] for r in study["runs"]),"run_keys":[study["study_id"]+"/"+r["run_id"] for r in study["runs"]]})
        summaries[study["dataset"]]={"counts":counts,"banks":banks,"study_hash":fingerprint(study)};combined.update({k:v for k,v in counts.items() if type(v) is int})
    for name,stages,required in (("whole_search",["search","search_selection"],["comparative"]),("whole_comparative",["comparative"],["bank_export","calibration","evaluation"]),("whole_calibration",["calibration"],["evaluation"])):
        plan["global_gates"][name]={"path":prefix+"/gate_receipts/"+name+".json","source_stages":stages,"required_for_stages":required,"required_hosts":["allocation"]}
    summary={"schema":"disabled-private-source-standardized-summary-v1","family":args.family,"plan_hash":fingerprint(plan),"counts":dict(combined),"per_dataset":summaries,
        "domain":p.DOMAIN,"source_anchor":p.SOURCE_ANCHOR,"source_canonical_methods":list(CANONICAL[args.family].values()),"source_exact_training_recipe":False,"resources_provisional":True,
        "new_comparison_policy":policy,"scientific_training_started":False,"GO_created":False,"full_249_coverage":False,"status":"DISABLED_PENDING_ROOT_RESOURCE_QA_MASTER_AND_HUMAN_GO"}
    for name,value in (("PLAN.json",plan),("SUMMARY.json",summary)):(output/name).write_text(canonical(value)+"\n");(output/name).chmod(0o444)
    print(canonical({"family":args.family,"plan_hash":fingerprint(plan),"counts":dict(combined),"scientific_fits":0,"GO_created":False}))


if __name__=="__main__":main()
