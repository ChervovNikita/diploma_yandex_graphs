"""Compile disabled126-fit boundary reference supplement from exact link custody.

Stdlib metadata/hashes only; allocation guard precedes project reads. Old link
packet is unchanged. Source TRAIN-only graph/SELECT/TEST protocol is retained.
"""
from __future__ import annotations
import argparse
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess

from adapters.link_boundary import BOUNDARY_SOURCE,CONTRACT,validate_config,validate_roster
from compile_link_matrix import native_parameters,replace,bind_custody,bind_sealed
from prepare_node_data import verify_allocation
from spec import canonical,fingerprint,validate

SEARCH_SEEDS=(73101,73103)
COMPARATIVE_SEEDS=(73201,73203,73207)
STYLES=("ordinary","equipped","original_shared4")
RUNTIME=("engine.py","metrics.py","spec.py","registry.py","partition.py","run_study.py","run_endpoint.py",
    "launcher.py","phase_closure.py","prepare_node_data.py","compile_link_matrix.py","compile_link_boundary_matrix.py",
    "adapters/__init__.py","adapters/link.py","adapters/link_search.py","adapters/link_endpoints.py",
    "adapters/link_boundary.py","adapters/link_boundary_endpoints.py")


def domain(predictor):
    result=[]
    for width in (64,128):
        for multiplier in (.5,1.,2.):
            p=replace(native_parameters(predictor),"encoder.hidden",width)
            p=replace(p,"optimizer.encoder_lr",.0082*multiplier)
            p=replace(p,"optimizer.predictor_lr",.0037*multiplier)
            result.append(p)
    return result


def method(predictor,style,bank="M1"):
    return "original_"+predictor+"_boundary_GNNM" if style=="original_shared4" else "source_"+predictor+"_"+style+"_"+bank


def make_study(template,boundary_source,source,profile,runtime):
    base=next(r["config"] for r in template["runs"] if r["kind"]=="model_fit")
    budget={"unit":"epochs","policy":"full_budget","maximum":100}
    selector={"metric":"hits@50","direction":"max","ties":"first","evaluation_interval":1,"checkpoint0":False,"save_every":1}
    runs=[];selections={};banks=[]
    def add(identifier,kind,label,seed,body,config,parents=()):
        adapter=(runtime+"/adapters/link_boundary.py:make_adapter" if kind=="model_fit" else
            runtime+"/adapters/link_search.py:select_configuration" if kind=="selection" else
            runtime+"/adapters/link_boundary_endpoints.py:"+("export_bank" if kind=="ensemble_assembly" else "evaluate_bank"))
        runs.append({"run_id":identifier,"kind":kind,"method":label,"split":0,"seed":seed,"body":body,
            "adapter":adapter,"config":config,"parents":list(parents),
            "budget":copy.deepcopy(budget) if kind=="model_fit" else {},"selection":copy.deepcopy(selector) if kind=="model_fit" else {}})
        return identifier
    def config(point,style,seed,body,stage,group=None):
        v={k:copy.deepcopy(base[k]) for k in ("link_inputs","link_source","sampler","metric","support","training")}
        initialization=seed+1000003*body;m=4 if style=="original_shared4" else 1
        v.update(copy.deepcopy(point),arm="native_independent4_body" if group else "native_single",
            initialization_seed=initialization,shuffle_seed=initialization,dropout_seed=initialization+3000007,
            boundary={"kind":"ordinary" if style=="ordinary" else "original_boundary","source":boundary_source},
            members=m,is_original_gnnm=m==4,source_optimizer_count=1,train_minibatches_per_epoch=17,
            graphmodelsource=copy.deepcopy(base["link_source"]),stage=stage,reservation_mib=profile["reservation_mib"],
            maximum_cuda_bytes=profile["reservation_mib"]*1024**2,estimated_seconds=profile["estimated_seconds_per_epoch"]*100)
        if group:v["independence_group"]=group
        return v
    for predictor in ("NCN","NCNC"):
        for style in STYLES:
            candidates=[]
            for point in domain(predictor):
                config_id=fingerprint(point)[:20]
                ids=[add("search-"+predictor+"-"+style+"-"+config_id+"-s"+str(seed),"model_fit",method(predictor,style),seed,0,
                    config(point,style,seed,0,"search")) for seed in SEARCH_SEEDS]
                candidates.append({"config_id":config_id,"parameters":point,"run_ids":ids})
            selections[predictor,style]=add("select-"+predictor+"-"+style,"selection",method(predictor,style),0,0,
                {"stage":"search_selection","ranking_metric":"hits@50","ranking_direction":"max","configurations":candidates,
                 "reservation_mib":256,"estimated_seconds":profile["estimated_selection_seconds"]},
                [i for c in candidates for i in c["run_ids"]])
        for seed in COMPARATIVE_SEEDS:
            for style in STYLES:
                prefix="compare-"+predictor+"-"+style+"-s"+str(seed)
                fits=[]
                for body in range(1 if style=="original_shared4" else 4):
                    v=config(domain(predictor)[0],style,seed,body,"comparative",None if style=="original_shared4" else prefix)
                    v["selected_config_from"]=selections[predictor,style]
                    fits.append(add(prefix+"-b"+str(body),"model_fit",method(predictor,style,"I4"),seed,body,v,[selections[predictor,style]]))
                shared={"query_input":template["data"]["evidence"]["endpoint_inputs"]["evaluate_queries"],
                    "fusion":"mean_raw_scores","support_policy":"full_TRAIN_only"}
                export=add(prefix+"-export","ensemble_assembly",method(predictor,style,"I4"),seed,0,
                    {**shared,"fit_parents":fits,"stage":"bank_export","reservation_mib":profile["reservation_mib"],
                     "estimated_seconds":profile["estimated_export_seconds"]},fits)
                exports=[(export,fits,method(predictor,style,"I4"))]
                if style!="original_shared4":
                    single=add(prefix+"-single-export","ensemble_assembly",method(predictor,style),seed,0,
                        {**shared,"fit_parents":fits[:1],"reuse_export_parent":export,"single_reference":"fixed_I4_body0",
                         "stage":"bank_export","reservation_mib":512,"estimated_seconds":1},[export,fits[0]])
                    exports.append((single,fits[:1],method(predictor,style)))
                for exp,parents,label in exports:
                    evaluation=add(exp+"-evaluate","evaluation",label,seed,0,
                        {**shared,"target_input":template["data"]["evidence"]["endpoint_inputs"]["evaluate_targets"],
                         "export_parent":exp,"metric":base["metric"],"stage":"evaluation","reservation_mib":512,"estimated_seconds":60},[exp])
                    banks.append({"method":label,"predictor":predictor,"style":style,"seed":seed,"fit_parents":parents,
                        "export":exp,"evaluation":evaluation,"genuine_I4":len(parents)==4,
                        "body0_reused":len(parents)==1 and style!="original_shared4","internal_shared_routes":4 if style=="original_shared4" else 1})
    study={"schema_version":1,"study_id":"original-boundary-ncn-ncnc-collab-v1","stage":"development","dataset":"ogbl-collab",
        "hypothesis":"Mandatory adapted original boundary GNNM reference with domain-matched native and boundary equipped M1/I4 controls.",
        "data":copy.deepcopy(template["data"]),"source":source,"runs":runs,
        "comparisons":[{"references":[method(p,"ordinary"),method(p,"ordinary","I4"),method(p,"equipped"),method(p,"equipped","I4")],
            "candidates":[method(p,"original_shared4")],"scope":"all three comparative seeds; exact shared TRAIN-only link source protocol"} for p in ("NCN","NCNC")],
        "release":{"policy":"whole_roster","primary_endpoint":"raw sealed official TEST Hits50 after whole combined closures",
            "global_plan_gates_required":True,"scientific_launch_enabled":False,"human_GO_required":True},
        "resources":{"wall_seconds":profile["wall_seconds"],"profile":profile,"status":"PROVISIONAL_EXCLUDED_UNTIL_ENGINEERING_RESOURCE_QUALIFICATION"}}
    study["data"]["evidence"]["original_GNNM_link"]={"status":"materialized_disabled_task_adaptation","contract":CONTRACT,
        "original_task_replication":False,"boundary_scope":"first GCN affine and terminal native decoder affine; native adjacency API retained"}
    validate(study)
    for row in runs:
        if row["kind"]=="model_fit":validate_config(row["config"],row);validate_roster(study,row)
    counts={"model_fits":126,"search_model_fits":72,"comparative_model_fits":54,"model_optimizer_objects":126,
        "physical_slow_bodies":126,"epochs_per_fit":100,"registered_epochs":12600,"train_minibatches_per_epoch":17,
        "registered_optimizer_updates":214200,"registered_member_backward_calls":367200,
        "training_native_encoder_traversals":367200,"top_level_training_decoder_query_traversals":734400,
        "recursive_NCNC_decoder_calls":"measured at execution; not bounded by two top-level calls",
        "search_selection_jobs":6,"bank_exports":30,"evaluation_jobs":30,"CAL_fit_jobs":0,
        "genuine_I4_banks":12,"reused_I4_body0":12,"original_shared4_banks":6,"run_records":192,
        "actual_optimizer_updates":0,"actual_predictive_scores":0,"scientific_fits":0}
    assert len(runs)==192 and sum(r["kind"]=="model_fit" for r in runs)==126
    return study,counts,banks


def binding(root,path):
    from adapters.link import file_sha
    path=Path(path).resolve()
    if not path.is_relative_to(root) or path.is_symlink() or not path.is_file():raise ValueError("Confine immutable packet")
    return {"path":str(path.relative_to(root)),"bytes":path.stat().st_size,"sha256":file_sha(path)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("project-root","template-study","boundary-source","runtime-package","qa-receipt","resources","output"):
        parser.add_argument("--"+name,required=True)
    args=parser.parse_args();verify_allocation()
    root=Path(args.project_root).resolve()
    def path(v):
        p=Path(v);p=p if p.is_absolute() else root/p
        if not p.resolve().is_relative_to(root):raise ValueError("Confine new link boundary packet")
        return p.resolve()
    template_path,runtime,output=map(path,(args.template_study,args.runtime_package,args.output))
    if output.exists() or runtime.is_relative_to(root/'training_pipeline_v2_20261012'):raise ValueError("Fresh output and frozen runtime required")
    template=json.loads(template_path.read_text());validate(template)
    manifest=next(r for r in template["source"]["files"] if r["path"].endswith('/fit_data/MANIFEST.json'))
    custody,files=bind_custody(root,path(manifest["path"]))
    if binding(root,path(manifest["path"]))!=manifest:raise ValueError("Old link provider custody changed")
    sealrow=next(r for r in template["source"]["files"] if r["path"].endswith('/sealed_endpoint/MANIFEST.json'))
    sealed,sealfiles=bind_sealed(root,path(sealrow["path"]));files+=sealfiles
    if binding(root,path(sealrow["path"]))!=sealrow or sealed["fit_manifest"]!=manifest:raise ValueError("Sealed source protocol ancestry changed")
    boundary=binding(root,path(args.boundary_source))
    if any(boundary[k]!=BOUNDARY_SOURCE[k] for k in BOUNDARY_SOURCE):raise ValueError("Pinned original BEBlock source required")
    resource_path,qa_path=map(path,(args.resources,args.qa_receipt));profile=json.loads(resource_path.read_text())
    qa=json.loads(qa_path.read_text())
    if qa.get("passed") is not True or qa.get("skips")!=0:raise ValueError("Passing exact frozen CPU/source qualification required")
    files += [binding(root,runtime/name) for name in RUNTIME]+[boundary,binding(root,template_path),binding(root,resource_path),binding(root,qa_path)]
    files=list({r["path"]:r for r in files}.values())
    source={"git_commit":subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(),"files":files,
        "working_tree_fingerprints_authoritative":True,"frozen_runtime_package":str(runtime.relative_to(root)),
        "author_protocol_replication":False,"original_task_replication":False}
    study,counts,banks=make_study(template,boundary,source,profile,str(runtime.relative_to(root)))
    prefix=str(output.relative_to(root));study_path=prefix+'/studies/'+study["study_id"]+'.json.gz'
    capacity={"allocation/gpu0":78*1024**3}
    if max(r["config"]["reservation_mib"] for r in study["runs"])*1024**2>capacity["allocation/gpu0"]:raise ValueError("Declared resource exceeds planning capacity")
    plan={"schema_version":1,"slots_per_gpu":2,"torch_threads_per_fit":2,"runtime_package":str(runtime.relative_to(root)),
        "runtime_launcher":str((runtime/'launcher.py').relative_to(root)),"output_root":prefix+'/execution',
        "worker_capacities_cuda_bytes":capacity,"studies":[{"path":study_path,"spec_hash":fingerprint(study)}],
        "groups":[{"id":study["study_id"],"gpu_assignment":"allocation/gpu0","allowed_workers":["allocation/gpu0"],
            "maximum_cuda_bytes":profile["reservation_mib"]*1024**2,"estimated_seconds":sum(r["config"]["estimated_seconds"] for r in study["runs"]),
            "run_keys":[study["study_id"]+'/'+r["run_id"] for r in study["runs"]]}],"barriers":{},"global_gates":{},
        "scientific_launch_enabled":False,"human_GO_required":True}
    for name,stages,required in [('whole_search',['search','search_selection'],['comparative']),('whole_comparative',['comparative'],['bank_export','evaluation'])]:
        plan['global_gates'][name]={"path":prefix+'/gate_receipts/'+name+'.json',"source_stages":stages,"required_for_stages":required,"required_hosts":['allocation']}
    output.mkdir(parents=True);(output/'studies').mkdir()
    (root/study_path).write_bytes(gzip.compress((canonical(study)+'\n').encode(),mtime=0))
    summary={"schema":"disabled-link-boundary-reference-v1","status":"PREPARED_DISABLED_RESOURCE_EXCLUDED_MASTER_AND_GO_PENDING",
        "plan_hash":fingerprint(plan),"counts":counts,"banks":banks,"contract":CONTRACT,"scientific_launch_enabled":False,
        "human_GO_required":True,"training_started":False,"GO_created":False,"resources_provisional":True,
        "source_role_tensors_decoded":False,"old_link_packet_unchanged":True,"calibration":"raw_only_absent"}
    (output/'PLAN.json').write_text(canonical(plan)+'\n');(output/'SUMMARY.json').write_text(canonical(summary)+'\n')
    print(canonical({"counts":counts,"plan_hash":summary['plan_hash'],"science":False,"GO":False}))


if __name__=='__main__':main()
