#!/usr/bin/env python3
"""Fixed J/F stages; source preparation itself provides no execution release."""
from argparse import ArgumentParser
import os
import sys
from time import perf_counter
from pilot_common import (STAGES, preflight, admit_invocation, fresh_output,
                          runtime_stdlib, qualification_required, atomic_json,
                          file_sha, lock_output, driver_module_custody)
from pilot_accounting import Attempts


def host_peak_bytes():
    import resource
    import sys
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform=="darwin" else peak*1024)


def arguments():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--root-release",required=True)
    parser.add_argument("--stage",required=True,choices=STAGES)
    parser.add_argument("--unit",required=True,choices=("pair","J","F"))
    parser.add_argument("--base-seed",required=True,type=int,choices=(0,))
    parser.add_argument("--output",required=True)
    parser.add_argument("--resume",action="store_true")
    return parser.parse_args()


def receipt(path):
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha(path)}


def main():
    started=perf_counter();args=arguments()
    context=preflight(args.root_release,args.stage)
    admit_invocation(context,args.stage,args.unit,args.base_seed,args.output,args.resume)
    if args.stage!="numerical":qualification_required(context,args.stage)
    output=fresh_output(args.output,resume=args.resume)
    output_lock=lock_output(output)
    attempts=Attempts(output,context,args.stage,args.unit,0,started=started)
    cuda_started=False
    try:
        if args.stage=="close":
            from pattern_close import run
            driver_module_custody(context)
            result=run(context,output,attempts);name="CLOSURE.json"
        else:
            require_torch_absent="torch" not in sys.modules
            from pilot_common import require
            require(require_torch_absent, "Torch must be absent before process workspace configuration")
            environment={"Torch_absent_before_configuration":require_torch_absent,
                         "CUBLAS_WORKSPACE_CONFIG_before":os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
            os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
            environment.update(Torch_absent_after_configuration="torch" not in sys.modules,
                               CUBLAS_WORKSPACE_CONFIG_after=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
            context["preimport_environment"]=environment
            attempts.row["preimport_environment"]=environment
            attempts.phase("stdlib_interpreter_and_distribution_admission")
            context["versions"]=runtime_stdlib(context)
            from pilot_model import runtime,modules,runtime_custody
            attempts.phase("normal_host_CUDA_source_and_binary_admission")
            device,sampler=runtime(context);cuda_started=True
            atomic_json(output/"RUNTIME_PROFILE_TRANSITION.json",context["runtime_profile_transition"])
            attempts.row["runtime_profile_transition_receipt"]=receipt(output/"RUNTIME_PROFILE_TRANSITION.json")
            attempts.phase("sealed_qualified_model_source_imports")
            mods=modules(context)
            driver_module_custody(context)
            runtime(context)
            data=None
            if args.stage!="numerical":
                from pilot_data import load_data
                driver_module_custody(context)
                attempts.phase("authenticated_complete_TRAIN_raw_VALID_load_and_transfer")
                data=load_data(context,device)
            if args.stage in ("numerical","full_graph"):
                from pattern_qualification import run
                driver_module_custody(context)
                runtime(context)
                result=run(context,mods,data,sampler,device,output,args.stage,attempts)
                name="QUALIFICATION.json"
            elif args.stage=="fit":
                from pattern_fit import fit
                from pilot_evaluate import evaluator
                driver_module_custody(context)
                runtime(context)
                attempts.phase("bound_official_VALID_evaluator")
                result=fit(context,mods,data,sampler,evaluator(context),device,output,args.unit,args.resume,attempts)
                name="COMPLETE.json"
            else:
                from pattern_diagnostics import run
                driver_module_custody(context)
                runtime(context)
                result=run(context,mods,data,sampler,device,output,attempts)
                name="COMPLETE.json"
            import torch
            attempts.phase("final_CUDA_synchronization_and_inclusive_peak_accounting")
            torch.cuda.synchronize(0)
            for key,observed in (("cuda_peak_allocated_bytes",torch.cuda.max_memory_allocated(0)),
                                 ("cuda_peak_reserved_bytes",torch.cuda.max_memory_reserved(0))):
                prior=attempts.row.get("observed_CUDA_peak_before_arm_resets",{}).get(key,0)
                result[key]=max(result.get(key,0),observed,prior)
            result["runtime_profile_transition"]=context["runtime_profile_transition"]
            result["runtime_profile_transition_receipt"]=receipt(output/"RUNTIME_PROFILE_TRANSITION.json")
            result["runtime_profile_final"]=runtime_custody(context)
            attempts.row["runtime_profile_final"]=result["runtime_profile_final"]
        attempts.phase("final_receipt_serialization")
        result["peak_host_rss_bytes"]=host_peak_bytes()
        atomic_json(output/name,result)
        result["inclusive_accounting"]=attempts.finish("COMPLETE")
        atomic_json(output/name,result)
        print("STAGE_COMPLETE stage="+args.stage+" unit="+args.unit+" seed=0 receipt="+str(output/name),flush=True)
    except Exception as error:
        runtime_failure={}
        if args.stage!="close":
            transition=context.get("runtime_profile_transition")
            if transition is not None:
                runtime_failure["runtime_profile_transition"]=transition
                try:
                    atomic_json(output/"RUNTIME_PROFILE_TRANSITION.json",transition)
                    runtime_failure["runtime_profile_transition_receipt"]=receipt(output/"RUNTIME_PROFILE_TRANSITION.json")
                    attempts.row["runtime_profile_transition_receipt"]=runtime_failure["runtime_profile_transition_receipt"]
                except Exception as transition_error:
                    runtime_failure["runtime_transition_receipt_failure"]={"exception_type":type(transition_error).__name__,"condition":str(transition_error)}
            module=sys.modules.get("torch")
            if module is not None:
                try:
                    cuda=getattr(module,"cuda",None)
                    if cuda is not None and cuda.is_initialized():
                        from pilot_model import observe_runtime
                        runtime_failure["runtime_profile_final"]=observe_runtime(context)
                        attempts.row["runtime_profile_final"]=runtime_failure["runtime_profile_final"]
                except Exception as observation_error:
                    runtime_failure["runtime_observation_failure"]={"exception_type":type(observation_error).__name__,"condition":str(observation_error)}
        attempts.row["failure_peak_host_rss_bytes"]=host_peak_bytes()
        if cuda_started:
            try:
                import torch
                torch.cuda.synchronize(0)
                prior=attempts.row.get("observed_CUDA_peak_before_arm_resets",{})
                attempts.row["failure_peak_allocated_bytes"]=max(torch.cuda.max_memory_allocated(0),prior.get("cuda_peak_allocated_bytes",0))
                attempts.row["failure_peak_reserved_bytes"]=max(torch.cuda.max_memory_reserved(0),prior.get("cuda_peak_reserved_bytes",0))
            except Exception as sync_error:
                attempts.row["failed_CUDA_accounting"]={"exception_type":type(sync_error).__name__,"condition":str(sync_error)}
        accounting=attempts.finish("FAILED",error)
        atomic_json(output/"FAILED.json",{"schema":"ncnc-pattern-failed-stage-v1","identity":context["identity"],
            "stage":args.stage,"unit":args.unit,"seed":0,"status":"FAILED","failure":attempts.row["failure"],
            "inclusive_accounting":accounting,"attempts_receipt":receipt(output/"ATTEMPTS.json"),
            "journal_receipt":receipt(output/"JOURNAL.json") if (output/"JOURNAL.json").exists() else None,
            "predictive_values_exposed":False,"test_file_opened":False,**runtime_failure})
        raise
    finally:
        os.close(output_lock)


if __name__=="__main__":
    main()
