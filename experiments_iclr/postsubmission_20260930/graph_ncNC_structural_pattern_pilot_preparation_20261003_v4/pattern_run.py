#!/usr/bin/env python3
"""Fixed J/F stages; source preparation itself provides no execution release."""
from argparse import ArgumentParser
from time import perf_counter
from pilot_common import (STAGES, preflight, admit_invocation, fresh_output,
                          runtime_stdlib, qualification_required, atomic_json,
                          file_sha, lock_output)
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
            result=run(context,output,attempts);name="CLOSURE.json"
        else:
            attempts.phase("stdlib_interpreter_and_distribution_admission")
            context["versions"]=runtime_stdlib(context)
            from pilot_model import runtime,modules
            attempts.phase("normal_host_CUDA_source_and_binary_admission")
            device,sampler=runtime(context);cuda_started=True
            attempts.phase("sealed_qualified_model_source_imports")
            mods=modules(context)
            data=None
            if args.stage!="numerical":
                from pilot_data import load_data
                attempts.phase("authenticated_complete_TRAIN_raw_VALID_load_and_transfer")
                data=load_data(context,device)
            if args.stage in ("numerical","full_graph"):
                from pattern_qualification import run
                result=run(context,mods,data,sampler,device,output,args.stage,attempts)
                name="QUALIFICATION.json"
            elif args.stage=="fit":
                from pattern_fit import fit
                from pilot_evaluate import evaluator
                attempts.phase("bound_official_VALID_evaluator")
                result=fit(context,mods,data,sampler,evaluator(context),device,output,args.unit,args.resume,attempts)
                name="COMPLETE.json"
            else:
                from pattern_diagnostics import run
                result=run(context,mods,data,sampler,device,output,attempts)
                name="COMPLETE.json"
            import torch
            attempts.phase("final_CUDA_synchronization_and_inclusive_peak_accounting")
            torch.cuda.synchronize(0)
            for key,observed in (("cuda_peak_allocated_bytes",torch.cuda.max_memory_allocated(0)),
                                 ("cuda_peak_reserved_bytes",torch.cuda.max_memory_reserved(0))):
                prior=attempts.row.get("observed_CUDA_peak_before_arm_resets",{}).get(key,0)
                result[key]=max(result.get(key,0),observed,prior)
        attempts.phase("final_receipt_serialization")
        result["peak_host_rss_bytes"]=host_peak_bytes()
        atomic_json(output/name,result)
        result["inclusive_accounting"]=attempts.finish("COMPLETE")
        atomic_json(output/name,result)
        print("STAGE_COMPLETE stage="+args.stage+" unit="+args.unit+" seed=0 receipt="+str(output/name),flush=True)
    except Exception as error:
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
        def receipt(path):
            return {"path":path.name,"bytes":path.stat().st_size,"sha256":file_sha(path)}
        atomic_json(output/"FAILED.json",{"schema":"ncnc-pattern-failed-stage-v1","identity":context["identity"],
            "stage":args.stage,"unit":args.unit,"seed":0,"status":"FAILED","failure":attempts.row["failure"],
            "inclusive_accounting":accounting,"attempts_receipt":receipt(output/"ATTEMPTS.json"),
            "journal_receipt":receipt(output/"JOURNAL.json") if (output/"JOURNAL.json").exists() else None,
            "predictive_values_exposed":False,"test_file_opened":False})
        raise
    finally:
        import os
        os.close(output_lock)


if __name__=="__main__":
    main()
