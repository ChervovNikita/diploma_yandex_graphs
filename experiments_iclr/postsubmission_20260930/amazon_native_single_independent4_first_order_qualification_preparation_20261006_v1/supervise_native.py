"""Disabled native engineering supervisor using reviewed owned wait4 helpers.

Publishes combined native qualifier receipt only after whole-child closure.
No numerical imports, fitting, source release or retry.
"""
import time
STARTED=time.monotonic()
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import sys
import traceback
from types import SimpleNamespace

SOURCE_RELEASED=False
TARGET="/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930"
PROCESS=("amazon_ordinary_shared_bank_two_gpu_scheduling_preparation_20261006_v1/process_supervisor.py","6106ee06d4764293cf58e77870f96a157bcc23f442936258a1d982c63dd740cf")
QUALIFIER=("amazon_native_single_independent4_first_order_qualification_preparation_20261006_v1/qualify_native.py","d3f2278ee8e0bbaf3a2294268dceccae757896f6c87d094a67330a88d4d2b53d")
REFERENCE=("amazon_native_single_independent4_sr_reference_preparation_20261006_v2/native_reference.py","0d8e677b4e2c9b3616a0795cc8f1b55758cc63d7048b2d42f614a295a1c1a6ef")


def require(value,message):
    if not value:raise RuntimeError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--execute-authorized",action="store_true");parser.add_argument("--source-root")
    parser.add_argument("--scope");parser.add_argument("--output")
    args=parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({"status":"DISABLED_NATIVE16_OWNED_ENGINEERING_SUPERVISOR","SOURCE_RELEASED":False,"numeric_imports":False}));return
    require(SOURCE_RELEASED is False and args.source_root==TARGET and socket.gethostname()=="peptide"
        and args.scope and args.output and sys.dont_write_bytecode
        and not any(n=="torch" or n.startswith("torch.") for n in sys.modules),"Exact fresh normal77/-B engineering supervisor required")
    root=Path(args.source_root).resolve()
    for relative,pin in (PROCESS,QUALIFIER,REFERENCE):require(sha(root/relative)==pin,"Exact reviewed engineering/source binding differs")
    name="_native16_owned_process_helpers";require(name not in sys.modules,"Fresh stdlib supervisor required")
    spec=importlib.util.spec_from_file_location(name,root/PROCESS[0]);process=importlib.util.module_from_spec(spec)
    sys.modules[name]=process;spec.loader.exec_module(process)
    require(process.SOURCE_RELEASED is False,"Original reviewed process helper remains disabled")
    ordinary=process.load_ordinary(root)
    root,output,scope_path=ordinary.deliberate_paths(SimpleNamespace(source_root=args.source_root,output=args.output,admission=args.scope))
    require(scope_path.stat().st_mode&0o222==0,"Immutable actual root native engineering scope required")
    scope=json.loads(scope_path.read_text())
    require(scope["engineering_execution_authorized"] is True and scope["fit_authorized"] is False
        and scope["W_SR_label_access_engineering_authorized"] is True and scope["A_scoring"] is False and scope["VALID_TEST_access"] is False
        and scope["native_reference_source_review_approved"] is True and scope["native_qualifier_source_review_approved"] is True
        and scope["native_supervisor_source_review_approved"] is True
        and scope["native_reference_worker_sha256"]==REFERENCE[1] and scope["native_qualifier_worker_sha256"]==QUALIFIER[1]
        and scope["native_supervisor_worker_sha256"]==sha(__file__) and scope["member_seeds"]==[17,1026,2035,3044],
        "Reviewed bounded native engineering supervision only")
    require(output==process.new_path(root,scope["supervisor_output_relative"]),"Assigned native supervisor output differs")
    child_output=process.new_path(root,scope["qualification_output_relative"]);require(child_output!=output,"Two distinct fresh outputs required")
    python=scope["runtime"]["python_resolved"]
    require(Path(python).is_absolute() and str(Path(python).resolve())==python and Path(python).is_file(),"Exact normal interpreter required")
    caps,watchdog=scope["resource_limits"],scope["external_watchdog_seconds"]
    grace,reap_timeout=scope["termination_grace_seconds"],scope["termination_reap_timeout_seconds"]
    process.validate_caps(caps,watchdog,grace,reap_timeout)
    uuid=scope["CUDA_VISIBLE_DEVICES"];require(isinstance(uuid,str) and uuid.startswith("GPU-") and "," not in uuid,"Single root-assigned UUID required")
    owner=ordinary.CreatedOutput(output,root);token=owner.creator_token
    receipt={"schema":"normal_native_independent_first_order_qualification_v1","status":"RUNNING",
        "supervisor_worker_sha256":sha(__file__),"qualifier_worker_sha256":QUALIFIER[1],"process_helper_sha256":PROCESS[1],
        "native_reference_disabled_source_sha256":REFERENCE[1],"scope_sha256":sha(scope_path),
        "launch_attempts":0,"children":[],"numeric_imports_in_supervisor":False,"model_fits":0,"persistent_updates":0,
        "A_scoring":False,"VALID_TEST_access":False,"full_native_context_supported":False}
    def save():
        receipt["supervisor_elapsed_seconds"]=time.monotonic()-STARTED
        ordinary.atomic(owner.verify(token)/"RESULT.json",receipt)
    old_handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    def interrupted(signum,frame):raise InterruptedError("Owned native supervisor interrupted: "+str(signum))
    code=1
    try:
        for signum in old_handlers:signal.signal(signum,interrupted)
        receipt["launch_attempts"]=1;save()
        child=process.launch(root,owner,token,"native16",python,root/QUALIFIER[0],
            ["--execute-authorized","--source-root",TARGET,"--scope",str(scope_path),"--output",str(child_output)],
            dict(os.environ,CUDA_VISIBLE_DEVICES=uuid),receipt["children"])
        save()
        while not process.watch(child,watchdog,grace,reap_timeout):time.sleep(0.25)
        save()
        result_row=process.immutable_descriptor(root,child_output/"RESULT.json")
        result=json.loads(ordinary.bound(root,result_row).read_text());process.resource_closure(child,result,caps)
        require(result["schema"]=="normal_native_independent_first_order_qualification_v1" and result["status"]=="PASS"
            and result["worker_sha256"]==QUALIFIER[1] and result["native_reference_disabled_source_sha256"]==REFERENCE[1]
            and result["root_scope_sha256"]==sha(scope_path) and result["CUDA_VISIBLE_DEVICES"]==uuid and result["member_seeds"]==scope["member_seeds"]
            and all(result[k] is True for k in ("full_native_context_supported","both_stages_checked","native_own_CE_gradient_Adam_RNG_checked",
                "fresh_seed_constructor_reset_checked","independent_parameter_and_Adam_ownership_checked","four_resident_models_and_histories_checked",
                "per_parameter_clock_semantics_checked","all_updated_states_discarded"))
            and result["model_fits"]==0 and result["persistent_updates"]==0 and result["W400_acquisitions"]==0
            and result["trained_checkpoints_saved"]==0 and not result["restoration_errors"],"Exact successful discarded native qualification required")
        require(sha(scope_path)==receipt["scope_sha256"] and all(sha(root/p)==h for p,h in (PROCESS,QUALIFIER,REFERENCE)),"Scope/source bytes changed")
        # Root binds this combined immutable RESULT to the fit admission: the
        # exact child support/context/timings plus successful external closure.
        receipt.update(result)
        receipt.update(status="PASS",schema="normal_native_independent_first_order_qualification_v1",
            supervisor_worker_sha256=sha(__file__),qualifier_worker_sha256=QUALIFIER[1],qualifier_result=result_row,
            scope_sha256=sha(scope_path),child_exit_code=0,wait4_closed=True,external_whole_child_resource_closure_verified=True,
            root_caps=caps,children=[child],launch_attempts=1,numeric_imports_in_supervisor=False)
        code=0
    except BaseException as error:receipt.update(status="FAIL",full_native_context_supported=False,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        for signum in old_handlers:signal.signal(signum,signal.SIG_IGN)
        errors=process.cleanup(receipt["children"],grace,reap_timeout);receipt["cleanup_errors"]=errors
        if errors:receipt.update(status="FAIL",full_native_context_supported=False);code=1
        for signum,handler in old_handlers.items():signal.signal(signum,handler)
        if code:receipt["full_native_context_supported"]=False
        try:save();owner.freeze(token)
        except BaseException as error:
            code=1;receipt.update(status="FAIL",full_native_context_supported=False,output_finalization_error=str(error))
            try:save();owner.freeze(token)
            except BaseException as recovery_error:receipt["output_failure_receipt_error"]=str(recovery_error)
    print(json.dumps({"status":receipt["status"],"child_exit_code":receipt.get("child_exit_code"),"model_fits":0,"error":receipt.get("error")}))
    raise SystemExit(code)


if __name__=="__main__":main()
