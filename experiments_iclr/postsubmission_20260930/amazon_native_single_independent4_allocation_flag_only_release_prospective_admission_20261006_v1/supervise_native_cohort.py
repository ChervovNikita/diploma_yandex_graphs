"""Disabled one-child native SINGLE/ENS4 cohort supervisor; no numerical imports.

Exact flag-only native V2 worker and actual combined native16 support; owned
process helpers/wait4/caps, fixed worker-derived counts and eight checkpoints.
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

SOURCE_RELEASED=True
TARGET="/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930"
PROCESS=("amazon_ordinary_shared_bank_two_gpu_scheduling_preparation_20261006_v1/process_supervisor.py","6106ee06d4764293cf58e77870f96a157bcc23f442936258a1d982c63dd740cf")
REFERENCE=("amazon_native_single_independent4_allocation_only_port_preparation_20261006_v2/native_reference.py","a75aa679788bcb6e892acdd16d7b2b514b10a94da435036050e43ed0f5dce545")
RELEASED_WORKER_SHA256="56ca97c68437d59a74f98733b9399a770bbbce08b67322648f70fe0c82fb3ffc"
PROTOCOL_SHA256="6e4110adaae9e9dc426943ca0e37100d7c7423fc8102be049a7182663ac6ec72"
QUALIFICATION_SHA256="d8122ff9acb242450f58999e01101832d1549563932d23603eb107a98abed7c0"
QUALIFICATION_BYTES=16041


def require(value,message):
    if not value:raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):h.update(block)
    return h.hexdigest()


def load(root,binding,name):
    path=root/binding[0]
    require(sha(path)==binding[1] and path.stat().st_mode&0o222==0 and name not in sys.modules,"Exact immutable fresh source required")
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def checkpoint(root,process,output,row,member,kind):
    # Names are the exact native worker checkpoint() format and phase constants.
    name="member"+str(member)+"_"+kind+".pt"
    require(row["path"]==name,"Only declared native fixed checkpoint name accepted")
    descriptor=process.immutable_descriptor(root,output/name)
    require(descriptor["bytes"]==row["bytes"] and descriptor["sha256"]==row["sha256"],"Frozen native checkpoint bytes differ")
    return descriptor


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--execute-authorized",action="store_true");parser.add_argument("--source-root")
    parser.add_argument("--scope");parser.add_argument("--output")
    args=parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({"status":"DISABLED_ONE_NATIVE_SINGLE_ENS4_COHORT_SUPERVISOR","SOURCE_RELEASED":SOURCE_RELEASED,"numeric_imports":False}));return 0
    require(SOURCE_RELEASED is True,"Disabled native cohort supervision source has no launch/fit authority")
    require(args.source_root==TARGET and socket.gethostname()=="anogena-2-0" and args.scope and args.output
        and sys.dont_write_bytecode and not any(n=="torch" or n.startswith("torch.") for n in sys.modules),"Exact fresh normal77/-B supervisor required")
    root=Path(args.source_root).resolve();process=load(root,PROCESS,"_native_cohort_owned_process_helpers")
    reference=load(root,REFERENCE,"_native_cohort_fixed_worker_constants")
    require(process.SOURCE_RELEASED is False and reference.SOURCE_RELEASED is False,"Reviewed disabled helper/source markers must stay unchanged")
    ordinary=process.load_ordinary(root)
    root,output,scope_path=ordinary.deliberate_paths(SimpleNamespace(source_root=args.source_root,output=args.output,admission=args.scope))
    require(scope_path.stat().st_mode&0o222==0,"Immutable actual root native supervision scope required")
    scope=json.loads(scope_path.read_text())
    own_disabled_sha=hashlib.sha256(Path(__file__).read_text().replace("SOURCE_RELEASED=True","SOURCE_RELEASED=False",1).encode()).hexdigest()
    review=json.loads(ordinary.bound(root,scope["source_review"]).read_text())
    require(scope["schema"]=="root_fresh_native_single_independent4_owned_supervision_v1"
        and scope["root_supervision_fit_authorized"] is True and scope["fixed_before_launch"] is True
        and scope["supervisor_worker_sha256"]==sha(__file__) and scope["source_review_approved"] is True
        and scope["one_owned_child_no_retry"] is True and scope["A_scoring"] is False and scope["VALID_TEST_access"] is False
        and review["status"].startswith("PASS_SOURCE") and review["supervisor_sha256"]==own_disabled_sha
        and review["execution_or_fit_authorized"] is False,"Exact reviewed once-only cohort supervision authority required")
    admission_path=ordinary.bound(root,scope["worker_admission"]);admission=json.loads(admission_path.read_text())
    qualification_row=scope["actual_combined_native_qualification"]
    require(qualification_row==admission["normal_native_context_qualification"] and qualification_row["bytes"]==QUALIFICATION_BYTES
        and qualification_row["sha256"]==QUALIFICATION_SHA256,"Exact actual combined native16 qualification binding required")
    qualified=json.loads(ordinary.bound(root,qualification_row).read_text())
    require(qualified["schema"]=="normal_native_independent_first_order_qualification_v1" and qualified["status"]=="PASS"
        and qualified["native_reference_disabled_source_sha256"]==REFERENCE[1] and qualified["process_helper_sha256"]==PROCESS[1]
        and qualified["supervisor_worker_sha256"]=="2b08f2198c5bb0f7db5ef0ee61d722cdd53437e85928fe96a7c916d0e5e6cb19"
        and qualified["qualifier_worker_sha256"]=="afbf8fb3c3efd201c11951590f65205e3d5569f2111305aaf769dd503357ea3a"
        and qualified["external_whole_child_resource_closure_verified"] is True and qualified["wait4_closed"] is True
        and qualified["child_exit_code"]==0 and not qualified["cleanup_errors"] and not qualified["restoration_errors"]
        and all(qualified[k] is True for k in ("full_native_context_supported","both_stages_checked","native_own_CE_gradient_Adam_RNG_checked",
            "fresh_seed_constructor_reset_checked","independent_parameter_and_Adam_ownership_checked","four_resident_models_and_histories_checked"))
        and qualified["member_seeds"]==list(reference.MEMBER_SEEDS) and qualified["input_identity"]==reference.INPUT_IDENTITY,
        "Complete exact normal native first-order support and owned resource closure required")
    member_keys=[str(m) for m in range(len(reference.MEMBER_SEEDS))]
    warm_kind,final_kind="W"+str(reference.W_UPDATES),"SR"+str(reference.SR_UPDATES)
    per_member=reference.W_UPDATES+reference.SR_UPDATES
    expected_total=len(member_keys)*per_member
    expected_checkpoints=2*len(member_keys)
    require(admission["schema"]=="root_fresh_native_single_independent4_SR_admission_v1" and admission["root_fit_authorized"] is True
        and admission["source_review_approved"] is True and admission["worker_sha256"]==RELEASED_WORKER_SHA256
        and admission["protocol_sha256"]==PROTOCOL_SHA256==sha((root/REFERENCE[0]).parent/"PROTOCOL.json")
        and admission["member_seeds"]==list(reference.MEMBER_SEEDS) and admission["W_updates"]==reference.W_UPDATES
        and admission["W_local_updates"]==reference.W_LOCAL and admission["SR_updates"]==reference.SR_UPDATES
        and admission["single_alias_member"]==0 and admission["all_W400_frozen_before_SR_decode"] is True
        and admission["old_checkpoint_inputs"] is False and admission["retry"] is False
        and admission["A_scoring"] is False and admission["VALID_TEST_access"] is False
        and admission["device"]=="cuda:0" and admission["CUDA_VISIBLE_DEVICES"]==qualified["CUDA_VISIBLE_DEVICES"],
        "Unmodified fixed native worker admission/protocol required")
    released_row=scope["released_worker"]
    flag_only=(root/REFERENCE[0]).read_bytes().replace(b"SOURCE_RELEASED = False",b"SOURCE_RELEASED = True",1)
    require(hashlib.sha256(flag_only).hexdigest()==RELEASED_WORKER_SHA256 and released_row["sha256"]==RELEASED_WORKER_SHA256
        and released_row["bytes"]==len(flag_only),"Exact flag-only native V2 release required")
    worker=ordinary.bound(root,released_row)
    require(sha(worker.parent/"PROTOCOL.json")==admission["protocol_sha256"],"Released worker protocol copy differs")
    require(output==process.new_path(root,scope["supervisor_output_relative"]),"Assigned new supervision output differs")
    child_output=process.new_path(root,scope["worker_output_relative"]);require(output!=child_output,"Two distinct fresh outputs required")
    python=qualified["runtime"]["python_resolved"]
    require(Path(python).is_absolute() and str(Path(python).resolve())==python and Path(python).is_file(),"Exact qualified normal interpreter required")
    caps,watchdog=admission["resource_limits"],admission["external_watchdog_seconds"]
    grace,reap_timeout=scope["termination_grace_seconds"],scope["termination_reap_timeout_seconds"]
    process.validate_caps(caps,watchdog,grace,reap_timeout)
    require(admission["external_watchdog_required"] is True,"Existing native worker external watch required")
    uuid=admission["CUDA_VISIBLE_DEVICES"]
    require(isinstance(uuid,str) and uuid.startswith("GPU-") and "," not in uuid,"One actual qualified physical UUID required")
    # Resource values, free-floor/aggregate placement and fit projection remain
    # root-owned in the immutable worker admission; no new availability query.
    owner=ordinary.CreatedOutput(output,root);token=owner.creator_token
    receipt={"schema":"fresh_native_single_independent4_owned_supervision_terminal_v1","status":"RUNNING",
        "supervisor_worker_sha256":sha(__file__),"native_reference_disabled_sha256":REFERENCE[1],"worker_sha256":RELEASED_WORKER_SHA256,
        "process_helper_sha256":PROCESS[1],"root_scope_sha256":sha(scope_path),"worker_admission":scope["worker_admission"],
        "protocol_sha256":PROTOCOL_SHA256,"worker_output_relative":scope["worker_output_relative"],
        "actual_combined_native_qualification":qualification_row,"CUDA_VISIBLE_DEVICES":uuid,"launch_attempts":0,"children":[],
        "numeric_imports_in_supervisor":False,"comparison_complete":False,"A_scoring":False,"VALID_TEST_access":False,
        "outcomes":{},"worker_derived_expected_counts":{"members":len(member_keys),"W_callbacks":len(member_keys)*reference.W_UPDATES,
            "SR_callbacks":len(member_keys)*reference.SR_UPDATES,"native_callbacks":expected_total,"native_Adam_steps":expected_total,
            "native_backward_APIs":expected_total,"full_checkpoint_writes":expected_checkpoints}}
    def save():
        receipt["supervisor_elapsed_seconds"]=time.monotonic()-STARTED
        ordinary.atomic(owner.verify(token)/"RESULT.json",receipt)
    old_handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    def interrupted(signum,frame):raise InterruptedError("Owned native cohort supervisor interrupted: "+str(signum))
    code=1
    try:
        for signum in old_handlers:signal.signal(signum,interrupted)
        receipt["launch_attempts"]=1;save()
        child=process.launch(root,owner,token,"native_cohort",python,worker,
            ["--execute-authorized","--source-root",TARGET,"--admission",str(admission_path),"--output",str(child_output)],
            dict(os.environ,CUDA_VISIBLE_DEVICES=uuid),receipt["children"])
        save()
        while not process.watch(child,watchdog,grace,reap_timeout):time.sleep(0.25)
        save()
        result_row=process.immutable_descriptor(root,child_output/"RESULT.json")
        result=json.loads(ordinary.bound(root,result_row).read_text())
        receipt.update(worker_result=result_row,worker_reported_status=result.get("status"),
            worker_reported_operation_accounting=result.get("operation_accounting"),
            worker_reported_checkpoint_cost=result.get("checkpoint_cost"),
            worker_reported_W400_states=result.get("W400_states"),worker_reported_endpoints=result.get("endpoints"),
            worker_reported_restoration_errors=result.get("restoration_errors"),
            worker_reported_whole_process_resources=result.get("whole_process_resources"))
        save();process.resource_closure(child,result,caps)
        require(result["status"]=="FRESH_NATIVE_SINGLE_INDEPENDENT4_COMPLETE_A_CLOSED" and result["comparison_complete"] is True
            and result["all_four_complete"] is True and result["all_W400_frozen_before_SR_decode"] is True and result["SR_labels_received"] is True
            and result["A_scoring"] is False and result["VALID_TEST_access"] is False and result["old_checkpoint_inputs"] is False
            and result["original_paper_scores_changed"] is False and not result["restoration_errors"],"Complete fixed fresh cohort successful result required")
        flags=("public_and_W_inputs_unchanged","public_and_SR_inputs_unchanged","cached_own_W400_model_Adam_RNG_immutable",
            "caller_RNG_restored","backend_restored","reversible_threads_restored","original_sources_and_process_state_verified",
            "all_acquired_W400_file_bytes_unchanged")
        require(all(result[k] is True for k in flags),"Exact native worker restoration/identity flags required")
        identity=result["created_output_identity"]
        require(not child_output.is_symlink() and identity["path"]==str(child_output)
            and (child_output.stat().st_dev,child_output.stat().st_ino)==(identity["device"],identity["inode"]),"Owned native worker directory identity changed")
        counts=result["operation_accounting"]
        require(set(counts)==set(member_keys) and all(set(counts[m])=={"W","S_R"} and all(
            counts[m][phase]==dict.fromkeys(("callback_attempts","backward_attempts","Adam_attempts","completed_updates"),updates)
            for phase,updates in (("W",reference.W_UPDATES),("S_R",reference.SR_UPDATES))) for m in member_keys),
            "Complete worker-derived per-member native horizons/attempt counts required")
        require(result["training_callbacks_W"]==len(member_keys)*reference.W_UPDATES and result["training_callbacks_SR"]==len(member_keys)*reference.SR_UPDATES
            and result["total_training_callbacks"]==result["native_Adam_steps"]==result["backward_APIs"]==expected_total
            and result["serving_callbacks"]==0,"Complete native cohort accounting differs")
        require(set(result["W400_states"])==set(result["endpoints"])==set(result["independent4_endpoints"])==set(member_keys)
            and result["independent4_endpoints"]==result["endpoints"],"All fixed W400 and final native outcomes required")
        cost=result["checkpoint_cost"]
        require(cost["write_attempts"]==cost["completed_frozen_writes"]==expected_checkpoints
            and cost["single_alias_extra_checkpoint_writes"]==0
            and cost["CPU_snapshots_hashing_and_checkpoint_work_included_in_whole_process_resources"] is True,"Exactly worker-declared checkpoint custody/cost required")
        checkpoint_bytes=0
        for member in range(len(member_keys)):
            key=str(member)
            warm=checkpoint(root,process,child_output,result["W400_states"][key],member,warm_kind)
            final=checkpoint(root,process,child_output,result["endpoints"][key],member,final_kind)
            receipt["outcomes"][key]={"seed":reference.MEMBER_SEEDS[member],"W400":warm,"SR2300":final}
            checkpoint_bytes+=warm["bytes"]+final["bytes"];save()
        require(cost["partial_and_complete_checkpoint_bytes_observed"]==checkpoint_bytes,"All checkpoint bytes/accounting must match retained outcomes")
        require(result["single_alias"]=={"member":0,"seed":reference.MEMBER_SEEDS[0],"endpoint":result["endpoints"]["0"],"additional_fit":False},
            "SINGLE remains fixed member0 endpoint alias, no fifth fit or selection")
        run_row=process.immutable_descriptor(root,child_output/"RUN.json");recipe=json.loads(ordinary.bound(root,run_row).read_text())
        require(recipe["source_sha256"]==RELEASED_WORKER_SHA256 and recipe["protocol_sha256"]==admission["protocol_sha256"]
            and recipe["admission_sha256"]==scope["worker_admission"]["sha256"] and recipe["member_seeds"]==list(reference.MEMBER_SEEDS)
            and recipe["single_alias_member"]==0 and recipe["input_identity"]==reference.INPUT_IDENTITY
            and recipe["native_source_pin"]==list(reference.PINS["native"])
            and recipe["W_local_updates"]==reference.W_LOCAL and recipe["W_global_updates"]==reference.W_UPDATES-reference.W_LOCAL
            and recipe["SR_global_updates"]==reference.SR_UPDATES and recipe["new_independent_W_acquisitions"]==len(member_keys)
            and recipe["copied_shared_history"] is False and recipe["S_R_received_only_after_all_W400_freeze"] is True
            and recipe["A_labels_received"] is False and recipe["A_scoring"] is False and recipe["VALID_TEST_access"] is False,
            "Exact source/protocol/admission/own-acquisition/input recipe custody required")
        require(sha(scope_path)==receipt["root_scope_sha256"] and sha(admission_path)==scope["worker_admission"]["sha256"]
            and sha(worker)==RELEASED_WORKER_SHA256 and all(sha(root/p)==h for p,h in (PROCESS,REFERENCE)),"Immutable scope/admission/source bytes changed")
        receipt.update(status="FRESH_NATIVE_SINGLE_ENS4_COHORT_COMPLETE_A_CLOSED",comparison_complete=True,worker_result=result_row,worker_run=run_row,
            operation_accounting=counts,checkpoint_cost=cost,single_alias=result["single_alias"],retained_checkpoint_bytes=checkpoint_bytes,
            native_callbacks=expected_total,native_backward_APIs=expected_total,native_Adam_steps=expected_total,
            full_checkpoint_writes=expected_checkpoints,serving_callbacks=0,child_exit_code=0,wait4_closed=True,
            external_whole_child_resource_closure_verified=True,root_resource_limits=caps)
        code=0
    except BaseException as error:receipt.update(status="FAIL_NATIVE_SINGLE_ENS4_COHORT",comparison_complete=False,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        for signum in old_handlers:signal.signal(signum,signal.SIG_IGN)
        errors=process.cleanup(receipt["children"],grace,reap_timeout);receipt["cleanup_errors"]=errors
        if errors:receipt.update(status="FAIL_NATIVE_SINGLE_ENS4_COHORT",comparison_complete=False);code=1
        for signum,handler in old_handlers.items():signal.signal(signum,handler)
        if code:receipt["comparison_complete"]=False
        try:save();owner.freeze(token)
        except BaseException as error:
            code=1;receipt.update(status="FAIL_NATIVE_SINGLE_ENS4_COHORT",comparison_complete=False,output_finalization_error=str(error))
            try:save();owner.freeze(token)
            except BaseException as recovery_error:receipt["output_failure_receipt_error"]=str(recovery_error)
    print(json.dumps({"status":receipt["status"],"comparison_complete":receipt["comparison_complete"],"child_exit_code":receipt.get("child_exit_code"),"error":receipt.get("error")}))
    return code


if __name__=="__main__":raise SystemExit(main())
