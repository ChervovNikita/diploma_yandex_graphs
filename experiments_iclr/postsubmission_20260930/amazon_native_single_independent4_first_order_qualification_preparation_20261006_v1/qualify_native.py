"""Disabled16-callback native local/global first-order engineering qualifier.

Four fresh primary native models/Adams; same-member independent standard CE
replays. All virtual updated states discarded, no W400 acquisition or fit.
"""
import time
STARTED = time.monotonic()
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import sys
import traceback
from types import SimpleNamespace

SOURCE_RELEASED = False
TARGET = "/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930"
REFERENCE = ("amazon_native_single_independent4_sr_reference_preparation_20261006_v2/native_reference.py", "0d8e677b4e2c9b3616a0795cc8f1b55758cc63d7048b2d42f614a295a1c1a6ef")
ATOL, RTOL = 2e-6, 2e-5


def require(value, message):
    if not value: raise RuntimeError(message)


def compare(actual, expected):
    import torch
    if isinstance(expected, torch.Tensor):
        require(isinstance(actual, torch.Tensor) and actual.dtype == expected.dtype and actual.shape == expected.shape, "Compared native coordinate structure differs")
        a, b = actual.detach().cpu(), expected.detach().cpu()
        require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
            and (torch.allclose(a, b, atol=ATOL, rtol=RTOL) if b.is_floating_point() else torch.equal(a, b)), "Native first-order complete-coordinate mismatch")
        return float((a.double()-b.double()).abs().max()) if b.numel() else 0.0
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), "Compared native dictionary differs")
        return max((compare(actual[k], expected[k]) for k in expected), default=0.0)
    if isinstance(expected, (list, tuple)):
        require(isinstance(actual, type(expected)) and len(actual) == len(expected), "Compared native sequence differs")
        return max((compare(a,b) for a,b in zip(actual, expected)), default=0.0)
    require(actual == expected, "Compared native scalar differs")
    return 0.0


def record(reference, model, optimizer, helpers, device):
    return {"state": reference.state(model, optimizer, helpers, device),
        "gradients": {n: helpers._cpu_tree(p.grad) for n,p in model.named_parameters()}}


def clocks(model, optimizer, local_steps, global_steps):
    """Observed real discarded-step counters, no fabricated W400 histories."""
    for name, parameter in model.named_parameters():
        expected = local_steps if name.startswith("pred_local.") else (global_steps if name.startswith(("global_attn.", "ln.", "pred_global.")) else local_steps+global_steps)
        if expected == 0: require(parameter not in optimizer.state, "Unvisited native branch gained Adam state")
        else: require(parameter in optimizer.state and float(optimizer.state[parameter]["step"].item()) == expected, "Own native stage-specific Adam clock differs: "+name)


def independent_step(model, optimizer, data, audit, helpers):
    """Independent ordinary cross_entropy oracle; no production step/loss call."""
    import torch
    import torch.nn.functional as F
    model.train(); optimizer.zero_grad(set_to_none=True)
    audit.attempt("callback_attempts", "independent_complete_native_forward_attempt")
    logits = model(data["features"], data["edge_index"])
    require(logits.shape == (24492, 5) and logits.dtype == torch.float32, "Oracle must use real complete native trajectory")
    helpers._finite(logits, "Nonfinite independent native logits")
    loss = F.cross_entropy(logits.index_select(0,data["training_ids"]), data["training_targets"])
    helpers._finite(loss, "Nonfinite independent native CE")
    audit.attempt("backward_attempts", "independent_standard_CE_backward_attempt"); loss.backward()
    for name, parameter in model.named_parameters():
        dormant = name.startswith("pred_local.") if model._global else name.startswith(("global_attn.", "ln.", "pred_global."))
        if dormant: require(parameter.grad is None, "Independent dormant native gradient appeared")
        else:
            require(parameter.requires_grad and parameter.grad is not None, "Independent active native gradient absent")
            helpers._finite(parameter.grad, "Nonfinite independent native gradient")
    audit.attempt("Adam_attempts", "independent_native_Adam_attempt"); optimizer.step()
    for p in model.parameters(): helpers._finite(p, "Nonfinite independent native parameter")
    for entry in optimizer.state.values():
        for value in entry.values():
            if isinstance(value,torch.Tensor): helpers._finite(value, "Nonfinite independent native Adam state")
    return float(loss.detach())


def native_checks(reference, native, helpers, accessor, root, device, primary_meter, oracle_meter, receipt, save):
    import torch
    public_b = (root/reference.PUBLIC_B_RELATIVE).resolve()
    preflight = helpers._public_identity_before_w(accessor,root,public_b,reference.INPUT_IDENTITY)
    require(preflight["public_b_manifest_sha256"]==reference.INPUT_IDENTITY["public_b_manifest_sha256"], "Exact original label manifest required before W access")
    w = accessor.load_public_w(root,public_b,device=device)
    require(w["W_ids"].numel()==4898 and w["provenance"]["preprocessing"]==preflight["preprocessing"]
        and w["provenance"]["public_b_manifest"]["sha256"]==reference.INPUT_IDENTITY["public_b_manifest_sha256"], "Fixed original public/W label context required")
    local_data = {"features":w["features"],"edge_index":w["edge_index"],"training_ids":w["W_ids"],"training_targets":w["W_labels"]}
    del w
    local_inputs = {k:v.detach().clone() for k,v in local_data.items()}
    models, optimizers, construction = [],[],[]
    for member,seed in enumerate(reference.MEMBER_SEEDS):
        receipt["primary_native_constructor_attempts"] += 1
        primary_meter.member,primary_meter.phase,primary_meter.update = member,"setup",0
        primary_meter.event("primary_fresh_native_constructor_attempt",seed=seed);save()
        model,optimizer,created = reference.fresh(native,helpers,seed,device)
        models.append(model);optimizers.append(optimizer);construction.append(created)
    require(all(not optimizer.state for optimizer in optimizers), "All four primary native Adams must truly start empty")
    reference.independent_ownership(models,optimizers)
    for model,optimizer in zip(models,optimizers): clocks(model,optimizer,0,0)
    after_local, results, durations = [], {"local":{},"global":{}}, {"local":[],"global":[]}
    def check_member(member, stage, data, incoming):
        model,optimizer = models[member],optimizers[member]
        frozen = helpers._cpu_tree(incoming)
        primary_meter.member,primary_meter.phase,primary_meter.update = member,("S_R" if stage=="global" else "W"),1
        oracle_meter.member,oracle_meter.phase,oracle_meter.update = member,primary_meter.phase,1
        reference.restore_rng(incoming["rng"],device);model._global = stage=="global"
        torch.cuda.synchronize(device); started = time.monotonic()
        production_loss = reference.step(model,optimizer,data,primary_meter,helpers)
        torch.cuda.synchronize(device);durations[stage].append(time.monotonic()-started)
        primary_meter.counts[str(member)][primary_meter.phase]["completed_updates"] = 1
        production = record(reference,model,optimizer,helpers,device)
        clocks(model,optimizer,1,1 if stage=="global" else 0)
        # This temporary oracle restores ONLY the same member's incoming image.
        # It is not an independently acquired W400 history or a baseline fit.
        receipt["oracle_native_constructor_attempts"] += 1
        oracle_meter.event("same_member_oracle_native_constructor_attempt",seed=reference.MEMBER_SEEDS[member]);save()
        oracle,oracle_optimizer,_ = reference.fresh(native,helpers,reference.MEMBER_SEEDS[member],device)
        oracle.load_state_dict(helpers._cpu_tree(incoming["model"]),strict=True)
        oracle_optimizer.load_state_dict(helpers._cpu_tree(incoming["Adam"]))
        oracle._global = stage=="global";reference.restore_rng(incoming["rng"],device)
        independent_loss = independent_step(oracle,oracle_optimizer,data,oracle_meter,helpers)
        oracle_meter.counts[str(member)][oracle_meter.phase]["completed_updates"] = 1
        expected = record(reference,oracle,oracle_optimizer,helpers,device)
        clocks(oracle,oracle_optimizer,1,1 if stage=="global" else 0)
        error = compare(production,expected);reference.equal(production["state"]["rng"],expected["state"]["rng"])
        reference.equal(incoming,frozen)
        require(any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for name,p in model.named_parameters()
            if name.startswith(("lin_in.","h_lins.","local_convs.","global_attn."))), "Native propagation gradients are trivial")
        require(all(torch.equal(data[k],v) for k,v in (local_inputs if stage=="local" else sr_inputs).items()), "Native data tensors changed")
        reference.independent_ownership(models,optimizers)
        results[stage][str(member)] = {"seed":reference.MEMBER_SEEDS[member],"complete_gradient_parameter_Adam_max_error":error,
            "logical_RNG_exact":True,"production_loss":production_loss,"independent_standard_CE_loss":independent_loss,
            "observed_local_steps":1,"observed_global_steps":1 if stage=="global" else 0,
            "synchronized_production_update_elapsed_seconds":durations[stage][-1]}
        receipt["native_checks_completed"] = results;save()
        del oracle,oracle_optimizer,expected
        return production["state"]
    for member,(model,optimizer,created) in enumerate(zip(models,optimizers,construction)):
        reference.restore_rng(created["after_Adam_constructor"],device)
        incoming = reference.state(model,optimizer,helpers,device)
        after_local.append(check_member(member,"local",local_data,incoming))
    # Engineering scope explicitly permits real SR targets for discarded checks.
    # No W400 state/acquisition is fabricated or represented by these2updates.
    loaded = accessor.load_public_b(root,public_b,device=device)
    s,r = loaded["inner_indices"],loaded["query_indices"]
    ids,order = torch.sort(torch.cat((s,r)));targets = torch.cat((loaded["inner_labels"],loaded["query_labels"]))[order]
    require(s.numel()==2449 and r.numel()==2450 and not bool(torch.isin(s,r).any()) and ids.numel()==4899
        and not bool(torch.isin(ids,torch.cat((loaded["W_ids"],loaded["A_ids"]))).any())
        and torch.equal(loaded["features"],local_data["features"]) and torch.equal(loaded["edge_index"],local_data["edge_index"])
        and loaded["provenance"]["preprocessing"]==preflight["preprocessing"]
        and loaded["provenance"]["public_b_manifest"]["sha256"]==reference.INPUT_IDENTITY["public_b_manifest_sha256"], "Only original fixed SR on complete unchanged native context")
    global_data = {"features":local_data["features"],"edge_index":local_data["edge_index"],"training_ids":ids,"training_targets":targets}
    del loaded,s,r,ids,order,targets
    sr_inputs = {k:v.detach().clone() for k,v in global_data.items()}
    for member,(model,optimizer) in enumerate(zip(models,optimizers)):
        reference.equal(model.state_dict(),after_local[member]["model"]);reference.equal(optimizer.state_dict(),after_local[member]["Adam"])
        reference.restore_rng(after_local[member]["rng"],device);model._global=True
        incoming = reference.state(model,optimizer,helpers,device)
        check_member(member,"global",global_data,incoming)
        reference.equal(after_local[member]["model"]["pred_local.weight"],model.state_dict()["pred_local.weight"])
        reference.equal(after_local[member]["model"]["pred_local.bias"],model.state_dict()["pred_local.bias"])
    reference.independent_ownership(models,optimizers)
    require(all(torch.equal(local_data[k],v) for k,v in local_inputs.items()) and all(torch.equal(global_data[k],v) for k,v in sr_inputs.items()), "Public/role arrays changed")
    require(all(len(result)==4 for result in results.values()) and all(optimizer.state for optimizer in optimizers), "Both stages/all four real native histories required")
    return {"results":results,"native_local_update_elapsed_seconds_max":max(durations["local"]),
        "native_global_update_elapsed_seconds_max":max(durations["global"]),"four_resident_models_and_histories_checked":True,
        "per_parameter_clock_semantics_checked":True,"clock_sequence":"Real empty0 -> one local update -> one global update; local-head1/global-only1/backbone2",
        "W400_or_SR2300_fits_performed":False,"400_2700_fit_horizon_boundary_checks_executed":False,
        "all_updated_states_discarded":True,"trained_checkpoint_writes":0}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--execute-authorized",action="store_true");parser.add_argument("--source-root")
    parser.add_argument("--scope");parser.add_argument("--output")
    args=parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({"status":"DISABLED_NORMAL_NATIVE16_FIRST_ORDER_QUALIFIER","SOURCE_RELEASED":False,"numeric_imports":False}));return
    require(SOURCE_RELEASED is False and args.source_root==TARGET and socket.gethostname()=="peptide"
        and args.scope and args.output and sys.dont_write_bytecode
        and not any(n=="torch" or n.startswith("torch.") for n in sys.modules), "Exact normal77 fresh/-B engineering child required")
    root=Path(args.source_root).resolve();names=[]
    require(hashlib.sha256((root/REFERENCE[0]).read_bytes()).hexdigest()==REFERENCE[1], "Exact disabled native source differs")
    spec=importlib.util.spec_from_file_location("_native16_exact_reference",root/REFERENCE[0]);reference=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=reference;names.append(spec.name);spec.loader.exec_module(reference)
    require(reference.SOURCE_RELEASED is False, "Native fit source must remain disabled")
    ordinary=reference.load(root,reference.ORDINARY_UTILITIES,"_native16_path_utilities",names)
    root,output,scope_path=ordinary.deliberate_paths(SimpleNamespace(source_root=args.source_root,output=args.output,admission=args.scope))
    require(scope_path.stat().st_mode&0o222==0, "Immutable actual root engineering scope required")
    scope=json.loads(scope_path.read_text())
    require(scope["engineering_execution_authorized"] is True and scope["fit_authorized"] is False
        and scope["W_SR_label_access_engineering_authorized"] is True and scope["A_scoring"] is False and scope["VALID_TEST_access"] is False
        and scope["native_reference_source_review_approved"] is True and scope["native_qualifier_source_review_approved"] is True
        and scope["native_reference_worker_sha256"]==REFERENCE[1] and scope["native_qualifier_worker_sha256"]==ordinary.sha(__file__)
        and scope["member_seeds"]==list(reference.MEMBER_SEEDS), "Reviewed bounded native engineering checks only")
    caps=scope["resource_limits"];watchdog=scope["external_watchdog_seconds"]
    require(all(type(caps[k]) in (int,float) and math.isfinite(caps[k]) and caps[k]>0 for k in
        ("max_elapsed_seconds","max_process_rss_bytes","max_cuda_allocated_bytes","max_cuda_reserved_bytes"))
        and type(watchdog) in (int,float) and math.isfinite(watchdog) and watchdog>caps["max_elapsed_seconds"], "Root-frozen caps/external watchdog required")
    uuid=scope["CUDA_VISIBLE_DEVICES"]
    require(isinstance(uuid,str) and uuid.startswith("GPU-") and "," not in uuid and os.environ.get("CUDA_VISIBLE_DEVICES")==uuid, "Single root-pinned physical UUID required")
    owner=ordinary.CreatedOutput(output,root);token=owner.creator_token
    receipt={"schema":"normal_native_independent_first_order_qualification_v1","status":"RUNNING","worker_sha256":ordinary.sha(__file__),
        "native_reference_disabled_source_sha256":REFERENCE[1],"root_scope_sha256":ordinary.sha(scope_path),"member_seeds":list(reference.MEMBER_SEEDS),
        "CUDA_VISIBLE_DEVICES":uuid,"model_fits":0,"persistent_updates":0,"W400_acquisitions":0,"trained_checkpoints_saved":0,
        "A_scoring":False,"VALID_TEST_access":False,"native_checks_completed":{},
        "primary_native_constructor_attempts":0,"oracle_native_constructor_attempts":0}
    def resources():
        row=reference.resources("cuda:0");row["elapsed_seconds"]=time.monotonic()-STARTED;return row
    def save():
        receipt["elapsed_seconds_including_imports_setup_checks_restore"]=time.monotonic()-STARTED
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():receipt["whole_process_resources"]=resources()
        ordinary.atomic(owner.verify(token)/"RESULT.json",receipt)
    class Meter(reference.Audit):
        label="primary"
        def event(self,kind,**fields):super().event(kind,qualification_variant=self.label,**fields)
    primary,oracle=Meter(output),Meter(output);oracle.label="independent_oracle"
    old_path,python_rng=list(sys.path),random.getstate()
    old_env_present,old_env="CUBLAS_WORKSPACE_CONFIG" in os.environ,os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    old_handler,old_timer=signal.getsignal(signal.SIGALRM),signal.getitimer(signal.ITIMER_REAL)
    require(old_timer==(0.0,0.0),"Fresh child timer required")
    process=helpers=backend_before=old_threads=rng_before=None
    code,success=1,None
    try:
        def expired(signum,frame):raise TimeoutError("Root-frozen native qualifier deadline exceeded")
        signal.signal(signal.SIGALRM,expired);remaining=caps["max_elapsed_seconds"]-(time.monotonic()-STARTED)
        require(remaining>0,"No remaining root time");signal.setitimer(signal.ITIMER_REAL,remaining)
        config=scope["process_configuration"];os.environ["CUBLAS_WORKSPACE_CONFIG"]=config["CUBLAS_WORKSPACE_CONFIG"]
        sys.path.insert(0,scope["runtime"]["site_packages"])
        import torch
        require(str(Path(torch.__file__).resolve())==scope["runtime"]["torch_module_path"]
            and str(Path(sys.executable).resolve())==scope["runtime"]["python_resolved"] and not torch.cuda.is_initialized(),"Pinned normal runtime/pre-CUDA required")
        process=reference.load(root,reference.PINS["process"],"_native16_runtime_process",names);process.torch=torch
        backend_before,old_threads=process.backend_snapshot(),torch.get_num_threads()
        torch.use_deterministic_algorithms(config["deterministic_algorithms_enabled"],warn_only=config["warn_only"])
        torch.set_num_threads(config["intra_op_threads"]);torch.set_num_interop_threads(config["interop_threads"])
        require(not torch.cuda.is_initialized(),"Backend config must precede CUDA")
        identity=process.runtime_identity();require(identity==scope["expected_backend_runtime_metadata"] and torch.cuda.device_count()==1,"Qualified native runtime/singleGPU differs")
        free,total=torch.cuda.mem_get_info("cuda:0")
        require(type(scope["minimum_initial_cuda_free_bytes"]) is int and scope["minimum_initial_cuda_free_bytes"]>0
            and free>=scope["minimum_initial_cuda_free_bytes"],"Native qualifier initial free floor failed")
        receipt["initial_cuda_memory"]={"free_bytes":free,"total_bytes":total,"required_free_bytes":scope["minimum_initial_cuda_free_bytes"]}
        helpers=reference.load(root,reference.PINS["scientific_helpers"],"_native16_state_helpers",names)
        accessor=reference.load(root,reference.PINS["accessor"],"_native16_train_only_accessor",names)
        native=reference.load(root,reference.PINS["native"],"_native16_full_native_architecture",names)
        require(helpers.SOURCE_RELEASED is False and accessor.SOURCE_RELEASED is True,"Original helper/accessor gates differ")
        rng_before=helpers._rng("cuda:0");save()
        checked=native_checks(reference,native,helpers,accessor,root,"cuda:0",primary,oracle,receipt,save)
        require(all(all(all(value==1 for value in phase.values()) for phase in phases.values()) for phases in primary.counts.values())
            and primary.counts==oracle.counts and receipt["primary_native_constructor_attempts"]==4
            and receipt["oracle_native_constructor_attempts"]==8,"Exact16full-native forward/backward/temporaryAdam schedule and12constructors required")
        success={"full_native_context_supported":True,"both_stages_checked":True,"native_own_CE_gradient_Adam_RNG_checked":True,
            "fresh_seed_constructor_reset_checked":True,"independent_parameter_and_Adam_ownership_checked":True,**checked,
            "input_identity":reference.INPUT_IDENTITY,"runtime":{"hostname":socket.gethostname(),"python_resolved":str(Path(sys.executable).resolve()),
            "site_packages":scope["runtime"]["site_packages"],"torch_module_path":str(Path(torch.__file__).resolve())},
            "process_configuration":config,"backend_snapshot":process.backend_snapshot(),"backend_runtime_metadata":identity}
    except BaseException as error:receipt.update(status="FAIL",error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);receipt["primary_attempts"],receipt["independent_oracle_attempts"]=primary.snapshot(),oracle.snapshot()
        errors=[]
        def restore(name,fn):
            try:fn();receipt[name]=True
            except BaseException as error:errors.append({"restoration":name,"error":str(error),"traceback":traceback.format_exc()})
        if rng_before is not None:
            def rng():reference.restore_rng(rng_before,"cuda:0");reference.equal(helpers._rng("cuda:0"),rng_before)
            restore("caller_RNG_restored",rng)
        if backend_before is not None:restore("backend_restored",lambda:process.backend_restore(backend_before))
        if old_threads is not None:
            def threads():torch.set_num_threads(old_threads);require(torch.get_num_threads()==old_threads,"Intra-op restoration differs")
            restore("reversible_threads_restored",threads)
        random.setstate(python_rng);sys.path[:]=old_path
        if old_env_present:os.environ["CUBLAS_WORKSPACE_CONFIG"]=old_env
        else:os.environ.pop("CUBLAS_WORKSPACE_CONFIG",None)
        signal.signal(signal.SIGALRM,old_handler);signal.setitimer(signal.ITIMER_REAL,*old_timer)
        for name in names:sys.modules.pop(name,None)
        restore("original_sources_and_process_state_verified",lambda:require(random.getstate()==python_rng and sys.path==old_path
            and ("CUBLAS_WORKSPACE_CONFIG" in os.environ)==old_env_present and os.environ.get("CUBLAS_WORKSPACE_CONFIG")==old_env
            and signal.getsignal(signal.SIGALRM)==old_handler and signal.getitimer(signal.ITIMER_REAL)==old_timer
            and ordinary.sha(root/REFERENCE[0])==REFERENCE[1] and all(ordinary.sha(root/p)==h for p,h in [reference.ORDINARY_UTILITIES,*reference.PINS.values()]),
            "Original sources/reversible process state differ"))
        receipt["restoration_errors"]=errors;receipt["interop_restoration"]="One-time fresh-child configuration; not reversible."
        if success is not None and not errors:
            try:
                row=resources();require(all(row[m]<=caps[c] for m,c in (("elapsed_seconds","max_elapsed_seconds"),("process_peak_rss_bytes","max_process_rss_bytes"),
                    ("cuda_peak_allocated_bytes","max_cuda_allocated_bytes"),("cuda_peak_reserved_bytes","max_cuda_reserved_bytes"))),"Native whole-process resource cap exceeded")
                receipt.update(success);receipt["status"]="PASS";code=0
            except BaseException as error:receipt.update(status="FAIL",error=str(error),traceback=traceback.format_exc())
        if code:receipt["full_native_context_supported"]=False
        try:save();owner.freeze(token)
        except BaseException as error:
            code=1;receipt.update(status="FAIL",full_native_context_supported=False,output_finalization_error=str(error))
            try:save();owner.freeze(token)
            except BaseException as recovery_error:receipt["output_failure_receipt_error"]=str(recovery_error)
    print(json.dumps({"status":receipt["status"],"model_fits":0,"persistent_updates":0,"error":receipt.get("error")}))
    raise SystemExit(code)


if __name__=="__main__":main()
