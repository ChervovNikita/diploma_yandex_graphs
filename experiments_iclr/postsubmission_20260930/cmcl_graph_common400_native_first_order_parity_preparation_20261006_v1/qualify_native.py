"""Disabled once-only common400 CMCL native monolithic/replay/repeat parity.

Reuse ordinary paths/output/resources/RNG, reviewed process helpers and G0
context/native port. One independent monolithic scalar, streamed step and fixed
repeat: 20 native callbacks, 9 native and 2 small gradient APIs. No fit/scores.
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
REPOSITORY = "/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs"
PHASE = REPOSITORY + "/experiments_iclr/postsubmission_20260930"
DEVICE, ATOL, RTOL = "cuda:0", 2e-6, 2e-5
COMMON = {"path": "amazon_learnability_responsibility_strict_scientific_execution_root_20261006_v2/output/scientific_run/initial.pt",
    "bytes": 109671738, "sha256": "2e0e9b44767abc12b3dc896986ea2d4faeaa667c8682b95929f927d10718154f"}
ORIGIN = {"path": "amazon_common400_descriptor_binding_root_20261006_v1/ORIGIN_RUN.json",
    "bytes": 6402, "sha256": "2e18a77081437e47c0c4a0d59cb07b2a15500f660066cbed1c980f613c028ca5"}
PINS = {
    "caller": ("cmcl_graph_common400_H16_callable_preparation_20261006_v1/cmcl_H16.py", "d5fc0dd9c341eb67255dff856782eee58709e00912ca3afcdcf2f854052e55a7"),
    "cmcl": ("cmcl_graph_objective_transplant_source_preparation_20261006_v1/cmcl_loss.py", "4f04eb87381512c9f7506b6458917b83fd12228f3c7651c6291e902c99fe6ac9"),
    "ordinary": ("amazon_ordinary_shared_bank_own_pool_reference_preparation_20261006_v4/ordinary_reference.py", "5044a16f3f710aaf234057b115ab928d589af3a06a595940afbfa8876636c97b"),
    "compare": ("amazon_ordinary_shared_bank_first_order_qualification_preparation_20261006_v1/qualify.py", "d175dc6bf44886d874959cecd9833dc9f192f30b764bc7e9b938be83f2ad8af2"),
    "operator": ("learnability_weighted_graph_responsibility_operator_20261005_v2/response_operator.py", "fba3ca3d4bb35da0438923941d97bc4833004dd9fd385735c2352f343693b3e3"),
    "port": ("learnability_responsibility_native_amazon_sparse_port_20261005_v1/native_sparse_port.py", "a8ee35afda97afcb745c365a51f8e8647fe5e6a1350f654a5321c81e12abad86"),
    "helpers": ("amazon_learnability_responsibility_sequential_train_only_execution_preparation_20261006_v3/six_arm_worker.py", "f19a94be4102e74f30d2b779ca602e288cf40c446ae367ce9d9ab44091569ad1"),
    "accessor": ("amazon_learnability_responsibility_engineering_accessor_20261005_v1/train_only_accessor.py", "9360e69753b362eb66ac89f133f00addaf15ab0e8c56314c75d7dd5e96fecb85"),
    "native": ("amazon_polynormer_paired_family_source_preparation_20261003_v6/reused_models/native_polynormer.py", "9b4e533f46ae7f91a23a552f88bbb47a01359e224b53996cf1a68c10a865f6a8"),
    "boundary": ("amazon_polynormer_paired_family_source_preparation_20261003_v6/reused_models/backbone_boundary_adapter.py", "699b606ead00cb7bdd9be6cd58730a0687157c40cf594af620d1edc4c93b6bac"),
    "process": ("amazon_learnability_responsibility_strict_process_scientific_runner_preparation_20261006_v3/run_scientific.py", "65c63b0b62f49bc852bc4a1db3a47196d5c46c1355a328b8ef838397dc79a6bb"),
    "custody": ("learnability_responsibility_native_numerical_worker_preparation_20261005_v3/qualify.py", "baeac626bade87bd286fe7a93a95602d8dc1f57d3257d0eeeb7e4a8e3823c688")}
EXPECTED = {"native_callbacks": 20, "native_gradient_APIs": 9, "small_logit_gradient_APIs": 2,
    "temporary_shared_SGD_maps": 3, "temporary_private_SGD_maps": 12, "temporary_endpoint_constructions": 3}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def load_pin(root, key, names, row=None):
    """Small binding adapter; scope may name immutable exact-byte mirrors."""
    path = root / (PINS[key][0] if row is None else row["path"])
    require(not path.is_symlink() and path.resolve().is_relative_to(root)
        and path.stat().st_mode & 0o222 == 0
        and hashlib.sha256(path.read_bytes()).hexdigest() == PINS[key][1]
        and (row is None or path.stat().st_size == row["bytes"] and row["sha256"] == PINS[key][1]),
        "Immutable pinned source differs: " + key)
    name = "_cmcl_native_parity_" + key
    require(name not in sys.modules, "Fresh module identity required")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; names.append(name); spec.loader.exec_module(module)
    return module


def independent_reference(modules, forward, theta, phis, S, yS, R, yR, bill, save):
    """All four full native graphs, one jointly differentiated CMCL scalar.

    Independently written equations; no CMCL helper or candidate step creates
    the expected values. The same joint grad request also returns dL/dbank.
    """
    torch = sys.modules["torch"]
    helpers = modules["helpers"]
    core = {n:v.detach().requires_grad_(True) for n,v in theta.items()}
    private = tuple({n:v.detach().requires_grad_(True) for n,v in row.items()} for row in phis)
    values=[]
    for row in private:
        values.append(forward(core,row))
    bank = torch.stack(values)
    require(bank.shape == (4,24492,5) and bank.dtype == torch.float32, "Complete native reference logits required")
    def role(indices, targets):
        logp = torch.log_softmax(bank[:,indices],dim=-1)
        ce = -logp.gather(2,targets[None,:,None].expand(4,-1,1)).squeeze(-1)
        kl = -math.log(5)-logp.mean(dim=-1)
        with torch.no_grad():
            selected = torch.argsort((ce-0.75*kl).detach(),dim=0,stable=True)[:3]
            mask = torch.zeros_like(ce,dtype=torch.bool); mask.scatter_(0,selected,True)
        return torch.where(mask,ce,0.75*kl).sum(dim=0).mean(),selected,mask
    loss_S, owners_S, mask_S = role(S,yS)
    loss_R, owners_R, mask_R = role(R,yR)
    loss = 0.5*(loss_S+loss_R)
    parameters = tuple(core.values())+tuple(v for row in private for v in row.values())
    bill["native_gradient_APIs"] += 1;save()
    derivatives = torch.autograd.grad(loss,parameters+(bank,),allow_unused=True)
    offset = len(core)
    shared = {n:torch.zeros_like(v) if d is None else d.detach()
        for (n,v),d in zip(core.items(),derivatives[:offset])}
    gradients=[]
    for row in private:
        gradients.append({n:torch.zeros_like(v) if d is None else d.detach()
            for (n,v),d in zip(row.items(),derivatives[offset:offset+len(row)])})
        offset += len(row)
    require(offset == len(parameters) and derivatives[-1] is not None, "Complete native reference gradients required")
    bill["temporary_endpoint_constructions"] += 1; bill["temporary_shared_SGD_maps"] += 1;save()
    theta_next = {n:(v-0.001*shared[n]).detach() for n,v in theta.items()}
    next_private=[]
    for row,g in zip(phis,gradients):
        bill["temporary_private_SGD_maps"] += 1;save()
        next_private.append({n:(v-0.01*g[n]).detach() for n,v in row.items()})
    next_private=tuple(next_private)
    helpers._finite_tree((bank,loss,shared,gradients,theta_next,next_private))
    helpers._check_commit(theta,phis,theta_next,next_private)
    return {"loss":loss.detach(),"S_loss":loss_S.detach(),"R_loss":loss_R.detach(),
        "S_owner_indices":owners_S.detach(),"S_owner_mask":mask_S.detach(),
        "R_owner_indices":owners_R.detach(),"R_owner_mask":mask_R.detach(),
        "incoming_logits":bank.detach(),"logit_cotangents":derivatives[-1].detach(),
        "shared_gradient":shared,"private_gradients":tuple(gradients),
        "theta_next":theta_next,"phis_next":next_private}


def native_parity(modules, scope, root, receipt, save, checked_forward, bill, limits):
    """One exact G0 context, three discarded first-order constructions."""
    torch = sys.modules["torch"]
    import numpy
    ordinary, helpers, accessor, port, custody = (modules[k] for k in ("ordinary","helpers","accessor","port","custody"))
    compare = modules["compare"].compare
    common_path=ordinary.bound(root,COMMON);origin=json.loads(ordinary.bound(root,ORIGIN).read_text())
    require(origin["schema"]=="amazon_G0_six_arm_run_v1" and origin["A_labels_received"] is False
        and origin["A_scoring_performed"] is False and origin["recipe"]["worker_source_sha256"]==ordinary.COMMON_ORIGIN_WORKER,
        "Original common400 origin required")
    preflight=helpers._public_identity_before_w(accessor,root,root/ordinary.PUBLIC_B_RELATIVE,ordinary.INPUT_IDENTITY)
    data=accessor.load_public_b(root,root/ordinary.PUBLIC_B_RELATIVE,device=DEVICE)
    expected=origin["recipe"];provenance=data["provenance"]
    require(provenance["public_b_manifest"]["sha256"]==expected["public_b_manifest_sha256"]
        and provenance["public_graph"]["sha256"]==expected["public_graph_sha256"]
        and provenance["roles"]["sha256"]==expected["roles_sha256"]
        and provenance["preprocessing"]["edge_logical_sha256"]==expected["native_edge_logical_sha256"]
        and provenance["preprocessing"]==preflight["preprocessing"], "Complete original public/role/edge custody required")
    S,yS,R,yR=(data[k] for k in ("inner_indices","inner_labels","query_indices","query_labels"))
    require(data["features"].shape==(24492,300) and data["features"].dtype==torch.float32
        and S.numel()==2449 and R.numel()==2450 and not bool(torch.isin(S,R).any())
        and not bool(torch.isin(torch.cat((S,R)),torch.cat((data["W_ids"],data["A_ids"]))).any()),
        "Original disjoint S/R on full FP32 graph required")
    image=torch.load(common_path,map_location="cpu",weights_only=True)
    require(image["schema"]=="amazon_G0_frozen_state_v1" and image["id"]=="initial" and image["warm_updates"]==400
        and image["episodes"]==0 and image["warm_role"]=="W" and image["global_stage"] is True and image["eval_mode"] is True
        and image["construction"]["seed"]==17 and len(image["W_own_CE_trace"])==400 and image["recipe"]==origin["recipe"],
        "Exact last original W-only common400 checkpoint required")
    helpers._finite_tree(image)
    family,_=helpers._fresh_family(modules["native"],modules["boundary"],DEVICE)
    family.load_state_dict(image["family_state"],strict=True);family.set_global_stage(True);family.eval()
    ordinary.restore_rng(image["rng"],DEVICE)
    frozen_image=helpers._cpu_tree(image)
    forward,theta,phis,_=port._native_callback_and_state(family,data["features"],data["edge_index"],expected_nodes=24492,global_stage=True)
    incoming_rng=helpers._rng(DEVICE)
    custody.torch=torch;custody.np=numpy
    before=custody.snapshot(family,data["features"],data["edge_index"],torch.device(DEVICE))
    input_values=helpers._cpu_tree((theta,phis,S,yS,R,yR))
    input_tensors=(S,yS,R,yR,data["features"],data["edge_index"],*theta.values(),
        *(v for row in phis for v in row.values()),*family.parameters(),*family.buffers())
    input_metadata=[(id(t),t.untyped_storage()._cdata,t.untyped_storage().data_ptr(),t._version,
        tuple(t.shape),tuple(t.stride()),t.dtype,t.device,t.requires_grad,t.grad is None) for t in input_tensors]
    def unchanged():
        compare(helpers._cpu_tree((theta,phis,S,yS,R,yR)),input_values,exact=True)
        compare(image,frozen_image,exact=True)
        for t,meta in zip(input_tensors,input_metadata):
            require((id(t),t.untyped_storage()._cdata,t.untyped_storage().data_ptr(),t._version,
                tuple(t.shape),tuple(t.stride()),t.dtype,t.device,t.requires_grad,t.grad is None)==meta, "Original input/model object/storage/version/flags changed")
        return custody.unchanged(before,family,data["features"],data["edge_index"],torch.device(DEVICE))
    def passed(name,value):
        receipt["checks"].append({"name":name,"status":"PASS","result":value});save()
    records=[];states=[];rows=[]
    try:
        for phase in ("independent_monolithic","streamed_candidate","fixed_streamed_repeat"):
            ordinary.restore_rng(incoming_rng,DEVICE)
            receipt["last_started_phase"]=phase;save();limits()
            begun=time.monotonic();count_before=dict(bill)
            phase_forward=checked_forward(forward,phase)
            if phase=="independent_monolithic":
                diagnostic=independent_reference(modules,phase_forward,theta,phis,S,yS,R,yR,bill,save)
                source_counts=None
            else:
                source_counts=dict.fromkeys(modules["caller"].COUNTER_KEYS,0)
                diagnostic={}
                receipt["active_candidate_counts"]={"phase":phase,"counts":source_counts};save()
                try:
                    modules["caller"]._engineering_step({k:modules[k] for k in modules["caller"].PINS},
                        phase_forward,theta,phis,S,yS,R,yR,source_counts,
                        engineering_authorized=True,CPU_fixture_PASS_verified=True,
                        caller_source_review_approved=True,native_context_verified=True,diagnostics=diagnostic)
                finally:
                    bill["native_gradient_APIs"]+=source_counts["native_parameter_VJP_APIs"]
                    bill["small_logit_gradient_APIs"]+=source_counts["small_logit_grad_APIs"]
                    bill["temporary_endpoint_constructions"]+=source_counts["simultaneous_SGD_attempts"]
                    bill["temporary_shared_SGD_maps"]+=source_counts["shared_SGD_maps"]
                    bill["temporary_private_SGD_maps"]+=source_counts["private_SGD_row_maps"]
                    receipt.setdefault("candidate_phase_counts",{})[phase]=source_counts
                    receipt["active_candidate_counts"]=None;save()
            states.append(helpers._cpu_tree(helpers._rng(DEVICE)))
            records.append(helpers._cpu_tree(diagnostic));del diagnostic
            ordinary.restore_rng(incoming_rng,DEVICE)
            row={"phase":phase,"elapsed_seconds":time.monotonic()-begun,"bill_delta":{k:bill[k]-count_before[k] for k in bill},
                "candidate_counts":source_counts,"whole_process_resources":ordinary.resources(DEVICE)}
            rows.append(row);receipt["phase_resources"]=rows;save();limits()
            passed(phase+"_original_native_context_and_inputs_unchanged",unchanged())
        for index in (1,2):
            actual,expected=records[index],records[0]
            exact={k:expected[k] for k in ("S_owner_indices","S_owner_mask","R_owner_indices","R_owner_mask")}
            compare({k:actual[k] for k in exact},exact,exact=True)
            errors={k:compare(actual[k],value) for k,value in expected.items() if k not in exact}
            compare(states[index],states[0],exact=True)
            passed(("streamed" if index==1 else "fixed_repeat")+"_versus_independent_full_native",
                {"complete_coordinate_max_errors":errors,"owner_masks_indices_and_logical_RNG_exact":True})
        compare(records[2],records[1]);compare(states[2],states[1],exact=True)
        passed("fixed_repeat_versus_streamed_complete_values_and_RNG",{"all_coordinates_and_discrete_owners_checked":True})
        for diagnostic in records:
            for group in (diagnostic["shared_gradient"],*diagnostic["private_gradients"]):
                for name,value in group.items():
                    if name.startswith("local_head."):
                        require(bool((value==0).all()), "Dormant native gradient is not exact zero")
            require(bool((diagnostic["logit_cotangents"][:,~torch.isin(torch.arange(24492),torch.cat((S.cpu(),R.cpu()))) ]==0).all()),
                "Non-S/R nodes received native logit cotangents")
        require(bill==EXPECTED,"Frozen 20-forward/11-gradient/three-discarded-endpoint bill differs")
        passed("complete_fixed_attempt_bill_and_zero_unused_coordinates",dict(bill))
        require(ordinary.sha(common_path)==COMMON["sha256"] and ordinary.sha(root/ORIGIN["path"])==ORIGIN["sha256"],
            "Original common/source context bytes changed")
        receipt["common400_context"]={"common400":COMMON,"origin_run":ORIGIN,"N_D_C":[24492,300,5],"S_R":[2449,2450],
            "same_incoming_state_for_all_three":True,"global_stage_eval_FP32":True,"all_temporary_endpoints_discarded":True}
    finally:
        ordinary.restore_rng(incoming_rng,DEVICE)
        receipt["final_native_custody"]=unchanged();save()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--execute-authorized",action="store_true");parser.add_argument("--source-root")
    parser.add_argument("--scope");parser.add_argument("--scope-sha256");parser.add_argument("--output")
    args=parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({"status":"DISABLED_CMCL_NATIVE_FIRST_ORDER_PARITY","SOURCE_RELEASED":False,"numeric_imports":False}));return 0
    require(SOURCE_RELEASED is False and args.source_root==PHASE and socket.gethostname()=="anogena-2-0"
        and Path.cwd().resolve()==Path(REPOSITORY) and sys.dont_write_bytecode
        and not any(n=="torch" or n.startswith("torch.") for n in sys.modules),"Fresh exact allocation root invocation required")
    names=[];ordinary=load_pin(Path(PHASE),"ordinary",names)
    root,output,scope_path=ordinary.deliberate_paths(SimpleNamespace(source_root=args.source_root,output=args.output,admission=args.scope))
    require(scope_path.stat().st_mode & 0o222==0 and ordinary.sha(scope_path)==args.scope_sha256,"Root immutable exact scope required")
    scope=json.loads(scope_path.read_text())
    own={"path":str(Path(__file__).resolve().relative_to(root)),"bytes":Path(__file__).stat().st_size,"sha256":ordinary.sha(__file__)}
    ordinary.bound(root,own)
    require(scope["schema"]=="root_CMCL_native_common400_first_order_parity_scope_v1" and scope["executor"]==own
        and scope["root_engineering_invocation_authorized"] is True and scope["fixed_before_execution"] is True
        and scope["one_owned_invocation_no_retry"] is True and scope["no_fit_or_scoring"] is True
        and scope["A_VALID_TEST_access"] is False and scope["exclusive_native_parity_window"] is True
        and scope["root_scheduling_clearance_verified"] is True and scope["common400"]==COMMON and scope["origin_run"]==ORIGIN
        and scope["M_K_beta"]==[4,3,0.75] and scope["eta_core_private"]==[0.001,0.01]
        and scope["ATOL"]==ATOL and scope["RTOL"]==RTOL and scope["expected_bill"]==EXPECTED,
        "Exact prospective root engineering/math/count/scheduling scope required")
    reviews={}
    for key in ("root_scheduling_clearance","fresh_GPU_memory_review","caller_source_review","parity_source_review"):
        review=json.loads(ordinary.bound(root,scope[key]).read_text())
        reviews[key]=review
        if key.endswith("source_review"):
            expected_source=own if key=="parity_source_review" else {
                "path":PINS["caller"][0],"bytes":scope["deployed_sources"]["caller"]["bytes"],"sha256":PINS["caller"][1]}
            require(review["status"].startswith("PASS_SOURCE") and review["new_source_blockers"]==[]
                and review["execution_or_fit_authorized"] is False and review["source"]==expected_source,
                "New exact caller/parity source review required")
    require(scope["caller_source_review"]["bytes"]==7576
        and scope["caller_source_review"]["sha256"]=="b69f81737f383d5183b108c470740cb10f0d85505b1a5a5fc1ae68bddd48eaf7",
        "Exact actual independent callable review required")
    cpu=json.loads(ordinary.bound(root,scope["CPU_worker_PASS"]).read_text())
    terminal=json.loads(ordinary.bound(root,scope["CPU_owned_terminal_PASS"]).read_text())
    require(scope["CPU_worker_PASS"]["bytes"]==8530 and scope["CPU_worker_PASS"]["sha256"]=="d73ac3939f3857d6f96d42b5dc079721e912a8b6392486a76a44b931b88d18dc"
        and scope["CPU_owned_terminal_PASS"]["bytes"]==2970
        and scope["CPU_owned_terminal_PASS"]["sha256"]=="b65805fcaa585de3cd46bc58b4a7c19b0293c18604de97aee37b5c76e9d7d0d4"
        and cpu["status"]=="PASS_CPU_FLOAT64_CMCL_ANALYTIC_FIXTURES_ENGINEERING_ONLY" and cpu["completed_frozen_checks"]==19
        and all(row["status"]=="PASS" for row in cpu["checks"]) and cpu["restoration_errors"]==[]
        and cpu["attempts"]=={"positive_role_objective_calls":6,"positive_serving_calls":1,"ordinary_grad_API_calls":6,"rejection_calls":13}
        and cpu["model_fits"]==cpu["native_callbacks"]==cpu["optimizer_updates"]==0
        and cpu["actual_dataset_labels_read"] is False and cpu["A_VALID_TEST_access"] is False
        and cpu["helper"]["sha256"]==PINS["cmcl"][1] and cpu["executor"]["sha256"]=="c0553fef2bed858e9e6672642c2216cc0e026cda82d852300caac88a34b69edc"
        and terminal["status"]=="PASS_CPU_CMCL_OWNED_PUBLICATION_EXIT_RESOURCE_CLOSURE_ENGINEERING_ONLY"
        and terminal["worker_result"]["sha256"]==scope["CPU_worker_PASS"]["sha256"] and terminal["owned_child"]["exit_code"]==0
        and terminal["owned_child"]["reaped"] is True and terminal["owned_child"]["signals"]==[]
        and terminal["owned_child"]["watchdog_fired"] is False and terminal["wrapper_resource_closure_PASS"] is True,
        "Actual fixed CPU V2 PASS and owned closure required")
    caps=scope["resource_limits"]
    require(all(type(caps[k]) in (int,float) and math.isfinite(caps[k]) and caps[k]>0 for k in
        ("max_elapsed_seconds","max_process_rss_bytes","max_cuda_allocated_bytes","max_cuda_reserved_bytes"))
        and scope["external_terminal_resource_closure_required"] is True and scope["external_watchdog_seconds"]>caps["max_elapsed_seconds"]
        and type(scope["minimum_initial_cuda_free_bytes"]) is int and scope["minimum_initial_cuda_free_bytes"]>=caps["max_cuda_reserved_bytes"]+scope["cuda_headroom_bytes"]
        and scope["cuda_headroom_bytes"]>=4294967296,"Root prospective caps/free-memory budget and external closure required")
    clearance=reviews["root_scheduling_clearance"];memory=reviews["fresh_GPU_memory_review"]
    require(clearance["schema"]=="root_CMCL_native_parity_scheduling_clearance_v1"
        and clearance["fixed_before_execution"] is True and clearance["exclusive_native_parity_window"] is True
        and clearance["GPU_UUID"]==scope["GPU_UUID"] and clearance["no_fit_or_scoring"] is True
        and memory["schema"]=="root_CMCL_native_parity_fresh_memory_review_v1" and memory["root_reviewed"] is True
        and memory["GPU_UUID"]==scope["GPU_UUID"] and memory["resource_limits"]==caps
        and memory["cuda_headroom_bytes"]==scope["cuda_headroom_bytes"]
        and memory["minimum_initial_cuda_free_bytes"]==scope["minimum_initial_cuda_free_bytes"]
        and memory["actual_observed_free_bytes"]>=scope["minimum_initial_cuda_free_bytes"],
        "Actual fresh root GPU memory review and exclusive scheduling clearance required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES")==scope["GPU_UUID"] and scope["GPU_UUID"].startswith("GPU-")
        and "," not in scope["GPU_UUID"] and Path(sys.executable).absolute()==Path(scope["python_executable"]).absolute(),
        "Root-selected sole physical GPU and interpreter required")
    owner=ordinary.CreatedOutput(output,root);token=owner.creator_token
    bill=dict.fromkeys(EXPECTED,0)
    receipt={"schema":"CMCL_native_common400_first_order_parity_result_v1","status":"RUNNING","executor":own,
        "root_scope_sha256":args.scope_sha256,"checks":[],"bill":bill,"phase_resources":[],"restoration_errors":[],
        "model_fits":0,"persistent_updates":0,"checkpoint_or_prediction_writes":0,"A_VALID_TEST_access":False,
        "external_terminal_resource_closure_required":True,"CPU_worker_PASS":scope["CPU_worker_PASS"],"CPU_owned_terminal_PASS":scope["CPU_owned_terminal_PASS"]}
    def resources():
        row=ordinary.resources(DEVICE);row["elapsed_seconds"]=time.monotonic()-STARTED;return row
    def save():
        receipt["elapsed_seconds"]=time.monotonic()-STARTED
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():receipt["whole_process_resources"]=resources()
        ordinary.atomic(owner.verify(token)/"RESULT.json",receipt)
    def limits():
        row=resources()
        require(all(row[m]<=caps[k] for m,k in (("elapsed_seconds","max_elapsed_seconds"),("process_peak_rss_bytes","max_process_rss_bytes"),
            ("cuda_peak_allocated_bytes","max_cuda_allocated_bytes"),("cuda_peak_reserved_bytes","max_cuda_reserved_bytes"))),"Prospective whole-worker caps exceeded")
    def checked_forward(forward,phase):
        def called(theta,phi):
            limits();bill["native_callbacks"]+=1;receipt["last_native_callback_phase"]=phase;save()
            value=forward(theta,phi);limits();return value
        return called
    old_path=list(sys.path);python_rng=random.getstate();old_env_present="CUBLAS_WORKSPACE_CONFIG" in os.environ;old_env=os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    old_handler=signal.getsignal(signal.SIGALRM);old_timer=signal.getitimer(signal.ITIMER_REAL)
    require(old_timer==(0.0,0.0),"Fresh child timer required")
    process=helpers=None;backend_before=old_threads=numeric_rng=None;code=1
    modules={"ordinary":ordinary}
    try:
        def expired(signum,frame):raise TimeoutError("Root-fixed native parity deadline exceeded")
        signal.signal(signal.SIGALRM,expired);remaining=caps["max_elapsed_seconds"]-(time.monotonic()-STARTED)
        require(remaining>0,"No root parity time remains");signal.setitimer(signal.ITIMER_REAL,remaining)
        config=scope["process_configuration"];os.environ["CUBLAS_WORKSPACE_CONFIG"]=config["CUBLAS_WORKSPACE_CONFIG"]
        sys.path.insert(0,scope["site_packages"])
        import torch
        require(str(Path(torch.__file__).resolve())==scope["torch_module_path"] and str(Path(sys.executable).resolve())==scope["python_resolved"]
            and not torch.cuda.is_initialized(),"Exact pre-CUDA normal runtime required")
        require(set(scope["deployed_sources"])==set(PINS),"Complete exact source deployment required")
        require(scope["deployed_sources"]["ordinary"]=={
            "path":PINS["ordinary"][0],"bytes":Path(ordinary.__file__).stat().st_size,"sha256":PINS["ordinary"][1]},
            "Bootstrap ordinary source must match its admitted deployment")
        for key in PINS:
            if key!="ordinary":modules[key]=load_pin(root,key,names,scope["deployed_sources"][key])
        process=modules["process"];process.torch=torch;helpers=modules["helpers"]
        backend_before=process.backend_snapshot();old_threads=torch.get_num_threads()
        torch.use_deterministic_algorithms(True,warn_only=False)
        torch.set_num_threads(config["intra_op_threads"]);torch.set_num_interop_threads(config["interop_threads"])
        require(config=={"CUBLAS_WORKSPACE_CONFIG":":4096:8","intra_op_threads":1,"interop_threads":1}
            and not torch.cuda.is_initialized(),"Original strict before-CUDA process settings required")
        identity=process.runtime_identity();require(identity==scope["runtime_identity_expected"] and torch.cuda.device_count()==1
            and process.backend_snapshot()==scope["strict_backend_expected"],"Root original native runtime/backend/single GPU differs")
        free,total=torch.cuda.mem_get_info(DEVICE)
        receipt["initial_cuda_memory"]={"actual_free_bytes":free,"total_bytes":total,"required_free_bytes":scope["minimum_initial_cuda_free_bytes"]}
        require(free>=scope["minimum_initial_cuda_free_bytes"],"Fresh actual free memory below root prospective budget")
        for key in ("caller","cmcl","ordinary","compare","operator","helpers","process"):
            require(modules[key].SOURCE_RELEASED is False,"Borrowed source marker changed: "+key)
        require(modules["port"].PORT_RELEASED is False and modules["accessor"].SOURCE_RELEASED is True,"Native port/accessor gates differ")
        numeric_rng=helpers._rng(DEVICE);save()
        native_parity(modules,scope,root,receipt,save,checked_forward,bill,limits)
        require(bill==EXPECTED,"Complete prospective attempt bill differs");limits()
        receipt["numeric_PARITY_supported"]=True
    except BaseException as error:
        receipt.update(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        def restore(name,function):
            try:function();receipt[name]=True
            except BaseException as error:receipt["restoration_errors"].append({"check":name,"error":str(error),"traceback":traceback.format_exc()})
        if numeric_rng is not None:
            def rng():
                ordinary.restore_rng(numeric_rng,DEVICE);modules["compare"].compare(helpers._rng(DEVICE),numeric_rng,exact=True)
            restore("caller_numeric_RNG_restored",rng)
        if backend_before is not None:restore("backend_restored",lambda:process.backend_restore(backend_before))
        if old_threads is not None:
            def threads():
                torch.set_num_threads(old_threads);require(torch.get_num_threads()==old_threads,"Intra-op restoration differs")
            restore("reversible_threads_restored",threads)
        random.setstate(python_rng);sys.path[:]=old_path
        if old_env_present:os.environ["CUBLAS_WORKSPACE_CONFIG"]=old_env
        else:os.environ.pop("CUBLAS_WORKSPACE_CONFIG",None)
        signal.signal(signal.SIGALRM,old_handler);signal.setitimer(signal.ITIMER_REAL,*old_timer)
        for name in names:sys.modules.pop(name,None)
        restore("Python_path_environment_signal_module_state_restored",lambda:require(random.getstate()==python_rng and sys.path==old_path
            and ("CUBLAS_WORKSPACE_CONFIG" in os.environ)==old_env_present and os.environ.get("CUBLAS_WORKSPACE_CONFIG")==old_env
            and signal.getsignal(signal.SIGALRM)==old_handler and signal.getitimer(signal.ITIMER_REAL)==old_timer
            and not any(name in sys.modules for name in names),"Reversible process state differs"))
        restore("all_exact_source_scope_receipts_unchanged",lambda:require(ordinary.sha(scope_path)==args.scope_sha256
            and ordinary.sha(__file__)==own["sha256"] and all(ordinary.bound(root,row) for row in scope["deployed_sources"].values())
            and all(ordinary.bound(root,scope[key]) for key in ("root_scheduling_clearance","fresh_GPU_memory_review","caller_source_review","parity_source_review","CPU_worker_PASS","CPU_owned_terminal_PASS")),
            "Source/scope/review/evidence bytes changed"))
        receipt["interop_restoration"]="Fresh-child one-time initialization; not claimed reversible."
        if receipt.get("numeric_PARITY_supported") is True and not receipt["restoration_errors"]:
            try:limits();code=0
            except BaseException as error:receipt["final_resource_error"]=str(error)
        receipt["status"]="PASS_CMCL_NATIVE_COMMON400_FIRST_ORDER_PARITY_ENGINEERING_ONLY" if code==0 else "FAIL_CMCL_NATIVE_COMMON400_FIRST_ORDER_PARITY"
        receipt["numeric_PARITY_supported"]=code==0
        try:save();owner.freeze(token)
        except BaseException as error:
            code=1;receipt.update(status="FAIL_CMCL_NATIVE_COMMON400_FIRST_ORDER_PARITY",numeric_PARITY_supported=False,output_finalization_error=str(error))
            try:save();owner.freeze(token)
            except BaseException as recovery_error:receipt["failure_publication_error"]=str(recovery_error)
    print(json.dumps({"status":receipt["status"],"error":receipt.get("error"),"model_fits":0,"persistent_updates":0,"external_terminal_resource_closure_required":True}))
    return code


if __name__=="__main__":raise SystemExit(main())
