"""Disabled CMCL native step and fixed H16 callable for a reviewed context owner.

No CLI, loader, acquisition, supervisor or new runtime guard framework.
Reuse eval-only native callback, ordinary value/cotangent/RNG replay pattern,
existing pure SGD, complete commonW400 context and probability-mean serving.
"""
import hashlib
from pathlib import Path

SOURCE_RELEASED = True
ARM, HORIZON = "CMCL_exact_KL_M4_K3_beta075", 16
MEMBERS, NODES, CLASSES = 4, 24492, 5
ETA_CORE, ETA_PRIVATE = 0.001, 0.01
ATOL, RTOL = 2e-6, 2e-5
COMMON = {"path": "amazon_learnability_responsibility_strict_scientific_execution_root_20261006_v2/output/scientific_run/initial.pt",
    "bytes": 109671738, "sha256": "2e0e9b44767abc12b3dc896986ea2d4faeaa667c8682b95929f927d10718154f"}
ORIGIN = {"path": "amazon_common400_descriptor_binding_root_20261006_v1/ORIGIN_RUN.json",
    "bytes": 6402, "sha256": "2e18a77081437e47c0c4a0d59cb07b2a15500f660066cbed1c980f613c028ca5"}
PINS = {
    "cmcl": ("cmcl_graph_objective_transplant_source_preparation_20261006_v1/cmcl_loss.py",
        "4f04eb87381512c9f7506b6458917b83fd12228f3c7651c6291e902c99fe6ac9"),
    "ordinary": ("amazon_ordinary_shared_bank_own_pool_reference_preparation_20261006_v4/ordinary_reference.py",
        "5044a16f3f710aaf234057b115ab928d589af3a06a595940afbfa8876636c97b"),
    "compare": ("amazon_ordinary_shared_bank_first_order_qualification_preparation_20261006_v1/qualify.py",
        "d175dc6bf44886d874959cecd9833dc9f192f30b764bc7e9b938be83f2ad8af2"),
    "operator": ("learnability_weighted_graph_responsibility_operator_20261005_v2/response_operator.py",
        "fba3ca3d4bb35da0438923941d97bc4833004dd9fd385735c2352f343693b3e3"),
    "port": ("learnability_responsibility_native_amazon_sparse_port_20261005_v1/native_sparse_port.py",
        "a8ee35afda97afcb745c365a51f8e8647fe5e6a1350f654a5321c81e12abad86"),
    "helpers": ("amazon_learnability_responsibility_sequential_train_only_execution_preparation_20261006_v3/six_arm_worker.py",
        "f19a94be4102e74f30d2b779ca602e288cf40c446ae367ce9d9ab44091569ad1")}
COUNTER_KEYS = ("native_value_callbacks", "native_replay_callbacks", "small_logit_grad_APIs",
    "native_parameter_VJP_APIs", "simultaneous_SGD_attempts", "shared_SGD_maps", "private_SGD_row_maps",
    "completed_updates", "serving_callbacks")


def _engineering_step(modules, forward, theta, phis, S, yS, R, yR, counts, *,
        engineering_authorized=False, CPU_fixture_PASS_verified=False,
        caller_source_review_approved=False, native_context_verified=False, diagnostics=None):
    """One first-order streamed native SGD construction, no in-place commit.

    A separately reviewed owner must bind actual CPU evidence and native full
    eval/global/common400 context before enabling this pure engineering call.
    Return all next dictionaries together after every coupled gradient exists.
    Optional fresh diagnostics expose detached full values for native parity;
    the owner must discard them after qualification, without scoring or fitting.
    """
    if not (engineering_authorized is True and CPU_fixture_PASS_verified is True
            and caller_source_review_approved is True and native_context_verified is True):
        raise RuntimeError("Disabled: actual CPU PASS/new source review/native context admission required")
    if set(modules) != set(PINS) or set(counts) != set(COUNTER_KEYS):
        raise RuntimeError("Exact borrowed modules and fixed counters required")
    if diagnostics is not None and (not isinstance(diagnostics,dict) or diagnostics):
        raise RuntimeError("Engineering diagnostics must be a fresh caller-owned dictionary")
    for key, module in modules.items():
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != PINS[key][1]:
            raise RuntimeError("Pinned borrowed source differs: " + key)
        gate = "PORT_RELEASED" if key == "port" else "SOURCE_RELEASED"
        if getattr(module, gate) is not False:
            raise RuntimeError("Original pure borrowed source gate must stay false")
    import torch
    cmcl, ordinary, op, port, helpers = (modules[k] for k in ("cmcl", "ordinary", "operator", "port", "helpers"))
    if (len(phis) != MEMBERS or any(set(phi) != set(port.PRIVATE_NAMES) for phi in phis)
            or S.numel() != 2449 or R.numel() != 2450 or bool(torch.isin(S,R).any())
            or yS.shape != S.shape or yR.shape != R.shape
            or (cmcl.MEMBERS,cmcl.CLASSES,cmcl.OWNER_K,cmcl.BETA,cmcl.ROLE_MASS) != (4,5,3,0.75,0.5)
            or any(v.dtype != torch.float32 or str(v.device) != "cuda:0" for v in
                [*theta.values(),*(v for phi in phis for v in phi.values())])):
        raise RuntimeError("Fixed complete native FP32 bank and original S/R roles required")
    helpers._finite_tree((theta,phis))
    device = "cuda:0"
    initial_rng = helpers._rng(device)
    states, values, after_values = [], [], None
    succeeded = False
    try:
        # Same first-order value bank as ordinary.step, using the reviewed pure
        # eval-only native callback instead of its train()/Adam model wrapper.
        with torch.no_grad():
            for member in range(MEMBERS):
                states.append(helpers._rng(device))
                counts["native_value_callbacks"] += 1
                z = forward(theta,phis[member])
                if z.shape != (NODES,CLASSES) or z.dtype != torch.float32:
                    raise RuntimeError("Complete all-node FP32 native value path required")
                helpers._finite(z,"Nonfinite CMCL native value")
                values.append(z.detach().clone()); del z
        after_values = helpers._rng(device)
        # S and R are views of ONE all-member/all-node incoming value bank.
        bank = torch.stack(values).requires_grad_(True)
        loss, info = cmcl._engineering_role_objective(bank[:,S],yS,bank[:,R],yR,
            engineering_authorized=True)
        counts["small_logit_grad_APIs"] += 1
        cotangents, = torch.autograd.grad(loss,bank)
        shared_gradient = {n:torch.zeros_like(v) for n,v in theta.items()}
        private_gradients = []
        for member in range(MEMBERS):
            ordinary.restore_rng(states[member],device)
            core = {n:v.detach().requires_grad_(True) for n,v in theta.items()}
            private = {n:v.detach().requires_grad_(True) for n,v in phis[member].items()}
            counts["native_replay_callbacks"] += 1
            z = forward(core,private)
            helpers._finite(z,"Nonfinite CMCL native gradient replay")
            if z.shape != values[member].shape or not torch.allclose(z.detach(),values[member],atol=ATOL,rtol=RTOL):
                raise RuntimeError("Exact incoming native value/RNG replay parity failed")
            counts["native_parameter_VJP_APIs"] += 1
            derivatives = torch.autograd.grad(z,tuple(core.values())+tuple(private.values()),
                grad_outputs=cotangents[member],allow_unused=True)
            g_core = {n:torch.zeros_like(v) if d is None else d.detach()
                for (n,v),d in zip(core.items(),derivatives[:len(core)])}
            g_private = {n:torch.zeros_like(v) if d is None else d.detach()
                for (n,v),d in zip(private.items(),derivatives[len(core):])}
            for n in shared_gradient:shared_gradient[n] = shared_gradient[n]+g_core[n]
            private_gradients.append(g_private)
            del z,derivatives,core,private,g_core,g_private
        ordinary.restore_rng(after_values,device)
        helpers._finite_tree((shared_gradient,private_gradients))
        counts["simultaneous_SGD_attempts"] += 1
        counts["shared_SGD_maps"] += 1
        theta_next = op._detach(op._sgd(theta,shared_gradient,ETA_CORE))
        private_next = []
        for phi,gradient in zip(phis,private_gradients):
            counts["private_SGD_row_maps"] += 1
            private_next.append(op._detach(op._sgd(phi,gradient,ETA_PRIVATE)))
        private_next = tuple(private_next)
        helpers._check_commit(theta,phis,theta_next,private_next)
        stats = {"CMCL_loss":float(loss.detach()),"S_loss":float(info["S_loss"].detach()),
            "R_loss":float(info["R_loss"].detach()),
            "S_owner_counts":info["S"]["owner_mask"].sum(dim=1).tolist(),
            "R_owner_counts":info["R"]["owner_mask"].sum(dim=1).tolist(),
            "same_incoming_full_logits_for_S_R":True,"M":4,"K":3,"beta":0.75}
        if diagnostics is not None:
            diagnostics.update(loss=loss.detach(),S_loss=info["S_loss"].detach(),R_loss=info["R_loss"].detach(),
                S_owner_indices=info["S"]["owner_indices"].detach(),S_owner_mask=info["S"]["owner_mask"].detach(),
                R_owner_indices=info["R"]["owner_indices"].detach(),R_owner_mask=info["R"]["owner_mask"].detach(),
                incoming_logits=bank.detach(),logit_cotangents=cotangents.detach(),
                shared_gradient=shared_gradient,private_gradients=tuple(private_gradients),
                theta_next=theta_next,phis_next=private_next)
        succeeded = True
        return theta_next,private_next,stats
    finally:
        # Preserve the four-value-forward logical RNG on success; a failed
        # discarded construction restores its original incoming RNG.
        expected_rng=after_values if succeeded else initial_rng
        ordinary.restore_rng(expected_rng,device)
        modules["compare"].compare(helpers._rng(device),expected_rng,exact=True)


def run_H16(modules, forward, theta, phis, S, yS, R, yR, *, admission, progress):
    """Fixed CMCL continuation inside the existing reviewed context owner.

    That owner supplies immutable evidence/custody/resources and persists every
    fixed endpoint/failure. This callable neither loads nor writes any payload.
    """
    if SOURCE_RELEASED is not True:
        raise RuntimeError("Disabled CMCL H16 source has no native qualification or fit authority")
    if (admission.get("schema")!="root_common400_CMCL_H16_context_admission_v1" or admission.get("arm")!=ARM
            or admission.get("root_fit_authorized") is not True or admission.get("fixed_before_fit") is not True
            or admission.get("caller_source_review_approved") is not True
            or admission.get("CPU_fixture_PASS_verified") is not True
            or admission.get("new_native_CMCL_first_order_PARITY_PASS_verified") is not True
            or admission.get("native_context_verified") is not True
            or admission.get("common400") != COMMON or admission.get("origin_run") != ORIGIN
            or admission.get("H") != HORIZON or admission.get("M_K_beta") != [4,3,0.75]
            or admission.get("eta_core_private") != [ETA_CORE,ETA_PRIVATE]
            or admission.get("A_VALID_TEST_scoring") is not False
            or admission.get("optional_feature_sharing") is not False
            or admission.get("one_fixed_outcome_no_retry") is not True):
        raise RuntimeError("Actual CPU/native/source/context and distinct fixed root fit admission required")
    if set(modules)!=set(PINS) or not isinstance(progress,dict) or progress:
        raise RuntimeError("Exact borrowed modules and fresh caller-owned progress required")
    for key,module in modules.items():
        if (hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=PINS[key][1]
                or getattr(module,"PORT_RELEASED" if key=="port" else "SOURCE_RELEASED") is not False):
            raise RuntimeError("Pinned original pure borrowed module differs before serving")
    import torch
    cmcl, helpers = modules["cmcl"],modules["helpers"]
    counts = dict.fromkeys(COUNTER_KEYS,0)
    history=[]
    progress.update(status="RUNNING_CMCL_H16",counts=counts,history=history,last_completed_theta=theta,last_completed_phis=phis)
    def served(current_theta,current_phis):
        with torch.no_grad():
            rows=[]
            for phi in current_phis:
                counts["serving_callbacks"] += 1
                z=forward(current_theta,phi)
                if z.shape!=(NODES,CLASSES) or z.dtype!=torch.float32:
                    raise RuntimeError("Complete native FP32 fixed probability serving required")
                rows.append(z)
            bank=torch.stack(rows)
            pool=cmcl._engineering_mean_probabilities(bank,engineering_authorized=True)
            helpers._finite_tree((bank,pool))
            return {"member_logits":bank.detach(),"mean_softmax_probabilities":pool.detach(),
                "scoring_performed":False,"owners_used_for_serving":False}
    try:
        common_served=served(theta,phis);progress["common400_serving"]=common_served
        for update in range(HORIZON):
            theta_next,private_next,stats=_engineering_step(modules,forward,theta,phis,S,yS,R,yR,counts,
                engineering_authorized=True,CPU_fixture_PASS_verified=True,
                caller_source_review_approved=True,native_context_verified=True)
            theta,phis=theta_next,private_next
            counts["completed_updates"] += 1
            progress.update(last_completed_theta=theta,last_completed_phis=phis)
            history.append({"update":update+1,**stats})
        endpoint_served=served(theta,phis);progress["endpoint_serving"]=endpoint_served
        expected={"native_value_callbacks":4*HORIZON,"native_replay_callbacks":4*HORIZON,
            "small_logit_grad_APIs":HORIZON,"native_parameter_VJP_APIs":4*HORIZON,
            "simultaneous_SGD_attempts":HORIZON,"shared_SGD_maps":HORIZON,
            "private_SGD_row_maps":4*HORIZON,"completed_updates":HORIZON,"serving_callbacks":8}
        if counts!=expected:raise RuntimeError("Complete fixed CMCL H16 operation bill differs")
        progress["status"]="CMCL_H16_COMPLETE_A_CLOSED"
        return {"arm":ARM,"H":HORIZON,"theta":theta,"phis":phis,"history":history,"counts":counts,
            "common400_serving":common_served,"endpoint_serving":endpoint_served,
            "probability_mean_of_all4":True,"A_VALID_TEST_scoring":False,
            "checkpoint_or_fit_loader_added":False,"all_fixed_outcomes_required":True}
    except BaseException as error:
        progress.update(status="FAIL_CMCL_H16",error_type=type(error).__name__,error=str(error))
        raise
