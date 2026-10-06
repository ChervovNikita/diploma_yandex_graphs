"""Disabled exact-common400 utility H16 continuation; no acquisition or held scoring."""
import time
STARTED = time.monotonic()
import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import socket
import sys
import traceback

SOURCE_RELEASED = False
ARM, HORIZON = "first_order_utility_live", 16
HOSTS = {"peptide": "/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git",
         "anogena-2-0": "/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs"}
COMMON = {"path": "amazon_learnability_responsibility_strict_scientific_execution_root_20261006_v2/output/scientific_run/initial.pt",
          "bytes": 109671738, "sha256": "2e0e9b44767abc12b3dc896986ea2d4faeaa667c8682b95929f927d10718154f"}
ORIGIN = {"path": "amazon_common400_descriptor_binding_root_20261006_v1/ORIGIN_RUN.json",
          "bytes": 6402, "sha256": "2e18a77081437e47c0c4a0d59cb07b2a15500f660066cbed1c980f613c028ca5"}
PINS = {
 "qualifier": ("matched_first_order_private_gradient_utility_native_qualification_preparation_20261006_v6/qualify.py", "8b98dfa18ddd1b9990949984079a3654fd968f55346ee07ab5c29bfa59e0232e"),
 "utility": ("matched_first_order_private_gradient_utility_control_source_preparation_20261006_v1/utility_control.py", "83967dc045db903205ce6c67df963fb557a858175e107e7de205651ad0b28545"),
 "sequential": ("learnability_responsibility_sequential_autograd_vjp_preparation_20261006_v1/sequential_vjp.py", "1f9e704a1b95a98cec688d695094058594758a99bac2f057e2fe90e7f9315167"),
 "operator": ("learnability_weighted_graph_responsibility_operator_20261005_v2/response_operator.py", "fba3ca3d4bb35da0438923941d97bc4833004dd9fd385735c2352f343693b3e3"),
 "port": ("learnability_responsibility_native_amazon_sparse_port_20261005_v1/native_sparse_port.py", "a8ee35afda97afcb745c365a51f8e8647fe5e6a1350f654a5321c81e12abad86"),
 "accessor": ("amazon_learnability_responsibility_engineering_accessor_20261005_v1/train_only_accessor.py", "9360e69753b362eb66ac89f133f00addaf15ab0e8c56314c75d7dd5e96fecb85"),
 "native": ("amazon_polynormer_paired_family_source_preparation_20261003_v6/reused_models/native_polynormer.py", "9b4e533f46ae7f91a23a552f88bbb47a01359e224b53996cf1a68c10a865f6a8"),
 "boundary": ("amazon_polynormer_paired_family_source_preparation_20261003_v6/reused_models/backbone_boundary_adapter.py", "699b606ead00cb7bdd9be6cd58730a0687157c40cf594af620d1edc4c93b6bac"),
 "reference": ("amazon_learnability_responsibility_sequential_train_only_release_root_20261006_v2/six_arm_worker.py", "ad28d1c8a05a168aeadb6825183ef8d1d1a6cffea3b488c5a33aa3c90318c8f3"),
 "base": ("learnability_responsibility_native_numerical_worker_preparation_20261005_v3/qualify.py", "baeac626bade87bd286fe7a93a95602d8dc1f57d3257d0eeeb7e4a8e3823c688")}
COEFFICIENTS = {"eta_probe": .01, "eta_private": .01, "eta_core": .001, "extra_margin": .1,
 "pool_fraction": .5, "response_epsilon": .001, "entropy": 1., "graph": 1., "assignment_steps": 8,
 "assignment_rate": 1., "ratio_smoothing": .01, "reciprocal_smoothing": .01}
EPISODE = {"native_forward_calls": 40, "private_gradient_calls": 24, "native_vjp_calls": 8,
 "q_map_primal_calls": 30, "q_map_vjp_calls": 10, "small_query_vjp_calls": 1,
 "utility_margin_phi_vjp_calls": 8, "utility_dummy_cotangent_reverse_calls": 8,
 "utility_weighted_margin_private_gradient_calls": 4}


def require(condition, message):
    if not condition: raise AssertionError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def bound_path(phase, row):
    relative = Path(row["path"])
    require(not relative.is_absolute() and ".." not in relative.parts, "Relative in-phase binding required")
    path = phase / relative
    require(not path.is_symlink() and path.resolve().is_relative_to(phase) and path.stat().st_mode & 0o222 == 0
            and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Immutable bound bytes differ")
    return path


def bound_json(phase, row):
    return json.loads(bound_path(phase, row).read_text())


def admission(args):
    host = socket.gethostname()
    require(host in HOSTS and sys.platform.startswith("linux"), "Named authorized Linux host required")
    repo = Path(HOSTS[host]); phase = Path(args.source_root).absolute()
    require(phase == repo / "experiments_iclr/postsubmission_20260930" and phase.resolve() == phase
            and Path.cwd().resolve() == repo and not any(n == "torch" or n.startswith("torch.") for n in sys.modules),
            "Exact phase/repository and fresh process before Torch required")
    scope_path = Path(args.fit_scope).absolute()
    require(".." not in scope_path.parts and scope_path.is_relative_to(phase) and scope_path.resolve().is_relative_to(phase)
            and not scope_path.is_symlink() and scope_path.stat().st_mode & 0o222 == 0
            and sha(scope_path) == args.fit_scope_sha256, "Immutable exact root fit scope required")
    scope = json.loads(scope_path.read_text())
    require(scope["schema"] == "root_common400_utility_H16_fit_scope_v1" and scope["root_fit_authorized"] is True
            and scope["fixed_before_fit"] is True and scope["caller_sha256"] == sha(__file__)
            and scope["hostname"] == host and scope["repository"] == str(repo)
            and scope["arm"] == ARM and scope["H"] == HORIZON and scope["coefficients"] == COEFFICIENTS
            and scope["common400"] == COMMON and scope["origin_run"] == ORIGIN
            and scope["no_A_VALID_TEST_scoring"] is True and scope["original_six_arm_pilot_unchanged"] is True,
            "Separate matched utility-only H16 fit admission required")
    review = bound_json(phase, scope["caller_review"])
    require(review["caller_sha256"] == sha(__file__) and review["status"].startswith("PASS_SOURCE")
            and review["execution_or_fit_authorized"] is False, "Independent exact caller source review required")
    terminal = bound_json(phase, scope["native_utility_terminal"])
    qualified = bound_json(phase, scope["native_utility_worker_result"])
    native_scope = bound_json(phase, scope["native_utility_worker_scope"])
    native_supervision = bound_json(phase, scope["native_utility_supervision_scope"])
    result_relative = str(Path(scope["native_utility_terminal"]["path"]).parent / terminal["worker_result"]["path"])
    require(scope["native_utility_worker_result"] == dict(terminal["worker_result"], path=result_relative)
            and terminal["schema"] == "native_utility_owned_supervision_terminal_v1"
            and terminal["status"] == "PASS_SUPERVISOR_TERMINAL_RESOURCE_CLOSURE_NATIVE_SUPPORT_ONLY"
            and terminal["wrapper_resource_closure_PASS"] is True and terminal["owned_child"]["exit_code"] == 0
            and terminal["worker_scope"] == scope["native_utility_worker_scope"]
            and terminal["root_scope_sha256"] == scope["native_utility_supervision_scope"]["sha256"]
            and native_supervision["worker_scope"] == scope["native_utility_worker_scope"]
            and terminal["supervisor_sha256"] == "80611cae8ab2e557a11db72c99321bf5f5d550992dbf16f63374364f48f8f808"
            and qualified["status"] == "PASS_NATIVE_UTILITY_FP32_EPISODE_RECOMMIT_AND_ISOLATED_SUPPORT_RESOURCE_ONLY"
            and qualified["worker_sha256"] == PINS["qualifier"][1] and qualified["restoration_errors"] == []
            and qualified["actual_native_forward_total"] == 54 and qualified["actual_ordinary_grad_API_total"] == 88
            and qualified["native_reverse_constructions_excluding_maps_query"] == 77
            and qualified["model_fits"] == 0 and qualified["persistent_updates"] == 0
            and qualified["A_scoring"] is False and qualified["VALID_TEST_access"] is False,
            "Actual supervised native utility support/resource/restoration required before payload access")
    require(scope["runtime_identity_expected"] == qualified["runtime_identity"]
            and scope["strict_backend_expected"] == qualified["backend_during_native"]
            and scope["GPU_UUID"] == qualified["selected_GPU_UUID"] == native_scope["GPU_UUID"]
            and scope["python_executable"] == native_scope["python_executable"] == str(Path(sys.executable).absolute())
            and scope["python_executable_sha256"] == native_supervision["child_python_sha256"] == sha(sys.executable)
            and scope["site_packages"] == native_scope["site_packages"] and sys.dont_write_bytecode,
            "Fit must use the actual qualified native runtime/backend/device")
    require(scope["comparison_runtime_disposition"] in ("SAME_LIVE_RUNTIME_VERIFIED", "DIFFERENT_LIVE_RUNTIME_DISCLOSED_NO_METHOD_ONLY_ATTRIBUTION"),
            "Freeze LIVE/control runtime comparison interpretation")
    limits = scope["fit_resource_limits"]
    require(scope["external_fit_supervisor_required"] is True and scope["external_fit_terminal_closure_required"] is True
            and all(type(limits[k]) in (float, int) and math.isfinite(limits[k]) and limits[k] > 0 for k in
            ("max_elapsed_seconds", "max_process_rss_bytes", "max_cuda_allocated_bytes", "max_cuda_reserved_bytes")),
            "Distinct positive prospectively fixed H16 fit limits required")
    supervision = bound_json(phase, scope["external_fit_supervision_admission"])
    require(supervision["caller_sha256"] == sha(__file__) and supervision["fit_invocation_id"] == scope["fit_invocation_id"]
            and supervision["common400"] == COMMON and supervision["arm"] == ARM and supervision["H"] == HORIZON
            and supervision["resource_limits"] == limits and supervision["external_watchdog_seconds"] > limits["max_elapsed_seconds"]
            and supervision["one_owned_child_no_retry"] is True and supervision["reviewed_supervisor_sha256"],
            "Reviewed distinct fit supervision/resource admission required")
    supervisor_review=bound_json(phase,supervision["source_review"])
    bound_path(phase,supervision["supervisor_source"])
    require(supervisor_review["status"].startswith("PASS_SOURCE")
            and supervisor_review["supervisor_sha256"]==supervision["reviewed_supervisor_sha256"]==supervision["supervisor_source"]["sha256"]
            and supervisor_review["execution_or_fit_authorized"] is False,"Exact separate fit supervisor source review required")
    origin = bound_json(phase, ORIGIN)
    require(origin["schema"] == "amazon_G0_six_arm_run_v1" and origin["A_labels_received"] is False
            and origin["A_scoring_performed"] is False, "Exact common400 origin metadata required")
    common = bound_path(phase, COMMON)  # Hash only, after native/fit admission; no deserialization yet.
    relative = Path(scope["output_relative_path"])
    require(not relative.is_absolute() and ".." not in relative.parts, "Deliberate relative output required")
    output = phase / relative
    require(not output.exists() and not output.is_symlink() and output.resolve().is_relative_to(phase) and output.parent.is_dir(),
            "Fresh in-phase output with authorized existing parent required")
    return phase, scope, origin, common, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-authorized", action="store_true")
    parser.add_argument("--source-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--fit-scope"); parser.add_argument("--fit-scope-sha256")
    args = parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({"status": "DISABLED_COMMON400_UTILITY_H16_SOURCE_ONLY", "SOURCE_RELEASED": SOURCE_RELEASED})); return 0
    require(SOURCE_RELEASED is False and args.fit_scope and args.fit_scope_sha256, "Explicit separate root fit scope required")
    phase, scope, origin, common_path, output = admission(args)
    output.mkdir(mode=0o700, parents=False, exist_ok=False); identity = (output.stat().st_dev, output.stat().st_ino)
    def owned():
        require(not output.is_symlink() and (output.stat().st_dev, output.stat().st_ino) == identity, "Owned new output changed")
    def write(name, data):
        owned()
        with (output / name).open("x") as stream: json.dump(data, stream, indent=2, allow_nan=False); stream.write("\n")
        (output / name).chmod(0o444)
    def event(kind, **fields):
        owned()
        with (output / "ATTEMPTED_OPERATIONS.jsonl").open("a") as stream:
            stream.write(json.dumps({"kind": kind, "elapsed_seconds": time.monotonic()-STARTED, **fields}, allow_nan=False)+"\n")
    loaded, modules = [], {}
    completed, callbacks, grad_apis = 0, {"continuation": 0, "serving": 0}, 0
    meter = None; torch = numpy = None
    prior_path, prior_rng = list(sys.path), random.getstate()
    prior_env = {k:(k in os.environ, os.environ.get(k)) for k in ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG")}
    prior_signal, prior_timer = signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL)
    require(prior_timer == (0.,0.), "Fresh child without previous timer required")
    old_backend = old_threads = original_grad = old_numeric_rng = None
    error = None; endpoints = {}; restoration_errors = []
    def load(key):
        relative, expected = PINS[key]; path = phase / relative
        require(not path.is_symlink() and path.resolve().is_relative_to(phase)
                and sha(path) == expected and path.stat().st_mode & 0o222 == 0, "Exact readonly source differs: "+key)
        name = "_common400_utility_H16_"+key; require(name not in sys.modules, "Fresh source binding required")
        spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
        sys.modules[name]=module; loaded.append(name); spec.loader.exec_module(module); modules[key]=module; return module
    def resources():
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {"elapsed_seconds": time.monotonic()-STARTED, "process_peak_rss_bytes": rss*1024,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(0) if torch is not None and torch.cuda.is_initialized() else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(0) if torch is not None and torch.cuda.is_initialized() else None,
            "scope": "Worker capture through current stage; final publication/exit require external fit terminal closure; no peak reset"}
    def limits():
        r=resources(); caps=scope["fit_resource_limits"]
        require(r["elapsed_seconds"] <= caps["max_elapsed_seconds"] and r["process_peak_rss_bytes"] <= caps["max_process_rss_bytes"], "Fit elapsed/RSS cap exceeded")
        if r["cuda_peak_allocated_bytes"] is not None:
            require(r["cuda_peak_allocated_bytes"] <= caps["max_cuda_allocated_bytes"] and r["cuda_peak_reserved_bytes"] <= caps["max_cuda_reserved_bytes"], "Fit CUDA cap exceeded")
        return r
    def save_tensor(name, value):
        owned()
        with (output/name).open("xb") as stream: torch.save(value, stream)
        (output/name).chmod(0o444); return {"path": name, "bytes":(output/name).stat().st_size,"sha256":sha(output/name)}
    def source_gates_unchanged():
        for key,module in modules.items():
            require(sha(module.__file__)==PINS[key][1],"Loaded source bytes changed: "+key)
            if key in ("operator","utility","sequential","qualifier"):
                require(module.SOURCE_RELEASED is False,"Original pure gate changed: "+key)
            elif key=="port": require(module.PORT_RELEASED is False,"Original port gate changed")
            elif key in ("accessor","reference"): require(module.SOURCE_RELEASED is True,"Released helper gate changed: "+key)
    try:
        write("RUN.json", {"schema":"common400_utility_H16_run_v1","arm":ARM,"H":HORIZON,"common400":COMMON,"origin_run":ORIGIN,
            "origin_recipe":origin["recipe"],"fit_scope_sha256":args.fit_scope_sha256,"coefficients":COEFFICIENTS,
            "new_acquisition_updates":0,"A_scoring":False,"VALID_TEST_access":False,"original_six_arm_pilot_changed":False})
        def alarm(number, frame): raise TimeoutError("Root fixed H16 worker deadline exceeded")
        signal.signal(signal.SIGALRM, alarm); remaining=scope["fit_resource_limits"]["max_elapsed_seconds"]-(time.monotonic()-STARTED)
        require(remaining>0,"Deadline exhausted before numerical imports"); signal.setitimer(signal.ITIMER_REAL,remaining)
        qualifier=load("qualifier"); utility=load("utility"); sequential=load("sequential"); reference=load("reference"); base=load("base")
        os.environ["CUDA_VISIBLE_DEVICES"]=scope["GPU_UUID"]; os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
        sys.path.insert(0,scope["site_packages"])
        import torch as torch_module
        import numpy as numpy_module
        torch,numpy=torch_module,numpy_module; qualifier.torch=torch; base.torch,base.np=torch,numpy
        require(not torch.cuda.is_initialized(),"Strict configuration must precede CUDA")
        old_backend=qualifier.backend_snapshot(); old_threads=torch.get_num_threads()
        torch.use_deterministic_algorithms(True,warn_only=False); torch.set_num_threads(1); torch.set_num_interop_threads(1)
        old_numeric_rng={"numpy":numpy.random.get_state(),"torch_cpu":torch.get_rng_state().clone(),"cuda":torch.cuda.get_rng_state_all()}
        require(qualifier.backend_snapshot()==scope["strict_backend_expected"] and torch.cuda.device_count()==1
                and qualifier.runtime_identity()==scope["runtime_identity_expected"],"Actual qualified strict runtime/device differs")
        original_grad=torch.autograd.grad
        operator=load("operator"); port=load("port"); accessor=load("accessor"); native=load("native"); boundary=load("boundary")
        require(asdict(operator.Config())==COEFFICIENTS and operator.SOURCE_RELEASED is False and port.PORT_RELEASED is False
                and utility.SOURCE_RELEASED is False and sequential.SOURCE_RELEASED is False and accessor.SOURCE_RELEASED is True,
                "Exact config/original pure source gates differ")
        event("native_utility_and_fit_admission_passed_before_data_and_checkpoint_decode")
        data=accessor.load_public_b(phase,Path(scope["public_b_dir"]),device="cuda:0")
        provenance=data["provenance"]; expected=origin["recipe"]
        require(provenance["public_b_manifest"]["sha256"]==expected["public_b_manifest_sha256"]
                and provenance["public_graph"]["sha256"]==expected["public_graph_sha256"]
                and provenance["roles"]["sha256"]==expected["roles_sha256"]
                and provenance["preprocessing"]["edge_logical_sha256"]==expected["native_edge_logical_sha256"], "Exact common400 public/role/edge custody differs")
        require(data["features"].shape==(24492,300) and data["features"].dtype==torch.float32
                and data["W_ids"].numel()==4898 and data["B_ids"].numel()==9797
                and data["inner_indices"].numel()==2449 and data["query_indices"].numel()==2450,"Complete fixed context differs")
        require(sha(common_path)==COMMON["sha256"] and common_path.stat().st_size==COMMON["bytes"]
                and common_path.stat().st_mode & 0o222 == 0,"Common400 bytes changed before decode")
        payload=torch.load(common_path,map_location="cpu",weights_only=True)
        require(payload["schema"]=="amazon_G0_frozen_state_v1" and payload["id"]=="initial" and payload["warm_updates"]==400
                and payload["episodes"]==0 and payload["warm_role"]=="W" and payload["global_stage"] is True
                and payload["eval_mode"] is True and payload["recipe"]==origin["recipe"],"Exact common400 checkpoint header/recipe differs")
        family,_=reference._fresh_family(native,boundary,"cuda:0"); family.load_state_dict(payload["family_state"],strict=True)
        family.set_global_stage(True); family.eval(); reference._finite_tree(family.state_dict())
        rng=payload["rng"]; random.setstate(rng["python"])
        nr=rng["numpy"]; numpy.random.set_state((nr["name"],numpy.array(nr["keys"],dtype=numpy.uint32),nr["position"],nr["has_gauss"],nr["cached_gaussian"]))
        torch.set_rng_state(rng["torch_cpu"]); torch.cuda.set_rng_state(rng["cuda"],0)
        common_family=reference._cpu_tree(payload["family_state"]); del payload,rng,nr
        forward,theta,phis,_=port._native_callback_and_state(family,data["features"],data["edge_index"],expected_nodes=24492,global_stage=True)
        pairs=port._sparse_pairs(operator,data["inner_indices"],data["inner_labels"],data["edge_index"],node_count=24492,dtype=torch.float32)
        before=base.snapshot(family,data["features"],data["edge_index"],torch.device("cuda:0"))
        class Meter(utility.Counters):
            def add(self,key,stage):
                super().add(key,stage); event("utility_operation_attempt",operation=key,stage=stage,episode=completed+1)
        meter=Meter()
        def counted(core,private,kind="continuation"):
            callbacks[kind]+=1; event("native_callback_attempt",kind_scope=kind,episode=completed+1)
            return forward(core,private)
        def served(name):
            with torch.no_grad():
                logits=torch.stack([counted(theta,phi,"serving") for phi in phis]); require(logits.shape==(4,24492,5) and logits.dtype==torch.float32,"Complete native serving bank required")
                probabilities=torch.softmax(logits,dim=-1); pool=probabilities.mean(dim=0)
                reference._finite_tree((logits,probabilities,pool))
                return save_tensor(name,{"schema":"complete_native_probability_serving_bank_v1","node_ids":torch.arange(24492),
                    "state_id":"common400" if completed==0 else ARM,"episodes":completed,"common_state":COMMON,"origin_run":ORIGIN,
                    "native_member_logits":logits.cpu().clone(),"native_member_softmax_probabilities":probabilities.cpu().clone(),
                    "served_FP32_pool_probabilities":pool.cpu().clone(),"pool":"unweighted arithmetic mean of all4 native softmax probabilities",
                    "labels_received":False,"scoring_performed":False})
        endpoints["common400_predictions"]=served("common400_complete_predictions.pt")
        history=[]
        @contextmanager
        def grad_observer():
            def observed(*a,**k):
                nonlocal grad_apis
                grad_apis+=1; event("ordinary_grad_API_attempt",episode=completed+1); return original_grad(*a,**k)
            torch.autograd.grad=observed
            try: yield
            finally: torch.autograd.grad=original_grad
        for episode in range(HORIZON):
            limits(); event("episode_started",episode=episode+1); counts_before=dict(meter.total); f_before=callbacks["continuation"]; g_before=grad_apis
            with grad_observer():
                nt,np_,info,inspection=utility._engineering_episode(operator,PINS["operator"][1],port,PINS["port"][1],sequential,PINS["sequential"][1],
                    theta,phis,counted,pairs,data["inner_indices"],data["inner_labels"],data["query_indices"],data["query_labels"],
                    engineering_authorized=True,counters=meter,collect_inspection=False)
            require(inspection is None and torch.autograd.grad is original_grad,"Detached episode/observer restoration differs")
            actual={k:meter.total[k]-counts_before[k] for k in EPISODE}
            require(actual==EPISODE and callbacks["continuation"]-f_before==40 and grad_apis-g_before==63,"Complete actual episode bill differs")
            reference._check_commit(theta,phis,nt,np_); reference._finite_tree(info); qualifier.validate_diagnostics(info)
            theta,phis=nt,np_; history.append(reference._cpu_tree(info)); completed=episode+1
            event("episode_completed",episode=completed,counters=actual); limits()
        require(completed==16 and meter.total=={k:16*v for k,v in EPISODE.items()} and callbacks["continuation"]==640 and grad_apis==1008,"H16 actual bill/completion differs")
        base.unchanged(before,family,data["features"],data["edge_index"],torch.device("cuda:0"))
        endpoints["utility_predictions"]=served("utility_H16_complete_predictions.pt")
        reference._install(family,theta,phis,port.PRIVATE_NAMES)
        state=reference._cpu_tree(family.state_dict())
        endpoints["utility_state"]=save_tensor("utility_H16.pt",{"schema":"amazon_G0_frozen_state_v1","id":ARM,"episodes":16,"warm_updates":400,
            "warm_role":"W","recipe":origin["recipe"],"common_state":COMMON,"origin_run":ORIGIN,"global_stage":True,"eval_mode":True,"family_state":state,
            "coefficients":COEFFICIENTS,"fit_scope_sha256":args.fit_scope_sha256})
        endpoints["B_diagnostics"]=save_tensor("utility_H16_B_diagnostics.pt",{"episodes":tuple(history),"endpoint_vs_common":reference._distances(state,common_family),
            "interpretation":"Training S/R diagnostics only; no held-label selector; utility/paid finite costs remain distinct"})
        require(callbacks=={"continuation":640,"serving":8},"Complete endpoint callback custody differs"); source_gates_unchanged(); limits()
    except BaseException as e:
        error={"type":type(e).__name__,"error":str(e),"traceback":traceback.format_exc()}
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        def restore(label,function):
            try: function()
            except BaseException as e: restoration_errors.append({"label":label,"type":type(e).__name__,"error":str(e)})
        restore("source_bytes_and_process_gates",source_gates_unchanged)
        if original_grad is not None: restore("ordinary_grad",lambda:setattr(torch.autograd,"grad",original_grad))
        if old_backend is not None: restore("backend",lambda:modules["qualifier"].backend_restore(old_backend))
        if old_threads is not None: restore("intra_op",lambda:torch.set_num_threads(old_threads))
        if old_numeric_rng is not None:
            def numerical_rng_restore():
                numpy.random.set_state(old_numeric_rng["numpy"]); torch.set_rng_state(old_numeric_rng["torch_cpu"])
                torch.cuda.set_rng_state_all(old_numeric_rng["cuda"])
                require(torch.equal(torch.get_rng_state(),old_numeric_rng["torch_cpu"])
                        and all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),old_numeric_rng["cuda"])),"Numerical RNG restore failed")
            restore("NumPy_Torch_CPU_visible_CUDA_RNG",numerical_rng_restore)
        def python_restore():
            random.setstate(prior_rng); sys.path[:]=prior_path
            for name in loaded: sys.modules.pop(name,None)
            for name,(present,value) in prior_env.items():
                if present: os.environ[name]=value
                else: os.environ.pop(name,None)
            signal.signal(signal.SIGALRM,prior_signal); signal.setitimer(signal.ITIMER_REAL,*prior_timer)
        restore("python_env_signal_bindings",python_restore)
        event("terminal_capture",completed_episodes=completed)
        events=output/"ATTEMPTED_OPERATIONS.jsonl"; events.chmod(0o444)
        result={"schema":"common400_utility_H16_worker_terminal_v1","status":"COMPLETE_UTILITY_H16_WORKER_CAPTURE_PENDING_EXTERNAL_RESOURCE_CLOSURE" if error is None and not restoration_errors else "FAILED_UTILITY_H16",
            "arm":ARM,"H":16,"completed_episodes":completed,"common400":COMMON,"origin_run":ORIGIN,"endpoints":endpoints,
            "operation_counts":meter.snapshot() if meter is not None else None,"independent_native_callbacks":callbacks,"ordinary_grad_API_attempts":grad_apis,
            "worker_resources":resources(),"primary_error":error,"restoration_errors":restoration_errors,
            "scientific_continuations_attempted":1 if meter is not None else 0,"committed_core_updates":completed,
            "committed_private_row_updates":4*completed,"fit_scope_sha256":args.fit_scope_sha256,
            "new_acquisition_updates":0,"Adam_history_activated":False,"A_scoring":False,"VALID_TEST_access":False,"original_pilot_changed":False,
            "external_fit_terminal_closure_required":True,"interop_restoration":"Fresh child only; one-time1 initialization ends with child termination",
            "attempted_events":{"path":events.name,"bytes":events.stat().st_size,"sha256":sha(events)}}
        write("WORKER_TERMINAL.json",result)
    print(json.dumps({"status":result["status"],"completed_episodes":completed,"A_scoring":False,"external_closure_pending":True}))
    return 0 if error is None and not restoration_errors else 1


if __name__ == "__main__": raise SystemExit(main())
