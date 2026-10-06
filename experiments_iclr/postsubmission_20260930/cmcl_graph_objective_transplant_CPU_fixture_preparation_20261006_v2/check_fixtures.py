"""Disabled one-shot CPU float64 CMCL rational/analytic fixture executor.

No native model, data loader, accelerator API, optimizer, fit or supervisor.
Closed Python scalar/list expectations; exact reviewed helper stays unchanged.
"""
import time
STARTED = time.monotonic()
import argparse
from fractions import Fraction
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
REPOSITORY = "/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs"
PHASE = REPOSITORY + "/experiments_iclr/postsubmission_20260930"
PYTHON = PHASE + "/native_ncn_runtime_20261005_v1/.venv/bin/python"
SITE = REPOSITORY + "/.venv/lib/python3.11/site-packages"
TORCH_PATH = SITE + "/torch/__init__.py"
HELPER = {"path": "cmcl_graph_objective_transplant_source_preparation_20261006_v1/cmcl_loss.py",
    "bytes": 4330, "sha256": "4f04eb87381512c9f7506b6458917b83fd12228f3c7651c6291e902c99fe6ac9"}
PLAN = {"path": "cmcl_graph_objective_transplant_source_preparation_20261006_v1/SYNTHETIC_FIXTURE_PLAN.json",
    "bytes": 5092, "sha256": "4342d22604aad68e66710161e2235ba273b05ad87b42d6092f350e172c618c8b"}
HELPER_REVIEW = {"path": "cmcl_graph_objective_transplant_root_independent_source_review_20261006_v1/FINDINGS.json",
    "bytes": 1839, "sha256": "6f73a887ed6411d444b1f32280c8f68b15161bd322bedf83708ff226ffa45ff2"}
FIXTURES = ("uniform_tie_member_sum_and_zero_nonowner", "rational_probability_order_and_exact_closed_loss",
    "KL_direction_and_gradient_rejection", "owner_ranking_not_CE_only_or_all_member_anchor",
    "equal_aggregate_role_mass_unequal_node_counts", "complete_arithmetic_probability_serving")
ATOL, RTOL = 1e-12, 1e-12


def require(value, message):
    if not value: raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(root, row):
    relative = Path(row["path"])
    require(not relative.is_absolute() and ".." not in relative.parts, "Relative frozen source/metadata path required")
    path = root / relative
    require(not path.is_symlink() and path.resolve().is_relative_to(root) and path.is_file()
        and path.stat().st_mode & 0o222 == 0 and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"],
        "Exact immutable source/metadata binding differs")
    return path


def resources():
    return {"elapsed_seconds_including_imports_checks_and_restore": time.monotonic() - STARTED,
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024}


def limits(caps):
    row = resources()
    require(row["elapsed_seconds_including_imports_checks_and_restore"] <= caps["max_elapsed_seconds"]
        and row["process_peak_rss_bytes"] <= caps["max_process_rss_bytes"], "Root CPU fixture whole-worker caps exceeded")


def close(actual, expected):
    """Independent recursive scalar/list comparator, no helper tensor math."""
    if isinstance(expected, (tuple, list)):
        require(isinstance(actual, (tuple, list)) and len(actual) == len(expected), "Analytic reference structure differs")
        return max((close(a, e) for a, e in zip(actual, expected)), default=0.0)
    a, e = float(actual), float(expected)
    require(math.isfinite(a) and math.isfinite(e), "Nonfinite fixture scalar")
    error = abs(a-e)
    require(error <= ATOL + RTOL*abs(e), "Closed analytic fixture tolerance violated: " + str(error))
    return error


def cpu_backend():
    return {"deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "debug_mode": torch.get_deterministic_debug_mode(), "dtype": str(torch.get_default_dtype()),
        "default_device": str(torch.empty(0).device), "grad_enabled": torch.is_grad_enabled(),
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "mkldnn_deterministic_supported": hasattr(torch.backends.mkldnn, "deterministic"),
        "mkldnn_deterministic": getattr(torch.backends.mkldnn, "deterministic", None)}


def probability(a):
    return [a] + [(1-a)/4]*4


UNIFORM = [Fraction(1,5)]*5
RATIONAL = [probability(Fraction(1,2)), probability(Fraction(1,4)),
    probability(Fraction(1,8)), probability(Fraction(1,16))]
RANKING = [UNIFORM, probability(Fraction(2,5)), probability(Fraction(1,2)),
    [Fraction(1,10), Fraction(897,1000), Fraction(1,1000), Fraction(1,1000), Fraction(1,1000)]]


def expected_gradient(bank, owners, nodes, coefficient):
    # Explicit dCE/dz=P-e0 and dKL(U||P)/dz=P-U, evaluated from fractions.
    return [[[float(coefficient*((p - (1 if c == 0 else 0)) if m in owners else Fraction(3,4)*(p-Fraction(1,5))))
        for c,p in enumerate(row)] for _ in range(nodes)] for m,row in enumerate(bank)]


def main():
    global torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-authorized", action="store_true"); parser.add_argument("--source-root")
    parser.add_argument("--scope"); parser.add_argument("--scope-sha256"); parser.add_argument("--output")
    args = parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({"status": "DISABLED_CPU_CMCL_ANALYTIC_FIXTURE_EXECUTOR", "SOURCE_RELEASED": False,
            "source_import_or_numerical_execution": False})); return 0
    require(SOURCE_RELEASED is False and args.source_root == PHASE and socket.gethostname() == "anogena-2-0"
        and Path.cwd().absolute() == Path(REPOSITORY) and Path(sys.executable).absolute() == Path(PYTHON)
        and sys.dont_write_bytecode and args.scope and args.scope_sha256 and args.output
        and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not any(n == "torch" or n.startswith("torch.") for n in sys.modules),
        "Exact fresh allocation normal interpreter/-B and empty device exposure required before paths/imports")
    root = Path(PHASE).resolve()
    def deliberate(value, new):
        path = Path(value).absolute()
        require(".." not in path.parts and path.is_relative_to(Path(PHASE)) and not path.is_symlink()
            and path.resolve().is_relative_to(root), "Deliberate in-phase path required")
        path = path.resolve()
        require((not path.exists() and path.parent.is_dir()) if new else path.is_file(), "New output/frozen scope required")
        return path
    scope_path, output = deliberate(args.scope, False), deliberate(args.output, True)
    require(scope_path.stat().st_mode & 0o222 == 0 and sha(scope_path) == args.scope_sha256, "Exact readonly root CPU scope required")
    scope = json.loads(scope_path.read_text())
    own = {"path": str(Path(__file__).resolve().relative_to(root)), "bytes": Path(__file__).stat().st_size, "sha256": sha(__file__)}
    bound(root,own)
    require(scope["schema"] == "root_CMCL_CPU_float64_analytic_fixture_scope_v1"
        and scope["root_engineering_invocation_authorized"] is True and scope["fixed_before_execution"] is True
        and scope["executor"] == own and scope["helper"] == HELPER and scope["fixture_plan"] == PLAN
        and scope["helper_review"] == HELPER_REVIEW and scope["fixture_ids"] == list(FIXTURES)
        and scope["python_executable"] == PYTHON and scope["site_packages"] == SITE
        and scope["CPU_only"] is True and scope["CUDA_VISIBLE_DEVICES"] == ""
        and scope["one_owned_root_invocation_no_retry"] is True
        and scope["no_model_data_fit_scoring"] is True and scope["A_VALID_TEST_access"] is False
        and scope["ATOL"] == ATOL and scope["RTOL"] == RTOL, "Frozen exact six-fixture engineering scope differs")
    review = json.loads(bound(root, HELPER_REVIEW).read_text())
    require(review["status"] == "PASS_SOURCE_MATH_ONLY" and review["source"] == HELPER
        and review["new_source_blockers"] == [] and review["execution_or_fit_authorized"] is False, "Original root helper math review differs")
    executor_review = json.loads(bound(root, scope["executor_review"]).read_text())
    require(executor_review["status"].startswith("PASS_SOURCE") and executor_review["executor"] == own
        and executor_review["new_source_blockers"] == [] and executor_review["execution_or_fit_authorized"] is False,
        "New independent exact executor source review required")
    plan = json.loads(bound(root, PLAN).read_text())
    require([row["id"] for row in plan["fixtures"]] == list(FIXTURES), "Frozen original fixture order differs")
    caps = scope["resource_limits"]
    require(set(caps) == {"max_elapsed_seconds", "max_process_rss_bytes"}
        and all(type(v) in (int,float) and math.isfinite(v) and v > 0 for v in caps.values())
        and scope["external_terminal_resource_closure_required"] is True and scope["external_watchdog_required"] is True
        and type(scope["external_watchdog_seconds"]) in (int,float) and math.isfinite(scope["external_watchdog_seconds"])
        and scope["external_watchdog_seconds"]>caps["max_elapsed_seconds"], "Root CPU wall/RSS and external terminal closure required")
    helper_path = bound(root, HELPER)
    output.mkdir(parents=False, exist_ok=False); information = output.stat(); identity = (information.st_dev, information.st_ino)
    receipt = {"schema": "CMCL_CPU_float64_analytic_fixture_result_v1", "status": "RUNNING",
        "executor": own, "helper": HELPER, "fixture_plan": PLAN, "helper_review": HELPER_REVIEW,
        "executor_review": scope["executor_review"], "scope_sha256": sha(scope_path), "checks": [], "restoration_errors": [],
        "attempts": {"positive_role_objective_calls": 0, "positive_serving_calls": 0, "ordinary_grad_API_calls": 0, "rejection_calls": 0},
        "model_fits": 0, "native_callbacks": 0, "CUDA_API_calls": 0, "optimizer_updates": 0,
        "actual_dataset_labels_read": False, "A_VALID_TEST_access": False,
        "external_terminal_resource_closure_required": True, "engineering_only_not_research_evidence": True,
        "created_output_identity": {"path": str(output), "device": identity[0], "inode": identity[1]}}
    def save():
        require(not output.is_symlink() and (output.stat().st_dev, output.stat().st_ino) == identity, "Only owned fresh output may be written")
        receipt["whole_process_resources"] = resources()
        temporary = output/"RESULT.tmp"; temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
        temporary.replace(output/"RESULT.json")
    def check(name, function):
        limits(caps); receipt["last_started_check"] = name; save(); begun = time.monotonic()
        try:
            value = function(); row = {"name": name, "status": "PASS", "result": value}
        except BaseException as error:
            row = {"name": name, "status": "FAIL", "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
            if isinstance(error, (KeyboardInterrupt, SystemExit, TimeoutError, InterruptedError)):
                receipt["checks"].append(dict(row, elapsed_seconds=time.monotonic()-begun)); save(); raise
        receipt["checks"].append(dict(row, elapsed_seconds=time.monotonic()-begun)); save(); limits(caps)
    old_path, python_rng = list(sys.path), random.getstate()
    old_handler, old_timer = signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL)
    module_name = "_CMCL_reviewed_pure_loss_fixture_subject"
    helper = previous_rng = previous_backend = numpy = numpy_state = None
    inputs = []
    def byte_hash(tensor):
        return hashlib.sha256(bytes(tensor.detach().contiguous().view(torch.uint8).reshape(-1).tolist())).hexdigest()
    def register(tensor):
        meta = tensor.device.type == "meta"
        inputs.append({"tensor": tensor, "id": id(tensor), "storage": tensor.untyped_storage()._cdata,
            "pointer": tensor.untyped_storage().data_ptr(), "version": tensor._version,
            "shape": tuple(tensor.shape), "stride": tuple(tensor.stride()), "dtype": tensor.dtype,
            "device": tensor.device, "requires_grad": tensor.requires_grad, "hash": None if meta else byte_hash(tensor)})
        return tensor
    def logits(bank, nodes=1):
        return register(torch.tensor([[[math.log(float(p)) for p in row] for _ in range(nodes)] for row in bank],
            dtype=torch.float64, device="cpu", requires_grad=True))
    def targets(nodes=1):
        return register(torch.zeros(nodes, dtype=torch.int64, device="cpu"))
    def objective(bank_S, bank_R, nodes_S=1, nodes_R=1):
        a,b,ya,yb = logits(bank_S,nodes_S),logits(bank_R,nodes_R),targets(nodes_S),targets(nodes_R)
        receipt["attempts"]["positive_role_objective_calls"] += 1
        loss, info = helper._engineering_role_objective(a,ya,b,yb,engineering_authorized=True)
        return a,b,loss,info
    def gradient(value, values):
        receipt["attempts"]["ordinary_grad_API_calls"] += 1
        return torch.autograd.grad(value,values)
    def owners(info, expected):
        mask = [[m in expected]*info["S"]["owner_mask"].shape[1] for m in range(4)]
        require(info["S"]["owner_mask"].tolist() == mask, "Predetermined analytic owner mask differs")
        require(not info["S"]["owner_mask"].requires_grad and not info["S"]["owner_indices"].requires_grad, "Discrete owners not stopped")
    def reject(function):
        receipt["attempts"]["rejection_calls"] += 1
        try: function()
        except RuntimeError: return {"expected_RuntimeError_rejection": True}
        raise RuntimeError("Frozen invalid/default-disabled call was accepted")
    code = 1
    try:
        require(old_timer == (0.0,0.0), "Fresh CPU worker timer required")
        def expired(signum, frame): raise TimeoutError("Root frozen CPU fixture deadline exceeded")
        remaining=caps["max_elapsed_seconds"]-(time.monotonic()-STARTED)
        require(remaining>0,"Root CPU deadline exhausted before helper import")
        signal.signal(signal.SIGALRM,expired); signal.setitimer(signal.ITIMER_REAL,remaining)
        require(module_name not in sys.modules, "Fresh helper binding required")
        spec = importlib.util.spec_from_file_location(module_name,helper_path)
        helper = importlib.util.module_from_spec(spec); sys.modules[module_name] = helper; spec.loader.exec_module(helper)
        require(helper.SOURCE_RELEASED is False, "Exact helper source must stay disabled")
        for name,function in (("public_training_disabled_pre_Torch",lambda: helper.training_role_objective(None,None,None,None)),
            ("public_serving_disabled_pre_Torch",lambda: helper.mean_probability_serving(None)),
            ("engineering_loss_default_disabled_pre_Torch",lambda: helper._engineering_role_objective(None,None,None,None)),
            ("engineering_serving_default_disabled_pre_Torch",lambda: helper._engineering_mean_probabilities(None))):
            check(name,lambda function=function: reject(function))
        require(not any(n == "torch" or n.startswith("torch.") for n in sys.modules), "Disabled guard imported numerical package")
        sys.path.insert(0,SITE)
        import torch
        require(torch.__version__ == "2.1.2+cu118" and str(Path(torch.__file__).resolve()) == TORCH_PATH,
            "Exact existing allocation normal Torch required; no fallback")
        previous_backend = cpu_backend(); require(previous_backend["default_device"] == "cpu", "Existing default must be CPU")
        previous_rng = torch.get_rng_state().clone()
        numpy = sys.modules.get("numpy"); numpy_state = numpy.random.get_state() if numpy is not None else None
        receipt["runtime"] = {"hostname": socket.gethostname(), "python_executable": sys.executable,
            "torch_version": torch.__version__, "torch_module_path": str(Path(torch.__file__).resolve()), "CPU_backend_unchanged_basis": previous_backend}
        def uniform_case():
            a,b,loss,info = objective([UNIFORM]*4,[UNIFORM]*4); owners(info,{0,1,2})
            require(info["S"]["owner_indices"].tolist() == [[0],[1],[2]], "Uniform exact member-index ties differ")
            gs,gr = gradient(loss,(a,b)); expected = expected_gradient([UNIFORM]*4,{0,1,2},1,Fraction(1,2))
            return {"loss_max_error":close(float(loss),3*math.log(5)), "gradient_max_error":close([gs.tolist(),gr.tolist()],[expected,expected]),
                "nonowner_reference_exact_zero":True,"nonowner_gradient_error":close([gs[3].tolist(),gr[3].tolist()],[[[0.0]*5],[[0.0]*5]])}
        rational_loss = math.log(64)+Fraction(3,20)*math.log(Fraction(16*64**4,5**5*15**4))
        def rational_case():
            a,b,loss,info = objective(RATIONAL,RATIONAL); owners(info,{0,1,2})
            gs,gr = gradient(loss,(a,b)); expected = expected_gradient(RATIONAL,{0,1,2},1,Fraction(1,2))
            return {"loss_max_error":close(float(loss),rational_loss),"gradient_max_error":close([gs.tolist(),gr.tolist()],[expected,expected])}
        def KL_case():
            bank=[probability(Fraction(1,2))]*4; a,b,loss,info=objective(bank,bank)
            forward_KL=Fraction(13,5)*math.log(2)-math.log(5); reverse_KL=math.log(Fraction(5,4))
            require(abs(forward_KL-reverse_KL)>ATOL+RTOL*abs(reverse_KL), "Closed direction negative control trivial")
            g,=gradient(info["S"]["KL_uniform_to_member"][3,0],(a,))
            expected=[[[float(p-Fraction(1,5)) if m==3 else 0.0 for p in row]] for m,row in enumerate(bank)]
            return {"KL_max_error":close(float(info["S"]["KL_uniform_to_member"][3,0]),forward_KL),
                "loss_max_error":close(float(loss),3*math.log(2)+Fraction(3,4)*forward_KL),"gradient_max_error":close(g.tolist(),expected),"reverse_KL_rejected":True}
        def ranking_case():
            a,b,loss,info=objective(RANKING,RANKING); owners(info,{1,2,3})
            require(Fraction(897**3,5**5*10**19)<1, "Closed CMCL-versus-CE ranking proof differs")
            gs,gr=gradient(loss,(a,b)); expected=expected_gradient(RANKING,{1,2,3},1,Fraction(1,2))
            zero_error=close([gs[0].tolist(),gr[0].tolist()],[[[0.0]*5],[[0.0]*5]])
            return {"loss_max_error":close(float(loss),math.log(50)),"gradient_max_error":close([gs.tolist(),gr.tolist()],[expected,expected]),
                "CE_only_owners_would_be_0_1_2":True,"uniform_nonowner_anchor_rejected":True,"nonowner_gradient_error":zero_error}
        def roles_case():
            errors=[]
            for count in (2,4):
                a,b,loss,info=objective(RATIONAL,[UNIFORM]*4,1,count); owners(info,{0,1,2})
                gs,gr=gradient(loss,(a,b)); expected_S=expected_gradient(RATIONAL,{0,1,2},1,Fraction(1,2))
                expected_R=expected_gradient([UNIFORM]*4,{0,1,2},count,Fraction(1,2*count))
                errors.append({"R_nodes":count,"loss_max_error":close(float(loss),(rational_loss+3*math.log(5))/2),
                    "gradient_max_error":close([gs.tolist(),gr.tolist()],[expected_S,expected_R])})
            return {"role_duplication_checks":errors,"equal_aggregate_role_mass":True}
        def serving_case():
            a=logits(RATIONAL); receipt["attempts"]["positive_serving_calls"]+=1
            actual=helper._engineering_mean_probabilities(a,engineering_authorized=True)
            expected=[[Fraction(15,64)]+[Fraction(49,256)]*4]
            require(Fraction(15,64)!=Fraction(7,24), "Closed owner-subset negative control trivial")
            return {"serving_max_error":close(actual.tolist(),expected),"owner_only_mean_rejected":True}
        for name,function in zip(FIXTURES,(uniform_case,rational_case,KL_case,ranking_case,roles_case,serving_case)):check(name,function)
        a,b,ya,yb=logits(RATIONAL),logits(RATIONAL),targets(),targets()
        invalids = (
            ("wrong_M",register(torch.zeros((3,1,5),dtype=torch.float64,device="cpu")),ya,b,yb),
            ("wrong_C",register(torch.zeros((4,1,4),dtype=torch.float64,device="cpu")),ya,b,yb),
            ("empty_role",register(torch.zeros((4,0,5),dtype=torch.float64,device="cpu")),register(torch.zeros(0,dtype=torch.int64,device="cpu")),b,yb),
            ("out_of_range_target",a,register(torch.tensor([5],dtype=torch.int64,device="cpu")),b,yb),
            ("negative_target",a,register(torch.tensor([-1],dtype=torch.int64,device="cpu")),b,yb),
            ("non_int64_target",a,register(torch.tensor([0],dtype=torch.int32,device="cpu")),b,yb),
            ("nonfinite_logits",register(torch.full((4,1,5),float("nan"),dtype=torch.float64,device="cpu")),ya,b,yb),
            ("role_dtype_mismatch",a,ya,register(torch.zeros((4,1,5),dtype=torch.float32,device="cpu")),yb),
            ("role_device_mismatch_unmaterialized_meta",a,ya,register(torch.empty((4,1,5),dtype=torch.float64,device="meta")),yb))
        for name,x,y,z,w in invalids:
            check("negative_input_"+name,lambda x=x,y=y,z=z,w=w:reject(lambda:helper._engineering_role_objective(x,y,z,w,engineering_authorized=True)))
        require(receipt["attempts"] == {"positive_role_objective_calls":6,"positive_serving_calls":1,"ordinary_grad_API_calls":6,"rejection_calls":13},
            "Complete frozen fixture attempted bill differs")
        require(all(row["status"]=="PASS" for row in receipt["checks"]) and len(receipt["checks"])==19, "Any failed/missing frozen check rejects qualification")
        code=0
    except BaseException as error:
        receipt.update(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        def restore(name,function):
            try: function(); receipt[name]=True
            except BaseException as error: receipt["restoration_errors"].append({"check":name,"error":str(error),"traceback":traceback.format_exc()})
        if "torch" in globals():
            def input_custody():
                for row in inputs:
                    t=row["tensor"]
                    require(id(t)==row["id"] and t.untyped_storage()._cdata==row["storage"] and t.untyped_storage().data_ptr()==row["pointer"]
                        and t._version==row["version"] and tuple(t.shape)==row["shape"] and tuple(t.stride())==row["stride"]
                        and t.dtype==row["dtype"] and t.device==row["device"] and t.requires_grad==row["requires_grad"] and t.grad is None
                        and (row["hash"] is None or byte_hash(t)==row["hash"]),"Synthetic input value/object/storage/flags changed")
                receipt["synthetic_input_objects_verified"]=len(inputs)
            restore("all_synthetic_input_values_objects_storage_flags_unchanged",input_custody)
            if previous_rng is not None:
                unchanged=torch.equal(torch.get_rng_state(),previous_rng); torch.set_rng_state(previous_rng)
                restore("Torch_CPU_rng_unchanged_and_restored",lambda:require(unchanged and torch.equal(torch.get_rng_state(),previous_rng),"CPU RNG changed"))
            if previous_backend is not None:restore("CPU_backend_default_device_dtype_grad_and_threads_unchanged",lambda:require(cpu_backend()==previous_backend,"CPU settings changed"))
            if numpy_state is not None:
                now=numpy.random.get_state(); unchanged=now[0]==numpy_state[0] and now[1].tobytes()==numpy_state[1].tobytes() and now[2:]==numpy_state[2:]
                numpy.random.set_state(numpy_state)
                after=numpy.random.get_state()
                restore("existing_NumPy_rng_unchanged_and_restored",lambda:require(unchanged and after[0]==numpy_state[0]
                    and after[1].tobytes()==numpy_state[1].tobytes() and after[2:]==numpy_state[2:],"NumPy RNG changed"))
        unchanged=random.getstate()==python_rng; random.setstate(python_rng)
        restore("Python_rng_unchanged_and_restored",lambda:require(unchanged and random.getstate()==python_rng,"Python RNG changed"))
        sys.path[:]=old_path; sys.modules.pop(module_name,None)
        signal.signal(signal.SIGALRM,old_handler); signal.setitimer(signal.ITIMER_REAL,*old_timer)
        restore("path_helper_binding_timer_and_environment_unchanged",lambda:require(sys.path==old_path and module_name not in sys.modules
            and signal.getsignal(signal.SIGALRM)==old_handler and signal.getitimer(signal.ITIMER_REAL)==old_timer
            and os.environ.get("CUDA_VISIBLE_DEVICES")=="","Reversible process state differs"))
        restore("exact_sources_scope_and_review_bytes_unchanged",lambda:require(sha(__file__)==own["sha256"] and sha(scope_path)==args.scope_sha256
            and all(bound(root,row) for row in (HELPER,PLAN,HELPER_REVIEW,scope["executor_review"]))
            and (helper is None or helper.SOURCE_RELEASED is False),"Source/review/scope bytes changed"))
        if receipt["restoration_errors"]:code=1
        try: limits(caps)
        except BaseException as error:code=1;receipt["final_worker_resource_error"]=str(error)
        receipt["status"]="PASS_CPU_FLOAT64_CMCL_ANALYTIC_FIXTURES_ENGINEERING_ONLY" if code==0 else "FAIL_CPU_FLOAT64_CMCL_ANALYTIC_FIXTURES"
        receipt["completed_frozen_checks"]=len(receipt["checks"])
        result=None
        try:
            save();(output/"RESULT.json").chmod(0o444)
            result={"path":"RESULT.json","bytes":(output/"RESULT.json").stat().st_size,"sha256":sha(output/"RESULT.json")}
            publication={"schema":"CMCL_CPU_fixture_worker_publication_v1","status":receipt["status"],"result":result,"executor":own,
                "worker_exit_code_planned":code,"external_terminal_resource_closure_required":True,"fit_scoring_or_research_evidence":False}
            temporary=output/"PUBLICATION.tmp";temporary.write_text(json.dumps(publication,indent=2)+"\n");temporary.replace(output/"PUBLICATION.json");(output/"PUBLICATION.json").chmod(0o444)
        except BaseException as error:
            code=1;receipt.update(status="FAIL_CPU_FLOAT64_CMCL_ANALYTIC_FIXTURES",output_finalization_error=str(error))
            try:
                save();(output/"RESULT.json").chmod(0o444)
                result={"path":"RESULT.json","bytes":(output/"RESULT.json").stat().st_size,"sha256":sha(output/"RESULT.json")}
            except BaseException as recovery_error:receipt["failure_publication_error"]=str(recovery_error)
    print(json.dumps({"status":receipt["status"],"result":result,"engineering_only":True,"fit_authorized":False}))
    return code


if __name__=="__main__":raise SystemExit(main())
