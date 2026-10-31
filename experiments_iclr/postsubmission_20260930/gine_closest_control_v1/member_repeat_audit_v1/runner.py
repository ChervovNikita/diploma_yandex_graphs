"""Source-only prepared full-graph repeat audit. Root separately admits execution.

Exactly Reddit2/seed17/exact_member/INITIAL, p0, checkpoint_blocks=False.
Four serial passes, no backward, loss, score, optimizer, or tensor export.
This numerical engineering audit is not positive scientific evidence.
"""
import sys
import os
sys.path[:] = [p for p in sys.path if p and os.path.isabs(p)
               and any(os.path.commonpath([p, root]) == root for root in (sys.prefix, sys.base_prefix))]
sys.dont_write_bytecode = True
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import resource
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
PACKAGE = str(HERE.relative_to(PHASE))
V3 = "gine_closest_control_v1/member_gradient_probe_integration_v3/runner.py"
HELPER = "gine_closest_control_v1/member_gradient_probe_v1/probe.py"
SCIENTIFIC_SHA = "e4d1b4b7d96ae07e7588a854cb8a0d70e9af6612ac72d1df583c3c637c376c9b"
CELL = dict(task="Reddit2", seed=17, arm="exact_member")
PASSES = (("no_grad_A", False), ("no_grad_B", False), ("grad_A", True), ("grad_B", True))
FIXED = dict(child_wall_cap_seconds=120, termination_grace_seconds=5,
             combined_phase_wall_cap_seconds=21600, combined_phase_disk_cap_bytes=200 * 2**30,
             diagnostic_wall_cap_seconds=2400, diagnostic_artifact_cap_bytes=64 * 2**20,
             reserved_GPU_memory_cap_bytes=60 * 2**30, serial_repeat_atol=1e-6,
             complete_official_training_nodes=True, members=4, passes=[dict(name=n, grad_enabled=g) for n,g in PASSES],
             dropout_probability=0., checkpoint_blocks=False, scalar_reduction_node_chunk=8192,
             forward_calls=16, vjp_calls=0, loss_calls=0, optimizer_steps=0,
             validation_or_test_targets_used=False, original_score_recalculation=False,
             tensor_export=False, tolerance_amendment=False, scientific_positive_evidence=False)


def confined(path):
    path = Path(path).absolute()
    if not path.is_relative_to(PHASE):
        raise ValueError("Outside authorized project phase")
    current = PHASE
    for name in path.relative_to(PHASE).parts:
        current /= name
        if current.is_symlink():
            raise ValueError("Evidence/output symlinks refused")
    return path.resolve()


def sha(path):
    h = hashlib.sha256()
    with confined(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def checked(reference):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("Canonical path/hash reference required")
    path = confined(PHASE / reference["path"])
    if str(path.relative_to(PHASE)) != reference["path"] or sha(path) != reference["sha256"]:
        raise ValueError("Input identity changed: " + str(reference["path"]))
    return path


def ref(path):
    path = confined(path)
    return dict(path=str(path.relative_to(PHASE)), sha256=sha(path))


def read(reference):
    return json.loads(checked(reference).read_text())


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), module.__dict__)
    return module


def stage_paths():
    root = PACKAGE + "/runs/authorized1gpu_run01"
    return dict(stage="repeat", output=root+"/numerical", supervisor=root+"/supervisor", outer=root+"/outer")


def bootstrap(request):
    seal = read(request["source_seal"])
    if (seal.get("schema") != "gnnm-member-repeat-source-only-v1" or seal.get("execution_authorized") is not False
            or request["fixed_policy"] != FIXED or request["scientific_manifest_sha256"] != SCIENTIFIC_SHA):
        raise ValueError("Exact separately sealed numerical policy required")
    own = seal["importable_sources"]
    if set(own) != {PACKAGE+"/runner.py", V3, HELPER}:
        raise ValueError("Only new caller, unchanged v3 utility caller and original helper may be added")
    for path, wanted in own.items():
        checked(dict(path=path, sha256=wanted))
    manifest = read(dict(path="shared_message_predictive_v3/SOURCE_MANIFEST_v1.json", sha256=SCIENTIFIC_SHA))
    guard_row = next(r for r in manifest["files"] if r["relative_path"] == "import_guard.py")
    guard = load_source("_member_repeat_inherited_guard_v1", checked(dict(path="shared_message_predictive_v3/import_guard.py", sha256=guard_row["sha256"])))
    policy, tree = guard.verify_packet(SCIENTIFIC_SHA)
    policy = copy.deepcopy(policy)
    policy["importable_sha256"].update(own)
    for path in own:
        guard.inspect_tree((PHASE/path).parent, [path], policy["package_directories"])
    guard.installed_path_only()
    guard.verify_generator_installation(policy)
    guard.imported_modules(policy)
    sys.meta_path.insert(0, guard.PhaseFinder(policy))
    v3 = load_source("_member_repeat_unchanged_v3_utilities", PHASE/V3)
    inputs = read(seal["existing_evidence"])
    return guard, policy, v3, inputs, seal


class Budget:
    def __init__(self, request, request_ref, context, scientific, v3, inputs):
        self.request_ref, self.context, self.v3 = request_ref, context, v3
        self.prior = read(request["prior_accounting"])
        p = self.prior
        if (p.get("schema") != "gnnm-member-repeat-current-root-accounting-v1" or p.get("qualified") is not True
                or p.get("includes_all_prior_successes_failures_preparation_and_current_upload") is not True
                or p.get("includes_all_v3_stages_and_failed_primary") is not True
                or p.get("current_repeat_attempt_already_charged") is not False
                or p.get("retained_v3_prior") != inputs["retained_v3_prior"]
                or p.get("v3_stage_charges") != inputs["v3_stage_charges"]):
            raise ValueError("Fresh root accounting retaining v1/v2/v3 failures and all v3 stages required")
        keys = ("combined_phase_seconds", "combined_phase_allocated_disk_bytes", "diagnostic_seconds", "diagnostic_allocated_bytes")
        if any(type(p.get(k)) not in (int,float) or not math.isfinite(p[k]) or p[k] < 0 for k in keys):
            raise ValueError("Finite nonnegative current accounting required")
        old = read(inputs["retained_v3_prior"])
        for terminal in inputs["retained_v1_v2_terminals"]:
            read(terminal)
        seconds, stored = 0., 0
        for charge in inputs["v3_stage_charges"]:
            terminal = read(charge["terminal"])
            start_path = checked(charge["start"])
            start = json.loads(start_path.read_text())
            supervisor = confined(PHASE/charge["supervisor"])
            coordinator = read(charge["coordinator"])
            expected_exit = 1 if charge["stage"] == "primary48" else 0
            if (terminal.get("START_sha256") != charge["start"]["sha256"]
                    or terminal.get("root_request_unchanged") is not True or start.get("root_request") != inputs["v3_request"]
                    or terminal.get("child_exit_code") != expected_exit or terminal.get("complete") is not (expected_exit==0)
                    or terminal.get("timed_out") is not False or terminal.get("within_whole_cap") is not True
                    or coordinator.get("stage") != charge["stage"] or coordinator.get("request_sha256") != inputs["v3_request"]["sha256"]
                    or coordinator.get("request_unchanged") is not True or coordinator.get("returncode") != expected_exit):
                raise ValueError("Retained v3 terminal/coordinator identity or outcome changed")
            scientific.verify_outer_start(start_path, checked(inputs["v3_request"]), supervisor, start["child_argv"])
            if terminal.get("stdout_stderr_sha256") != sha(start_path.parent/"stdout_stderr.log"):
                raise ValueError("Retained v3 outer stdout changed")
            for name, expected in terminal["inner_evidence"].items():
                file = confined(supervisor/name)
                if expected != dict(bytes=file.stat().st_size, sha256=sha(file)):
                    raise ValueError("Retained v3 inner evidence changed")
            if expected_exit==0:
                scientific.completed(supervisor, charge["start"])
            values = (terminal.get("whole_supervised_seconds"), coordinator.get("seconds"))
            if any(type(s) not in (int,float) or not math.isfinite(s) or s < 0 for s in values):
                raise ValueError("Complete prior wall charges required")
            seconds += max(values)
            stored += sum(v3.allocated(PHASE/charge[key]) for key in ("output", "supervisor", "outer"))
        if (p["combined_phase_seconds"] < old["combined_phase_seconds"]+seconds
                or p["diagnostic_seconds"] < old["diagnostic_seconds"]+seconds
                or p["combined_phase_allocated_disk_bytes"] < old["combined_phase_allocated_disk_bytes"]+stored
                or p["diagnostic_allocated_bytes"] < old["diagnostic_allocated_bytes"]+stored):
            raise ValueError("Fresh accounting omitted retained complete v3 charges")
        if p["combined_phase_seconds"]+120 > 21600 or p["diagnostic_seconds"]+120 > 2400:
            raise ValueError("Full 120-second envelope does not fit existing remaining budgets")
        outer = read(context["outer_start"])
        if outer["whole_cap_seconds"] != 120 or outer["termination_grace_seconds"] != 5:
            raise ValueError("Exact 120-second child and five-second grace required")
        age = max(0., (datetime.now(timezone.utc)-datetime.fromisoformat(outer["start_UTC"])).total_seconds())
        self.start = time.monotonic()-age
        if (PHASE/stage_paths()["outer"]/"TERMINAL.json").exists():
            raise ValueError("Terminal attempt refuses retry")
        self.check()

    def check(self, new_bytes=0):
        elapsed = time.monotonic()-self.start
        p = self.prior
        if elapsed >= 115 or p["combined_phase_seconds"]+elapsed >= 21600 or p["diagnostic_seconds"]+elapsed >= 2400:
            raise RuntimeError("Unchanged child/phase/diagnostic wall cap exhausted")
        stored = self.v3.allocated(HERE/"runs")
        if (p["combined_phase_allocated_disk_bytes"]+stored+new_bytes+2**20 > 200*2**30
                or p["diagnostic_allocated_bytes"]+stored+new_bytes+2**20 > 64*2**20):
            raise RuntimeError("Unchanged phase/diagnostic artifact cap exhausted")
        checked(self.request_ref)

    def write(self, path, value):
        path = confined(path)
        if not path.is_relative_to(HERE):
            raise ValueError("New numerical output outside this audit packet")
        raw = (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
        self.check(len(raw)+4096)
        with path.open("xb") as stream:
            stream.write(raw)


def scale(torch, values):
    count, square, maximum = values.numel(), 0., 0.
    for block in values.split(8192):
        v = block.to(dtype=torch.float64)
        if not bool(torch.isfinite(v).all()):
            raise ValueError("Nonfinite official-training logits")
        square += float(v.square().sum())
        maximum = max(maximum, float(v.abs().max()))
    return dict(elements=count, maxabs=maximum, rms=math.sqrt(square/count))


def contrast(torch, left, right):
    if left.shape != right.shape or left.dtype != torch.float32 or right.dtype != torch.float32:
        raise ValueError("Identical complete training-logit shapes and original float32 required")
    count, square, maximum, above = left.numel(), 0., 0., 0
    for a,b in zip(left.split(8192),right.split(8192)):
        # Preserve original helper float32 subtraction and its absolute threshold.
        delta = (a-b).abs()
        if not bool(torch.isfinite(delta).all()):
            raise ValueError("Nonfinite repeat differences")
        maximum = max(maximum,float(delta.max()))
        above += int((delta > 1e-6).sum())
        square += float(delta.to(dtype=torch.float64).square().sum())
    l,r = scale(torch,left),scale(torch,right)
    rms = math.sqrt(square/count)
    ms,rs = max(l["maxabs"],r["maxabs"]),max(l["rms"],r["rms"])
    return dict(elements=count, maxabs_error=maximum, rms_error=rms, above_original_atol_count=above,
                above_original_atol_fraction=above/count, original_serial_repeat_atol=1e-6,
                original_absolute_repeat_check_passed=maximum<=1e-6,
                left_scale=l,right_scale=r,maxabs_error_over_pair_maxabs=(maximum/ms if ms else None),
                rms_error_over_pair_rms=(rms/rs if rs else None),zero_scale_ratio_policy="undefined/null")


def invariance(torch, v3, before, model):
    after = v3.snapshot(torch,model)
    result = dict(parameter_and_buffer_state_unchanged=after["parameters"]==before["parameters"],
                  dropout_probability_unchanged=after["p"]==before["p"],module_modes_unchanged=after["modes"]==before["modes"],
                  Torch_CPU_RNG_unchanged=torch.equal(after["cpu_rng"],before["cpu_rng"]),
                  Torch_all_CUDA_RNG_unchanged=len(after["cuda_rng"])==len(before["cuda_rng"]) and all(torch.equal(a,b) for a,b in zip(after["cuda_rng"],before["cuda_rng"])),
                  all_parameter_grad_fields_remain_None=all(p.grad is None for p in model.parameters()))
    result["all_checked_invariants_hold"] = all(result.values())
    result["parameter_and_buffer_fingerprint"] = after["parameters"]
    return result


def numerical(torch, admitted, inputs, v3, budget, output):
    from shared_message_predictive_v3.core import load_inputs, fingerprint
    from shared_message_quality_v3.data import load_family
    from shared_message_quality_v3 import models,counts
    row = inputs["state"]
    original_inputs = read(inputs["original_input_table"])
    if (original_inputs["states"][0] != row or original_inputs["binding"] != inputs["binding"]
            or original_inputs["tasks"]["Reddit2"] != inputs["task_bindings"]
            or inputs["task_bindings"] != {key:admitted["tasks"]["Reddit2"][key] for key in
                ("acquisition","engineering_phase","prepared_manifest","parameter_control_report")}
            or read(inputs["failed_primary"])["message"] != "Serial repeats do not match the pooled-cotangent forward"):
        raise ValueError("Original failed first-state and complete-data bindings differ")
    fit = read(row["source_fit_end"])
    if (row["cell"] != CELL or row["role"] != "INITIAL" or row["execution"]["checkpoint_blocks"] is not False
            or fit.get("status") != "completed" or fit.get("cell") != CELL or fit.get("epochs") != 300
            or fit.get("initial") != row["checkpoint"] or fit.get("execution") != row["execution"]
            or admitted["execution_policy"]["Reddit2"]["exact_member"] != row["execution"]):
        raise ValueError("Only exact failed INITIAL state and original noncheckpointed serial execution admitted")
    export,lineage,metadata,directory = load_inputs(admitted,"Reddit2")
    if lineage.get("source_export_contains_only_training_targets") is not True or lineage.get("heldout_target_values_loaded") is not False:
        raise ValueError("Original training-only acquisition lineage required")
    family = load_family(metadata,directory,"exact_member",device="cuda:0")
    x,ids = export["x"].to("cuda:0"),export["training_target_node_ids"].to("cuda:0")
    del export  # Training labels may be deserialized by loader; never selected or used here.
    if x.shape != (counts.TASKS["Reddit2"]["nodes"],counts.TASKS["Reddit2"]["features"]) or ids.ndim!=1 or not ids.numel():
        raise ValueError("Complete official graph/features and training IDs required")
    payload = torch.load(checked(row["checkpoint"]),map_location="cpu",weights_only=False)
    if (payload.get("schema") != "gnnm-shared-message-checkpoint-v2" or payload.get("cell") != CELL
            or payload.get("epoch") != 0 or payload.get("binding") != inputs["binding"]
            or payload.get("input_lineage") != lineage or payload.get("execution") != row["execution"]):
        raise ValueError("Exact existing INITIAL checkpoint provenance required")
    model = models.construct("Reddit2","exact_member",seed=17,checkpoint_blocks=False).to("cuda:0")
    model.load_state_dict(payload["model_state"],strict=True)
    del payload
    state_hash = fingerprint({name:v.detach().cpu().contiguous() for name,v in model.state_dict().items()})
    if (state_hash != row["initial_parameter_fingerprint"] or model.members != 4 or model.checkpoint_blocks
            or sum(p.numel() for p in model.parameters()) != counts.plan("Reddit2")["parameter_counts"]["exact_member"]):
        raise ValueError("Original source state, member count or parameters differ")
    preparation_memory = v3.memory(torch)
    torch.cuda.reset_peak_memory_stats(0)
    original = v3.snapshot(torch,model)
    saved_modes,saved_p = [(m,m.training) for m in model.modules()],model.p
    logits,forward_rows,pair_rows = {},[],[]
    started = time.perf_counter()
    try:
        model.train(True); model.p = 0.
        baseline = v3.snapshot(torch,model)
        for pass_name,grad in PASSES:
            logits[pass_name] = []
            for member in range(4):
                budget.check()
                s,d = len(family.ledger.sparse),len(family.ledger.dense)
                torch.cuda.synchronize(0); tick=time.perf_counter()
                with family.ledger.scope("repeat_"+pass_name+"_member%d"%member),torch.set_grad_enabled(grad):
                    full = model.forward_member(family,x,member,colors=(0,2),dropout_key=None)
                    actual_mode = dict(grad_enabled=torch.is_grad_enabled(),training=model.training,p=model.p,
                                       checkpoint_blocks=model.checkpoint_blocks,output_requires_grad=full.requires_grad,
                                       output_grad_fn_type=(type(full.grad_fn).__name__ if full.grad_fn is not None else None))
                    values = full[ids].detach().to("cpu").contiguous()
                    del full
                torch.cuda.synchronize(0)
                elapsed=time.perf_counter()-tick
                invariant = invariance(torch,v3,baseline,model)
                sparse,dense = family.ledger.sparse[s:],family.ledger.dense[d:]
                measured = dict(pass_name=pass_name,member=member,training_logit_shape=list(values.shape),dtype=str(values.dtype),
                                mode=actual_mode,invariants=invariant,scale=scale(torch,values),forward_seconds=elapsed,
                                sparse_ledger_rows=sparse,dense_ledger_rows=dense,sparse_calls=len(sparse),dense_calls=len(dense),memory=v3.memory(torch))
                budget.write(output/("FORWARD_%02d_%s_member%d.json"%(len(forward_rows),pass_name,member)),measured)
                forward_rows.append(measured)
                if (not invariant["all_checked_invariants_hold"] or actual_mode["grad_enabled"] is not grad
                        or actual_mode["output_requires_grad"] is not grad or len(sparse)!=2 or len(dense)!=2
                        or any(r["direction"]!="forward" for r in sparse)
                        or any(r["scope"].startswith("checkpoint_") for r in sparse)):
                    raise ValueError("State/RNG/.grad invariance or exact serial forward execution changed")
                logits[pass_name].append(values)
                print(json.dumps(dict(pass_name=pass_name,member=member,complete_forward_calls=len(forward_rows))),flush=True)
        for left,right in itertools.combinations([n for n,g in PASSES],2):
            budget.check()
            members=[dict(member=m,**contrast(torch,logits[left][m],logits[right][m])) for m in range(4)]
            count=sum(r["elements"] for r in members)
            above=sum(r["above_original_atol_count"] for r in members)
            pair_max=max(r["maxabs_error"] for r in members)
            pair_rms=math.sqrt(sum(r["elements"]*r["rms_error"]**2 for r in members)/count)
            pair_scale_max=max(max(r["left_scale"]["maxabs"],r["right_scale"]["maxabs"]) for r in members)
            pair_scale_rms=max(math.sqrt(sum(r["elements"]*r[side]["rms"]**2 for r in members)/count)
                               for side in ("left_scale","right_scale"))
            pair=dict(left=left,right=right,comparison="same-member repeat across two executions; no cross-member prediction comparison",members=members,
                      aggregate=dict(elements=count,maxabs_error=pair_max,rms_error=pair_rms,
                                     pair_scale_maxabs=pair_scale_max,pair_scale_rms=pair_scale_rms,
                                     maxabs_error_over_pair_maxabs=(pair_max/pair_scale_max if pair_scale_max else None),
                                     rms_error_over_pair_rms=(pair_rms/pair_scale_rms if pair_scale_rms else None),
                                     above_original_atol_count=above,above_original_atol_fraction=above/count))
            budget.write(output/("PAIR_%02d_%s_vs_%s.json"%(len(pair_rows),left,right)),pair)
            pair_rows.append(pair)
    finally:
        model.p=saved_p
        for module,flag in saved_modes:
            module.training=flag
    restored=invariance(torch,v3,original,model)
    if not restored["all_checked_invariants_hold"]:
        raise ValueError("Restoring original model p/modes/state/RNG/.grad failed")
    result=dict(cell=CELL,role="INITIAL",checkpoint=row["checkpoint"],model_parameter_fingerprint=state_hash,
                input_lineage=lineage,execution=row["execution"],graph_nodes=family.full.nodes,training_nodes=ids.numel(),
                actual_dtype=str(x.dtype),training_targets_used=False,forward_calls=len(forward_rows),vjp_calls=0,
                sparse_ledger_summary=family.ledger.summary(),preparation_memory=preparation_memory,memory=v3.memory(torch),
                state_restoration=restored,forwards=forward_rows,pairs=pair_rows,
                numerical_repeat_seconds_including_reductions=time.perf_counter()-started,
                cpu_peak_RSS_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                original_threshold_preserved=True,any_pair_exceeds_original_threshold=any(m["above_original_atol_count"] for p in pair_rows for m in p["members"]),
                no_numerical_policy_amendment_or_scientific_conclusion=True)
    del logits,model,family,x,ids,baseline,original
    v3.release(torch)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request",type=Path,required=True)
    parser.add_argument("--supervisor",type=Path,required=True)
    args=parser.parse_args()
    request_ref=ref(args.request); request=read(request_ref)
    if (request.get("schema")!="gnnm-member-repeat-root-request-v1" or request.get("root_authorized") is not True
            or request.get("completed_development_family_qualified") is not True or request.get("stage")!=stage_paths()):
        raise ValueError("Separate root admission of this exact numerical audit required")
    guard,policy,v3,inputs,seal=bootstrap(request)
    approval=read(request["root_approval"])
    payload_hash=hashlib.sha256(json.dumps({k:v for k,v in request.items() if k!="root_approval"},sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if approval.get("qualified") is not True or approval.get("approved_payload_sha256")!=payload_hash:
        raise ValueError("Root approval does not bind exact request")
    from shared_message_predictive_v3 import contract as scientific
    from shared_message_predictive_v3.core import configure
    admitted=read(inputs["binding"])
    if admitted["source_manifest_sha256"]!=SCIENTIFIC_SHA or request["gpu_uuid"]!=admitted["gpu_uuid"]:
        raise ValueError("Original source/runtime/GPU binding changed")
    scientific.runtime_before_import(admitted)
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise ValueError("Original sole GPU visibility required")
    context=scientific.live(args.supervisor,Path(__file__),args.request,
                            dict(gpu_uuid=admitted["gpu_uuid"],runtime_policy=admitted["runtime_policy"],per_child_wall_cap_seconds=120))
    if confined(args.supervisor)!=PHASE/stage_paths()["supervisor"]:
        raise ValueError("Exact new supervisor directory required")
    budget=Budget(request,request_ref,context,scientific,v3,inputs)
    output=confined(PHASE/stage_paths()["output"]); output.mkdir(parents=True,exist_ok=False)
    budget.write(output/"START.json",dict(request=request_ref,live=context,training_authorized=False,
                 numerical_repeat_audit_authorized=True,source_seal=request["source_seal"],existing_evidence=seal["existing_evidence"]))
    try:
        import torch
        gpu=configure(admitted)
        measured=numerical(torch,admitted,inputs,v3,budget,output)
        modules=guard.imported_modules(policy); guard.verify_imported_receipts(modules,policy,require_generated=True)
        budget.write(output/"RESULT.json",dict(schema="gnnm-member-repeat-numerical-result-v1",audit_completed=True,
                 scientific_positive_evidence=False,request=request_ref,live=context,gpu=gpu,source_seal=request["source_seal"],
                 imported_phase_modules=modules,runtime_policy=admitted["runtime_policy"],measurements=measured,
                 precision=dict(default_dtype=str(torch.get_default_dtype()),float32_matmul_precision=torch.get_float32_matmul_precision(),
                                CUDA_matmul_TF32=torch.backends.cuda.matmul.allow_tf32,cudnn_TF32=torch.backends.cudnn.allow_tf32,
                                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),CUBLAS_WORKSPACE_CONFIG=os.environ.get("CUBLAS_WORKSPACE_CONFIG")),
                 source_or_helper_changed=False,original_threshold_relaxed=False,logits_or_gradients_serialized=False,
                 authoritative_wall_charge="complete bounded outer terminal and coordinator, including failures and teardown"))
    except BaseException as error:
        failure=dict(type=type(error).__name__,message=str(error),request=request_ref,live=context,
                     partial_scalar_outputs_preserved=True,automatic_retry_authorized=False,
                     original_threshold_relaxed=False,scientific_positive_evidence=False)
        if "torch" in locals() and torch.cuda.is_initialized():
            failure.update(failed_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),failed_peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        with (output/"FAILURE.json").open("x") as stream:
            json.dump(failure,stream,indent=2,allow_nan=False);stream.write("\n")
        raise


if __name__=="__main__":
    main()
