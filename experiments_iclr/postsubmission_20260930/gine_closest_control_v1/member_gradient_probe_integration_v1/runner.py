"""Training-only fixed-state gradient companion. Source preparation is not admission.

Three separately supervised stages: small numerical qualification, complete
Products memory gate, and the complete 48-state dropout-free primary.
No optimizer, fit, validation/test targets, predictions, or gradient exports.
"""
import sys
import os
sys.path[:] = [p for p in sys.path if p and os.path.isabs(p)
               and any(os.path.commonpath([p, root]) == root for root in (sys.prefix, sys.base_prefix))]
sys.dont_write_bytecode = True
import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
PACKAGE = str(HERE.relative_to(PHASE))
SCIENTIFIC_SHA = "e4d1b4b7d96ae07e7588a854cb8a0d70e9af6612ac72d1df583c3c637c376c9b"
STAGES = ("qualification", "products_memory", "primary48")
ARMS = ("exact_member", "shared_mean_message", "single_base64", "single_parameters")
TASKS = ("Reddit2", "ogbn-products")
SEEDS = (17, 29, 43)
ROLES = ("INITIAL", "FINAL")
FIXED = dict(child_wall_cap_seconds=600, termination_grace_seconds=5,
             diagnostic_wall_cap_seconds=2400, diagnostic_artifact_cap_bytes=64 * 2**20,
             combined_phase_wall_cap_seconds=21600, combined_phase_disk_cap_bytes=200 * 2**30,
             reserved_GPU_memory_cap_bytes=60 * 2**30, secondary_enabled=False,
             numeric_atol=1e-5, numeric_rtol=1e-5, serial_repeat_atol=1e-6,
             strict_training_target_only=True, optimizer_steps=0,
             original_score_recalculation=False,
             same_weight_same_forward_adjoint_control_measured=False,
             actual_AdamW_directional_response_measured=False,
             peer_feedback_harm_identified=False)


def confined(path):
    path = Path(path).absolute()
    if not path.is_relative_to(PHASE):
        raise ValueError("Outside the authorized phase")
    part = PHASE
    for name in path.relative_to(PHASE).parts:
        part /= name
        if part.is_symlink():
            raise ValueError("Phase symlink refused")
    return path.resolve()


def sha(path):
    h = hashlib.sha256()
    with confined(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def checked(ref):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("Canonical path/hash reference required")
    p = confined(PHASE / ref["path"])
    if str(p.relative_to(PHASE)) != ref["path"] or sha(p) != ref["sha256"]:
        raise ValueError("Input identity changed: " + str(ref["path"]))
    return p


def ref(path):
    path = confined(path)
    return dict(path=str(path.relative_to(PHASE)), sha256=sha(path))


def json_read(reference):
    return json.loads(checked(reference).read_text())


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), module.__dict__)
    return module


def bootstrap(request):
    seal = json_read(request["source_seal"])
    if seal.get("schema") != "gnnm-member-gradient-companion-source-v1" or seal.get("execution_authorized") is not False:
        raise ValueError("Separate source-only seal required")
    if request["fixed_policy"] != FIXED or request["scientific_manifest_sha256"] != SCIENTIFIC_SHA:
        raise ValueError("Declared source or fixed policy changed")
    own = seal["importable_sources"]
    if set(own) != {PACKAGE + "/runner.py", "gine_closest_control_v1/member_gradient_probe_v1/probe.py"}:
        raise ValueError("Only one caller and the existing helper are importable")
    for path, wanted in own.items():
        checked(dict(path=path, sha256=wanted))
    if own[PACKAGE + "/runner.py"] != sha(Path(__file__)):
        raise ValueError("Calling source changed")
    scientific = PHASE / "shared_message_predictive_v3"
    raw_manifest = checked(dict(path="shared_message_predictive_v3/SOURCE_MANIFEST_v1.json", sha256=SCIENTIFIC_SHA))
    row = next(r for r in json.loads(raw_manifest.read_text())["files"] if r["relative_path"] == "import_guard.py")
    guard = load_source("_member_gradient_inherited_guard_v1", checked(dict(path="shared_message_predictive_v3/import_guard.py", sha256=row["sha256"])))
    policy, tree = guard.verify_packet(SCIENTIFIC_SHA)
    # An explicit separate caller policy adds just these two exact sources.
    # The inherited files and package roots remain immutable.
    policy = copy.deepcopy(policy)
    policy["importable_sha256"].update(own)
    guard.inspect_tree(HERE, [PACKAGE + "/runner.py"], policy["package_directories"])
    helper_path = PHASE / "gine_closest_control_v1/member_gradient_probe_v1/probe.py"
    guard.inspect_tree(helper_path.parent, [str(helper_path.relative_to(PHASE))], policy["package_directories"])
    guard.installed_path_only()
    guard.verify_generator_installation(policy)
    guard.imported_modules(policy)
    sys.meta_path.insert(0, guard.PhaseFinder(policy))
    helper = load_source("_member_gradient_helper_v1", helper_path)
    inputs = json_read(seal["input_identities"])
    if inputs.get("schema") != "gnnm-member-gradient-existing-input-identities-v1":
        raise ValueError("Existing input table required")
    return guard, policy, helper, inputs, seal


def expected_stages():
    return [dict(stage=s, output=PACKAGE + "/runs/authorized1gpu_run01/" + s,
                 supervisor="protocols/logged_runs/member_gradient_" + s + "_authorized1gpu_run01",
                 outer="protocols/bounded_runs/member_gradient_" + s + "_authorized1gpu_run01") for s in STAGES]


def allocated(root):
    if not root.exists():
        return 0
    rows = list(root.rglob("*"))
    if any(p.is_symlink() for p in rows):
        raise ValueError("Output evidence symlink refused")
    return sum(p.stat().st_blocks * 512 for p in rows if p.is_file())


class Budget:
    def __init__(self, request, request_ref, stage, context, scientific):
        self.request, self.request_ref, self.scientific = request, request_ref, scientific
        self.prior = json_read(request["prior_accounting"])
        p = self.prior
        if (p.get("schema") != "gnnm-member-gradient-current-root-accounting-v1" or p.get("qualified") is not True
                or p.get("includes_all_prior_successes_failures_preparation_and_current_upload") is not True
                or p.get("declared_stages_already_charged") is not False):
            raise ValueError("Current root accounting, including failed work, required")
        keys = ("combined_phase_seconds", "combined_phase_allocated_disk_bytes", "diagnostic_seconds", "diagnostic_allocated_bytes")
        if any(type(p.get(k)) not in (int, float) or not math.isfinite(p[k]) or p[k] < 0 for k in keys):
            raise ValueError("Finite nonnegative current accounting required")
        self.previous, self.previous_seconds = [], 0.
        index = STAGES.index(stage)
        for i, row in enumerate(expected_stages()):
            out = PHASE / row["outer"]
            if i < index:
                start = json.loads((out / "START.json").read_text())
                terminal = json.loads((out / "TERMINAL.json").read_text())
                seconds = terminal.get("whole_supervised_seconds")
                if (terminal.get("START_sha256") != sha(out / "START.json") or start.get("root_request") != request_ref
                        or terminal.get("root_request_unchanged") is not True
                        or terminal.get("stdout_stderr_sha256") != sha(out / "stdout_stderr.log")
                        or start.get("whole_cap_seconds") != 600 or start.get("termination_grace_seconds") != 5
                        or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0):
                    raise ValueError("Prior stage terminal provenance invalid")
                self.previous_seconds += seconds  # Actual failed overrun remains chargeable.
                self.previous.append(dict(stage=row["stage"], seconds=seconds, terminal=ref(out / "TERMINAL.json")))
                if terminal.get("complete") is not True or terminal.get("child_exit_code") != 0:
                    raise RuntimeError("Prior failure charged; no continuation or retry admitted")
                scientific.completed(PHASE / row["supervisor"], ref(out / "START.json"))
                result = json.loads((PHASE / row["output"] / "RESULT.json").read_text())
                if result.get("qualified") is not True or result.get("request") != request_ref:
                    raise ValueError("Prior stage did not qualify this request")
            elif i > index and any((PHASE / row[k]).exists() for k in ("outer", "supervisor", "output")):
                raise ValueError("Stage order changed")
            elif i == index and (out / "TERMINAL.json").exists():
                raise ValueError("Terminal current attempt refuses retry")
        remaining = (len(STAGES) - index) * 600
        if (p["combined_phase_seconds"] + self.previous_seconds + remaining > 21600
                or p["diagnostic_seconds"] + self.previous_seconds + remaining > 2400):
            raise ValueError("Full remaining stage caps do not fit existing phase limits")
        outer = json_read(context["outer_start"])
        self.start = time.monotonic() - max(0., (datetime.now(timezone.utc) - datetime.fromisoformat(outer["start_UTC"])).total_seconds())
        self.context = context
        self.check()

    def check(self, new_bytes=0):
        p = self.prior
        elapsed = time.monotonic() - self.start
        if (elapsed >= 595 or p["combined_phase_seconds"] + self.previous_seconds + elapsed >= 21600
                or p["diagnostic_seconds"] + self.previous_seconds + elapsed >= 2400):
            raise RuntimeError("Declared child/phase wall cap exhausted")
        total = sum(allocated(PHASE / row[k]) for row in expected_stages() for k in ("output", "outer", "supervisor"))
        if (p["combined_phase_allocated_disk_bytes"] + total + new_bytes + 2**20 > 200 * 2**30
                or p["diagnostic_allocated_bytes"] + total + new_bytes + 2**20 > 64 * 2**20):
            raise RuntimeError("Declared disk cap exhausted")
        checked(self.request_ref)

    def write(self, path, value):
        raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        self.check(len(raw) + 4096)
        with confined(path).open("xb") as stream:
            stream.write(raw)


def memory(torch):
    row = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(0), peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
               live_allocated_bytes=torch.cuda.memory_allocated(0), live_reserved_bytes=torch.cuda.memory_reserved(0))
    if row["peak_reserved_bytes"] > 60 * 2**30 or row["live_reserved_bytes"] > 60 * 2**30:
        raise RuntimeError("Existing 60 GiB reserved GPU memory cap exceeded")
    return row


def release(torch):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)


def close_values(torch, actual, expected):
    a, b = torch.as_tensor(actual, dtype=torch.float64), torch.as_tensor(expected, dtype=torch.float64)
    if a.shape != b.shape or not torch.allclose(a, b, atol=1e-5, rtol=1e-5):
        raise ValueError("Numerical qualification differs")
    return float((a - b).abs().max()) if a.numel() else 0.


def snapshot(torch, model):
    from shared_message_predictive_v3.core import fingerprint
    return dict(parameters=fingerprint(model.state_dict()), p=model.p,
                modes=[m.training for m in model.modules()], cpu_rng=torch.get_rng_state().clone(),
                cuda_rng=[x.clone() for x in torch.cuda.get_rng_state_all()])


def unchanged(torch, before, model):
    after = snapshot(torch, model)
    if (after["parameters"] != before["parameters"] or after["p"] != before["p"] or after["modes"] != before["modes"]
            or not torch.equal(after["cpu_rng"], before["cpu_rng"])
            or any(not torch.equal(a, b) for a, b in zip(after["cuda_rng"], before["cuda_rng"]))
            or any(p.grad is not None for p in model.parameters())):
        raise ValueError("Helper altered state, mode, RNG, or .grad")


def qualification(torch, helper, budget):
    from torch.nn import functional as F
    from propagation_cost_engineering_v1.complete_models import CompleteCSR, CompleteFamily, OperationLedger
    from propagation_cost_impl_v1.dropout import DropoutKey
    from shared_message_quality_v3 import models
    n, d, c = 11, 5, 3
    columns = [j for i in range(n) for j in sorted(((i - 1) % n, (i + 1) % n))]
    csr = torch.sparse_csr_tensor(torch.arange(0, 2*n + 1, 2, device="cuda:0"),
        torch.tensor(columns, device="cuda:0"), torch.full((2*n,), .5, device="cuda:0"), size=(n,n))
    ledger = OperationLedger()
    metadata = dict(nodes=n, canonical_edges=2*n, P_edge_only_fingerprint="qualification-ring-only")
    family = CompleteFamily({"full": CompleteCSR("full", csr, csr, ledger, n)}, metadata, ledger, "cuda:0")
    x = (torch.arange(n*d, device="cuda:0", dtype=torch.float32).reshape(n,d) % 13 - 6) / 7
    ids = torch.tensor([0, 2, 3, 6, 8, 10], device="cuda:0")
    y = torch.tensor([0, 1, 2, 0, 2, 1], device="cuda:0")
    rows = []
    for internal in ("exact_member", "shared_mean", "single_D64", "single_matched"):
        budget.check()
        print(json.dumps(dict(stage="qualification",internal_arm=internal,check="checkpointed_p0_forward")),flush=True)
        model = models.construct(dict(features=d, classes=c), internal, seed=17, checkpoint_blocks=True).to("cuda:0")
        original_p = model.p
        model.train(); model.p = 0.
        ledger.sparse.clear(); ledger.dense.clear()
        with torch.enable_grad():
            checkpointed = model(family, x, colors=(0,2)).detach()
        if not any(r["scope"] == "checkpoint_forward" for r in ledger.sparse):
            raise ValueError("Qualification did not actually enter source checkpointing")
        model.eval()
        with torch.no_grad():
            ordinary = model(family, x, colors=(0,2))
        forward_error = close_values(torch, checkpointed, ordinary)
        del checkpointed, ordinary
        model.p = original_p; model.train()
        architecture = "single" if internal.startswith("single_") else internal
        for disabled in (True, False):
            print(json.dumps(dict(stage="qualification",internal_arm=internal,dropout_disabled=disabled,
                                  check="native_vjp_helper_geometry_and_restoration")),flush=True)
            key = None if disabled else DropoutKey(0,17,1)
            model.p = 0. if disabled else original_p
            named = list(model.named_parameters()); names, parameters = zip(*named)
            output = model(family, x, colors=(0,2), dropout_key=key)[:, ids]
            gradients = []
            for member in output:
                raw = torch.autograd.grad(F.cross_entropy(member,y), parameters, retain_graph=True, allow_unused=True)
                gradients.append({name: (torch.zeros_like(p, device="cpu", dtype=torch.float64) if g is None
                    else g.detach().to(device="cpu", dtype=torch.float64)) for (name,p),g in zip(named,raw)})
            native = torch.autograd.grad(F.cross_entropy(output.mean(0),y), parameters, allow_unused=True)
            pooled = {name: (torch.zeros_like(p, device="cpu", dtype=torch.float64) if g is None
                else g.detach().to(device="cpu", dtype=torch.float64)) for (name,p),g in zip(named,native)}
            serial_error = None
            if internal == "exact_member":
                with torch.no_grad():
                    q = torch.softmax(output.mean(0),dim=-1)
                    q[torch.arange(ids.numel(),device="cuda:0"),y] -= 1.
                    q /= 4*ids.numel()
                serial = {name: torch.zeros_like(p, device="cpu", dtype=torch.float64) for name,p in named}
                for m in range(4):
                    values = model.forward_member(family,x,m,colors=(0,2),dropout_key=key)[ids]
                    raw = torch.autograd.grad(values, parameters, grad_outputs=q, allow_unused=True)
                    for (name,p),g in zip(named,raw):
                        if g is not None:
                            serial[name] += g.detach().to(device="cpu",dtype=torch.float64)
                    del values, raw
                serial_error = max(close_values(torch, serial[name], pooled[name]) for name in names)
                del q, serial
            del output, native
            groups = {k: sorted(v) for k,v in helper._groups(set(names), architecture == "single").items()}
            reference = helper._summary(torch, gradients, pooled, groups)
            del gradients, pooled
            model.p = original_p
            # Deliberately mixed module flags verify exact restoration.
            model.eval(); next(iter(model.blocks)).train(True)
            before = snapshot(torch, model)
            measured = helper.probe(model,family,x,ids,y,architecture=architecture,
                                    dropout_disabled=disabled,dropout_key=key)
            unchanged(torch,before,model)
            if not any(r["scope"] == "checkpoint_recompute" for r in measured["sparse_ledger_rows"]):
                raise ValueError("Helper qualification did not retain checkpoint recomputation")
            geometry_error = 0.
            for block in groups:
                for field in ("member_gradient_gram", "pooled_gradient_norm_squared", "own_dot_pooled", "jensen_extra_dot_pooled"):
                    geometry_error = max(geometry_error,close_values(torch,measured["shared_blocks"][block][field],reference[block][field]))
            model.train(); model.p = original_p
            with torch.no_grad():
                first = model(family,x,colors=(0,2),dropout_key=DropoutKey(0,17,1))
                second = model(family,x,colors=(0,2),dropout_key=DropoutKey(0,17,1))
            if not torch.equal(first,second):
                raise ValueError("Repeated fixed-key forwards differ")
            rows.append(dict(internal_arm=internal, dropout_disabled=disabled,
                checkpointed_p0_vs_eval_max_abs_error=forward_error,
                direct_serial_pooled_vjp_vs_native_max_abs_error=serial_error,
                helper_vs_native_geometry_max_abs_error=geometry_error,
                fixed_key_repeat_exact=True, state_modes_rng_and_grad_unchanged=True))
            del first,second,measured,reference,before
            ledger.sparse.clear(); ledger.dense.clear()
        del model, named, parameters
        release(torch)
    return dict(checks=rows, fixture_nodes=n, fixture_features=d, fixture_classes=c,
                fixture_is_numerical_only=True, actual_dataset_loaded=False,
                direct_serial_vjp_qualified_for_both_p0_and_keyed_dropout=True)


def load_task(admitted, task):
    import torch
    from shared_message_predictive_v3.core import load_inputs
    from shared_message_quality_v3.data import load_family
    export, lineage, metadata, directory = load_inputs(admitted, task)
    family = load_family(metadata,directory,"exact_member",device="cuda:0")
    x = export["x"].to("cuda:0")
    ids = export["training_target_node_ids"].to("cuda:0")
    y = export["training_targets"].reshape(-1).to(dtype=torch.long,device="cuda:0")
    if lineage.get("source_export_contains_only_training_targets") is not True or lineage.get("heldout_target_values_loaded") is not False:
        raise ValueError("Training-only exported target lineage required")
    return family,x,ids,y,lineage


def measure_state(torch, helper, admitted, binding_ref, row, data, budget):
    from shared_message_quality_v3 import models,counts
    from shared_message_predictive_v3.core import fingerprint
    family,x,ids,y,lineage = data
    budget.check()
    fit_end = json_read(row["source_fit_end"])
    if (fit_end.get("status") != "completed" or fit_end.get("cell") != row["cell"]
            or fit_end.get("epochs") != 300 or fit_end.get("execution") != row["execution"]
            or fit_end.get("initial" if row["role"] == "INITIAL" else "final_checkpoint") != row["checkpoint"]):
        raise ValueError("State reference differs from its completed fit metadata")
    path = checked(row["checkpoint"])
    payload = torch.load(path,map_location="cpu",weights_only=False)
    if (payload.get("schema") != "gnnm-shared-message-checkpoint-v2" or payload.get("cell") != row["cell"]
            or payload.get("epoch") != (0 if row["role"] == "INITIAL" else 300)
            or payload.get("binding") != binding_ref or payload.get("input_lineage") != lineage
            or payload.get("execution") != row["execution"]):
        raise ValueError("Checkpoint state identity differs from the completed cell")
    internal = admitted["model_arm_adapters"][row["cell"]["arm"]]
    model = models.construct(row["cell"]["task"],internal,seed=row["cell"]["seed"],
                             checkpoint_blocks=row["execution"]["checkpoint_blocks"]).to("cuda:0")
    model.load_state_dict(payload["model_state"],strict=True)
    state_hash = fingerprint(model.state_dict())
    if row["role"] == "INITIAL" and state_hash != row["initial_parameter_fingerprint"]:
        raise ValueError("Initial parameter identity differs")
    del payload
    if sum(p.numel() for p in model.parameters()) != counts.plan(row["cell"]["task"])["parameter_counts"][internal]:
        raise ValueError("Parameter count differs")
    before = snapshot(torch,model)
    preparation_memory = memory(torch)
    torch.cuda.reset_peak_memory_stats(0)
    result = helper.probe(model,family,x,ids,y,architecture=("single" if internal.startswith("single_") else internal))
    unchanged(torch,before,model)
    if row["execution"]["checkpoint_blocks"] and not any(r["scope"] == "checkpoint_recompute" for r in result["sparse_ledger_rows"]):
        raise ValueError("Full checkpoint-required state omitted checkpoint recomputation")
    result.update(cell=row["cell"],role=row["role"],checkpoint=row["checkpoint"],
                  model_parameter_fingerprint=state_hash,input_lineage=lineage,
                  execution=row["execution"],memory=memory(torch),
                  preparation_memory=preparation_memory,
                  cpu_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    family.ledger.sparse.clear(); family.ledger.dense.clear()
    del model,before
    release(torch)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=STAGES)
    p.add_argument("--request",type=Path,required=True)
    p.add_argument("--supervisor",type=Path,required=True)
    args = p.parse_args()
    request_ref = ref(args.request)
    request = json_read(request_ref)
    if (request.get("schema") != "gnnm-member-gradient-root-request-v1" or request.get("root_authorized") is not True
            or request.get("completed_development_family_qualified") is not True
            or request.get("stages") != expected_stages()):
        raise ValueError("Separate root admission of this exact diagnostic family required")
    guard,policy,helper,inputs,seal = bootstrap(request)
    approval = json_read(request["root_approval"])
    payload_hash = hashlib.sha256(json.dumps({k:v for k,v in request.items() if k != "root_approval"},sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if approval.get("qualified") is not True or approval.get("approved_payload_sha256") != payload_hash:
        raise ValueError("Root approval does not bind this request")
    from shared_message_predictive_v3 import contract as scientific
    from shared_message_predictive_v3.core import configure
    admitted = json_read(inputs["binding"])
    binding_ref = inputs["binding"]
    if admitted["source_manifest_sha256"] != SCIENTIFIC_SHA or request["gpu_uuid"] != admitted["gpu_uuid"]:
        raise ValueError("Immutable scientific runtime binding differs")
    if inputs["tasks"] != {task: {key: admitted["tasks"][task][key] for key in
            ("acquisition","engineering_phase","prepared_manifest","parameter_control_report")} for task in TASKS}:
        raise ValueError("Declared existing input IDs differ from the bound development inputs")
    scientific.runtime_before_import(admitted)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Declared single GPU visibility required")
    live_policy = dict(gpu_uuid=admitted["gpu_uuid"],runtime_policy=admitted["runtime_policy"],per_child_wall_cap_seconds=600)
    context = scientific.live(args.supervisor,Path(__file__),args.request,live_policy)
    selected = expected_stages()[STAGES.index(args.stage)]
    if confined(args.supervisor) != PHASE / selected["supervisor"]:
        raise ValueError("Exact stage supervisor required")
    budget = Budget(request,request_ref,args.stage,context,scientific)
    output = confined(PHASE / selected["output"])
    output.mkdir(parents=True,exist_ok=False)
    budget.write(output / "START.json",dict(request=request_ref,live=context,stage=args.stage,
                 previous_stages=budget.previous,scientific_training_authorized=False,
                 training_gradient_diagnostic_authorized=True,validation_or_test_targets_loaded=False))
    started = time.perf_counter()
    try:
        import torch
        gpu = configure(admitted)
        if args.stage == "qualification":
            measured = qualification(torch,helper,budget)
        else:
            expected = [dict(cell=dict(task=t,seed=s,arm=a),role=r) for t in TASKS for s in SEEDS for a in ARMS for r in ROLES]
            if [dict(cell=r["cell"],role=r["role"]) for r in inputs["states"]] != expected:
                raise ValueError("Exactly all48 INITIAL/FINAL states required")
            rows = inputs["states"]
            if args.stage == "products_memory":
                rows = [r for r in rows if r["cell"] == dict(task="ogbn-products",seed=17,arm="shared_mean_message") and r["role"] == "FINAL"]
                if len(rows) != 1 or rows[0]["execution"]["checkpoint_blocks"] is not True:
                    raise ValueError("Complete Products shared FINAL checkpoint gate required")
            completed, data, last_task = [], None, None
            for index,row in enumerate(rows):
                if row["execution"] != admitted["execution_policy"][row["cell"]["task"]][row["cell"]["arm"]]:
                    raise ValueError("Execution flags differ from existing source fit")
                if row["cell"]["task"] != last_task:
                    if data is not None:
                        del data; release(torch)
                    data = load_task(admitted,row["cell"]["task"])
                    last_task = row["cell"]["task"]
                value = measure_state(torch,helper,admitted,binding_ref,row,data,budget)
                file = output / ("%02d_%s_seed%d_%s_%s.json" % (index,"reddit2" if last_task == "Reddit2" else "products",row["cell"]["seed"],row["cell"]["arm"],row["role"]))
                budget.write(file,value)
                completed.append(ref(file))
                print(json.dumps(dict(stage=args.stage,completed_states=len(completed),expected_states=len(rows),peak_reserved_bytes=value["memory"]["peak_reserved_bytes"])),flush=True)
                del value
            del data; release(torch)
            measured = dict(states=completed,complete_states=len(completed),primary_complete=len(completed)==48,
                            full_products_shared_checkpoint_memory_qualified=args.stage=="products_memory")
        modules = guard.imported_modules(policy)
        guard.verify_imported_receipts(modules,policy,require_generated=True)
        budget.write(output / "RESULT.json",dict(schema="gnnm-member-gradient-stage-result-v1",qualified=True,
            stage=args.stage,request=request_ref,live=context,gpu=gpu,source_seal=request["source_seal"],
            existing_input_identities=seal["input_identities"],imported_phase_modules=modules,
            runtime_policy=admitted["runtime_policy"],training_targets_only=True,validation_or_test_targets_loaded=False,
            optimizer_steps=0,logits_or_gradients_serialized=False,measurements=measured,
            diagnostic_seconds_excluding_outer_teardown=time.perf_counter()-started,
            authoritative_wall_charge="complete bounded outer terminal, including failure and teardown"))
    except BaseException as error:
        failure=dict(type=type(error).__name__,message=str(error),stage=args.stage,request=request_ref,
            partial_outputs_preserved=True,automatic_retry_or_selective_promotion_authorized=False,
            wall_charge="all actual bounded outer wall, including failed work",live=context)
        if "torch" in locals() and torch.cuda.is_initialized():
            failure["failed_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
            failure["failed_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
        # Failure evidence must survive an exhausted ordinary write budget.
        with (output / "FAILURE.json").open("x") as stream:
            json.dump(failure,stream,indent=2,allow_nan=False);stream.write("\n")
        raise


if __name__ == "__main__":
    main()
