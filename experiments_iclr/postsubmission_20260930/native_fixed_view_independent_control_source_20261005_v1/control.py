"""Proposed ordinary native fixed-view independent control; no default execution.

This source reuses the existing native constructor, acquisition and selection
functions. Only the per-member objective adds that member's assigned fixed view.
The existing 30-fit protocol supplies input identities, never permission to run
this separately proposed control. No dataset or saved-outcome reader is supplied.
"""
import builtins
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent
REFERENCE = "accuracy_first_graph_view_reference_source_preparation_20261004_v2"
BANK = "accuracy_first_graph_view_source_preparation_20261004_v2"
KIND = "native_fixed_view_member"

# Existing source descriptors, not a new manifest or scientific protocol.
BOUND = {
    "reference_manifest": (REFERENCE + "/MANIFEST.json", 2991,
        "3a054568e54baed6ee6526256b6d37fa5f82b449257d7983c1ef54eace8236e0"),
    "reference_protocol": (REFERENCE + "/PROTOCOL.json", 5113,
        "05d13ba716857c70a399cf1b308dc9ed1d76596afbb72067d98cf3e97a107546"),
    "reference_bindings": (REFERENCE + "/SOURCE_BINDINGS.json", 3179,
        "3222e89e4e25d08f20b06ac2fb272fcab282604b28ef72f888153c0cf046ad5e"),
    "admission": (REFERENCE + "/admission.py", 1408,
        "850f46ad71b41cc7735cd341867a622dd7d82c423f51106e53fffbfa521d3473"),
    "contract": (REFERENCE + "/contract.py", 2633,
        "72cfbc82f87d1830735c17a52fcf49fef1fa3d4be52187feccdffe060671f593"),
    "runtime": (REFERENCE + "/runtime.py", 2664,
        "df48aff13a7d273e3418f33ea07a60019e200cd5f9592c297f6159b1d782fad9"),
    "native_reference": (REFERENCE + "/native_reference.py", 8026,
        "20465859822a71cad8f6ad95775fbd7aca76f5dd4bb41b9cc593ec27d048ace7"),
    "driver": (REFERENCE + "/driver.py", 11011,
        "a4c8c302812c5bbcdc5b53a985bca8ac3975473bc9bf19ab28439506ffabc8f9"),
    "schedule": (BANK + "/schedule.py", 3051,
        "bd01e4653b49a9b1b8683a27848de7f90f3feef098f5b4eb16167aafebd800cf"),
    "native_cpu_fixture": (REFERENCE + "/test_numerical.py", 9183,
        "11c06cd844a7716fcd46d7f6e396c90450d8857d1a6bee330258172585fdc8b7"),
    "paired_inputs": ("accuracy_first_graph_view_paired_plan_root_20261004_v1/PAIRED_PROTOCOL.json", 19441,
        "cfbf6094482ff28e907eb92e8ab6d50429e4c13518cfa954b6c14dd621ae77ac"),
}


def descriptor(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def declared(key):
    path, size, sha256 = BOUND[key]
    return {"path": path, "bytes": size, "sha256": sha256}


def verify(record):
    path = Path(record["path"])
    actual = descriptor(path if path.is_absolute() else PHASE / path)
    if actual["bytes"] != record["bytes"] or actual["sha256"] != record["sha256"]:
        raise ValueError("Bound source/input changed: " + record["path"])
    return actual


def verify_source_bindings():
    """Hash source only; do not traverse manifest payloads containing test receipts."""
    records = {key: verify(declared(key)) for key in BOUND}
    inherited = json.loads(Path(records["reference_bindings"]["path"]).read_text())["files"]
    for key in ("native", "helpers", "views", "native_recipe"):
        records[key] = verify(inherited[key])
    paired = json.loads(Path(records["paired_inputs"]["path"]).read_text())
    if paired["reference_manifest_sha256"] != BOUND["reference_manifest"][2] or \
            paired["source_protocols"]["reference"]["sha256"] != BOUND["reference_protocol"][2]:
        raise ValueError("Existing paired input/source provenance differs")
    return records, inherited, paired


def _module(key, record, imports=None):
    """Compile verified bytes without caches or bare-module name collisions."""
    actual = verify(record)
    data = Path(actual["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != actual["sha256"]:
        raise ValueError("Source changed before compilation")
    name = "_native_fixed_view_control_" + key
    module = ModuleType(name)
    module.__file__ = actual["path"]
    replacements = {} if imports is None else imports

    def source_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in replacements:
            return replacements[name]
        return builtins.__import__(name, globals, locals, fromlist, level)

    module.__dict__["__builtins__"] = dict(vars(builtins), __import__=source_import)
    sys.modules[name] = module  # Required by inherited stdlib dataclasses.
    exec(compile(data, actual["path"], "exec"), module.__dict__)
    return module


def source_functions():
    """Load only stdlib source. Native numerical modules remain unloaded."""
    records, inherited, paired = verify_source_bindings()
    modules = {key: _module(key, declared(key)) for key in ("admission", "contract", "runtime")}
    modules["views"] = _module("views", inherited["views"])
    modules["helpers"] = _module("helpers", inherited["helpers"])
    modules["schedule"] = _module("schedule", declared("schedule"), {"views": modules["views"]})
    modules["native_reference"] = _module("native_reference", declared("native_reference"), modules)
    modules["driver"] = _module("driver", declared("driver"), modules)
    modules["native_cpu_fixture"] = _module("native_cpu_fixture", declared("native_cpu_fixture"), modules)
    return SimpleNamespace(**modules, records=records, inherited=inherited, paired=paired)


def load_runtime(*, numerical=False):
    """Explicit numerical loading only; neither this call nor CLI starts fitting."""
    if numerical is not True:
        raise RuntimeError("Source-only default; explicit numerical entry required")
    started = time.perf_counter()
    sources = source_functions()
    import numpy
    import torch
    native = _module("native", sources.inherited["native"])
    return SimpleNamespace(torch=torch, numpy=numpy, native=native,
        helpers=sources.helpers, views=sources.views, sources=sources,
        manifest_sha256=BOUND["reference_manifest"][2],
        protocol_sha256=BOUND["reference_protocol"][2],
        provenance=tuple(sources.records[key] for key in ("native", "helpers", "views")),
        runtime_load_seconds_shared=time.perf_counter() - started)


def member_spec(sources, split, member):
    original = sources.contract.fit_spec("native_member", split, member)
    view_seed = dict(sources.contract.BLOCKS)[split]
    assigned = sources.schedule.assignment("tied_persistent", view_seed, 1)[member]
    return dict(original, kind=KIND, assigned_view=assigned, view_seed=view_seed)


def pass_plan(spec):
    if spec["kind"] != KIND or spec["assigned_view"] not in ("equal", "different"):
        raise ValueError("Declared fixed semantic member view required")
    return (("native", 1.0), (spec["assigned_view"], 1.0))


def build_member(rt, split, member, device="cpu"):
    """Fresh native constructor/reset and Adam under the existing independent seed."""
    member_spec(rt.sources, split, member)
    return rt.sources.native_reference.build(rt, "native_member", split, member, device)


def assert_independent_ownership(rt, members):
    """Optional in-memory four-member audit; reject Parameter or storage reuse."""
    if len(members) != 4:
        raise ValueError("Four independently acquired model/optimizer pairs required")
    parameter_owners, storage_owners = set(), set()
    for model, optimizer in members:
        rt.sources.native_reference.optimizer_names(model, optimizer)
        ids = {id(p) for p in model.parameters()}
        storage = {(str(p.device), p.untyped_storage().data_ptr()) for p in model.parameters()}
        if parameter_owners.intersection(ids) or storage_owners.intersection(storage):
            raise ValueError("Independent native members share parameters/storage")
        parameter_owners.update(ids)
        storage_owners.update(storage)


def accumulate_member_gradients(rt, model, optimizer, x, graphs, train_ids, train_labels, spec, update, costs):
    """Complete TRAIN CE(native)+CE(assigned view), without a parameter update."""
    native = rt.sources.native_reference
    if type(model._global) is not bool or model._global != rt.sources.contract.stage_at(update):
        raise ValueError("Actual native stage differs")
    torch = rt.torch
    index = torch.tensor(train_ids, dtype=torch.long, device=x.device)
    targets = torch.tensor(train_labels, dtype=torch.long, device=x.device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    value = 0.0
    for graph, coefficient in pass_plan(spec):
        costs["native_training_forwards" if graph == "native" else "assigned_view_training_forwards"] += 1
        logits = model(x, graphs[graph])
        if logits.dtype != torch.float32 or tuple(logits.shape) != (x.shape[0], 5) or \
                not torch.isfinite(logits).all().item():
            raise ValueError("Complete finite FP32 native logits required")
        logp = torch.nn.functional.log_softmax(logits, dim=1).index_select(0, index)
        loss = torch.nn.functional.nll_loss(logp, targets) * coefficient
        if not torch.isfinite(loss).item():
            raise ValueError("Nonfinite native/view CE")
        costs["training_backwards"] += 1
        loss.backward()
        value += float(loss.detach().cpu())
        del logits, logp, loss
    native.check_gradients(rt, model)
    return value


def train_member_step(rt, model, optimizer, x, graphs, train_ids, train_labels, spec, update, costs):
    """Accumulate the two TRAIN passes, then perform one ordinary Adam step."""
    value = accumulate_member_gradients(rt, model, optimizer, x, graphs, train_ids, train_labels, spec, update, costs)
    native, torch = rt.sources.native_reference, rt.torch
    costs["optimizer_step_attempts"] += 1
    optimizer.step()
    costs["optimizer_steps_completed"] += 1
    native.optimizer_names(model, optimizer)
    if any(not torch.isfinite(p).all().item() for p in model.parameters()) or \
            any(isinstance(v, torch.Tensor) and not torch.isfinite(v).all().item()
                for state in optimizer.state.values() for v in state.values()):
        raise ValueError("Nonfinite native parameter/Adam state")
    return value


def verify_cpu_components(rt):
    """Bound native synthetic component check: zero optimizer steps and zero fits.

    Reuse the existing native-reference CPU fixture's recipe, graph and labels.
    Compare production accumulation to independently computed per-pass gradients
    at the same parameters/RNG. Clones here are oracle copies only; the four
    audited members are each freshly constructed under their own native seed.
    No data, saved outcomes, checkpoints or trace files are read or written.
    """
    started = time.perf_counter()
    sources, torch = rt.sources, rt.torch
    fixture = sources.native_cpu_fixture
    fixture.RT = rt
    first, first_optimizer, x, bundle, graphs = fixture.NumericalReferences().fixture("native_member")
    members = [(first, first_optimizer)]
    for member in range(1, 4):
        model, optimizer, _ = sources.native_reference.build(rt, "native_member", 0, member, "cpu",
                                                            synthetic_recipe=fixture.RECIPE)
        members.append((model, optimizer))
    assert_independent_ownership(rt, members)
    role = bundle["role"]
    ids = torch.tensor(role.train_ids, dtype=torch.long)
    targets = torch.tensor(role.train_labels, dtype=torch.long)
    initial = [rt.helpers.cpu_tree(rt, model.state_dict()) for model, _ in members]
    cases, totals = [], _costs()
    for member, (model, optimizer) in enumerate(members):
        spec = member_spec(sources, role.split, member)
        for global_stage, update in ((False, 1), (True, 201)):
            model._global = global_stage
            oracle = deepcopy(model)
            oracle.train()
            parameters = tuple(oracle.parameters())
            expected = [None for _ in parameters]
            before = rt.helpers.cpu_tree(rt, rt.helpers.rng_state(rt, "cpu"))
            actual_loss = accumulate_member_gradients(rt, model, optimizer, x, graphs,
                role.train_ids, role.train_labels, spec, update, totals)
            after = rt.helpers.cpu_tree(rt, rt.helpers.rng_state(rt, "cpu"))
            rt.helpers.restore_rng(rt, "cpu", before)
            expected_loss = 0.0
            # Explicit oracle objective; do not use the implementation's pass_plan.
            assigned = sources.schedule.assignment("tied_persistent", spec["view_seed"], update)[member]
            for graph in ("native", assigned):
                logits = oracle(x, graphs[graph])
                loss = torch.nn.functional.cross_entropy(logits.index_select(0, ids), targets)
                gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
                expected_loss += float(loss.detach())
                for index, gradient in enumerate(gradients):
                    if gradient is not None:
                        expected[index] = gradient if expected[index] is None else expected[index] + gradient
            # The tolerances are the existing bound native CPU check's tolerances.
            torch.testing.assert_close(torch.tensor(actual_loss), torch.tensor(expected_loss), rtol=2e-5, atol=2e-7)
            active = 0
            for parameter, gradient in zip(model.parameters(), expected):
                if gradient is None:
                    if parameter.grad is not None:
                        raise AssertionError("Dormant native gradient unexpectedly present")
                else:
                    if parameter.grad is None:
                        raise AssertionError("Active native gradient missing")
                    torch.testing.assert_close(parameter.grad, gradient, rtol=2e-5, atol=2e-7)
                    active += 1
            replay_after = rt.helpers.rng_state(rt, "cpu")
            for key in after:
                equal = torch.equal(after[key], replay_after[key]) if isinstance(after[key], torch.Tensor) else \
                        after[key] == replay_after[key]
                if not equal:
                    raise AssertionError("Native/view RNG opportunity differs: " + key)
            if optimizer.state:
                raise AssertionError("Component verification performed an optimizer update")
            cases.append({"member": member, "seed": spec["seed"], "assigned_view": assigned,
                          "global": global_stage, "complete_TRAIN_rows_each_pass": len(role.train_ids),
                          "active_native_parameter_gradients_checked": active})
    for (model, optimizer), expected in zip(members, initial):
        for name, value in model.state_dict().items():
            if not torch.equal(value, expected[name]):
                raise AssertionError("Component verification changed native state: " + name)
        if optimizer.state:
            raise AssertionError("Component verification created Adam moments")
    return {"status": "PASS_BOUNDED_NATIVE_CPU_COMPONENTS_ONLY", "control_source": descriptor(__file__),
            "fixture_source": declared("native_cpu_fixture"), "fresh_independent_native_members": 4,
            "independent_parameter_storage": "disjoint", "cases": cases,
            "production_native_forwards": totals["native_training_forwards"],
            "production_assigned_view_forwards": totals["assigned_view_training_forwards"],
            "production_backwards": totals["training_backwards"],
            "oracle_native_and_view_forwards": 2 * len(cases),
            "oracle_autograd_grad_calls": 2 * len(cases), "oracle_state_copies": len(cases),
            "optimizer_steps": 0, "physical_fits": 0, "scientific_training": False,
            "seconds": time.perf_counter() - started}


def _validate_context(rt, context, spec, bundle, x, validation_labels, device):
    sources = rt.sources
    verify_source_bindings()
    rt.views.verify_bundle(bundle)
    source_sha256 = descriptor(__file__)["sha256"]
    if context.get("schema") != "native-fixed-view-independent-control-context-v1" or \
            context.get("execution_authorized") is not True or \
            context.get("control_source_sha256") != source_sha256 or \
            context.get("reference_manifest_sha256") != rt.manifest_sha256 or \
            context.get("reference_protocol_sha256") != rt.protocol_sha256:
        raise ValueError("Separate root-admitted exact control/source context required")
    if verify(context["paired_input_protocol"])["sha256"] != BOUND["paired_inputs"][2]:
        raise ValueError("Exact existing paired input identities required")
    role = bundle["role"]
    if role.split != spec["split"] or bundle["coverage"]["view_seed"] != spec["view_seed"]:
        raise ValueError("Existing assigned split/view seed required")
    coverage = sources.paired["official_TRAIN_coverage_by_split"][spec["split"]]["descriptor"]
    if verify(context["coverage"]) != verify(coverage):
        raise ValueError("Exact existing official TRAIN coverage required")
    saved = json.loads(Path(verify(context["coverage"])["path"]).read_text())
    expected = dict(bundle["coverage"], coverage_origin="official_TRAIN",
                    scientific_freeze_eligible=bundle["coverage"]["coverage_eligible"])
    if saved != expected or saved["scientific_freeze_eligible"] is not True:
        raise ValueError("Complete eligible official TRAIN fixed-view bundle required")
    features = sources.driver.feature_identity(x)
    labels_sha256 = sources.contract.digest(validation_labels)
    if context.get("features") != features or context.get("validation_labels_sha256") != labels_sha256 or \
            context.get("selected_device") != rt.helpers.device_provenance(device):
        raise ValueError("Live feature/validation/device admission differs")
    sources.admission.validate_paired_inputs(sources.paired, spec["split"],
        {"role": role.identity(), "native_edges_sha256": bundle["coverage"]["native_edges_sha256"],
         "feature_identity": features, "validation_labels_sha256": labels_sha256})
    policy = sources.paired["runtime_policy"]
    if rt.torch.are_deterministic_algorithms_enabled() != policy["deterministic_algorithms"] or \
            rt.torch.is_deterministic_algorithms_warn_only_enabled() != policy["warn_only"] or \
            rt.torch.backends.cuda.matmul.allow_tf32 != policy["cuda_matmul_TF32"] or \
            rt.torch.backends.cudnn.allow_tf32 != policy["cudnn_TF32"] or \
            os.environ.get("CUBLAS_WORKSPACE_CONFIG") != policy["CUBLAS_WORKSPACE_CONFIG"]:
        raise ValueError("Existing common deterministic runtime policy required")


def _costs():
    return {key: 0 for key in ("physical_member_acquisition_attempts", "physical_members_acquired",
        "native_training_forwards", "assigned_view_training_forwards", "training_backwards",
        "optimizer_step_attempts", "optimizer_steps_completed", "native_selection_forwards",
        "native_restore_verification_forwards", "completed_actual_updates", "checkpoint_bytes_written")}


def run_member(rt, *, execute=False, split, member, x, bundle, validation_labels,
               context, output, device="cpu"):
    """Train one independently selected member after separate prospective admission.

    The caller supplies projected inputs; the original 30-fit plan is unchanged.
    Failure retains spent-cost/trace receipts and propagates; no retry is offered.
    """
    if execute is not True:
        raise RuntimeError("Scientific control fitting disabled by default")
    started = time.perf_counter()
    sources, torch = rt.sources, rt.torch
    device = sources.admission.reference_device(rt, device, rt.helpers.resolve_device)
    spec = member_spec(sources, split, member)
    role = bundle["role"]
    if role.nodes != 24492 or len(role.train_ids) != 12246 or len(role.val_ids) != 6123 or \
            len(validation_labels) != len(role.val_ids) or tuple(x.shape) != (24492, 300) or \
            x.dtype != torch.float32 or not torch.isfinite(x).all().item() or \
            any(type(y) is not int or y not in range(5) for y in validation_labels):
        raise ValueError("Existing full raw Amazon TRAIN/VALIDATION contract required")
    _validate_context(rt, context, spec, bundle, x, validation_labels, device)
    output = sources.admission.create_output_directory(output)
    costs = _costs()
    costs["runtime_load_seconds_shared"] = rt.runtime_load_seconds_shared
    costs["counting_rule"] = "Every attempted forward/backward/acquisition is charged, including failures."
    failure, handle = None, None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        costs["CUDA_allocated_bytes_before_member"] = torch.cuda.memory_allocated(device)
    try:
        x = x.to(device)
        edges = dict(bundle["views"], native=bundle["native"])
        graphs = {name: torch.tensor(edges[name], dtype=torch.long, device=device).t().contiguous()
                  for name, _ in pass_plan(spec)}
        costs["physical_member_acquisition_attempts"] += 1
        acquired = time.perf_counter()
        model, optimizer, construction = build_member(rt, split, member, device)
        costs["physical_members_acquired"] += 1
        costs["member_acquisition_seconds"] = time.perf_counter() - acquired
        costs["native_parameter_count"] = sum(p.numel() for p in model.parameters())
        costs["native_parameter_bytes"] = sum(p.numel() * p.element_size() for p in model.parameters())
        bindings = {"control_source": descriptor(__file__), "fit": spec, "context": context,
                    "reference_manifest_sha256": rt.manifest_sha256,
                    "reference_protocol_sha256": rt.protocol_sha256,
                    "source_provenance": sources.records,
                    "selected_device": rt.helpers.device_provenance(device)}
        selector, best_image, selected_logits = sources.contract.Selector(), None, None
        with (output / "TRACE.jsonl").open("x") as trace:
            for update in range(1, 2701):
                if update == 201:
                    local_record = sources.driver.save_image(rt, output / "SELECTED_LOCAL.pt", best_image)
                    transition_record = sources.native_reference.transition(rt, model, optimizer, device, best_image, 200)
                loss = train_member_step(rt, model, optimizer, x, graphs, role.train_ids, role.train_labels,
                                         spec, update, costs)
                costs["native_selection_forwards"] += 1
                logits = sources.native_reference.evaluate(rt, model, x, graphs["native"])
                correct = sources.driver.correct_count(rt, logits, role.val_ids, validation_labels)
                improved = selector.observe(correct, len(role.val_ids), update)
                selection = {"actual_update": update, "actual_local_updates": min(update, 200),
                             "actual_global_updates": max(0, update - 200),
                             "global": sources.contract.stage_at(update),
                             "VAL_correct": correct, "VAL_count": len(role.val_ids)}
                if improved:
                    best_image = sources.native_reference.snapshot(rt, model, optimizer, device, selection, bindings)
                    selected_logits = rt.helpers.cpu_tree(rt, logits)
                    best_image["selected_raw_logits"] = selected_logits
                costs["completed_actual_updates"] = update
                trace.write(json.dumps(dict(selection, TRAIN_CE=loss, strict_selected=improved,
                    assigned_view=spec["assigned_view"], spent_passes=dict(costs)), sort_keys=True) + "\n")
                trace.flush()
        sources.native_reference.restore(rt, model, optimizer, device, best_image, bindings)
        costs["native_restore_verification_forwards"] += 1
        actual = sources.native_reference.evaluate(rt, model, x, graphs["native"])
        if not torch.equal(actual.cpu(), selected_logits) or \
                sources.driver.correct_count(rt, actual, role.val_ids, validation_labels) != selector.best:
            raise ValueError("Exact selected native function/accuracy restoration differs")
        final_record = sources.driver.save_image(rt, output / "SELECTED_FINAL.pt", best_image)
        construction_record = sources.driver.save_image(rt, output / "CONSTRUCTION.pt", construction)
        result = {"schema": "native-fixed-view-independent-member-v1", "bindings": bindings,
                  "completed_actual_updates": 2700, "selection": best_image["selection"],
                  "local_checkpoint": local_record, "final_checkpoint": final_record,
                  "construction": construction_record, "transition": transition_record,
                  "selected_native_logits_identity": sources.driver.feature_identity(selected_logits),
                  "TRAIN_fit_diagnostics": sources.driver.metrics(rt, actual, role.train_ids, role.train_labels),
                  "VAL_exploratory_selected": sources.driver.metrics(rt, actual, role.val_ids, validation_labels)}
        with (output / "RESULT.json").open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
        handle = {"result": result, "selected_native_raw_logits": selected_logits}
        return handle
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        costs["checkpoint_bytes_written"] = sum(path.stat().st_size for path in output.iterdir()
            if path.is_file() and path.suffix == ".pt")  # Includes partial failed writes.
        costs["member_driver_seconds_through_outputs"] = time.perf_counter() - started
        costs["failure"] = failure
        costs["peak_process_RSS_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * \
            (1 if sys.platform == "darwin" else 1024)
        costs["RSS_scope"] = "Process lifetime high water mark; includes any earlier activity."
        if device.type == "cuda":
            costs["CUDA_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            costs["CUDA_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        costs["output_bytes_before_cost_receipt"] = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
        with (output / "COST.json").open("x") as stream:
            json.dump(costs, stream, indent=2, sort_keys=True)
            stream.write("\n")
        if handle is not None:
            handle["cost_receipt"] = descriptor(output / "COST.json")


def pool_members(rt, completed):
    """Fixed arithmetic native probability pool of four own selected members.

    Accept completed in-memory handles; no checkpoint or result reader is needed.
    This function has no additional selector, weight fitting or member exclusion.
    """
    started = time.perf_counter()
    if len(completed) != 4:
        raise ValueError("Exactly four completed physical fixed-view members required")
    ordered, common, checkpoint_paths = {}, None, set()
    for handle in completed:
        result = handle["result"]
        binding = result["bindings"]
        spec = binding["fit"]
        if spec != member_spec(rt.sources, spec["split"], spec["member"]) or \
                result["schema"] != "native-fixed-view-independent-member-v1" or \
                result["completed_actual_updates"] != 2700 or spec["member"] in ordered or \
                binding["control_source"] != descriptor(__file__):
            raise ValueError("All four declared independently selected members required")
        context = {key: value for key, value in binding.items() if key != "fit"}
        if common is None:
            common = context
        elif common != context:
            raise ValueError("Control/source/input/device context differs")
        path = result["final_checkpoint"]["path"]
        if path in checkpoint_paths:
            raise ValueError("Physical selected checkpoint reuse is forbidden")
        checkpoint_paths.add(path)
        value = handle["selected_native_raw_logits"]
        if value.dtype != rt.torch.float32 or tuple(value.shape) != (24492, 5) or \
                not rt.torch.isfinite(value).all().item() or \
                rt.sources.driver.feature_identity(value) != result["selected_native_logits_identity"]:
            raise ValueError("Complete finite selected native logits required")
        ordered[spec["member"]] = handle
    members = [ordered[member] for member in range(4)]
    logits = rt.torch.stack([handle["selected_native_raw_logits"] for handle in members])
    probability = rt.torch.softmax(logits, dim=-1).mean(0)
    return {"native_fixed_view_independent4_probability": probability,
            "member_native_raw_logits": logits,
            "selected_stages_and_updates": [handle["result"]["selection"] for handle in members],
            "physical_cost_receipts": [handle["cost_receipt"] for handle in members],
            "physical_member_checkpoints": [handle["result"]["final_checkpoint"] for handle in members],
            "pool_cost": {"seconds": time.perf_counter() - started, "graph_forwards": 0,
                "member_native_logits_stack_bytes": logits.numel() * logits.element_size(),
                "served_probability_bytes": probability.numel() * probability.element_size()}}


if __name__ == "__main__":
    print(json.dumps({"status": "SEPARATELY_PROPOSED_SOURCE_CONTROL", "execution_authorized": False,
                      "numerical_imports": False, "scientific_training": False,
                      "original_30_fit_plan_changed": False}, sort_keys=True))
