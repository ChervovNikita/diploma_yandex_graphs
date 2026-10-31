"""Source-candidate native Polynormer Q/K and actual boundary-R/S/B baseline.

Standard-library imports only until make_adapter. Exact source operations are
reused; fixed old TRAIN guards/Adam LR are explicitly provider-bound. Complete
local/global transition and source RNG ownership remain represented. This is a
new standardized source candidate, not native author CLI score replication.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import types

try:
    from .classifier import ClassifierAdapter
    from ..spec import confined, normalized_id, ORIGINAL_PAPER_DATASETS
except ImportError:
    from adapters.classifier import ClassifierAdapter
    from spec import confined, normalized_id, ORIGINAL_PAPER_DATASETS

SOURCES = {
    "operator": ("pre_sigmoid_qk_operator_adapter_source_20261009_v1/operator_adapter.py", "be9974399e090b424c57fe4f77ba10126f575d8e89454945ce51f0f7c61b9631"),
    "native": ("wikics_staged_private_graph_residual_method_preparation_20261007_v4/vendor/native_polynormer.py", "9b4e533f46ae7f91a23a552f88bbb47a01359e224b53996cf1a68c10a865f6a8"),
    "models": ("portable_internal_be_public_interface_20261007_v2/core/models.py", "2be6d872e962d9fdd341883b0d1060170fe0a176f7a8d6829d988fc6f8e7f9c3"),
    "factors": ("portable_internal_be_public_interface_20261007_v2/core/factors.py", "9f185dfeb05a059f6c5d84062e1b6226ab29a288b4c7ab8fac08f8dfb523f9c3"),
    "objectives": ("portable_internal_be_public_interface_20261007_v2/core/objectives.py", "9208e677ab5cfb286862216991453f8592a1d373120019a6e2ff14c38fad5bcf"),
    "selection": ("portable_internal_be_public_interface_20261007_v2/core/selection.py", "c48c08adfea5d08b945a38e3edf55522d6dbfa44418bf902fb46700a8cc11d20"),
    "portable": ("portable_internal_be_public_interface_20261007_v2/portable.py", "29beab567eeb4c070c432e2ce31990c88ec2d8c8618aa5dfb85ebedcf4f0ca14"),
    "driver": ("portable_internal_be_public_interface_20261007_v2/train.py", "ea486cd59b33c1a5dcf9b0cf19672690e469e011f2b2db4781913d102625729a"),
    "recipe": ("portable_internal_be_public_interface_20261007_v2/recipes/wikics.json", "d93ca649500ee016ce5dd1a4fe96ed5c4515c96a4f97bb9bce3623f65ccc7c2d"),
    "original_boundary": ("common_wrapper_native_source_20261010_v1/models.py", "07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303"),
}
OPERATORS = ("native_tied", "active_reversible_exp", "pre_sigmoid_split", "full_qk")
ARMS = {operator + "_" + kind: (operator, kind, 4 if kind == "be_init" else 1)
        for operator in OPERATORS for kind in ("single", "be_init")}
ARMS.update(bare_native_single=("bare", "single", 1), original_gnnm=("original_boundary", "boundary4", 4))
I4 = {operator + "_independent4": operator + "_single" for operator in OPERATORS}
I4["bare_native_independent4"] = "bare_native_single"
CONTRACT = {
    "architecture": "native Polynormer; distinct from PolyFormer;hidden512/local7/global2/heads1/beta-1/pre_lnFalse",
    "initializer": "source native reset;BE first-stem Rademacher seed+900001;all later factors1;gamma/delta0;fullq deep-copy k with independent storage",
    "seed_law": "actual_seed=base+1009*body;source native reset actual_seed;private stream actual_seed+1009*route+300001",
    "objective": "source F-only;view_then_member;two full ownCE views;all old-state backwards before Adam;shared .5/M,private .5",
    "optimizer": "source Adam;declared LR;eps1e-8;weight_decay0;default betas;one optimizer/physical body",
    "transition": "after100local updates restore strict-first SELECT-best-local model+Adam only;preserve end-local global/private RNG;then native globalTrue",
    "serving": "native/QK probability mean;original RSB boundary raw-logit mean",
    "original_boundary": "actual source BEBlock input and local/global heads;bias-free Xavier W;input RademacherR;outputR1;allS1,B0;no internal factors",
    "old_protocol": "WikiCS fixed TRAIN580/VALID5274;1100total/100local;source driver unchanged",
    "new_protocol": "new GRAPH/TRAIN/SELECT/CAL/evaluate;all masks;declared seeds/domain;5000complete/100local;strict first maximum SELECT;no checkpoint0/earlystop;common500-step scalar CAL",
    "author_CLI_or_published_score_equality": False,
}


def receipt(root, path):
    root = Path(root).resolve();path = Path(path)
    if not path.is_absolute():
        path = confined(root, str(path))
    path = path.resolve()
    if not path.is_relative_to(root) or path.is_symlink():
        raise ValueError("Confined exact source/input required")
    raw = path.read_bytes()
    return {"path": str(path.relative_to(root)), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def bound(root, row):
    path = confined(root, row["path"])
    if receipt(root, path) != row:
        raise ValueError("Exact native Polynormer source/input changed")
    return path


def load_module(root, row, tag):
    path = bound(root, row);name = "polynormer_qk_" + tag + "_" + row["sha256"][:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        result = importlib.util.module_from_spec(spec);sys.modules[name] = result;spec.loader.exec_module(result)
    return sys.modules[name]


def source_functions(root, bindings, torch, train_rows, lr):
    """Reuse source mathematical functions; replace two fixed TRAIN guards/LR."""
    if set(bindings) != set(SOURCES) or any(bindings[key]["sha256"] != value[1] for key, value in SOURCES.items()):
        raise ValueError("Complete exact Q/K/native/core/RSB source set required")
    tree = ast.parse(bound(root, bindings["operator"]).read_bytes())
    names = ("require", "_global_forward", "_attach", "_streamed_f_step")
    originals = [next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name) for name in names]
    functions = copy.deepcopy(originals);changes = []
    for node in functions:
        for child in ast.walk(node):
            if node.name == "_streamed_f_step" and isinstance(child, ast.Constant) and child.value == 580:
                child.value = int(train_rows);changes.append("complete_TRAIN_row_guard")
            if node.name == "_attach" and isinstance(child, ast.keyword) and child.arg == "lr":
                if not isinstance(child.value, ast.Constant) or child.value.value != .001:
                    raise ValueError("Exact source Adam LR anchor changed")
                child.value.value = float(lr);changes.append("declared_Adam_LR")
    if sorted(changes) != ["complete_TRAIN_row_guard", "complete_TRAIN_row_guard", "declared_Adam_LR"]:
        raise ValueError("Only two TRAIN guards and Adam LR are prospective source bindings")
    ns = {"ast": ast, "copy": copy, "types": types, "torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), "<pinned_native_Polynormer_QK_math>", "exec"), ns)
    ns["source_function_AST_hashes"] = {node.name: hashlib.sha256(ast.dump(node).encode()).hexdigest() for node in originals}
    ns["explicit_provider_changes"] = changes
    return ns


def original_block(root, bindings, torch):
    tree = ast.parse(bound(root, bindings["original_boundary"]).read_bytes())
    blocks = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BEBlock"]
    if len(blocks) != 1:
        raise ValueError("One exact original R/S/B block class required")
    ns = {"torch": torch, "nn": torch.nn, "__name__": "source_original_RSB_native_Polynormer"}
    exec(compile(ast.Module(body=blocks, type_ignores=[]), "<exact_source_original_BEBlock>", "exec"), ns)
    return ns["BEBlock"], hashlib.sha256(ast.dump(blocks[0]).encode()).hexdigest()


def attach_boundary(root, bindings, torch, model):
    """Actual original blocks, distinct from FactorLinear all-map replacement."""
    source_block, AST_hash = original_block(root, bindings, torch)
    class BoundaryLinear(source_block):
        def __init__(self, linear, first):
            super().__init__(linear.weight.shape[1], linear.weight.shape[0], first, "cpu", 4)
            self.member = 0
        def forward(self, x):
            return source_block.forward(self, x, self.member)
    wrapper = model.models[0];body = wrapper.body
    for name, first in (("lin_in", True), ("pred_local", False), ("pred_global", False)):
        linear = getattr(body, name)
        if not isinstance(linear, torch.nn.Linear):
            raise ValueError("Fresh native Polynormer input/prediction Linear required")
        setattr(body, name, BoundaryLinear(linear, first))
    wrapper.member = 0;original_forward = wrapper.forward.__func__
    def routed_forward(self, batch):
        blocks = [self.body.lin_in, self.body.pred_local, self.body.pred_global]
        old = [block.member for block in blocks]
        for block in blocks: block.member = self.member
        try:
            return original_forward(self, batch)
        finally:
            for block, member in zip(blocks, old): block.member = member
    wrapper.forward = types.MethodType(routed_forward, wrapper)
    model.independent = False;model.members = 4;model.arm = "original_gnnm_boundary4"
    return BoundaryLinear, {"BEBlock_AST_sha256": AST_hash, "sites": ["lin_in", "pred_local", "pred_global"],
                            "native_boundary_W_bias_replaced": True, "internal_factors_installed": False}


def make_adapter(*, torch, project_root, study, run, device):
    import json
    import numpy as np
    c = run["config"];root = Path(project_root).resolve();arm = c["polynormer_qk_arm"]
    if (normalized_id(study["dataset"]) in ORIGINAL_PAPER_DATASETS or arm not in ARMS or c["source_contract"] != CONTRACT
            or torch.get_default_dtype() != torch.float32):
        raise ValueError("Declared new-data native Polynormer source arm required")
    operator, kind, members = ARMS[arm]
    if (c["members"] != members or c["physical_slow_bodies"] != 1 or c["source_optimizer_count"] != 1
            or c["width"] != 512 or c["depth"] != 7 or c["global_layers"] != 2 or c["heads"] != 1
            or c["local_transition_updates"] != 100 or c["learning_rate"] not in (.0005, .001, .002)
            or (c["in_dropout"], c["dropout"], c["global_dropout"]) not in ((.5,.5,.5),(.5,.3,.5),(.5,.5,.3))):
        raise ValueError("Frozen source body/useful explicit finite search domain required")
    payloads = {role: torch.load(bound(root, row), map_location="cpu", weights_only=True) for role, row in c["node_inputs"].items()}
    graph, train, select = (payloads[key] for key in ("graph", "train", "select"))
    if set(graph) != {"x", "edge_index"} or set(train) != {"ids", "y"} or set(select) != {"ids", "y"}:
        raise ValueError("Separate factual GRAPH/TRAIN/SELECT required")
    if (graph["x"].dtype != torch.float32 or graph["x"].ndim != 2 or not torch.isfinite(graph["x"]).all()
            or graph["edge_index"].dtype != torch.long or graph["edge_index"].ndim != 2 or graph["edge_index"].shape[0] != 2
            or train["ids"].dtype != torch.long or train["y"].dtype != torch.long or train["ids"].shape != train["y"].shape
            or len(torch.unique(train["ids"])) != len(train["ids"]) or torch.isin(train["ids"], select["ids"]).any()):
        raise ValueError("Finite native graph/disjoint categorical TRAIN/SELECT required")
    sources = c["polynormer_qk_sources"]
    functions = source_functions(root, sources, torch, len(train["ids"]), c["learning_rate"])
    factors = load_module(root, sources["factors"], "factors")
    old = sys.modules.get("factors")
    try:
        sys.modules["factors"] = factors
        models = load_module(root, sources["models"], "models")
    finally:
        if old is None: sys.modules.pop("factors", None)
        else: sys.modules["factors"] = old
    native = load_module(root, sources["native"], "native")
    core = {"factors": factors, "models": models, "objectives": load_module(root, sources["objectives"], "objectives"),
            "selection": load_module(root, sources["selection"], "selection")}
    recipe = json.loads(bound(root, sources["recipe"]).read_text())
    options = {**recipe["model"], "in_channels": graph["x"].shape[1], "out_channels": c["classes"],
               "in_dropout": c["in_dropout"], "dropout": c["dropout"], "global_dropout": c["global_dropout"]}
    actual_seed = run["seed"] + 1009 * run["body"]
    random.seed(actual_seed);np.random.seed(actual_seed);torch.manual_seed(actual_seed)
    device = torch.device(device);cuda_index = (device.index or 0) if device.type == "cuda" else None
    if cuda_index is not None: torch.cuda.manual_seed(actual_seed)
    model = models.Ensemble("wikics", "be_init" if kind == "be_init" else "single", actual_seed, options, (native, None))
    boundary_type = None;boundary_record = None
    if kind == "boundary4":
        boundary_type, boundary_record = attach_boundary(root, sources, torch, model)
    model = model.to(device);model.set_global(False)
    initial_optimizer = torch.optim.Adam(model.parameters(), lr=c["learning_rate"], eps=1e-8, weight_decay=0.)
    session = SimpleNamespace(torch=torch, np=np, task="wikics", arm="be_init" if kind in ("be_init", "boundary4") else "single",
                              seed=actual_seed, device=device, cuda_index=cuda_index, cuda_devices=[] if cuda_index is None else [cuda_index],
                              model=model, optimizers=[initial_optimizer], core=core, steps=0)
    if operator not in ("bare", "original_boundary"):
        functions["_attach"](session, operator, bound(root, sources["native"]))
    optimizer = session.optimizers[0]
    if len(session.optimizers) != 1 or {id(p) for group in optimizer.param_groups for p in group["params"]} != {id(p) for p in model.parameters()}:
        raise ValueError("Exactly one native Adam owner per physical body required")
    session.streams = []
    for member in range(members):
        with torch.random.fork_rng(devices=session.cuda_devices):
            torch.manual_seed(actual_seed + 1009*member + 300001)
            stream = {"cpu": torch.get_rng_state().clone()}
            if cuda_index is not None:
                torch.cuda.manual_seed(actual_seed + 1009*member + 300001)
                stream["cuda"] = torch.cuda.get_rng_state(cuda_index).clone()
            session.streams.append(stream)
    session._operator_failed = False
    session.operator_work = dict(update_attempts=0, completed_updates=0, member_view_forwards=0, member_view_backwards=0, Adam_steps=0)
    x = graph["x"].to(device);edges = graph["edge_index"].to(device);ids = train["ids"].to(device);y = train["y"].to(device)
    full_batch = {"x": x, "edge_index": edges, "ids": torch.arange(len(x), device=device)}
    train_batch = {**full_batch, "ids": ids}
    session.train_step = types.MethodType(functions["_streamed_f_step"], session)
    local = {"best": None, "step": None, "snapshot": None, "transition_completed": False}
    def cpu_tree(value):
        if torch.is_tensor(value): return value.detach().cpu().clone()
        if isinstance(value, dict): return {key: cpu_tree(field) for key, field in value.items()}
        if isinstance(value, list): return [cpu_tree(field) for field in value]
        if isinstance(value, tuple): return tuple(cpu_tree(field) for field in value)
        return copy.deepcopy(value)
    def logits_fn(current, training, step):
        rows = []
        for member, stream in enumerate(session.streams):
            with torch.random.fork_rng(devices=session.cuda_devices):
                torch.set_rng_state(stream["cpu"])
                if cuda_index is not None: torch.cuda.set_rng_state(stream["cuda"], cuda_index)
                rows.append(current.member_forward(full_batch, member)[0])
                stream["cpu"] = torch.get_rng_state()
                if cuda_index is not None: stream["cuda"] = torch.cuda.get_rng_state(cuda_index)
        return torch.stack(rows)
    def custom_train(adapter, step):
        transition = 0
        if session._operator_failed: raise ValueError("No automatic retry of failed exact-source update")
        if step != session.steps + 1: raise ValueError("Source complete-update order/continuation differs")
        if step == 101:
            if local["transition_completed"] or local["snapshot"] is None or any(body.body._global for body in model.models):
                raise ValueError("Exactly one declared SELECT-best-local transition required")
            # Model and Adam restore only. Live end-local private/global RNG and
            # Session counters intentionally remain at the100-update boundary.
            try:
                model.load_state_dict(local["snapshot"]["model"], strict=True)
                optimizer.load_state_dict(local["snapshot"]["optimizer"])
                model.set_global(True);local["transition_completed"] = True;transition = 1
            except BaseException:
                session._operator_failed = True
                raise
        result = session.train_step(train_batch, y)
        return {"objective": float(result["loss"]), "ce": float(result["own_mean"]), "regularization": 0.,
                "operation_counts": {"optimizer_updates": 1, "model_forwards": 2*members, "route_predictions": 2*members,
                                     "backward_calls": 2*members, "local_model_Adam_restores": transition, "supervised_examples": 2*len(ids)}}
    def capture():
        nr = np.random.get_state()
        return {"streams": copy.deepcopy(session.streams), "numpy_rng": [nr[0], nr[1].tolist(), *nr[2:]],
                "steps": session.steps, "operator_failed": session._operator_failed, "operator_work": copy.deepcopy(session.operator_work),
                "body_global": [body.body._global for body in model.models], "local": cpu_tree(local),
                "factor_members": {name: module.member for name, module in model.named_modules() if isinstance(module, factors.FactorLinear)},
                "boundary_members": {name: module.member for name, module in model.named_modules() if boundary_type is not None and isinstance(module, boundary_type)},
                "wrapper_members": [getattr(body, "member", None) for body in model.models]}
    def restore(value):
        session.streams[:] = copy.deepcopy(value["streams"]);session.steps = value["steps"];session._operator_failed = value["operator_failed"]
        session.operator_work = copy.deepcopy(value["operator_work"]);nr = value["numpy_rng"];np.random.set_state((nr[0], np.array(nr[1], dtype=np.uint32), *nr[2:]))
        for body, active, member in zip(model.models, value["body_global"], value["wrapper_members"]):
            body.set_global(active)
            if member is not None: body.member = member
        local.clear();local.update(cpu_tree(value["local"]))
        modules = dict(model.named_modules())
        for key in ("factor_members", "boundary_members"):
            for name, member in value[key].items(): modules[name].member = member
    adapter = ClassifierAdapter(torch, torch.nn.functional, model, optimizer, logits_fn=logits_fn, train_ids=ids, train_y=y,
                                select_ids=select["ids"].to(device), select_y=select["y"].to(device), regularizer=lambda current: x.new_zeros(()),
                                fusion="mean_logits" if kind == "boundary4" else "mean_probabilities",
                                selector={"metric": "accuracy", "direction": "max", "ties": "first", "patience_steps": 5001, "save_every": 1},
                                custom_train=custom_train, external_capture=capture, external_restore=restore,
                                provenance={"run_id": run["run_id"], "data": study["data"], "source": study["source"], "config": c,
                                            "actual_model_seed": actual_seed, "source_contract": CONTRACT, "boundary": boundary_record,
                                            "source_math_AST_hashes": functions["source_function_AST_hashes"], "source_provider_changes": functions["explicit_provider_changes"]})
    native_observer = adapter.observe_validation
    def observe_validation(step, validation):
        decision = native_observer(step, validation)
        metric = validation["selection_metric"]
        if step <= 100 and (local["best"] is None or metric > local["best"]):
            if any(body.body._global for body in model.models): raise ValueError("Local SELECT used global branch")
            local.update(best=metric, step=step, snapshot={"model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict())})
        return decision
    adapter.observe_validation = observe_validation
    adapter.source_session = session
    return adapter
