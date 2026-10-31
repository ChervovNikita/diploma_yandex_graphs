"""Pinned masked CORE v3 Session; provider/LR adaptation is explicit."""
from contextlib import contextmanager
import copy
import hashlib
import random
import time
from types import SimpleNamespace

from . import private_polyformer_source as p

SOURCE = {
    "method": {"path": "masked_context_be_source_prototype_20261010_v3/method.py", "bytes": 18445, "sha256": "f0edca0ad2af132559c67819b2ed84fd2511c98ea1ecb4bd5944eeed1bfa0970"},
    "native": {"path": "masked_context_be_source_prototype_20261010_v3/native.py", "bytes": 3524, "sha256": "a7ea5d5bd897ef3b927de3c08bd4966dc94f89105c3f4f9ae258be31a1765c35"},
    "plan": {"path": "masked_context_be_source_prototype_20261010_v3/plan.py", "bytes": 5912, "sha256": "698a36af2f5c804a6a4b1611255e4de60b75961cae0c812dfa49c2f0f322ece6"}}
CONDITIONS = ("shared4_own", "shared4_masked_ce", "shared4_core", "shared4_core_shuffled", "shared4_core_rewired",
    "single_native", "independent4_native", "single_four_view_core", "untied4_shared_decoder_core")
LAW = {
    "shared4_own": (1,4,1,4,4), "shared4_masked_ce": (1,4,1,8,4),
    "shared4_core": (1,4,2,8,4), "shared4_core_shuffled": (1,4,2,8,4), "shared4_core_rewired": (1,4,2,8,4),
    "single_native": (1,1,1,1,1), "independent4_native": (4,4,4,4,4),
    "single_four_view_core": (1,1,2,5,5), "untied4_shared_decoder_core": (4,4,5,8,4)}
PENDING_STAGE2 = {"shared4_masked_ce", "shared4_core_shuffled", "shared4_core_rewired", "single_four_view_core", "untied4_shared_decoder_core"}


def method_namespace(torch, project_root, config, x, dataset):
    import numpy as np
    factors = SimpleNamespace(**p.factor_definitions(torch, project_root))
    recipe = p.recipe(config, x, dataset)
    env = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional, "random": random, "time": time,
        "hashlib": hashlib, "contextmanager": contextmanager, "SimpleNamespace": SimpleNamespace,
        "NATIVE_MONO": recipe, "RULE": {}, "CONDITIONS": CONDITIONS,
        "factors_module": lambda: factors, "construct": lambda recipe, seed, device: p.constructor(torch, project_root, recipe, seed, device),
        "seeded": lambda seed, device: p.seeded(torch, np, seed, device),
        "native_preprocess": lambda x, edges, K: p.mono_tokens(torch, project_root, x, edges, K=K, preprocessing="pinned_mono_base"),
        "NODES": len(x), "FEATURES": x.shape[1], "CLASSES": config["classes"],
        "BASE_LR": config["learning_rate"], "ATTN_LR": config["learning_rate"]/10}
    p.definitions(project_root, SOURCE["plan"], ("condition",), env)
    transform = p.ProviderConstants({(int,19717): "NODES", (int,500): "FEATURES", (int,3): "CLASSES",
        (float,.005): "BASE_LR", (float,.0005): "ATTN_LR"})
    p.definitions(project_root, SOURCE["method"], ("balanced_hash_partition", "negative_indices", "core_loss", "verify_auxiliary_edges", "DropoutStreams", "Session"), env, transform)
    env["bridge_substitutions"] = transform.changed
    return SimpleNamespace(**env)


def validate_config(study, run):
    c = p.common_config_check(study, run, unit="epochs")
    name = c.get("core_condition")
    if name not in CONDITIONS: raise ValueError("Exact typed CORE condition required")
    bodies, members, owners, forwards, backwards = LAW[name]
    if (c.get("architecture") != "masked_core" or c.get("members") != members or c.get("physical_slow_bodies") != bodies
            or c.get("source_optimizer_count") != owners or c.get("optimizer_calls_per_cycle") != owners
            or c.get("macro_cycle_law") != "one_complete_old_state_all_owner_cycle" or c.get("core_source") != SOURCE
            or c.get("fusion") != "mean_probabilities" or "source_arm" in c
            or c.get("genuine_own_selected_I4") is not False):
        raise ValueError("Exact CORE old-state body/owner and common-bank selector law required")
    return c


def make_adapter(*, torch, project_root, study, run, device, execution_context=None):
    c = validate_config(study, run)
    return make_source_adapter(torch=torch, project_root=project_root, study=study, run=run, device=device,
        execution_context=execution_context)


def make_source_adapter(*, torch, project_root, study, run, device, execution_context=None, normalization=None, factorized_m1=False):
    c = run["config"]
    authority = p.UpdateAuthority(execution_context)
    graph, train, select = p.load_roles(torch, project_root, c["node_inputs"], c["classes"])
    method = method_namespace(torch, project_root, c, graph["x"], study["dataset"])
    auxiliary = None
    if c["core_condition"] == "shared4_core_rewired":
        payload = torch.load(p.bound(project_root, c["auxiliary_graph"]), map_location="cpu", weights_only=True)
        if set(payload) != {"edge_index"}: raise ValueError("Target-free exact rewired graph required")
        auxiliary = payload["edge_index"]
    seed = run["seed"] +1000003*run["body"]
    if normalization is None:
        session = method.Session(c["core_condition"], seed, graph["x"], graph["edge_index"], train["ids"], train["y"],
            device=device, auxiliary_edges=auxiliary)
    else:
        session = normalization(method, seed, {"x": graph["x"], "edge_index": graph["edge_index"], "train_ids": train["ids"], "train_y": train["y"]}, device)
    if factorized_m1:
        p.require(session.spec["members"] == 1 and session.decoder is None and not session.spec["masked"], "Factual source single reference only")
        prior = dict(session.bodies[0].named_parameters())
        sites = session.factors.install_factors(session.bodies[0], 1); session.bodies[0].to(device)
        p.require(sites == 23 and all(dict(session.bodies[0].named_parameters())[n] is v for n,v in prior.items()), "Matched all-map M1 retains native source body")
        session.spec["shared_BE"] = True
        session.optimizers = [session._optimizer(session.bodies[0])]
    session.select_ids, session.select_y = select["ids"].to(device), select["y"].to(device)
    class Container(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.bodies = session.bodies
            if session.decoder is not None: self.decoder = session.decoder
        def forward(self):
            return session.factual_probabilities()[1]
    model = Container()
    def training(adapter, step):
        authority.require("scientific")
        before = dict(session.counters); result = session.train_step()
        delta = {k: session.counters[k]-before[k] for k in before}
        counts = {"optimizer_updates": delta["optimizer_steps"], "source_optimizer_calls": delta["optimizer_steps"],
            "source_optimizer_objects": len(session.optimizers), "macro_cycles": 1 if c["architecture"] == "masked_core" else 0,
            "model_forwards": delta["factual_forwards"]+delta["masked_forwards"], "factual_forwards": delta["factual_forwards"],
            "masked_forwards": delta["masked_forwards"], "route_predictions": delta["factual_forwards"]+delta["masked_forwards"],
            "backward_calls": delta["backwards"], "supervised_examples": (delta["factual_forwards"]+delta["masked_forwards"])*len(train["ids"])}
        return {"objective": result["scalar_objective"], "ce": result["factual_ce"], "regularization": 0.,
            "operation_counts": counts, "objective_components": result}
    def capture():
        import numpy as np
        return {"streams": copy.deepcopy(session.streams.states), "negative_rng": [g.getstate() for g in session.negative_generators],
            "ownership_rng": session.ownership_generator.getstate(), "counters": dict(session.counters),
            "runtime": p.runtime_capture(torch, session.bodies), "numpy_rng": copy.deepcopy(np.random.get_state())}
    def restore(saved):
        import numpy as np
        session.streams.states = copy.deepcopy(saved["streams"])
        for generator, state in zip(session.negative_generators, saved["negative_rng"]): generator.setstate(state)
        session.ownership_generator.setstate(saved["ownership_rng"]); session.counters = dict(saved["counters"])
        p.runtime_restore(torch, session.bodies, saved["runtime"]); np.random.set_state(saved["numpy_rng"])
        if normalization is not None: session.normalization_verify(session)
    adapter = p.source_adapter(torch, study, run, session, model, session.bodies, session.optimizers,
        lambda model, training, step: model(), training, capture, restore)
    adapter.source_session, adapter.source_method = session, method
    return adapter
