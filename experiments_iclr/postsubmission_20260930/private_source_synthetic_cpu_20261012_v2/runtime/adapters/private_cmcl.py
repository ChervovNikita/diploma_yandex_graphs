"""Exact pinned CMCL Session.step with explicit new-provider source bridge."""
import ast
from contextlib import contextmanager
import copy
import math
import random
from types import SimpleNamespace

from . import private_polyformer_source as p

SOURCE = {
    "session": {"path": "private_cmcl_polyformer_source_20261010_v1/session.py", "bytes": 19193, "sha256": "4e1dd3a94ca5a3c0a1ac5045fc7323b8a1607323fe713eb4b25e44167ea45087"},
    "objective": {"path": "private_cmcl_polyformer_source_20261010_v1/objective.py", "bytes": 2536, "sha256": "8c1ec6c9e95b0696610b3435cd1a687950b65327c2d18b14fcea0f2ef77d2506"},
    "plan": {"path": "private_cmcl_polyformer_source_20261010_v1/plan.py", "bytes": 1232, "sha256": "f01309a1ac41bf5581abe64fad46fc9cdd154cd4ca180ed8af7322595c40aa7c"},
    "native": {"path": "private_cmcl_polyformer_source_20261010_v1/native.py", "bytes": 5764, "sha256": "d04a00421d416fdf352c3787cdd11b4c1b29790d8ea4b1275a4559a0be9a230e"},
    "state": {"path": "sehgnn_IMDB_TRAIN_VALID_native_reference_source_20261009_v2/state_helpers.py", "bytes": 2420, "sha256": "00059ffe54571f59bb17084c64e0dcd3f50d3b4f2b31de785e4306beb989b512"}}
CONDITIONS = ("own_floor", "private_cmcl", "all_block_cmcl", "vanilla_cmcl", "private_uniform", "private_constant_credit")
COUNTS = {name: {"model_forwards": 8 if name in CONDITIONS[1:4] else 4,
    "backward_calls": 8 if name in ("private_cmcl", "private_uniform", "private_constant_credit") else 4,
    "optimizer_updates": 1} for name in CONDITIONS}


def build_session(torch, project_root, *, name, seed, graph, train, select, config, dataset,
                  execution_context=None, engineering=False):
    import numpy as np
    caps = p.UpdateAuthority(execution_context, engineering=engineering)
    classes, nodes, features = config["classes"], len(graph["x"]), graph["x"].shape[1]
    recipe = p.recipe(config, graph["x"], dataset)
    device = torch.device(config.get("device", "cpu"))
    if device.type == "cuda": device = torch.device("cuda", 0 if device.index is None else device.index)
    factors = SimpleNamespace(**p.factor_definitions(torch, project_root))
    state = p.source_namespace(torch, project_root, SOURCE["state"])
    native_count, private_count = p.coordinate_counts(features, classes)
    def build(seed, device, caps):
        model = p.constructor(torch, project_root, recipe, seed, device)
        prior = dict(model.named_parameters())
        sites = factors.install_factors(model, 4); model.to(device)
        named = dict(model.named_parameters())
        shared = tuple(v for n, v in named.items() if not n.endswith((".r", ".s")))
        private = tuple(v for n, v in named.items() if n.endswith((".r", ".s")))
        p.require(sites == 23 and len(private) == 46 and all(named[n] is v for n, v in prior.items()), "All23 source affine sites and retained native objects required")
        p.require(sum(v.numel() for v in shared) == native_count and sum(v.numel() for v in private) == private_count,
            "Provider-shaped exact native/private coordinate inventory required")
        p.require(all(torch.equal(v, torch.ones_like(v)) for v in private), "Original unit factors required")
        return model, factors, shared, private
    env = {"torch": torch, "np": np, "math": math, "contextmanager": contextmanager,
        "CLOSED": caps, "require": p.require, "MEMBERS": 4, "CLASSES": classes, "OWNER_K": 3, "BETA": .75,
        "LAMBDA": 1., "CONDITIONS": CONDITIONS, "SEEDS": (seed,), "NATIVE": recipe, "build": build,
        "state_helpers": lambda caps: state, "source_gate": lambda identity, caps: p.require(identity == config["identity"], "Exact bridge input identity required"),
        "seeded": lambda seed, device, caps: p.seeded(torch, np, seed, device), "NODES": nodes, "FEATURES": features,
        "PROVIDER_SPLIT": config["identity"]["split"], "BASE_LR": config["learning_rate"], "ATTN_LR": config["learning_rate"] / 10}
    p.definitions(project_root, SOURCE["plan"], ("condition",), env)
    p.definitions(project_root, SOURCE["objective"], ("node_terms", "assignment", "member_terms"), env)
    # Three class guard, provider row/feature guard, identity split and declared
    # LR substitutions only. Session.step and objective functions are exact.
    transform = p.ProviderConstants({(int, 19717): "NODES", (int, 500): "FEATURES", (int, 190111): "PROVIDER_SPLIT",
        (int, 3): "CLASSES", (float, .005): "BASE_LR", (float, .0005): "ATTN_LR"})
    p.definitions(project_root, SOURCE["session"], ("gradient_stamp", "Session"), env, transform)
    identity = config["identity"]
    tokens = [z.to(device) for z in p.mono_tokens(torch, project_root, graph["x"], graph["edge_index"], K=2, preprocessing="author_undirected_then_mono")]
    inputs = SimpleNamespace(split_seed=identity["split"], counts={"nodes": nodes, "classes": classes},
        input_custody_sha256=identity["input_custody_sha256"], factual=SimpleNamespace(list_mat=tokens),
        train=SimpleNamespace(ids=train["ids"].to(device), targets=train["y"].to(device), custody_sha256=identity["TRAIN_custody_sha256"]),
        validation=SimpleNamespace(ids=select["ids"].to(device), targets=select["y"].to(device), custody_sha256=identity["VALID_custody_sha256"]))
    result = env["Session"](name, seed, inputs, device, p.SourceCosts(), identity, caps)
    result.select_ids, result.select_y = inputs.validation.ids, inputs.validation.targets
    result.bridge_substitutions = transform.changed
    return result


def validate_config(study, run):
    c = p.common_config_check(study, run, unit="optimizer_updates")
    name = c.get("cmcl_condition")
    if (name not in CONDITIONS or c.get("architecture") != "private_cmcl" or c.get("members") != 4
            or c.get("physical_slow_bodies") != 1 or c.get("source_optimizer_count") != 1
            or c.get("source_optimizer_calls_per_cycle") != 1 or c.get("cmcl_source") != SOURCE
            or c.get("fusion") != "mean_probabilities" or "source_arm" in c):
        raise ValueError("Exact typed private CMCL source/condition/one-owner law required")
    return c


def make_adapter(*, torch, project_root, study, run, device, execution_context=None):
    c = validate_config(study, run)
    graph, train, select = p.load_roles(torch, project_root, c["node_inputs"], c["classes"])
    config = {**c, "device": str(device)}
    session = build_session(torch, project_root, name=c["cmcl_condition"], seed=run["seed"]+1000003*run["body"],
        graph=graph, train=train, select=select, config=config, dataset=study["dataset"], execution_context=execution_context)
    def logits(model, training, step):
        bank = torch.stack([session.forward(m) for m in range(4)])
        session.counters["serving_fullgraph_forwards"] += 4
        return bank
    def training(adapter, step):
        session.caps.require("scientific")
        before = dict(session.counters); result = session.step()
        counts = dict(COUNTS[c["cmcl_condition"]], supervised_examples=4*len(train["ids"]), route_predictions=COUNTS[c["cmcl_condition"]]["model_forwards"])
        counts.update(source_optimizer_objects=1, source_optimizer_calls=1)
        expected = {"optimizer_steps": 1, "updates": 1, "gradient_fullgraph_forwards": 4,
            "score_fullgraph_forwards": 4 if session.spec["assignment"] else 0}
        p.require(all(session.counters[k]-before[k] == value for k, value in expected.items()), "Observed exact CMCL source work counts required")
        ce = sum(row["own_CE"] for row in result["member_terms"])/4
        return {"objective": sum((row["floor"] or 0)+(row["auxiliary"] or 0) for row in result["member_terms"]),
            "ce": ce, "regularization": 0., "operation_counts": counts,
            "objective_components": result, "gradient_field_is_joint_scalar": not session.spec["private_auxiliary_only"]}
    def capture():
        return {"streams": copy.deepcopy(session.streams), "counters": dict(session.counters),
            "runtime": p.runtime_capture(torch, [session.model]), "numpy_rng": session.state.capture_rng(session.np, torch)["numpy"]}
    def restore(saved):
        session.streams = copy.deepcopy(saved["streams"]); session.counters = dict(saved["counters"])
        p.runtime_restore(torch, [session.model], saved["runtime"])
        n = saved["numpy_rng"]; session.np.random.set_state((n[0], session.np.asarray(n[1], dtype=session.np.uint32), n[2], n[3], n[4]))
        session.verify_optimizer()
    adapter = p.source_adapter(torch, study, run, session, session.model, [session.model], [session.optimizer], logits, training, capture, restore)
    adapter.source_session = session
    return adapter
