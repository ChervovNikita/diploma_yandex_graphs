"""Source Adam convention: only R/S coupled decay and epsilon divided by4."""
from . import masked_core as core
from . import private_polyformer_source as p

SOURCE = {"path": "pubmed_own4_M_normalization_source_20261010_v1/adapter.py", "bytes": 4002,
    "sha256": "8bc01fde14bdab3280709f0af1c9ade3c121d3afa82070af63cb70d4af0e4500"}


def normalization_namespace(torch, root, config, x):
    native, private = p.coordinate_counts(x.shape[1], config["classes"])
    env = {"ENABLED": False, "M": 4, "CONDITION": "shared4_own_M_normalized", "EXPECTED_TOTAL": native+private,
        "EXPECTED_PRIVATE": private, "BASE_LR": config["learning_rate"], "ATTN_LR": config["learning_rate"]/10}
    transform = p.ProviderConstants({(int,2125647): "EXPECTED_TOTAL", (int,55772): "EXPECTED_PRIVATE",
        (float,.005): "BASE_LR", (float,.0005): "ATTN_LR"})
    p.definitions(root, SOURCE, ("fresh", "verify", "work"), env, transform)
    return env


def validate_config(study, run):
    c = p.common_config_check(study, run, unit="optimizer_updates")
    if (c.get("architecture") != "private_adam" or c.get("core_condition") != "shared4_own" or c.get("members") != 4
            or c.get("physical_slow_bodies") != 1 or c.get("source_optimizer_count") != 1
            or c.get("source_optimizer_calls_per_cycle") != 1 or c.get("core_source") != core.SOURCE
            or c.get("private_adam_source") != SOURCE
            or c.get("private_adam_normalization") != "private_R_S_epsilon_and_weight_decay_divided_by_M"
            or c.get("fusion") != "mean_probabilities" or "source_arm" in c):
        raise ValueError("Exact private epsilon/decay convention; unchanged mean own gradients required")
    return c


def make_adapter(*, torch, project_root, study, run, device, execution_context=None):
    c = validate_config(study, run)
    graph, _, _ = p.load_roles(torch, project_root, c["node_inputs"], c["classes"])
    source = normalization_namespace(torch, project_root, c, graph["x"])
    def fresh(method, seed, data, device):
        session = source["fresh"](method, seed, data, device=device, numerical_admitted=True)
        session.normalization_verify = source["verify"]
        return session
    return core.make_source_adapter(torch=torch, project_root=project_root, study=study, run=run, device=device,
        execution_context=execution_context, normalization=fresh)
