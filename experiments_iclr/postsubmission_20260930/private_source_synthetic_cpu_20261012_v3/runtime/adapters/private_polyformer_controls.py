"""Ordinary/factorized M1 and actual source boundary GNNM controls."""
from . import private_polyformer_source as p
from . import masked_core as core
from . import polyformer_boundary as boundary

CONDITIONS = ("ordinary_m1", "factorized_m1", "original_boundary", "shared4_own")


def validate_config(study, run):
    c = p.common_config_check(study, run, unit="optimizer_updates")
    arm = c.get("control_condition")
    if arm not in CONDITIONS or c.get("source_optimizer_count") != 1 or c.get("physical_slow_bodies") != 1:
        raise ValueError("Separate one-owner native reference fit required")
    if (c.get("core_condition") != ("shared4_own" if arm == "shared4_own" else "single_native") or c.get("core_source") != core.SOURCE
            or c.get("architecture") != {"ordinary_m1":"single", "factorized_m1":"factorized_single", "original_boundary":"original_gnnm", "shared4_own":"internal_be"}[arm]
            or c.get("members") != (4 if arm in ("original_boundary", "shared4_own") else 1)
            or c.get("fusion") != ("mean_logits" if arm == "original_boundary" else "mean_probabilities")):
        raise ValueError("Exact ordinary, equipped M1 or original source-boundary reference law required")
    return c


def make_adapter(*, torch, project_root, study, run, device, execution_context=None):
    c = validate_config(study, run)
    if c["control_condition"] != "original_boundary":
        return core.make_source_adapter(torch=torch, project_root=project_root, study=study, run=run, device=device,
            execution_context=execution_context, factorized_m1=c["control_condition"] == "factorized_m1")
    graph, train, select = p.load_roles(torch, project_root, c["node_inputs"], c["classes"])
    authority = p.UpdateAuthority(execution_context)
    tokens = [z.to(device) for z in p.mono_tokens(torch, project_root, graph["x"], graph["edge_index"], K=2, preprocessing="author_undirected_then_mono")]
    seed = run["seed"]+1000003*run["body"]
    model = boundary.build_bank(torch, project_root, tokens, native=c["native"], classes=c["classes"], dataset=study["dataset"], members=4,
        initialization_seed=seed, dropout_seeds=[seed+300001+1009*m for m in range(4)])
    optimizer = boundary.grouped_optimizer(torch, model, {**c, "attn_lr": c["learning_rate"]/10, "weight_decay":0., "attn_weight_decay":0.})
    def training(adapter, step):
        authority.require("scientific")
        result = boundary.own_training(adapter, step)
        result["operation_counts"].update(source_optimizer_objects=1, source_optimizer_calls=1)
        return result
    adapter = p.ClassifierAdapter(torch, torch.nn.functional, model, optimizer, logits_fn=lambda model, training, step: model(),
        train_ids=train["ids"].to(device), train_y=train["y"].to(device), select_ids=select["ids"].to(device), select_y=select["y"].to(device),
        regularizer=lambda model: tokens[0].new_zeros(()), fusion="mean_logits", selector={"metric":"accuracy","direction":"max","ties":"first","save_every":1,"patience_steps":5001},
        provenance={"run_id":run["run_id"],"data":study["data"],"source":study["source"],"config":c,
            "actual_model_seed":seed,
            "boundary_source":boundary.BOUNDARY_SOURCE,"replaced_sites":["lin1","lin3"],"same_backbone_source_adaptation":True},
        external_capture=model.capture_runtime_state, external_restore=model.restore_runtime_state, custom_train=training)
    return adapter
