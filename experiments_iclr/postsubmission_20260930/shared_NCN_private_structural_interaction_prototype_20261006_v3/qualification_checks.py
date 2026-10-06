"""Small numerical branch/ownership/capacity checks; invoked only by TRAIN gate."""
import copy
import hashlib
import json
import time


def branch_gate(torch, model, topology, device, arm):
    kind = "count" if arm == "count_only" else "blind" if arm == "structure_blind" else "structural"
    neighbors = [{2, 3, 4, 5}, {2, 3, 4, 5}, {0, 1, 3}, {0, 1, 2}, {0, 1, 5}, {0, 1, 4},
                 set(), set(), {10, 11}, {10, 11}, {8, 9}, {8, 9}]
    info = topology.enumerate_queries(neighbors, [(0, 1), (6, 7), (8, 9)], kind)
    empty = topology.enumerate_queries([set(), set()], [(0, 1)], kind)
    h = torch.linspace(-.5, .5, 12*256, device=device).reshape(12, 256)
    certificates = []
    for original in model.branches:
        # Isolate probes and native Adam history from the official first fixture.
        branch = copy.deepcopy(original); members = list(range(branch.r.shape[0])); work = {}
        def charge(key, value): work[key] = work.get(key, 0)+value
        optimizer = torch.optim.Adam(branch.parameters(), lr=.001, betas=(.9, .999), eps=1e-8,
                                     weight_decay=0., foreach=False, fused=False)
        optimizer.zero_grad(set_to_none=True); out = branch(h, info, members, charge)
        if not bool((out == 0).all()): raise ValueError("Initial residual is not exact zero")
        out.sum().backward()
        if not all(p.grad is not None and bool((p.grad == 0).all()) for p in branch.psi.parameters()) or branch.r.grad is None or not bool((branch.r.grad == 0).all()):
            raise ValueError("Initial Psi/gate derivatives must be connected exact zeros")
        if branch.a.grad is None or not bool((branch.a.grad != 0).any()): raise ValueError("Active startup readout derivative absent")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True); moved = branch(h, info, members, charge)
        active_query_ids = set(info["query_ids"])
        inactive_query_ids = [q for q in range(len(info["queries"])) if q not in active_query_ids]
        if inactive_query_ids and not bool((moved[inactive_query_ids] == 0).all()):
            raise ValueError("Queries without this control's enumerated witnesses acquired a residual")
        moved.sum().backward()
        if not any(p.grad is not None and bool((p.grad != 0).any()) for p in branch.psi.parameters()) or branch.r.grad is None or not bool((branch.r.grad != 0).any()):
            raise ValueError("Psi/gate did not activate after actual Adam readout update")
        if kind == "count" and not bool((branch.psi[0].weight.grad[:, 5:] == 0).all()):
            raise ValueError("Count-only padded input columns received gradients")
        optimizer.step()
        previous = {id(p): {key: value.detach().clone() for key, value in optimizer.state[p].items()} for p in branch.parameters()}
        optimizer.zero_grad(set_to_none=True); empty_out = branch(h, empty, members, charge)
        empty_out.sum().backward()
        if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in branch.parameters()):
            raise ValueError("Empty-CN route disconnects declared parameters")
        if kind != "count" and not all(bool((p.grad == 0).all()) for p in branch.parameters()):
            raise ValueError("Empty-witness anchor derivatives are not exact zeros")
        optimizer.step()
        if kind != "count":
            for p in branch.parameters():
                state, old = optimizer.state[p], previous[id(p)]
                if float(state["step"]) != 3 or not torch.allclose(state["exp_avg"], old["exp_avg"]*.9, rtol=2e-7, atol=1e-12) or not torch.equal(state["exp_avg_sq"], old["exp_avg_sq"]*.999):
                    raise ValueError("Native Adam empty-witness history did not decay with zero gradients")
        certificates.append({"startup_zero_output_and_Psi_gate_gradients": True, "startup_readout_derivative_nonzero": True,
            "actual_Adam_readout_then_Psi_gate_activation": True, "empty_CN_gradients_connected": True,
            "empty_witness_native_Adam_moment_decay": kind != "count", "count_empty_CN_can_learn_intercept": kind == "count",
            "count_padded_columns_inactive": kind == "count", "synthetic_work": work})
        del branch, optimizer, previous, out, moved, empty_out
    return certificates


def edge_product_certificate(torch, branch, topology, device):
    """Configured MLP certificate, not a trained result or full-GNN theorem."""
    network = copy.deepcopy(branch)
    with torch.no_grad():
        for p in network.parameters(): p.zero_()
        network.psi[0].weight[0, 256] = 1
        network.psi[0].weight[1, 256] = -1
        network.psi[2].weight[0, 0] = 1
        network.psi[2].weight[0, 1] = -1
        network.a[:, 0] = 1
    h = torch.zeros(6, 256, device=device); h[2:, 0] = h.new_tensor([1., 1., -1., -1.])
    ga = [{2, 3, 4, 5}, {2, 3, 4, 5}, {0, 1, 3}, {0, 1, 2}, {0, 1, 5}, {0, 1, 4}]
    gb = [{2, 3, 4, 5}, {2, 3, 4, 5}, {0, 1, 4}, {0, 1, 5}, {0, 1, 2}, {0, 1, 3}]
    work = {}
    def charge(key, value): work[key] = work.get(key, 0)+value
    with torch.no_grad():
        a = network(h, topology.enumerate_queries(ga, [(0, 1), (1, 0)], "structural"), [0], charge)
        b = network(h, topology.enumerate_queries(gb, [(0, 1), (1, 0)], "structural"), [0], charge)
    if not torch.equal(a, h.new_tensor([[1.], [1.]])) or not torch.equal(b, h.new_tensor([[-1.], [-1.]])):
        raise ValueError("Actual symmetric nonseparable MLP product certificate failed")
    return {"passed": True, "matching_residuals": [1., -1.], "gate": .5,
            "scope": "configured_shared_MLP_conditional_fixed_H; no_prediction_or_novelty_claim", "synthetic_work": work}


def draw_identity(negative, order, episodes):
    value = {"negative_bank": negative.tolist(), "outer_order": order,
             "episodes": [{"endpoint": row[0], "matched_random": row[1], "kept_positive_ids": row[2]} for row in episodes]}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def witness_coverage(topology, geometry, train, negative, order, episodes, nodes):
    """Paid CPU enumeration of the complete frozen first TRAIN cycle.

    All graphs are the actual current union-masked supports. Complete native
    negatives are inspected under every such support, with no query restoration.
    Output contains counts, work and draw identity, never quality scores.
    """
    started = time.monotonic(); positives, negatives = train.tolist(), negative.tolist()
    roles, rows, work = {}, [], {}; outer_seen = set(); bank_nonempty_ids = set()
    for index, (endpoint, _, kept, _) in enumerate(episodes):
        neighbors = geometry.neighbors([positives[i] for i in kept], nodes)
        selections = {"outer_positive": [positives[i] for i in endpoint["outer_pos_ids"]],
                      "outer_negative": [negatives[i] for i in endpoint["outer_neg_ids"]],
                      "inner_positive": [positives[i] for route in endpoint["inner"] for i in route["pos_ids"]],
                      "inner_negative": [negatives[i] for route in endpoint["inner"] for i in route["neg_ids"]],
                      "complete_native_negative_bank": negatives}
        outer_seen.update(endpoint["outer_pos_ids"]); episode_counts = {}
        for role, queries in selections.items():
            info = topology.enumerate_queries(neighbors, queries, "count")
            stats = info["statistics"]
            counts = {"queries": len(stats), "CN_zero_queries": sum(s[0] == 0 for s in stats),
                      "CN_one_queries": sum(s[0] == 1 for s in stats), "CN_at_least_two_queries": sum(s[0] >= 2 for s in stats),
                      "internal_edge_nonempty_queries": sum(s[1] > 0 for s in stats),
                      "internal_edge_instances": int(sum(s[1] for s in stats)),
                      "maximum_CN": int(max((s[0] for s in stats), default=0)),
                      "maximum_internal_edges": int(max((s[1] for s in stats), default=0))}
            if role == "complete_native_negative_bank": bank_nonempty_ids.update(i for i, s in enumerate(stats) if s[1] > 0)
            accumulated = roles.setdefault(role, {key: 0 for key in counts})
            for key, value in counts.items():
                accumulated[key] = max(accumulated[key], value) if key.startswith("maximum_") else accumulated[key]+value
            for key, value in info["work"].items(): work[key] = work.get(key, 0)+value
            episode_counts[role] = counts
        rows.append({"episode": index+1, "current_support_directed_entries": 2*len(kept), "counts": episode_counts})
    if outer_seen != set(range(len(positives))) or roles["outer_positive"]["queries"] != len(positives):
        raise ValueError("Complete witness coverage missed a frozen outer TRAIN positive")
    for values in roles.values():
        values["internal_edge_nonempty_fraction"] = values["internal_edge_nonempty_queries"]/values["queries"] if values["queries"] else None
    return {"scope": "paid_TRAIN_only_complete_frozen_cycle_witness_coverage", "cycle": 1,
            "episodes": len(episodes), "outer_positive_unique_ids_seen": len(outer_seen),
            "native_negative_bank_entries": len(negatives), "native_negative_ids_with_any_current_support_witness": len(bank_nonempty_ids),
            "roles": roles, "per_episode_counts": rows, "topology_work": work,
            "inclusive_CPU_seconds": time.monotonic()-started, "draw_identity_sha256": draw_identity(negative, order, episodes),
            "all_supports_current_union_masked": True, "target_restoration": False, "quality_scoring": False,
            "interpretation": "Internal witnesses need K4 minus query edge. Sparsity can make this branch inactive; this does not diagnose exact-statistic collisions or common STRICT errors."}


def parameter_audit(torch, model, shared, private, partition, input_dim):
    names = dict(model.named_parameters())
    if any(p.dtype != torch.float32 or not p.requires_grad for p in names.values()):
        raise ValueError("Full learned native/structural FP32 blocks required")
    counts = {"shared_role": sum(names[n].numel() for n in shared), "private_role": sum(names[n].numel() for n in private)}
    delta = (input_dim-3703)*256
    expected = {
        "structural_private": (1460322+delta, 25752), "structure_blind": (1460322+delta, 25752),
        "count_only": (1460322+delta, 25752), "structural_tied": (1460322+delta, 25644),
        "informed_single": (999521+delta, 463398), "informed_independent4": (3998084+4*delta, 1853592)}[partition["arm"]]
    if tuple(counts[key] for key in ("shared_role", "private_role")) != expected:
        raise ValueError("Actual control parameter count differs from reviewed geometry")
    independent = partition["arm"] == "informed_independent4"
    if len(model.branches) != (4 if independent else 1): raise ValueError("Incorrect Psi sharing boundary")
    psi_parameters = [p for branch in model.branches for p in branch.psi.parameters()]
    if len({p.data_ptr() for p in psi_parameters}) != len(psi_parameters): raise ValueError("Independent Psi storage aliases")
    if independent:
        for block in ("encoder", "predictor"):
            params = [p for route in model.base.routes for p in getattr(route, block).parameters()]
            if len({p.data_ptr() for p in params}) != len(params): raise ValueError("Independent native block storage aliases")
    if not independent and hasattr(model, "routes"): raise ValueError("Shared control acquired independent route scaling")
    counts.update(total=sum(counts.values()), Psi_networks=len(model.branches),
                  Psi_parameters=sum(p.numel() for p in psi_parameters),
                  new_private_parameters=sum(branch.r.numel()+branch.a.numel() for branch in model.branches),
                  active_message_input_columns=5 if partition["arm"] == "count_only" else 768,
                  effective_capacity_equal=False if partition["arm"] in ("count_only", "structural_tied") else None,
                  topology_information=partition["structural"]["kind"], independent_storage_verified=independent)
    return counts
