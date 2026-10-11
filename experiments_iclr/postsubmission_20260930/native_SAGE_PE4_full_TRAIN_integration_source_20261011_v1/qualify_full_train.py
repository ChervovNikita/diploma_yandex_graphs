"""One actual complete-TRAIN packing qualification; stdlib-only import."""
from pathlib import Path
import json

TOLERANCES = {
    "output": {"rtol": 2e-4, "atol": 3e-5},
    "gradient": {"rtol": 2e-3, "atol": 2e-6},
    "Adam_parameters": {"rtol": 2e-3, "atol": 5e-5},
    "Adam_first_moment": {"rtol": 3e-3, "atol": 2e-7},
    "Adam_second_moment": {"rtol": 4e-3, "atol": 2e-9},
}


def clone(torch, value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: clone(torch, v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone(torch, v) for v in value)
    return value


def exact(torch, left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact(torch, left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(exact(torch, a, b) for a, b in zip(left, right))
    return left == right


def mapping(model, bodies):
    rows = []
    sites = [("input", model.input, [b.input_linear for b in bodies]),
             ("final_norm", model.final_norm, [b.output_normalization for b in bodies]),
             ("output", model.output, [b.output_linear for b in bodies])]
    for l in range(2):
        sites.extend([(f"norms.{l}", model.norms[l], [b.residual_modules[l].normalization for b in bodies]),
                      (f"neighbors.{l}", model.neighbors[l], [b.residual_modules[l].module.conv.lin_l for b in bodies]),
                      (f"roots.{l}", model.roots[l], [b.residual_modules[l].module.conv.lin_r for b in bodies]),
                      (f"first.{l}", model.first[l], [b.residual_modules[l].module.feed_forward_module.linear_1 for b in bodies]),
                      (f"second.{l}", model.second[l], [b.residual_modules[l].module.feed_forward_module.linear_2 for b in bodies])])
    for name, group, native in sites:
        for field in ("weight", "bias"):
            p = getattr(group, field, None)
            if p is not None:
                rows.append((name + "." + field, p, [getattr(b, field) for b in native]))
    if len(rows) != 24 or {n for n, _, _ in rows} != set(dict(model.named_parameters())):
        raise ValueError("Every registered grouped parameter must have a native counterpart")
    if {id(p) for _, _, native in rows for p in native} != {id(p) for b in bodies for p in b.parameters()}:
        raise ValueError("Every original native parameter must appear exactly once")
    return rows


def qualify(torch, routes, packed, Model, set_seed, batch, train_ids, train_y, output, budget):
    """Consume only full factual public X/E and compact580 TRAIN ids/labels."""
    device = batch["x"].device
    if (batch["x"].shape != (11701, 300) or batch["x"].dtype != torch.float32
            or train_ids.shape != (580,) or train_y.shape != (580,)
            or train_ids.dtype != torch.long or train_y.dtype != torch.long
            or len(torch.unique(train_ids)) != 580 or train_ids.min().item() < 0 or train_ids.max().item() >= 11701
            or train_y.min().item() < 0 or train_y.max().item() >= 10
            or batch["edge_index"].ndim != 2 or batch["edge_index"].shape[0] != 2
            or batch["edge_index"].dtype != torch.long
            or batch["edge_index"].min().item() < 0 or batch["edge_index"].max().item() >= 11701):
        raise ValueError("Full original11701x300 graph and580 compact TRAIN rows required")
    budget.check(torch, device)
    native = []
    for m in range(4):
        set_seed(7301 + 1000003 * m)
        native.append(Model("SAGE", 2, 300, 59, 10, 1, 8, "LayerNorm", .2))
    model = packed.pack_native(torch, native)
    model.to(device)
    for b in native:
        b.to(device)
    rows = mapping(model, native)
    packed_params, native_params = list(model.parameters()), [p for b in native for p in b.parameters()]
    pointers = [p.untyped_storage().data_ptr() for p in packed_params + native_params]
    if len(set(pointers)) != len(pointers):
        raise ValueError("Packed tensors and four native bodies must have distinct storages")
    for _, p, originals in rows:
        if p.shape[0] != 4 or not p.is_contiguous():
            raise ValueError("Four disjoint contiguous member-coordinate slices required")
        for m, original in enumerate(originals):
            if not torch.equal(p[m], original):
                raise ValueError("Packing reset or changed native constructor values")
        offsets = [(p[m].storage_offset(), p[m].storage_offset() + p[m].numel()) for m in range(4)]
        if any(offsets[m][1] > offsets[m + 1][0] for m in range(3)):
            raise ValueError("Private member coordinate ranges overlap")
    optim = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0)
    native_optim = torch.optim.AdamW(native_params, lr=.001, weight_decay=0)
    model.check_optimizer_ownership(optim)
    ids = [id(p) for g in native_optim.param_groups for p in g["params"]]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in native_params}:
        raise ValueError("Reference optimizer ownership differs")
    streams = []
    for m in range(4):
        set_seed(7301 + 1000003 * m + 3000007)
        streams.append({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device)})
    native_streams = clone(torch, streams)
    adapters = [routes.NativeModelAdapter(torch, b) for b in native]
    report = {"seed": 7301, "width_contract": packed.width_contract(), "tolerances": TOLERANCES,
              "TRAIN_native_route_trajectories": 0, "discarded_backwards": 0, "discarded_AdamW_calls": 0,
              "grouped_parameter_tensors": len(rows), "reference_parameter_tensors": len(native_params),
              "all_parameter_initial_values_exact": True, "storage_and_ownership": True,
              "steps": [], "maximum_absolute_differences": {}}

    def publish(phase):
        report["phase"] = phase
        path = Path(output) / "QUALIFIER_PROGRESS.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    publish("native_constructor_values_and_ownership_checked")

    def close(name, a, b, kind):
        difference = (a.detach() - b.detach()).abs().max().item()
        report["maximum_absolute_differences"][kind] = max(report["maximum_absolute_differences"].get(kind, 0.), difference)
        if not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item() or not torch.allclose(a, b, **TOLERANCES[kind]):
            raise ValueError(f"Full-TRAIN {kind} differs at {name}; max_abs={difference}")

    def forwards():
        nonlocal streams, native_streams
        model.train()
        publish("packed_full_TRAIN_forward_pending")
        out = model(dict(batch, ids=train_ids), route_scope=lambda m: routes.existing_member_rng_scope(torch, streams, m, device.index or 0))
        report["TRAIN_native_route_trajectories"] += 4
        logits, hidden = [], []
        for m, (b, adapter) in enumerate(zip(native, adapters)):
            b.train(); adapter.check(batch)
            publish(f"native_member{m}_full_TRAIN_forward_pending")
            with routes.existing_member_rng_scope(torch, native_streams, m, device.index or 0):
                l, h = adapter.native(batch)
            logits.append(l[train_ids]); hidden.append(h[train_ids])
            report["TRAIN_native_route_trajectories"] += 1
        reference = torch.stack(logits), torch.stack(hidden)
        if out[0].shape != (4, 580, 10) or out[1].shape != (4, 580, 59):
            raise ValueError("Full-TRAIN collector dimensions differ")
        if not exact(torch, streams, native_streams):
            raise ValueError("Packed and native dropout stream consumption differs")
        close("logits", out[0], reference[0], "output")
        close("hidden", out[1], reference[1], "output")
        publish("paired_full_TRAIN_forward_and_stream_check_complete")
        budget.check(torch, device)
        return out, reference

    for update in range(2):
        optim.zero_grad(); native_optim.zero_grad()
        current, reference = forwards()
        loss = torch.nn.functional.cross_entropy(current[0].flatten(0, 1), train_y.repeat(4))
        native_loss = torch.nn.functional.cross_entropy(reference[0].flatten(0, 1), train_y.repeat(4))
        close("mean own CE", loss, native_loss, "output")
        publish("packed_own_CE_backward_pending")
        loss.backward(); report["discarded_backwards"] += 1
        publish("native_own_CE_backward_pending")
        native_loss.backward(); report["discarded_backwards"] += 1
        for name, p, originals in rows:
            if p.grad is None or not torch.isfinite(p.grad).all().item():
                raise ValueError("A grouped parameter has no finite gradient")
            for m, original in enumerate(originals):
                if original.grad is None or not p.grad[m].ne(0).any().item() or not original.grad.ne(0).any().item():
                    raise ValueError("A whole parameter/member has no live TRAIN gradient")
                close(name + f"[{m}]", p.grad[m], original.grad, "gradient")
        publish("packed_AdamW_pending")
        optim.step(); report["discarded_AdamW_calls"] += 1
        publish("native_AdamW_pending")
        native_optim.step(); report["discarded_AdamW_calls"] += 1
        for name, p, originals in rows:
            for m, original in enumerate(originals):
                close(name + f"[{m}]", p[m], original, "Adam_parameters")
                for key, kind in (("exp_avg", "Adam_first_moment"), ("exp_avg_sq", "Adam_second_moment")):
                    close(name + key + f"[{m}]", optim.state[p][key][m], native_optim.state[original][key], kind)
                if not exact(torch, optim.state[p]["step"], native_optim.state[original]["step"]):
                    raise ValueError("Adam step metadata differs")
        report["steps"].append({"discarded_update": update + 1, "all_parameters_live_finite": True,
                                "native_gradient_Adam_comparison": True, "member_streams_equal": True})
        torch.cuda.synchronize(device); budget.check(torch, device)
        publish(f"discarded_update{update + 1}_complete")
        del current, reference, loss, native_loss

    snapshot = clone(torch, {"packed": model.state_dict(), "native": [b.state_dict() for b in native],
                            "packed_optimizer": optim.state_dict(), "native_optimizer": native_optim.state_dict(),
                            "streams": streams, "native_streams": native_streams})
    state_path = Path(output) / "discarded_TRAIN_packing_state.pt"
    torch.save(snapshot, state_path)
    publish("complete_discarded_TRAIN_state_saved")
    with torch.no_grad():
        expected, native_expected = forwards()
        expected, native_expected = clone(torch, expected), clone(torch, native_expected)
    replay_streams, replay_native_streams = clone(torch, streams), clone(torch, native_streams)
    with torch.no_grad():
        for p in packed_params + native_params:
            p.zero_()
        for optimizer in (optim, native_optim):
            for state in optimizer.state.values():
                for value in state.values():
                    if torch.is_tensor(value):
                        value.zero_()
    streams, native_streams = [], []
    restored = torch.load(state_path, map_location="cpu", weights_only=True)
    model.load_state_dict(restored["packed"], strict=True)
    for b, state in zip(native, restored["native"]):
        b.load_state_dict(state, strict=True)
    optim.load_state_dict(restored["packed_optimizer"]); native_optim.load_state_dict(restored["native_optimizer"])
    streams, native_streams = restored["streams"], restored["native_streams"]
    actual = clone(torch, {"packed": model.state_dict(), "native": [b.state_dict() for b in native],
                          "packed_optimizer": optim.state_dict(), "native_optimizer": native_optim.state_dict(),
                          "streams": streams, "native_streams": native_streams})
    if not exact(torch, snapshot, restored) or not exact(torch, snapshot, actual):
        raise ValueError("Full model/Adam/stream restoration failed")
    publish("full_model_Adam_stream_restore_exact")
    with torch.no_grad():
        replay, native_replay = forwards()
        for i in range(2):
            close("packed restored replay", replay[i].cpu(), expected[i], "output")
            close("native restored replay", native_replay[i].cpu(), native_expected[i], "output")
    if not exact(torch, streams, replay_streams) or not exact(torch, native_streams, replay_native_streams):
        raise ValueError("Persistent streams failed exact restored replay")
    if (report["TRAIN_native_route_trajectories"], report["discarded_backwards"], report["discarded_AdamW_calls"]) != (32, 4, 4):
        raise ValueError("Entire one fixed full-TRAIN qualification must close")
    model.check_optimizer_ownership(optim)
    report.update({"status": "PASS", "strict_full_state_restore": True, "restored_TRAIN_replay": True,
                   "discarded_state_bytes": state_path.stat().st_size,
                   "actual_packed_operation_counts": model.operation_counts,
                   "VALID_payload_deserializations": 0, "VALID_scoring_calls": 0, "TEST_access": False,
                   "scientific_fits": 0, "scientific_fit_admitted": False})
    torch.cuda.synchronize(device); budget.check(torch, device)
    report.update(budget.costs(torch, device))
    publish("PASS_discarded_full_TRAIN_qualification")
    return report
