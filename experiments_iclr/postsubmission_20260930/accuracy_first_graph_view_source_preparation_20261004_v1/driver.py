"""Isolated all-TRAIN eight-pass bank training. CLI is source-only by default."""
import json
import math
import hashlib
from pathlib import Path
import time
from bank import (NATIVE_RECIPE, build_bank, check_gradients, cpu_tree, optimizer_names,
                  restore, snapshot, stage, transition)
from runtime import ROOT, file_descriptor, verify_descriptor
from schedule import (BankClock, CONDITIONS, Selector, TOTAL_UPDATES, pass_plan,
                      schedule_identity, stage_at)
from views import construct_views, digest, verify_bundle


def prepare_views(rt, role, raw_edge_index, view_seed):
    """Admitted future entry: native preprocessing exactly precedes fixed masks."""
    from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
    edge = to_undirected(raw_edge_index)
    edge, _ = remove_self_loops(edge)
    edge, _ = add_self_loops(edge, num_nodes=role.nodes)
    return construct_views(role, tuple(map(tuple, edge.t().cpu().tolist())), view_seed)


def edge_tensors(rt, bundle, device):
    all_edges = dict(bundle["views"], native=bundle["native"])
    return {name: rt.torch.tensor(edges, dtype=rt.torch.long, device=device).t().contiguous()
            for name, edges in all_edges.items()}


def train_step(rt, model, optimizer, x, graphs, train_ids, train_labels, condition, schedule_seed, update):
    if stage(model) != (stage_at(update) == "global"):
        raise ValueError("Actual bank schedule and every core flag must agree")
    torch = rt.torch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    index = torch.tensor(train_ids, dtype=torch.long, device=x.device)
    labels = torch.tensor(train_labels, dtype=torch.long, device=x.device)
    loss_value = 0.0
    for member, view in pass_plan(condition, schedule_seed, update):
        z = model.forward_member(x, graphs[view], member)
        if z.dtype != torch.float32 or tuple(z.shape) != (x.shape[0], 5):
            raise ValueError("Complete FP32 five-class logits required")
        if not torch.isfinite(z).all().item():
            raise ValueError("Nonfinite logits")
        logp = torch.nn.functional.log_softmax(z, dim=1).index_select(0, index)
        scaled_loss = torch.nn.functional.nll_loss(logp, labels) / 4
        if not torch.isfinite(scaled_loss).item():
            raise ValueError("Nonfinite bank loss")
        # All forwards see the same pre-step parameters; no tape retained for the next pass.
        scaled_loss.backward()
        loss_value += float(scaled_loss.detach().cpu())
        del scaled_loss, logp, z
    check_gradients(rt, model)
    optimizer.step()
    optimizer_names(model, optimizer)
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all().item():
            raise ValueError("Nonfinite post-step parameter: " + name)
    for state in optimizer.state.values():
        if any(isinstance(v, torch.Tensor) and not torch.isfinite(v).all().item() for v in state.values()):
            raise ValueError("Nonfinite Adam state")
    return loss_value


def eval_logits(rt, model, x, native):
    model.eval()
    with rt.torch.no_grad():
        logits = model(x, native)
    if logits.dtype != rt.torch.float32 or tuple(logits.shape) != (4, x.shape[0], 5) or \
            not rt.torch.isfinite(logits).all().item():
        raise ValueError("Complete finite FP32 bank logits required")
    return logits


def correct_count(rt, logits, ids, labels):
    prediction = rt.torch.softmax(logits, dim=-1).mean(dim=0).argmax(-1)
    index = rt.torch.tensor(ids, dtype=rt.torch.long, device=logits.device)
    target = rt.torch.tensor(labels, dtype=rt.torch.long, device=logits.device)
    return int((prediction.index_select(0, index) == target).sum().item())


def metrics(rt, logits, ids, labels):
    torch = rt.torch
    index = torch.tensor(ids, dtype=torch.long, device=logits.device)
    targets = torch.tensor(labels, dtype=torch.long, device=logits.device)
    z = logits.index_select(1, index)
    logp = torch.log_softmax(z.to(torch.float64), dim=-1)
    pooled_logp = torch.logsumexp(logp, dim=0) - math.log(4)
    prediction = torch.softmax(z, dim=-1).mean(dim=0).argmax(-1)
    rows = torch.arange(len(ids), device=logits.device)
    f1 = []
    for label in range(5):
        tp = int(((prediction == label) & (targets == label)).sum())
        fp = int(((prediction == label) & (targets != label)).sum())
        fn = int(((prediction != label) & (targets == label)).sum())
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {"accuracy": float((prediction == targets).to(torch.float64).mean()),
            "NLL": float(-pooled_logp[rows, targets].mean()), "macro_F1": sum(f1) / 5,
            "macro_F1_absent_class_rule": "zero when 2TP+FP+FN is zero",
            "member_accuracy": [float((z[m].argmax(-1) == targets).to(torch.float64).mean()) for m in range(4)],
            "member_NLL": [float(-logp[m, rows, targets].mean()) for m in range(4)]}


def feature_identity(x):
    return {"dtype": str(x.dtype), "shape": list(x.shape),
            "logical_sha256": hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}


def verify_freeze(rt, bundle, freeze):
    """Coverage must be saved from actual TRAIN data before a future freeze."""
    verify_bundle(bundle)
    if freeze.get("schema") != "accuracy-first-graph-view-freeze-v1" or \
            freeze.get("source_manifest_sha256") != rt.manifest_sha256 or \
            freeze.get("protocol_sha256") != rt.protocol_sha256 or \
            tuple(freeze.get("conditions", ())) != CONDITIONS:
        raise ValueError("Explicit source/protocol/all-condition freeze binding required")
    descriptor = freeze["official_TRAIN_coverage"]
    actual = verify_descriptor(descriptor)
    saved = json.loads(Path(actual["path"]).read_text())
    expected = dict(bundle["coverage"], coverage_origin="official_TRAIN",
                    scientific_freeze_eligible=bundle["coverage"]["coverage_eligible"])
    if saved != expected or saved["scientific_freeze_eligible"] is not True:
        raise ValueError("Exact eligible official TRAIN coverage must precede scientific freeze")


def save_image(rt, path, image):
    with Path(path).open("xb") as stream:
        rt.torch.save(image, stream)
    return file_descriptor(path)


def run_training(rt, *, execute=False, condition, block_seed, schedule_seed, x, bundle,
                 validation_labels, freeze, output, device="cpu"):
    """External root-admitted call only. No acquisition, TEST labels or tuning."""
    if execute is not True:
        raise RuntimeError("Scientific training disabled by default")
    started = time.perf_counter()
    verify_freeze(rt, bundle, freeze)
    role = bundle["role"]
    if role.nodes != 24492 or len(role.train_ids) != 12246 or len(role.val_ids) != 6123 or \
            len(validation_labels) != len(role.val_ids) or \
            tuple(x.shape) != (24492, 300) or x.dtype != rt.torch.float32 or \
            not rt.torch.isfinite(x).all().item():
        raise ValueError("Complete official Amazon roles/raw FP32 features required")
    if any(type(y) is not int or y not in range(5) for y in validation_labels):
        raise ValueError("Five-class validation targets required")
    if condition not in CONDITIONS:
        raise ValueError("Condition not in frozen comparison")
    block = {"split": role.split, "block_seed": block_seed, "schedule_seed": schedule_seed,
             "view_seed": bundle["coverage"]["view_seed"]}
    planned = json.loads((ROOT / "PROTOCOL.json").read_text())["prospective_blocks"]
    if block not in freeze.get("blocks", ()) or block not in planned or \
            freeze.get("features") != feature_identity(x) or \
            freeze.get("validation_labels_sha256") != digest(validation_labels):
        raise ValueError("Frozen split/seeds/features/validation targets differ")
    output = Path(output)
    output.mkdir(exist_ok=False)
    graphs = edge_tensors(rt, bundle, device)
    x = x.to(device)
    model, optimizer, construction = build_bank(rt, condition, block_seed, device)
    bindings = {"manifest_sha256": rt.manifest_sha256, "protocol_sha256": rt.protocol_sha256,
                "runtime_source_provenance": list(rt.source_provenance),
                "condition": condition, "block_seed": block_seed, "native_recipe": NATIVE_RECIPE,
                "coverage": freeze["official_TRAIN_coverage"], "role": role.identity(),
                "validation_labels_sha256": digest(validation_labels),
                "features": feature_identity(x),
                "schedule": schedule_identity(condition, schedule_seed)}
    clock, selector, best_image, local_image = BankClock(), Selector(), None, None
    selected_logits = None
    with (output / "TRACE.jsonl").open("x") as trace:
        for _ in range(TOTAL_UPDATES):
            if clock.actual_update == 200:
                local_image = best_image
                local_record = save_image(rt, output / "SELECTED_LOCAL.pt", local_image)
                transition_record = transition(rt, model, optimizer, device, local_image, clock)
            update = clock.advance()
            loss = train_step(rt, model, optimizer, x, graphs, role.train_ids, role.train_labels,
                              condition, schedule_seed, update)
            logits = eval_logits(rt, model, x, graphs["native"])
            correct = correct_count(rt, logits, role.val_ids, validation_labels)
            improved = selector.observe(correct, len(role.val_ids), update)
            event = dict(clock.record(), stage=stage_at(update), global_stage=stage(model),
                         TRAIN_CE=loss, VAL_correct=correct, VAL_count=len(role.val_ids),
                         strict_selected=improved, assigned_views=list(pass_plan(condition, schedule_seed, update)[4:]),
                         native_training_passes=4, probe_training_passes=4, native_selection_passes=4)
            if improved:
                selection = dict(clock.record(), global_=stage(model), VAL_correct=correct,
                                 VAL_count=len(role.val_ids), schedule_cursor=update)
                selection["global"] = selection.pop("global_")
                best_image = snapshot(rt, model, optimizer, device, selection, bindings)
                selected_logits = cpu_tree(rt, logits)
                best_image["selected_raw_logits"] = selected_logits
            trace.write(json.dumps(event, sort_keys=True) + "\n")
            trace.flush()
    completed_clock = clock.record()
    restore(rt, model, optimizer, device, best_image, bindings)
    final_logits = eval_logits(rt, model, x, graphs["native"])
    if not rt.torch.equal(final_logits.cpu(), selected_logits) or \
            correct_count(rt, final_logits, role.val_ids, validation_labels) != selector.best:
        raise ValueError("Selected stage/function/pooled selector restoration differs")
    final_record = save_image(rt, output / "SELECTED_FINAL.pt", dict(best_image, selected_raw_logits=selected_logits))
    train_metrics = metrics(rt, final_logits, role.train_ids, role.train_labels)
    val_metrics = metrics(rt, final_logits, role.val_ids, validation_labels)
    construction_record = save_image(rt, output / "CONSTRUCTION.pt", construction)
    result = {"schema": "accuracy-first-graph-view-result-v1", "bindings": bindings,
              "completed_clock": completed_clock, "selection": best_image["selection"],
              "construction": construction_record,
              "local_checkpoint": local_record, "final_checkpoint": final_record,
              "transition": transition_record, "driver_seconds_through_metric_and_checkpoint_outputs": time.perf_counter() - started,
              "training_graph_passes": 8 * TOTAL_UPDATES, "training_backward_passes": 8 * TOTAL_UPDATES,
              "selection_graph_passes": 4 * TOTAL_UPDATES,
              "TRAIN_fit_diagnostics": train_metrics,
              "VAL_exploratory_selected": val_metrics,
              "interpretation": "exploratory validation-associated development; no unseen TEST or novelty claim"}
    with (output / "RESULT.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


if __name__ == "__main__":
    print(json.dumps({"status": "SOURCE_ONLY_NO_EXECUTION", "scientific_training": False,
                      "numerical_imports": False, "protocol": file_descriptor(ROOT / "PROTOCOL.json")}, indent=2))
