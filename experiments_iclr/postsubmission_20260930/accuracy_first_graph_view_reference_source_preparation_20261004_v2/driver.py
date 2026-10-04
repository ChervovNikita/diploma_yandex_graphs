"""Caller-driven native references. CLI performs no numerical or scientific work."""
import hashlib
import json
import math
from pathlib import Path
import time
from admission import create_output_directory, reference_device, validate_paired_inputs
from contract import BLOCKS, Selector, digest, fit_spec, independent_inventory, pass_plan, stage_at
from native_reference import build, evaluate, restore, snapshot, train_step, transition
from runtime import ROOT, descriptor, verify


def feature_identity(x):
    return {"dtype": str(x.dtype), "shape": list(x.shape),
            "logical_sha256": hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}


def correct_count(rt, logits, ids, labels):
    if logits.ndim == 2:
        prediction = logits.argmax(-1)
    else:
        prediction = rt.torch.softmax(logits, dim=-1).mean(0).argmax(-1)
    index = rt.torch.tensor(ids, dtype=rt.torch.long, device=logits.device)
    target = rt.torch.tensor(labels, dtype=rt.torch.long, device=logits.device)
    return int((prediction.index_select(0, index) == target).sum().item())


def metrics(rt, logits, ids, labels):
    torch = rt.torch
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    if logits.dtype != torch.float32 or logits.shape[0] not in (1, 4) or logits.shape[2] != 5:
        raise ValueError("Fixed single/four-member FP32 reporting contract")
    index = torch.tensor(ids, dtype=torch.long, device=logits.device)
    y = torch.tensor(labels, dtype=torch.long, device=logits.device)
    z = logits.index_select(1, index)
    logp = torch.log_softmax(z.to(torch.float64), dim=-1)
    mixture = torch.logsumexp(logp, dim=0) - math.log(z.shape[0])
    prediction = z[0].argmax(-1) if z.shape[0] == 1 else torch.softmax(z, dim=-1).mean(0).argmax(-1)
    row = torch.arange(len(ids), device=logits.device)
    f1 = []
    for label in range(5):
        tp = int(((prediction == label) & (y == label)).sum())
        fp = int(((prediction == label) & (y != label)).sum())
        fn = int(((prediction != label) & (y == label)).sum())
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {"accuracy": float((prediction == y).to(torch.float64).mean()),
            "NLL": float(-mixture[row, y].mean()), "macro_F1": sum(f1) / 5,
            "absent_class_F1": 0.0,
            "member_accuracy": [float((z[m].argmax(-1) == y).to(torch.float64).mean()) for m in range(z.shape[0])],
            "member_NLL": [float(-logp[m, row, y].mean()) for m in range(z.shape[0])]}


def validate_context(rt, context, bundle, x, validation_labels):
    rt.views.verify_bundle(bundle)
    if context.get("schema") != "accuracy-first-reference-context-v1" or \
            context.get("reference_manifest_sha256") != rt.manifest_sha256 or \
            context.get("reference_protocol_sha256") != rt.protocol_sha256:
        raise ValueError("Explicit reference source/protocol context required")
    actual = verify(context["coverage"])
    saved = json.loads(Path(actual["path"]).read_text())
    expected = dict(bundle["coverage"], coverage_origin="official_TRAIN",
                    scientific_freeze_eligible=bundle["coverage"]["coverage_eligible"])
    if saved != expected or saved["scientific_freeze_eligible"] is not True:
        raise ValueError("Exact official full-TRAIN fixed-view coverage required")
    if context.get("features") != feature_identity(x) or \
            context.get("validation_labels_sha256") != digest(validation_labels):
        raise ValueError("Paired feature/validation identity differs")
    paired = json.loads(Path(verify(context["paired_protocol"])["path"]).read_text())
    sources = json.loads((ROOT / "SOURCE_BINDINGS.json").read_text())["files"]
    if paired.get("reference_manifest_sha256") != rt.manifest_sha256 or \
            paired.get("bank_manifest_sha256") != sources["bank_manifest"]["sha256"]:
        raise ValueError("Separate paired protocol must bind both exact sources")
    validate_paired_inputs(paired, bundle["role"].split,
                          {"role": bundle["role"].identity(),
                           "native_edges_sha256": bundle["coverage"]["native_edges_sha256"],
                           "feature_identity": context["features"],
                           "validation_labels_sha256": context["validation_labels_sha256"]})


def save_image(rt, path, image):
    path = Path(path).resolve()
    with path.open("xb") as stream:
        rt.torch.save(image, stream)
    return descriptor(path)


def run_reference(rt, *, execute=False, kind, split, member=None, x, bundle,
                  validation_labels, context, output, device="cpu"):
    if execute is not True:
        raise RuntimeError("Scientific reference fitting disabled by default")
    started = time.perf_counter()
    device = reference_device(rt, device, rt.helpers.resolve_device)
    spec = fit_spec(kind, split, member)
    role = bundle["role"]
    if role.split != split or role.nodes != 24492 or len(role.train_ids) != 12246 or \
            len(role.val_ids) != 6123 or len(validation_labels) != len(role.val_ids) or \
            tuple(x.shape) != (24492, 300) or x.dtype != rt.torch.float32 or \
            not rt.torch.isfinite(x).all().item() or \
            any(type(y) is not int or y not in range(5) for y in validation_labels):
        raise ValueError("Complete raw Amazon/full official TRAIN/VALIDATION contract required")
    validate_context(rt, context, bundle, x, validation_labels)
    if bundle["coverage"]["view_seed"] != dict(BLOCKS)[split]:
        raise ValueError("Predeclared paired view seed required")
    output = create_output_directory(output)
    x = x.to(device)
    edges = dict(bundle["views"], native=bundle["native"])
    graphs = {name: rt.torch.tensor(edge, dtype=rt.torch.long, device=device).t().contiguous()
              for name, edge in edges.items() if name in dict(pass_plan(kind))}
    model, optimizer, construction = build(rt, kind, split, member, device)
    bindings = {"reference_manifest_sha256": rt.manifest_sha256,
                "reference_protocol_sha256": rt.protocol_sha256, "fit": spec,
                "role": role.identity(), "coverage": context["coverage"],
                "features": context["features"], "validation_labels_sha256": context["validation_labels_sha256"],
                "paired_protocol": context["paired_protocol"], "source_provenance": list(rt.provenance),
                "selected_device": rt.helpers.device_provenance(device)}
    selector, best_image, selected_logits = Selector(), None, None
    with (output / "TRACE.jsonl").open("x") as trace:
        for update in range(1, 2701):
            if update == 201:
                local_record = save_image(rt, output / "SELECTED_LOCAL.pt", best_image)
                transition_record = transition(rt, model, optimizer, device, best_image, 200)
            loss = train_step(rt, model, optimizer, x, graphs, role.train_ids, role.train_labels, kind, update)
            logits = evaluate(rt, model, x, graphs["native"])
            correct = correct_count(rt, logits, role.val_ids, validation_labels)
            improved = selector.observe(correct, len(role.val_ids), update)
            selection = {"actual_update": update, "actual_local_updates": min(update, 200),
                         "actual_global_updates": max(0, update - 200), "global": stage_at(update),
                         "VAL_correct": correct, "VAL_count": len(role.val_ids)}
            if improved:
                best_image = snapshot(rt, model, optimizer, device, selection, bindings)
                selected_logits = rt.helpers.cpu_tree(rt, logits)
                best_image["selected_raw_logits"] = selected_logits
            trace.write(json.dumps(dict(selection, TRAIN_CE=loss, strict_selected=improved,
                                        training_passes=len(pass_plan(kind)), backward_passes=len(pass_plan(kind)),
                                        native_selection_passes=1), sort_keys=True) + "\n")
            trace.flush()
    restore(rt, model, optimizer, device, best_image, bindings)
    actual = evaluate(rt, model, x, graphs["native"])
    if not rt.torch.equal(actual.cpu(), selected_logits) or \
            correct_count(rt, actual, role.val_ids, validation_labels) != selector.best:
        raise ValueError("Exact selected native function/accuracy restoration differs")
    final_record = save_image(rt, output / "SELECTED_FINAL.pt", best_image)
    construction_record = save_image(rt, output / "CONSTRUCTION.pt", construction)
    result = {"schema": "graph-view-native-reference-result-v1", "bindings": bindings,
              "completed_actual_updates": 2700, "selection": best_image["selection"],
              "local_checkpoint": local_record, "final_checkpoint": final_record,
              "construction": construction_record, "transition": transition_record,
              "training_passes": 2700 * len(pass_plan(kind)), "backward_passes": 2700 * len(pass_plan(kind)),
              "native_selection_passes": 2700, "TRAIN_fit_diagnostics": metrics(rt, actual, role.train_ids, role.train_labels),
              "VAL_exploratory_selected": metrics(rt, actual, role.val_ids, validation_labels)}
    result["driver_seconds_through_checkpoints_and_metrics"] = time.perf_counter() - started
    with (output / "RESULT.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def pool_conventional(rt, results):
    """Reopen four selected tensors; member0 single is an explicit physical-fit alias."""
    ordered = independent_inventory(results)
    logits = []
    for result in ordered:
        record = verify(result["final_checkpoint"])
        image = rt.torch.load(record["path"], map_location="cpu", weights_only=True)
        if image["bindings"] != result["bindings"] or image["selection"] != result["selection"]:
            raise ValueError("Selected conventional image identity differs")
        value = image["selected_raw_logits"]
        if value.dtype != rt.torch.float32 or tuple(value.shape) != (24492, 5) or \
                not rt.torch.isfinite(value).all().item():
            raise ValueError("Complete conventional raw logits required")
        logits.append(value)
    return {"independent4_raw_logits": rt.torch.stack(logits),
            "independent4_probability": rt.torch.softmax(rt.torch.stack(logits), dim=-1).mean(0),
            "native_single_raw_logits": logits[0], "native_single_alias": ordered[0]["final_checkpoint"],
            "selected_stages_and_updates": [r["selection"] for r in ordered],
            "physical_cost_references": [r["final_checkpoint"] for r in ordered]}


if __name__ == "__main__":
    print(json.dumps({"status": "SOURCE_ONLY", "numerical_imports": False,
                      "scientific_training": False, "reference_families": 3,
                      "new_bank_conditions": 0}, indent=2))
