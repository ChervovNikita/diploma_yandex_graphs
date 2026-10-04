"""Full-TRAIN ordinary native GNNM quality reference. Source-only CLI."""
import json
from pathlib import Path
import time
import bank
import bank_driver
from admission import create_output_directory, reference_device, validate_paired_inputs
from runtime import ROOT, file_descriptor, verify_descriptor
from schedule import BankClock, Selector, stage_at
from views import digest, verify_bundle

BLOCKS = ((0, 17), (1, 29), (2, 43))


def native_train_step(rt, model, optimizer, x, native, train_ids, train_labels, update):
    """Exactly mean4 native CE: four streaming forwards/backwards, one Adam step."""
    if bank.stage(model) != (stage_at(update) == "global"):
        raise ValueError("Actual native GNNM stage differs")
    torch = rt.torch
    ids = torch.tensor(train_ids, dtype=torch.long, device=x.device)
    labels = torch.tensor(train_labels, dtype=torch.long, device=x.device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    value = 0.0
    for member in range(4):
        z = model.forward_member(x, native, member)
        if z.dtype != torch.float32 or tuple(z.shape) != (x.shape[0], 5) or not torch.isfinite(z).all().item():
            raise ValueError("Complete finite FP32 native member logits required")
        logp = torch.nn.functional.log_softmax(z, dim=1).index_select(0, ids)
        loss = torch.nn.functional.nll_loss(logp, labels) / 4
        if not torch.isfinite(loss).item():
            raise ValueError("Nonfinite native GNNM CE")
        loss.backward()
        value += float(loss.detach().cpu())
        del z, logp, loss
    bank.check_gradients(rt, model)
    optimizer.step()
    bank.optimizer_names(model, optimizer)
    if any(not torch.isfinite(p).all().item() for p in model.parameters()) or \
            any(isinstance(v, torch.Tensor) and not torch.isfinite(v).all().item()
                for entry in optimizer.state.values() for v in entry.values()):
        raise ValueError("Nonfinite native GNNM parameter/Adam state")
    return value


def validate_context(rt, context, bundle, x, validation_labels):
    verify_bundle(bundle)
    if context.get("schema") != "accuracy-first-native-gnnm-context-v1" or \
            context.get("manifest_sha256") != rt.manifest_sha256 or \
            context.get("protocol_sha256") != rt.protocol_sha256:
        raise ValueError("Explicit native-reference source/protocol context required")
    if context.get("role") != bundle["role"].identity() or \
            context.get("native_edges_sha256") != bundle["coverage"]["native_edges_sha256"] or \
            context.get("features") != bank_driver.feature_identity(x) or \
            context.get("validation_labels_sha256") != digest(validation_labels):
        raise ValueError("Full-TRAIN/native graph/feature/validation pairing differs")
    paired = json.loads(Path(verify_descriptor(context["paired_protocol"])["path"]).read_text())
    predecessor = json.loads((ROOT / "SOURCE_BINDINGS.json").read_text())["bank_source_manifest"]
    if paired.get("native_gnnm_manifest_sha256") != rt.manifest_sha256 or \
            paired.get("bank_manifest_sha256") != predecessor["sha256"]:
        raise ValueError("Separate paired protocol must bind native-reference and bank sources")
    validate_paired_inputs(paired, bundle["role"].split,
                          {"role": context["role"],
                           "native_edges_sha256": context["native_edges_sha256"],
                           "feature_identity": context["features"],
                           "validation_labels_sha256": context["validation_labels_sha256"]})


def save_image(rt, path, image):
    """Preserve the bound image helper while recording an absolute path."""
    return bank_driver.save_image(rt, Path(path).resolve(), image)


def run_native_reference(rt, *, execute=False, split, x, bundle, validation_labels,
                         context, output, device="cpu"):
    if execute is not True:
        raise RuntimeError("Native GNNM scientific fitting disabled by default")
    started = time.perf_counter()
    device = reference_device(rt, device, bank.resolve_device)
    if split not in dict(BLOCKS):
        raise ValueError("Unplanned fixed block")
    seed = dict(BLOCKS)[split]
    role = bundle["role"]
    if role.split != split or role.nodes != 24492 or len(role.train_ids) != 12246 or \
            len(role.val_ids) != 6123 or len(validation_labels) != len(role.val_ids) or \
            tuple(x.shape) != (24492, 300) or x.dtype != rt.torch.float32 or \
            not rt.torch.isfinite(x).all().item() or \
            any(type(y) is not int or y not in range(5) for y in validation_labels):
        raise ValueError("Complete official all-TRAIN/raw Amazon reference inputs required")
    validate_context(rt, context, bundle, x, validation_labels)
    output = create_output_directory(output)
    x = x.to(device)
    native = rt.torch.tensor(bundle["native"], dtype=rt.torch.long, device=device).t().contiguous()
    model, optimizer, construction = bank.build_bank(rt, "tied_persistent", seed, device)
    bindings = {"manifest_sha256": rt.manifest_sha256, "protocol_sha256": rt.protocol_sha256,
                "reference": "ordinary_native_gnnm_full_TRAIN", "split": split, "seed": seed,
                "builder": "byte-bound bank v2 build_bank(tied_persistent)",
                "role": role.identity(), "native_edges_sha256": context["native_edges_sha256"],
                "features": context["features"], "validation_labels_sha256": context["validation_labels_sha256"],
                "paired_protocol": context["paired_protocol"],
                "source_provenance": list(rt.source_provenance),
                "selected_device": bank.device_provenance(device)}
    clock, selector, selected, selected_logits = BankClock(), Selector(), None, None
    with (output / "TRACE.jsonl").open("x") as trace:
        for _ in range(2700):
            if clock.actual_update == 200:
                local_record = save_image(rt, output / "SELECTED_LOCAL.pt", selected)
                transition_record = bank.transition(rt, model, optimizer, device, selected, clock)
            update = clock.advance()
            loss = native_train_step(rt, model, optimizer, x, native, role.train_ids, role.train_labels, update)
            logits = bank_driver.eval_logits(rt, model, x, native)
            correct = bank_driver.correct_count(rt, logits, role.val_ids, validation_labels)
            improved = selector.observe(correct, len(role.val_ids), update)
            selection = dict(clock.record(), global_=bank.stage(model), VAL_correct=correct, VAL_count=len(role.val_ids))
            selection["global"] = selection.pop("global_")
            if improved:
                selected = bank.snapshot(rt, model, optimizer, device, selection, bindings)
                selected_logits = bank.cpu_tree(rt, logits)
                selected["selected_raw_logits"] = selected_logits
            trace.write(json.dumps(dict(selection, TRAIN_CE=loss, strict_selected=improved,
                                        native_training_passes=4, backward_passes=4, native_selection_passes=4)) + "\n")
            trace.flush()
    bank.restore(rt, model, optimizer, device, selected, bindings)
    final_logits = bank_driver.eval_logits(rt, model, x, native)
    if not rt.torch.equal(final_logits.cpu(), selected_logits) or \
            bank_driver.correct_count(rt, final_logits, role.val_ids, validation_labels) != selector.best:
        raise ValueError("Exact selected native GNNM function/pooled score restoration differs")
    final_record = save_image(rt, output / "SELECTED_FINAL.pt", selected)
    construction_record = save_image(rt, output / "CONSTRUCTION.pt", construction)
    result = {"schema": "ordinary-native-gnnm-reference-result-v1", "bindings": bindings,
              "completed_clock": clock.record(), "selection": selected["selection"],
              "local_checkpoint": local_record, "final_checkpoint": final_record,
              "construction": construction_record, "transition": transition_record,
              "training_passes": 10800, "backward_passes": 10800, "native_selection_passes": 10800,
              "TRAIN_fit_diagnostics": bank_driver.metrics(rt, final_logits, role.train_ids, role.train_labels),
              "VAL_exploratory_selected": bank_driver.metrics(rt, final_logits, role.val_ids, validation_labels)}
    result["driver_seconds_through_checkpoints_and_metrics"] = time.perf_counter() - started
    with (output / "RESULT.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True);stream.write("\n")
    return result


if __name__ == "__main__":
    print(json.dumps({"status": "SOURCE_ONLY", "numerical_imports": False, "scientific_training": False,
                      "reference": "ordinary_native_gnnm_full_TRAIN", "new_bank_conditions": 0}, indent=2))
