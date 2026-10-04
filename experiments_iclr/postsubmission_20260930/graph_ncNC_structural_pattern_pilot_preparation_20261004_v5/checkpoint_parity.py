"""Prospectively fixed V3/V4 engineering parity; only a released stage runs it.

The saved V3 model class is loaded from its exact immutable source packet.
Reference and candidate trajectories are sequential, never two live GPU graphs.
No result is a scientific donor, a selection candidate, or a project metric.
"""
import importlib.util
from pathlib import Path
import torch
from graph_ops import Graph
from pilot_common import (require, verify_manifest, file_sha,
                          ENGINEERING_SEED, ENGINEERING_SIGN_SEED)
from pilot_model import train_flag
from pilot_state import (snapshot, restore_snapshot, state_digest, cpu_clone,
                         rng_state, rng_digest, restore_rng)
from pattern_model import make_pattern
from pattern_train import batch_forward, finite
from pattern_checks import nested_close

REFERENCE_NAME = "graph_ncNC_structural_pattern_pilot_preparation_20261003_v3"
REFERENCE_SHA = "fa7b2a7a2c6ec83362f3c820fb4f7ad5288e5cc9fb0ee5139614d6690f3f7f89"
PROBE_RECORD_IDS = (0, 2, 3, 5)
PROBE_NEGATIVE_PAIRS = ((7, 9), (8, 10), (9, 11), (10, 12))


def saved_v3_factory():
    root = Path(__file__).resolve().parent.parent / REFERENCE_NAME
    manifest = verify_manifest(root, REFERENCE_SHA)
    pin = next(row for row in manifest["files"] if row["path"] == "pattern_model.py")
    path = root / "pattern_model.py"
    require(file_sha(path) == pin["sha256"], "Saved V3 pattern model differs")
    spec = importlib.util.spec_from_file_location("saved_V3_checkpoint_parity_pattern_model", path)
    require(spec is not None and spec.loader is not None, "Saved V3 loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_pattern, {"manifest_sha256": REFERENCE_SHA,
                                "model_sha256": pin["sha256"],
                                "model_bytes": pin.get("bytes", pin.get("size"))}


def gradient_image(model):
    return {name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for name, parameter in model.named_parameters()}


def forward_image(main, records):
    return cpu_clone({"main": main.detach(), "records": [
        {"query": row["query"], "logits": row["logits"].detach(),
         "rows": row["rows"], "labels": row["labels"],
         "values": {name: value.detach() for name, value in row["values"].items()},
         "t": row["detail"]["t"].detach(),
         "native_q": row["detail"]["native_q"],
         "raw_left": row["detail"]["raw_left"], "raw_right": row["detail"]["raw_right"],
         "raw_scores": [value.detach() for value in row["detail"]["raw_score_tensors"]],
         "neighbors": {name: getattr(row["detail"]["neighbors"], name)
                       for name in ("common", "left", "right")}}
        for row in records]})


def actual_step(model, optimizer, data, teacher, negatives, ids, arm):
    optimizer.zero_grad(set_to_none=True)
    main, records = batch_forward(model, data, teacher, negatives, ids)
    scores = [value for row in records for value in row["detail"]["raw_score_tensors"]]
    isolated = torch.autograd.grad(main, scores, allow_unused=True, retain_graph=True)
    require(all(value is None for value in isolated), "Checkpoint changes target detach boundary")
    auxiliary = sum(row["values"][arm].mean() for row in records)
    total = main + auxiliary
    require(bool(torch.isfinite(total)), "Nonfinite checkpoint-parity objective")
    forward = forward_image(main, records)
    forward["selected_auxiliary"] = auxiliary.detach().cpu().clone()
    forward["total"] = total.detach().cpu().clone()
    support = {name: teacher.support_digest(row["query"], row["detail"]["neighbors"], row["labels"])
               for name, row in zip(("positive", "negative"), records)}
    before_backward = cpu_clone(rng_state())
    total.backward()
    # All scorer recomputation must be RNG-neutral to the canonical caller.
    require(rng_digest(rng_state()) == rng_digest(before_backward), "Checkpoint backward advances caller RNG")
    finite(model, optimizer, gradients=True)
    gradients = gradient_image(model)
    optimizer.step()
    finite(model, optimizer, gradients=False)
    return {"forward": forward, "support": support, "forward_RNG": before_backward,
            "gradients": gradients, "next_state": snapshot(model, optimizer)}


def extra_forward_probes(model, optimizer, initial, data, teacher, negatives, ids):
    # The auxiliary-disabled path is direct, including its original dropout.
    restore_snapshot(model, optimizer, initial)
    train_flag(model, True)
    main, records = batch_forward(model, data, teacher, negatives, ids, auxiliary_grad=False)
    require(all(not value.requires_grad for row in records
                for value in row["detail"]["raw_score_tensors"]), "Disabled auxiliary acquired a gradient")
    disabled = {"forward": forward_image(main, records), "RNG": cpu_clone(rng_state())}
    del main, records
    # Empty captured queries still run every full-node native xlin/decode call.
    restore_snapshot(model, optimizer, initial)
    train_flag(model, True)
    optimizer.zero_grad(set_to_none=True)
    graph = Graph.mask_train_batch(data["pairs"], ids, len(data["x"]))
    h = model.encoder(data["x"], graph)
    query = torch.empty((0, 2), dtype=torch.long, device=h.device)
    logits, detail = model.decoder.pattern_forward(h, graph, query, auxiliary_grad=True)
    before_backward = cpu_clone(rng_state())
    detail["t"].sum().backward()
    require(rng_digest(rng_state()) == rng_digest(before_backward), "Empty-query recomputation advances RNG")
    empty = {"logits": logits.detach().cpu().clone(), "t": detail["t"].detach().cpu().clone(),
             "native_q": detail["native_q"].detach().cpu().clone(), "RNG": before_backward,
             "gradients": gradient_image(model)}
    return {"auxiliary_disabled": disabled, "empty_auxiliary": empty}


def trajectory(factory, mods, data, teacher, device, arm, *, probes):
    model, optimizer = factory(mods, ENGINEERING_SEED, device,
                               engineering_sign_seed=ENGINEERING_SIGN_SEED)
    initial = snapshot(model, optimizer)
    train_flag(model, True)
    ids = torch.tensor(PROBE_RECORD_IDS, device=device, dtype=torch.long)
    negatives = data["pairs"].clone()
    negatives[ids] = torch.tensor(PROBE_NEGATIVE_PAIRS, device=device, dtype=torch.long)
    steps = [actual_step(model, optimizer, data, teacher, negatives, ids, arm) for _ in range(2)]
    extras = extra_forward_probes(model, optimizer, initial, data, teacher, negatives, ids) if probes else None
    # Returned images own CPU copies; no graph/model/Adam GPU object escapes.
    result = {"initial": initial, "steps": steps, "extra_forward_probes": extras}
    del model, optimizer, negatives, ids
    torch.cuda.synchronize(0)
    return result


def checkpoint_parity(mods, data, teacher, device, *, full_graph):
    require(len(data["pairs"]) >= 6 and len(data["x"]) >= 13, "Fixed parity fixture too small")
    if full_graph:
        require(len(data["x"]) == 235868 and len(data["pairs"]) == 1179052,
                "Bounded query probe requires the unchanged complete real graph")
    caller_rng = cpu_clone(rng_state())
    reference_factory, source = saved_v3_factory()
    cases = []
    try:
        for arm in ("J", "F"):
            reference = trajectory(reference_factory, mods, data, teacher, device, arm, probes=(arm == "J"))
            candidate = trajectory(make_pattern, mods, data, teacher, device, arm, probes=(arm == "J"))
            require(state_digest(reference["initial"]) == state_digest(candidate["initial"]),
                    "Checkpoint constructor/initial model/Adam/RNG/flags differ")
            for number, (expected, actual) in enumerate(zip(reference["steps"], candidate["steps"]), start=1):
                nested_close(actual["forward"], expected["forward"], "checkpoint forward " + arm + str(number))
                nested_close(actual["gradients"], expected["gradients"], "checkpoint all gradients " + arm + str(number))
                nested_close(actual["next_state"]["models"], expected["next_state"]["models"], "checkpoint next model " + arm + str(number))
                nested_close(actual["next_state"]["optimizer"], expected["next_state"]["optimizer"], "checkpoint next Adam " + arm + str(number))
                require(actual["support"] == expected["support"], "Checkpoint support/teacher order differs")
                require(state_digest(actual["forward_RNG"]) == state_digest(expected["forward_RNG"])
                        and state_digest(actual["next_state"]["rng"]) == state_digest(expected["next_state"]["rng"])
                        and actual["next_state"]["flags"] == expected["next_state"]["flags"],
                        "Checkpoint forward/next-step RNG or flags differ")
            if arm == "J":
                for probe_name in ("auxiliary_disabled", "empty_auxiliary"):
                    expected = reference["extra_forward_probes"][probe_name]
                    actual = candidate["extra_forward_probes"][probe_name]
                    # The qualified nested arithmetic comparator handles
                    # tensors; RNG also contains a NumPy array, hashed exactly.
                    require(state_digest(actual["RNG"]) == state_digest(expected["RNG"]),
                            "Disabled/empty scorer RNG differs")
                    nested_close({key: value for key, value in actual.items() if key != "RNG"},
                                 {key: value for key, value in expected.items() if key != "RNG"},
                                 "disabled and empty scorer probe " + probe_name)
            cases.append({"arm": arm, "reference_updates": 2, "candidate_updates": 2,
                          "forward_support_all_gradients_next_model_Adam_equal": True,
                          "forward_backward_next_step_RNG_flags_exact": True})
            del reference, candidate
    finally:
        restore_rng(caller_rng)
    require(rng_digest(rng_state()) == rng_digest(caller_rng), "Parity probe donated caller RNG")
    return {"schema": "ncnc-pattern-activation-checkpoint-parity-v1", "status": "PASS",
            "reference_manifest_sha256": source["manifest_sha256"], "reference_model": source,
            "complete_real_graph": full_graph, "query_records": list(PROBE_RECORD_IDS),
            "negative_pairs_for_these_records": [list(pair) for pair in PROBE_NEGATIVE_PAIRS],
            "actual_updates_each_variant_per_arm": 2, "audit_optimizer_updates": 8,
            "audit_member_trajectory_updates": 32, "cases": cases,
            "arithmetic_tolerance": "unchanged atol=rtol=128*finfo(dtype).eps",
            "forward_gradient_next_step_arithmetic_equal": True, "RNG_flags_exact": True,
            "main_target_detach_boundary_verified": True,
            "auxiliary_disabled_and_empty_query_forward_backward_probes": True,
            "reference_and_candidate_GPU_graphs_sequential": True,
            "caller_RNG_restored": True, "state_donor": False, "project_metric_computed": False,
            "native_full_batch_or_existing_complete_resource_work_reduced": False}
