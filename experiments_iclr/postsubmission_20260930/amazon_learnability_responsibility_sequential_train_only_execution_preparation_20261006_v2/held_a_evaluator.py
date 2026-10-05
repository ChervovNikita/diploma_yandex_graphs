"""Disabled separate final A evaluator, admitted only by seven frozen states.

All seven state hashes, exact H16 metadata, complete native parameter schemas,
and all seven served predictions are checked before the A artifact is opened.
Run in a separate process from custody preparation and B-only fitting.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

SOURCE_RELEASED = False
PACKET = Path(__file__).resolve().parent
ARMS = ("live", "uniform", "margins", "graph_free", "permuted", "stop_q")
POOL_MIN = {"accuracy": 0.0025, "NLL": 0.005, "Brier": 0.001}
MEMBER_MAX = {"accuracy": 0.005, "NLL": 0.01, "Brier": 0.002}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _metrics(probabilities, stable_log_probabilities, labels):
    import torch
    count = labels.numel()
    if count == 0:
        return {"n": 0, "accuracy": None, "NLL": None, "Brier": None,
                "FP32_target_zeros": None, "FP32_argmax_ties": None}
    _require(probabilities.dtype == torch.float32 and stable_log_probabilities.dtype == torch.float64,
             "Metric precision convention differs")
    rows = torch.arange(count)
    prediction = probabilities.argmax(dim=-1)  # First class index wins a tie.
    target = torch.nn.functional.one_hot(labels, num_classes=5).double()
    return {"n": count, "accuracy": float((prediction == labels).double().mean()),
            "NLL": float(-stable_log_probabilities[rows, labels].mean()),
            "Brier": float((probabilities.double() - target).square().sum(dim=-1).mean()),
            "FP32_target_zeros": int((probabilities[rows, labels] == 0).sum()),
            "FP32_argmax_ties": int(((probabilities == probabilities.max(dim=-1, keepdim=True).values).
                                     sum(dim=-1) > 1).sum())}


def _score(logits, member_probabilities, pool_probabilities, labels):
    import torch
    log_members = torch.log_softmax(logits.double(), dim=-1)
    log_pool = torch.logsumexp(log_members, dim=0) - math.log(4)
    members = [_metrics(member_probabilities[m], log_members[m], labels) for m in range(4)]
    pool = _metrics(pool_probabilities, log_pool, labels)
    cells = []
    for cls in range(5):
        selected = labels == cls
        cells.append({"class": cls, "n": int(selected.sum()),
                      "pool": _metrics(pool_probabilities[selected], log_pool[selected], labels[selected]),
                      "members": [_metrics(member_probabilities[m, selected], log_members[m, selected],
                                            labels[selected]) for m in range(4)]})
    raw_prediction = logits.argmax(dim=-1)
    unanimous = (raw_prediction == raw_prediction[0]).all(dim=0)
    return {"pool": pool, "members": members, "class_cells": cells,
            "raw_all4_unanimous_count": int(unanimous.sum()),
            "raw_all4_unanimous_wrong_count": int((unanimous & (raw_prediction[0] != labels)).sum()),
            "raw_member_logit_argmax_ties": [int(((logits[m] == logits[m].max(
                dim=-1, keepdim=True).values).sum(dim=-1) > 1).sum()) for m in range(4)]}


def _gate(scores):
    live = scores["live"]
    pool_comparisons = {}
    for name in ("initial", "uniform", "margins", "graph_free", "permuted", "stop_q"):
        reference = scores[name]["pool"]
        gains = {"accuracy": live["pool"]["accuracy"] - reference["accuracy"],
                 "NLL": reference["NLL"] - live["pool"]["NLL"],
                 "Brier": reference["Brier"] - live["pool"]["Brier"]}
        pool_comparisons[name] = {"gains": gains,
                                 "pass": all(gains[key] >= value for key, value in POOL_MIN.items())}
    member_comparisons = []
    for member in range(4):
        comparisons = {}
        for name in ("initial", "uniform"):
            reference = scores[name]["members"][member]
            current = live["members"][member]
            harm = {"accuracy": reference["accuracy"] - current["accuracy"],
                    "NLL": current["NLL"] - reference["NLL"],
                    "Brier": current["Brier"] - reference["Brier"]}
            comparisons[name] = {"harm": harm,
                                 "pass": all(harm[key] <= value for key, value in MEMBER_MAX.items())}
        member_comparisons.append({"member": member, "comparisons": comparisons})
    return {"fixed_metric_gate_pass": all(row["pass"] for row in pool_comparisons.values())
            and all(row["pass"] for member in member_comparisons
                    for row in member["comparisons"].values()),
            "pool_comparisons": pool_comparisons, "member_comparisons": member_comparisons,
            "broader_hypothesis_rejected_by_a_G0_failure": False,
            "support_interpretation": "Report the unchanged-warm response/Q measurements without an empirical "
                "scale threshold or tuning; zero responses are interpretable finite results, not a broad rejection"}


def evaluate_all(project_root, custody_dir, run_dir, output_dir, *, device="cpu"):
    """Exact final full-A scores; no selector, retraining or postprocessor."""
    if not SOURCE_RELEASED:
        raise RuntimeError("Disabled source-only held A evaluator")
    import hashlib
    import importlib.util
    import sys

    bindings = json.loads((PACKET / "SOURCE_BINDINGS.json").read_text(encoding="utf-8"))
    row = bindings["files"]["worker"]
    path = Path(project_root).resolve() / row["path"]
    _require(not path.is_symlink() and path.resolve().is_relative_to(Path(project_root).resolve())
             and hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], "Worker binding differs")
    spec = importlib.util.spec_from_file_location("_amazon_G0_final_evaluator_worker", path)
    worker = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = worker
    spec.loader.exec_module(worker)
    _require(worker.SOURCE_RELEASED is True, "Released reviewed worker required")
    accessor = worker._module(project_root, bindings, "accessor")
    _require(accessor.SOURCE_RELEASED is True, "Released reviewed public+B accessor required")
    run_dir, custody_dir = Path(run_dir).resolve(), Path(custody_dir).resolve()
    complete_path = run_dir / "COMPLETE.json"
    _require(complete_path.is_file() and complete_path.stat().st_mode & 0o222 == 0
             and not (run_dir / "FAILURE.json").exists(), "Incomplete or failed scientific run")
    complete = worker._read(complete_path)
    _require(complete["schema"] == "amazon_G0_seven_states_complete_v1"
             and complete["warm_updates"] == 400 and complete["warm_role"] == "W"
             and complete["diagnostic_extra_member_forwards"] == 16
             and complete["diagnostic_extra_private_gradients"] == 8 and complete["episodes_per_arm"] == 16
             and complete["arms_in_order"] == list(ARMS) and set(complete["endpoints"]) == set(ARMS)
             and complete["A_labels_received"] is False and complete["A_scoring_performed"] is False,
             "Require the common state and all six exact H16 endpoints")
    recipe = complete["recipe"]
    _require(recipe["queue_sha256"] == worker._sha(PACKET / "QUEUE.json")
             and recipe["bindings_sha256"] == worker._sha(PACKET / "SOURCE_BINDINGS.json")
             and recipe["worker_source_sha256"] == bindings["files"]["worker"]["sha256"],
             "Scientific recipe binding differs")
    state_rows = {"initial": complete["common"], **complete["endpoints"]}
    # All immutable state bytes exist before any A label artifact access.
    state_paths = {name: accessor._verify(run_dir, artifact, frozen=True)
                   for name, artifact in state_rows.items()}
    accessor._verify(run_dir, complete["initial_response"], frozen=True)
    custody_path = custody_dir / "CUSTODY.json"
    _require(custody_path.stat().st_mode & 0o222 == 0, "Frozen custody required")
    custody = worker._read(custody_path)
    _require(custody["schema"] == "amazon_G0_TRAIN_label_custody_v2"
             and custody["visibility"] == accessor.VISIBILITY
             and custody["A_labels"]["path"] == "evaluator_a/A_LABELS.npz"
             and custody["public_b_manifest"]["path"] == "public_b/PUBLIC_B_MANIFEST.json"
             and custody["public_b_manifest"]["sha256"] == recipe["public_b_manifest_sha256"],
             "A/public-B custody differs")
    accessor._verify(custody_dir, custody["public_b_manifest"], frozen=True)
    data = accessor.load_public_b(project_root, custody_dir / "public_b", device=device)
    _require(data["provenance"]["roles"]["sha256"] == recipe["roles_sha256"]
             and data["provenance"]["public_graph"]["sha256"] == recipe["public_graph_sha256"]
             and data["provenance"]["preprocessing"]["edge_logical_sha256"] == recipe["native_edge_logical_sha256"],
             "Public role/graph binding differs")
    native = worker._module(project_root, bindings, "native_model")
    boundary = worker._module(project_root, bindings, "boundary_model")
    import numpy as np
    import torch
    family, _ = worker._fresh_family(native, boundary, device)
    expected = family.state_dict()

    def load_state(name):
        payload = torch.load(state_paths[name], map_location="cpu", weights_only=True)
        _require(payload["schema"] == "amazon_G0_frozen_state_v1" and payload["id"] == name
                 and payload["recipe"] == recipe and payload["warm_updates"] == 400
                 and payload["warm_role"] == "W"
                 and payload["episodes"] == (0 if name == "initial" else 16)
                 and payload["global_stage"] is True and payload["eval_mode"] is True
                 and (name == "initial" or payload["common_state"] == complete["common"]),
                 "Checkpoint is not the exact frozen endpoint: " + name)
        values = payload["family_state"]
        _require(set(values) == set(expected), "Incomplete native family bank: " + name)
        for key, value in values.items():
            _require(isinstance(value, torch.Tensor) and value.dtype == torch.float32
                     and value.device.type == "cpu" and value.shape == expected[key].shape
                     and bool(torch.isfinite(value).all()), "Invalid native state tensor: " + key)
        return values

    # Semantic validation of ALL states also precedes opening A.
    for name in state_rows:
        values = load_state(name)
        del values
    output = Path(output_dir).resolve()
    output.mkdir(parents=False, exist_ok=False)
    predictions, prediction_rows = {}, {}
    for name in state_rows:
        family.load_state_dict(load_state(name), strict=True)
        family.set_global_stage(True)
        family.eval()
        with torch.no_grad():
            logits = family(data["features"], data["edge_index"])
            _require(logits.dtype == torch.float32 and logits.shape == (4, 24492, 5)
                     and bool(torch.isfinite(logits).all()), "Native served prediction failed: " + name)
            a_device_logits = logits.index_select(1, data["A_ids"])
            # Native-device FP32 serving first, then save the exact served values.
            device_probabilities = torch.softmax(a_device_logits, dim=-1)
            device_pool = device_probabilities.mean(dim=0)
            a_logits = a_device_logits.cpu().clone()
            probabilities = device_probabilities.cpu().clone()
            pool = device_pool.cpu().clone()
        prediction = {"A_ids": data["A_ids"].cpu().clone(), "native_FP32_logits": a_logits,
                      "served_FP32_member_probabilities": probabilities,
                      "served_FP32_pool_probabilities": pool}
        prediction_rows[name] = worker._save_state(output / (name + "_A_predictions.pt"), prediction)
        predictions[name] = prediction
        del logits, a_device_logits, device_probabilities, device_pool
    _require(set(prediction_rows) == set(state_rows), "Missing served endpoint")
    # This is the FIRST A label artifact hash/read in this evaluator.
    a_path = accessor._verify(custody_dir, custody["A_labels"], frozen=True)
    a_labels = accessor._read_compact(np, a_path, data["A_ids"].cpu().tolist())
    labels = torch.tensor(a_labels.tolist(), dtype=torch.long)
    scores = {name: _score(value["native_FP32_logits"], value["served_FP32_member_probabilities"],
                           value["served_FP32_pool_probabilities"], labels)
              for name, value in predictions.items()}
    result = {"schema": "amazon_G0_fixed_full_A_scores_v1", "n": labels.numel(),
              "complete_run": worker._descriptor(complete_path), "states": state_rows,
              "custody_manifest": worker._descriptor(custody_path), "A_labels": custody["A_labels"],
              "predictions": prediction_rows, "initial_response": complete["initial_response"],
              "preprocessing": data["provenance"]["preprocessing"],
              "scores": scores, "gate": _gate(scores),
              "evaluator_source_sha256": worker._sha(Path(__file__)),
              "A_labels_used_for_fitting_or_selection": False,
              "historical_or_custodian_A_bytes_unread_claim": False,
              "later_whole_bank_ordinary_controls_still_required": True}
    worker._write(output / "SCORES.json", result)
    return worker._descriptor(output / "SCORES.json")
