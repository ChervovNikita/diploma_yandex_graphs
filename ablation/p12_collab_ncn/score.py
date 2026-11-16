"""Evaluate validation-selected NCN weights after training is frozen."""
import hashlib
import json
from pathlib import Path
import torch
from ablation.common.resumable import atomic_torch_save, atomic_write_json
from .data import load_bundle
from .models import NCN
from .train import predict, metrics

def score_fit(training_dir, out, device="cpu"):
    training_dir, out = Path(training_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    completed = json.loads((training_dir / "done.json").read_text())
    checkpoint = training_dir / "selected_model.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != completed["selected_checkpoint_sha256"]:
        raise ValueError("NCN selected checkpoint changed after completion")
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = artifact["config"]
    expected = json.loads((training_dir / "config.json").read_text())
    if config != expected or any(completed.get(key) != value for key, value in config.items()):
        raise ValueError("NCN completed checkpoint/config identity differs")
    if config["smoke_updates"]:
        raise ValueError("Scientific scoring requires a completed scientific fit")
    bundle = load_bundle(config["data_dir"], torch.device(device), release_test=True)
    if artifact["identity"]["data_sha256"] != bundle.identity:
        raise ValueError("NCN dataset changed before scoring")
    model = NCN(bundle.x.shape[1], config).to(device)
    model.load_state_dict(artifact["model"])
    predictions = dict(artifact["predictions"])
    test_predictions, test_members = predict(
        model, bundle, config["testbs"], test_only=True, return_members=True)
    predictions.update(test_predictions)
    member_predictions = dict(artifact["member_predictions"])
    member_predictions.update(test_members)
    output = dict(artifact, predictions=predictions, member_predictions=member_predictions,
                  test_edge_order_sha256=bundle.test_edge_order_sha256,
                  test_role="selected_checkpoint_once")
    output.pop("model", None)
    atomic_torch_save(output, out / "predictions.pt")
    result = dict(completed, **metrics(predictions), test_role=output["test_role"],
                  test_edge_order_sha256=bundle.test_edge_order_sha256)
    result["test_metric"] = result["test_hits@50"]
    result["test_evaluated"] = True
    reference = training_dir / "reference100_model.pt"
    if reference.exists():
        if hashlib.sha256(reference.read_bytes()).hexdigest() != completed["reference100_checkpoint_sha256"]:
            raise ValueError("NCN reference100 checkpoint changed")
        reference_state = torch.load(reference, map_location="cpu", weights_only=False)
        if reference_state["config"] != config or reference_state["identity"] != artifact["identity"]:
            raise ValueError("NCN reference100 source/config/data differs")
        model.load_state_dict(reference_state["model"])
        reference_predictions = dict(reference_state["predictions"])
        reference_members = dict(reference_state["member_predictions"])
        test_predictions, test_members = predict(
            model, bundle, config["testbs"], test_only=True, return_members=True)
        reference_predictions.update(test_predictions)
        reference_members.update(test_members)
        reference_output = dict(output, predictions=reference_predictions, member_predictions=reference_members,
                                selected_epoch=reference_state["selected_epoch"])
        atomic_torch_save(reference_output, out / "reference100_predictions.pt")
        reference_metrics = metrics(reference_predictions)
        result.update({"reference100_" + key: value for key, value in reference_metrics.items()})
        result["reference100_val_metric"] = reference_metrics["valid_hits@50"]
        result["reference100_test_metric"] = reference_metrics["test_hits@50"]
    atomic_write_json(result, out / "done.json")
    return result
