# SPDX-License-Identifier: Apache-2.0
"""Inactive full-graph native-single/MLP competence screen; no TEST truth input."""
import argparse
import gc
import importlib.metadata
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

from common import (PACKET, append_jsonl, failure_record, load_adapter,
                    process_peak_rss_bytes, read_roles, require_release,
                    sha256_file, write_json)
from native_placement import make_native_placed_factory


def seed_all(np, torch, seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng(np, torch):
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(np, torch, state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@contextmanager
def evaluation_rng(np, torch, seed, epoch):
    """One prospective eval stream, restored afterward; no hidden draw averaging."""
    before = capture_rng(np, torch)
    seed_all(np, torch, seed + 2000003 + epoch)
    try:
        yield
    finally:
        restore_rng(np, torch, before)


def synchronize(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_mlp(torch):
    """Standard same-Torch feature anchor; no topology or native source import."""
    nn, F = torch.nn, torch.nn.functional

    class FeatureMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Linear(10, 64)
            self.lin12 = nn.Linear(64, 64)
            self.lin2 = nn.Linear(64, 2)

        def forward(self, x):
            x = F.dropout(x, p=.4, training=self.training)
            x = F.elu(self.lin1(x))
            x = F.dropout(x, p=.2, training=self.training)
            x = F.elu(self.lin12(x))
            x = F.dropout(x, p=.2, training=self.training)
            return F.log_softmax(self.lin2(x), dim=-1)

    return FeatureMLP()


def metrics(torch, roc_auc_score, logp, labels):
    if not torch.isfinite(logp).all().item():
        raise FloatingPointError("Nonfinite role log probabilities")
    probability = logp.exp()[:, 1]
    return {"auroc": float(roc_auc_score(labels.detach().cpu().numpy(), probability.detach().cpu().numpy())),
            "nll": float(torch.nn.functional.nll_loss(logp, labels).item()),
            "accuracy": float((logp.argmax(-1) == labels).float().mean().item()),
            "brier": float((probability - labels.float()).square().mean().item())}


def evaluate(np, torch, roc_auc_score, model, data, seed, epoch):
    model.eval()
    with evaluation_rng(np, torch, seed, epoch), torch.no_grad():
        logp = model(data["x"])
        if logp.shape != (data["x"].shape[0], 2):
            raise ValueError("Unexpected binary full-graph output shape")
        role_logp = {role: logp[data[role + "_index"]].detach().cpu()
                     for role in ("train", "valid")}
        scores = {role: metrics(torch, roc_auc_score, role_logp[role], data[role + "_y"].cpu())
                  for role in ("train", "valid")}
    return scores, role_logp


def parameter_counts(model):
    parameters = list(model.parameters())  # native members here have no sharing
    return {"active": sum(p.numel() for p in parameters if p.requires_grad),
            "total": sum(p.numel() for p in parameters),
            "unique_parameter_objects": len(parameters),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in parameters)}


def static_topology_bytes(model):
    """Unique storage for source plain graph tensors; parameter bytes separate."""
    builder = getattr(model, "laplacian_builder", None)
    seen, total = set(), 0
    for owner in (model, builder):
        if owner is None:
            continue
        for name in ("edge_index", "full_left_right_idx", "left_right_idx", "vertex_tril_idx",
                     "diag_indices", "tril_indices", "fixed_diag_indices", "fixed_tril_indices", "deg"):
            value = getattr(owner, name, None)
            if value is not None:
                storage = value.untyped_storage()
                identity = (str(value.device), storage.data_ptr())
                if identity not in seen:
                    seen.add(identity)
                    total += storage.nbytes()
    return total


def fit_one(np, torch, roc_auc_score, adapter, protocol, config, seed, family,
            data, role_meta, identity, output, device):
    config_id = config["id"] if family == "native_single" else "feature_mlp_10_64_64_2"
    run_id = family + "__" + config_id + "__seed" + str(seed)
    folder = output / run_id
    folder.mkdir(exist_ok=False)
    record = {"run_id": run_id, "family": family, "config_id": config_id, "seed": seed,
              "status": "started", "experiment_classification": "original_benchmark_exploratory",
              "identity": identity, "selector": protocol["checkpoint_rule"],
              "train_label_opportunity": role_meta["role_counts"]["train"],
              "validation_label_opportunity": role_meta["role_counts"]["valid"],
              "all_training_rows_used_each_epoch": True, "test_truth_present": False,
              "uses_topology": family == "native_single", "full_graph_features_used": True,
              "canonical_directed_entries_actual": role_meta["support_counts"]["canonical_directed"],
              "rng": {"initialization_and_training_seed": seed,
                      "evaluation_seed_rule": "base_seed+2000003+epoch; snapshot/restore training streams",
                      "python_numpy_cpu_cuda_owned": True,
                      "serving_draws": 1}}
    write_json(folder / "RESULT.json", record)
    stage, model, restored, optimizer = "construct", None, None, None
    groups, sheaf, other = None, None, None
    started = time.perf_counter()
    counters = {"train_forwards": 0, "validation_forwards": 0, "restored_report_forwards": 0}
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        seed_all(np, torch, seed)
        if family == "native_single":
            native_args = dict(config["native_args"], graph_size=data["x"].shape[0],
                               input_dim=data["x"].shape[1], output_dim=2, device=str(device))
            factory = make_native_placed_factory(torch, adapter, data["cpu_edge_index"],
                                                 data["edge_index"], native_args)
            model = factory()
            sheaf, other = model.grouped_parameters()
            groups = [{"params": sheaf, "weight_decay": protocol["optimizer"]["sheaf_decay"]},
                      {"params": other, "weight_decay": protocol["optimizer"]["weight_decay"]}]
            record["native_args"] = native_args
            record["topology_construction"] = "original_CPU_constructor_then_explicit_static_tensor_transfer"
        else:
            factory = lambda: make_mlp(torch).to(device)
            model = factory()
            groups = [{"params": list(model.parameters()), "weight_decay": protocol["optimizer"]["weight_decay"]}]
            record["mlp_args"] = {"widths": [10, 64, 64, 2], "activation": "ELU",
                                  "input_dropout": .4, "hidden_dropout": .2}
        optimizer = torch.optim.Adam(groups, lr=protocol["optimizer"]["lr"])
        synchronize(torch, device)
        record["construction_seconds"] = time.perf_counter() - started
        record["construction_cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        record["parameters"] = parameter_counts(model)
        record["static_topology_bytes"] = static_topology_bytes(model)
        record["native_saved_transport_payload_bytes"] = (
            config["native_args"]["layers"] * role_meta["support_counts"]["canonical_undirected"]
            * config["native_args"]["d"] ** 2 * data["x"].element_size()
            if family == "native_single" else 0)
        if family == "feature_mlp" and record["parameters"]["active"] != 4994:
            raise RuntimeError("Feature MLP parameter opportunity differs from fixed architecture")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        best_key, best_epoch, stale, checkpoint_seconds = None, None, 0, 0.0
        checkpoint_path = folder / "SELECTED_STATE.pt"
        for epoch in range(1, protocol["optimizer"]["max_epochs"] + 1):
            record["last_attempted_epoch"] = epoch
            stage = "train_epoch_" + str(epoch)
            tick = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logp = model(data["x"])
            counters["train_forwards"] += 1
            loss = torch.nn.functional.nll_loss(logp[data["train_index"]], data["train_y"])
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Nonfinite TRAIN NLL")
            record["last_train_objective_nll"] = float(loss.item())
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in model.parameters()):
                raise FloatingPointError("Nonfinite native/MLP gradient")
            optimizer.step()
            del logp, loss
            synchronize(torch, device)
            train_seconds = time.perf_counter() - tick
            stage = "validation_epoch_" + str(epoch)
            tick = time.perf_counter()
            scores, role_logp = evaluate(np, torch, roc_auc_score, model, data, seed, epoch)
            counters["validation_forwards"] += 1
            synchronize(torch, device)
            validation_seconds = time.perf_counter() - tick
            key = (scores["valid"]["auroc"], -scores["valid"]["nll"], -epoch)
            selected = best_key is None or key > best_key
            if selected:
                best_key, best_epoch, stale = key, epoch, 0
                record["selected_epoch"] = epoch
                stage = "save_selected_state_" + str(epoch)
                tick = time.perf_counter()
                checkpoint = {"schema": "owned_nsd_train_valid_checkpoint_v1", "identity": identity,
                              "family": family, "config_id": config_id, "seed": seed, "epoch": epoch,
                              "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                              "optimizer_state": optimizer.state_dict(), "training_rng_state": capture_rng(np, torch),
                              "role_logp": role_logp, "scores": scores,
                              "test_truth_saved": False}
                temporary = checkpoint_path.with_suffix(".tmp")
                torch.save(checkpoint, temporary)
                temporary.replace(checkpoint_path)
                checkpoint_seconds += time.perf_counter() - tick
                del checkpoint
            else:
                stale += 1
            append_jsonl(folder / "HISTORY.jsonl", {"epoch": epoch, "train": scores["train"],
                         "valid": scores["valid"], "selected": selected, "stale_epochs": stale,
                         "train_seconds": train_seconds, "validation_seconds": validation_seconds})
            del role_logp
            record["epochs_completed"] = epoch
            if stale >= protocol["optimizer"]["patience"]:
                break
        record["epochs_completed"] = epoch
        record["selected_epoch"] = best_epoch
        record["checkpoint_write_seconds"] = checkpoint_seconds
        record["fit_cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        record["fit_cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        # Release the trained model before reconstructing; report the saved state,
        # not last-epoch predictions. Fresh constructor rebuilds plain graph attrs.
        del model, optimizer
        model, optimizer = None, None
        del groups
        if family == "native_single":
            del sheaf, other
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        stage = "load_owned_selected_checkpoint"
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (saved["identity"] != identity or saved["config_id"] != config_id
                or saved["seed"] != seed or saved["family"] != family):
            raise RuntimeError("Owned checkpoint identity mismatch")
        stage = "fresh_reconstruction"
        restored = factory()
        restored.load_state_dict(saved["state_dict"], strict=True)
        if any(not torch.equal(value.cpu(), saved["state_dict"][name])
               for name, value in restored.state_dict().items()):
            raise RuntimeError("Reconstructed parameter/buffer state differs from selected checkpoint")
        stage = "reconstructed_train_validation_report"
        scores, reconstructed = evaluate(np, torch, roc_auc_score, restored, data, seed, saved["epoch"])
        counters["restored_report_forwards"] += 1
        parity = {role: float((reconstructed[role] - saved["role_logp"][role]).abs().max().item())
                  for role in ("train", "valid")}
        record["restore_max_abs_role_logp_difference"] = parity
        if any(value > 1e-3 for value in parity.values()):
            raise RuntimeError("Gross reconstructed selected-state role logp difference exceeded 1e-3")
        auroc_delta = {role: abs(scores[role]["auroc"] - saved["scores"][role]["auroc"])
                       for role in ("train", "valid")}
        record["restore_auroc_absolute_difference"] = auroc_delta
        record["restore_role_prediction_changes"] = {role: int((reconstructed[role].argmax(-1) != saved["role_logp"][role].argmax(-1)).sum().item()) for role in ("train", "valid")}
        for role in ("train", "valid"):
            if auroc_delta[role] > 1e-3:
                raise RuntimeError("Gross reconstructed selected-state AUROC difference exceeded 1e-3")
        record.update({"status": "complete", "scores": scores,
                       "checkpoint_sha256": sha256_file(checkpoint_path),
                       "reported_state": "fresh_constructor_loaded_selected_checkpoint",
                       "selected_scores_before_reconstruction": saved["scores"],
                       "restore_max_abs_role_logp_difference": parity,
                       "restore_auroc_absolute_difference": auroc_delta,
                       "restore_parameter_buffer_state_exact": True})
    except Exception as error:
        record.update({"status": "failed", "failure": failure_record(error, stage)})
        append_jsonl(output / "FAILURES.jsonl", record)
    finally:
        try:
            synchronize(torch, device)
        except Exception as error:
            record["cost_finalization_failure"] = failure_record(error, "final_cuda_synchronize")
            if record["status"] != "failed":
                record.update({"status": "failed", "failure": record["cost_finalization_failure"]})
                append_jsonl(output / "FAILURES.jsonl", record)
        record["complete_attempt_cuda_peak_allocated_bytes"] = None
        if device.type == "cuda":
            try:
                record["complete_attempt_cuda_peak_allocated_bytes"] = max(
                    record.get("construction_cuda_peak_allocated_bytes") or 0,
                    torch.cuda.max_memory_allocated(device))
            except Exception as error:
                record["cost_query_failure"] = failure_record(error, "final_cuda_peak_query")
        record["complete_attempt_seconds"] = time.perf_counter() - started
        record["process_peak_rss_bytes_cumulative"] = process_peak_rss_bytes()
        record["forward_counts"] = counters
        record["cost_includes_failures_construction_checkpoint_restore"] = True
        write_json(folder / "RESULT.json", record)
        groups, sheaf, other = None, None, None
        del model, optimizer, restored
        gc.collect()
        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception as error:
                record["cleanup_failure"] = failure_record(error, "cuda_empty_cache")
                write_json(folder / "RESULT.json", record)
    return record


def select_architecture(np, protocol, records, train_y):
    complete = [record for record in records if record["status"] == "complete"]
    native = [record for record in complete if record["family"] == "native_single"]
    mlp = [record for record in complete if record["family"] == "feature_mlp"]
    panel_complete = len(native) == 12 and len(mlp) == 3
    result = {"panel_complete": panel_complete, "native_successes": len(native), "mlp_successes": len(mlp),
              "failures": len(records) - len(complete), "competence": "unestablished",
              "architecture_frozen": False, "test_metrics": "unopened and absent",
              "experiment_classification": "original_benchmark_exploratory",
              "fresh_dataset_or_unused_split_confirmation": False,
              "original_paper_scores_modified": False}
    if not panel_complete:
        result["selection_status"] = "incomplete_no_freeze"
        return result
    candidates = []
    for config in protocol["configs"]:
        subset = [record for record in native if record["config_id"] == config["id"]]
        candidates.append({"id": config["id"], "mean_validation_auroc": float(np.mean([r["scores"]["valid"]["auroc"] for r in subset])),
                           "mean_validation_nll": float(np.mean([r["scores"]["valid"]["nll"] for r in subset])),
                           "active_parameters": subset[0]["parameters"]["active"]})
    candidates.sort(key=lambda value: (-value["mean_validation_auroc"], value["mean_validation_nll"],
                                       value["active_parameters"], value["id"]))
    chosen = candidates[0]
    subset = [record for record in native if record["config_id"] == chosen["id"]]
    proportion = float(np.mean(train_y))
    prior_nll = -(proportion * np.log(proportion) + (1 - proportion) * np.log(1 - proportion))
    mlp_mean = float(np.mean([record["scores"]["valid"]["auroc"] for record in mlp]))
    fit_gate = all(r["scores"]["train"]["nll"] < prior_nll and r["scores"]["valid"]["auroc"] > .5 for r in subset)
    reference_gate = chosen["mean_validation_auroc"] >= mlp_mean - .02
    result.update({"selection_status": "selected_validation_only", "candidates": candidates,
                   "selected_configuration": chosen["id"], "selected_configuration_summary": chosen,
                   "train_class_prior_nll": float(prior_nll), "feature_mlp_mean_validation_auroc": mlp_mean,
                   "provisional_fit_gate": fit_gate, "provisional_reference_gate": reference_gate,
                   "competence": "provisional_numeric_gates_passed_requires_curve_review" if fit_gate and reference_gate else "weak_requires_prospective_amendment",
                   "architecture_frozen": False,
                   "freeze_requires_root_curve_stability_review": True})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--release")
    parser.add_argument("--roles")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    protocol = json.loads((PACKET / "RUNNER_PROTOCOL.json").read_text())
    if not args.execute:
        print(json.dumps({"status": "inactive", "action": "screen", "native_fits": 12, "mlp_fits": 3,
                          "configs": [c["id"] for c in protocol["configs"]], "seeds": protocol["seeds"],
                          "classification": "original_benchmark_exploratory", "test_truth_input": False}))
        return
    if not args.roles or not args.output:
        parser.error("--roles and --output are required for execution")
    release, output = require_release(args, "screen", args.roles)
    write_json(output / "ACTIVATION.json", {"release_sha256": sha256_file(args.release),
                "source_seal_sha256": release["source_seal_sha256"], "protocol": protocol})
    records = []
    schedule = [("native_single", config, seed) for config in protocol["configs"] for seed in protocol["seeds"]]
    schedule += [("feature_mlp", {}, seed) for seed in protocol["seeds"]]
    try:
        import numpy as np
        import torch
        from sklearn.metrics import roc_auc_score
        arrays, role_meta, role_metadata_sha = read_roles(np, args.roles)
        if role_meta.get("exposure_classification") != "original_paper_benchmark_exploratory":
            raise RuntimeError("Role exposure classification does not match corrected Tolokers status")
        device = torch.device(args.device)
        if str(device) != args.device or (device.type == "cuda" and device.index is None):
            raise ValueError("Use a final canonical device string, e.g. cpu or cuda:0")
        torch.use_deterministic_algorithms(release["deterministic_algorithms"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        data = {name: torch.from_numpy(value).to(device) for name, value in arrays.items()}
        data["cpu_edge_index"] = torch.from_numpy(arrays["edge_index"])
        versions = {}
        for package in ("torch", "numpy", "scikit-learn", "torch-geometric", "torch-sparse", "torch-scatter", "torch-householder"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        expected = release.get("expected_runtime_versions", {})
        if not expected or any(versions.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Runtime versions do not match root release qualification")
        write_json(output / "RUNTIME_AND_ROLES.json", {"versions": versions, "device": str(device),
                    "cuda": torch.version.cuda, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "allow_tf32": False, "roles": role_meta,
                    "actual_numeric_array_shapes": {name: list(value.shape) for name, value in arrays.items()}})
        identity = {"source_seal_sha256": release["source_seal_sha256"], "role_archive_sha256": sha256_file(args.roles),
                    "role_metadata_sha256": role_metadata_sha, "runner_protocol_sha256": sha256_file(PACKET / "RUNNER_PROTOCOL.json"),
                    "native_commit": protocol["native_commit"], "root_release_sha256": sha256_file(args.release)}
        # Loading adapter.py needs only Torch. Native eager extensions import on
        # factory construction inside each retained fit attempt; MLP remains usable.
        adapter = load_adapter()
        for family, config, seed in schedule:
            if family == "feature_mlp" and "edge_index" in data:
                del data["edge_index"]  # anchor receives features/roles, no topology tensor
                del data["cpu_edge_index"]
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            record = fit_one(np, torch, roc_auc_score, adapter, protocol, config, seed, family,
                             data, role_meta, identity, output, device)
            records.append(record)
            append_jsonl(output / "ALL_FITS.jsonl", record)
        summary = select_architecture(np, protocol, records, arrays["train_y"])
        summary["role_identity"] = identity
        write_json(output / "SCREEN_SUMMARY.json", summary)
        print(json.dumps({"status": summary["selection_status"], "summary": str(output / "SCREEN_SUMMARY.json"),
                          "competence": summary["competence"], "failures": summary["failures"]}))
        if summary["failures"]:
            raise SystemExit(1)
    except Exception as error:
        failure = failure_record(error, "screen_setup_or_summary")
        write_json(output / "SCREEN_FAILURE.json", failure)
        for family, config, seed in schedule[len(records):]:
            record = {"status": "failed_before_fit", "family": family, "config_id": config.get("id", "feature_mlp_10_64_64_2"),
                      "seed": seed, "failure": failure, "source_seal_sha256": release["source_seal_sha256"],
                      "role_archive_sha256": release["input_sha256"], "test_truth_present": False}
            append_jsonl(output / "ALL_FITS.jsonl", record)
            append_jsonl(output / "FAILURES.jsonl", record)
        raise


if __name__ == "__main__":
    main()
