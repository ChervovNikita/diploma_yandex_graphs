# SPDX-License-Identifier: Apache-2.0
"""Original V2 fit_one training/selection with separately declared new report policy.

Only restored-output gates are removed; actual finite reconstructed serving scores
and materiality are recorded. Original V2 file/reports are never modified.
"""
import gc
import time
from support import replay_materiality


def bind(original):
    names = ('write_json', 'append_jsonl', 'failure_record', 'seed_all', 'make_native_placed_factory',
             'make_mlp', 'synchronize', 'parameter_counts', 'static_topology_bytes', 'evaluate',
             'capture_rng', 'sha256_file', 'process_peak_rss_bytes')
    globals().update({name: getattr(original, name) for name in names})


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
        record["restore_parameter_buffer_state_exact"] = True
        stage = "reconstructed_train_validation_report"
        scores, reconstructed = evaluate(np, torch, roc_auc_score, restored, data, seed, saved["epoch"])
        counters["restored_report_forwards"] += 1
        record.update(scores=scores, selected_scores_before_reconstruction=saved["scores"],
                      score_source="fresh_reconstructed_selected_serving")
        write_json(folder / "RESULT.json", record)
        parity = {role: float((reconstructed[role] - saved["role_logp"][role]).abs().max().item())
                  for role in ("train", "valid")}
        record["restore_max_abs_role_logp_difference"] = parity
        auroc_delta = {role: abs(scores[role]["auroc"] - saved["scores"][role]["auroc"])
                       for role in ("train", "valid")}
        record["restore_auroc_absolute_difference"] = auroc_delta
        record["restore_role_prediction_changes"] = {role: int((reconstructed[role].argmax(-1) != saved["role_logp"][role].argmax(-1)).sum().item()) for role in ("train", "valid")}
        record["replay_materiality"] = replay_materiality(torch, scores, reconstructed, saved["scores"], saved["role_logp"])
        record["replay_policy"] = "exact_state_finite_serving_recorded_materiality"
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
