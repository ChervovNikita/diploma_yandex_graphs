# SPDX-License-Identifier: Apache-2.0
"""Inactive two-checkpoint predictive-materiality report. Never trains/selects."""
import argparse
import ast
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import random
import socket
import time
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
RUNNER = PHASE / "private_sheaf_train_valid_runner_20261009_v2"
FIXED = (("d4_f16_L2", 2207, 405), ("d4_f16_L4", 2207, 372))
PROVIDERS = ("torch", "numpy", "scikit-learn", "torch-geometric",
             "torch-sparse", "torch-scatter", "torch-householder")
HELPERS = ("seed_all", "capture_rng", "restore_rng", "evaluation_rng",
           "synchronize", "metrics", "evaluate", "parameter_counts",
           "static_topology_bytes")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def local_source_gate():
    seal = json.loads((HERE / "SEAL.json").read_text())
    require(seal["execution_enabled"] is False and
            sha(HERE / "MANIFEST.json") == seal["manifest_sha256"], "Diagnostic source seal")
    for row in json.loads((HERE / "MANIFEST.json").read_text())["files"]:
        path = (HERE / row["path"]).resolve(strict=True)
        require(path.is_relative_to(HERE) and path.stat().st_size == row["bytes"]
                and sha(path) == row["sha256"], "Changed diagnostic payload")
    bindings = json.loads((HERE / "SOURCE_BINDINGS.json").read_text())
    for row in bindings["inputs"].values():
        path = (PHASE / row["path"]).resolve(strict=True)
        require(path.is_relative_to(PHASE) and path.stat().st_size == row["bytes"]
                and sha(path) == row["sha256"], "Changed original helper source")
    return seal, bindings


def activation(args):
    require(args.execute and args.release, "Inactive: explicit root release required")
    seal, bindings = local_source_gate()
    release = json.loads(Path(args.release).read_text())
    require(release.get("enabled") is True and release.get("release_owner") == "root"
            and release.get("action") == "native15_saved_state_predictive_materiality",
            "Wrong root action")
    require(release.get("diagnostic_manifest_sha256") == seal["manifest_sha256"],
            "Release/source mismatch")
    require(release.get("all15_terminal_reaped") is True
            and release.get("original_family_status") == "incomplete_no_freeze"
            and release.get("original_failed_attempts_preserved") is True
            and release.get("architecture_selection_allowed") is False
            and release.get("test_truth_excluded") is True
            and release.get("maximum_fresh_model_forwards") == 2,
            "Closed-family scope/forward custody required")
    require(release.get("server_hostname") == socket.gethostname(), "Server-only host mismatch")
    require(release.get("runner_source_seal_sha256") == bindings["inputs"]["runner_seal"]["sha256"]
            and release.get("runner_protocol_sha256") == bindings["inputs"]["protocol"]["sha256"],
            "Original runner source/protocol mismatch")
    receipts = release.get("receipt_bindings", {})
    require(set(receipts) == {"closed_native15", "owner_release", "runtime_qualification",
                              "role_custody", "original_screen_release"}, "Exact custody receipts required")
    for row in receipts.values():
        require(sha(row["path"]) == row["sha256"], "Root receipt hash mismatch")
    original_release = json.loads(Path(receipts["original_screen_release"]["path"]).read_text())
    require(original_release.get("enabled") is True and original_release.get("action") == "screen",
            "Original screen release required")
    roles = Path(release["roles_archive"]).resolve(strict=True)
    require(sha(roles) == release["role_archive_sha256"]
            and sha(roles.parent / "ROLE.json") == release["role_metadata_sha256"],
            "Exact original role custody required")
    require(original_release.get("input_sha256") == release["role_archive_sha256"]
            and original_release.get("role_metadata_sha256") == release["role_metadata_sha256"]
            and original_release.get("source_seal_sha256") == release["runner_source_seal_sha256"],
            "Role/source mismatch with original screen")
    require(release["device"] == original_release["device"]
            and type(release["deterministic_algorithms"]) is bool
            and release["deterministic_algorithms"] == original_release["deterministic_algorithms"]
            and release.get("allow_tf32") is False, "Original numerical policy required")
    require(set(release["expected_runtime_versions"]) == set(PROVIDERS), "All actual provider versions required")
    rows = release["checkpoints"]
    require([(r["config_id"], r["seed"], r["selected_epoch"]) for r in rows] == list(FIXED),
            "Only the two fixed owned selected checkpoints are admitted")
    root = Path(release["original_screen_output_directory"]).resolve(strict=True)
    require(str(root) == original_release["output_directory"], "Original screen directory must remain exact")
    for row in rows:
        run_id = "native_single__" + row["config_id"] + "__seed" + str(row["seed"])
        checkpoint = root / run_id / "SELECTED_STATE.pt"
        result = root / run_id / "RESULT.json"
        require(str(checkpoint) == row["path"] and sha(checkpoint) == row["sha256"]
                and str(result) == row["original_result_path"]
                and sha(result) == row["original_result_sha256"], "Owned attempt/file custody mismatch")
        original = json.loads(result.read_text())
        require(original["status"] == "failed" and original["config_id"] == row["config_id"]
                and original["seed"] == row["seed"] and original["family"] == "native_single"
                and original["epochs_completed"] == 500
                and original["selected_epoch"] == row["selected_epoch"], "Original failed status must persist")
    output = Path(release["output_directory"]).resolve()
    require(str(output) == release["output_directory"] and not output.exists()
            and not output.is_relative_to(root) and not output.is_relative_to(HERE)
            and not output.is_relative_to(RUNNER)
            and not output.is_relative_to(PHASE / "private_sheaf_native_source_design_20261009_v1"),
            "A fresh separate server-only output is required")
    output.mkdir(parents=True, exist_ok=False)
    return release, original_release, roles, output


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_helpers():
    """Reuse unchanged helper ASTs; fit/optimizer/selection/main are excluded."""
    tree = ast.parse((RUNNER / "baseline_runner.py").read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    require(all(name in functions for name in HELPERS), "Original helper definition missing")
    module = ast.fix_missing_locations(ast.Module(body=[functions[name] for name in HELPERS], type_ignores=[]))
    namespace = {"random": random, "contextmanager": contextmanager}
    exec(compile(module, str(RUNNER / "baseline_runner.py") + ":unchanged-materiality-helpers", "exec"), namespace)
    return namespace


def inspect_one(np, torch, roc_auc_score, common, placed, helpers, adapter, protocol,
                row, data, identity, output, device):
    run_id = "native_single__" + row["config_id"] + "__seed" + str(row["seed"])
    folder = output / run_id
    folder.mkdir(exist_ok=False)
    record = {"run_id": run_id, "config_id": row["config_id"], "seed": row["seed"],
              "selected_epoch": row["selected_epoch"], "original_attempt_status": "failed",
              "diagnostic_status": "started", "identity": identity,
              "checkpoint_sha256": row["sha256"], "original_result_sha256": row["original_result_sha256"],
              "family_status": "incomplete_no_freeze", "architecture_selection_allowed": False,
              "training_forwards": 0, "backwards": 0, "optimizer_steps": 0,
              "fresh_model_forward_attempts": 0, "fresh_model_forwards_completed": 0,
              "retry_or_reselection": False, "raw_outputs_server_only": True,
              "delta_convention": "replayed minus original; absolute element changes are reported separately"}
    model, saved, replay = None, None, None
    start, cpu_start, stage = time.perf_counter(), time.process_time(), "load_owned_checkpoint"
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        tick = time.perf_counter()
        saved = torch.load(row["path"], map_location="cpu", weights_only=True)
        record["checkpoint_load_seconds"] = time.perf_counter() - tick
        require(saved["schema"] == "owned_nsd_train_valid_checkpoint_v1"
                and saved["identity"] == identity["original_checkpoint_identity"]
                and saved["family"] == "native_single" and saved["config_id"] == row["config_id"]
                and saved["seed"] == row["seed"] and saved["epoch"] == row["selected_epoch"]
                and saved["test_truth_saved"] is False, "Exact selected checkpoint identity required")
        stage, tick = "fresh_native_constructor", time.perf_counter()
        helpers["seed_all"](np, torch, row["seed"])
        config = next(c for c in protocol["configs"] if c["id"] == row["config_id"])
        native_args = dict(config["native_args"], graph_size=data["x"].shape[0],
                           input_dim=data["x"].shape[1], output_dim=2, device=str(device))
        factory = placed.make_native_placed_factory(torch, adapter, data["cpu_edge_index"],
                                                    data["edge_index"], native_args)
        model = factory()
        helpers["synchronize"](torch, device)
        record["native_args"] = native_args
        record["construction_seconds"] = time.perf_counter() - tick
        record["parameters"] = helpers["parameter_counts"](model)
        record["static_topology_bytes"] = helpers["static_topology_bytes"](model)
        stage, tick = "strict_selected_state_load", time.perf_counter()
        model.load_state_dict(saved["state_dict"], strict=True)
        require(all(torch.equal(value.cpu(), saved["state_dict"][name])
                    for name, value in model.state_dict().items()), "Exact selected parameter/buffer restoration required")
        record["parameter_buffer_restore_exact"] = True
        helpers["synchronize"](torch, device)
        record["strict_state_load_seconds"] = time.perf_counter() - tick
        stage, tick = "one_fresh_full_graph_evaluation", time.perf_counter()
        record["fresh_model_forward_attempts"] = 1
        write(folder / "MATERIALITY.json", record)
        scores, replay = helpers["evaluate"](np, torch, roc_auc_score, model, data,
                                               row["seed"], row["selected_epoch"])
        record["fresh_model_forwards_completed"] = 1
        helpers["synchronize"](torch, device)
        record["evaluation_including_role_metrics_seconds"] = time.perf_counter() - tick
        stage, tick = "original_vs_replayed_materiality", time.perf_counter()
        record["roles"] = {}
        for role in ("train", "valid"):
            original = saved["role_logp"][role]
            fresh = replay[role]
            require(original.dtype == torch.float32 and original.shape == fresh.shape
                    and original.shape == (data[role + "_y"].numel(), 2)
                    and bool(torch.isfinite(original).all()), "Finite exact original role output required")
            labels = data[role + "_y"].cpu()
            original_metrics = helpers["metrics"](torch, roc_auc_score, original, labels)
            logp_change = (fresh - original).abs()
            probability_change = (fresh.exp() - original.exp()).abs()
            record["roles"][role] = {
                "rows": original.shape[0], "classes": 2,
                "original_metrics_recomputed_from_saved_role_logp": original_metrics,
                "original_selected_metrics_recorded": saved["scores"][role],
                "replayed_metrics": scores[role],
                "signed_metric_differences": {key: scores[role][key] - original_metrics[key]
                                              for key in ("auroc", "nll", "accuracy", "brier")},
                "saved_score_vs_recomputed_original_differences": {
                    key: original_metrics[key] - saved["scores"][role][key]
                    for key in ("auroc", "nll", "accuracy", "brier")},
                "max_abs_logp_change": float(logp_change.max().item()),
                "mean_abs_logp_change": float(logp_change.mean().item()),
                "max_abs_probability_change": float(probability_change.max().item()),
                "mean_abs_probability_change": float(probability_change.mean().item()),
                "element_change_population": "all role rows and both native float32 class columns",
                "decision_change_count": int((fresh.argmax(-1) != original.argmax(-1)).sum().item())}
        record["comparison_seconds"] = time.perf_counter() - tick
        stage, tick = "save_server_only_raw_role_outputs", time.perf_counter()
        raw = folder / "ROLE_OUTPUTS.pt"
        torch.save({"schema": "native15_predictive_materiality_raw_roles_v1",
                    "identity": identity, "run_id": run_id,
                    "original_role_logp": saved["role_logp"], "replayed_role_logp": replay,
                    "server_only": True, "labels_saved": False, "test_truth_saved": False}, raw)
        record["raw_output_sha256"] = sha(raw)
        record["raw_output_bytes"] = raw.stat().st_size
        record["raw_save_seconds"] = time.perf_counter() - tick
        record["diagnostic_status"] = "materiality_reported"
    except Exception as error:
        record["diagnostic_status"] = "unavailable_failure_retained"
        record["failure"] = common.failure_record(error, stage)
    finally:
        try:
            helpers["synchronize"](torch, device)
            record["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            record["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        except Exception as error:
            record["cost_query_failure"] = common.failure_record(error, "final_cost_query")
        record["attempt_wall_seconds"] = time.perf_counter() - start
        record["attempt_cpu_seconds"] = time.process_time() - cpu_start
        record["process_peak_rss_bytes_cumulative"] = common.process_peak_rss_bytes()
        record["cost_includes_failures_constructor_load_evaluation_compare_and_raw_storage"] = True
        write(folder / "MATERIALITY.json", record)
        del model, saved, replay
        try:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as error:
            record["cleanup_failure"] = common.failure_record(error, "final_cleanup")
            record["diagnostic_status"] = "unavailable_failure_retained"
            write(folder / "MATERIALITY.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--release")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "inactive_source_only", "fixed_checkpoints": list(FIXED),
                          "maximum_fresh_model_forwards": 2, "training_or_selection": False}))
        return
    start, cpu_start = time.perf_counter(), time.process_time()
    release, original_release, roles, output = activation(args)
    summary = {"diagnostic_status": "started", "original_family_status": "incomplete_no_freeze",
               "original_failed_attempts_preserved": True, "architecture_selection_allowed": False,
               "thresholds_or_acceptance_gate_added": False, "training_forwards": 0,
               "backwards": 0, "optimizer_steps": 0, "maximum_fresh_model_forwards": 2,
               "root_release_sha256": sha(args.release), "receipt_bindings": release["receipt_bindings"],
               "records": []}
    write(output / "ACTIVATION.json", {"release": release, "source_only_packet_activation": True})
    try:
        import numpy as np
        import torch
        from sklearn.metrics import roc_auc_score
        versions = {name: importlib.metadata.version(name) for name in PROVIDERS}
        require(versions == release["expected_runtime_versions"], "Actual provider version mismatch")
        require(all(versions.get(k) == v for k, v in original_release["expected_runtime_versions"].items()),
                "Runtime differs from original qualified screen")
        require(torch.version.cuda == release["expected_torch_cuda"]
                and platform.python_version() == release["expected_python_version"], "Actual CUDA/Python mismatch")
        device = torch.device(release["device"])
        require(str(device) == release["device"] and (device.type != "cuda" or device.index is not None),
                "Canonical original device required")
        torch.use_deterministic_algorithms(release["deterministic_algorithms"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        common = module_at("native15_materiality_original_common", RUNNER / "common.py")
        require(common.verify_source_seal() == release["runner_source_seal_sha256"], "Original full source seal mismatch")
        placed = module_at("native15_materiality_original_placement", RUNNER / "native_placement.py")
        helpers = exact_helpers()
        tick = time.perf_counter()
        arrays, metadata, metadata_sha = common.read_roles(np, roles)
        require(metadata_sha == release["role_metadata_sha256"], "Role metadata mismatch")
        data = {name: torch.from_numpy(value).to(device) for name, value in arrays.items()}
        data["cpu_edge_index"] = torch.from_numpy(arrays["edge_index"])
        helpers["synchronize"](torch, device)
        summary["role_load_transfer_seconds"] = time.perf_counter() - tick
        protocol = json.loads((RUNNER / "RUNNER_PROTOCOL.json").read_text())
        original_identity = {"source_seal_sha256": release["runner_source_seal_sha256"],
                             "role_archive_sha256": release["role_archive_sha256"],
                             "role_metadata_sha256": metadata_sha,
                             "runner_protocol_sha256": release["runner_protocol_sha256"],
                             "native_commit": protocol["native_commit"],
                             "root_release_sha256": release["receipt_bindings"]["original_screen_release"]["sha256"]}
        identity = {"original_checkpoint_identity": original_identity,
                    "diagnostic_manifest_sha256": release["diagnostic_manifest_sha256"],
                    "root_diagnostic_release_sha256": sha(args.release)}
        summary["runtime_and_identity"] = {"identity": identity, "versions": versions,
                                          "python": platform.python_version(), "torch_cuda": torch.version.cuda,
                                          "device": str(device), "hostname": socket.gethostname(),
                                          "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                          "allow_tf32": False, "roles": metadata,
                                          "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                          "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
        adapter = common.load_adapter()
        for row in release["checkpoints"]:
            summary["records"].append(inspect_one(np, torch, roc_auc_score, common, placed,
                                                helpers, adapter, protocol, row, data, identity, output, device))
        summary["diagnostic_status"] = "materiality_reported" if all(
            r["diagnostic_status"] == "materiality_reported" for r in summary["records"]) else "unavailable_failure_retained"
    except Exception as error:
        summary["diagnostic_status"] = "unavailable_failure_retained"
        summary["failure"] = {"exception_type": type(error).__name__, "message": str(error)}
    finally:
        recorded = {r["config_id"] for r in summary["records"]}
        for config_id, seed, epoch in FIXED:
            if config_id not in recorded:
                path = output / ("native_single__" + config_id + "__seed" + str(seed)) / "MATERIALITY.json"
                if path.exists():
                    retained = json.loads(path.read_text())
                    retained["diagnostic_status"] = "unavailable_failure_retained"
                    retained["outer_failure"] = summary.get("failure")
                    summary["records"].append(retained)
                else:
                    summary["records"].append({"config_id": config_id, "seed": seed,
                        "selected_epoch": epoch, "original_attempt_status": "failed",
                        "diagnostic_status": "not_executed_setup_failure",
                        "fresh_model_forward_attempts": 0, "fresh_model_forwards_completed": 0,
                        "training_forwards": 0, "backwards": 0, "optimizer_steps": 0,
                        "failure": summary.get("failure")})
        summary["fresh_model_forward_attempts"] = sum(r["fresh_model_forward_attempts"] for r in summary["records"])
        summary["fresh_model_forwards_completed"] = sum(r["fresh_model_forwards_completed"] for r in summary["records"])
        summary["complete_wall_seconds_including_activation_hashing"] = time.perf_counter() - start
        summary["complete_cpu_seconds"] = time.process_time() - cpu_start
        write(output / "MATERIALITY_SUMMARY.json", summary)
    print(json.dumps({"status": summary["diagnostic_status"], "family_status": "incomplete_no_freeze",
                      "forward_attempts": summary["fresh_model_forward_attempts"], "raw_outputs_server_only": True}))
    if summary["diagnostic_status"] != "materiality_reported":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
