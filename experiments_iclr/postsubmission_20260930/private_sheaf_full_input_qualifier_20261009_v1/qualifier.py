# SPDX-License-Identifier: Apache-2.0
"""Inactive authentic full-graph NSD engineering qualifier; TRAIN truth only."""
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

PACKET = Path(__file__).resolve().parent
RUNNER = PACKET.parent / "private_sheaf_train_valid_runner_20261009_v1"
ROLE_KEYS = {"x", "edge_index", "train_index", "train_y", "valid_index", "valid_y"}
TRAIN_KEYS = ("x", "edge_index", "train_index", "train_y")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_stdlib_helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_chain():
    own = json.loads((PACKET / "SOURCE_SEAL.json").read_text())
    for relative, expected in own["sha256"].items():
        if sha256(PACKET / relative) != expected:
            raise RuntimeError("Qualifier source seal mismatch: " + relative)
    common = load_stdlib_helper("nsd_qualifier_common", RUNNER / "common.py")
    runner_seal = common.verify_source_seal()
    return common, sha256(PACKET / "SOURCE_SEAL.json"), runner_seal


def activate(release_path, roles, output, device, overlay, direct_mode):
    common, own_seal, runner_seal = verify_chain()
    release = json.loads(Path(release_path).read_text())
    checks = (release.get("enabled") is True, release.get("release_owner") == "root",
              release.get("action") == "qualify_full_input", release.get("prior_exposure_audit_complete") is True,
              release.get("authentic_train_valid_roles_authorized") is True,
              release.get("runtime_overlay_import_authorized") is True,
              release.get("source_seal_sha256") == own_seal,
              release.get("runner_source_seal_sha256") == runner_seal,
              release.get("input_sha256") == sha256(roles),
              release.get("role_metadata_sha256") == sha256(Path(roles).parent / "ROLE.json"),
              release.get("device") == device, release.get("direct_index_check") == direct_mode,
              release.get("dependency_overlay") == (str(Path(overlay).resolve()) if overlay else None),
              type(release.get("deterministic_algorithms")) is bool,
              release.get("allow_tf32") is False)
    if not all(checks):
        raise RuntimeError("Root release/source/input/runtime placement binding mismatch")
    if direct_mode == "skip" and not release.get("direct_index_check_skip_reason"):
        raise RuntimeError("Skipping the one direct check requires a recorded root reason")
    target = Path(output).resolve()
    if target.exists() or release.get("output_directory") != str(target):
        raise RuntimeError("Qualification requires the fresh root-bound output directory")
    target.mkdir(parents=True, exist_ok=False)
    return common, release, target


def read_train_input(np, common, archive):
    """Authenticate six-field role archive; decode only graph/features/TRAIN."""
    meta = json.loads((Path(archive).parent / "ROLE.json").read_text())
    if (meta.get("schema") != "nsd_train_valid_roles_v1" or meta.get("dataset") != "tolokers"
            or meta.get("official_split") != 0 or meta.get("official_split_count") != 10
            or meta.get("exposure_classification") != "original_paper_benchmark_exploratory"
            or meta.get("archive_sha256") != sha256(archive)):
        raise ValueError("Authentic corrected Tolokers role identity mismatch")
    with np.load(archive, allow_pickle=False) as loaded:
        if set(loaded.files) != ROLE_KEYS:
            raise ValueError("Unknown role keys; TEST truth or full y are not accepted")
        arrays = {name: loaded[name] for name in TRAIN_KEYS}
    for name, value in arrays.items():
        if value.dtype.hasobject or value.dtype.kind not in "fiub":
            raise ValueError("Non-numeric TRAIN input: " + name)
        if common.array_hash(value) != meta["array_sha256"][name]:
            raise ValueError("Authenticated TRAIN input array hash mismatch: " + name)
    x, edge = arrays["x"], arrays["edge_index"]
    if x.dtype != np.float32 or x.shape != (11758, 10) or not np.isfinite(x).all():
        raise ValueError("Expected complete finite official float32 feature matrix")
    n = x.shape[0]
    if edge.dtype != np.int64 or edge.ndim != 2 or edge.shape[0] != 2 or not edge.shape[1]:
        raise ValueError("Expected actual canonical int64 directed support")
    if edge.min() < 0 or edge.max() >= n or np.any(edge[0] == edge[1]):
        raise ValueError("Malformed canonical native support")
    keys = edge[0] * n + edge[1]
    if np.any(keys[1:] <= keys[:-1]) or not np.array_equal(keys, np.sort(edge[1] * n + edge[0])):
        raise ValueError("Support must be sorted unique and exactly reverse-paired")
    if (meta["support_counts"]["canonical_directed"] != edge.shape[1]
            or meta["support_counts"]["canonical_undirected"] * 2 != edge.shape[1]
            or meta.get("support_convention") != "pyg_2.6.1_to_undirected_row_coalesce"):
        raise ValueError("Actual canonical support count/convention mismatch")
    index, labels = arrays["train_index"], arrays["train_y"]
    if (index.dtype != np.int64 or labels.dtype != np.int64 or index.ndim != 1
            or labels.ndim != 1 or not len(index) or len(index) != len(labels)
            or len(index) != meta["role_counts"]["train"] or index.min() < 0 or index.max() >= n
            or np.any(index[1:] <= index[:-1])
            or not np.array_equal(np.unique(labels), np.array([0, 1], dtype=np.int64))):
        raise ValueError("Invalid authentic all-TRAIN role")
    return arrays, meta


def seed_all(np, torch, seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def finite_parameters_and_optimizer(torch, model, optimizer):
    if any(not torch.isfinite(p).all().item() for p in model.parameters()):
        raise FloatingPointError("Nonfinite native parameters after Adam")
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all().item():
                raise FloatingPointError("Nonfinite native Adam state")


def native_groups(model, optimizer_config):
    sheaf, other = model.grouped_parameters()
    return [{"params": sheaf, "weight_decay": optimizer_config["sheaf_decay"]},
            {"params": other, "weight_decay": optimizer_config["weight_decay"]}]


def topology_storage_bytes(model, placement):
    seen, total = set(), 0
    for owner, names in ((model, ("edge_index", "time_range")),
                         (model.laplacian_builder, placement.BUILDER_TENSORS)):
        for name in names:
            tensor = getattr(owner, name, None)
            if tensor is not None:
                storage = tensor.untyped_storage()
                key = (str(tensor.device), storage.data_ptr())
                if key not in seen:
                    seen.add(key)
                    total += storage.nbytes()
    return total


def compare_full_static_indices(np, torch, adapter, placement, placed, data, args, seed, folder):
    """One authentic full-graph/config comparator; exact source construction."""
    seed_all(np, torch, seed)
    tick = time.perf_counter()
    direct_factory = adapter.make_native_factory(data["edge_index"], args)
    direct = direct_factory()  # source may do many CUDA scalar reads; no rewrite
    sync(torch, data["x"].device)
    record = {"config_id": "d2_f32_L2", "scope": "one_authentic_complete_graph",
              "direct_construction_seconds": time.perf_counter() - tick,
              "static_fields": {}, "validation_metrics": False, "test_truth": False}
    for owner, names in (("model", ("edge_index", "time_range")),
                         ("builder", placement.BUILDER_TENSORS)):
        lhs = placed if owner == "model" else placed.laplacian_builder
        rhs = direct if owner == "model" else direct.laplacian_builder
        for name in names:
            a, b = getattr(lhs, name, None), getattr(rhs, name, None)
            if a is None and b is None:
                continue
            if a is None or b is None or a.dtype != b.dtype or a.shape != b.shape:
                raise RuntimeError("Direct/placed static tensor schema mismatch: " + name)
            equal = torch.equal(a.detach().cpu(), b.detach().cpu())
            record["static_fields"][owner + "." + name] = {"shape": list(a.shape), "dtype": str(a.dtype),
                                                            "exact_equal": equal, "device": str(a.device)}
            if not equal:
                write_json(folder / "DIRECT_STATIC_INDEX_COMPARISON.json", record)
                raise RuntimeError("Direct/placed exact static index mismatch: " + name)
    initialization_equal = all(torch.equal(value.cpu(), direct.state_dict()[name].cpu())
                               for name, value in placed.state_dict().items())
    record["native_seed_initialization_state_exact_equal"] = initialization_equal
    record["status"] = "passed" if initialization_equal else "failed"
    write_json(folder / "DIRECT_STATIC_INDEX_COMPARISON.json", record)
    del direct
    gc.collect()
    if not initialization_equal:
        raise RuntimeError("Direct/placed seeded native initialization mismatch")
    return record


def qualify_one(np, torch, common, adapter, placement, config, seed, optimizer_config,
                data, identity, target, direct_mode, skip_reason):
    folder = target / config["id"]
    folder.mkdir()
    record = {"config_id": config["id"], "seed": seed, "identity": identity, "status": "started",
              "scope": "complete_graph_all_TRAIN_rows_one_Adam_update",
              "selector": "fixed post-one-update engineering state; no validation selection",
              "validation_arrays_decoded": False, "validation_metrics": False, "test_truth_access": False,
              "forward_counts": {"train": 0, "post_update_train_check": 0, "fresh_reconstruction": 0}}
    write_json(folder / "RESULT.json", record)
    started = time.perf_counter()
    stage = "native_CPU_topology_construction_and_transfer"
    model, rebuilt, optimizer, rebuilt_optimizer, groups = None, None, None, None, None
    try:
        device = data["x"].device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        args = dict(config["native_args"], graph_size=data["x"].shape[0], input_dim=data["x"].shape[1],
                    output_dim=2, device=str(device))
        factory = placement.make_native_placed_factory(torch, adapter, data["cpu_edge_index"], data["edge_index"], args)
        seed_all(np, torch, seed)
        tick = time.perf_counter()
        model = factory()
        sync(torch, device)
        record["construction_seconds"] = time.perf_counter() - tick
        record["native_args"] = args
        record["active_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        record["unique_static_topology_storage_bytes"] = topology_storage_bytes(model, placement)
        record["all_TRAIN_label_opportunity"] = len(data["train_y"])
        record["actual_canonical_directed_entries"] = data["edge_index"].shape[1]
        if config["id"] == "d2_f32_L2":
            stage = "one_full_graph_direct_static_index_comparison"
            if direct_mode == "full":
                try:
                    record["direct_static_index_check"] = compare_full_static_indices(
                        np, torch, adapter, placement, model, data, args, seed, folder)
                except Exception as error:
                    record["direct_static_index_check"] = {
                        "status": "failed", "failure": common.failure_record(error, stage),
                        "does_not_authorize_placement": True}
                    write_json(folder / "DIRECT_STATIC_INDEX_COMPARISON_FAILURE.json", record["direct_static_index_check"])
                    # Keep all four prescribed own-task updates reviewable even
                    # if direct construction is impractical or static parity fails.
                    # The summary retains the failed placement gate explicitly.
            else:
                record["direct_static_index_check"] = {"status": "not_run", "reason": skip_reason,
                                                        "placement_static_parity": "pending"}
                write_json(folder / "DIRECT_STATIC_INDEX_COMPARISON.json", record["direct_static_index_check"])
        # Reset a recorded operation stream so the optional index comparator
        # does not alter dropout or the source CPU SVD jitter for the own update.
        seed_all(np, torch, seed + 1000003)
        groups = native_groups(model, optimizer_config)
        optimizer = torch.optim.Adam(groups, lr=optimizer_config["lr"])
        stage = "own_TRAIN_forward_backward_Adam"
        tick = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logp = model(data["x"])
        record["forward_counts"]["train"] += 1
        if logp.shape != (data["x"].shape[0], 2) or not torch.isfinite(logp).all().item():
            raise FloatingPointError("Nonfinite/malformed complete native TRAIN output")
        loss = torch.nn.functional.nll_loss(logp[data["train_index"]], data["train_y"])
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Nonfinite own TRAIN NLL")
        record["own_TRAIN_nll_before_update"] = float(loss.item())
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in model.parameters() if p.requires_grad):
            raise FloatingPointError("Missing/nonfinite native own-task gradients")
        optimizer.step()
        finite_parameters_and_optimizer(torch, model, optimizer)
        del logp, loss
        sync(torch, device)
        record["one_train_update_seconds"] = time.perf_counter() - tick
        stage = "fixed_post_update_TRAIN_state_save"
        model.eval()
        with torch.no_grad():
            full = model(data["x"])
            record["forward_counts"]["post_update_train_check"] += 1
            if not torch.isfinite(full).all().item():
                raise FloatingPointError("Nonfinite native post-update output")
            train_logp = full[data["train_index"]].cpu()
        del full
        post_nll = torch.nn.functional.nll_loss(train_logp, data["train_y"].cpu())
        if not torch.isfinite(post_nll).item():
            raise FloatingPointError("Nonfinite post-update own TRAIN NLL")
        record["own_TRAIN_nll_after_update"] = float(post_nll.item())
        numpy_rng = np.random.get_state()
        checkpoint = {"schema": "owned_full_input_nsd_engineering_checkpoint_v1", "identity": identity,
                      "config_id": config["id"], "seed": seed, "selector": record["selector"],
                      "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                      "optimizer_state": optimizer.state_dict(), "train_logp": train_logp,
                      "torch_cpu_rng": torch.get_rng_state(),
                      "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                      "python_rng": random.getstate(),
                      "numpy_rng": [numpy_rng[0], numpy_rng[1].tolist(), int(numpy_rng[2]), int(numpy_rng[3]), float(numpy_rng[4])],
                      "validation_truth_or_metrics_saved": False, "test_truth_saved": False}
        checkpoint_path = folder / "SELECTED_ENGINEERING_STATE.pt"
        tick = time.perf_counter()
        torch.save(checkpoint, checkpoint_path)
        record["checkpoint_write_seconds"] = time.perf_counter() - tick
        del checkpoint, train_logp
        model, optimizer, groups = None, None, None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        stage = "fresh_native_reconstruction_and_Adam_restore"
        tick = time.perf_counter()
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if saved["identity"] != identity or saved["config_id"] != config["id"] or saved["seed"] != seed:
            raise RuntimeError("Owned engineering checkpoint identity mismatch")
        rebuilt = factory()
        rebuilt.load_state_dict(saved["state_dict"], strict=True)
        if any(not torch.equal(value.cpu(), saved["state_dict"][name]) for name, value in rebuilt.state_dict().items()):
            raise RuntimeError("Fresh strict native parameter/buffer roundtrip mismatch")
        rebuilt_optimizer = torch.optim.Adam(native_groups(rebuilt, optimizer_config), lr=optimizer_config["lr"])
        rebuilt_optimizer.load_state_dict(saved["optimizer_state"])
        finite_parameters_and_optimizer(torch, rebuilt, rebuilt_optimizer)
        random.setstate(saved["python_rng"])
        n = saved["numpy_rng"]
        np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
        torch.set_rng_state(saved["torch_cpu_rng"])
        if saved["torch_cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["torch_cuda_rng"])
        if not torch.equal(torch.get_rng_state(), saved["torch_cpu_rng"]):
            raise RuntimeError("Selected-state CPU RNG roundtrip mismatch")
        if saved["torch_cuda_rng"] and any(not torch.equal(a, b) for a, b in
                                           zip(torch.cuda.get_rng_state_all(), saved["torch_cuda_rng"])):
            raise RuntimeError("Selected-state CUDA RNG roundtrip mismatch")
        rebuilt.eval()
        with torch.no_grad():
            full = rebuilt(data["x"])
            record["forward_counts"]["fresh_reconstruction"] += 1
            if not torch.isfinite(full).all().item():
                raise FloatingPointError("Nonfinite reconstructed full native output")
            reconstructed = full[data["train_index"]].cpu()
        del full
        difference = float((reconstructed - saved["train_logp"]).abs().max().item())
        if difference > 1e-6:
            raise RuntimeError("Fresh selected-state TRAIN logp roundtrip exceeds 1e-6")
        sync(torch, device)
        record.update({"status": "passed", "fresh_reconstruction_seconds": time.perf_counter() - tick,
                       "state_parameters_buffers_exact": True, "Adam_state_restored_finite": True,
                       "owned_Python_NumPy_CPU_CUDA_rng_restored": True,
                       "max_abs_reconstructed_TRAIN_logp_difference": difference,
                       "checkpoint_sha256": sha256(checkpoint_path), "all_outputs_gradients_Adam_finite": True})
    except Exception as error:
        record.update({"status": "failed", "failure": common.failure_record(error, stage)})
    finally:
        record["complete_attempt_seconds"] = time.perf_counter() - started
        record["process_peak_rss_bytes_cumulative"] = common.process_peak_rss_bytes()
        if data["x"].device.type == "cuda":
            try:
                sync(torch, data["x"].device)
                record["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(data["x"].device)
                record["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(data["x"].device)
            except Exception as error:
                record["resource_query_failure"] = common.failure_record(error, "resource_finalization")
                record.update({"status": "failed", "failure": record["resource_query_failure"]})
        write_json(folder / "RESULT.json", record)
        model, rebuilt, optimizer, rebuilt_optimizer, groups = None, None, None, None, None
        gc.collect()
        if data["x"].device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    return record


def qualify_roles(roles=None, output=None, device="cpu", release_path=None, dependency_overlay=None,
                  direct_index_check="full", execute=False):
    """Optional public callable, inactive unless explicitly root-released."""
    protocol = json.loads((PACKET / "PROTOCOL.json").read_text())
    if not execute:
        return {"status": "inactive", "scope": "four_authentic_full_graph_own_TRAIN_updates",
                "configs": [config["id"] for config in protocol["configs"]], "validation_metrics": False,
                "test_access": False, "direct_index_check": "one_full_graph/config_if_root_selects_practical"}
    if not roles or not output or not release_path or direct_index_check not in ("full", "skip"):
        raise ValueError("Explicit roles/output/root release and supported direct mode are required")
    common, release, target = activate(release_path, roles, output, device, dependency_overlay, direct_index_check)
    write_json(target / "ACTIVATION.json", {"root_release_sha256": sha256(release_path), "scope": protocol})
    records = []
    try:
        if dependency_overlay:
            sys.path.insert(0, str(Path(dependency_overlay).resolve()))
        import numpy as np
        import torch
        arrays, metadata = read_train_input(np, common, roles)
        final_device = torch.device(device)
        if str(final_device) != device or (final_device.type == "cuda" and final_device.index is None):
            raise ValueError("Use the exact final device, e.g. cpu or cuda:0")
        torch.use_deterministic_algorithms(release["deterministic_algorithms"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        data = {name: torch.from_numpy(value).to(final_device) for name, value in arrays.items()}
        data["cpu_edge_index"] = torch.from_numpy(arrays["edge_index"])
        adapter = common.load_adapter()
        adapter.load_native_general_class()  # requires real eager householder
        householder_origin = getattr(sys.modules.get("torch_householder"), "__file__", None)
        if dependency_overlay and (householder_origin is None or not
                Path(householder_origin).resolve().is_relative_to(Path(dependency_overlay).resolve())):
            raise RuntimeError("Real householder module was not imported from the root-bound overlay")
        placement = load_stdlib_helper("nsd_qualifier_placement", RUNNER / "native_placement.py")
        versions = {name: importlib.metadata.version(name) for name in
                    ("torch", "numpy", "torch-geometric", "torch-sparse", "torch-scatter", "torch-householder")}
        expected = release.get("expected_runtime_versions", {})
        if not expected or any(versions.get(name) != version for name, version in expected.items()):
            raise RuntimeError("Native overlay/runtime versions do not match root release")
        identity = {"qualifier_source_seal_sha256": release["source_seal_sha256"],
                    "runner_source_seal_sha256": release["runner_source_seal_sha256"],
                    "role_archive_sha256": sha256(roles), "role_metadata_sha256": release["role_metadata_sha256"],
                    "native_commit": protocol["native_commit"], "root_release_sha256": sha256(release_path)}
        write_json(target / "RUNTIME_AND_INPUT_IDENTITY.json", {"versions": versions, "identity": identity,
                    "device": device, "cuda": torch.version.cuda, "dependency_overlay": dependency_overlay,
                    "householder_origin": householder_origin,
                    "support_counts_actual": metadata["support_counts"], "full_input_shape": list(arrays["x"].shape),
                    "all_TRAIN_rows": len(arrays["train_y"]), "decoded_arrays": list(TRAIN_KEYS),
                    "validation_truth_decoded": False, "validation_metrics": False, "test_truth_access": False})
        for config in protocol["configs"]:
            record = qualify_one(np, torch, common, adapter, placement, config, protocol["seed"], protocol["optimizer"],
                                 data, identity, target, direct_index_check, release.get("direct_index_check_skip_reason"))
            records.append(record)
            common.append_jsonl(target / "ALL_PRESETS.jsonl", record)
        all_passed = all(record["status"] == "passed" for record in records)
        static_passed = records[0].get("direct_static_index_check", {}).get("status") == "passed"
        static_failed = records[0].get("direct_static_index_check", {}).get("status") == "failed"
        summary = {"status": "failed" if not all_passed else "static_parity_failed" if static_failed else "passed" if static_passed else "updates_passed_static_parity_pending",
                   "all_four_full_task_updates_passed": all_passed, "one_full_graph_direct_static_index_parity_passed": static_passed,
                   "failures": [record["config_id"] for record in records if record["status"] != "passed"],
                   "validation_metrics": False, "validation_truth_decoded": False, "test_truth_access": False,
                   "competence": "unestablished; engineering-only one-step qualification", "identity": identity,
                   "training_and_reconstruction_source": "unmodified native class and sealed CPU-topology placement"}
        write_json(target / "QUALIFICATION_SUMMARY.json", summary)
        return summary
    except Exception as error:
        failure = common.failure_record(error, "qualifier_setup_or_summary")
        write_json(target / "QUALIFICATION_FAILURE.json", failure)
        for config in protocol["configs"][len(records):]:
            common.append_jsonl(target / "ALL_PRESETS.jsonl", {"config_id": config["id"], "status": "failed_before_update", "failure": failure})
        raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--release")
    parser.add_argument("--roles")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dependency-overlay")
    parser.add_argument("--direct-index-check", choices=("full", "skip"), default="full")
    args = parser.parse_args()
    result = qualify_roles(args.roles, args.output, args.device, args.release, args.dependency_overlay,
                           args.direct_index_check, args.execute)
    print(json.dumps(result, allow_nan=False))
    if args.execute and result["status"] in ("failed", "static_parity_failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
