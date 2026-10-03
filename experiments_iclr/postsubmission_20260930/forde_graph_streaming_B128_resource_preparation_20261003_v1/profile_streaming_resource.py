"""Source-only Q02 plan: one complete B128 forward/private-gradient per recipe.

Numerical imports and data access occur only after a separate root release.
No checkpoint, optimizer, prediction metric, TEST label or full-B128 oracle.
"""
import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback


PACKET = Path(__file__).resolve().parent
PHASE = PACKET.parent
V4_NAME = "graph_conditional_response_native_source_preparation_20261003_v4"
WARM_NAME = "amazon_ratings_native_warm_study_preparation_20261003_v2"
REFERENCE_NAME = "forde_graph_source_style_engineering_20261003_v1"
STREAMING_NAME = "forde_graph_direct_streaming_engineering_20261003_v1"
RECIPES = ("source_defaults", "roman_mono")
ROLE_SEED = "condresp-amazon-v1|split=0|opt=17|roles"
TARGET_PREFIX = "forde-Q02-resource-only-v1|split=0|seed=17|B=128"
STATE_RECEIPT = "Q02_source_fixed_seed17_sinusoidal_all_private_v1_excluded_from_donors"
PRECISION_RECEIPT = "native_float32_CPU_cache_and_row_powers_then_byte_exact_CUDA_q_direct_streaming_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def confined(value):
    path = Path(value)
    path = path if path.is_absolute() else PHASE / path
    require(".." not in path.parts and path.is_relative_to(PHASE), "Phase-confined path required")
    require(not any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(PHASE)),
            "Symlink paths forbidden")
    return path


def descriptor(value):
    path = confined(value)
    data = path.read_bytes()
    return {"path": str(path.relative_to(PHASE)), "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data)}


def verify(row):
    require(isinstance(row, dict) and set(row) == {"path", "sha256", "bytes"},
            "Exact bound file descriptor required")
    path = confined(row["path"])
    require(descriptor(path) == row, "Bound artifact changed: " + str(path))
    return path


def read_json(path):
    return json.loads(confined(path).read_text())


def verify_prepared_source():
    manifest = read_json(PACKET / "MANIFEST.json")
    require(manifest["schema"] == "forde-streaming-B128-resource-source-manifest-v1", "Source schema differs")
    for row in manifest["files"]:
        require(not Path(row["path"]).is_absolute() and ".." not in Path(row["path"]).parts,
                "Source payload path escapes packet")
        verify(dict(row, path=str((PACKET / row["path"]).relative_to(PHASE))))
    expected = sorted(row["path"] for row in manifest["files"])
    actual = sorted(str(p.relative_to(PACKET)) for p in PACKET.rglob("*") if p.is_file()
                    and p.name not in ("MANIFEST.json", "SEAL.json"))
    require(expected == actual and len(expected) == manifest["payload_count"], "Source inventory differs")
    seal = read_json(PACKET / "SEAL.json")
    require(seal["manifest_sha256"] == descriptor(PACKET / "MANIFEST.json")["sha256"], "Source seal differs")
    bindings = read_json(PACKET / "SOURCE_BINDINGS.json")
    # These contain source/engineering/metadata only, never model or label payloads.
    for row in bindings["source_payloads"] + bindings["predecessor_metadata"]:
        verify(row)
    return descriptor(PACKET / "MANIFEST.json"), bindings


def admission(release_path, output):
    manifest, bindings = verify_prepared_source()
    release_path = confined(release_path)
    output = confined(output)
    require(not release_path.is_relative_to(PACKET), "Separate root release required")
    require(output.is_relative_to(confined(bindings["execution_output_parent"])) and
            not output.is_relative_to(PACKET) and not output.exists(), "Fresh bound execution output required")
    release = read_json(release_path)
    expected = read_json(PACKET / "RELEASE_TEMPLATE.json")
    require(set(release) == set(expected), "Exact root release schema required")
    for key in ("schema", "kind", "scope", "data_manifest", "data_projection_release",
                "public_graph", "split0_train", "runtime_receipt", "device", "source_state",
                "recipes", "logical_B", "members", "test_labels_authorized",
                "validation_labels_authorized", "selected_checkpoint_use", "scientific_donor_eligible",
                "optimizer_update_authorized", "automatic_retry_authorized", "Q01_execution_receipt"):
        require(release[key] == expected[key], "Root release scope differs: " + key)
    require(release["packet_manifest"] == manifest and release["execution_authorized"] is True and
            release["root_observed_source_review"] is True and release["root_adopted_Q01"] is True and
            release["root_observed_Q01_exit0"] is True and
            release["output"] == str(output.relative_to(PHASE)), "Root source/Q01 adoption required")
    review = release["independent_source_review"]
    require(isinstance(review, dict) and Path(review["path"]).name == "REVIEW.json" and
            not confined(review["path"]).is_relative_to(PACKET), "Separate independent source review required")
    verify(review)
    q01 = json.loads(verify(release["Q01_execution_receipt"]).read_text())
    require(q01["status"] == "SYNTHETIC_SCOPE_QUALIFIED" and q01["all_native_cases_qualified"] is True and
            q01["all_coefficient_cases_qualified"] is True and not q01["unexpected_failures"] and
            q01["CUDA_initialized_after"] is False, "Exact prior tiny CPU qualification required")
    memory = release["minimum_free_device_bytes"]
    require(type(memory) is int and 0 < memory <= bindings["retained_runtime"]["runtime"]["hardware"]["total_memory"],
            "Root-selected positive available-memory precondition required")
    require(release["root_resource_availability_approved"] is True, "Root resource availability approval required")
    return release, descriptor(release_path), bindings, output


def numerical_imports(release, bindings):
    sys.dont_write_bytecode = True
    # Do not reuse a conflicting numerical/native module from the runner process.
    for prefix in ("torch", "numpy", "scipy", "torch_geometric", "core", "native_source",
                   "forde_graph", "direct_streaming", "_q02_pinned_warm_common"):
        require(not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules),
                "Fresh ordinary interpreter required: " + prefix)
    warm_path = PHASE / WARM_NAME / "common.py"
    spec = importlib.util.spec_from_file_location("_q02_pinned_warm_common", warm_path)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    torch, actual = common.configure_runtime(release["device"])
    supplied = json.loads(verify(release["runtime_receipt"]).read_text())
    require(supplied == bindings["retained_runtime"] and supplied["runtime"] == actual,
            "Fresh actual numerical files/settings/device differ from bound runtime")
    require(not torch.is_autocast_enabled("cpu") and not torch.is_autocast_enabled("cuda"),
            "Native autocast-off execution required")
    for name in (V4_NAME, REFERENCE_NAME, STREAMING_NAME):
        sys.path.insert(0, str(PHASE / name))
    modules = {name: importlib.import_module(name) for name in
               ("native_source.adapter", "native_source.tokens", "train_roles", "forde_graph", "direct_streaming")}
    owners = {"native_source.adapter": V4_NAME, "native_source.tokens": V4_NAME, "train_roles": V4_NAME,
              "forde_graph": REFERENCE_NAME, "direct_streaming": STREAMING_NAME}
    for name, module in modules.items():
        require(Path(module.__file__).resolve().is_relative_to(PHASE / owners[name]), "Imported source owner differs")
    return torch, actual, modules


def narrow_data(release, bindings, torch, modules):
    import numpy as np
    data = json.loads(verify(release["data_manifest"]).read_text())
    producer = json.loads(verify(release["data_projection_release"]).read_text())
    require(data == bindings["retained_data_metadata"] and producer == bindings["retained_producer_release"],
            "Actual projected metadata differs")
    require(data["public_graph"] == release["public_graph"] and data["train_labels"]["0"] == release["split0_train"] and
            data["producer_release"] == release["data_projection_release"] and
            producer["execution_authorized"] is True and producer["raw_label_payload_decode_disclosed"] is True and
            not data["test_label_artifacts"] and data["test_labels_used_for_fitting_grouping_selection_or_scoring"] is False,
            "Authorized TRAIN-only projection provenance differs")
    # Hash only these two allowed NPZ payloads. Raw/VAL/TEST label files are never opened.
    with np.load(verify(release["public_graph"]), allow_pickle=False) as arrays:
        require(set(arrays.files) == {"features", "edge_index", "train_mask", "val_mask", "test_mask"},
                "Public graph keys differ")
        x = torch.from_numpy(arrays["features"].copy())
        edge = torch.from_numpy(arrays["edge_index"].copy())
        train_mask = torch.from_numpy(arrays["train_mask"].copy())
    require(tuple(x.shape) == (24492, 300) and x.dtype == torch.float32 and bool(torch.isfinite(x).all()),
            "Complete native float32 Amazon features required")
    require(edge.ndim == 2 and edge.shape[0] == 2 and edge.dtype == torch.int64 and edge.numel() and
            int(edge.min()) >= 0 and int(edge.max()) < x.shape[0], "Complete native edge records required")
    roles_module = modules["train_roles"]
    native_ids = roles_module.select_native_train_column(
        train_mask, node_count=24492, expected_split_count=10, official_split_id=0,
        release_and_mask_receipt=object_sha((release["data_manifest"], release["public_graph"])))
    with np.load(verify(release["split0_train"]), allow_pickle=False) as arrays:
        require(set(arrays.files) == {"ids", "labels"} and arrays["ids"].dtype == np.int64 and
                arrays["labels"].dtype == np.int64 and arrays["ids"].ndim == arrays["labels"].ndim == 1,
                "Compact TRAIN schema differs")
        ids, labels = tuple(arrays["ids"].tolist()), tuple(arrays["labels"].tolist())
    require(ids == native_ids.ids and len(ids) == len(labels) == 12246, "Official split0 TRAIN IDs differ")
    roles = roles_module.fixed_stratified_roles(
        native_ids, labels, class_count=5, role_seed=ROLE_SEED,
        explicit_role_split_receipt="prospective_floor4n_over5_v1")
    ordered = sorted(roles.fit, key=lambda node: (hashlib.sha256(f"{TARGET_PREFIX}|node={node}".encode()).hexdigest(), node))
    require(len(ordered) >= 128, "Fixed fit role lacks B128; no replacement")
    targets = torch.tensor(ordered[:128], dtype=torch.long)
    channels = torch.tensor([dict(roles.train_labels)[node] for node in ordered[:128]], dtype=torch.long)
    return x, edge, targets, channels, roles


class Profile:
    def __init__(self, torch, device, receipt, output):
        self.torch, self.device, self.receipt, self.output = torch, device, receipt, output
        self.case = None

    def save(self):
        temporary = self.output / "PROFILE_RECEIPT.tmp"
        with temporary.open("w") as stream:
            json.dump(self.receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, self.output / "PROFILE_RECEIPT.json")

    def snapshot(self):
        torch, device = self.torch, self.device
        free, total = torch.cuda.mem_get_info(device)
        return {"allocated_bytes": torch.cuda.memory_allocated(device),
                "reserved_bytes": torch.cuda.memory_reserved(device),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "device_free_bytes": free, "device_total_bytes": total,
                "process_peak_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}

    @contextmanager
    def stage(self, name):
        torch, device = self.torch, self.device
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        record = {"name": name, "status": "RUNNING", "start": self.snapshot()}
        records = self.receipt["initial_stages"] if self.case is None else self.case["stages"]
        records.append(record)
        self.receipt["active_stage"] = name
        self.save()
        started = time.monotonic()
        synchronized_end = False
        try:
            yield
            torch.cuda.synchronize(device)
            synchronized_end = True
            record["status"] = "COMPLETE"
        except Exception:
            record["status"] = "FAILED"
            try:
                torch.cuda.synchronize(device)
                synchronized_end = True
            except Exception as error:
                record["failure_synchronization_error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            record["wall_seconds"] = time.monotonic() - started
            record["end_synchronization_completed"] = synchronized_end
            if synchronized_end:
                record["synchronized_wall_seconds"] = record["wall_seconds"]
            try:
                record["end"] = self.snapshot()
            except Exception as error:
                record["end_snapshot_error"] = {"type": type(error).__name__, "message": str(error)}
            self.save()


def summary(value):
    detached = value.detach()
    require(bool(torch.isfinite(detached).all()), "Nonfinite resource diagnostic")
    return {"shape": list(detached.shape), "dtype": str(detached.dtype),
            "minimum": float(detached.min()), "maximum": float(detached.max()),
            "maximum_absolute": float(detached.abs().max())}


def state_image(model, graph_module):
    return {name: graph_module.tensor_digest(value) for name, value in model.state_dict().items()}


def one_recipe(recipe, release, bindings, modules, profile, x, edge, targets, channels, roles):
    adapter, tokens = modules["native_source.adapter"], modules["native_source.tokens"]
    graph, streaming = modules["forde_graph"], modules["direct_streaming"]
    torch, device = profile.torch, profile.device
    free, _ = torch.cuda.mem_get_info(device)
    require(free >= release["minimum_free_device_bytes"], "Root memory availability precondition failed before recipe")
    case = {"recipe": recipe, "status": "RUNNING", "stages": [], "logical_B": 128, "members": 4,
            "resource_only_state": STATE_RECEIPT, "scientific_donor_eligible": False}
    profile.receipt["cases"].append(case)
    profile.case = case
    started = time.monotonic()
    with profile.stage("cold_CPU_model_and_fixed_all_private_state"):
        fields = bindings["recipes"][recipe]
        spec = adapter.NativeSpec(dataset="amazon-ratings", num_features=300, num_classes=5,
                                  **fields, recipe_receipt="retained_transfer_" + recipe + "_resource_only")
        model = adapter.build_all_layer_polyformer(spec, seed=17, members=4,
                                                   composition_receipt="unchanged_native_v4_Q02_resource_only")
        with torch.no_grad():
            for site_index, site in enumerate(model.site_names):
                factor = model.core.get_submodule(site)
                for member in range(4):
                    for factor_index, name in enumerate(("R", "S")):
                        value = getattr(factor, name)[member]
                        coordinates = torch.arange(value.numel(), dtype=torch.float32)
                        value.copy_(1 + .17 * torch.sin((member + 1) * (coordinates + 1)
                                                       + (site_index + 1) * (factor_index + 1)))
        names = graph.setup_all_private(model, bank_receipt=STATE_RECEIPT)
        before = state_image(model, graph)
        case.update(spec=fields, all_private_names=list(names), all_private_tensor_count=len(names),
                    all_private_scalars=sum(dict(model.named_parameters())[name].numel() for name in names),
                    total_model_scalars=sum(p.numel() for p in model.parameters()), initial_state_sha256=object_sha(before))
    with profile.stage("cold_CPU_complete_native_operator_tokens_and_bound_rows"):
        identity = tokens.CacheIdentity(
            dataset_release_sha256=bindings["retained_data_metadata"]["raw_release"]["sha256"],
            features_sha256=graph.tensor_digest(x), native_edge_records_sha256=graph.tensor_digest(edge),
            edge_attributes_sha256="none_native_unweighted", removed_units_sha256="none_complete_native_graph",
            fit_role_labels_sha256=object_sha(roles.train_labels), view="native", K=spec.K, base="mono",
            normalization_source_sha256=bindings["gcn_norm_source_sha256"], runtime_precision_receipt=PRECISION_RECEIPT)
        bank = tokens.build_uncached_mono_tokens(x, edge, None, identity)
        complete_tokens = torch.stack(bank.tokens, dim=1).detach()
        cpu_batch = graph.BoundGraphBatch.create(
            bank.operator.detach(), complete_tokens, targets, maximum_power=spec.K,
            feature_identity=identity.features_sha256, precision_receipt=PRECISION_RECEIPT)
        case["cache"] = {"identity": identity.__dict__, "identity_sha256": identity.fingerprint,
                         "complete_node_count": 24492, "original_feature_count": 300,
                         "complete_token_shape": list(complete_tokens.shape),
                         "operator_shape": list(cpu_batch.operator.shape),
                         "operator_coalesced": cpu_batch.operator.is_coalesced(),
                         "operator_stored_entries": cpu_batch.operator._nnz(),
                         "row_powers_shape": list(cpu_batch.powers.shape),
                         "bound_bytes": {name: getattr(cpu_batch, name + "_sha256") for name in
                                         ("operator", "tokens", "targets", "grams", "powers")}}
        del bank, complete_tokens
    with profile.stage("CUDA_complete_cache_and_model_transfer_plus_byte_custody"):
        # New explicit device-custody path. No CUDA sparse recomputation or altered bytes.
        batch = replace(cpu_batch, **{name: getattr(cpu_batch, name).to(device) for name in
                                     ("operator", "complete_tokens", "targets", "grams", "powers")})
        model.to(device)
        selected_channels = channels.to(device)
        batch.validate()
        require(state_image(model, graph) == before, "CPU-to-CUDA model bytes changed")
        del cpu_batch
    parameters = [dict(model.named_parameters())[name] for name in names]
    with torch.autocast(device_type="cuda", enabled=False):
        with profile.stage("CUDA_live_native_q_and_retained_predictor_graph_B128_M4"):
            q, logits = graph.selected_coefficients(model, batch, selected_channels)
            require(tuple(q.shape) == (4, 128, spec.K + 1, 300), "Whole q shape differs")
            case["q"] = summary(q)
        with profile.stage("CUDA_streaming_forward_and_complete_global_repulsion_cotangent"):
            repulsion, details = streaming.evaluate_streaming(q, batch, backend=streaming.BACKEND)
            ce = sum(torch.nn.functional.cross_entropy(member, selected_channels) for member in logits)
            objective = ce + repulsion
            require(bool(torch.isfinite(objective)), "Source CE+R is nonfinite")
            case["objective"] = {"formula": "sum_member_mean_TRAIN_CE_plus_R", "finite": True,
                                 "predictive_metrics_computed": False}
            case["streamed_statistics"] = {key: summary(details[key]) for key in
                                           ("norm2", "distance", "similarity", "bandwidth", "distance_cotangent")}
        with profile.stage("CUDA_single_complete_all_private_first_gradient_with_live_mixed_path"):
            gradients = torch.autograd.grad(objective, parameters, create_graph=False, retain_graph=False,
                                            allow_unused=False)
            require(len(gradients) == len(names), "Private tensor count differs")
            case["all_private_gradients"] = {name: summary(value) for name, value in zip(names, gradients)}
            case["streaming_accounting"] = details["accounting"]
            require(details["accounting"]["forward_target_reconstructions"] == 128 and
                    len(details["accounting"]["backward_calls"]) == 1 and
                    details["accounting"]["backward_calls"][0]["target_reconstructions"] == 128 and
                    details["accounting"]["backward_calls"][0]["normalization_and_A_VJPs"] == 128,
                    "Complete streamed forward/backward accounting differs")
        with profile.stage("CUDA_final_state_and_complete_cache_custody"):
            batch.validate()
            custody = {"state_bytes_unchanged": state_image(model, graph) == before,
                       "all_module_modes_eval": all(not module.training for module in model.modules()),
                       "parameter_grad_fields_unwritten": all(p.grad is None for p in model.parameters()),
                       "every_R_S_and_only_R_S_trainable": set(n for n, p in model.named_parameters() if p.requires_grad) == set(names),
                       "member_selection_restored": all(model.core.get_submodule(site).active_member is None for site in model.site_names),
                       "native_forward_idle": model._in_forward is False}
            require(all(custody.values()), "Final model custody failed")
            case["custody"] = custody
    case["profile_complete_wall_seconds"] = time.monotonic() - started
    case["GPU_peak_allocated_bytes"] = max(stage["end"]["peak_allocated_bytes"] for stage in case["stages"])
    case["GPU_peak_reserved_bytes"] = max(stage["end"]["peak_reserved_bytes"] for stage in case["stages"])
    case["bound_batch_validation_calls"] = 5
    case["status"] = "RESOURCE_PROFILE_COMPLETED_NO_NUMERICAL_OR_SCIENTIFIC_PROMOTION"
    # Own object/cache cleanup only. No other process, job, GPU or filesystem state is changed.
    del gradients, objective, ce, repulsion, details, q, logits, parameters, model, batch, selected_channels
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    case["after_own_cleanup"] = profile.snapshot()
    profile.save()
    profile.case = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    release, release_record, bindings, output = admission(args.release, args.output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "forde-streaming-B128-resource-execution-receipt-v1", "UTC_started": utc(),
               "status": "RUNNING", "root_release": release_record,
               "packet_manifest": descriptor(PACKET / "MANIFEST.json"), "root_release_contents": release,
               "logical_B": 128, "members": 4, "recipes": list(RECIPES),
               "source_fixed_state_excluded_from_donors": True, "initial_stages": [], "cases": [],
               "full_B128_oracle_materialized": False, "checkpoint_loaded_or_saved": False,
               "optimizer_created_or_updated": False, "TEST_or_VAL_labels_opened": False,
               "numerical_or_scientific_equivalence_admitted": False, "automatic_retry": False}
    torch_local = None
    started = time.monotonic()
    try:
        global torch
        torch, runtime, modules = numerical_imports(release, bindings)
        torch_local = torch
        receipt["actual_runtime"] = runtime
        profile = Profile(torch, torch.device(release["device"]), receipt, output)
        require(profile.snapshot()["device_free_bytes"] >= release["minimum_free_device_bytes"],
                "Root memory availability precondition failed before data work")
        with profile.stage("allowed_projected_public_and_split0_TRAIN_bytes_decode_and_fixed_targets"):
            x, edge, targets, channels, roles = narrow_data(release, bindings, torch, modules)
            receipt["target_receipt"] = {"role_seed": ROLE_SEED, "fit_count": len(roles.fit),
                                         "control_count": len(roles.control), "role_labels_sha256": object_sha(roles.train_labels),
                                         "target_prefix": TARGET_PREFIX, "ordered_ids": targets.tolist(),
                                         "ordered_TRAIN_channels": channels.tolist(),
                                         "ordered_ids_sha256": modules["forde_graph"].tensor_digest(targets),
                                         "ordered_channels_sha256": modules["forde_graph"].tensor_digest(channels),
                                         "retry_rebalance_or_outcome_selection": False}
        for recipe in RECIPES:
            one_recipe(recipe, release, bindings, modules, profile, x, edge, targets, channels, roles)
        # Only source/engineering/metadata and the already allowed two data payloads.
        with profile.stage("final_source_and_allowed_payload_preservation"):
            verify_prepared_source()
            for name in ("data_manifest", "data_projection_release", "public_graph", "split0_train", "runtime_receipt"):
                verify(release[name])
        receipt["status"] = "BOTH_B128_RESOURCE_PROFILES_COMPLETED_SOURCE_FIXED_DONOR_EXCLUDED"
    except Exception as error:
        receipt["status"] = "FAILED_NO_RETRY_NO_FALLBACK"
        receipt["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        if receipt["cases"] and receipt["cases"][-1]["status"] == "RUNNING":
            receipt["cases"][-1]["status"] = "FAILED"
        if torch_local is not None:
            try:
                receipt["failure_GPU_snapshot"] = Profile(torch_local, release["device"], receipt, output).snapshot()
            except Exception as snapshot_error:
                receipt["failure_GPU_snapshot_error"] = str(snapshot_error)
    finally:
        receipt["whole_process_body_wall_seconds"] = time.monotonic() - started
        receipt["process_peak_RSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt["UTC_finished"] = utc()
        with (output / "PROFILE_RECEIPT.json").open("w") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"status": receipt["status"], "receipt": str(output / "PROFILE_RECEIPT.json")}, allow_nan=False))
    return 0 if receipt["status"] == "BOTH_B128_RESOURCE_PROFILES_COMPLETED_SOURCE_FIXED_DONOR_EXCLUDED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
