"""Separately released full-Amazon B2 CPU-cache/CUDA numerical oracle.

Source preparation alone imports no numerical library or data payload. No
checkpoint, optimizer, predictive scoring, fallback or post-outcome tolerance.
"""
import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
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
RECIPES = ("source_defaults", "roman_mono")


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
    require(path.is_relative_to(PHASE) and ".." not in path.parts, "Phase-confined path required")
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
            "Exact file descriptor required")
    path = confined(row["path"])
    require(descriptor(path) == row, "Bound artifact changed: " + str(path))
    return path


def read_json(path):
    return json.loads(confined(path).read_text())


def verify_source():
    manifest = read_json(PACKET / "MANIFEST.json")
    require(manifest["schema"] == "forde-small-real-B2-oracle-source-manifest-v1", "Source schema differs")
    for row in manifest["files"]:
        relative = Path(row["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Packet path escapes")
        verify(dict(row, path=str((PACKET / relative).relative_to(PHASE))))
    actual = sorted(str(p.relative_to(PACKET)) for p in PACKET.rglob("*") if p.is_file()
                    and p.name not in ("MANIFEST.json", "SEAL.json"))
    expected = sorted(row["path"] for row in manifest["files"])
    require(actual == expected and len(expected) == manifest["payload_count"], "Source inventory differs")
    require(read_json(PACKET / "SEAL.json")["manifest_sha256"] ==
            descriptor(PACKET / "MANIFEST.json")["sha256"], "Source seal differs")
    bindings = read_json(PACKET / "SOURCE_BINDINGS.json")
    for row in bindings["source_and_metadata_files"]:
        verify(row)
    return descriptor(PACKET / "MANIFEST.json"), bindings


def admission(release_path, output):
    manifest, bindings = verify_source()
    release_path, output = confined(release_path), confined(output)
    require(not release_path.is_relative_to(PACKET), "Separate root release required")
    require(output.is_relative_to(confined(bindings["execution_output_parent"])) and
            not output.exists() and not output.is_relative_to(PACKET), "Fresh bound execution output required")
    release = read_json(release_path)
    template = read_json(PACKET / "RELEASE_TEMPLATE.json")
    require(set(release) == set(template), "Exact root release schema required")
    mutable = {"execution_authorized", "root_observed_source_review", "packet_manifest",
               "independent_source_review", "root_adopted_Q01", "root_observed_Q01_exit0",
               "root_adopted_Q02_resource_only", "root_observed_Q02_exit0",
               "root_resource_availability_approved", "minimum_free_device_bytes", "output"}
    for key in set(template) - mutable:
        require(release[key] == template[key], "Root release changes fixed scope: " + key)
    require(release["packet_manifest"] == manifest and release["output"] == str(output.relative_to(PHASE)),
            "Release manifest/output differs")
    for key in mutable - {"packet_manifest", "independent_source_review", "minimum_free_device_bytes", "output"}:
        require(release[key] is True, "Root adoption/availability required: " + key)
    review = release["independent_source_review"]
    require(isinstance(review, dict) and Path(review["path"]).name == "REVIEW.json" and
            not confined(review["path"]).is_relative_to(PACKET), "Fresh separate source review required")
    verify(review)
    q01 = read_json(verify(release["Q01_execution_receipt"]))
    require(q01["status"] == "SYNTHETIC_SCOPE_QUALIFIED" and q01["all_native_cases_qualified"] is True and
            q01["all_coefficient_cases_qualified"] is True and not q01["unexpected_failures"] and
            q01["CUDA_initialized_after"] is False, "Bound tiny CPU Q01 required")
    require(q01["tolerances_predeclared_before_execution"]["torch.float32"] == bindings["tolerances"],
            "Predeclared Q01 float32 thresholds differ")
    q02 = read_json(verify(release["Q02_resource_receipt"]))
    exit_receipt = read_json(verify(release["Q02_external_exit_receipt"]))
    require(q02["status"] == "BOTH_B128_RESOURCE_PROFILES_COMPLETED_SOURCE_FIXED_DONOR_EXCLUDED" and
            q02["packet_manifest"] == bindings["Q02_source_manifest"] and exit_receipt["exit_code"] == 0 and
            [case["recipe"] for case in q02["cases"]] == list(RECIPES), "Actual Q02 resource passage/exit required")
    require(q02["numerical_or_scientific_equivalence_admitted"] is False,
            "Q02 must retain resource-only scope")
    minimum = release["minimum_free_device_bytes"]
    require(type(minimum) is int and 0 < minimum <= bindings["retained_runtime"]["runtime"]["hardware"]["total_memory"],
            "Positive root memory precondition required")
    return release, descriptor(release_path), bindings, output, q02


def load_helpers(bindings):
    # The pinned Q02 file has standard-library-only top level. Its main is not called.
    sys.dont_write_bytecode = True
    path = verify(bindings["Q02_helper_source"])
    spec = importlib.util.spec_from_file_location("_q03_pinned_q02_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Profile:
    def __init__(self, torch, device, receipt, output):
        self.torch, self.device, self.receipt, self.output = torch, device, receipt, output
        self.case = None

    def save(self):
        temporary = self.output / "ORACLE_RECEIPT.tmp"
        temporary.write_text(json.dumps(self.receipt, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, self.output / "ORACLE_RECEIPT.json")

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
        record = {"name": name, "status": "ENTERING", "UTC_started": utc()}
        records = self.receipt["initial_stages"] if self.case is None else self.case["stages"]
        records.append(record)
        self.receipt["active_stage"] = name
        started = time.monotonic()
        try:
            self.save()
            self.torch.cuda.synchronize(self.device)
            self.torch.cuda.reset_peak_memory_stats(self.device)
            record["start"] = self.snapshot()
            record["status"] = "RUNNING"
            yield
            self.torch.cuda.synchronize(self.device)
            record["status"] = "COMPLETE"
        except Exception as error:
            record["status"] = "FAILED"
            record["failure"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            record["wall_seconds_including_entry"] = time.monotonic() - started
            try:
                record["end"] = self.snapshot()
            except Exception as error:
                record["end_snapshot_error"] = {"type": type(error).__name__, "message": str(error)}
            if record["status"] == "COMPLETE":
                self.receipt["active_stage"] = None
            self.save()


def finite(value, name):
    require(bool(torch.isfinite(value).all()), name + " is nonfinite; no replacement")


def metric(a, b, atol, rtol):
    # Same elementwise criterion as pinned Q01. Reference b sets relative scale.
    a, b = a.detach(), b.detach()
    require(a.shape == b.shape and a.dtype == b.dtype, "Comparison shape/dtype differs")
    finite(a, "Alternative comparison tensor")
    finite(b, "Reference comparison tensor")
    error = (a - b).abs()
    scaled = error / (atol + rtol * b.abs())
    return {"shape": list(a.shape), "dtype": str(a.dtype), "finite": True,
            "max_absolute_error": float(error.max()), "max_scaled_error": float(scaled.max()),
            "within_predeclared_tolerance": bool((scaled <= 1).all())}


def require_comparisons(records):
    require(all(row["within_predeclared_tolerance"] for row in records.values()),
            "Predeclared numerical tolerance failed; no widening/retry/fallback")


def compare_private(a, b, names, tolerance, graph):
    records = {}
    for name, alternative, reference in zip(names, a, b):
        row = metric(alternative, reference, tolerance["gradient_absolute"], tolerance["gradient_relative"])
        row.update(alternative_sha256=graph.tensor_digest(alternative),
                   reference_sha256=graph.tensor_digest(reference),
                   alternative_maximum_absolute=float(alternative.detach().abs().max()),
                   reference_maximum_absolute=float(reference.detach().abs().max()))
        records[name] = row
    require(len(records) == len(names), "Every private tensor comparison required")
    return records


def private_gradients(loss, parameters):
    return torch.autograd.grad(loss, parameters, retain_graph=True, create_graph=False, allow_unused=False)


def independent_full_x(model, cpu_batch, x, edge, channels, identity, tokens, graph, device):
    full, logits, contexts = [], [], []
    for member in range(4):
        leaf = x.clone().requires_grad_(True)
        bank = tokens.build_uncached_mono_tokens(leaf, edge, None, identity)
        complete = torch.stack(bank.tokens, dim=1)
        require(graph.tensor_digest(bank.operator) == cpu_batch.operator_sha256 and
                graph.tensor_digest(complete) == cpu_batch.tokens_sha256,
                "Independent live CPU cache bytes differ from bound native cache")
        # This indexing/transfer remains differentiable from all N original feature rows.
        rows = complete[cpu_batch.targets].to(device)
        z = model.forward_member(rows, member)
        gradients = []
        for row in range(2):
            gradient = torch.autograd.grad(z[row, channels[row]], leaf,
                                           create_graph=True, retain_graph=True, allow_unused=False)[0]
            require(gradient.device.type == "cpu" and tuple(gradient.shape) == (24492, 300),
                    "Genuine complete CPU X derivative required")
            gradients.append(gradient.to(device))
        full.append(torch.stack(gradients))
        logits.append(z)
        contexts.append((leaf, bank, complete, rows))
    return torch.stack(full), torch.stack(logits), contexts


def materialized_full_x(full, graph):
    # Original direct full-feature recipe, with fixed stopped reference and epsilons.
    norm2 = full.square().sum(dim=(-1, -2))
    graph.finite_nonnegative(norm2, "Independent full-X norm2")
    normalized = full / torch.sqrt(norm2[..., None, None] + 1e-24)
    distance = (normalized[:, None] - normalized.detach()[None, :]).square().sum(dim=(-1, -2))
    similarity = (normalized[:, None] * normalized.detach()[None, :]).sum(dim=(-1, -2))
    r, bandwidth = graph.repulsion(distance)
    kernel = torch.exp(-distance / bandwidth.unsqueeze(0)).mean(dim=2)
    for name, value in (("normalized", normalized), ("distance", distance), ("similarity", similarity),
                        ("repulsion", r), ("bandwidth", bandwidth), ("kernel", kernel)):
        finite(value, "Independent " + name)
    return r, {"norm2": norm2, "normalized": normalized, "distance": distance,
               "similarity": similarity, "bandwidth": bandwidth, "kernel": kernel}


def one_recipe(recipe, release, bindings, q02, helper, modules, profile, x, edge, targets, channels):
    adapter, tokens = modules["native_source.adapter"], modules["native_source.tokens"]
    graph, streaming = modules["forde_graph"], modules["direct_streaming"]
    tolerance, device = bindings["tolerances"], profile.device
    expected = next(case for case in q02["cases"] if case["recipe"] == recipe)
    case = {"recipe": recipe, "status": "RUNNING", "stages": [], "members": 4, "logical_B": 2}
    profile.receipt["cases"].append(case)
    profile.case = case
    profile.receipt.update(active_recipe=recipe, active_stage="before_recipe_memory_precondition")
    profile.save()
    started = time.monotonic()
    require(profile.snapshot()["device_free_bytes"] >= release["minimum_free_device_bytes"],
            "Root memory precondition failed before recipe")
    with profile.stage("CPU_same_source_fixed_state_and_complete_bound_native_cache"):
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
        del factor, value, coordinates
        names = graph.setup_all_private(model, bank_receipt=helper.STATE_RECEIPT)
        before = helper.state_image(model, graph)
        require(object_sha(before) == expected["initial_state_sha256"] and
                list(names) == expected["all_private_names"], "Exact Q02 source-fixed state/private set differs")
        identity = tokens.CacheIdentity(
            dataset_release_sha256=bindings["retained_data_metadata"]["raw_release"]["sha256"],
            features_sha256=graph.tensor_digest(x), native_edge_records_sha256=graph.tensor_digest(edge),
            edge_attributes_sha256="none_native_unweighted", removed_units_sha256="none_complete_native_graph",
            fit_role_labels_sha256=q02["target_receipt"]["role_labels_sha256"], view="native", K=spec.K, base="mono",
            normalization_source_sha256=bindings["gcn_norm_source_sha256"],
            runtime_precision_receipt=helper.PRECISION_RECEIPT)
        bank = tokens.build_uncached_mono_tokens(x, edge, None, identity)
        cpu_batch = graph.BoundGraphBatch.create(bank.operator.detach(), torch.stack(bank.tokens, dim=1).detach(),
            targets, maximum_power=spec.K, feature_identity=identity.features_sha256,
            precision_receipt=helper.PRECISION_RECEIPT)
        require(cpu_batch.operator_sha256 == expected["cache"]["bound_bytes"]["operator"] and
                cpu_batch.tokens_sha256 == expected["cache"]["bound_bytes"]["tokens"],
                "CPU complete operator/token bytes differ from actual Q02")
        case.update(state_sha256=object_sha(before), private_names=list(names),
                    private_tensor_count=len(names), complete_token_shape=list(cpu_batch.complete_tokens.shape),
                    row_powers_shape=list(cpu_batch.powers.shape), cache_identity=identity.__dict__)
        del bank
    with profile.stage("CUDA_byte_exact_descriptor_model_transfer_and_selected_q"):
        batch = replace(cpu_batch, **{name: getattr(cpu_batch, name).to(device) for name in
                                     ("operator", "complete_tokens", "targets", "grams", "powers")})
        model.to(device)
        selected_channels = channels.to(device)
        batch.validate()
        require(helper.state_image(model, graph) == before, "Model transfer changed bytes")
        parameters = [dict(model.named_parameters())[name] for name in names]
        q, logits = graph.selected_coefficients(model, batch, selected_channels)
        require(tuple(q.shape) == (4, 2, spec.K + 1, 300), "Complete B2 coefficient shape differs")
    with torch.autocast(device_type="cuda", enabled=False):
        with profile.stage("CPU_live_complete_X_native_recurrences_and_differentiable_CUDA_reference"):
            full_reference, reference_logits, contexts = independent_full_x(
                model, cpu_batch, x, edge, selected_channels, identity, tokens, graph, device)
            finite(full_reference, "Complete independent full-X derivative")
        with profile.stage("CUDA_streamed_and_materialized_direct_forward_values_full_comparisons"):
            streaming_r, streamed = streaming.evaluate_streaming(q, batch, backend=streaming.BACKEND)
            direct_r, direct = graph.evaluate_coefficients(q, batch, backend="direct")
            reference_r, reference = materialized_full_x(full_reference, graph)
            full_from_q = torch.einsum("bkn,mbkf->mbnf", batch.powers, q)
            values = {
                "selected_logits_vs_live_X": metric(logits, reference_logits, tolerance["value_absolute"], tolerance["value_relative"]),
                "full_X_pullback_vs_live_X": metric(full_from_q, full_reference, tolerance["value_absolute"], tolerance["value_relative"]),
                "direct_normalized_vs_live_X": metric(direct["normalized"], reference["normalized"], tolerance["value_absolute"], tolerance["value_relative"])}
            for label, other, r in (("coefficient_direct", direct, direct_r), ("live_X", reference, reference_r)):
                for key in ("norm2", "distance", "similarity", "bandwidth", "kernel"):
                    values[label + "_" + key] = metric(streamed[key], other[key], tolerance["value_absolute"], tolerance["value_relative"])
                values[label + "_R"] = metric(streaming_r.reshape(1), r.reshape(1), tolerance["value_absolute"], tolerance["value_relative"])
            del other, r
            for target in range(2):
                normalized, _ = streaming.direct_target(q.detach(), batch.powers, target)
                values["streamed_normalized_vs_live_X_target" + str(target)] = metric(
                    normalized, reference["normalized"][:, target:target + 1], tolerance["value_absolute"], tolerance["value_relative"])
            del normalized, _
            ce = sum(torch.nn.functional.cross_entropy(z, selected_channels) for z in logits)
            reference_ce = sum(torch.nn.functional.cross_entropy(z, selected_channels) for z in reference_logits)
            values["CE_vs_live_X"] = metric(ce.reshape(1), reference_ce.reshape(1), tolerance["value_absolute"], tolerance["value_relative"])
            values["CE_plus_R_vs_live_X"] = metric((ce + streaming_r).reshape(1), (reference_ce + reference_r).reshape(1), tolerance["value_absolute"], tolerance["value_relative"])
            case["value_comparisons"] = values
            case["full_X_output_hashes"] = {"coefficient_pullback": graph.tensor_digest(full_from_q),
                                           "independent_live_X": graph.tensor_digest(full_reference)}
            profile.save()
            require_comparisons(values)
        with profile.stage("CUDA_q_force_streamed_vs_materialized_coefficient_direct"):
            streaming_force = torch.autograd.grad(streaming_r, q, retain_graph=True)[0]
            direct_force = torch.autograd.grad(direct_r, q, retain_graph=True)[0]
            case["q_force_comparison"] = metric(streaming_force, direct_force,
                tolerance["gradient_absolute"], tolerance["gradient_relative"])
            profile.save()
            require_comparisons({"q_force": case["q_force_comparison"]})
            del streaming_force, direct_force
        for endpoint in ("R", "CE_plus_R"):
            with profile.stage("complete_all_private_gradients_" + endpoint + "_both_materialized_references"):
                losses = (streaming_r, direct_r, reference_r) if endpoint == "R" else (
                    ce + streaming_r, ce + direct_r, reference_ce + reference_r)
                alternative = private_gradients(losses[0], parameters)
                coefficient = private_gradients(losses[1], parameters)
                independent = private_gradients(losses[2], parameters)
                comparisons = {
                    "coefficient_direct": compare_private(alternative, coefficient, names, tolerance, graph),
                    "independent_live_X": compare_private(alternative, independent, names, tolerance, graph)}
                case["private_" + endpoint + "_comparisons"] = comparisons
                profile.save()
                for records in comparisons.values():
                    require_comparisons(records)
                del losses, alternative, coefficient, independent
        with profile.stage("complete_state_cache_and_independent_X_final_custody"):
            batch.validate()
            cpu_batch.validate()
            custody = {"state_unchanged": helper.state_image(model, graph) == before,
                       "eval_modes": all(not module.training for module in model.modules()),
                       "parameter_grad_fields_unwritten": all(p.grad is None for p in model.parameters()),
                       "all_and_only_R_S_trainable": set(n for n, p in model.named_parameters() if p.requires_grad) == set(names),
                       "member_selection_idle": all(model.core.get_submodule(site).active_member is None for site in model.site_names),
                       "native_forward_idle": not model._in_forward,
                       "independent_X_leaves_unwritten": all(leaf.grad is None and graph.tensor_digest(leaf) == identity.features_sha256 for leaf, _, _, _ in contexts),
                       "independent_operator_and_tokens_unchanged": all(
                           graph.tensor_digest(bank.operator) == cpu_batch.operator_sha256 and
                           graph.tensor_digest(complete) == cpu_batch.tokens_sha256 for _, bank, complete, _ in contexts)}
            case["custody"] = custody
            case["streaming_accounting"] = streamed["accounting"]
            require(all(custody.values()), "Final state/cache/reference custody failed")
            calls = streamed["accounting"]["backward_calls"]
            require(streamed["accounting"]["forward_target_reconstructions"] == 2 and len(calls) == 3 and
                    all(call["target_reconstructions"] == 2 and call["normalization_and_A_VJPs"] == 2 and
                        call["self_direction_cotangent_maximum"] == 0 for call in calls),
                    "Fixed forward/three first-gradient endpoint accounting differs")
    with profile.stage("own_reference_and_CUDA_allocator_cleanup"):
        del q, logits, full_reference, reference_logits, contexts, streamed, direct, reference, full_from_q
        del ce, reference_ce, streaming_r, direct_r, reference_r, parameters, model, batch, cpu_batch
        del selected_channels
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    case["GPU_peak_allocated_bytes"] = max(stage["end"]["peak_allocated_bytes"] for stage in case["stages"])
    case["GPU_peak_reserved_bytes"] = max(stage["end"]["peak_reserved_bytes"] for stage in case["stages"])
    case["complete_wall_seconds_including_cleanup"] = time.monotonic() - started
    case["status"] = "B2_FIXED_ENDPOINTS_WITHIN_PREDECLARED_TOLERANCES_NO_SCIENTIFIC_PROMOTION"
    profile.case = None
    profile.receipt.update(active_recipe=None, active_stage=None)
    profile.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    release, release_record, bindings, output, q02 = admission(args.release, args.output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "forde-small-real-B2-oracle-execution-receipt-v1", "UTC_started": utc(),
               "status": "RUNNING", "root_release": release_record, "root_release_contents": release,
               "packet_manifest": descriptor(PACKET / "MANIFEST.json"), "initial_stages": [], "cases": [],
               "tolerances_predeclared_before_execution": bindings["tolerances"],
               "endpoint": bindings["endpoint"], "scientific_donor_eligible": False,
               "B128_numerical_equivalence_admitted": False, "CUDA_sparse_recurrence_equivalence_admitted": False,
               "predictive_metrics_computed": False, "checkpoint_loaded_or_saved": False,
               "optimizer_created_or_updated": False, "VAL_TEST_or_raw_label_payloads_opened": False,
               "automatic_retry_or_tolerance_widening": False}
    started = time.monotonic()
    profile, data_started = None, False
    try:
        global torch
        receipt["active_stage"] = "runtime_binding"
        helper = load_helpers(bindings)
        torch, runtime, modules = helper.numerical_imports(release, bindings, receipt)
        profile = Profile(torch, torch.device(release["device"]), receipt, output)
        receipt["active_stage"] = "before_data_memory_precondition"
        require(profile.snapshot()["device_free_bytes"] >= release["minimum_free_device_bytes"], "Root pre-data memory precondition failed")
        with profile.stage("allowed_public_split0_TRAIN_hash_decode_exact_Q02_targets"):
            data_started = True
            x, edge, all_targets, all_channels, roles = helper.narrow_data(release, bindings, torch, modules)
            graph = modules["forde_graph"]
            prior = q02["target_receipt"]
            require(all_targets.tolist() == prior["ordered_ids"] and all_channels.tolist() == prior["ordered_TRAIN_channels"] and
                    object_sha(roles.train_labels) == prior["role_labels_sha256"], "Reconstructed Q02 roles/ordered targets/channels differ")
            targets, channels = all_targets[:2].clone(), all_channels[:2].clone()
            require(targets.tolist() == bindings["fixed_target_ids"] and channels.tolist() == bindings["fixed_channels"], "Fixed B2 endpoint differs")
            receipt["targets"] = {"ids": targets.tolist(), "TRAIN_channels": channels.tolist(),
                                  "ids_sha256": graph.tensor_digest(targets), "channels_sha256": graph.tensor_digest(channels)}
        for recipe in RECIPES:
            one_recipe(recipe, release, bindings, q02, helper, modules, profile, x, edge, targets, channels)
        receipt["status"] = "BOTH_FIXED_B2_CPU_CACHE_CUDA_ENDPOINTS_QUALIFIED_NO_B128_OR_SCIENTIFIC_PROMOTION"
    except Exception as error:
        receipt["status"] = "FAILED_NO_RETRY_NO_TOLERANCE_WIDENING_NO_FALLBACK"
        receipt["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        if receipt["cases"] and receipt["cases"][-1]["status"] == "RUNNING":
            receipt["cases"][-1]["status"] = "FAILED"
    finally:
        preservation_started = time.monotonic()
        try:
            verify_source()
            for name in ("data_manifest", "data_projection_release", "runtime_receipt"):
                verify(release[name])
            if data_started:
                for name in ("public_graph", "split0_train"):
                    verify(release[name])
            receipt["preservation"] = {"source_and_allowed_payloads_match": True,
                                       "allowed_data_hashes_rechecked": data_started}
        except Exception as error:
            receipt["status"] = "FAILED_NO_RETRY_NO_TOLERANCE_WIDENING_NO_FALLBACK"
            receipt["preservation"] = {"source_and_allowed_payloads_match": False,
                "failure": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}}
        receipt["preservation"]["wall_seconds"] = time.monotonic() - preservation_started
        if profile is not None:
            try:
                receipt["final_GPU_snapshot"] = profile.snapshot()
            except Exception as error:
                receipt["final_GPU_snapshot_error"] = str(error)
        receipt["whole_process_body_wall_seconds"] = time.monotonic() - started
        receipt["process_peak_RSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt["UTC_finished"] = utc()
        (output / "ORACLE_RECEIPT.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": receipt["status"], "receipt": str(output / "ORACLE_RECEIPT.json")}, allow_nan=False))
    return 0 if receipt["status"] == "BOTH_FIXED_B2_CPU_CACHE_CUDA_ENDPOINTS_QUALIFIED_NO_B128_OR_SCIENTIFIC_PROMOTION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
