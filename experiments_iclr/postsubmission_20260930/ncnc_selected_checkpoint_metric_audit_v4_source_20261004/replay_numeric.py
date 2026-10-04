"""Shared successor core. Historical bytes are conditional; 128eps is evidence only."""
from pathlib import Path
from time import perf_counter
import importlib
import math
import os
import sys
from replay_gate import require, bound, ARMS, file_sha
from replay_contract import validate_journal_payload, validate_selected_payload, validate_valid_receipt
from audit_contract import LABELS, PROFILE_CONTRACT, schedule, terminal, PASS


def configure_environment(*, already_imported=False):
    require(already_imported or "torch" not in sys.modules, "Configure common cuBLAS before Torch import")
    expected = PROFILE_CONTRACT["cublas_workspace_config"]
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (None, expected), "Conflicting cuBLAS profile; no fallback")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected


def original_modules(context):
    root = context["paths"]["driver_root"]
    sys.path.insert(0, str(root))
    result = {}
    for name in ("pilot_common", "pilot_model", "pilot_data", "pilot_state", "pilot_evaluate"):
        module = importlib.import_module(name)
        require(Path(module.__file__).resolve() == root / (name + ".py"), "Original driver module shadowed")
        result[name] = module
    return result


def trusted_load(torch, root, pin):
    from hashlib import sha256
    path = bound(pin, root=root)
    with path.open("rb") as handle:
        digest, size = sha256(), 0
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk); size += len(chunk)
        require(size == pin["bytes"] and digest.hexdigest() == pin["sha256"], "Open checkpoint custody changed")
        handle.seek(0)
        return torch.load(handle, map_location="cpu", weights_only=False)


def set_profile(torch, label):
    require(label in LABELS, "Unscheduled inference label")
    torch.use_deterministic_algorithms(label != "False_anchor", warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("highest")


def expected_profile(label):
    require(label in LABELS, "Unscheduled profile receipt")
    return {"label": label, "deterministic_algorithms": label != "False_anchor", "warn_only": False,
            "default_dtype": "torch.float32", "TF32_matmul": False, "TF32_cudnn": False,
            "autocast_cuda": False, "autocast_cpu": False, "float32_matmul_precision": "highest",
            "cudnn_benchmark": False, "cudnn_deterministic": False, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}


def profile_receipt(torch, label):
    value = {"label": label, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
             "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
             "default_dtype": str(torch.get_default_dtype()),
             "TF32_matmul": torch.backends.cuda.matmul.allow_tf32, "TF32_cudnn": torch.backends.cudnn.allow_tf32,
             "autocast_cuda": torch.is_autocast_enabled("cuda"), "autocast_cpu": torch.is_autocast_enabled("cpu"),
             "float32_matmul_precision": torch.get_float32_matmul_precision(),
             "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
             "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    require(value == expected_profile(label), "Actual dtype/TF32/autocast/strict profile differs from slot")
    return value


def tensor_custody(data, data_api):
    import torch
    keys = {"x", "pairs", "raw_edge_index", "valid_positive", "valid_negative"}
    require(keys <= set(data), "Complete canonical data/query tensors required")
    require(data["x"].dtype == torch.float32 and data["pairs"].dtype == data["raw_edge_index"].dtype == data["valid_positive"].dtype == data["valid_negative"].dtype == torch.int64, "Exact graph/query dtype differs")
    require(tuple(data["valid_positive"].shape) == (60084, 2) and tuple(data["valid_negative"].shape) == (100000, 2), "Canonical query coverage differs")
    return {k: data_api.tensor_sha(data[k]) for k in sorted(keys)}


def private_receipt(value, *, seen=None, depth=0):
    """Preserve JSON receipt fields; mark malformed nodes without exporting tensor bodies."""
    if value is None or type(value) in (str, int, bool): return value
    if type(value) is float:
        if math.isfinite(value): return value
        return {"nonfinite_float": "nan" if math.isnan(value) else "positive_infinity" if value > 0 else "negative_infinity"}
    if depth >= 16: return {"unavailable_node": "receipt_depth_limit", "type": type(value).__name__}
    if type(value) not in (dict, list, tuple): return {"unsupported_receipt_type": type(value).__name__, "tensor_or_object_body_exported": False}
    seen = set() if seen is None else seen
    if id(value) in seen: return {"unavailable_node": "cyclic_receipt", "type": type(value).__name__}
    seen.add(id(value))
    try:
        if type(value) is dict:
            if all(type(key) is str for key in value):
                return {key: private_receipt(child, seen=seen, depth=depth + 1) for key, child in value.items()}
            return {"dictionary_with_non_string_keys": [{"key": private_receipt(key, seen=seen, depth=depth + 1), "value": private_receipt(child, seen=seen, depth=depth + 1)} for key, child in value.items()]}
        return [private_receipt(child, seen=seen, depth=depth + 1) for child in value]
    finally: seen.remove(id(value))


def valid_arrays(positive, negative, receipt, data_api):
    import torch
    validate_valid_receipt(receipt)
    require(positive.device.type == negative.device.type == "cpu" and positive.dtype == negative.dtype == torch.float32, "Original CPU float32 score vectors required")
    require(tuple(positive.shape) == (60084,) and tuple(negative.shape) == (100000,) and bool(torch.isfinite(positive).all()) and bool(torch.isfinite(negative).all()), "Invalid full-pool finite score vectors")
    actual = {"positive": data_api.tensor_sha(positive), "negative": data_api.tensor_sha(negative)}
    require(actual == receipt["score_digests"], "Scalar score digest is detached from actual typed ordered array")
    return actual


def engineering_report(candidate, reference):
    """Literal absolute-plus-relative128eps receipt; never consulted by PASS."""
    import torch
    require(candidate.dtype == reference.dtype == torch.float32 and candidate.shape == reference.shape and bool(torch.isfinite(candidate).all()) and bool(torch.isfinite(reference).all()), "Engineering arrays differ in dtype/shape/finiteness")
    eps = float(torch.finfo(torch.float32).eps)
    delta = (candidate.double() - reference.double()).abs()
    limit = 128 * eps + 128 * eps * reference.double().abs()
    return {"name": "engineering_absolute_plus_relative_128eps_fp32", "atol": 128 * eps, "rtol": 128 * eps,
            "acceptance_authority": False, "scientific_forward_error_bound": False,
            "max_absolute_error": float(delta.max()), "outside_budget_rows": int((delta > limit).sum()),
            "all_rows_within_engineering_budget": bool((delta <= limit).all())}


def authenticate_pool(rows, *, expected_member, expected_binding, pool, historical_digest, data_api):
    """The full authenticated array stays in process; a scalar label cannot substitute."""
    require(type(historical_digest) is str and len(historical_digest) == 64 and all(c in "0123456789abcdef" for c in historical_digest), "Malformed immutable historical digest")
    flags = {}
    for label in LABELS:
        row = rows[label]
        require(row["member"] == expected_member and row["label"] == label and row["validated"] is True and row["before_profile"] == row["after_profile"] == expected_profile(label) and row["binding"] == expected_binding, "Wrong cell/state/row/member/label/profile reference")
        actual = valid_arrays(row["positive"], row["negative"], row["receipt"], data_api)
        flags[label] = actual[pool] == historical_digest
    chosen = next((label for label in LABELS if flags[label]), None)
    return (rows[chosen][pool] if chosen else None), {"available": chosen is not None, "label": chosen,
        "historical_digest": historical_digest, "original_digest_equal_by_label": flags,
        "historical_arithmetic_status": "measured_against_authenticated_bytes" if chosen else "unavailable",
        "unavailable_reason": None if chosen else "No predeclared validated full vector authenticates historical bytes"}


def pooling_guard(evaluate):
    function = evaluate.mean_native_scores
    require(function.__module__ == evaluate.__name__ and function.__name__ == "mean_native_scores" and Path(function.__code__.co_filename).resolve() == Path(evaluate.__file__).resolve(), "Pinned original stack(dim=1).mean(1) API was replaced")


def reconstruct_i4(members, evaluate, data_api):
    """Newly reconstructed mean; original member order0,1,2,3 is mandatory."""
    require([r["member"] for r in members] == list(range(4)), "I4 historical member order differs")
    require(all(r["positive"] is not None and r["negative"] is not None for r in members), "Partial references cannot reconstruct I4")
    import torch
    before_profile = profile_receipt(torch, "True2")
    pooling_guard(evaluate)
    positive, negative = evaluate.mean_native_scores([(r["positive"], r["negative"], {}) for r in members])
    after_profile = profile_receipt(torch, "True2")
    require(before_profile == after_profile, "I4 reconstruction profile changed")
    receipt = {"score_digests": {"positive": data_api.tensor_sha(positive), "negative": data_api.tensor_sha(negative)},
               "historical_pooled_digest_available": False, "pooled_digest_kind": "newly_reconstructed_from_authenticated_historical_member_bytes",
               "member_order": list(range(4)), "pooling_api": "pilot_evaluate.mean_native_scores: torch.stack(dim=1).mean(1)",
               "dtype": "torch.float32", "reconstruction_profile_before": before_profile, "reconstruction_profile_after": after_profile, "member_reference_provenance": [r["provenance"] for r in members]}
    return positive, negative, receipt


def reference_metric_receipt(positive, negative, evaluate, metric, saved_hits50):
    hits = evaluate.hits50(metric, positive, negative)
    return {"reference_hits50": hits, "saved_selected_hits50_equal": hits == saved_hits50}


def hook(context, hooks, name, **kwargs):
    if hooks and name in hooks:
        require(context.get("synthetic_only") is True, "Injection is qualification-only")
        hooks[name](**kwargs)


def run(context, gate, accounting, *, fabricated_data=None, hooks=None):
    require(gate["full_success"] and gate["unique_fits_completed"] == 35 and gate["complete_served_cells"] == 25, "Full immutable35/25 required before numerical imports")
    require((fabricated_data is not None) == bool(context.get("synthetic_only")) == bool(gate.get("synthetic_only")), "Fabricated input requires admitted synthetic entry")
    require(not hooks or context.get("synthetic_only") is True, "Ordinary audit cannot inject faults")
    configure_environment(already_imported=bool(context.get("synthetic_only")))
    api = original_modules(context)
    common, model, data_api, state_api, evaluate = (api[n] for n in ("pilot_common", "pilot_model", "pilot_data", "pilot_state", "pilot_evaluate"))
    accounting.phase("original_runtime_admission")
    common.runtime_stdlib(context)
    # The inherited runtime authority stays False; each fixed serving profile follows admission.
    if "torch" in sys.modules: set_profile(sys.modules["torch"], "False_anchor")
    device, _ = model.runtime(context)
    import torch
    accounting.cuda_started = True
    torch.cuda.reset_peak_memory_stats(0)
    mods = model.modules(context); mods["design"].validate_plan(context["plan"])
    metric = evaluate.evaluator(context)
    prepared, failures = {}, []
    accounting.phase("all25_exact_selected_state_contracts")
    for (unit, seed), row in sorted(gate["units"].items()):
        payload = trusted_load(torch, row["root"], row["journal"]["state_file"])
        state = validate_journal_payload(payload, row["journal"], row["complete"], context["identity"], unit, seed, mods["design"].select_validation_candidate)
        if unit == "native_bank4":
            cost = next(c for c in gate["costs"] if (c["unit"], c["base_seed"]) == (unit, seed))
            cost["candidate101_extra_VALID_wall_seconds"] = sum(r["wall_seconds"] for r in state["ensemble_candidates"][-1]["extra_VALID_evaluations"])
        for cell in [c for c in gate["cells"] if (c["unit"], c["base_seed"]) == (unit, seed)]:
            selected = trusted_load(torch, row["root"], cell["checkpoint"])
            saved, digests = validate_selected_payload(selected, cell, state, context["identity"], state_api.state_digest)
            prepared[(cell["arm"], seed)] = {"snapshots": saved, "state_digests": [state_api.state_digest(s) for s in saved], "score_digests": digests}
    require(len(prepared) == 25, "All25 states required before first scorer entry")
    accounting.phase("complete_canonical_TRAIN_VALID_tensor_load")
    if fabricated_data is None: data = data_api.load_data(context, device)
    else:
        require(context["admission_release"]["schema"] == "ncnc-selected-checkpoint-metric-audit-synthetic-root-release-v4" and context["authority"]["files"] == {} and fabricated_data["fabricated_inputs_only"] is True, "Fabricated scope differs")
        require(tuple(fabricated_data["x"].shape) == (16, 128) and tuple(fabricated_data["pairs"].shape) == (13, 2), "Reviewed fabricated graph differs")
        data = fabricated_data
    data_digest = tensor_custody(data, data_api)
    if fabricated_data is not None: require(data_digest == context["synthetic_tensor_digests"], "Admitted fabricated typed rows changed")
    context["loaded_tensor_guard"] = lambda: require(tensor_custody(data, data_api) == data_digest, "Loaded canonical tensor/row custody changed")
    source_eval_sha = file_sha(evaluate.__file__)
    by_cell = {(c["arm"], c["base_seed"]): c for c in gate["cells"]}
    private_rows, outputs, stopped = [], {}, False
    cell_details = []
    def retain_private():
        accounting.private({"slots": private_rows, "slot_ledger": accounting.ledger.rows, "cells": cell_details, "failures": failures, "no_subset_publication": True})
    accounting.phase("fixed120_no_retry_scorer_slots")
    for slot in schedule():
        arm, seed, member, label = (slot[k] for k in ("arm", "base_seed", "member", "label"))
        item = prepared[(arm, seed)]; saved = item["snapshots"][member]
        instance = optimizer = None
        receipt = None
        current_slot = {**slot, "returned": False, "validated": False, "validation_status": "ATTEMPT_STARTED",
                        "returned_original_receipt": None, "post_guard_receipt": None, "guard_failure": None,
                        "returned_original_receipt_evidence_only": True, "validation_scope": "post_guard_required_receipt_fields_arrays_state_RNG_rows_profile",
                        "receipt_encoding": "JSON fields preserved; nonfinite/cyclic/unsupported nodes explicitly marked; tensor bodies excluded"}
        try:
            accounting.ledger.event(slot["slot"], "attempted")
            private_rows.append(current_slot); retain_private(); accounting.progress()
            set_profile(torch, label)
            hook(context, hooks, "after_profile", slot=slot, torch=torch)
            before_profile = profile_receipt(torch, label)
            context["loaded_tensor_guard"]()
            require(state_api.state_digest(saved) == item["state_digests"][member], "Retained selected snapshot mutated")
            began = perf_counter()
            instance, optimizer = model.make_native(mods, seed + 5 * member if arm == ARMS[1] else seed, 70 if arm == ARMS[4] else 64, device) if arm in (ARMS[0], ARMS[1], ARMS[4]) else model.make_factorized(mods, seed, device)
            state_api.restore_snapshot(instance, optimizer, saved, restore_random=True)
            hook(context, hooks, "after_restore", slot=slot, instance=instance, optimizer=optimizer, state_api=state_api, torch=torch)
            restored = state_api.snapshot(instance, optimizer)
            require(state_api.state_digest(restored) == item["state_digests"][member], "Exact restored models/Adam/flags/RNG differs before call")
            fixed = state_api.state_digest({"models": restored["models"], "optimizer": restored["optimizer"]})
            before_rng = state_api.rng_digest(state_api.rng_state())
            hook(context, hooks, "before_call", slot=slot, instance=instance, optimizer=optimizer)
            require(file_sha(evaluate.__file__) == source_eval_sha, "Scorer source bytes changed")
            mode = "private" if arm == ARMS[2] else "pooled_after_clamp" if arm == ARMS[3] else None
            accounting.ledger.event(slot["slot"], "entered_original_scorer")
            positive, negative, receipt = evaluate.score_valid(instance, data, mods, mode=mode)
            accounting.ledger.event(slot["slot"], "returned")
            # Snapshot the original returned receipt before hooks or any post-return validation.
            current_slot.update(returned=True, validated=False, validation_status="RETURNED_PENDING_GUARDS",
                                returned_original_receipt=private_receipt(receipt))
            retain_private()
            hook(context, hooks, "after_call", slot=slot, positive=positive, negative=negative, receipt=receipt, data=data, torch=torch)
            context["loaded_tensor_guard"]()
            actual = valid_arrays(positive, negative, receipt, data_api)
            require(state_api.rng_digest(state_api.rng_state()) == before_rng, "Scoring changed restored RNG")
            after = state_api.snapshot(instance, optimizer, rng=saved["rng"])
            require(state_api.state_digest({"models": after["models"], "optimizer": after["optimizer"]}) == fixed, "Scoring changed model/optimizer")
            require(all(flag is False for tree in after["flags"] for flag in tree.values()), "Prescribed original eval transition differs")
            after_profile = profile_receipt(torch, label)
            require(before_profile == after_profile, "Profile changed during scorer")
            outputs[(arm, seed, member, label)] = {"member": member, "label": label, "positive": positive.clone(), "negative": negative.clone(), "receipt": receipt,
                "before_profile": before_profile, "after_profile": after_profile, "validated": True,
                "binding": {"arm": arm, "base_seed": seed, "member": member, "selected_state_digest": item["state_digests"][member], "canonical_data_digests": data_digest, "original_identity": context["identity"]}}
            current_slot.update(validated=True, validation_status="VALIDATED", state_digest=item["state_digests"][member], typed_score_digests=actual,
                original_score_digest_equal=actual == item["score_digests"][member], before_profile=before_profile, after_profile=after_profile,
                canonical_query_digests={k: data_digest[k] for k in ("valid_positive", "valid_negative")}, post_guard_receipt=private_receipt(receipt),
                construction_restore_and_score_wall_seconds=perf_counter() - began)
            accounting.ledger.event(slot["slot"], "completed_validated"); accounting.progress()
            retain_private()
        except Exception as error:
            current_slot.update(validation_status="VALIDATED_WITH_LATER_ACCOUNTING_FAILURE" if current_slot["validated"] else "RETURNED_GUARD_FAILED" if current_slot["returned"] else "FAILED_BEFORE_RETURN",
                                post_guard_receipt=private_receipt(receipt),
                                guard_failure={"phase": "after_validation_accounting" if current_slot["validated"] else "post_return" if current_slot["returned"] else "before_return", "exception_type": type(error).__name__, "condition": str(error)})
            accounting.ledger.fail(slot["slot"], error); accounting.ledger.stop("slot_error")
            failures.append({"category": "FAILED_SCORER_OR_SLOT_GUARD", "slot": slot["id"], "exception_type": type(error).__name__, "condition": str(error)})
            stopped = True
            retain_private()
            break
        finally:
            if instance is not None: del instance, optimizer
    # No scorer calls occur below. Fixed strict True2 applies to CPU metric/reference assembly.
    set_profile(torch, "True2")
    # Missing historical vectors are never synthesized.
    metric_checks, reference_checks = 0, 0
    for arm in ARMS:
        for seed in range(5):
            cell = by_cell[(arm, seed)]; item = prepared[(arm, seed)]
            count = len(item["snapshots"])
            current = {"arm": arm, "base_seed": seed, "served_metrics": {}, "member_references": [], "engineering_discrepancies": []}
            cell_details.append(current)
            if not all((arm, seed, m, label) in outputs for m in range(count) for label in LABELS):
                cell["status"] = "NOT_COMPLETED_TERMINAL_STOP"; current["status"] = cell["status"]; retain_private(); continue
            rows = [{label: outputs[(arm, seed, m, label)] for label in LABELS} for m in range(count)]
            repeat = True
            for m in range(count):
                repeat = repeat and rows[m]["True1"]["receipt"]["score_digests"] == rows[m]["True2"]["receipt"]["score_digests"]
            served = {}
            for label in LABELS:
                label_rows = [(rows[m][label]["positive"], rows[m][label]["negative"], rows[m][label]["receipt"]) for m in range(count)]
                if arm == ARMS[1]: pooling_guard(evaluate)
                positive, negative = evaluate.mean_native_scores(label_rows) if arm == ARMS[1] else label_rows[0][:2]
                served[label] = (positive, negative)
                hits = evaluate.hits50(metric, positive, negative); metric_checks += 1
                current["served_metrics"][label] = {"hits50": hits, "saved_selected_hits50_equal": hits == cell["selection"]["hits50"], "acceptance_authority": label != "False_anchor",
                    "digests": {"positive": data_api.tensor_sha(positive), "negative": data_api.tensor_sha(negative)}}
                retain_private()
            repeat = repeat and current["served_metrics"]["True1"]["digests"] == current["served_metrics"]["True2"]["digests"]
            current["strict_True_observed_exact_repeat_bytes"] = repeat
            if not repeat: failures.append({"category": "FAILED_STRICT_REPEAT", "arm": arm, "base_seed": seed})
            strict_metric = all(current["served_metrics"][label]["saved_selected_hits50_equal"] for label in ("True1", "True2"))
            if not strict_metric: failures.append({"category": "FAILED_STRICT_METRIC", "arm": arm, "base_seed": seed})
            refs = []
            for m in range(count):
                pools, provenance = {}, {"member": m, "selected_state_digest": item["state_digests"][m]}
                for pool in ("positive", "negative"):
                    pools[pool], provenance[pool] = authenticate_pool(rows[m], expected_member=m, expected_binding={"arm": arm, "base_seed": seed, "member": m, "selected_state_digest": item["state_digests"][m], "canonical_data_digests": data_digest, "original_identity": context["identity"]}, pool=pool, historical_digest=item["score_digests"][m][pool], data_api=data_api)
                    if pools[pool] is not None:
                        for label in ("True1", "True2"):
                            current["engineering_discrepancies"].append({"member": m, "pool": pool, "candidate_label": label, "reference_label": provenance[pool]["label"],
                                "scope": "candidate_vs_authenticated_historical_member_bytes", **engineering_report(rows[m][label][pool], pools[pool])})
                current["member_references"].append(provenance)
                refs.append({"member": m, **pools, "provenance": provenance})
            reference_metric_equal = None
            if arm == ARMS[1] and all(r[pool] is not None for r in refs for pool in ("positive", "negative")):
                hook(context, hooks, "before_reference_mean", references=refs, evaluate=evaluate)
                rp, rn, reconstructed = reconstruct_i4(refs, evaluate, data_api)
                checked_metric = reference_metric_receipt(rp, rn, evaluate, metric, cell["selection"]["hits50"]); reference_checks += 1
                reference_metric_equal = checked_metric["saved_selected_hits50_equal"]
                current["I4_reference_mean"] = {**reconstructed, **checked_metric,
                    "runtime_authority_sha256": context["identity"]["runtime_authority_sha256"], "runtime_authority_schema": context["runtime"]["schema"], "canonical_query_digests": {k: data_digest[k] for k in ("valid_positive", "valid_negative")}}
                if not reference_metric_equal: failures.append({"category": "FAILED_I4_REFERENCE_METRIC", "arm": arm, "base_seed": seed})
                for label in ("True1", "True2"):
                    for index, pool in enumerate(("positive", "negative")):
                        current["engineering_discrepancies"].append({"pool": pool, "candidate_label": label, "scope": "candidate_vs_newly_reconstructed_historical_member_reference_mean",
                            **engineering_report(served[label][index], (rp, rn)[index])})
            elif arm == ARMS[1]: current["I4_reference_mean"] = {"available": False, "historical_pooled_digest_available": False, "reason": "At least one authenticated member/pool unavailable"}
            passed = repeat and strict_metric and reference_metric_equal is not False
            cell.update(status="PASS" if passed else "FAILED_METRIC_AUDIT", replayed_VALID_hits50=current["served_metrics"]["True1"]["hits50"], selected_hits50_equal=strict_metric)
            current["status"] = cell["status"]; retain_private()
    require(metric_checks <= 75 and reference_checks <= 5, "Declared metric-check denominator exceeded")
    if not failures:
        require(accounting.ledger.counts()["completed_validated"] == 120 and metric_checks == 75 and all(c["status"] == "PASS" for c in gate["cells"]), "All25/fixed120 strict audit is incomplete")
    retain_private()
    return gate["cells"], {"status": terminal(failures), "failures": failures, "work": accounting.ledger.counts(),
        "cells_served_metric_checks_completed": sum(len(c["served_metrics"]) == 3 for c in cell_details),
        "scheduled_served_metric_checks_planned": 75, "scheduled_served_metric_checks_completed": metric_checks,
        "I4_authenticated_reference_mean_checks_maximum": 5, "I4_authenticated_reference_mean_checks_completed": reference_checks,
        "metric_checks_maximum_total": 80, "historical_references_required_for_checkpoint_metric_PASS": False,
        "engineering_128eps_acceptance_authority": False, "old_exact_replay_qualified": False,
        "strict_repeat_scope": "observed bytes in this admitted invocation; no universal kernel determinism claim",
        "terminal_stop_after_slot_error": stopped, "training_updates": 0, "retries": 0}
