#!/usr/bin/env python3
"""Future root-released ordinary-runtime fabricated qualification; reuse existing fixture."""
from argparse import ArgumentParser
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import copy
import json
import sys
from replay_gate import require, bound, canonical, verify_manifest, descriptor, ARMS
from replay_run import execute, Accounting, atomic_json
from replay_numeric import original_modules, trusted_load, configure_environment, set_profile
from audit_contract import PROFILE_CONTRACT, PASS, sum_case_work
from replay_synthetic_admission import preflight, admitted_input_custody
from replay_synthetic_cases import clone_fixture, alter_own_payload, corrupt_after_metadata, corrupt_admitted_release, assert_failed_suppression, make_historical_references_unavailable, inconsistent_I4_reference, assert_provisional_receipts, assert_terminal_receipts, assert_unvalidated_return
from audit_synthetic_contract_cases import run as shared_contract_cases


def existing_fixture(context, api, device):
    import torch
    spec = bound(context["release"]["existing_fabricated_fixture"], decode=True)
    require(spec["schema"] == "ncnc-owned-existing-fabricated-full25-fixture-descriptor-v3" and spec["fabricated_inputs_only"] is True and spec["new_training_updates"] == 0, "Authenticated existing fabricated fixture only")
    root = canonical(spec["fixture_root"]); verify_manifest(root, spec["fixture_manifest_sha256"])
    require(spec["identity"]["family_id"] == "FABRICATED_NCNC_REPLAY_QUALIFICATION_ONLY" and spec["identity"]["synthetic_only"] is True and Path(spec["identity"]["family_lock_output_directory"]).is_relative_to(root), "Fixture cannot borrow study family identity")
    for row in spec["unit_custody"]:
        require(canonical(row["output_directory"]).is_relative_to(root) and canonical(row["physical_terminal"]["path"]).is_relative_to(root), "Fabricated custody escaped fixture")
    require(canonical(spec["family_lock"]["path"]).is_relative_to(root) and canonical(spec["tensor_custody"]["path"]).parent == root, "Fabricated lock/tensors escaped fixture")
    tensor_pin = {**spec["tensor_custody"], "path": Path(spec["tensor_custody"]["path"]).name}
    payload = trusted_load(torch, root, tensor_pin)
    require(payload["schema"] == "ncnc-replay-owned-fabricated-tensors-v1" and payload["fabricated_inputs_only"] is True and set(payload["tensors"]) == {"x", "pairs", "raw_edge_index", "valid_positive", "valid_negative"}, "Only bound fabricated canonical tensors admitted")
    data = {k: v.to(device) for k, v in payload["tensors"].items()}; data["fabricated_inputs_only"] = True
    require({k: api["pilot_data"].tensor_sha(v) for k, v in payload["tensors"].items()} == spec["tensor_digests"], "Existing fabricated typed tensor bytes differ")
    fixture_context = {**context, "identity": spec["identity"], "release": {**context["release"], "family_lock": spec["family_lock"], "unit_custody": spec["unit_custody"]},
                       "synthetic_tensor_custody": spec["tensor_custody"], "synthetic_tensor_digests": spec["tensor_digests"]}
    return {"context": fixture_context, "fixture_root": root, "data": data, "spec": spec}


def run(context, started):
    output = context["output"]; output.mkdir(parents=True, mode=0o700, exist_ok=False)
    plan = json.loads((Path(__file__).parent / "FABRICATED_QUALIFICATION_PLAN.json").read_text())
    cases = [{"case": name, "status": "NOT_ATTEMPTED", "work": {"planned": 120 if name in plan["full_core_cases"] else 0, "attempted": 0, "entered_original_scorer": 0, "returned": 0, "completed_validated": 0}} for name in plan["shared_contract_cases"] + plan["full_core_cases"]]
    peak_rows = []
    active_case = None
    current = None
    result = {"schema": "ncnc-selected-checkpoint-metric-audit-synthetic-qualification-v4", "status": "IN_PROGRESS",
              "identity": context["original_identity"], "sidecar_manifest_sha256": context["sidecar_manifest_sha256"], "source_entry": descriptor(Path(__file__).resolve()),
              "runtime_authority": context["release"]["runtime_authority"], "profile_contract": PROFILE_CONTRACT, "fabricated_inputs_only": True,
              "study_lock_data_outcome_checkpoint_accessed": False, "TEST_opened": False, "scientific_fit_updates": 0, "new_training_updates": 0,
              "existing_fabricated_fixture": context["release"]["existing_fabricated_fixture"], "cases": cases, "automatic_retry": False,
              "old_FAILED_qualification_repaired": False, "qualification_inference_beyond_baseline120_charged_by_actual_slots": True}
    atomic_json(output / "QUALIFICATION.json", result)
    code, fixture, torch = 1, None, None
    try:
        configure_environment()
        api = original_modules(context); api["pilot_common"].runtime_stdlib(context)
        device, _ = api["pilot_model"].runtime(context)
        import torch as torch_module
        torch = torch_module; torch.cuda.reset_peak_memory_stats(0)
        fixture = existing_fixture(context, api, device)
        peak_rows.append({"allocated": torch.cuda.max_memory_allocated(0), "reserved": torch.cuda.max_memory_reserved(0)})
        with patch.object(api["pilot_data"], "load_data", side_effect=RuntimeError("STUDY_DATA_ACCESS_FORBIDDEN_IN_FABRICATED_QUALIFICATION")):
            for row in shared_contract_cases(api, context):
                next(c for c in cases if c["case"] == row["case"]).update(row)
            for name in plan["full_core_cases"]:
                active_case = next(c for c in cases if c["case"] == name)
                active_case["status"] = "IN_PROGRESS"
                current = None
                target = output / ("inputs_" + name)
                current = clone_fixture(fixture, target, api, torch); current["output"] = output / ("result_" + name)
                release_copy = target / "ADMITTED_SYNTHETIC_RELEASE_COPY.json"; release_copy.write_bytes(context["release_path"].read_bytes())
                current["release_path"] = release_copy
                data = {k: v.clone() if torch.is_tensor(v) else v for k, v in fixture["data"].items()}
                hooks = {}
                I4_fault = inconsistent_I4_reference(current, api, torch) if name == "strict_metric" else None
                if name == "historical_unavailable_all25": make_historical_references_unavailable(current, api, torch)
                if name == "wrong_selected_state": alter_own_payload(current, api, torch, "selected_state")
                if name == "wrong_restored_state":
                    def wrong_state(**kw):
                        if kw["slot"]["slot"] == 1:
                            with torch.no_grad(): next(iter(kw["instance"][0].parameters())).add_(1.)
                    hooks["after_restore"] = wrong_state
                if name == "wrong_profile":
                    def wrong_profile(**kw):
                        if kw["slot"]["slot"] == 2: torch.use_deterministic_algorithms(False, warn_only=False)
                    hooks["after_profile"] = wrong_profile
                if name in ("wrong_query_row", "wrong_dtype", "strict_repeat", "strict_metric", "late_after_scorer"):
                    saved = next(c for c in json.loads(Path(current["release"]["family_lock"]["path"]).read_text())["cells"] if c["arm"] == ARMS[0] and c["base_seed"] == 0)["selection"]["hits50"]
                    def after_call(**kw):
                        slot = kw["slot"]["slot"]
                        if name == "strict_metric" and kw["slot"]["arm"] == ARMS[1] and kw["slot"]["base_seed"] == 0 and kw["slot"]["label"] == "False_anchor":
                            kw["positive"].fill_(I4_fault["positive_value"]); kw["negative"].fill_(I4_fault["negative_value"])
                            kw["receipt"]["score_digests"] = dict(I4_fault["typed_digests"])
                        if name == "wrong_query_row" and slot == 1: kw["data"]["valid_positive"][0, 0] = (kw["data"]["valid_positive"][0, 0] + 1) % 16
                        elif name == "wrong_dtype" and slot == 1:
                            kw["positive"].data = kw["positive"].double(); kw["receipt"]["wall_seconds"] = float("nan")
                        elif name == "late_after_scorer" and slot == 120: raise RuntimeError("FABRICATED_LATE_RETURN_GUARD_ERROR")
                        elif name == "strict_repeat" and slot == 3:
                            kw["positive"][0] = torch.nextafter(kw["positive"][0], torch.tensor(float("inf"), dtype=torch.float32))
                            kw["receipt"]["score_digests"]["positive"] = api["pilot_data"].tensor_sha(kw["positive"])
                        elif name == "strict_metric" and slot in (2, 3):
                            kw["positive"].fill_(2. if saved != 1. else 0.); kw["negative"].fill_(1.)
                            kw["receipt"]["score_digests"] = {"positive": api["pilot_data"].tensor_sha(kw["positive"]), "negative": api["pilot_data"].tensor_sha(kw["negative"])}
                    hooks["after_call"] = after_call
                if name in ("late_before_scorer", "late_original_scorer_error"):
                    def before_call(**kw):
                        if kw["slot"]["slot"] == 120:
                            if name == "late_before_scorer": raise RuntimeError("FABRICATED_LATE_BEFORE_SCORER_ERROR")
                            def fail_encoder(*args, **kwargs): raise RuntimeError("FABRICATED_ERROR_INSIDE_ORIGINAL_SCORER")
                            kw["instance"][0].forward = fail_encoder
                    hooks["before_call"] = before_call
                with ExitStack() as stack:
                    if name == "accounting_error": stack.enter_context(patch.object(Accounting, "finish", side_effect=RuntimeError("FABRICATED_FINISH_ERROR")))
                    if name == "mutation_during_finish":
                        original_finish = Accounting.finish
                        def mutated_finish(accounting):
                            receipt = original_finish(accounting); assert_provisional_receipts(accounting.context); corrupt_admitted_release(accounting.context); return receipt
                        stack.enter_context(patch.object(Accounting, "finish", mutated_finish))
                    def after_accounting(ctx):
                        assert_provisional_receipts(ctx); corrupt_admitted_release(ctx)
                    child_code, child = execute(current, perf_counter(), fabricated_data=data, hooks=hooks,
                        synthetic_after_metadata=corrupt_after_metadata if name == "custody_after_metadata" else None,
                        synthetic_after_accounting=after_accounting if name == "mutation_after_accounting" else None)
                assert_terminal_receipts(current, child)
                active_case.update(child_exit_code=child_code, child_status=child["status"], work=child["work"], child_result=descriptor(current["output"] / "AUDIT_RESULT.json"))
                peak_rows.append({"allocated": child.get("audit_cuda_peak_allocated_bytes", 0), "reserved": child.get("audit_cuda_peak_reserved_bytes", 0)})
                if name in ("baseline_all25", "historical_unavailable_all25"):
                    require(child_code == 0 and child["status"] == PASS and child["summary"] is not None and all(c["status"] == "PASS" for c in child["cells"]), "Ordinary full25 strict True metric/repeat baseline must actually pass; no repair or retry")
                    require(child["work"]["planned"] == child["work"]["attempted"] == child["work"]["entered_original_scorer"] == child["work"]["returned"] == child["work"]["completed_validated"] == 120, "Full120 baseline counts differ")
                    if name == "historical_unavailable_all25":
                        details = json.loads((current["output"] / "PRIVATE_AUDIT_DETAILS.json").read_text())
                        require(all(not member[pool]["available"] for c in details["cells"] for member in c["member_references"] for pool in ("positive", "negative")) and child["checkpoint_metric_audit"]["I4_authenticated_reference_mean_checks_completed"] == 0, "Missing history must remain unavailable while strict checkpoint/metric audit stands")
                else:
                    assert_failed_suppression(child_code, child)
                    if name == "late_before_scorer": require((child["work"]["attempted"], child["work"]["entered_original_scorer"], child["work"]["returned"], child["work"]["completed_validated"]) == (120, 119, 119, 119), "Late-before actual counts differ")
                    if name == "late_original_scorer_error": require((child["work"]["attempted"], child["work"]["entered_original_scorer"], child["work"]["returned"], child["work"]["completed_validated"]) == (120, 120, 119, 119), "Late original scorer error counts differ")
                    if name == "late_after_scorer": require((child["work"]["attempted"], child["work"]["entered_original_scorer"], child["work"]["returned"], child["work"]["completed_validated"]) == (120, 120, 120, 119), "Late-after actual counts differ")
                    if name in ("mutation_during_finish", "mutation_after_accounting"): require(child["status"] == "FAILED_FINAL_CUSTODY", "Final post-accounting mutation was not caught")
                    if name == "strict_repeat": require(any(f["category"] == "FAILED_STRICT_REPEAT" for f in child["failures"]), "Repeat mismatch reached the wrong gate")
                    if name == "strict_metric":
                        require(any(f["category"] == "FAILED_STRICT_METRIC" for f in child["failures"]) and any(f["category"] == "FAILED_I4_REFERENCE_METRIC" for f in child["failures"]) and child["status"] == "FAILED_I4_REFERENCE_METRIC", "Conditional I4 mean failure/strict metric precedence reached the wrong gate")
                        details = json.loads((current["output"] / "PRIVATE_AUDIT_DETAILS.json").read_text())
                        I4_cell = next(c for c in details["cells"] if c["arm"] == ARMS[1] and c["base_seed"] == 0)
                        require(I4_cell["strict_True_observed_exact_repeat_bytes"] is True and all(I4_cell["served_metrics"][label]["saved_selected_hits50_equal"] for label in ("True1", "True2")) and I4_cell["I4_reference_mean"]["saved_selected_hits50_equal"] is False and all(member[pool]["available"] and member[pool]["label"] == "False_anchor" for member in I4_cell["member_references"] for pool in ("positive", "negative")), "Inconsistent authenticated I4 mean did not propagate independently of strict I4 candidate metrics")
                    if name in ("wrong_query_row", "wrong_dtype", "late_after_scorer"):
                        assert_unvalidated_return(current, slot=120 if name == "late_after_scorer" else 1, expected_returned_rows=120 if name == "late_after_scorer" else 1, expected_validated_rows=119 if name == "late_after_scorer" else 0)
                        if name == "wrong_dtype":
                            details = json.loads((current["output"] / "PRIVATE_AUDIT_DETAILS.json").read_text())
                            require(details["slots"][0]["post_guard_receipt"]["wall_seconds"] == {"nonfinite_float": "nan"} and type(details["slots"][0]["returned_original_receipt"]["wall_seconds"]) is float, "Malformed post-guard receipt was not safely retained without changing the original return snapshot")
                active_case["status"] = "PASS"
                result["actual_work"] = sum_case_work(cases); atomic_json(output / "QUALIFICATION.json", result)
            require({r["case"] for r in cases} == set(plan["full_core_cases"]) | set(plan["shared_contract_cases"]) and all(r["status"] == "PASS" for r in cases), "Fixed qualification case coverage differs")
            final_context = {**fixture["context"], "release_path": context["release_path"]}
            result["final_admitted_input_custody"] = admitted_input_custody(final_context)
            verify_manifest(fixture["fixture_root"], fixture["spec"]["fixture_manifest_sha256"])
        result["status"], code = "PASS", 0
    except Exception as error:
        if active_case is not None:
            active_case["status"] = "FAILED"
            active_case["failure"] = {"exception_type": type(error).__name__, "condition": str(error)}
            if current is not None and (current["output"] / "AUDIT_ATTEMPT.json").exists():
                active_case["work"] = json.loads((current["output"] / "AUDIT_ATTEMPT.json").read_text())["work"]
        result.update(status="FAILED", failure={"exception_type": type(error).__name__, "condition": str(error)})
    finally:
        if torch is not None:
            try:
                torch.cuda.synchronize(0)
                result["qualification_cuda_peak_allocated_bytes"] = max([r["allocated"] for r in peak_rows] + [torch.cuda.max_memory_allocated(0)])
                result["qualification_cuda_peak_reserved_bytes"] = max([r["reserved"] for r in peak_rows] + [torch.cuda.max_memory_reserved(0)])
            except Exception as error:
                result.update(status="FAILED", CUDA_accounting_failure={"exception_type": type(error).__name__, "condition": str(error)}); code = 1
        result.update(case_count=len(cases), actual_work=sum_case_work(cases), inclusive_wall_seconds=perf_counter() - started,
                      terminal_write_tail_measured=False, universal_kernel_determinism_certified=False, scientific_equivalence_certified=False)
        atomic_json(output / "QUALIFICATION.json", result)
    return code


def main():
    parser = ArgumentParser(description=__doc__); parser.add_argument("--root-release", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); return run(preflight(args.root_release, args.output), perf_counter())


if __name__ == "__main__": raise SystemExit(main())
