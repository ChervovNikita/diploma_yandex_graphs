#!/usr/bin/env python3
"""Separately root-authorized ordinary-runtime QA on fabricated inputs only."""
from argparse import ArgumentParser
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import json
import subprocess
import sys
from replay_gate import require, descriptor, ARMS
from replay_run import execute, Accounting, atomic_json
from replay_numeric import original_modules
from replay_synthetic_admission import preflight, admitted_input_custody
from replay_synthetic_fixtures import build
from replay_synthetic_cases import (clone_fixture, alter_own_payload, corrupt_after_metadata,
                                    corrupt_admitted_release, assert_failed_suppression)


def resource_recheck(context):
    release = context["release"]
    command = ["nvidia-smi", "-i", release["cuda_visible_devices"], "--query-gpu=uuid,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    fields = [v.strip() for v in result.stdout.strip().split(",")]
    require(len(fields) == 4 and fields[0] == release["cuda_visible_devices"], "Synthetic dispatch GPU identity differs")
    memory = {"UUID": fields[0], "total_MiB": int(fields[1]), "used_MiB": int(fields[2]), "free_MiB": int(fields[3])}
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    require(memory["free_MiB"] >= release["minimum_GPU_free_MiB"] and available >= release["minimum_host_MemAvailable_bytes"], "Synthetic resource floor is not available; no retry/job interference")
    return {"UTC": datetime.now(timezone.utc).isoformat(), "GPU_memory": memory, "host_MemAvailable_bytes": available, "no_other_job_mutation": True}


def direct_api_checks(api, mods, context, device):
    import torch
    evaluate = api["pilot_evaluate"]
    metric = evaluate.evaluator(context)
    negative = torch.arange(100000, dtype=torch.float32)
    positive = torch.tensor([-1., 99950., 99951., 100001.]).repeat(15021)
    require(evaluate.hits50(metric, positive, negative) == .5, "Original official strict50th-negative tie rule differs")
    for p, n in ((positive[:-1], negative), (positive.double(), negative), (positive, negative[:-1])):
        try:
            evaluate.hits50(metric, p, n)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Original official invalid coverage/dtype was accepted")
    rows = [(torch.full((60084,), 10. if i == 0 else 0.), torch.ones(100000), {}) for i in range(4)]
    pooled = evaluate.mean_native_scores(rows)
    require(evaluate.hits50(metric, *pooled) == 1. and sum(evaluate.hits50(metric, row[0], row[1]) for row in rows) / 4 == .25, "Original I4 raw-logit mean must precede Hits50")
    iterator = mods["native_utils"].PermIterator(device, 131079, 131072, False)
    batches = list(iterator)
    require([len(b) for b in batches] == [131072, 7] and torch.equal(torch.cat(batches), torch.arange(131079, device=device)), "Original canonical evaluation order/tail differs")
    return ["original_official_Hits50_full_pools_strict_ties_and_invalid_coverage_dtype",
            "original_equal_raw_logit_mean_before_metric", "original_canonical_eval_iterator_and131072_plus7_tail"]


def run(context, started):
    output = context["output"]
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    counts = {"actual_engineering_updates": 0, "reference_score_valid_calls": 0,
              "attempted_score_valid_calls": 0, "actual_original_score_valid_invocations": 0}
    checks, cases, peak_rows = [], [], []
    result = {"schema": "ncnc-selected-state-replay-synthetic-qualification-v1", "status": "IN_PROGRESS",
              "identity": context["original_identity"], "sidecar_manifest_sha256": context["sidecar_manifest_sha256"],
              "root_release": descriptor(context["release_path"]), "authorized_invocation": context["release"]["authorized_invocations"][0],
              "source_entry": descriptor(Path(__file__).resolve()), "actual_argv": list(sys.argv),
              "runtime_authority": context["release"]["runtime_authority"], "fabricated_inputs_only": True,
              "study_lock_data_outcome_checkpoint_accessed": False, "TEST_opened": False,
              "scientific_fit_updates": 0, "checks": checks, "cases": cases, "actual_work": counts,
              "metadata100epochs_101candidates_not_execution_claim": True, "automatic_retry": False}
    atomic_json(output / "QUALIFICATION.json", result)
    code = 1
    torch = None
    try:
        result["dispatch_resource_observation"] = resource_recheck(context)
        api = original_modules(context)
        api["pilot_common"].runtime_stdlib(context)
        device, _ = api["pilot_model"].runtime(context)
        import torch as torch_module
        torch = torch_module
        torch.cuda.reset_peak_memory_stats(0)
        mods = api["pilot_model"].modules(context)
        mods["design"].validate_plan(context["plan"])
        checks.extend(direct_api_checks(api, mods, context, device))
        original_score = api["pilot_evaluate"].score_valid
        state = {"fault": None, "case_calls": 0, "routes": {}}
        def measured_score(instance, data, current_mods, **kwargs):
            counts["attempted_score_valid_calls"] += 1
            state["case_calls"] += 1
            if state["fault"] == "scoring" and state["case_calls"] == 40:
                raise RuntimeError("ADMITTED_SYNTHETIC_SCORING_FAULT")
            total = sum(p.numel() for _, p in api["pilot_model"].named_parameters(instance))
            route = "native64" if total == 38147 else "native70" if total == 44663 else kwargs.get("mode")
            require(route in ("native64", "native70", "private", "pooled_after_clamp"), "Unexpected actual model/mode route")
            state["routes"][route] = state["routes"].get(route, 0) + 1
            counts["actual_original_score_valid_invocations"] += 1
            return original_score(instance, data, current_mods, **kwargs)
        with patch.object(api["pilot_data"], "load_data", side_effect=RuntimeError("STUDY_DATA_ACCESS_FORBIDDEN_IN_QUALIFICATION")), patch.object(api["pilot_evaluate"], "score_valid", side_effect=measured_score):
            fixture = build(context, output / "fabricated_reference", api, mods, device, counts)
            peak_rows.append({"phase": "fabricated_reference_creation", "allocated_bytes": torch.cuda.max_memory_allocated(0), "reserved_bytes": torch.cuda.max_memory_reserved(0)})
            checks.append("actual_native64_native70_I4_privateF4_pooledF4_populated_Adam_mixed_flags_all_RNG_restoration_and_N64_distinct_I4_epoch")
            case_names = ("baseline_all25", "custody_after_metadata", "selected_state", "raw_score", "selected_metric", "scoring", "accounting", "admitted_release_after_scoring")
            for name in case_names:
                case_inputs = output / ("inputs_" + name)
                case_context = clone_fixture(fixture, case_inputs, api, torch)
                case_context["output"] = output / ("result_" + name)
                # An owned exact-byte copy allows a custody fault without changing the real root release.
                release_copy = case_inputs / "ADMITTED_SYNTHETIC_RELEASE_COPY.json"
                release_copy.write_bytes(context["release_path"].read_bytes())
                case_context["release_path"] = release_copy
                before = dict(counts)
                state.update(fault="scoring" if name == "scoring" else None, case_calls=0, routes={})
                if name in ("selected_state", "raw_score", "selected_metric"):
                    alter_own_payload(case_context, api, torch, name)
                with ExitStack() as stack:
                    if name == "accounting":
                        stack.enter_context(patch.object(Accounting, "finish", side_effect=RuntimeError("ADMITTED_SYNTHETIC_TERMINAL_ACCOUNTING_FAULT")))
                    case_code, case_result = execute(case_context, perf_counter(), fabricated_data=fixture["data"],
                        synthetic_after_metadata=corrupt_after_metadata if name == "custody_after_metadata" else None,
                        synthetic_before_custody=corrupt_admitted_release if name == "admitted_release_after_scoring" else None)
                call_delta = counts["attempted_score_valid_calls"] - before["attempted_score_valid_calls"]
                if name == "baseline_all25":
                    require(case_code == 0 and case_result["status"] == "ALL25_SELECTED_STATE_REPLAY_PASS" and len(case_result["cells"]) == 25 and all(c["status"] == "PASS" for c in case_result["cells"]) and case_result["summary"] is not None, "Actual fabricated all25 core replay did not pass")
                    require(call_delta == 40 and state["routes"] == {"native64": 25, "native70": 5, "private": 5, "pooled_after_clamp": 5}, "Actual all25/40 constructor/scoring/mode routing differs")
                    details = json.loads((case_context["output"] / "PRIVATE_REPLAY_DETAILS.json").read_text())
                    require(len(details["cells"]) == 25 and sum(r["score_valid_calls"] for r in details["cells"]) == 40 and all(r["selected_hits50_equal"] and all(m["original_score_digest_equal"] for m in r["member_receipts"]) for r in details["cells"]), "Actual raw-score/selected-Hits/nominal member receipts differ")
                    require(case_result["final_input_custody"]["output_freshness_rechecked"] is False and case_result["replay_accounting"]["inclusive_wall_seconds"] >= 0 and case_result["replay_cuda_peak_allocated_bytes"] > 0 and case_result["replay_cuda_peak_reserved_bytes"] >= case_result["replay_cuda_peak_allocated_bytes"], "Actual complete custody/accounting/CUDA peak checks differ")
                    checks.append("actual_production_core_all25_40_calls_full_pools_hashes_Hits_routing_disclosure_and_CUDA_accounting")
                else:
                    assert_failed_suppression(case_code, case_result)
                    require(call_delta == (0 if name in ("custody_after_metadata", "selected_state") else 40), "Injected case actual scoring work differs from reviewed plan")
                    if name == "accounting":
                        require(case_result["status"] == "FAILED_REPLAY_ACCOUNTING" and case_result["replay_accounting"]["inclusive_wall_seconds"] is None, "Accounting fault did not retain explicit unknown finalization cost")
                    checks.append("actual_shared_failure_retention_null_all25_" + name)
                cases.append({"case": name, "status": "PASS", "expected_child_exit": case_code,
                              "child_result": descriptor(case_context["output"] / "REPLAY_RESULT.json"),
                              "attempted_score_valid_calls": call_delta, "observed_routes": dict(state["routes"]),
                              "input_lock": case_context["release"]["family_lock"], "fabricated_inputs_only": True})
                peak_rows.append({"phase": name, "allocated_bytes": torch.cuda.max_memory_allocated(0), "reserved_bytes": torch.cuda.max_memory_reserved(0)})
                atomic_json(output / "QUALIFICATION.json", result)
            # Check the original admission once more; failed-case copies remain private evidence.
            final_context = {**fixture["context"], "release_path": context["release_path"]}
            result["final_admitted_input_custody"] = admitted_input_custody(final_context)
        require(counts == {"actual_engineering_updates": 40, "reference_score_valid_calls": 44,
                           "attempted_score_valid_calls": 284, "actual_original_score_valid_invocations": 283}, "Complete actual qualification work count differs; no fabricated PASS")
        result.update(status="PASS", check_count=len(checks), case_count=len(cases), original_sources_changed=False,
                      fixture_N64_I4_seed0_distinct_selected_snapshots=True)
        code = 0
    except Exception as error:
        result.update(status="FAILED", failure={"exception_type": type(error).__name__, "condition": str(error)},
                      check_count=len(checks), case_count=len(cases))
    finally:
        if torch is not None:
            try:
                torch.cuda.synchronize(0)
                result["qualification_cuda_peak_allocated_bytes"] = max([p["allocated_bytes"] for p in peak_rows] + [torch.cuda.max_memory_allocated(0)])
                result["qualification_cuda_peak_reserved_bytes"] = max([p["reserved_bytes"] for p in peak_rows] + [torch.cuda.max_memory_reserved(0)])
            except Exception as error:
                result.update(status="FAILED", CUDA_accounting_failure={"exception_type": type(error).__name__, "condition": str(error)})
                code = 1
        result["phase_peaks"] = peak_rows
        result["inclusive_qualification_wall_seconds"] = perf_counter() - started
        result["terminal_write_tail_measured"] = False
        result["UTC"] = datetime.now(timezone.utc).isoformat()
        atomic_json(output / "QUALIFICATION.json", result)
    print("SYNTHETIC_QUALIFICATION_TERMINAL status=" + result["status"] + " cases=" + str(len(cases)) + " receipt=" + str(output / "QUALIFICATION.json"), flush=True)
    return code


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root-release", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = perf_counter()
    context = preflight(args.root_release, args.output)
    return run(context, started)


if __name__ == "__main__":
    raise SystemExit(main())
