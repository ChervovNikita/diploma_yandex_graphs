"""Disabled fixed same-state LIVE diagnostic; no qualification or fitting.

One complete LIVE episode, three fresh responses at its single returned theta+
from ORIGINAL phis, then two direct complete logits for EACH original member.
Tolerance failures are described, never relaxed, promoted, or retried.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import sys
import time
import traceback

SOURCE_RELEASED = False
FULL_SIX = ("learnability_responsibility_sequential_full_six_preparation_20261006_v1/qualify.py",
            "67a898c606d1e94bf680c84023d832ac73d4620081e79fd232800459b041a780")
BASE = ("learnability_responsibility_native_numerical_worker_preparation_20261005_v3/qualify.py",
        "baeac626bade87bd286fe7a93a95602d8dc1f57d3257d0eeeb7e4a8e3823c688")
ROOT_QUALIFIER = ("learnability_responsibility_sequential_numerical_preparation_20261006_v2/qualify.py",
                  "3fdf78eaabb73868510f38dbf5de88e976364c0f03996d5325f23e3bd10dc5ae")
SEQUENTIAL = ("learnability_responsibility_sequential_autograd_vjp_preparation_20261006_v1/sequential_vjp.py",
              "1f9e704a1b95a98cec688d695094058594758a99bac2f057e2fe90e7f9315167")
FAILURE = {"path": "learnability_responsibility_sequential_full_six_execution_root_20261006_v1/RESULT.json",
           "sha256": "e5826a3a38b7fe2a1d8b51abf7722488cd16182c1a70b028a1a629c5c063a6b0", "bytes": 20046}
ATOL, RTOL = 2e-6, 2e-5
DEADLINE = 900
RESPONSE_COUNTS = {"native_forward_calls": 12, "private_gradient_calls": 8,
    "native_vjp_calls": 0, "q_map_primal_calls": 10, "q_map_vjp_calls": 0, "small_query_vjp_calls": 0}
TOTAL_COUNTS = {"native_forward_calls": 92, "private_gradient_calls": 48,
    "native_vjp_calls": 12, "q_map_primal_calls": 60, "q_map_vjp_calls": 10, "small_query_vjp_calls": 1}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def append_json(path, row):
    # Each attempted operation/comparison closes its file before work continues.
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def coordinate(flat_index, shape):
    result = []
    for size in reversed(shape):
        result.append(flat_index % size)
        flat_index //= size
    return list(reversed(result))


def tensor_summary(actual, expected):
    row = {"kind": "tensor", "expected_shape": list(expected.shape),
           "expected_dtype": str(expected.dtype), "expected_device": str(expected.device)}
    if not isinstance(actual, torch.Tensor):
        return dict(row, structure_mismatch="actual is not a tensor", coordinates_compared=0)
    row.update(actual_shape=list(actual.shape), actual_dtype=str(actual.dtype), actual_device=str(actual.device))
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.device != expected.device:
        return dict(row, structure_mismatch="shape/dtype/device differs", coordinates_compared=0)
    a, e = actual.detach(), expected.detach()
    floating = expected.is_floating_point()
    violation = ~torch.isclose(a, e, atol=ATOL, rtol=RTOL) if floating else a != e
    finite_a, finite_e = torch.isfinite(a), torch.isfinite(e)
    row.update(coordinates_compared=a.numel(), tolerance_rule="torch.allclose(actual,expected,atol=2e-6,rtol=2e-5)" if floating else "exact integer equality",
               original_tolerance_pass=bool(torch.allclose(a, e, atol=ATOL, rtol=RTOL)) if floating else bool(torch.equal(a, e)),
               original_tolerance_violation_count=int(violation.sum()),
               actual_nonfinite_count=int((~finite_a).sum()), expected_nonfinite_count=int((~finite_e).sum()))
    ac, ec = a.cpu().double().reshape(-1), e.cpu().double().reshape(-1)
    delta = (ac - ec).abs()
    all_finite = bool(finite_a.all() and finite_e.all())
    row.update(actual_scale_max_abs=number(ac.abs().max()) if ac.numel() else 0.0,
               expected_scale_max_abs=number(ec.abs().max()) if ec.numel() else 0.0,
               actual_scale_rms=number(ac.square().mean().sqrt()) if ac.numel() else 0.0,
               expected_scale_rms=number(ec.square().mean().sqrt()) if ec.numel() else 0.0,
               abs_error_max=number(delta.max()) if delta.numel() else 0.0,
               abs_error_rms=number(delta.square().mean().sqrt()) if delta.numel() else 0.0)
    mask = violation.cpu().reshape(-1)
    limits = (ATOL + RTOL * e.abs()).cpu().double().reshape(-1) if floating else torch.zeros_like(ec)
    def sample(index):
        return {"index": coordinate(index, tuple(a.shape)), "flat_index": index,
                "actual": number(ac[index]), "expected": number(ec[index]),
                "abs_error": number(delta[index]), "original_tolerance_limit": number(limits[index]),
                "violates_original_tolerance": bool(mask[index])}
    indices = mask.nonzero().flatten()[:8].tolist()
    row["first_violating_coordinates"] = [sample(i) for i in indices]
    row["all_values_finite"] = all_finite
    if delta.numel() and all_finite:
        row["max_abs_coordinate"] = sample(int(delta.argmax()))
    return row


def compare_tree(actual, expected, label, path, output):
    """Exhaustive tree walk in original expected order; no numeric early abort."""
    summary = {"label": label, "root_path": path, "tensor_leaves": 0, "coordinates_compared": 0,
               "original_tolerance_violation_count": 0, "structure_mismatches": 0,
               "exact_scalar_metadata_mismatches": 0, "first_failing_path": None,
               "max_abs_error": 0.0, "max_abs_error_path": None}
    def record(row, where):
        row.update(comparison=label, path=where)
        append_json(output / "COMPARISON_COORDINATES.jsonl", row)
        if row["kind"] == "tensor":
            summary["tensor_leaves"] += 1
            summary["coordinates_compared"] += row.get("coordinates_compared", 0)
            summary["original_tolerance_violation_count"] += row.get("original_tolerance_violation_count", 0)
            error = row.get("abs_error_max")
            if isinstance(error, (int, float)) and error > summary["max_abs_error"]:
                summary["max_abs_error"], summary["max_abs_error_path"] = error, where
        if "structure_mismatch" in row:
            summary["structure_mismatches"] += 1
        if row.get("scalar_metadata_mismatch"):
            summary["exact_scalar_metadata_mismatches"] += 1
        failed = ("structure_mismatch" in row or row.get("scalar_metadata_mismatch")
                  or row.get("original_tolerance_violation_count", 0)
                  or row.get("actual_nonfinite_count", 0) or row.get("expected_nonfinite_count", 0))
        if failed and summary["first_failing_path"] is None:
            summary["first_failing_path"] = where
    def walk(a, e, where):
        if isinstance(e, torch.Tensor):
            record(tensor_summary(a, e), where)
        elif isinstance(e, dict):
            if not isinstance(a, dict):
                record({"kind": "structure", "structure_mismatch": "expected dictionary"}, where)
                return
            if a.keys() != e.keys():
                record({"kind": "structure", "structure_mismatch": "dictionary keys differ",
                        "missing": [str(k) for k in e if k not in a], "extra": [str(k) for k in a if k not in e]}, where)
            for key in e:
                if key in a:
                    walk(a[key], e[key], where + "[" + json.dumps(str(key)) + "]")
        elif isinstance(e, (tuple, list)):
            if not isinstance(a, type(e)) or len(a) != len(e):
                record({"kind": "structure", "structure_mismatch": "sequence kind/length differs"}, where)
                return
            for i, (left, right) in enumerate(zip(a, e)):
                walk(left, right, where + "[" + str(i) + "]")
        else:
            record({"kind": "scalar_metadata", "scalar_metadata_mismatch": a != e,
                    "actual": repr(a), "expected": repr(e)}, where)
    walk(actual, expected, path)
    return summary


def response_package(fresh):
    response = fresh["response"]
    return {"diagnostics": response.diagnostics, "adapted": fresh["adapted"],
            "raw_costs": response.raw_costs, "q_blocks": response.q_blocks,
            "probe_private_gradients": response.probe_private_gradients,
            "main_private_gradients": fresh["main_private_gradients"]}


def run(args, receipt, save):
    global torch
    repo, phase, output = Path(args.repository).resolve(), Path(args.source_root).resolve(), Path(args.output).resolve()
    runtime = repo / "experiments_iclr/postsubmission_20260930/native_ncn_runtime_20261005_v1/.venv/bin/python"
    require(socket.gethostname() == "anogena-2-0" and Path(sys.executable).absolute() == runtime
            and sys.dont_write_bytecode, "Wrong fixed host/runtime or missing -B")
    # This source uses the previously reviewed SHA-checking loader; it is not run during preparation.
    import importlib.util
    data = (phase / FULL_SIX[0]).read_bytes()
    require(hashlib.sha256(data).hexdigest() == FULL_SIX[1], "Failed full-six qualifier source differs")
    spec = importlib.util.spec_from_file_location("same_state_full_six", phase / FULL_SIX[0])
    full = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = full
    spec.loader.exec_module(full)
    require(full.BASE == BASE and full.ROOT_QUALIFIER == ROOT_QUALIFIER and full.SEQUENTIAL == SEQUENTIAL,
            "Failed qualifier dependency pins differ")
    base = full.load("same_state_base", phase / BASE[0], BASE[1])
    raw = (phase / FAILURE["path"]).read_bytes()
    require(len(raw) == FAILURE["bytes"] and hashlib.sha256(raw).hexdigest() == FAILURE["sha256"], "Original failure receipt differs")
    failed = json.loads(raw)
    require(failed["status"] == "FAIL_ENGINEERING" and failed["worker_sha256"] == FULL_SIX[1]
            and failed["candidate_pin"] == list(SEQUENTIAL) and failed["original_qualifier_pin"] == list(BASE)
            and failed["root_qualifier_pin"] == list(ROOT_QUALIFIER) and failed["last_attempted_arm"] == "live"
            and failed["source_pins"] == {key: list(value) for key, value in base.PINS.items()}
            and failed["error"] == "Tensor mismatch, max abs error=1.5035271644592285e-05"
            and failed["caller_states_flags_and_gates_unchanged"] is True
            and failed["model_fits"] == 0 and failed["persistent_updates"] == 0
            and failed["A_scoring"] is False and failed["VALID_TEST_access"] is False,
            "Exact original LIVE diagnostic failure required; do not promote it")
    receipt["preserved_original_failure"] = dict(FAILURE)
    receipt["original_failure_status"] = failed["status"]
    receipt["prerequisite_receipts"] = full.read_prerequisites(phase, base.PINS)
    site = repo / ".venv/lib/python3.11/site-packages"
    require(site.is_dir(), "Missing normal runtime; no fallback")
    sys.path.insert(0, str(site))
    import torch
    import numpy
    require(Path(torch.__file__).resolve().is_relative_to(repo), "Unexpected Torch runtime")
    base.torch, base.np = torch, numpy
    full.torch, full.base = torch, base
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    require(str(device) == "cuda:0", "Same fixed CUDA0 FP32 context required")
    modules = {name: full.load("same_state_" + name, phase / path, digest)
               for name, (path, digest) in base.PINS.items()}
    op, port = modules["operator"], modules["port"]
    seq = full.load("same_state_sequential", phase / SEQUENTIAL[0], SEQUENTIAL[1])
    accessor = full.load("same_state_accessor", phase / full.ACCESSOR[0], full.ACCESSOR[1])
    def gates():
        require(SOURCE_RELEASED is False and full.SOURCE_RELEASED is False and op.SOURCE_RELEASED is False
                and port.PORT_RELEASED is False and seq.SOURCE_RELEASED is False and accessor.SOURCE_RELEASED is True,
                "Existing source/process gates changed")
    gates()
    projection = Path(args.public_b_dir).resolve()
    manifest = (projection / "PUBLIC_B_MANIFEST.json").read_bytes()
    require(len(manifest) == full.PUBLIC_B_MANIFEST["bytes"]
            and hashlib.sha256(manifest).hexdigest() == full.PUBLIC_B_MANIFEST["sha256"], "Exact existing public+B projection differs")
    data = accessor.load_public_b(phase, projection, device=str(device))
    require(data["provenance"] == failed["public_B_binding"]["provenance"], "Original failed graph/preprocessing/roles/projection identity differs")
    x, edges = data["features"], data["edge_index"]
    s, ys, r, yr = data["inner_indices"], data["inner_labels"], data["query_indices"], data["query_labels"]
    require(x.shape == (24492, 300) and x.dtype == torch.float32 and s.numel() == 2449 and r.numel() == 2450
            and data["W_ids"].numel() == 4898 and data["B_ids"].numel() == 9797
            and not bool(torch.isin(torch.cat((s, r)), data["A_ids"]).any()), "Fixed full context/roles differ")
    receipt["source_pins"], receipt["public_B_provenance"] = base.PINS, data["provenance"]
    family = base.build_family(modules["native"], modules["boundary"], device, torch.float32, 17)
    forward, theta, phis, _ = port._native_callback_and_state(family, x, edges, expected_nodes=24492, global_stage=True)
    pairs = port._sparse_pairs(op, s, ys, edges, node_count=24492, dtype=torch.float32)
    before = base.snapshot(family, x, edges, device)
    originals = ({name: value.clone() for name, value in theta.items()},
                 tuple({name: value.clone() for name, value in phi.items()} for phi in phis))
    flags = ([v.requires_grad for v in theta.values()], [[v.requires_grad for v in phi.values()] for phi in phis])
    nt, theta_plus_before = None, None
    callback_counts, meters = {}, []
    phase_name = "episode"
    def event(kind, **fields):
        append_json(output / "ATTEMPTED_OPERATIONS.jsonl", {"kind": kind, "phase": phase_name,
                    "monotonic_seconds": time.monotonic(), **fields})
    def counted(core, private):
        callback_counts[phase_name] = callback_counts.get(phase_name, 0) + 1
        event("physical_native_callback_attempt")
        return forward(core, private)
    class DurableCounters(seq.Counters):
        def add(self, key, stage):
            super().add(key, stage)
            event("staged_operation_attempt", operation=key, stage=stage)
    def restore():
        result = base.unchanged(before, family, x, edges, device)
        require(flags == ([v.requires_grad for v in theta.values()], [[v.requires_grad for v in phi.values()] for phi in phis]), "Caller gradient flags changed")
        require(all(torch.equal(originals[0][n], v) for n, v in theta.items())
                and all(torch.equal(old[n], v) for old, phi in zip(originals[1], phis) for n, v in phi.items()), "Original theta/phi input mutated")
        if nt is not None and theta_plus_before is not None:
            require(all(torch.equal(theta_plus_before[n], v) for n, v in nt.items())
                    and all(not v.requires_grad for v in nt.values()), "Single returned theta+ mutated")
        gates()
        return result
    receipt["comparisons"] = []
    try:
        meter = DurableCounters(); meters.append(meter)
        event("phase_started"); save()
        nt, np_, info, inspection = seq._engineering_episode(op, base.PINS["operator"][1], port, base.PINS["port"][1],
            theta, phis, counted, pairs, s, ys, r, yr, control="live", engineering_authorized=True,
            counters=meter, collect_inspection=False)
        require(meter.total == seq.expected_episode_counts("live") and callback_counts["episode"] == 48, "LIVE episode accounting differs")
        full.validate_episode(op, theta, phis, nt, np_, info, inspection)
        theta_plus_before = {name: value.clone() for name, value in nt.items()}
        restore(); receipt["episode_completed"] = True; save()
        freshs = []
        for repeat in range(3):
            phase_name = "fresh_response_" + str(repeat + 1)
            meter = DurableCounters(); meters.append(meter)
            event("phase_started"); save()
            fresh = seq._engineering_response(op, base.PINS["operator"][1], port, base.PINS["port"][1],
                nt, phis, counted, pairs, s, ys, r, yr, control="live", engineering_authorized=True, counters=meter)
            require(meter.total == RESPONSE_COUNTS and callback_counts[phase_name] == 12, "Fresh response accounting differs")
            package = response_package(fresh); freshs.append(package)
            # The original qualifier compares private states FIRST, then projected diagnostics.
            receipt["comparisons"].append(compare_tree(np_, package["adapted"], "episode_vs_fresh_" + str(repeat + 1), "$private", output))
            expected = package["diagnostics"]
            receipt["comparisons"].append(compare_tree({k: info[k] for k in expected}, expected,
                "episode_vs_fresh_" + str(repeat + 1), "$diagnostics", output))
            restore(); receipt["fresh_responses_completed"] = repeat + 1; save()
            del fresh
        for left, right in ((0, 1), (0, 2), (1, 2)):
            receipt["comparisons"].append(compare_tree(freshs[left], freshs[right],
                "fresh_" + str(left + 1) + "_vs_fresh_" + str(right + 1), "$response_package", output))
            save()
        phase_name = "direct_original_member_logits"
        for member in range(4):
            event("direct_member_pair_started", member=member)
            with torch.no_grad():
                first = counted(nt, phis[member]).detach().clone()
                second = counted(nt, phis[member]).detach().clone()
            require(first.shape == second.shape == (24492, 5) and first.dtype == second.dtype == torch.float32, "Complete direct member logits required")
            receipt["comparisons"].append(compare_tree(first, second, "direct_repeat_member_" + str(member), "$native_logits[member=" + str(member) + "]", output))
            del first, second
            restore(); receipt["direct_member_pairs_completed"] = member + 1; save()
        total = {key: sum(m.total[key] for m in meters) for key in TOTAL_COUNTS}
        total["native_forward_calls"] += callback_counts["direct_original_member_logits"]
        require(total == TOTAL_COUNTS and sum(callback_counts.values()) == 92, "Complete diagnostic accounting differs")
        receipt["actual_complete_counts"] = total
        receipt["primal_assignment_iterations"] = 8 * total["q_map_primal_calls"]
        receipt["all_diagnostic_and_private_coordinates_compared"] = True
        del nt, np_, info, freshs, theta_plus_before, package, expected
        nt, theta_plus_before = None, None
        receipt["constructed_states_discarded"] = True
    finally:
        receipt["attempted_counter_snapshots"] = [m.snapshot() for m in meters]
        receipt["independent_physical_callback_attempts"] = dict(callback_counts)
        receipt["restoration_on_exit"] = restore()
        receipt["caller_states_flags_and_gates_unchanged"] = True
        save()
    return {"nodes": 24492, "features": 300, "dtype": "torch.float32", "device": "cuda:0", "seed": 17,
            "one_live_episode_only": True, "three_fresh_same_theta_plus_original_phi_responses": True,
            "two_direct_passes_each_of_four_original_members": True, "original_failure_promoted": False,
            "qualification_pass_claim": False, "scientific_fit_or_predictive_utility_qualified": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-authorized", action="store_true")
    parser.add_argument("--source-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--repository", default="/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output")
    parser.add_argument("--public-b-dir")
    args = parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({"status": "DISABLED_SAME_STATE_DIAGNOSTIC", "numeric_imports": False, "SOURCE_RELEASED": SOURCE_RELEASED}))
        return
    if not args.output or not args.public_b_dir:
        parser.error("Explicit future engineering execution requires fresh output and exact existing public+B projection")
    require(SOURCE_RELEASED is False, "This diagnostic never flips its source guard")
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = {"status": "RUNNING_DIAGNOSTIC_NOT_QUALIFICATION", "UTC": datetime.now(timezone.utc).isoformat(),
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "candidate_pin": SEQUENTIAL,
        "failed_qualifier_pin": FULL_SIX, "original_failure": FAILURE, "fixed_total_deadline_seconds": DEADLINE,
        "fixed_FP32_atol": ATOL, "fixed_FP32_rtol": RTOL, "planned_counts": TOTAL_COUNTS,
        "hostname": socket.gethostname(), "python": sys.executable, "model_fits": 0, "persistent_updates": 0,
        "A_scoring": False, "VALID_TEST_access": False, "predictive_evidence": False,
        "qualification_pass_claim": False, "original_failure_promoted": False}
    def save():
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if "torch" in globals() and torch.cuda.is_initialized():
            receipt["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
            receipt["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(args.device)
        temporary = output / "RESULT.tmp"
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, output / "RESULT.json")
    def expired(signum, frame):
        raise TimeoutError("Fixed 900-second same-state diagnostic deadline exceeded")
    previous = signal.signal(signal.SIGALRM, expired); signal.alarm(DEADLINE)
    try:
        save(); receipt.update(run(args, receipt, save))
        receipt["status"] = "DIAGNOSTIC_COMPLETE_NO_QUALIFICATION"
    except BaseException as error:
        receipt.update(status="FAIL_DIAGNOSTIC_EXECUTION", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
        save()
        for path in output.iterdir():
            path.chmod(0o444)
    print(json.dumps({"status": receipt["status"], "elapsed_seconds": receipt["elapsed_seconds"], "error": receipt.get("error"),
                      "qualification_pass_claim": False, "original_failure_promoted": False}))
    raise SystemExit(0 if receipt["status"] == "DIAGNOSTIC_COMPLETE_NO_QUALIFICATION" else 1)


if __name__ == "__main__":
    main()
