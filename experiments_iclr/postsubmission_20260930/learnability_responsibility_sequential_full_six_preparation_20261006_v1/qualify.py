"""Disabled all-six full-context FP32 engineering worker; no fit or scoring.

Exact SHA-bound V2 synthetic and one-LIVE resource passes are prerequisites.
Six episodes share the unchanged initial state. Every commit is independently
recomputed at returned theta+ from ORIGINAL phi; no full monolithic/FD retry.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import signal
import socket
import sys
import time
import traceback

SOURCE_RELEASED = False
ROOT_QUALIFIER = ("learnability_responsibility_sequential_numerical_preparation_20261006_v2/qualify.py",
                  "3fdf78eaabb73868510f38dbf5de88e976364c0f03996d5325f23e3bd10dc5ae")
BASE = ("learnability_responsibility_native_numerical_worker_preparation_20261005_v3/qualify.py",
        "baeac626bade87bd286fe7a93a95602d8dc1f57d3257d0eeeb7e4a8e3823c688")
SEQUENTIAL = ("learnability_responsibility_sequential_autograd_vjp_preparation_20261006_v1/sequential_vjp.py",
              "1f9e704a1b95a98cec688d695094058594758a99bac2f057e2fe90e7f9315167")
SCIENTIFIC = ("amazon_learnability_responsibility_train_only_execution_preparation_20261005_v3/six_arm_worker.py",
              "68c7902f340432fe3e7d04593244f6af84ebe6c30968af4fac93ce38fd53d53a")
ACCESSOR = ("amazon_learnability_responsibility_engineering_accessor_20261005_v1/train_only_accessor.py",
            "9360e69753b362eb66ac89f133f00addaf15ab0e8c56314c75d7dd5e96fecb85")
PUBLIC_B_MANIFEST = {"sha256": "da03b6c14615da934443c9b8f36b68f9d643e2b23e04ebe17dab7dacf630df55",
                     "bytes": 1568}
PREREQUISITES = {
    "synthetic": {"path": "learnability_responsibility_sequential_synthetic_execution_root_20261006_v1/RESULT.json",
                  "sha256": "e11754ad266b52cc5ddf8efd75087a5791976ac7105d33c6281b1ab6702f9674", "bytes": 29024},
    "one_live": {"path": "learnability_responsibility_sequential_full_execution_root_20261006_v1/RESULT.json",
                 "sha256": "ac67aa8eb33861a9e1c171d12404c8977bdcc02196bdb6e303ad06fdbeb096c2", "bytes": 11125},
}
CONTROLS = ("live", "uniform", "margins", "graph_free", "permuted", "stop_q")
RECOMPUTE_COUNTS = {"native_forward_calls": 12, "private_gradient_calls": 8,
    "native_vjp_calls": 0, "q_map_primal_calls": 10, "q_map_vjp_calls": 0, "small_query_vjp_calls": 0}
TOTAL_COUNTS = {"native_forward_calls": 328, "private_gradient_calls": 180,
    "native_vjp_calls": 52, "q_map_primal_calls": 220, "q_map_vjp_calls": 40, "small_query_vjp_calls": 6}


def load(name, path, digest):
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("Pinned source differs: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_prerequisites(phase, pins):
    result = {}
    for name, row in PREREQUISITES.items():
        data = (phase / row["path"]).read_bytes()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("Exact prerequisite receipt differs: " + name)
        prior = json.loads(data)
        if (prior["worker_sha256"] != ROOT_QUALIFIER[1] or prior["candidate_pin"] != list(SEQUENTIAL)
                or prior["original_qualifier_pin"] != list(BASE)
                or prior["source_pins"] != {key: list(value) for key, value in pins.items()}
                or prior["caller_states_and_flags_and_gates_unchanged"] is not True
                or prior["restoration_on_exit"]["native_parameters_modes_attributes_buffers_inputs_and_rng_unchanged"] is not True
                or prior["model_fits"] != 0 or prior["persistent_updates"] != 0):
            raise ValueError("Prerequisite identity/restoration differs: " + name)
        if name == "synthetic":
            required = {"full_coordinate_parity_" + label for label in CONTROLS}
            required.add("unchanged_same_state_fine_directional_FD_stop_fixedQ_and_unused")
            if (prior["status"] != "PASS_SYNTHETIC_SEQUENTIAL_PARITY_ONLY" or prior["mode"] != "synthetic"
                    or prior["nodes"] != 15 or prior["dtype"] != "torch.float64" or prior["device"] != "cpu"
                    or prior["all_six_controls_compared"] is not True
                    or not required.issubset({check["name"] for check in prior["checks"]})):
                raise ValueError("Exact all-six synthetic pass required")
        else:
            checks = {check["name"]: check["result"] for check in prior["checks"]}
            live = checks["complete_full_graph_sequential_higher_order_state_map"]
            if (prior["status"] != "PASS_FULL_SEQUENTIAL_RESOURCE_ONLY" or prior["mode"] != "full"
                    or prior["nodes"] != 24492 or prior["dtype"] != "torch.float32"
                    or prior["device"] != "cuda:0" or live["complete_live_episode"] is not True
                    or live["virtual_query_loss_finite"] is not True or not live["min_assignment"] > 0):
                raise ValueError("Exact one-LIVE full resource pass required")
        result[name] = dict(row)
    return result


def compare(actual, expected):
    """Fixed original FP32 recommit tolerance; all entries and finite values."""
    if isinstance(expected, torch.Tensor):
        if (not isinstance(actual, torch.Tensor) or actual.shape != expected.shape
                or actual.dtype != expected.dtype or actual.device != expected.device
                or not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all())):
            raise AssertionError("Recomputed tensor structure/finite values differ")
        if not expected.is_floating_point():
            if not torch.equal(actual, expected):
                raise AssertionError("Recomputed integer tensor differs")
            return 0.0
        return base.close(actual, expected, atol=2e-6, rtol=2e-5)
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError("Recomputed dictionary keys differ")
        return max((compare(actual[key], expected[key]) for key in expected), default=0.0)
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            raise AssertionError("Recomputed sequence structure differs")
        return max((compare(left, right) for left, right in zip(actual, expected)), default=0.0)
    if actual != expected:
        raise AssertionError("Recomputed scalar metadata differs")
    return 0.0


def validate_episode(op, theta, phis, nt, np_, info, inspection):
    # These objective/diagnostic/assignment checks retain reviewed V2 semantics.
    if inspection is not None:
        raise AssertionError("Full resource mode retained engineering inspection")
    if not bool(torch.isfinite(info["virtual_query_loss"])):
        raise AssertionError("Nonfinite complete virtual query objective")
    for row, q in zip(info["pairs"], info["assignments"]):
        if not bool(torch.isfinite(q).all()) or not bool(q.min() > 0):
            raise AssertionError("Actual complete assignment is nonfinite/nonpositive")
        for value in row.values():
            if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                raise AssertionError("Nonfinite actual assignment diagnostic")
        if float(row["row_residual"]) > 3e-6 or float(row["column_residual"]) > 2e-3:
            raise AssertionError("Actual complete assignment lost fixed feasibility tolerance")
    if not bool(torch.isfinite(info["probe_own_ce_change"]).all()):
        raise AssertionError("Nonfinite actual private-probe CE change")
    gradient = {name: (theta[name] - value) / op.Config().eta_core for name, value in nt.items()}
    if not any(bool(torch.count_nonzero(value)) for name, value in gradient.items()
               if not name.startswith("local_head.")):
        raise AssertionError("Full sequential state map made no active shared change")
    if any(not bool(torch.isfinite(value).all()) for bank in (nt, *np_) for value in bank.values()):
        raise AssertionError("Nonfinite complete output state")
    base.zero_unused(gradient, phis, np_)


def run(args, receipt, check):
    global torch, base
    repo, phase = Path(args.repository).resolve(), Path(args.source_root).resolve()
    runtime = repo / "experiments_iclr/postsubmission_20260930/native_ncn_runtime_20261005_v1/.venv/bin/python"
    if socket.gethostname() != "anogena-2-0" or Path(sys.executable).absolute() != runtime or not sys.dont_write_bytecode:
        raise RuntimeError("Wrong normal host/runtime or missing -B")
    root_worker = load("full_six_root_qualifier", phase / ROOT_QUALIFIER[0], ROOT_QUALIFIER[1])
    base = load("full_six_original_qualifier", phase / BASE[0], BASE[1])
    if root_worker.BASE != BASE or root_worker.SEQUENTIAL != SEQUENTIAL:
        raise ValueError("Reviewed V2 prerequisite source pins differ")
    receipt["prerequisite_receipts"] = read_prerequisites(phase, base.PINS)
    site = repo / ".venv/lib/python3.11/site-packages"
    if not site.is_dir():
        raise RuntimeError("Missing normal runtime; no fallback")
    sys.path.insert(0, str(site))
    import torch
    import numpy
    if not Path(torch.__file__).resolve().is_relative_to(repo):
        raise RuntimeError("Unexpected Torch runtime")
    base.torch, base.np = torch, numpy
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if str(device) != "cuda:0":
        raise ValueError("Same successful one-LIVE cuda:0 FP32 context required")
    modules = {name: load("full_six_" + name, phase / path, digest)
               for name, (path, digest) in base.PINS.items()}
    op, port = modules["operator"], modules["port"]
    seq = load("full_six_candidate", phase / SEQUENTIAL[0], SEQUENTIAL[1])
    scientific = load("full_six_fixed_permutation", phase / SCIENTIFIC[0], SCIENTIFIC[1])
    accessor = load("full_six_safe_public_b", phase / ACCESSOR[0], ACCESSOR[1])
    if (SOURCE_RELEASED is not False or op.SOURCE_RELEASED is not False or port.PORT_RELEASED is not False
            or seq.SOURCE_RELEASED is not False or scientific.SOURCE_RELEASED is not False
            or accessor.SOURCE_RELEASED is not True):
        raise AssertionError("Original/reviewed process release gates differ")
    projection = Path(args.public_b_dir).resolve()
    manifest_bytes = (projection / "PUBLIC_B_MANIFEST.json").read_bytes()
    if (args.public_b_sha != PUBLIC_B_MANIFEST["sha256"] or len(manifest_bytes) != PUBLIC_B_MANIFEST["bytes"]
            or hashlib.sha256(manifest_bytes).hexdigest() != PUBLIC_B_MANIFEST["sha256"]):
        raise ValueError("Root-bound existing public+B projection differs")
    data = accessor.load_public_b(phase, projection, device=str(device))
    x, edges = data["features"], data["edge_index"]
    s, ys, r, yr = data["inner_indices"], data["inner_labels"], data["query_indices"], data["query_labels"]
    if (x.shape != (24492, 300) or x.dtype != torch.float32 or s.numel() != 2449 or r.numel() != 2450
            or data["W_ids"].numel() != 4898 or data["B_ids"].numel() != 9797
            or bool(torch.isin(torch.cat((s, r)), data["A_ids"]).any())):
        raise ValueError("Complete existing safe native context/roles differ")
    receipt["source_pins"] = base.PINS
    receipt["public_B_binding"] = {"directory": str(projection), "manifest_sha256": args.public_b_sha,
                                  "accessor_pin": ACCESSOR, "provenance": data["provenance"]}
    family = base.build_family(modules["native"], modules["boundary"], device, torch.float32, 17)
    forward, theta, phis, _ = port._native_callback_and_state(family, x, edges, expected_nodes=24492, global_stage=True)
    pairs = port._sparse_pairs(op, s, ys, edges, node_count=24492, dtype=torch.float32)
    before = base.snapshot(family, x, edges, device)
    input_before = ({name: value.clone() for name, value in theta.items()},
                    tuple({name: value.clone() for name, value in phi.items()} for phi in phis))
    flags = ([value.requires_grad for value in theta.values()],
             [[value.requires_grad for value in phi.values()] for phi in phis])
    episode_meter, recompute_meter = seq.Counters(), seq.Counters()
    total = dict.fromkeys(TOTAL_COUNTS, 0)

    def restore():
        result = base.unchanged(before, family, x, edges, device)
        if flags != ([value.requires_grad for value in theta.values()],
                     [[value.requires_grad for value in phi.values()] for phi in phis]):
            raise AssertionError("Caller requires_grad flags changed")
        if any(not torch.equal(input_before[0][name], theta[name]) for name in theta):
            raise AssertionError("Caller shared input mutated")
        if any(not torch.equal(old[name], phi[name]) for old, phi in zip(input_before[1], phis) for name in phi):
            raise AssertionError("Caller private input mutated")
        if (SOURCE_RELEASED is not False or op.SOURCE_RELEASED is not False or port.PORT_RELEASED is not False or seq.SOURCE_RELEASED is not False
                or scientific.SOURCE_RELEASED is not False or accessor.SOURCE_RELEASED is not True):
            raise AssertionError("Original/reviewed process gates changed")
        return result

    def arm(label, control, selected_pairs):
        calls = [0]
        def counted(core, private):
            calls[0] += 1
            return forward(core, private)
        nt, np_, info, inspection = seq._engineering_episode(op, base.PINS["operator"][1],
            port, base.PINS["port"][1], theta, phis, counted, selected_pairs, s, ys, r, yr,
            control=control, engineering_authorized=True, counters=episode_meter, collect_inspection=False)
        expected = seq.expected_episode_counts(control)
        if episode_meter.total != expected or calls[0] != expected["native_forward_calls"]:
            raise AssertionError("Actual episode/callback counters differ")
        validate_episode(op, theta, phis, nt, np_, info, inspection)
        # Fresh engine/value passes at returned theta+, always ORIGINAL private rows.
        calls[0] = 0
        fresh = seq._engineering_response(op, base.PINS["operator"][1], port, base.PINS["port"][1],
            nt, phis, counted, selected_pairs, s, ys, r, yr, control=control,
            engineering_authorized=True, counters=recompute_meter)
        if recompute_meter.total != RECOMPUTE_COUNTS or calls[0] != RECOMPUTE_COUNTS["native_forward_calls"]:
            raise AssertionError("Actual independent response/callback counters differ")
        state_error = compare(np_, fresh["adapted"])
        fresh_info = fresh["response"].diagnostics
        diagnostic_error = compare({key: info[key] for key in fresh_info}, fresh_info)
        base.zero_unused({name: (theta[name] - nt[name]) / op.Config().eta_core for name in theta}, phis, np_)
        restoration = restore()
        for key in total:
            total[key] += episode_meter.total[key] + recompute_meter.total[key]
        return {"arm": label, "control": control, "private_recommit_max_error": state_error,
                "all_original_committed_diagnostic_max_error": diagnostic_error,
                "episode_counters": episode_meter.snapshot(), "fresh_response_counters": recompute_meter.snapshot(),
                "fixed_FP32_atol": 2e-6, "fixed_FP32_rtol": 2e-5,
                "all_committed_private_coordinates": sum(value.numel() for phi in np_ for value in phi.values()),
                "virtual_query_loss": float(info["virtual_query_loss"]),
                "min_assignment": min(float(value.min()) for value in info["assignments"]),
                "same_initial_state_restored": restoration, "constructed_states_discarded": True}

    try:
        permutation = scientific._permutation(data)
        receipt["fixed_permutation"] = {"source_pin": SCIENTIFIC,
            "sha256": hashlib.sha256(permutation.detach().cpu().numpy().tobytes()).hexdigest(),
            "same_label": bool(torch.equal(ys[permutation], ys)), "node_count": int(permutation.numel())}
        permuted = port._sparse_pairs(op, s, ys, edges, node_count=24492, dtype=torch.float32,
                                     affinity_permutation=permutation)
        check("additional_original_sparse_products_and_assignment_feasibility", lambda:
            base.reference_products_and_assignment(op, pairs, dtype=torch.float32, device=device))
        for label in CONTROLS:
            episode_meter, recompute_meter = seq.Counters(), seq.Counters()
            receipt["last_attempted_arm"] = label
            control = "live" if label == "permuted" else label
            selected = permuted if label == "permuted" else pairs
            check("complete_full_FP32_episode_and_original_phi_recommit_" + label,
                  lambda l=label, c=control, p=selected: arm(l, c, p))
        if total != TOTAL_COUNTS:
            raise AssertionError("Complete six-arm plus recommit totals differ")
    finally:
        receipt["attempted_episode_counters"] = episode_meter.snapshot()
        receipt["attempted_recompute_counters"] = recompute_meter.snapshot()
        receipt["completed_arm_counter_totals"] = dict(total)
        try:
            receipt["restoration_on_exit"] = restore()
            receipt["caller_states_flags_and_gates_unchanged"] = True
        except BaseException:
            receipt["restoration_error"] = traceback.format_exc()
            raise
    return {"nodes": 24492, "dtype": "torch.float32", "device": str(device), "seed": 17,
            "all_six_full_FP32_episodes_and_original_phi_recommits": True,
            "actual_complete_counter_totals": total, "additional_sparse_oracle_outside_counter_totals": True,
            "full_monolithic_higher_order_retry": False, "full_graph_FD": False,
            "A_label_access": False, "B_labels_only": True, "scientific_fit_or_predictive_utility_qualified": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-authorized", action="store_true")
    parser.add_argument("--source-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--repository", default="/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output")
    parser.add_argument("--public-b-dir")
    parser.add_argument("--public-b-sha", default=PUBLIC_B_MANIFEST["sha256"])
    args = parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({"status": "DISABLED_FULL_SIX_ENGINEERING_WORKER", "numeric_imports": False}))
        return
    if not all((args.output, args.public_b_dir, args.public_b_sha)):
        parser.error("Fresh output and root-bound existing public+B directory/manifest SHA required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "RUNNING", "UTC": datetime.now(timezone.utc).isoformat(),
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidate_pin": SEQUENTIAL, "original_qualifier_pin": BASE, "root_qualifier_pin": ROOT_QUALIFIER,
        "hostname": socket.gethostname(), "python": sys.executable, "checks": [],
        "fixed_total_deadline_seconds": 900, "model_fits": 0, "persistent_updates": 0,
        "A_scoring": False, "VALID_TEST_access": False, "predictive_evidence": False}
    started = time.monotonic()
    def save():
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        (output / "RESULT.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    def check(name, function):
        receipt["last_started_check"] = name
        save()
        beginning = time.monotonic()
        result = function()
        receipt["checks"].append({"name": name, "elapsed_seconds": time.monotonic() - beginning, "result": result})
        save()
    def expired(signum, frame):
        raise TimeoutError("Fixed 900-second full-six engineering deadline exceeded")
    previous_handler = signal.signal(signal.SIGALRM, expired)
    signal.alarm(900)
    try:
        save()
        receipt.update(run(args, receipt, check))
        receipt["status"] = "PASS_FULL_SIX_FP32_EPISODE_RECOMMIT_RESOURCE_ONLY"
    except BaseException as error:
        receipt.update(status="FAIL_ENGINEERING", error_type=type(error).__name__,
                       error=str(error), traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        if "torch" in globals() and torch.cuda.is_initialized():
            receipt["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
            receipt["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(args.device)
        save()
    print(json.dumps({"status": receipt["status"], "checks_completed": len(receipt["checks"]),
                      "elapsed_seconds": receipt["elapsed_seconds"], "error": receipt.get("error")}))
    raise SystemExit(0 if receipt["status"].startswith("PASS_") else 1)


if __name__ == "__main__":
    main()
