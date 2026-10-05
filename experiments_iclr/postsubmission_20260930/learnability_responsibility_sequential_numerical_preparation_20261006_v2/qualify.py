"""Frozen root-owned sequential/native engineering qualification; never a fit.

Synthetic mode compares the complete native architecture at the unchanged
15-node float64 fixture. Full mode runs one exact float32 public/B episode
after the synthetic pass. No evaluator, checkpoint or model-state commit.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import signal
import socket
import sys
import time
import traceback

BASE = ("learnability_responsibility_native_numerical_worker_preparation_20261005_v3/qualify.py",
        "baeac626bade87bd286fe7a93a95602d8dc1f57d3257d0eeeb7e4a8e3823c688")
SEQUENTIAL = ("learnability_responsibility_sequential_autograd_vjp_preparation_20261006_v1/sequential_vjp.py",
              "1f9e704a1b95a98cec688d695094058594758a99bac2f057e2fe90e7f9315167")
CONTROLS = ("live", "uniform", "margins", "graph_free", "permuted", "stop_q")


def load(name, path, expected):
    import importlib.util
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Source differs: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compare(actual, expected, *, gradient=False):
    """Fixed inherited all-coordinate tolerance, with finite values required."""
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise AssertionError("Tensor structure differs")
        if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
            raise AssertionError("Nonfinite compared tensor")
        if not expected.is_floating_point():
            if not torch.equal(actual, expected):
                raise AssertionError("Integer tensor differs")
            return 0.0
        return base.close(actual, expected, atol=1e-8 if gradient else 1e-10,
                          rtol=1e-6 if gradient else 1e-8)
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError("Dictionary keys differ")
        return max((compare(actual[k], expected[k], gradient=gradient) for k in expected), default=0.0)
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            raise AssertionError("Sequence structure differs")
        return max((compare(a, b, gradient=gradient) for a, b in zip(actual, expected)), default=0.0)
    if actual != expected:
        raise AssertionError("Scalar/metadata differs: " + repr((actual, expected)))
    return 0.0


def reference_details(op, theta, phis, forward, pairs, s, ys, control):
    """Independent oracle recomputation using only original math functions."""
    cfg = op.Config()
    before = torch.stack([forward(theta, phi)[s] for phi in phis])
    probes = tuple(torch.func.grad(op._own_ce, argnums=1)(theta, phi, forward, s, ys)
                   for phi in phis)
    probe_states = tuple(op._sgd(phi, g, cfg.eta_probe) for phi, g in zip(phis, probes))
    after = torch.stack([forward(theta, phi)[s] for phi in probe_states])
    raw = tuple(op._margin(before, pair) if control == "margins" else
                op._margin(after, pair) - op._margin(before, pair) for pair in pairs)
    q = tuple(op._assignment_map(op._normalized_cost(value, cfg.response_epsilon)[0],
                                pair.laplacian, cfg, 0.0 if control == "graph_free" else cfg.graph)
              for pair, value in zip(pairs, raw))
    if control == "uniform":
        q = tuple(torch.full_like(value, 0.25) for value in q)
    main = tuple(torch.func.grad(op._main_loss, argnums=1)(theta, phi, q, member,
                forward, s, ys, pairs, 5, cfg) for member, phi in enumerate(phis))
    return {"raw": raw, "q": q, "probe": probes, "main": main}


def parity(op, port, seq, theta, phis, forward, pairs, s, ys, r, yr, control, meter):
    cfg = op.Config()
    counted_calls = [0]
    def counted(core, private):
        counted_calls[0] += 1
        return forward(core, private)
    next_theta, next_phis, info, inspection = seq._engineering_episode(
        op, base.PINS["operator"][1], port, base.PINS["port"][1], theta, phis,
        counted, pairs, s, ys, r, yr, control=control,
        engineering_authorized=True, counters=meter, collect_inspection=True)
    expected_counts = seq.expected_episode_counts(control)
    if meter.total != expected_counts or counted_calls[0] != expected_counts["native_forward_calls"]:
        raise AssertionError("Attempted counters/callback calls differ")
    outer = inspection["outer"]
    oracle = lambda core, private: base.outer(op, core, private, forward, pairs, s, ys, r, yr, control)
    shared, private = torch.func.grad(oracle, argnums=(0, 1))(theta, phis)
    errors = {
        "every_shared_gradient_coordinate": compare(outer["shared_gradient"], shared, gradient=True),
        "every_outer_private_gradient_coordinate": compare(outer["outer_private_gradients"], private, gradient=True),
        "query_loss": compare(outer["virtual_query_loss"], oracle(theta, phis)),
    }
    adapted, original_info = op._private_response(theta, phis, forward, s, ys,
        pairs, 5, cfg, control, collect_diagnostics=True)
    details = reference_details(op, theta, phis, forward, pairs, s, ys, control)
    errors.update({
        "virtual_private_states": compare(outer["adapted"], adapted),
        "original_diagnostics_at_theta": compare(outer["response"].diagnostics, original_info),
        "all_raw_costs": compare(outer["response"].raw_costs, details["raw"]),
        "all_Q_values": compare(outer["response"].q_blocks, details["q"]),
        "all_probe_private_partials": compare(outer["response"].probe_private_gradients, details["probe"], gradient=True),
        "all_main_private_partials": compare(outer["main_private_gradients"], details["main"], gradient=True),
        "recomputed_main_private_partials": compare(outer["recomputed_main_private_partials"], details["main"], gradient=True),
        "query_logits": compare(outer["query_logits"], torch.stack([forward(theta, phi)[r] for phi in adapted])),
    })
    # Exercise the unchanged complete public monolithic state map and restore gate.
    if port.PORT_RELEASED is not False:
        raise AssertionError("Unexpected port gate")
    port.PORT_RELEASED = True
    try:
        ref_theta, ref_phis, ref_info = port.native_sparse_episode(
            op, base.PINS["operator"][1], theta, phis, forward, pairs, s, ys, r, yr, control=control)
    finally:
        port.PORT_RELEASED = False
    errors["theta_plus_every_coordinate"] = compare(next_theta, ref_theta)
    errors["committed_private_every_coordinate"] = compare(next_phis, ref_phis)
    errors["complete_original_committed_diagnostics"] = compare(
        {key: info[key] for key in ref_info}, ref_info)
    # A separate original-phi recomputation excludes using virtual phi as commit.
    fresh, _ = op._private_response(next_theta, phis, forward, s, ys, pairs, 5, cfg, control)
    errors["original_phi_recompute_at_theta_plus"] = compare(next_phis, fresh)
    plus_details = reference_details(op, next_theta, phis, forward, pairs, s, ys, control)
    committed = inspection["committed"]
    errors["committed_raw_costs"] = compare(committed["response"].raw_costs, plus_details["raw"])
    errors["committed_probe_partials"] = compare(committed["response"].probe_private_gradients, plus_details["probe"], gradient=True)
    errors["committed_main_partials"] = compare(committed["main_private_gradients"], plus_details["main"], gradient=True)
    base.zero_unused(outer["shared_gradient"], phis, next_phis)
    if control == "stop_q":
        anchors = original_info["assignments"]
        def fixed_q(core):
            partial = torch.func.grad(op._main_loss, argnums=1)
            states = tuple(op._sgd(phi, partial(core, phi, anchors, member, forward,
                s, ys, pairs, 5, cfg), cfg.eta_private) for member, phi in enumerate(phis))
            return op._query_objective(core, states, forward, r, yr, cfg)
        errors["stop_Q_vs_independent_fixed_Q_shared_derivative"] = compare(
            outer["shared_gradient"], torch.func.grad(fixed_q)(theta), gradient=True)
    return {"control": control, "max_absolute_errors": errors,
            "actual_counters": meter.snapshot(), "independent_callback_calls": counted_calls[0],
            "shared_coordinates": sum(v.numel() for v in theta.values()),
            "outer_private_coordinates": sum(v.numel() for phi in phis for v in phi.values()),
            "dormant_shared_tangent_and_private_state_exact": True,
            "public_monolithic_gate_restored": port.PORT_RELEASED is False}


def resource_episode(op, port, seq, theta, phis, forward, pairs, s, ys, r, yr, meter):
    nt, np_, info, inspection = seq._engineering_episode(op, base.PINS["operator"][1],
        port, base.PINS["port"][1], theta, phis, forward, pairs, s, ys, r, yr,
        engineering_authorized=True, counters=meter, collect_inspection=False)
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
    if not any(bool(torch.count_nonzero(v)) for n, v in gradient.items() if not n.startswith("local_head.")):
        raise AssertionError("Full sequential state map made no active shared change")
    if any(not bool(torch.isfinite(v).all()) for bank in (nt, *np_) for v in bank.values()):
        raise AssertionError("Nonfinite complete output state")
    base.zero_unused(gradient, phis, np_)
    return {"complete_live_episode": True, "actual_counters": meter.snapshot(),
            "shared_coordinates": sum(v.numel() for v in theta.values()),
            "min_assignment": min(float(q.min()) for q in info["assignments"]),
            "virtual_query_loss_finite": bool(torch.isfinite(info["virtual_query_loss"])),
            "constructed_states_discarded": True, "persistent_updates": 0,
            "full_graph_directional_FD": False}


def run(args, receipt, check):
    global torch, base
    repo, phase = Path(args.repository).resolve(), Path(args.source_root).resolve()
    runtime = repo / "experiments_iclr/postsubmission_20260930/native_ncn_runtime_20261005_v1/.venv/bin/python"
    if socket.gethostname() != "anogena-2-0" or Path(sys.executable).absolute() != runtime or not sys.dont_write_bytecode:
        raise RuntimeError("Wrong host/runtime or missing -B")
    base = load("sequential_original_qualifier", phase / BASE[0], BASE[1])
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
    modules = {n: load("sequential_oracle_" + n, phase / p, h) for n, (p, h) in base.PINS.items()}
    op, port = modules["operator"], modules["port"]
    seq = load("sequential_candidate", phase / SEQUENTIAL[0], SEQUENTIAL[1])
    if op.SOURCE_RELEASED is not False or port.PORT_RELEASED is not False or seq.SOURCE_RELEASED is not False:
        raise AssertionError("Original/candidate release gates differ")
    device = torch.device(args.device)
    if args.mode == "synthetic":
        if device.type != "cpu":
            raise ValueError("Frozen synthetic fixture requires CPU float64")
        dtype, count, seed = torch.float64, 15, 106005
        generator = torch.Generator().manual_seed(126005)
        x = torch.randn(count, 300, dtype=dtype, generator=generator)
        support = {(i, i) for i in range(count)}
        for i in range(count - 1):
            support.update({(i, i + 1), (i + 1, i)})
        for i, j in [(0, 4), (1, 6), (2, 7), (3, 8), (4, 9), (0, 9)]:
            support.update({(i, j), (j, i)})
        edges = torch.tensor(sorted(support), dtype=torch.long).transpose(0, 1).contiguous()
        s, r, ys, yr = torch.arange(10), torch.arange(10, 15), torch.arange(10) % 5, torch.arange(5)
    else:
        prior_path = Path(args.synthetic_pass)
        if hashlib.sha256(prior_path.read_bytes()).hexdigest() != args.synthetic_sha:
            raise ValueError("Synthetic receipt differs")
        prior = json.loads(prior_path.read_text())
        if prior["status"] != "PASS_SYNTHETIC_SEQUENTIAL_PARITY_ONLY" or prior["candidate_pin"] != list(SEQUENTIAL):
            raise ValueError("Synthetic prerequisite failed/differs")
        accessor = load("sequential_safe_public_b", Path(args.accessor), args.accessor_sha)
        if accessor.SOURCE_RELEASED is not True:
            raise ValueError("Engineering accessor is disabled")
        payload = accessor.load_public_b(phase, Path(args.public_b_dir), device=str(device))
        x, edges = payload["features"], payload["edge_index"]
        s, ys, r, yr = payload["inner_indices"], payload["inner_labels"], payload["query_indices"], payload["query_labels"]
        if (x.shape != (24492, 300) or x.dtype != torch.float32 or s.numel() != 2449 or r.numel() != 2450
                or payload["W_ids"].numel() != 4898 or payload["B_ids"].numel() != 9797
                or bool(torch.isin(torch.cat((s, r)), payload["A_ids"]).any())):
            raise ValueError("Complete safe native context/roles differ")
        dtype, count, seed = torch.float32, 24492, 17
    family = base.build_family(modules["native"], modules["boundary"], device, dtype, seed)
    forward, theta, phis, shared_names = port._native_callback_and_state(family, x, edges,
        expected_nodes=count, global_stage=True)
    pairs = port._sparse_pairs(op, s, ys, edges, node_count=count, dtype=dtype)
    before = base.snapshot(family, x, edges, device)
    input_before = ({n: p.clone() for n, p in theta.items()}, tuple({n: p.clone() for n, p in phi.items()} for phi in phis))
    flags = ([p.requires_grad for p in theta.values()], [[p.requires_grad for p in phi.values()] for phi in phis])
    meter = seq.Counters()
    receipt["source_pins"] = base.PINS
    receipt["attempted_counter_snapshot"] = meter.snapshot()
    try:
        if args.mode == "synthetic":
            permutation = (torch.arange(s.numel()) + 5) % s.numel()
            permuted = port._sparse_pairs(op, s, ys, edges, node_count=count, dtype=dtype,
                                         affinity_permutation=permutation)
            for label in CONTROLS:
                control = "live" if label == "permuted" else label
                selected_pairs = permuted if label == "permuted" else pairs
                meter = seq.Counters()
                receipt["last_attempted_control"] = label
                check("full_coordinate_parity_" + label, lambda c=control, p=selected_pairs:
                    parity(op, port, seq, theta, phis, forward, p, s, ys, r, yr, c, meter))
            check("unchanged_same_state_fine_directional_FD_stop_fixedQ_and_unused", lambda:
                base.directional_native(op, forward, theta, phis, pairs, s, ys, r, yr, dtype=dtype, device=device))
        else:
            check("full_graph_sparse_products_and_assignment_feasibility", lambda:
                base.reference_products_and_assignment(op, pairs, dtype=dtype, device=device))
            check("complete_full_graph_sequential_higher_order_state_map", lambda:
                resource_episode(op, port, seq, theta, phis, forward, pairs, s, ys, r, yr, meter))
    finally:
        receipt["attempted_counter_snapshot"] = meter.snapshot()
        # Verify restoration on both success and failure; preserve original exception.
        try:
            receipt["restoration_on_exit"] = base.unchanged(before, family, x, edges, device)
            if flags != ([p.requires_grad for p in theta.values()], [[p.requires_grad for p in phi.values()] for phi in phis]):
                raise AssertionError("Caller requires_grad flags changed")
            if any(not torch.equal(input_before[0][n], theta[n]) for n in theta):
                raise AssertionError("Caller shared input mutated")
            if any(not torch.equal(old[n], phi[n]) for old, phi in zip(input_before[1], phis) for n in phi):
                raise AssertionError("Caller private input mutated")
            if op.SOURCE_RELEASED is not False or port.PORT_RELEASED is not False or seq.SOURCE_RELEASED is not False:
                raise AssertionError("Process release gates not restored")
            receipt["caller_states_and_flags_and_gates_unchanged"] = True
        except BaseException:
            receipt["restoration_error"] = traceback.format_exc()
            raise
    return {"torch_version": torch.__version__, "torch_module_path": torch.__file__,
            "nodes": count, "dtype": str(dtype), "device": str(device),
            "native_shared_coordinates": sum(v.numel() for v in theta.values()),
            "all_six_controls_compared": args.mode == "synthetic",
            "A_labels_loaded_by_worker": False, "B_labels_only": args.mode == "full",
            "custodian_joint_TRAIN_including_A_decode_preserved": args.mode == "full",
            "original_coarse_failure_and_full_OOM_preserved": True,
            "predictive_or_novelty_evidence": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-authorized", action="store_true")
    parser.add_argument("--mode", choices=("synthetic", "full"), default="synthetic")
    parser.add_argument("--source-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--repository", default="/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    parser.add_argument("--synthetic-pass")
    parser.add_argument("--synthetic-sha")
    parser.add_argument("--accessor")
    parser.add_argument("--accessor-sha")
    parser.add_argument("--public-b-dir")
    args = parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({"status": "DISABLED_ROOT_ENGINEERING_WORKER", "numeric_imports": False}))
        return
    if not args.output or (args.mode == "full" and not all((args.synthetic_pass, args.synthetic_sha,
            args.accessor, args.accessor_sha, args.public_b_dir))):
        parser.error("Fresh output and complete full-mode bindings required")
    if args.mode == "synthetic" and any((args.synthetic_pass, args.synthetic_sha, args.accessor, args.accessor_sha, args.public_b_dir)):
        parser.error("Synthetic mode accepts no dataset bindings")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "RUNNING", "UTC": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode, "candidate_pin": SEQUENTIAL, "original_qualifier_pin": BASE,
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "hostname": socket.gethostname(), "python": sys.executable, "checks": [],
        "model_fits": 0, "persistent_updates": 0, "A_scoring": False, "VALID_TEST_access": False}
    deadline = 600 if args.mode == "synthetic" else 900
    receipt["fixed_total_deadline_seconds"] = deadline
    started = time.monotonic()
    def save():
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        (output / "RESULT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    def check(name, fn):
        receipt["last_started_check"] = name
        save()
        begin = time.monotonic()
        result = fn()
        receipt["checks"].append({"name": name, "elapsed_seconds": time.monotonic() - begin, "result": result})
        save()
    def expired(signum, frame):
        raise TimeoutError("Fixed engineering deadline exceeded: " + str(deadline))
    save()
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(deadline)
    try:
        receipt.update(run(args, receipt, check))
        receipt["status"] = "PASS_SYNTHETIC_SEQUENTIAL_PARITY_ONLY" if args.mode == "synthetic" else "PASS_FULL_SEQUENTIAL_RESOURCE_ONLY"
    except BaseException as error:
        receipt.update(status="FAIL_ENGINEERING", error_type=type(error).__name__,
                       error=str(error), traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        if "torch" in globals() and args.device.startswith("cuda") and torch.cuda.is_initialized():
            receipt["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
            receipt["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(args.device)
        save()
    print(json.dumps({"status": receipt["status"], "checks_completed": len(receipt["checks"]),
                      "elapsed_seconds": receipt["elapsed_seconds"], "error": receipt.get("error")}))
    raise SystemExit(0 if receipt["status"].startswith("PASS_") else 1)


if __name__ == "__main__":
    main()
