"""Disabled paired common400 bridge qualification; source preparation only.

Root and joint source review precede the one explicit engineering invocation.
No monolithic native episode, fit, scoring, state install or automatic retry.
"""
import time
STARTED = time.monotonic()
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import resource
import signal
import socket
import sys
import traceback

SOURCE_RELEASED = False
REPOSITORY = "/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs"
DEVICE, WORKSPACE = "cuda:0", ":4096:8"
VALUE_ATOL, VALUE_RTOL = 2e-6, 2e-5
GRAD_ATOL, GRAD_RTOL = 1e-8, 1e-6
LIMITS = {"max_elapsed_seconds": 1800, "max_process_rss_bytes": 8589934592,
          "max_cuda_allocated_bytes": 79456894976, "max_cuda_reserved_bytes": 83751862272}
EXTERNAL_WATCHDOG_SECONDS = 1850
PINS = {'accessor': {'bytes': 19074,
              'path': 'amazon_learnability_responsibility_engineering_accessor_20261005_v1/train_only_accessor.py',
              'sha256': '9360e69753b362eb66ac89f133f00addaf15ab0e8c56314c75d7dd5e96fecb85'},
 'adapter': {'bytes': 15753,
             'path': 'public_path_responsibility_two_hop_disabled_implementation_preparation_20261006_v1/bridge_sparse_adapter.py',
             'sha256': 'bd8988c1e64ba8416927b3e9f6b79962d22dac2448c079476f99f4672fd6f2e0'},
 'adapter_independent_review': {'bytes': 9003,
                                'path': 'public_path_responsibility_two_hop_independent_static_review_20261006_v1/FINDINGS.json',
                                'sha256': 'aca03b9ded6f7de326d2911f4a04611b1624cead3bf635523e7bd72f22ff1c88'},
 'adapter_manifest': {'bytes': 1212,
                      'path': 'public_path_responsibility_two_hop_disabled_implementation_preparation_20261006_v1/MANIFEST.json',
                      'sha256': '91664082a371c778c48af22f3c8c42cad52f65d4d337f020dbfa5ff5845ad5c4'},
 'adapter_seal': {'bytes': 451,
                  'path': 'public_path_responsibility_two_hop_disabled_implementation_preparation_20261006_v1/SEAL.json',
                  'sha256': 'c1575c863fbf67bdd3caa14d0eb91e270f321b558efca23f9b64e3aa544cf5a3'},
 'bank_descriptor_receipt': {'bytes': 8163,
                             'path': 'amazon_common400_descriptor_binding_root_20261006_v1/DESCRIPTOR_RECEIPT.json',
                             'sha256': 'cd1e4cdb2b9d68a6a5a3166dfcf412ad2a53541ace30b4850ea60e992626f298'},
 'boundary': {'bytes': 6797,
              'path': 'amazon_polynormer_paired_family_source_preparation_20261003_v6/reused_models/backbone_boundary_adapter.py',
              'sha256': '699b606ead00cb7bdd9be6cd58730a0687157c40cf594af620d1edc4c93b6bac'},
 'census': {'bytes': 8313,
            'path': 'amazon_fixed_pair_S_graph_support_census_execution_root_20261006_v2/COMPACT_SUMMARY.json',
            'sha256': '015fb3ee020f1f57f04aa050ed74a577a39eb150a1dc2979fbc363fa790f4d88'},
 'certificate': {'bytes': 9573,
                 'path': 'learnability_responsibility_actual_strict_all_six_certificate_root_20261006_v1/CERTIFICATE.json',
                 'sha256': 'aac583de6a4777a16fbf09a38aa85607ffbc11d426583d558aa56d138719b13f'},
 'custody': {'bytes': 29855,
             'path': 'learnability_responsibility_native_numerical_worker_preparation_20261005_v3/qualify.py',
             'sha256': 'baeac626bade87bd286fe7a93a95602d8dc1f57d3257d0eeeb7e4a8e3823c688'},
 'full': {'bytes': 21539,
          'path': 'learnability_responsibility_sequential_full_six_preparation_20261006_v1/qualify.py',
          'sha256': '67a898c606d1e94bf680c84023d832ac73d4620081e79fd232800459b041a780'},
 'helpers': {'bytes': 41361,
             'path': 'amazon_learnability_responsibility_sequential_train_only_execution_preparation_20261006_v3/six_arm_worker.py',
             'sha256': 'f19a94be4102e74f30d2b779ca602e288cf40c446ae367ce9d9ab44091569ad1'},
 'native': {'bytes': 7129,
            'path': 'amazon_polynormer_paired_family_source_preparation_20261003_v6/reused_models/native_polynormer.py',
            'sha256': '9b4e533f46ae7f91a23a552f88bbb47a01359e224b53996cf1a68c10a865f6a8'},
 'operator': {'bytes': 14428,
              'path': 'learnability_weighted_graph_responsibility_operator_20261005_v2/response_operator.py',
              'sha256': 'fba3ca3d4bb35da0438923941d97bc4833004dd9fd385735c2352f343693b3e3'},
 'ordinary': {'bytes': 34961,
              'path': 'amazon_ordinary_shared_bank_own_pool_reference_preparation_20261006_v4/ordinary_reference.py',
              'sha256': '5044a16f3f710aaf234057b115ab928d589af3a06a595940afbfa8876636c97b'},
 'origin': {'bytes': 6402,
            'path': 'amazon_common400_descriptor_binding_root_20261006_v1/ORIGIN_RUN.json',
            'sha256': '2e18a77081437e47c0c4a0d59cb07b2a15500f660066cbed1c980f613c028ca5'},
 'port': {'bytes': 11837,
          'path': 'learnability_responsibility_native_amazon_sparse_port_20261005_v1/native_sparse_port.py',
          'sha256': 'a8ee35afda97afcb745c365a51f8e8647fe5e6a1350f654a5321c81e12abad86'},
 'process': {'bytes': 25151,
             'path': 'amazon_learnability_responsibility_strict_process_scientific_runner_preparation_20261006_v3/run_scientific.py',
             'sha256': '65c63b0b62f49bc852bc4a1db3a47196d5c46c1355a328b8ef838397dc79a6bb'},
 'protocol': {'bytes': 15122,
              'path': 'amazon_learnability_responsibility_train_response_protocol_20261005_v2/PROTOCOL.json',
              'sha256': '27f8a8d85aec4e81114ed9ecd8554fcfaa6dd4ccf1eaf7678cc0f5eeeb477d99'},
 'sequential': {'bytes': 25998,
                'path': 'learnability_responsibility_sequential_autograd_vjp_preparation_20261006_v1/sequential_vjp.py',
                'sha256': '1f9e704a1b95a98cec688d695094058594758a99bac2f057e2fe90e7f9315167'}}
COMMON = {"path": "amazon_learnability_responsibility_strict_scientific_execution_root_20261006_v2/output/scientific_run/initial.pt",
          "bytes": 109671738, "sha256": "2e0e9b44767abc12b3dc896986ea2d4faeaa667c8682b95929f927d10718154f"}
PUBLIC_B_RELATIVE = "learnability_responsibility_native_full_execution_root_20261005_v1/roles/public_b"
EPISODE_COUNTS = {"native_forward_calls": 48, "private_gradient_calls": 24,
    "native_vjp_calls": 12, "q_map_primal_calls": 30, "q_map_vjp_calls": 10, "small_query_vjp_calls": 1}
RESPONSE_COUNTS = {"native_forward_calls": 12, "private_gradient_calls": 8,
    "native_vjp_calls": 0, "q_map_primal_calls": 10, "q_map_vjp_calls": 0, "small_query_vjp_calls": 0}


def require(value, message):
    if not value:
        raise AssertionError(message)


def terminal_resources(receipt):
    """Retain terminal whole-child resources and every independent cap breach."""
    previous = receipt.get("whole_child_resources", {})
    row = {"elapsed_seconds": time.monotonic() - STARTED,
           "process_peak_rss_bytes": None,
           "cuda_peak_allocated_bytes": previous.get("cuda_peak_allocated_bytes", 0),
           "cuda_peak_reserved_bytes": previous.get("cuda_peak_reserved_bytes", 0)}
    errors = receipt.setdefault("terminal_resource_errors", [])
    def sample(label, call):
        try: return call()
        except BaseException as error:
            errors.append({"resource": label, "error_type": type(error).__name__,
                           "error": str(error), "traceback": traceback.format_exc()})
    row["process_peak_rss_bytes"] = sample("process_peak_rss_bytes",
        lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    torch = sys.modules.get("torch")
    if torch is not None and sample("CUDA_initialized", lambda: torch.cuda.is_initialized()):
        for key, call in (("cuda_peak_allocated_bytes", lambda: torch.cuda.max_memory_allocated(DEVICE)),
                          ("cuda_peak_reserved_bytes", lambda: torch.cuda.max_memory_reserved(DEVICE))):
            value = sample(key, call)
            if value is not None: row[key] = max(row[key], value)
    receipt["whole_child_resources"] = row
    receipt["terminal_resource_cap_breaches"] = [
        {"resource": key, "observed": row[key], "cap": LIMITS[cap], "cap_name": cap}
        for key, cap in (("elapsed_seconds", "max_elapsed_seconds"),
                         ("process_peak_rss_bytes", "max_process_rss_bytes"),
                         ("cuda_peak_allocated_bytes", "max_cuda_allocated_bytes"),
                         ("cuda_peak_reserved_bytes", "max_cuda_reserved_bytes"))
        if row[key] is not None and row[key] > LIMITS[cap]]


def bootstrap(path, digest):
    """One stdlib bootstrap; subsequent loads use the pinned existing loader."""
    require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, "Qualifier loader differs")
    name = "_bridge_existing_full_qualifier"
    require(name not in sys.modules, "Fresh qualifier module required")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MeteredLaplacian:
    """Count actual action/adjoint events without changing their arithmetic."""
    def __init__(self, original, counts, tag):
        self.original, self.counts, self.tag = original, counts, tag

    def __matmul__(self, q):
        self.counts[self.tag + "_action_attempts"] = self.counts.get(self.tag + "_action_attempts", 0) + 1
        result = self.original @ q
        if result.requires_grad:
            def adjoint(value):
                key = self.tag + "_output_adjoint_events"
                self.counts[key] = self.counts.get(key, 0) + 1
                return value
            result.register_hook(adjoint)
        return result


def independent_dense_banks(op, data, originals, banks, compare):
    """Qualification-only public-incidence matrices, independent of adapter factors.

    This oracle is not supplied by the adapter and does not change its source.
    No full V-by-V adjacency is built; terminal-interior incidence is temporary.
    """
    torch = sys.modules["torch"]
    edges, S = data["edge_index"], data["inner_indices"]
    keep = edges[0] != edges[1]
    source, destination = edges[0, keep], edges[1, keep]
    degrees = torch.bincount(source, minlength=24492)
    live, control, checks = [], [], []
    for original, sparse, permuted in zip(originals, banks.live, banks.bridge_permuted):
        T = S[original.nodes]; n = T.numel()
        location = torch.full((24492,), -1, dtype=torch.long, device=edges.device)
        location[T] = torch.arange(n, device=edges.device)
        a, b = location[source], location[destination]
        direct = torch.zeros((n, n), dtype=torch.float32, device=edges.device)
        direct_keep = (a >= 0) & (b >= 0)
        direct[a[direct_keep], b[direct_keep]] = 1.0
        bridge_keep = (a >= 0) & (b < 0)
        interiors, column = torch.unique(destination[bridge_keep], sorted=True, return_inverse=True)
        incidence = torch.zeros((n, interiors.numel()), dtype=torch.float32, device=edges.device)
        incidence[a[bridge_keep], column] = 1.0
        C = (incidence / degrees[interiors].to(torch.float32)[None, :]) @ incidence.T
        C.diagonal().zero_()  # Definition offdiag(C), not a numerical repair.
        dD, dB = direct.sum(1), C.sum(1)
        denominator = 1.0 + dD.max() + dB.max()
        LD, LB = torch.diag(dD) - direct, torch.diag(dB) - C
        p = permuted.laplacian.permutation
        require(not torch.equal(LB, LB[p][:, p]), "Frozen bridge control is identity for a pair; no rescue permutation")
        compare(sparse.laplacian.normalization, denominator)
        require(sparse.laplacian.direct is permuted.laplacian.direct, "Control changed direct object")
        require(torch.equal(sparse.laplacian.direct.row, original.laplacian.row)
                and torch.equal(sparse.laplacian.direct.column, original.laplacian.column)
                and torch.equal(sparse.laplacian.direct.weight, torch.ones_like(original.laplacian.weight)),
                "Direct support/unit weights changed")
        compare(sparse.laplacian.direct.degree, dD)
        for label, candidate, bridge, target in (("live", sparse, LB, live),
                                               ("bridge_permuted", permuted, LB[p][:, p], control)):
            matrix = (LD + bridge) / denominator
            require(bool(torch.isfinite(matrix).all()) and bool((C >= 0).all()), "Invalid nonnegative public construction")
            compare(matrix, matrix.T)
            compare(matrix @ torch.ones((n, 4), device=edges.device) / 4.0,
                    torch.zeros((n, 4), device=edges.device))
            compare(candidate.laplacian @ (torch.ones((n, 4), device=edges.device) / 4.0),
                    torch.zeros((n, 4), device=edges.device))
            eigenvalues = torch.linalg.eigvalsh(matrix)
            require(float(eigenvalues.min()) >= -VALUE_ATOL and float(eigenvalues.max()) <= 2.0 + VALUE_ATOL,
                    "PSD/construction norm check failed")
            # Every matrix coordinate is checked through the sparse operator.
            action_error = compare(candidate.laplacian @ torch.eye(n, device=edges.device), matrix)
            q = torch.sin(torch.arange(n * 4, device=edges.device).reshape(n, 4) / 97.0).requires_grad_(True)
            v = torch.cos(torch.arange(n * 4, device=edges.device).reshape(n, 4) / 89.0)
            sparse_gradient, = torch.autograd.grad((candidate.laplacian @ q * v).sum(), (q,))
            compare(sparse_gradient, matrix.T @ v, gradient=True)
            # Complete original eight-step Q map: first and second raw-cost credit.
            map_errors = []
            derivatives = []
            for lap in (candidate.laplacian, matrix):
                raw = q.detach().clone().requires_grad_(True)
                cost, _, _ = op._normalized_cost(raw, op.Config().response_epsilon)
                assignment = op._assignment_map(cost, lap, op.Config(), op.Config().graph)
                first, = torch.autograd.grad((assignment * v).sum(), (raw,), create_graph=True)
                second, = torch.autograd.grad((first * v).sum(), (raw,))
                derivatives.append((assignment.detach(), first.detach(), second.detach()))
            map_errors.append(compare(derivatives[0][0], derivatives[1][0]))
            map_errors.extend(compare(derivatives[0][j], derivatives[1][j], gradient=True) for j in (1, 2))
            target.append(op.Pair(original.classes, original.nodes, original.targets, original.competitors, matrix))
            checks.append({"classes": list(original.classes), "bank": label,
                "all_matrix_coordinate_max_error": action_error, "complete_Q_first_second_credit_max_errors": map_errors,
                "minimum_eigenvalue": float(eigenvalues.min()), "maximum_eigenvalue": float(eigenvalues.max()),
                "fixed_bridge_nonidentity": True, "direct_edge_invariance": True})
        del incidence, C, LD, LB
    return tuple(live), tuple(control), checks


def native_checks(modules, data, image, receipt, save, measured):
    torch = sys.modules["torch"]
    ordinary, helper, op, port, seq, base, full, adapter = (modules[k] for k in
        ("ordinary", "helpers", "operator", "port", "sequential", "custody", "full", "adapter"))
    S, yS, R, yR = (data[k] for k in ("inner_indices", "inner_labels", "query_indices", "query_labels"))
    family, _ = helper._fresh_family(modules["native"], modules["boundary"], DEVICE)
    family.load_state_dict(image["family_state"], strict=True); family.set_global_stage(True); family.eval()
    ordinary.restore_rng(image["rng"], DEVICE)
    forward, theta, phis, _ = port._native_callback_and_state(family, data["features"], data["edge_index"], expected_nodes=24492, global_stage=True)
    originals = port._sparse_pairs(op, S, yS, data["edge_index"], node_count=24492, dtype=torch.float32)
    before = base.snapshot(family, data["features"], data["edge_index"], torch.device(DEVICE))
    incoming = helper._cpu_tree((theta, phis, S, yS, R, yR, image))
    tensors = (S, yS, R, yR, data["features"], data["edge_index"], *theta.values(),
               *(v for row in phis for v in row.values()), *family.parameters(), *family.buffers())
    def metadata(t):
        return (id(t), t.untyped_storage()._cdata, t.untyped_storage().data_ptr(), t._version,
                tuple(t.shape), tuple(t.stride()), t.dtype, t.device, t.requires_grad, t.grad is None)
    tensor_metadata = [metadata(t) for t in tensors]
    all_cuda_rng = [v.clone() for v in torch.cuda.get_rng_state_all()]
    full.torch, full.base, base.torch = torch, base, torch
    def compare(actual, expected, gradient=False):
        if gradient:
            if isinstance(expected, torch.Tensor):
                require(isinstance(actual, torch.Tensor) and actual.shape == expected.shape and actual.dtype == expected.dtype
                        and actual.device == expected.device and bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()),
                        "Gradient structure/finite values differ")
                return base.close(actual, expected, atol=GRAD_ATOL, rtol=GRAD_RTOL)
            require(type(actual) is type(expected), "Gradient container type differs")
            if isinstance(expected, dict):
                require(actual.keys() == expected.keys(), "Gradient keys differ")
                return max((compare(actual[k], expected[k], True) for k in expected), default=0.0)
            require(isinstance(expected, (tuple, list)) and len(actual) == len(expected), "Gradient sequence differs")
            return max((compare(a, b, True) for a, b in zip(actual, expected)), default=0.0)
        return full.compare(actual, expected)
    def unchanged():
        require([metadata(t) for t in tensors] == tensor_metadata, "Input/model identities/storage/versions/flags changed")
        compare(helper._cpu_tree((theta, phis, S, yS, R, yR, image)), incoming)
        # Immutable caller values must be exact, beyond the comparison tolerance.
        def exact(a, b):
            if isinstance(b, torch.Tensor): require(torch.equal(a, b), "Caller value changed")
            elif isinstance(b, dict):
                for k in b: exact(a[k], b[k])
            elif isinstance(b, (tuple, list)):
                for x, y in zip(a, b): exact(x, y)
            else: require(a == b, "Caller metadata changed")
        exact(helper._cpu_tree((theta, phis, S, yS, R, yR, image)), incoming)
        current_cuda_rng = torch.cuda.get_rng_state_all()
        require(len(current_cuda_rng) == len(all_cuda_rng)
                and all(torch.equal(a, b) for a, b in zip(current_cuda_rng, all_cuda_rng)), "Visible CUDA RNG changed")
        require(SOURCE_RELEASED is False and op.SOURCE_RELEASED is False and port.PORT_RELEASED is False
                and seq.SOURCE_RELEASED is False and adapter.ADAPTER_RELEASED is False and helper.SOURCE_RELEASED is False,
                "Original/disabled source gates changed")
        return base.unchanged(before, family, data["features"], data["edge_index"], torch.device(DEVICE))
    calls = {}
    def episode(label, pairs, inspection):
        meter = seq.Counters(); count = [0]; actions = {}
        wrapped = tuple(op.Pair(p.classes, p.nodes, p.targets, p.competitors,
            MeteredLaplacian(p.laplacian, actions, label)) for p in pairs)
        def counted(core, private):
            count[0] += 1; return forward(core, private)
        try:
            result = seq._engineering_episode(op, PINS["operator"]["sha256"], port, PINS["port"]["sha256"],
                theta, phis, counted, wrapped, S, yS, R, yR, control="live", engineering_authorized=True,
                counters=meter, collect_inspection=inspection)
            require(meter.total == EPISODE_COUNTS and count[0] == 48, "Episode native/staged counters differ")
            full.validate_episode(op, theta, phis, *result[:3], None)
            return result
        finally:
            calls[label] = {"sequential": meter.snapshot(), "actual_native_callbacks": count[0], "laplacian_events": actions}
            receipt["attempted_native_work"] = calls; save()
    def fresh(label, theta_plus, pairs):
        meter = seq.Counters(); count = [0]
        def counted(core, private):
            count[0] += 1; return forward(core, private)
        try:
            value = seq._engineering_response(op, PINS["operator"]["sha256"], port, PINS["port"]["sha256"],
                theta_plus, phis, counted, pairs, S, yS, R, yR, control="live", engineering_authorized=True, counters=meter)
            require(meter.total == RESPONSE_COUNTS and count[0] == 12, "Fresh original-phi response counters differ")
            return value
        finally:
            calls[label] = {"sequential": meter.snapshot(), "actual_native_callbacks": count[0]}
            receipt["attempted_native_work"] = calls; save()
    def recommit(label, result, pairs):
        nt, np_, info, inspection = result
        value = fresh(label + "_fresh_original_phi", nt, pairs)
        compare(np_, value["adapted"])
        compare({k: info[k] for k in value["response"].diagnostics}, value["response"].diagnostics)
        if inspection is not None:
            committed = inspection["committed"]
            compare(committed["main_private_gradients"], value["main_private_gradients"], gradient=True)
            compare(committed["response"].raw_costs, value["response"].raw_costs)
            compare(committed["response"].probe_private_gradients, value["response"].probe_private_gradients, gradient=True)
        base.zero_unused({k: (theta[k] - nt[k]) / op.Config().eta_core for k in theta}, phis, np_)
        del value
    try:
        def baseline():
            value = episode("direct_only_baseline", originals, False)
            recommit("direct_only_baseline", value, originals)
            del value
            return unchanged()
        measured("direct_only_common400_baseline", baseline)
        permutation = helper._permutation(data)
        banks = measured("sparse_adapter_preparation_and_coverage", lambda: adapter._prepare_bridge_pair_banks(
            op, port, PINS["operator"]["sha256"], PINS["port"]["sha256"], S, yS, data["edge_index"],
            node_count=24492, dtype=torch.float32, affinity_permutation=permutation,
            protocol_sha256=PINS["protocol"]["sha256"], census_summary_sha256=PINS["census"]["sha256"]))
        require(banks.minimum_restoration_gate and len(banks.static_coverage) == 10, "Every live pair must newly couple a former isolate")
        receipt["actual_static_coverage"] = banks.static_coverage; save()
        references = measured("qualification_only_dense_operator_and_Q_derivative_oracles", lambda:
            independent_dense_banks(op, data, originals, banks, compare))
        receipt["operator_checks"] = references[2]; save()
        for label, pairs, reference_pairs in (("live", banks.live, references[0]),
                                               ("bridge_permuted", banks.bridge_permuted, references[1])):
            candidate = measured(label + "_sparse_episode", lambda: episode(label + "_sparse", pairs, True))
            reference = measured(label + "_dense_reference_episode", lambda: episode(label + "_dense_oracle", reference_pairs, True))
            nt, np_, info, inspection = candidate; rt, rp, ri, reference_inspection = reference
            compare(nt, rt); compare(np_, rp)
            compare({k: v for k, v in info.items() if not k.startswith("sequential_")},
                    {k: v for k, v in ri.items() if not k.startswith("sequential_")})
            a, b = inspection["outer"], reference_inspection["outer"]
            errors = {}
            for key in ("shared_gradient", "outer_private_gradients", "main_private_gradients",
                        "recomputed_main_private_partials", "q_cotangents", "raw_cost_cotangents"):
                errors[key] = compare(a[key], b[key], gradient=True)
            compare(a["response"].probe_private_gradients, b["response"].probe_private_gradients, gradient=True)
            errors["query_logit_cotangents"] = compare(a["query_logit_cotangents"], b["query_logit_cotangents"], gradient=True)
            for key in ("virtual_query_loss", "query_logits", "adapted"):
                compare(a[key], b[key])
            compare(a["response"].raw_costs, b["response"].raw_costs)
            compare(a["response"].q_blocks, b["response"].q_blocks)
            for stage in ("outer", "committed"):
                compare(inspection[stage]["main_private_gradients"], reference_inspection[stage]["main_private_gradients"], gradient=True)
                response_a = inspection[stage]["response"]
                response_b = reference_inspection[stage]["response"]
                compare(response_a.raw_costs, response_b.raw_costs)
                compare(response_a.q_blocks, response_b.q_blocks)
                compare(response_a.probe_private_gradients, response_b.probe_private_gradients, gradient=True)
                for raw_a, raw_b in zip(response_a.raw_costs, response_b.raw_costs):
                    compare(op._normalized_cost(raw_a, op.Config().response_epsilon),
                            op._normalized_cost(raw_b, op.Config().response_epsilon))
            del response_a, response_b, raw_a, raw_b
            # Independently anchored Q: private partial owns phi, with Q fixed.
            def anchored_private_oracle():
                anchored_calls = 0
                try:
                    for member, phi in enumerate(phis):
                        private = {k: v.detach().clone().requires_grad_(True) for k, v in phi.items()}
                        def anchored_forward(core, row):
                            nonlocal anchored_calls
                            anchored_calls += 1; return forward(core, row)
                        loss = op._main_loss(theta, private, tuple(q.detach() for q in a["response"].q_blocks),
                            member, anchored_forward, S, yS, pairs, 5, op.Config())
                        receipt["anchored_private_gradient_attempts"] = receipt.get("anchored_private_gradient_attempts", 0) + 1; save()
                        derivative = torch.autograd.grad(loss, tuple(private.values()), allow_unused=True)
                        expected = {k: torch.zeros_like(v) if g is None else g.detach()
                                    for (k, v), g in zip(private.items(), derivative)}
                        compare(a["main_private_gradients"][member], expected, gradient=True)
                        del loss, derivative, expected, private
                    require(anchored_calls == 4, "Anchored-Q oracle callback count differs")
                finally:
                    receipt.setdefault("anchored_Q_native_callbacks", {})[label] = anchored_calls; save()
            measured(label + "_anchored_Q_private_oracle", anchored_private_oracle)
            measured(label + "_fresh_original_phi_recommit", lambda: recommit(label, candidate, pairs))
            receipt.setdefault("paired_checks", []).append({"bank": label, "derivative_max_errors": errors,
                "all_shared_private_Q_coordinates_compared": True, "independent_Q_private_partial": True,
                "original_phi_recommit": True, "virtual_states_discarded": True, "restoration": unchanged()})
            del a, b, nt, np_, info, inspection, rt, rp, ri, reference_inspection, candidate, reference
            save()
        actual_native = sum(v["actual_native_callbacks"] for v in calls.values()) + sum(receipt["anchored_Q_native_callbacks"].values())
        require(len(receipt["paired_checks"]) == 2 and len(receipt["operator_checks"]) == 20
                and actual_native == 284 and receipt["anchored_private_gradient_attempts"] == 8,
                "Complete paired/oracle invocation matrix differs")
        receipt["actual_complete_native_callback_total"] = actual_native
    finally:
        receipt["attempted_native_work"] = calls
        receipt["native_and_caller_restoration"] = unchanged(); save()


def run(args, receipt, save):
    phase = Path(args.source_root).resolve(); repo = Path(REPOSITORY)
    require(socket.gethostname() == "anogena-2-0" and phase == repo / "experiments_iclr/postsubmission_20260930"
            and Path.cwd().resolve() == repo and Path(sys.executable).absolute() ==
            phase / "native_ncn_runtime_20261005_v1/.venv/bin/python" and sys.dont_write_bytecode,
            "Exact admitted host/phase/native runtime/-B required")
    require(not any(k == "torch" or k.startswith("torch.") for k in sys.modules), "Fresh child before Torch required")
    full = bootstrap(phase / PINS["full"]["path"], PINS["full"]["sha256"])
    loaded = ["_bridge_existing_full_qualifier"]
    modules = {"full": full}
    for key in ("ordinary", "process", "helpers", "custody", "operator", "port", "sequential", "adapter", "accessor"):
        row = PINS[key]
        if "ordinary" in modules:
            path = modules["ordinary"].bound(phase, row)
        else:
            path = phase / row["path"]
            require(path.stat().st_mode & 0o222 == 0 and path.stat().st_size == row["bytes"], "Readonly bootstrap source required")
        name = "_bridge_paired_" + key
        require(name not in sys.modules, "Fresh source module required")
        modules[key] = full.load(name, path, row["sha256"]); loaded.append(name)
    ordinary, process, helper = modules["ordinary"], modules["process"], modules["helpers"]
    for row in PINS.values(): ordinary.bound(phase, row)
    certificate = json.loads(ordinary.bound(phase, PINS["certificate"]).read_text())
    origin = json.loads(ordinary.bound(phase, PINS["origin"]).read_text())
    require(certificate["status"] == "PASS_ALL_SIX_FULL_FP32_COMMIT_RESOURCE"
            and certificate["candidate_sha256"] == PINS["sequential"]["sha256"]
            and certificate["original_phi_recompute_qualified"] is True
            and certificate["state_input_RNG_alias_gate_restoration_passed"] is True
            and certificate["model_fits"] == 0 and certificate["persistent_updates"] == 0
            and certificate["A_scoring"] is False and certificate["VALID_TEST_access"] is False,
            "Existing strict sequential qualification prerequisite differs")
    require(origin["schema"] == "amazon_G0_six_arm_run_v1" and origin["A_labels_received"] is False
            and origin["A_scoring_performed"] is False and origin["recipe"]["worker_source_sha256"] == ordinary.COMMON_ORIGIN_WORKER,
            "Immutable common400 origin differs")
    require(Path(args.output).is_absolute() and not Path(args.output).is_symlink()
            and Path(args.output).resolve().is_relative_to(phase), "Fresh deliberate in-phase output required")
    require(modules["adapter"].ADAPTER_RELEASED is False and modules["operator"].SOURCE_RELEASED is False
            and modules["port"].PORT_RELEASED is False and modules["sequential"].SOURCE_RELEASED is False
            and helper.SOURCE_RELEASED is False and ordinary.SOURCE_RELEASED is False and process.SOURCE_RELEASED is False
            and full.SOURCE_RELEASED is False and modules["accessor"].SOURCE_RELEASED is True,
            "Original and disabled process/source gates differ")
    require(not any(k == "torch" or k.startswith("torch.") for k in sys.modules),
            "Only stdlib source helpers may load before strict workspace configuration")
    old_path, old_env, old_python_rng = list(sys.path), ("CUBLAS_WORKSPACE_CONFIG" in os.environ, os.environ.get("CUBLAS_WORKSPACE_CONFIG")), random.getstate()
    old_signal, old_timer = signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL)
    require(old_timer == (0.0, 0.0), "Fresh child without previous alarm required")
    torch = numpy = None; backend = threads = numeric_rng = None
    def alarm(number, frame): raise TimeoutError("Whole-child bridge qualification cap exceeded; no retry")
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, LIMITS["max_elapsed_seconds"] - (time.monotonic() - STARTED)))
    peaks = {"cuda_peak_allocated_bytes": 0, "cuda_peak_reserved_bytes": 0}
    def resources():
        if torch is not None and torch.cuda.is_initialized():
            peaks["cuda_peak_allocated_bytes"] = max(peaks["cuda_peak_allocated_bytes"], torch.cuda.max_memory_allocated(DEVICE))
            peaks["cuda_peak_reserved_bytes"] = max(peaks["cuda_peak_reserved_bytes"], torch.cuda.max_memory_reserved(DEVICE))
        return {"elapsed_seconds": time.monotonic() - STARTED,
                "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, **peaks}
    def limits():
        row = resources(); receipt["whole_child_resources"] = row
        for key, cap in (("elapsed_seconds", "max_elapsed_seconds"), ("process_peak_rss_bytes", "max_process_rss_bytes"),
                         ("cuda_peak_allocated_bytes", "max_cuda_allocated_bytes"), ("cuda_peak_reserved_bytes", "max_cuda_reserved_bytes")):
            require(row[key] <= LIMITS[cap], "Whole-child limit exceeded: " + key)
    def measured(label, call):
        limits(); torch.cuda.synchronize(DEVICE)
        resources()  # Preserve the whole-child maxima before each interval reset.
        start = time.monotonic(); allocated = torch.cuda.memory_allocated(DEVICE)
        reserved = torch.cuda.memory_reserved(DEVICE)
        torch.cuda.reset_peak_memory_stats(DEVICE)
        try:
            with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True, record_shapes=False) as profile:
                value = call()
            return value
        finally:
            torch.cuda.synchronize(DEVICE)
            phase_peaks = {"cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(DEVICE),
                           "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(DEVICE)}
            resources()
            row = {"phase": label, "elapsed_seconds_including_profiler": time.monotonic() - start,
                "allocated_at_start": allocated, "reserved_at_start": reserved,
                "phase_peak_allocated_bytes": phase_peaks["cuda_peak_allocated_bytes"],
                "phase_peak_reserved_bytes": phase_peaks["cuda_peak_reserved_bytes"],
                "incremental_peak_allocated_bytes_above_start": phase_peaks["cuda_peak_allocated_bytes"] - allocated,
                "incremental_peak_reserved_bytes_above_start": phase_peaks["cuda_peak_reserved_bytes"] - reserved,
                "profiler_events": [{"key": e.key, "count": e.count, "self_cpu_time_total_us": e.self_cpu_time_total,
                    "self_cuda_time_total_us": e.self_cuda_time_total} for e in profile.key_averages()
                    if any(k in e.key.lower() for k in ("index_add", "scatter", "indexselectbackward", "indexbackward", "mm", "autograd"))]}
            receipt.setdefault("measured_phases", []).append(row); save(); limits()
    body_error = None
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = WORKSPACE
        site = repo / ".venv/lib/python3.11/site-packages"
        require(site.is_dir(), "Exact normal runtime required; no fallback")
        sys.path.insert(0, str(site))
        import torch
        import numpy
        process.torch, modules["custody"].torch, modules["custody"].np = torch, torch, numpy
        require(not torch.cuda.is_initialized() and Path(torch.__file__).resolve().is_relative_to(repo), "Premature CUDA or different Torch")
        backend = process.backend_snapshot(); threads = torch.get_num_threads()
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        require(process.backend_snapshot() == certificate["strict_backend_snapshot"], "Existing qualified strict backend differs")
        runtime = process.runtime_identity()
        require(all(runtime[k] == v for k, v in certificate["strict_runtime_metadata"].items())
                and {"path": runtime["PyG_module_path"], "version": runtime["PyG_version"]} == certificate["loaded_PyG_identity"],
                "Existing qualified runtime/device identity differs")
        receipt["strict_runtime_identity"] = runtime
        numeric_rng = (numpy.random.get_state(), torch.get_rng_state().clone(), [v.clone() for v in torch.cuda.get_rng_state_all()])
        for key in ("native", "boundary"):
            name = "_bridge_paired_" + key
            require(name not in sys.modules, "Fresh native source module required")
            modules[key] = full.load(name, ordinary.bound(phase, PINS[key]), PINS[key]["sha256"])
            loaded.append(name)
        projection = phase / PUBLIC_B_RELATIVE
        preflight = helper._public_identity_before_w(modules["accessor"], phase, projection, ordinary.INPUT_IDENTITY)
        data = modules["accessor"].load_public_b(phase, projection, device=DEVICE)
        expected, provenance = origin["recipe"], data["provenance"]
        require(all(provenance[k]["sha256"] == expected[v] for k, v in
            (("public_b_manifest", "public_b_manifest_sha256"), ("public_graph", "public_graph_sha256"), ("roles", "roles_sha256")))
            and provenance["preprocessing"]["edge_logical_sha256"] == expected["native_edge_logical_sha256"]
            and provenance["preprocessing"] == preflight["preprocessing"], "Same immutable common400 public/S/R custody required")
        require(data["features"].shape == (24492, 300) and data["features"].dtype == torch.float32
            and data["inner_indices"].numel() == 2449 and data["query_indices"].numel() == 2450
            and not bool(torch.isin(torch.cat((data["inner_indices"], data["query_indices"])),
                                   torch.cat((data["W_ids"], data["A_ids"]))).any()), "Full native disjoint S/R required")
        common_path = ordinary.bound(phase, COMMON)
        image = torch.load(common_path, map_location="cpu", weights_only=True)
        require(image["schema"] == "amazon_G0_frozen_state_v1" and image["id"] == "initial" and image["warm_updates"] == 400
                and image["episodes"] == 0 and image["warm_role"] == "W" and image["global_stage"] is True
                and image["eval_mode"] is True and image["recipe"] == expected and image["construction"]["seed"] == 17,
                "Exact frozen common400 initial bank required")
        helper._finite_tree(image)
        receipt["input_provenance"] = provenance; save()
        native_checks(modules, data, image, receipt, save, measured)
        require(process.backend_snapshot() == certificate["strict_backend_snapshot"], "Backend changed during qualification")
        limits()
    except BaseException as error:
        body_error = error
        receipt["qualification_body_error"] = {"error_type": type(error).__name__,
            "error": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        errors = []
        def restore(label, call):
            try: call(); receipt.setdefault("process_restored", []).append(label)
            except BaseException as error:
                errors.append({"restoration": label, "error_type": type(error).__name__,
                               "error": str(error), "traceback": traceback.format_exc()})
        restore("cancel_qualification_alarm", lambda: signal.setitimer(signal.ITIMER_REAL, 0))
        if numeric_rng is not None:
            def restore_numeric():
                numpy.random.set_state(numeric_rng[0]); torch.set_rng_state(numeric_rng[1]); torch.cuda.set_rng_state_all(numeric_rng[2])
                restored = numpy.random.get_state()
                require(restored[0] == numeric_rng[0][0] and numpy.array_equal(restored[1], numeric_rng[0][1])
                        and restored[2:] == numeric_rng[0][2:] and torch.equal(torch.get_rng_state(), numeric_rng[1])
                        and len(torch.cuda.get_rng_state_all()) == len(numeric_rng[2])
                        and all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), numeric_rng[2])), "Numerical RNG restoration failed")
            restore("NumPy_Torch_CPU_visible_CUDA_RNG", restore_numeric)
        if backend is not None: restore("strict_backend", lambda: process.backend_restore(backend))
        if threads is not None:
            def restore_threads():
                torch.set_num_threads(threads)
                require(torch.get_num_threads() == threads, "Intra-op restoration failed")
            restore("intra_op_threads", restore_threads)
        receipt["interop_restore"] = "One-time fresh-child initialization; not restored and ends with child termination."
        def restore_python_rng():
            random.setstate(old_python_rng)
            require(random.getstate() == old_python_rng, "Python RNG restoration failed")
        restore("Python_RNG", restore_python_rng)
        def restore_path():
            sys.path[:] = old_path
            require(sys.path == old_path, "Path restoration failed")
        restore("sys_path", restore_path)
        def restore_environment():
            if old_env[0]: os.environ["CUBLAS_WORKSPACE_CONFIG"] = old_env[1]
            else: os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            require(("CUBLAS_WORKSPACE_CONFIG" in os.environ, os.environ.get("CUBLAS_WORKSPACE_CONFIG")) == old_env,
                    "Environment restoration failed")
        restore("CUBLAS_WORKSPACE_CONFIG", restore_environment)
        def restore_signal():
            signal.signal(signal.SIGALRM, old_signal); signal.setitimer(signal.ITIMER_REAL, *old_timer)
            require(signal.getsignal(signal.SIGALRM) == old_signal
                    and signal.getitimer(signal.ITIMER_REAL) == old_timer, "Signal restoration failed")
        restore("signal_and_timer", restore_signal)
        for key, module in modules.items():
            restore("source_bytes_" + key, lambda key=key, module=module:
                require(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == PINS[key]["sha256"], "Loaded source bytes changed"))
        def check_gates():
            require(SOURCE_RELEASED is False and modules["adapter"].ADAPTER_RELEASED is False
                    and modules["port"].PORT_RELEASED is False and modules["accessor"].SOURCE_RELEASED is True
                    and all(modules[k].SOURCE_RELEASED is False for k in ("operator", "sequential", "helpers", "ordinary", "process", "full")),
                    "Loaded original/disabled process gates changed")
        restore("original_disabled_source_gates", check_gates)
        for name in loaded: restore("unload_" + name, lambda name=name: sys.modules.pop(name, None))
        receipt["restoration_errors"] = errors
        try: receipt["whole_child_resources"] = resources()
        except BaseException as error:
            receipt.setdefault("terminal_resource_errors", []).append({"resource": "whole_child_peak_snapshot",
                "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        terminal_resources(receipt)
        try: save()
        except BaseException as error:
            receipt.setdefault("cleanup_receipt_errors", []).append({"error_type": type(error).__name__,
                "error": str(error), "traceback": traceback.format_exc()})
        if body_error is None:
            require(not errors, "Process restoration failure")
            require(not receipt.get("terminal_resource_errors"), "Terminal whole-child resource accounting failure")
            require(not receipt["terminal_resource_cap_breaches"], "Whole-child terminal resource cap exceeded")
            require(not receipt.get("cleanup_receipt_errors"), "Cleanup receipt preservation failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-authorized", action="store_true")
    parser.add_argument("--source-root")
    parser.add_argument("--output")
    args = parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({"status": "DISABLED_PAIRED_BRIDGE_QUALIFICATION_SOURCE_ONLY", "SOURCE_RELEASED": SOURCE_RELEASED, "numeric_imports": False})); return
    require(SOURCE_RELEASED is False and args.source_root and args.output, "Separate root engineering release after joint source review required")
    require(args.source_root == str(Path(REPOSITORY) / "experiments_iclr/postsubmission_20260930")
            and socket.gethostname() == "anogena-2-0", "Exact future admitted phase/host required before output creation")
    output = Path(args.output).absolute()
    require(not output.exists() and not output.is_symlink() and output.parent.is_dir()
            and ".." not in output.parts and output.resolve().is_relative_to(Path(args.source_root).resolve()),
            "New in-phase output with existing parent required")
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    identity = (output.stat().st_dev, output.stat().st_ino)
    receipt = {"status": "RUNNING_PAIRED_BRIDGE_ENGINEERING_ONLY", "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "adapter_pin": PINS["adapter"], "immutable_common400": COMMON, "whole_child_limits": LIMITS,
        "external_watchdog_seconds": EXTERNAL_WATCHDOG_SECONDS, "measured_phases": [],
        "model_fits": 0, "persistent_updates": 0, "A_scoring": False, "VALID_TEST_access": False,
        "full_native_monolithic_episode": False, "automatic_retry_or_cap_expansion": False}
    def save():
        require(not output.is_symlink() and (output.stat().st_dev, output.stat().st_ino) == identity, "Created output ownership changed")
        receipt["whole_child_elapsed_seconds"] = time.monotonic() - STARTED
        temporary = output / "RESULT.tmp"
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, output / "RESULT.json")
    code = 1
    try:
        save(); run(args, receipt, save)
        receipt["status"] = "PASS_PAIRED_BRIDGE_NATIVE_PARITY_RECOMMIT_RESOURCE_ONLY"; code = 0
    except BaseException as error:
        receipt.update(status="FAIL_PAIRED_BRIDGE_ENGINEERING", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        terminal_resources(receipt)
        if receipt.get("terminal_resource_errors") or receipt["terminal_resource_cap_breaches"]:
            receipt["status"] = "FAIL_PAIRED_BRIDGE_ENGINEERING"; code = 1
            if "error" not in receipt:
                receipt.update(error_type="AssertionError", error="Terminal whole-child resource accounting or cap failure")
        save(); (output / "RESULT.json").chmod(0o444)
    print(json.dumps({"status": receipt["status"], "error": receipt.get("error"), "model_fits": 0, "persistent_updates": 0}))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
