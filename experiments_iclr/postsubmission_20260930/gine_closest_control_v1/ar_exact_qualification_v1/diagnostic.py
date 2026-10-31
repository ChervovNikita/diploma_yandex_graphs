"""Prospective CPU exact-law diagnostic. Authored only; not executed/compiled.

Source/Torch imports occur only in main after future root admission and pins.
No RNG, optimizer, fit, scientific dataset, real source Model or GPU is used.
"""
import argparse
import hashlib
import importlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import types


BASE = Path(__file__).resolve().parent
RESEARCH = BASE.parents[1]
SOURCE = RESEARCH / "new_learning_gap_independent_v1/ar_mixture_source_v1"
TOL = {"value_atol": 5e-12, "density_atol": 2e-12,
       "gradient_atol": 2e-10, "gradient_rtol": 2e-8, "zero_atol": 2e-10}
LIMITS = {"cases": 16, "max_nodes": 4, "max_branches": 3,
          "source_objective_calls": 900, "autograd_calls": 512,
          "charged_decoder_steps": 40000, "internal_wall_seconds": 110}
FIXTURES = (
    ("edgeless4", 4, (), (2, 0, 3, 1), (0.5, 1.25, 2.0, 3.0)),
    ("path4", 4, ((0, 1), (1, 2), (2, 3)), (2, 0, 3, 1), (0.5, 2.0, 1.25, 3.0)),
    ("triangle_isolate4", 4, ((0, 1), (1, 2), (0, 2)), (2, 3, 0, 1), (0.4, 1.3, 2.7, 0.8)),
    ("empty0", 0, (), (), ()),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def verify_source():
    pins = json.loads((BASE / "SOURCE_PINS.json").read_text(encoding="utf-8"))
    require(sha(SOURCE / "SOURCE_PACKET_MANIFEST_v1.json") == pins["manifest_sha256"], "Source manifest changed")
    require(sha(SOURCE / "SOURCE_PACKET_SEAL_v1.json") == pins["seal_sha256"], "Source seal changed")
    manifest = json.loads((SOURCE / "SOURCE_PACKET_MANIFEST_v1.json").read_text(encoding="utf-8"))
    for row in manifest["source_and_document_files"]:
        path = SOURCE / row["file"]
        require(path.resolve().is_relative_to(SOURCE.resolve()), "Source manifest path escapes packet")
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Source bytes changed: " + row["file"])
        require(pins["files"][row["file"]] == row["sha256"], "Source pin differs from sealed manifest")
    for packet_name, record in pins["additional_packets"].items():
        folder = SOURCE.parent / packet_name
        require(sha(folder / "SOURCE_PACKET_MANIFEST_v1.json") == record["manifest_sha256"], "Additional source manifest changed")
        require(sha(folder / "SOURCE_PACKET_SEAL_v1.json") == record["seal_sha256"], "Additional source seal changed")
        for name, expected in record["files"].items():
            require(sha(folder / name) == expected, "Additional source file changed: " + name)
    return pins


def import_existing_source():
    root_name = "root_admitted_packets_v1"
    package_root = types.ModuleType(root_name)
    package_root.__path__ = [str(SOURCE.parent)]
    sys.modules[root_name] = package_root
    name = root_name + ".ar_mixture_source_v1"
    result = {key: importlib.import_module(name + "." + key)
              for key in ("domain", "source_encoder", "work_accounting", "ar_decoder", "mixture_objective")}
    head_root = root_name + ".shared_transition_heads_source_v1"
    result["head_decoder"] = importlib.import_module(head_root + ".shared_head_decoder")
    result["head_work"] = importlib.import_module(head_root + ".work_accounting")
    return result


class Checks:
    def __init__(self, torch_api):
        self.torch = torch_api
        self.autograd_calls = 0
        self.source_objective_calls = 0
        self.records = []
        self.active_case = None
        self.active_ledger = None
        self.active_literal_counts = None

    def close(self, actual, expected, name, atol=None, rtol=0.0):
        atol = TOL["value_atol"] if atol is None else atol
        require(self.torch.isfinite(actual).all().item() and self.torch.isfinite(expected).all().item(), name + ": nonfinite comparison")
        require(self.torch.allclose(actual, expected, atol=atol, rtol=rtol), name + ": tolerance failure")

    def grad(self, value, parameters):
        self.autograd_calls += 1
        require(self.autograd_calls <= LIMITS["autograd_calls"], "Autograd-call cap reached")
        result = self.torch.autograd.grad(value, tuple(parameters.values()), retain_graph=True, allow_unused=True)
        result = {name: self.torch.zeros_like(parameter) if gradient is None else gradient.detach()
                  for (name, parameter), gradient in zip(parameters.items(), result)}
        require(all(self.torch.isfinite(g).all().item() for g in result.values()), "Nonfinite gradient")
        return result

    def compare_gradients(self, actual, expected, name):
        require(set(actual) == set(expected), "Gradient parameter set differs")
        maximum = 0.0
        for key in actual:
            self.close(actual[key], expected[key], name + "/" + key,
                       TOL["gradient_atol"], TOL["gradient_rtol"])
            maximum = max(maximum, (actual[key] - expected[key]).abs().max().item())
        return maximum

    def source_objective(self, source_function, **kwargs):
        self.source_objective_calls += 1
        require(self.source_objective_calls <= LIMITS["source_objective_calls"], "Source objective-call cap reached")
        result = source_function(**kwargs)
        require(result.surrogate.requires_grad and result.generating_log_q.requires_grad, "Generating-score gradient missing")
        for value in (result.detached_rewards, result.detached_log_q_mix,
                      result.detached_log_gamma, result.objective_without_log_partition):
            require(not value.requires_grad, "Source detached field retains gradient")
        return result


def make_parameters(torch_api, mode, m, d=2, h=2):
    p = d + 2
    shapes = {"W_x": (3 * h, p), "W_h": (3 * h, h), "W_o": (2, h)}
    if mode == "untied":
        shapes = {name: (m, *shape) for name, shape in shapes.items()}
    shapes.update({"b_x": (m, 3 * h), "b_h": (m, 3 * h), "b_o": (m, 2), "h0": (m, h)})
    if mode == "factored":
        shapes.update({"r_x": (m, p), "s_x": (m, 3 * h), "r_h": (m, h),
                       "s_h": (m, 3 * h), "r_o": (m, h), "s_o": (m, 2)})
    tensors = {}
    for index, (name, shape) in enumerate(shapes.items()):
        values = torch_api.arange(math.prod(shape), dtype=torch_api.float64, device="cpu")
        if name.startswith(("r_", "s_")):
            values = 0.8 + 0.012 * (values % 9) + 0.003 * index
        else:
            values = 0.019 * ((values % 13) - 6) + 0.003 * (index + 1)
        tensors[name] = values.reshape(shape).clone()
    return tensors


def parameter_snapshot(parameters):
    text = json.dumps([(name, tuple(value.shape), value.detach().tolist())
                       for name, value in parameters.items()], separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_storage_and_counts(torch_api, decoder, parameters, supplied, mode, m):
    storage_addresses = [value.untyped_storage().data_ptr() for value in parameters.values()]
    require(len(storage_addresses) == len(set(storage_addresses)), "Accidental parameter-storage alias")
    require(all(value.device.type == "cpu" and value.dtype == torch_api.float64 and value.is_contiguous()
                for value in parameters.values()), "CPU float64 contiguous parameters required")
    for name, value in decoder.named_parameters():
        require(value.data_ptr() == supplied[name].data_ptr(), "Factory injection storage contract changed")
        shared = name in ("W_x", "W_h", "W_o") and mode != "untied"
        if not shared and m > 1:
            spans = sorted((value[branch].data_ptr(), value[branch].numel() * value.element_size()) for branch in range(m))
            require(all(start + size <= next_start for (start, size), (next_start, _) in zip(spans, spans[1:])),
                    "Private branch slices overlap: " + name)
    w, p, h = 40, 4, 2
    expected = w + m * (p + 15 * h + 4) if mode == "factored" else (w + 7 * h + 2) * m
    actual = sum(parameter.numel() for parameter in decoder.parameters())
    require(actual == expected, "Actual decoder parameter count differs from source formula")
    require(sum(value.numel() for name, value in parameters.items() if name.startswith("proxy/")) == 8,
            "Diagnostic proxy count differs")
    return actual


def record_expected(expected, *, n, edges, kind, m, repetitions=1, retain=False, factored=False):
    def add(key, value):
        expected[key] = expected.get(key, 0) + value
    p, h = 4, 2
    add(kind + "_complete_decoder_trajectories", repetitions)
    add(kind + "_decoder_steps", n * repetitions)
    add("decoder_dense_multiply_terms", n * repetitions * (3 * h * p + 3 * h * h + 2 * h))
    if factored:
        add("decoder_diagonal_multiply_elements", n * repetitions * (p + 8 * h + 2))
    add("decoder_gate_bias_additions", n * repetitions * (6 * h + 2))
    add("decoder_nonlinearity_elements", n * repetitions * 3 * h)
    add("hard_core_neighbor_inspection_bound", 2 * len(edges) * repetitions)
    if retain:
        add("generating_score_backward_trajectories", repetitions)
    if kind == "generation":
        add("caller_uniform_values_consumed", n * repetitions)


def run_case(torch_api, modules, checks, fixture, mode, m):
    name, n, edges, order, fugacities = fixture
    require(n <= LIMITS["max_nodes"] and m <= LIMITS["max_branches"], "Case dimension cap reached")
    domain_module, work = modules["domain"], modules["work_accounting"]
    domain = domain_module.ScheduleDomain(n, edges, order, "diagnostic/" + name, "fixed/" + name)
    audit = work.EncoderAudit("DIAGNOSTIC_LINEAR_H_PROXY_NOT_SOURCE_ENCODER", sha(Path(__file__)),
                             "fixed-linear-proxy-only", "None", 2, 0, 0, 6 * n)
    ledger = work.WorkLedger(audit)
    expected_counts = {"static_graph_encodings": 1, "encoder_message_passes": 0, "encoder_edge_visits": 0,
                       "encoder_dense_multiply_terms": 6 * n, "encoded_node_values": 2 * n}
    checks.active_case = {"fixture": name, "mode": mode, "members": m}
    checks.active_ledger = ledger
    checks.active_literal_counts = expected_counts
    proxy_w = torch_api.nn.Parameter(torch_api.tensor(((0.13, -0.07), (0.04, 0.09), (-0.05, 0.11)), dtype=torch_api.float64))
    proxy_b = torch_api.nn.Parameter(torch_api.tensor((0.02, -0.03), dtype=torch_api.float64))
    features = torch_api.tensor([(1.0, (v + 1) * 0.2, math.log(fugacities[v]) * 0.1)
                                for v in range(n)], dtype=torch_api.float64).reshape(n, 3)
    h_values = features @ proxy_w + proxy_b
    require(h_values.requires_grad, "Shared diagnostic H gradient missing")
    ledger.record_encoding(n)
    encoded = modules["source_encoder"].EncodedGraph(h_values, domain, audit.source_recipe_id)
    supplied = make_parameters(torch_api, mode, m)
    decoder = modules["ar_decoder"].make_recurrent_decoder(torch_api=torch_api, mode=mode,
        n_branches=m, embedding_dimension=2, hidden_dimension=2, parameter_tensors=supplied, ledger=ledger)
    parameters = {"proxy/W": proxy_w, "proxy/b": proxy_b}
    parameters.update({"decoder/" + key: value for key, value in decoder.named_parameters()})
    parameter_count = check_storage_and_counts(torch_api, decoder, parameters, supplied, mode, m)
    unchanged_pin = parameter_snapshot(parameters)
    all_assignments = tuple(itertools.product((0, 1), repeat=n))
    feasible = tuple(y for y in all_assignments if all(not (y[u] and y[v]) for u, v in edges))
    require(len(all_assignments) <= 16 and feasible, "Assignment-work bound or support violation")
    q_rows, infeasible_calls = [], 0
    for y in all_assignments:
        independent_feasibility = all(not (y[u] and y[v]) for u, v in edges)
        require(domain.feasible(y) == independent_feasibility, "Domain support disagrees with independent constraint check")
        if not independent_feasibility:
            for branch in range(m):
                score = decoder.log_density(encoded, y, branch=branch, retain_score_gradient=False)
                require(torch_api.isneginf(score).item() and not score.requires_grad, "Infeasible density is not zero")
                infeasible_calls += 1
            continue
        row = []
        for branch in range(m):
            score = decoder.log_density(encoded, y, branch=branch, retain_score_gradient=True)
            record_expected(expected_counts, n=n, edges=edges, kind="density", m=m, retain=True, factored=mode == "factored")
            forward = decoder.log_density(encoded, y, branch=branch, retain_score_gradient=False)
            record_expected(expected_counts, n=n, edges=edges, kind="density", m=m, factored=mode == "factored")
            require(score.requires_grad and not forward.requires_grad, "Density retention flag incorrect")
            checks.close(score.detach(), forward, "Retained/forward law", TOL["density_atol"])
            require(torch_api.isfinite(score).item() and score.exp().item() > 0, "Feasible support lost positive mass")
            row.append(score)
        q_rows.append(torch_api.stack(row))
    log_q = torch_api.stack(q_rows)
    q = log_q.exp()
    k = len(feasible)
    checks.close(q.sum(dim=0), q.new_ones((m,)), "Every branch normalizes")
    mixture = domain_module.FixedMixture((1.0, 2.0, 4.0) if m == 3 else (1.0,))
    pi = q.new_tensor(mixture.probabilities)
    log_pi = q.new_tensor(mixture.log_probabilities)
    log_mix = torch_api.logsumexp(log_q + log_pi, dim=1)
    q_mix = log_mix.exp()
    checks.close(q_mix.sum(), q.new_ones(()), "Mixture normalizes")
    log_fugacities = q.new_tensor([math.log(value) for value in fugacities])
    labels = q.new_tensor(feasible).reshape(k, n)
    log_gamma = (labels * log_fugacities).sum(dim=1)
    log_z = torch_api.logsumexp(log_gamma, dim=0)
    target = (log_gamma - log_z).exp()
    checks.close(target.sum(), q.new_ones(()), "Target normalizes")
    exact_mixture = (q_mix * (log_mix - log_gamma + log_z)).sum()
    exact_mean = (pi * (q * (log_q - log_gamma[:, None] + log_z)).sum(dim=0)).sum()
    mutual_information = (pi * (q * (log_q - log_mix[:, None])).sum(dim=0)).sum()
    checks.close(exact_mean.detach() - exact_mixture.detach(), mutual_information.detach(), "Mixture KL identity")
    reference = {"mixture_reverse_kl": checks.grad(exact_mixture, parameters),
                 "mean_branch_reverse_kl": checks.grad(exact_mean, parameters)}
    zero_reference = {key: torch_api.zeros_like(value) for key, value in parameters.items()}
    for term in ((q_mix.detach() * log_mix).sum(), (pi * (q.detach() * log_q).sum(dim=0)).sum()):
        gradients = checks.grad(term, parameters)
        for key in gradients:
            checks.close(gradients[key], zero_reference[key], "Explicit normalized density derivative/" + key, TOL["zero_atol"])

    samples = tuple(domain_module.GeneratedSchedule(y, branch, domain.graph_id, domain.order_id)
                    for branch in range(m) for y in feasible)
    cross = decoder.cross_member_log_densities(encoded, samples)
    record_expected(expected_counts, n=n, edges=edges, kind="density", m=m,
                    repetitions=len(samples), retain=True, factored=mode == "factored")
    record_expected(expected_counts, n=n, edges=edges, kind="density", m=m,
                    repetitions=len(samples) * (m - 1), factored=mode == "factored")
    expected_counts["cross_member_density_entries"] = len(samples) * m
    expected_counts["mixture_logsumexp_entries"] = 0
    origin_mask = cross.new_zeros(cross.shape)
    for index, sample in enumerate(samples):
        origin_mask[index, sample.generating_branch] = 1
        checks.close(cross[index].detach(), log_q[index % k].detach(), "Complete cross densities", TOL["density_atol"])
    non_origin_gradients = checks.grad((cross * (1 - origin_mask)).sum(), parameters)
    for key in non_origin_gradients:
        checks.close(non_origin_gradients[key], zero_reference[key], "Non-generating replay gradient/" + key, TOL["zero_atol"])
    if m > 1:
        one_branch = checks.grad(log_q[0, 0], parameters)
        for key, value in one_branch.items():
            if not key.startswith("decoder/"):
                continue
            ordinary_name = key.removeprefix("decoder/")
            is_shared = ordinary_name in ("W_x", "W_h", "W_o") and mode != "untied"
            if not is_shared:
                checks.close(value[1:], torch_api.zeros_like(value[1:]), "Private branch isolation/" + key, TOL["zero_atol"])

    baseline = domain_module.IndependentBaseline((0.3, -0.4, 0.2)[:m], "predeclared_constant",
                                               "protocol-v1-fixed-nonzero-values", True)
    objective = modules["mixture_objective"].fixed_mixture_reverse_kl_surrogate
    expected_surrogates = {kind: cross.new_zeros(()) for kind in reference}
    for index, sample in enumerate(samples):
        quadrature_weight = (pi[sample.generating_branch] * q[index % k, sample.generating_branch]).detach()
        for kind in reference:
            result = checks.source_objective(objective, torch_api=torch_api, encoded=encoded,
                samples=(sample,), cross_log_q=cross[index:index + 1], mixture=mixture,
                log_fugacities=log_fugacities, baseline=baseline, sampling_design="iid_mixture",
                objective_kind=kind, ledger=ledger)
            expected_counts["mixture_logsumexp_entries"] += m
            expected_surrogates[kind] = expected_surrogates[kind] + quadrature_weight * result.surrogate
    gradient_errors = {}
    for kind in reference:
        gradient_errors[kind] = checks.compare_gradients(checks.grad(expected_surrogates[kind], parameters),
                                                        reference[kind], "Exact expected source estimator/" + kind)
    if n:
        require(max(reference["mixture_reverse_kl"]["proxy/W"].abs().max().item(),
                    reference["mixture_reverse_kl"]["proxy/b"].abs().max().item()) > 1e-12,
                "Shared proxy derivative check is vacuous")
    if m == 1:
        checks.close(exact_mixture.detach(), exact_mean.detach(), "M=1 value coincidence")
        checks.compare_gradients(reference["mixture_reverse_kl"], reference["mean_branch_reverse_kl"], "M=1 gradient coincidence")

    counts = tuple(range(1, m + 1))
    batch = tuple(domain_module.GeneratedSchedule(feasible[(branch + j) % k], branch, domain.graph_id, domain.order_id)
                  for branch, count in enumerate(counts) for j in range(count))
    batch_cross = decoder.cross_member_log_densities(encoded, batch)
    record_expected(expected_counts, n=n, edges=edges, kind="density", m=m,
                    repetitions=len(batch), retain=True, factored=mode == "factored")
    record_expected(expected_counts, n=n, edges=edges, kind="density", m=m,
                    repetitions=len(batch) * (m - 1), factored=mode == "factored")
    expected_counts["cross_member_density_entries"] += len(batch) * m
    batch_log_gamma = batch_cross.new_tensor([sum(y * lf for y, lf in zip(sample.assignment, log_fugacities.tolist())) for sample in batch])
    batch_generating = torch_api.stack([batch_cross[index, sample.generating_branch] for index, sample in enumerate(batch)])
    finite_batch_errors = {}
    for design in ("iid_mixture", "stratified_per_branch"):
        weights_literal = [(1 / len(batch)) if design == "iid_mixture" else mixture.probabilities[sample.generating_branch] / counts[sample.generating_branch]
                           for sample in batch]
        weights = batch_cross.new_tensor(weights_literal)
        checks.close(weights.sum(), weights.new_ones(()), "Finite sample weights sum")
        if design == "stratified_per_branch":
            for branch in range(m):
                total = sum(weight for sample, weight in zip(batch, weights_literal) if sample.generating_branch == branch)
                checks.close(weights.new_tensor(total), pi[branch], "Stratified branch total")
        for kind in reference:
            result = checks.source_objective(objective, torch_api=torch_api, encoded=encoded, samples=batch,
                cross_log_q=batch_cross, mixture=mixture, log_fugacities=log_fugacities,
                baseline=baseline, sampling_design=design, objective_kind=kind, ledger=ledger)
            expected_counts["mixture_logsumexp_entries"] += len(batch) * m
            density_reward = torch_api.logsumexp(batch_cross.detach() + log_pi, dim=1) if kind == "mixture_reverse_kl" else batch_generating.detach()
            reward = density_reward - batch_log_gamma
            fixed_baselines = batch_cross.new_tensor([baseline.values_by_branch[s.generating_branch] for s in batch])
            manual = (weights * (reward - fixed_baselines).detach() * batch_generating).sum()
            checks.close(result.objective_without_log_partition, (weights * reward).sum().detach(), "Finite batch objective weights")
            checks.close(result.surrogate.detach(), manual.detach(), "Finite batch surrogate weights")
            finite_batch_errors[design + "/" + kind] = checks.compare_gradients(checks.grad(result.surrogate, parameters),
                                                        checks.grad(manual, parameters), "Finite batch gradient weights")

    q_values = q.detach().tolist()
    vectors = ((0.0,) * n, (0.875,) * n, tuple((0.125, 0.375, 0.625, 0.875)[i] for i in range(n)))
    minimum_threshold_distance = None
    for branch in range(m):
        for uniforms in vectors:
            prefix = {}
            for step, node in enumerate(order):
                compatible = [idx for idx, y in enumerate(feasible) if all(y[v] == value for v, value in prefix.items())]
                mass = sum(q_values[idx][branch] for idx in compatible)
                active_mass = sum(q_values[idx][branch] for idx in compatible if feasible[idx][node] == 1)
                require(mass > 0, "Enumerated prefix has zero probability")
                conditional_one = active_mass / mass
                blocked = any(prefix.get(v) == 1 for u, v in edges if u == node) or any(prefix.get(u) == 1 for u, v in edges if v == node)
                if blocked:
                    require(active_mass == 0, "Blocked enumerated prefix permits activation")
                else:
                    distance = abs(uniforms[step] - conditional_one)
                    require(distance > TOL["density_atol"], "Finite generation point lies on an ambiguous numerical threshold")
                    minimum_threshold_distance = distance if minimum_threshold_distance is None else min(minimum_threshold_distance, distance)
                prefix[node] = int(uniforms[step] < conditional_one)
            expected_assignment = tuple(prefix[v] for v in range(n))
            generated = decoder.generate(encoded, branch=branch, uniforms=uniforms)
            record_expected(expected_counts, n=n, edges=edges, kind="generation", m=m, factored=mode == "factored")
            require(generated.assignment == expected_assignment, "Generation disagrees with enumerated conditional law")
            require(generated.graph_id == domain.graph_id and generated.order_id == domain.order_id,
                    "Generated graph/order identity changed")
            require(generated.generating_branch == branch and domain.feasible(generated.assignment), "Generated support/member incorrect")
            require(q_values[feasible.index(generated.assignment)][branch] > 0, "Generated assignment has zero teacher-forced probability")
    if n:
        rejected = False
        try:
            decoder.generate(encoded, branch=0, uniforms=(1.0,) * n)
        except ValueError:
            rejected = True
        require(rejected, "Uniform endpoint 1 was not rejected")

    require(ledger.counts == expected_counts, "Source ledger differs from literal forward/retention accounting")
    require(parameter_snapshot(parameters) == unchanged_pin, "Parameters changed during diagnostic")
    require(ledger.counts.get("density_decoder_steps", 0) + ledger.counts.get("generation_decoder_steps", 0) <= LIMITS["charged_decoder_steps"], "Per-case step bound exceeded")
    return {"fixture": name, "mode": mode, "members": m, "order": order, "fugacities": fugacities,
        "feasible_assignments": k, "total_assignments": len(all_assignments), "infeasible_uncharged_density_returns": infeasible_calls,
        "decoder_parameters": parameter_count, "shared_proxy_parameters": 8,
        "all_parameter_tensor_names_checked": list(parameters), "no_parameter_storage_alias": True,
        "exact_gradient_max_absolute_errors": gradient_errors, "finite_batch_gradient_errors": finite_batch_errors,
        "finite_batch_branch_counts": counts, "fixed_pi": mixture.probabilities,
        "minimum_pointwise_uniform_threshold_distance": minimum_threshold_distance,
        "source_ledger": ledger.counts, "literal_expected_ledger": expected_counts,
        "proxy_forwards": 1, "parameters_unchanged": True,
        "interpretation": "correctness fixture only; no optimization, accuracy or novelty claim"}


def run_head_case(torch_api, modules, checks, head_mode):
    """Same neural source operations, independent per-member replay wrapper.

    The reference repeats source _transition/_head calls; it contains no GRU or
    affine-head equations. Tracing wraps calls only, preserving values/graphs.
    """
    name, n, edges, order, fugacities = FIXTURES[1]
    m, d, h, p = 3, 2, 2, 4
    contracts = modules["domain"]
    domain = contracts.ScheduleDomain(n, edges, order, "diagnostic/head/path4", "fixed/head/path4")
    audit = modules["work_accounting"].EncoderAudit("DIAGNOSTIC_LINEAR_H_PROXY_NOT_SOURCE_ENCODER",
        sha(Path(__file__)), "fixed-linear-proxy-only", "None", d, 0, 0, 6 * n)
    production_ledger = modules["head_work"].SharedTransitionLedger(audit)
    reference_ledger = modules["head_work"].SharedTransitionLedger(audit)
    checks.active_case = {"fixture": name, "head_mode": head_mode, "members": m}
    checks.active_ledger = reference_ledger
    checks.active_literal_counts = None
    proxy_w = torch_api.nn.Parameter(torch_api.tensor(((0.13, -0.07), (0.04, 0.09), (-0.05, 0.11)), dtype=torch_api.float64))
    proxy_b = torch_api.nn.Parameter(torch_api.tensor((0.02, -0.03), dtype=torch_api.float64))
    features = torch_api.tensor([(1.0, (v + 1) * 0.2, math.log(fugacities[v]) * 0.1) for v in range(n)], dtype=torch_api.float64)
    encoded = modules["source_encoder"].EncodedGraph(features @ proxy_w + proxy_b, domain, audit.source_recipe_id)
    production_ledger.record_encoding(n)
    shapes = {"W_x": (3 * h, p), "W_h": (3 * h, h), "b_x": (3 * h,),
              "b_h": (3 * h,), "h0": (h,), "b_o": (m, 2)}
    if head_mode == "ordinary_heads":
        shapes["W_o"] = (m, 2, h)
    else:
        shapes.update({"W_o": (2, h), "r_o": (m, h), "s_o": (m, 2)})
    supplied = {}
    for index, (key, shape) in enumerate(shapes.items()):
        values = torch_api.arange(math.prod(shape), dtype=torch_api.float64)
        values = 0.8 + 0.012 * (values % 9) + 0.003 * index if key.startswith(("r_", "s_")) else 0.019 * ((values % 13) - 6) + 0.003 * (index + 1)
        supplied[key] = values.reshape(shape).clone()
    decoder = modules["head_decoder"].make_shared_transition_decoder(torch_api=torch_api,
        head_mode=head_mode, n_branches=m, embedding_dimension=d, hidden_dimension=h,
        parameter_tensors=supplied, ledger=production_ledger)
    parameters = {"proxy/W": proxy_w, "proxy/b": proxy_b}
    parameters.update({"decoder/" + key: value for key, value in decoder.named_parameters()})
    addresses = [value.untyped_storage().data_ptr() for value in parameters.values()]
    require(len(addresses) == len(set(addresses)), "Head-control parameter-storage alias")
    require(all(value.device.type == "cpu" and value.dtype == torch_api.float64 for value in parameters.values()), "Head CPU/dtype mismatch")
    actual_parameters = sum(value.numel() for value in decoder.parameters())
    require(actual_parameters == (68 if head_mode == "ordinary_heads" else 72), "Head-control parameter count mismatch")
    for key in ("b_o", "W_o" if head_mode == "ordinary_heads" else "r_o", "b_o" if head_mode == "ordinary_heads" else "s_o"):
        value = getattr(decoder, key)
        spans = [(value[b].data_ptr(), value[b].numel() * value.element_size()) for b in range(m)]
        require(all(a + size <= b for (a, size), (b, _) in zip(spans, spans[1:])), "Private head slices overlap")
    parameter_pin = parameter_snapshot(parameters)
    trace, phase = {}, ["reference"]
    original_transition, original_head = decoder._transition, decoder._head_log_probabilities
    def traced_transition(*args):
        key = phase[0] + "/transitions"
        trace[key] = trace.get(key, 0) + 1
        return original_transition(*args)
    def traced_head(*args):
        key = phase[0] + "/heads"
        trace[key] = trace.get(key, 0) + 1
        grad_key = phase[0] + ("/heads_grad_enabled" if torch_api.is_grad_enabled() else "/heads_no_grad")
        trace[grad_key] = trace.get(grad_key, 0) + 1
        return original_head(*args)
    decoder._transition, decoder._head_log_probabilities = traced_transition, traced_head
    feasible = tuple(y for y in itertools.product((0, 1), repeat=n) if all(not (y[u] and y[v]) for u, v in edges))
    k = len(feasible)
    def naive_member_replay(y, branch):
        reference_ledger.record_shared_trajectory(kind="density", n_nodes=n, n_edges=len(edges),
            embedding_dimension=d, hidden_dimension=h, heads_per_step=1,
            factored_heads=head_mode == "factored_heads", retain_generating_score_graph=True)
        state, previous, prefix = decoder.h0, 0, [None] * n
        score = encoded.H.sum() * 0 + decoder.h0.sum() * 0
        for node in order:
            blocked = domain.blocked(node, prefix)
            state = decoder._transition(state, encoded.H[node], previous, blocked)
            score = score + decoder._head_log_probabilities(branch, state, blocked)[y[node]]
            prefix[node], previous = y[node], y[node]
        return score
    log_q = torch_api.stack([torch_api.stack([naive_member_replay(y, branch) for branch in range(m)]) for y in feasible])
    q = log_q.exp()
    require(torch_api.isfinite(log_q).all().item() and (q > 0).all().item(), "Head feasible support lost finite positive mass")
    checks.close(q.sum(dim=0), q.new_ones((m,)), "Head member laws normalize")
    phase[0] = "forward_reuse"
    infeasible_calls = 0
    for y in itertools.product((0, 1), repeat=n):
        if not all(not (y[u] and y[v]) for u, v in edges):
            values = decoder.full_log_densities(encoded, y, generating_branch=None)
            require(torch_api.isneginf(values).all().item() and not values.requires_grad, "Head infeasible law is not zero")
            infeasible_calls += 1
    for index, y in enumerate(feasible):
        values = decoder.full_log_densities(encoded, y, generating_branch=None)
        require(not values.requires_grad, "Forward-only reused density retains gradient")
        checks.close(values, log_q[index].detach(), "Independent/reused head densities", TOL["density_atol"])
    phase[0] = "retained_reuse"
    samples = tuple(contracts.GeneratedSchedule(y, branch, domain.graph_id, domain.order_id) for branch in range(m) for y in feasible)
    cross = decoder.cross_member_log_densities(encoded, samples)
    checks.active_ledger = production_ledger
    for index in range(len(samples)):
        checks.close(cross[index].detach(), log_q[index % k].detach(), "Reused head cross density", TOL["density_atol"])
    per_branch_errors = {}
    for branch in range(m):
        per_branch_errors[str(branch)] = checks.compare_gradients(checks.grad(cross[branch * k, branch], parameters),
                                     checks.grad(log_q[0, branch], parameters), "Independent/reused generating-score gradients")
    mixture = contracts.FixedMixture((1.0, 2.0, 4.0))
    pi, log_pi = q.new_tensor(mixture.probabilities), q.new_tensor(mixture.log_probabilities)
    log_mix = torch_api.logsumexp(log_q + log_pi, dim=1)
    q_mix = log_mix.exp()
    log_fugacities = q.new_tensor([math.log(value) for value in fugacities])
    log_gamma = (q.new_tensor(feasible) * log_fugacities).sum(dim=1)
    log_z = torch_api.logsumexp(log_gamma, dim=0)
    exact = {"mixture_reverse_kl": (q_mix * (log_mix - log_gamma + log_z)).sum(),
             "mean_branch_reverse_kl": (pi * (q * (log_q - log_gamma[:, None] + log_z)).sum(dim=0)).sum()}
    reference_gradients = {key: checks.grad(value, parameters) for key, value in exact.items()}
    baseline = contracts.IndependentBaseline((0.3, -0.4, 0.2), "predeclared_constant", "protocol-v1-fixed-nonzero-values", True)
    expected_scores = {key: cross.new_zeros(()) for key in exact}
    for index, sample in enumerate(samples):
        weight = (pi[sample.generating_branch] * q[index % k, sample.generating_branch]).detach()
        for kind in exact:
            result = checks.source_objective(modules["mixture_objective"].fixed_mixture_reverse_kl_surrogate,
                torch_api=torch_api, encoded=encoded, samples=(sample,), cross_log_q=cross[index:index + 1],
                mixture=mixture, log_fugacities=log_fugacities, baseline=baseline,
                sampling_design="iid_mixture", objective_kind=kind, ledger=production_ledger)
            expected_scores[kind] = expected_scores[kind] + weight * result.surrogate
    gradient_errors = {key: checks.compare_gradients(checks.grad(expected_scores[key], parameters), reference_gradients[key],
                                                    "Exact head estimator/" + key) for key in exact}
    require(max(reference_gradients["mixture_reverse_kl"]["proxy/W"].abs().max().item(),
                reference_gradients["mixture_reverse_kl"]["proxy/b"].abs().max().item()) > 1e-12,
            "Shared head H-proxy check is vacuous")
    expected_trace = {"reference/transitions": k * m * n, "reference/heads": k * m * n,
        "reference/heads_grad_enabled": k * m * n,
        "forward_reuse/transitions": k * n, "forward_reuse/heads": k * n * m, "forward_reuse/heads_no_grad": k * n * m,
        "retained_reuse/transitions": k * m * n, "retained_reuse/heads": k * m * n * m,
        "retained_reuse/heads_grad_enabled": k * m * n, "retained_reuse/heads_no_grad": k * m * n * (m - 1)}
    require(trace == expected_trace, "Actual head/transition call trace differs from reuse route")
    require(reference_ledger.counts["density_shared_recurrent_trajectories"] == k * m, "Naive reference replay count differs")
    require(reference_ledger.counts["density_head_step_evaluations"] == k * m * n, "Naive reference head count differs")
    require(production_ledger.counts["density_shared_recurrent_trajectories"] == k + k * m, "Production must replay recurrence once per configuration")
    require(production_ledger.counts["density_head_step_evaluations"] == (k + k * m) * n * m, "Production all-head count differs")
    require(production_ledger.counts["retained_generating_score_graphs"] == k * m, "Reused retained graph count differs")
    require(production_ledger.counts["cross_member_density_entries"] == k * m * m, "Reused cross matrix count differs")
    require(production_ledger.counts["mixture_logsumexp_entries"] == 2 * k * m * m, "Reused objective reduction count differs")
    def literal_head_counts(trajectories, heads_per_step, retained, encoder):
        steps, head_steps = trajectories * n, trajectories * n * heads_per_step
        result = {"density_shared_recurrent_trajectories": trajectories, "density_shared_recurrent_steps": steps,
            "density_head_step_evaluations": head_steps, "recurrent_dense_multiply_terms": steps * (3 * h * p + 3 * h * h),
            "head_dense_multiply_terms": head_steps * 2 * h, "shared_recurrent_bias_additions": steps * 6 * h,
            "private_head_bias_additions": head_steps * 2, "shared_recurrent_nonlinearity_elements": steps * 3 * h,
            "hard_core_neighbor_inspection_bound": trajectories * 2 * len(edges), "retained_generating_score_graphs": retained}
        if head_mode == "factored_heads":
            result["private_head_diagonal_multiply_elements"] = head_steps * (h + 2)
        if encoder:
            result.update(static_graph_encodings=1, encoder_message_passes=0, encoder_edge_visits=0,
                          encoder_dense_multiply_terms=6 * n, encoded_node_values=n * d,
                          cross_member_density_entries=k * m * m, mixture_logsumexp_entries=2 * k * m * m)
        return result
    literal_production = literal_head_counts(k + k * m, m, k * m, True)
    literal_reference = literal_head_counts(k * m, 1, k * m, False)
    require(production_ledger.counts == literal_production, "Head production ledger differs from complete literal accounting")
    require(reference_ledger.counts == literal_reference, "Head naive ledger differs from complete literal accounting")
    require(parameter_snapshot(parameters) == parameter_pin, "Head parameters changed")
    return {"fixture": name, "head_mode": head_mode, "members": m, "feasible_assignments": k,
        "decoder_parameters": actual_parameters, "shared_proxy_parameters": 8,
        "all_parameter_tensor_names_checked": list(parameters), "same_neural_source_operations": True,
        "infeasible_uncharged_density_returns": infeasible_calls,
        "independent_replay_gradient_errors": per_branch_errors, "exact_gradient_errors": gradient_errors,
        "actual_forward_call_trace": trace, "expected_forward_call_trace": expected_trace,
        "source_ledger": production_ledger.counts, "independent_replay_ledger": reference_ledger.counts,
        "literal_expected_ledger": literal_production, "literal_independent_replay_ledger": literal_reference,
        "charged_recurrent_steps": production_ledger.counts["density_shared_recurrent_steps"] + reference_ledger.counts["density_shared_recurrent_steps"],
        "proxy_forwards": 1, "parameters_unchanged": True,
        "interpretation": "Head reuse correctness/control only; no storage, latency, memory or accuracy superiority claim"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--admission-id", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    require(output.is_relative_to(BASE) and output != BASE and output.is_dir(), "Precreated private output folder inside diagnostic packet required")
    require(args.admission_id.strip(), "Recorded root admission ID required")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    started = time.monotonic()
    terminal = {"status": "started", "admission_id": args.admission_id, "limits": LIMITS, "tolerances": TOL,
                "source_imported": False, "torch_imported": False, "case_results": [],
                "no_rng_optimizer_dataset_or_real_encoder": True}
    checks = None
    try:
        def wall_failure(signum, frame):
            raise TimeoutError("Internal 110-second diagnostic wall cap reached")
        signal.signal(signal.SIGALRM, wall_failure)
        signal.alarm(LIMITS["internal_wall_seconds"])
        terminal["source_pins"] = verify_source()
        import torch
        terminal["torch_imported"] = True
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.set_default_dtype(torch.float64)
        terminal["torch_version"] = torch.__version__
        require(torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1, "Thread cap not applied")
        modules = import_existing_source()
        terminal["source_imported"] = True
        checks = Checks(torch)
        specifications = [(fixture, mode, 1 if mode == "single" else 3)
                          for fixture in FIXTURES for mode in ("factored", "single", "untied")]
        specifications += [(FIXTURES[1], "factored", 1), (FIXTURES[1], "untied", 1)]
        require(len(specifications) + 2 == LIMITS["cases"], "Case count differs from sealed protocol")
        total_steps = 0
        for fixture, mode, members in specifications:
            result = run_case(torch, modules, checks, fixture, mode, members)
            terminal["case_results"].append(result)
            ledger = result["source_ledger"]
            total_steps += ledger.get("density_decoder_steps", 0) + ledger.get("generation_decoder_steps", 0)
            require(total_steps <= LIMITS["charged_decoder_steps"], "Total charged decoder step cap reached")
            print(json.dumps({"completed_case": len(terminal["case_results"]), "fixture": fixture[0], "mode": mode, "members": members}), flush=True)
        for head_mode in ("ordinary_heads", "factored_heads"):
            result = run_head_case(torch, modules, checks, head_mode)
            terminal["case_results"].append(result)
            total_steps += result["charged_recurrent_steps"]
            require(total_steps <= LIMITS["charged_decoder_steps"], "Total charged decoder step cap reached")
            print(json.dumps({"completed_case": len(terminal["case_results"]), "head_mode": head_mode}), flush=True)
        terminal.update(status="passed", autograd_calls=checks.autograd_calls,
                        source_objective_calls=checks.source_objective_calls, charged_decoder_steps=total_steps)
        terminal["qualification_scope"] = "Only these deterministic complete tiny laws and this H proxy; real encoder/RNG/performance/scalability unqualified"
    except BaseException as error:
        terminal.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        signal.alarm(0)
        if checks is not None:
            terminal["autograd_calls"] = checks.autograd_calls
            terminal["source_objective_calls"] = checks.source_objective_calls
            terminal["last_active_case"] = checks.active_case
            terminal["last_active_case_partial_source_ledger"] = dict(checks.active_ledger.counts) if checks.active_ledger is not None else None
            terminal["last_active_case_partial_literal_counts"] = checks.active_literal_counts
        terminal["elapsed_wall_seconds_including_imports"] = time.monotonic() - started
        (output / "diagnostic_terminal.json").write_text(json.dumps(terminal, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": terminal["status"], "terminal": str(output / "diagnostic_terminal.json")}), flush=True)
    return 0 if terminal["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
