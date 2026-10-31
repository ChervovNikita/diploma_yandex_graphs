"""Unexecuted exhaustive metrics over the pinned no-gradient evaluator.

The optimizer never receives oracle probabilities or batched density gradients.
All metrics retain every complete feasible assignment; no renormalization of q.
"""
import hashlib
import json
import math


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def tiny_identity_parity(*, torch, modules, evaluator, decoder, arm, evaluation):
    # The four qualified v3 arithmetic cases, now applied to both added paths.
    cases = (("empty0", 0, (), ()), ("edgeless3", 3, (), (2, 0, 1)),
             ("path3", 3, ((0, 1), (1, 2)), (2, 0, 1)),
             ("triangle_isolate4", 4, ((0, 1), (1, 2), (0, 2)), (2, 3, 0, 1)))
    records = []
    with torch.no_grad():
        for name, n, edges, order in cases:
            domain = modules["domain"].ScheduleDomain(n, edges, order, "tiny/" + name, "fixed/" + name)
            H = (torch.arange(n * 16).reshape(n, 16).to(torch.float64) % 11 - 5) / 17
            encoded = modules["encoder"].EncodedGraph(H, domain, "arithmetic-parity-proxy-only")
            rows = [tuple((mask >> node) & 1 for node in range(n)) for mask in range(1 << n)]
            rows = [row for row in rows if domain.feasible(row)]
            counts = {}
            vector = evaluator.all_member_log_densities(torch_api=torch, decoder=decoder, encoded=encoded,
                assignments=rows, family=arm["family"], counts=counts)
            scalar = torch.stack([torch.stack([decoder.log_density(encoded, row, branch=m,
                retain_score_gradient=False) for m in range(arm["members"])]) for row in rows])
            require(not vector.requires_grad and not scalar.requires_grad, "Tiny identity parity retained gradient")
            require(tuple(vector.shape) == (len(rows), arm["members"]), "Tiny identity omitted a member")
            require(torch.allclose(vector, scalar, atol=evaluation["parity_atol"], rtol=0), "Tiny identity scalar/batched parity failure")
            records.append({"case": name, "binary_states": 1 << n, "feasible_states": len(rows),
                "maximum_absolute_error": (vector - scalar).abs().max().item(), "vector_counts": counts})
    return {"status": "completed_parity_pass", "all_tiny_states_checked": records,
            "scope": "New identity arithmetic paths only; proxy H is never a fitted feature or scientific result"}


def evaluate_complete_law(*, torch, evaluator, decoder, encoded, weighted_states,
                          partition, arm, evaluation, on_chunk):
    require(encoded.domain.n_nodes == 15 and partition > 0, "Complete spatial15 target required")
    require(len(weighted_states) > 0 and sum(weight for _, weight in weighted_states) == partition,
            "Exact integer target weights/partition must agree")
    masks = [mask for mask, _ in weighted_states]
    require(masks == sorted(set(masks)), "Complete ascending-mask support required")
    F, M = len(weighted_states), arm["members"]
    counts, digest = {}, hashlib.sha256()
    member_mass = torch.zeros(M)
    branch_kl = torch.zeros(M)
    target_marginals = torch.zeros(15)
    mixture_marginals = torch.zeros(15)
    mixture_mass = kl_q_p = kl_p_q = tv = entropy_q = entropy_p = mutual_information = 0.0
    covered = 0
    with torch.no_grad():
        for start in range(0, F, evaluation["chunk_rows"]):
            states = weighted_states[start:start + evaluation["chunk_rows"]]
            rows = [tuple((mask >> node) & 1 for node in range(15)) for mask, _ in states]
            require(all(encoded.domain.feasible(row) for row in rows), "Oracle returned infeasible row")
            logq = evaluator.all_member_log_densities(torch_api=torch, decoder=decoder, encoded=encoded,
                assignments=rows, family=arm["family"], counts=counts)
            require(not logq.requires_grad and tuple(logq.shape) == (len(rows), M), "Full no-gradient member law required")
            logp = encoded.H.new_tensor([math.log(weight) - math.log(partition) for _, weight in states])
            logmix = torch.logsumexp(logq, dim=-1) - math.log(M)
            q, p, qm = logq.exp(), logp.exp(), logmix.exp()
            labels = encoded.H.new_tensor(rows)
            target_marginals += (p[:, None] * labels).sum(dim=0)
            mixture_marginals += (qm[:, None] * labels).sum(dim=0)
            require(torch.isfinite(logq).all().item() and (q > 0).all().item() and (qm > 0).all().item(),
                    "Feasible positive law underflow or nonfinite log density")
            member_mass += q.sum(dim=0)
            branch_kl += (q * (logq - logp[:, None])).sum(dim=0)
            mixture_mass += qm.sum().item()
            kl_q_p += (qm * (logmix - logp)).sum().item()
            kl_p_q += (p * (logp - logmix)).sum().item()
            tv += (0.5 * (qm - p).abs().sum()).item()
            entropy_q += (-qm * logmix).sum().item()
            entropy_p += (-p * logp).sum().item()
            mutual_information += (q * (logq - logmix[:, None])).sum().item() / M
            digest.update(json.dumps({"masks": [mask for mask, _ in states], "logq": logq.tolist()}, separators=(",", ":")).encode())
            covered += len(states)
            on_chunk({"required_feasible_states": F, "completed_feasible_states": covered, "counts": dict(counts)})
    tolerance = evaluation["normalization_atol"]
    require(covered == F, "Whole-law evaluation incomplete")
    require(torch.allclose(member_mass, torch.ones_like(member_mass), atol=tolerance, rtol=0), "Member law failed frozen normalization gate")
    require(abs(mixture_mass - 1) <= tolerance, "Mixture law failed frozen normalization gate")
    require(all(math.isfinite(x) for x in (kl_q_p, kl_p_q, tv, entropy_q, entropy_p, mutual_information)),
            "Whole-law metric nonfinite")
    # Small floating signed errors are reported unchanged, never clamped to zero.
    return {"complete_feasible_states": F, "complete_binary_states": 32768,
            "structural_zero_states": 32768 - F, "support_complete": True,
            "member_mass": member_mass.tolist(), "mixture_mass": mixture_mass,
            "kl_q_p": kl_q_p, "kl_p_q": kl_p_q, "total_variation": tv,
            "mixture_entropy": entropy_q, "target_entropy": entropy_p,
            "mean_branch_kl_q_p": branch_kl.mean().item(), "branch_kl_q_p": branch_kl.tolist(),
            "member_information": mutual_information, "full_member_loglaw_sha256": digest.hexdigest(),
            "target_node_marginals": target_marginals.tolist(), "mixture_node_marginals": mixture_marginals.tolist(),
            "node_marginal_errors_q_minus_p": (mixture_marginals - target_marginals).tolist(),
            "node_marginal_mean_absolute_error": (mixture_marginals - target_marginals).abs().mean().item(),
            "node_marginal_maximum_absolute_error": (mixture_marginals - target_marginals).abs().max().item(),
            "vector_counts": counts, "normalization_atol": tolerance,
            "metric_arithmetic": "Complete integer oracle; float64 log-density evaluation and sums; q never renormalized"}


def actual_H_parity(*, torch, evaluator, decoder, encoded, weighted_states, arm, evaluation):
    F, M = len(weighted_states), arm["members"]
    indices = sorted({j * (F - 1) // 15 for j in range(16)})
    rows = [tuple((weighted_states[index][0] >> node) & 1 for node in range(15)) for index in indices]
    counts = {}
    with torch.no_grad():
        vector = evaluator.all_member_log_densities(torch_api=torch, decoder=decoder, encoded=encoded,
            assignments=rows, family=arm["family"], counts=counts)
        if arm["family"] == "shared":
            scalar = torch.stack([decoder.full_log_densities(encoded, row, generating_branch=None) for row in rows])
        else:
            scalar = torch.stack([torch.stack([decoder.log_density(encoded, row, branch=m, retain_score_gradient=False)
                for m in range(M)]) for row in rows])
    require(not vector.requires_grad and not scalar.requires_grad, "Parity retained gradient")
    require(tuple(vector.shape) == tuple(scalar.shape) == (len(indices), M), "Final parity omitted a member density")
    require(torch.allclose(vector, scalar, atol=evaluation["parity_atol"], rtol=0), "Frozen actual-H parity failure")
    return {"selected_ordinals": indices, "selection": "Frozen 16-slot spread in complete ascending-mask support; no score selection",
            "all_members": M, "maximum_absolute_error": (vector - scalar).abs().max().item(), "vector_counts": counts,
            "coverage": "Additional implementation gate; final law still uses all feasible states"}
