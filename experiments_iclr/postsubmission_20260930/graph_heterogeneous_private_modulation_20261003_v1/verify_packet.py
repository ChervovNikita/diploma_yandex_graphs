#!/usr/bin/env python3
"""Read-only stdlib integrity/count/bookkeeping/rational-identity verifier."""
import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path


PACKET = Path(__file__).resolve().parent
WORKSPACE = PACKET.parent.parent
V24_SHA = "3618a9d22dbbc6369f850f2fb680ccb3e8727344998e0ec5ea945802c58ba2eb"
GEM_SHA = "98e36e1490d7bce9bd90430c739c891948d058bd511f677bf35c60025af8f3ae"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(name):
    return json.loads((PACKET / name).read_text())


def workspace_file(name):
    relative = Path(name)
    assert not relative.is_absolute() and ".." not in relative.parts, name
    path = WORKSPACE / relative
    assert path.is_file() and not path.is_symlink(), name
    assert path.resolve().is_relative_to(WORKSPACE), name
    return path


def vec(values):
    return [Fraction(value) for value in values]


def dot(x, y):
    assert len(x) == len(y)
    return sum((a * b for a, b in zip(x, y)), Fraction(0))


def row_map(x, matrix):
    return [sum((x[i] * matrix[i][j] for i in range(len(x))), Fraction(0))
            for j in range(len(matrix[0]))]


def cp_gradients(c, q, u, gradients):
    c_grad = [sum((q[r] * dot(u, gradients[m][r]) for r in range(len(q))), Fraction(0))
              for m in range(len(c))]
    q_grad = [sum((c[m] * dot(u, gradients[m][r]) for m in range(len(c))), Fraction(0))
              for r in range(len(q))]
    u_grad = [sum((c[m] * q[r] * gradients[m][r][j]
                   for m in range(len(c)) for r in range(len(q))), Fraction(0))
              for j in range(len(u))]
    return c_grad, q_grad, u_grad


def verify():
    seal = read_json("SEAL.json")
    manifest = PACKET / "MANIFEST.sha256"
    assert digest(manifest) == seal["manifest_sha256"]
    names = []
    for line in manifest.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        assert match, "Malformed manifest"
        expected, name = match.groups()
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        path = PACKET / relative
        assert path.is_file() and not path.is_symlink(), name
        assert path.resolve().is_relative_to(PACKET), name
        assert digest(path) == expected, name
        names.append(name)
    actual = {str(f.relative_to(PACKET)) for f in PACKET.rglob("*")
              if f.is_file() and f.name not in {"MANIFEST.sha256", "SEAL.json"}}
    assert len(names) == len(set(names)) == seal["files_sealed"]
    assert set(names) == actual

    inputs = read_json("INPUT_BINDINGS.json")
    assert inputs["consulted_index_sha256"] == V24_SHA
    assert inputs["immutable_GEM_closure_manifest_sha256"] == GEM_SHA
    for row in inputs["inputs"]:
        assert digest(workspace_file(row["path"])) == row["sha256"], row["path"]
    index_binding = next(row for row in inputs["inputs"] if row["path"].endswith(
        "index_v24/LITERATURE_INDEX.json"))
    index = json.loads(workspace_file(index_binding["path"]).read_text())
    assert len(index["paper_records"]) == 108
    scope = read_json("READ_SCOPES.json")
    methods = scope["methods"]
    ids = [row["canonical_id"] for row in methods]
    assert ids == ["arXiv:2505.12027v2", "arXiv:2605.00731v1", "arXiv:2112.14936v1"]
    for sid in ["2505.12027", "2605.00731", "2112.14936"]:
        assert not any(sid in row["canonical_id"] for row in index["paper_records"])
    accounting = scope["accounting"]
    expected = {
        "primary_method_scopes_this_packet": 3, "new_method_record_ids_absent_from_v24": 3,
        "primary_unique_ids_retrieved": 3, "abs_html_retrieval_rows": 6,
        "new_full_primary_reads": 0, "retained_scoped_method_rereads": 0,
        "author_source_reads": 0, "abstracts_read": 3, "new_method_driver_or_launch_adoptions": 0,
        "paragraph_heading_blocks": 150, "paragraph_blocks": 106, "heading_blocks": 44,
        "display_math_nodes": 32, "supplementary_math_nodes": 16,
        "distinct_math_nodes_or_fragments": 48, "complete_printed_algorithm_figures": 1,
        "previously_unknown_paper_identities_claimed": False,
    }
    for key, value in expected.items():
        assert accounting[key] == value, key
    assert inputs["read_accounting"] == accounting
    totals = {key: 0 for key in methods[0]["counts"]}
    for row in methods:
        sid = row["canonical_id"].split(":", 1)[1]
        assert len(row["file_bindings"]) == 12
        for binding in row["file_bindings"]:
            assert digest(workspace_file(binding["path"])) == binding["sha256"]
        blocks = read_json(f"primary/{sid}_html_blocks.json")
        display = read_json(f"primary/{sid}_html_display_math.json")
        all_math = read_json(f"primary/{sid}_html_all_math.json")
        selected_indices = [i for lo, hi in row["paragraph_blocks_inclusive"] for i in range(lo, hi + 1)]
        assert len(selected_indices) == len(set(selected_indices))
        assert all(0 <= i < len(blocks) and blocks[i]["index"] == i for i in selected_indices)
        selected = [blocks[i] for i in selected_indices]
        counts = row["counts"]
        assert len(selected) == counts["paragraph_heading_blocks"]
        assert sum(b["tag"] == "p" for b in selected) == counts["paragraph_blocks"]
        assert sum(re.fullmatch(r"h[1-6]", b["tag"]) is not None for b in selected) == counts["heading_blocks"]
        assert counts["paragraph_blocks"] + counts["heading_blocks"] == len(selected)
        assert row["display_math_indices_used"] == row["display_math_indices_previewed"] == list(range(len(display)))
        assert len(display) == counts["display_math_nodes"]
        assert [d["all_math_index"] for d in display] == row["display_to_all_math_indices"]
        for d in display:
            a = all_math[d["all_math_index"]]
            assert d["id"] == a["id"] and d["alttext"] == a["alttext"]
        supplemental = row["supplementary_all_math_indices_used"]
        assert len(supplemental) == len(set(supplemental)) == counts["supplementary_math_nodes"]
        assert all(0 <= i < len(all_math) for i in supplemental)
        assert not set(supplemental) & set(row["display_to_all_math_indices"])
        assert len(set(supplemental) | set(row["display_to_all_math_indices"])) == counts["distinct_math_nodes_or_fragments"]
        algorithms = read_json(f"primary/{sid}_html_algorithms.json")
        if row["algorithm_figure_read"]:
            chosen = [alg for alg in algorithms if alg["id"] == row["algorithm_figure_read"]]
            assert len(chosen) == 1 and 300 < len(chosen[0]["text"]) < 6000
            assert counts["complete_printed_algorithm_figures"] == 1
        else:
            assert counts["complete_printed_algorithm_figures"] == 0
        for key in totals:
            totals[key] += counts[key]
    for key, value in totals.items():
        assert accounting[key] == value, key

    retrievals = read_json("PRIMARY_RETRIEVAL.json")
    assert len(retrievals) == 6 and {row["canonical_id"] for row in retrievals} == set(ids)
    assert {(row["canonical_id"], row["kind"]) for row in retrievals} == {
        (sid, kind) for sid in ids for kind in ["abs", "html"]}
    for row in retrievals:
        assert row["status"] == 200 and row["url"].startswith("https://arxiv.org/")
        path = workspace_file(row["path"])
        assert digest(path) == row["sha256"] and path.stat().st_size == row["bytes"]
        assert len(row["mechanical_extracts"]) == 5
        for extract in row["mechanical_extracts"].values():
            path = workspace_file(extract["path"])
            assert digest(path) == extract["sha256"]
            assert len(json.loads(path.read_text())) == extract["items"]
    discovery = read_json("DISCOVERY_RECEIPT.json")["rows"] + read_json("DISCOVERY_RETRY_RECEIPT.json")["rows"]
    assert len(discovery) == 6
    assert sum(row.get("status") == 200 for row in discovery) == 4
    assert sum("error" in row for row in discovery) == 2
    assert sum(len(row["entries"]) for row in discovery) == 53
    for row in discovery:
        if row.get("status") == 200:
            assert digest(workspace_file(row["path"])) == row["sha256"]

    papers = read_json("PAPER_CONCLUSIONS.json")
    assert papers["read_accounting"] == accounting
    assert [row["canonical_id"] for row in papers["primary_methods"]] == ids
    assert all(row["source_benefits_transferred"] is False for row in papers["primary_methods"])
    composition = papers["composition"]
    assert composition["naive_separable_member_relation_equivalence"] is True
    assert composition["local_effective_coordinate_rank_distinction"] is True
    for key in ["complete_HGT_function_class_strict_expansion_claimed", "new_CP_hypernetwork_adapter_or_ensemble_principle_claimed",
                "complete_equivalent_predecessor_established", "global_absence_certificate",
                "new_inference_router_or_auxiliary_diversity_objective", "graph_edge_computation_reduction_claimed",
                "current_compute_availability_used_to_reject", "adopted_method_driver_launch"]:
        assert composition[key] is False, key
    assert composition["measured_gain"] is None

    candidate = read_json("CANDIDATE_SPEC.json")
    assert candidate["member_count"] == 4 and candidate["CP_residual_rank"] == 1
    assert candidate["new_router_auxiliary_loss_or_projection"] is False
    init = candidate["initialization"]
    assert init["delta"] == 0.01 and init["c_numerators"] == [-3, -1, 1, 3]
    assert init["c_denominator"] == "sqrt(5)" and init["R_at_least"] == 2
    assert init["same_initial_global_BE_function"] is False
    assert init["all_elementary_product_Jacobians_nonzero"] is True
    assert init["loss_gradient_or_useful_diversity_guaranteed"] is False
    assert init["standard_BE_and_dropout_diversity_attributed_separately"] is True
    test = candidate["representative_test"]
    assert test["graphs"] == ["HGB-DBLP", "HGB-ACM"]
    assert test["complete_released_graphs"] is True and test["analyst_selected_subgraphs"] is False
    assert test["master_seeds"] == [131, 137, 139, 149, 151]
    assert len(test["arms"]) == len(set(test["arms"])) == 9
    assert "HGT_wider_global_message_BE" in test["arms"]
    assert "HGT_unrestricted_relation_output_BE" in test["arms"]
    assert test["downstream_configurations"] == 9 * 2 * 5 == 90
    assert test["original_transductive_split_percentages"] == {"TRAIN": 24, "VALIDATION": 6, "TEST": 70}
    assert test["unrestricted_factor_initialization"]
    assert test["wider_control_rule"]
    assert test["practical_gate"]["not_a_significance_or_power_claim"] is True
    assert candidate["adopted_method_driver_launch"] is False
    dataset = read_json("DATASET_AND_BASELINE_SCOPE.json")
    for key in ["author_repository_visited", "dataset_or_split_files_retrieved", "availability_independently_verified",
                "strong_baseline_execution_admitted"]:
        assert dataset[key] is False, key

    resource = read_json("RESOURCE_ESTIMATE.json")
    counts = resource["study_counts"]
    assert counts["downstream_configurations"] == counts["arms"] * counts["graphs"] * counts["master_seeds_per_graph"] == 90
    assert counts["native_single_model_jobs"] == counts["joint_BE_optimizer_jobs"] == counts["untied_member_optimizer_jobs"] == 40
    assert counts["total_downstream_optimizer_jobs"] == 40 + 40 + 40 == 120
    assert counts["joint_complete_member_trajectory_run_equivalents"] == 40 * 4 == 160
    assert counts["total_complete_encoder_trajectory_run_equivalents"] == 40 + 40 + 160 == 240
    assert resource["parameter_formulas_per_HGT_layer"]["M4_d64_global_BE_fast_parameters"] == 2 * 4 * 64 == 512
    assert 2 * 4 * 64 + 4 + 64 == 580
    assert resource["baseline_continuation_device_hours"] is None
    assert resource["scientific_merit_independent_of_current_availability"] is True
    for key in ["new_measurements", "homogeneous_graph_timings_transferred", "new_inference_router",
                "speed_claim", "current_GPU_busy_or_small_used_to_reject_science", "execution_or_reservation_requested"]:
        assert resource[key] is False, key

    witness = read_json("SYMBOLIC_WITNESS.json")
    rank = witness["local_rank"]
    a, b, c, q, W = (vec(rank[key]) for key in ["a", "b", "c", "q", "W"])
    u = Fraction(rank["u"])
    effective = [[a[m] * (b[m] + c[m] * q[r] * u) * W[r] for r in range(2)] for m in range(2)]
    assert effective == [vec(row) for row in rank["effective_map"]]
    assert effective[0][0] * effective[1][1] - effective[0][1] * effective[1][0] == Fraction(rank["determinant"]) == 1
    global_map = [[a[m] * b[m] * W[r] for r in range(2)] for m in range(2)]
    assert global_map[0][0] * global_map[1][1] - global_map[0][1] * global_map[1][0] == 0
    naive = witness["naive_separable"]
    x, a, b, alpha, beta = (vec(naive[key]) for key in ["x", "a", "b", "alpha", "beta"])
    W = [vec(row) for row in naive["W"]]
    direct = row_map([x[i] * a[i] * alpha[i] for i in range(2)], W)
    direct = [direct[j] * b[j] * beta[j] for j in range(2)]
    typed_W = [[alpha[i] * W[i][j] * beta[j] for j in range(2)] for i in range(2)]
    composed = row_map([x[i] * a[i] for i in range(2)], typed_W)
    composed = [composed[j] * b[j] for j in range(2)]
    assert direct == composed
    null = witness["initialization_nulls"]
    G = [vec(row) for row in null["G_common"]]
    identical_gradients = [G, G]
    u, c0, qn, q0, ca, allzero = (vec(null[key]) for key in ["u", "c_zero", "q_nonzero", "q_zero", "c_antithetic", "all_zero"])
    grad_c, grad_q, grad_u = cp_gradients(c0, qn, u, identical_gradients)
    assert grad_c == [-4, -4] and grad_q == [0, 0] and grad_u == [0, 0]
    assert cp_gradients(allzero, allzero, allzero, identical_gradients) == ([0, 0], [0, 0], [0, 0])
    assert sum(ca) == 0
    assert cp_gradients(ca, q0, u, identical_gradients) == ([0, 0], [0, 0], [0, 0])
    # Antithetic mean cancels a purely linear pooled common-response contribution.
    assert [sum(ca[m] * qn[r] * u[j] for m in range(2)) / 2 for r in range(2) for j in range(2)] == [0] * 4
    init_witness = witness["prospective_initialization"]
    numerators = vec(init_witness["gamma_numerators"])
    denominator_square = Fraction(init_witness["gamma_square_denominator"])
    delta = Fraction(init_witness["delta"])
    assert sum(numerators) == 0
    assert sum(v * v for v in numerators) / (4 * denominator_square) == Fraction(init_witness["mean_square_gamma"]) == 1
    assert max(v * v for v in numerators) * delta * delta / denominator_square == Fraction(init_witness["maximum_absolute_factor_delta_squared"])
    assert all(v != 0 for v in numerators) and delta != 0
    assert witness["graph_specific_benefit_claimed"] is False
    assert witness["source_candidate_or_trained_state_realized"] is False

    receipt = read_json("SCOPE_RECEIPT.json")
    assert receipt["read_accounting"] == accounting
    assert receipt["subagents_spawned"] == 0 and receipt["literature_task_stops_at_this_sealed_packet"] is True
    for key in ["active_source_inspection_or_edit", "active_protocol_index_ledger_status_manuscript_edit",
                "author_source_or_driver_access", "model_dataset_checkpoint_logits_native_results_or_logs_access",
                "ML_model_execution", "GPU_SSH_remote_execution", "new_method_driver_launch_adoption",
                "current_resource_availability_used_to_reject", "writes_outside_packet", "GENLINK_access_or_use",
                "publication_numeric_claim_transferred"]:
        assert receipt[key] is False, key
    assert receipt["publication_numeric_incidental_exposure"] is True
    assert seal["primary_method_scopes"] == 3
    assert seal["full_primary_reads"] == seal["author_code_reads"] == seal["retained_primary_rereads"] == 0
    return {"status": "PASS", "files_sealed": len(names), "external_input_bindings": len(inputs["inputs"]),
            "new_scoped_primary_methods": 3, "new_method_ids_absent_v24": 3,
            "full_primary_reads": 0, "author_code_reads": 0, "retained_primary_rereads": 0,
            "paragraph_heading_blocks": 150, "paragraph_blocks": 106, "heading_blocks": 44,
            "separately_inspected_math_nodes_or_fragments": 48, "complete_printed_algorithms": 1,
            "prospective_operations": 1, "prospective_arms": 9, "downstream_configurations": 90,
            "downstream_training_job_bookkeeping_count": 120, "device_hours": None,
            "scope": "Integrity, declared reading scope, symbolic local identities/nulls and bookkeeping only; no full-HGT expressive-power, scientific utility, source compatibility, author-code or novelty certificate."}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
