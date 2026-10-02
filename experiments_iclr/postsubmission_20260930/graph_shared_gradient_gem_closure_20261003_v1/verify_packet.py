#!/usr/bin/env python3
"""Read-only stdlib packet integrity, accounting and rational identities."""
import hashlib
import json
import math
import re
from fractions import Fraction
from pathlib import Path


PACKET = Path(__file__).resolve().parent
WORKSPACE = PACKET.parent.parent
PRIOR_MANIFEST_SHA = "53b430ead234682bb8dcfe1703d213f60bccc4422a42f99930c329f01f6574c8"
V23_SHA = "5f41b02d198cc29f6ca86ad1507dc4c7dedc66ba2b14543b29aec16c70a8bc48"


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


def dot(x, y):
    assert len(x) == len(y)
    return sum((a * b for a, b in zip(x, y)), Fraction(0))


def vector(x):
    return [Fraction(a) for a in x]


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
    actual = {
        str(p.relative_to(PACKET)) for p in PACKET.rglob("*")
        if p.is_file() and p.name not in {"MANIFEST.sha256", "SEAL.json"}
    }
    assert len(names) == len(set(names)) == seal["files_sealed"]
    assert set(names) == actual

    inputs = read_json("INPUT_BINDINGS.json")
    assert inputs["immutable_predecessor_manifest_sha256"] == PRIOR_MANIFEST_SHA
    assert inputs["consulted_index_sha256"] == V23_SHA
    for row in inputs["inputs"]:
        assert digest(workspace_file(row["path"])) == row["sha256"], row["path"]
    old_manifest = next(row for row in inputs["inputs"] if row["path"].endswith(
        "graph_shared_gradient_routing_20261003_v1/MANIFEST.sha256"))
    assert old_manifest["sha256"] == PRIOR_MANIFEST_SHA
    old_seal = next(row for row in inputs["inputs"] if row["path"].endswith(
        "graph_shared_gradient_routing_20261003_v1/SEAL.json"))
    assert json.loads(workspace_file(old_seal["path"]).read_text())["manifest_sha256"] == PRIOR_MANIFEST_SHA
    index_row = next(row for row in inputs["inputs"] if row["path"].endswith(
        "index_v23/LITERATURE_INDEX.json"))
    assert index_row["sha256"] == V23_SHA
    index = json.loads(workspace_file(index_row["path"]).read_text())
    assert len(index["paper_records"]) == 106
    for sid in ["1706.08840", "1812.00420"]:
        assert not any(sid in row["canonical_id"] for row in index["paper_records"])

    scopes = read_json("READ_SCOPES.json")
    accounting = scopes["accounting"]
    expected = {
        "primary_method_scopes_this_packet": 2,
        "first_scoped_method_identity_with_no_retained_primary_citation": 2,
        "retained_abstract_only_citation_method_scope_upgrades": 0,
        "new_method_record_ids_absent_from_v23": 2,
        "new_full_primary_reads": 0,
        "retained_scoped_method_rereads": 0,
        "saved_SAM_MAML_method_rereads": 0,
        "saved_PCGrad_OGD_method_rereads": 0,
        "author_source_reads": 0,
        "primary_unique_ids_retrieved": 2,
        "abs_html_retrieval_rows": 4,
        "new_method_driver_or_launch_adoptions": 0,
        "paragraph_heading_blocks": 52,
        "paragraph_blocks": 46,
        "heading_blocks": 6,
        "complete_printed_algorithm_figures": 2,
        "display_math_nodes": 14,
        "supplementary_math_nodes": 56,
        "display_supplementary_overlaps": 6,
        "distinct_math_nodes_or_fragments": 64,
        "scoped_optimization_proof_or_argument_text_read": True,
        "full_paper_or_complete_proof_audit": False,
    }
    for key, value in expected.items():
        assert accounting[key] == value, key
    assert inputs["read_accounting"] == accounting
    assert len(scopes["methods"]) == 2
    ids = [row["canonical_id"] for row in scopes["methods"]]
    assert ids == ["arXiv:1706.08840v1", "arXiv:1812.00420v1"]
    assert scopes["preview_disclosures"]
    counts = {key: 0 for key in scopes["methods"][0]["counts"]}
    for row in scopes["methods"]:
        sid = row["canonical_id"].split(":", 1)[1]
        bindings = row["file_bindings"]
        assert len(bindings) == 11
        assert len({r["path"] for r in bindings}) == 11
        for binding in bindings:
            assert digest(workspace_file(binding["path"])) == binding["sha256"]
        blocks = read_json(f"primary/{sid}_html_blocks.json")
        display_math = read_json(f"primary/{sid}_html_display_math.json")
        all_math = read_json(f"primary/{sid}_html_all_math.json")
        selected = []
        for lo, hi in row["paragraph_blocks_inclusive"]:
            assert 0 <= lo <= hi < len(blocks)
            selected.extend(blocks[i] for i in range(lo, hi + 1))
        p_count = sum(block["tag"] == "p" for block in selected)
        h_count = sum(re.fullmatch(r"h[1-6]", block["tag"]) is not None for block in selected)
        assert len(selected) == p_count + h_count
        assert len(selected) == row["counts"]["paragraph_heading_blocks"]
        assert p_count == row["counts"]["paragraph_blocks"]
        assert h_count == row["counts"]["heading_blocks"]
        display_indices = row["display_math_indices_used"]
        assert display_indices == row["display_math_indices_previewed"]
        assert len(set(display_indices)) == row["counts"]["display_math_nodes"]
        display_all = row["display_to_all_math_indices"]
        assert len(display_indices) == len(display_all)
        for di, ai in zip(display_indices, display_all):
            assert display_math[di]["id"] == all_math[ai]["id"]
            assert display_math[di]["alttext"] == all_math[ai]["alttext"]
        supplemental = set()
        for lo, hi in row["supplementary_all_math_indices_inclusive"]:
            assert 0 <= lo <= hi < len(all_math)
            supplemental.update(range(lo, hi + 1))
        assert len(supplemental) == row["counts"]["supplementary_math_nodes"]
        assert len(supplemental & set(display_all)) == row["counts"]["display_supplementary_overlaps"]
        assert len(supplemental | set(display_all)) == row["counts"]["distinct_math_nodes_or_fragments"]
        algorithms = read_json(f"primary/{sid}_html_algorithms.json")
        chosen = [alg for alg in algorithms if alg["attrs"].get("id") == row["complete_algorithm_id"]]
        assert len(chosen) == 1
        assert len(chosen[0]["text"]) > 300
        assert len(chosen[0]["text"]) < 6000, "Overbroad algorithm extraction"
        assert row["counts"]["complete_printed_algorithm_figures"] == 1
        for key in counts:
            counts[key] += row["counts"][key]
    for key, value in counts.items():
        assert accounting[key] == value, key
    assert accounting["distinct_math_nodes_or_fragments"] == (
        accounting["display_math_nodes"] + accounting["supplementary_math_nodes"]
        - accounting["display_supplementary_overlaps"])

    retrieval = read_json("PRIMARY_RETRIEVAL.json")
    assert len(retrieval) == 4
    assert {row["canonical_id"] for row in retrieval} == set(ids)
    assert {(row["canonical_id"], row["kind"]) for row in retrieval} == {
        (sid, kind) for sid in ids for kind in ["abs", "html"]}
    for row in retrieval:
        assert row["status"] == 200 and row["url"].startswith("https://arxiv.org/")
        assert digest(workspace_file(row["path"])) == row["sha256"]
        for extract in row["mechanical_extracts"].values():
            path = workspace_file(extract["path"])
            assert digest(path) == extract["sha256"]
            assert len(json.loads(path.read_text())) == extract["items"]

    conclusions = read_json("PAPER_CONCLUSIONS.json")
    assert conclusions["read_accounting"] == accounting
    assert [row["canonical_id"] for row in conclusions["primary_methods"]] == ids
    assert all(row["source_benefits_transferred"] is False for row in conclusions["primary_methods"])
    composition = conclusions["composition"]
    assert composition["closest_consulted_QP_predecessor"] == "GEM simultaneous inequality Euclidean projection"
    assert composition["prospective_total_arms"] == 7
    for key in ["new_heads_or_inference_router", "new_projection_or_negative_correlation_principle_claimed",
                "complete_equivalent_predecessor_established", "global_absence_certificate",
                "current_compute_availability_used_to_reject", "adopted_method_driver_launch"]:
        assert composition[key] is False, key
    assert composition["measured_gain"] is None

    candidate = read_json("CANDIDATE_SPEC.json")
    assert candidate["immutable_shared_gradient_v1"]["manifest_sha256"] == PRIOR_MANIFEST_SHA
    assert candidate["immutable_shared_gradient_v1"]["modified"] is False
    assert candidate["prospective_additional_comparators"] == 1
    for key in ["new_direction_or_framework", "new_heads_or_inference_capacity",
                "source_qualification_or_execution", "adopted_method_driver_launch"]:
        assert candidate[key] is False, key
    control = read_json("PROSPECTIVE_MEMBER_CE_CONTROL.json")
    assert control["member_count"] == 4
    assert control["window_native_steps"] == candidate["window_native_steps"] == 32
    assert control["correction_norm_cap_relative_native_displacement"] == 1.0
    assert control["projection"]["fixed_active_set_order_maximum"] == 16
    assert control["projection"]["pooled_progress_equality"] == "a^T c=0"
    assert control["projection"]["simultaneous_member_constraints"] == "k_m^T(d0+c)<=0 for every member"
    assert control["projection"]["equivalent_H_columns"] == "-k_m"
    assert len(control["guard_differences"]["same"]) == 3
    assert control["guard_differences"]["different"] and control["guard_differences"]["units"]
    assert control["nonidentical_eligibility"]
    assert len(control["remaining_controls"]) == 6
    assert control["adopted_method_driver_launch"] is False
    test = control["paired_test"]
    assert candidate["prospective_paired_test"] == test
    assert test["graphs"] == ["Squirrel", "Photo"] and test["complete_graphs"] is True
    assert test["seeds"] == [101, 103, 107]
    assert test["arms_if_adopted"] == 7 and test["fits"] == 42
    gate = control["practical_gate_if_adopted"]
    assert gate["mean_NLL_gain_over_every_other_arm_at_least_nats"] == 0.01
    assert gate["mean_accuracy_decline_at_most_percentage_points"] == 0.5
    assert gate["gain_sign_consistent_all_three_seeds"] is True
    assert gate["not_a_power_or_significance_claim"] is True

    resource = read_json("RESOURCE_ESTIMATE.json")
    receipt_path = workspace_file(resource["permitted_receipt"]["path"])
    assert digest(receipt_path) == resource["permitted_receipt"]["sha256"]
    receipt = json.loads(receipt_path.read_text())
    summaries = resource["inherited_native_continuation_observations"]
    assert summaries == receipt["summaries"]
    per_arm_hours = 3 * sum(summaries[g]["mean_seconds"] for g in ["Squirrel", "Photo"]) / 3600
    forecast = resource["baseline_continuation_forecast"]
    assert forecast["arms"] == 7 and forecast["fits"] == 42
    assert math.isclose(forecast["device_hours"], 7 * per_arm_hours, abs_tol=1e-10)
    assert math.isclose(forecast["six_arm_baseline_hours"], 6 * per_arm_hours, abs_tol=1e-10)
    assert math.isclose(forecast["seventh_arm_increment_hours"], per_arm_hours, abs_tol=1e-10)
    assert math.isclose(forecast["four_arm_receipt_baseline_hours"], 4 * per_arm_hours, abs_tol=1e-10)
    assert forecast["reported_hours_rounded"] == 31.52
    added = resource["prospective_member_CE_work"]
    assert added["attempts_maximum"] == 2 * 3 * 32 == 192
    assert added["member_forward_route_equivalents_maximum"] == 192 * 12 == 2304
    assert added["member_backward_route_equivalents_maximum"] == 192 * 20 == 3840
    work = resource["all_projected_arms_work"]
    assert work["arms"] == 4 and work["attempts_maximum"] == 4 * 192 == 768
    assert work["member_forward_route_equivalents_maximum"] == 768 * 12 == 9216
    assert work["member_backward_route_equivalents_maximum"] == 768 * 20 == 15360
    extra = resource["known_control_extra_work"]
    assert extra["aggregate_extra_member_backward_route_equivalents_maximum"] == (
        15360 + extra["PCGrad_extra_member_backward_route_equivalents_maximum"]
        + extra["shared_pooled_extra_member_backward_route_equivalents_maximum"]) == 18432
    assert resource["memory_formula"]["extra_shared_float32_vectors_at_most"] == 7
    assert resource["memory_formula"]["extra_shared_vector_bytes"] == "28 * P_shared"
    assert resource["scientific_merit_independent_of_current_availability"] is True
    for key in ["new_inference_capacity", "current_GPU_busy_or_small_used_to_reject_science",
                "execution_or_reservation_requested", "new_resource_measurement"]:
        assert resource[key] is False, key

    witness = read_json("SYMBOLIC_WITNESS.json")
    mapping = witness["GEM_mapping"]
    z = {key: vector(mapping[key]) for key in ["h", "q", "d0", "d", "g", "v", "c", "a"]}
    assert z["q"] == [-x for x in z["h"]]
    assert z["g"] == [-x for x in z["d0"]]
    assert z["v"] == [-x for x in z["d"]]
    assert z["d"] == [x + y for x, y in zip(z["d0"], z["c"])]
    assert dot(z["h"], z["d"]) == dot(z["q"], z["v"]) == 0
    dv = [x - y for x, y in zip(z["v"], z["g"])]
    assert dot(dv, dv) == dot(z["c"], z["c"]) == 1
    assert dot(z["a"], z["c"]) == 0
    assert dot(z["a"], z["d"]) == dot(z["a"], z["d0"]) == -1
    projected_h = [h - dot(z["a"], z["h"]) * a / dot(z["a"], z["a"])
                   for h, a in zip(z["h"], z["a"])]
    lam = Fraction(mapping["lambda"])
    assert z["c"] == [lam * x for x in projected_h]
    b = -dot(z["h"], z["d0"])
    gram = dot(z["h"], projected_h)
    assert b == gram == lam == 1
    assert lam * b - lam * lam * gram / 2 == dot(z["c"], z["c"]) / 2
    assert dot(z["c"], z["c"]) <= dot(z["d0"], z["d0"])
    cancel = witness["A_GEM_cancellation"]
    h1, h2, d0, alpha = (vector(cancel[key]) for key in ["h1", "h2", "d0", "alpha"])
    average = [alpha[0] * x + alpha[1] * y for x, y in zip(h1, h2)]
    assert average == [0, 0] and dot(average, d0) == 0
    assert dot(h1, d0) == -1 and dot(h2, d0) == 1
    impossible = witness["pooled_equality_infeasibility"]
    a, h, d0 = (vector(impossible[key]) for key in ["a", "h", "d0"])
    assert a == h and dot(a, d0) == -1
    sign = {key: Fraction(value) for key, value in witness["A_GEM_v1_GEM_dual_sign"].items()
            if key != "claim"}
    assert sign["printed_dual_quadratic_coefficient"] == Fraction(1, 2)
    assert sign["printed_dual_linear_coefficient"] > 0
    assert sign["nonnegative_minimizer"] == 0
    assert (sign["negative_gradient_row"] * sign["nonnegative_minimizer"] + sign["g"]
            == sign["printed_restoration"] == -1)
    assert sign["prior_gradient"] * sign["printed_restoration"] == sign["primal_inner_product"] < 0
    assert witness["graph_specific_benefit_claimed"] is False
    assert witness["source_candidate_or_trained_state_realized"] is False

    scope = read_json("SCOPE_RECEIPT.json")
    assert scope["read_accounting"] == accounting and scope["subagents_spawned"] == 0
    assert scope["prospective_comparator_only"] is True
    for key in ["active_source_inspection", "active_source_protocol_index_root_ledger_status_manuscript_changes",
                "model_dataset_checkpoint_logits_or_logs_access", "ML_model_execution", "GPU_or_remote_execution",
                "adopted_method_driver_launch", "current_compute_constraints_decide_scientific_merit",
                "writes_outside_packet", "predecessor_changed"]:
        assert scope[key] is False, key
    assert seal["primary_method_scopes"] == 2 and seal["full_primary_reads"] == 0
    assert seal["author_code_reads"] == seal["retained_method_rereads"] == 0
    assert seal["immutable_predecessor_manifest_sha256"] == PRIOR_MANIFEST_SHA
    return {
        "status": "PASS", "files_sealed": len(names),
        "external_input_bindings": len(inputs["inputs"]),
        "primary_method_scopes": 2, "new_method_ids_absent_v23": 2,
        "full_primary_reads": 0, "author_code_reads": 0, "retained_method_rereads": 0,
        "paragraph_heading_blocks": 52, "paragraph_blocks": 46, "heading_blocks": 6,
        "complete_printed_algorithms": 2, "distinct_math_nodes_or_fragments": 64,
        "prospective_total_arms": 7, "prospective_fits": 42,
        "baseline_continuation_forecast_device_hours": forecast["device_hours"],
        "added_comparator_attempts_maximum": 192,
        "immutable_predecessor_manifest_sha256": PRIOR_MANIFEST_SHA,
        "scope": "Integrity, declared accounting, forecast arithmetic and rational identities only; no source qualification, utility, author-code reproduction or novelty certified.",
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
