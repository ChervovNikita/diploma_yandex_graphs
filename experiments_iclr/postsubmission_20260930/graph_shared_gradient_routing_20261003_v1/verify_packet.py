#!/usr/bin/env python3
"""Read-only stdlib integrity/accounting checks; no ML/data execution."""
import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path


PACKET = Path(__file__).resolve().parent
WORKSPACE = PACKET.parent.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(name):
    return json.loads((PACKET / name).read_text())


def workspace_file(name):
    path = WORKSPACE / name
    assert path.is_file() and not path.is_symlink(), name
    assert path.resolve().is_relative_to(WORKSPACE), name
    return path


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
        assert digest(path) == expected, name
        names.append(name)
    actual = {
        str(p.relative_to(PACKET)) for p in PACKET.rglob("*")
        if p.is_file() and p.name not in {"MANIFEST.sha256", "SEAL.json"}
    }
    assert len(names) == len(set(names)) == seal["files_sealed"]
    assert set(names) == actual

    inputs = read_json("INPUT_BINDINGS.json")
    for row in inputs["inputs"]:
        assert digest(workspace_file(row["path"])) == row["sha256"], row["path"]
    scopes = read_json("READ_SCOPES.json")
    accounting = scopes["accounting"]
    expected = {
        "primary_method_scopes_this_packet": 2,
        "first_scoped_method_identity_with_no_retained_primary_citation": 1,
        "retained_abstract_only_citation_method_scope_upgrades": 1,
        "new_method_record_ids_absent_from_v21": 2,
        "new_full_primary_reads": 0,
        "retained_scoped_method_rereads": 0,
        "saved_SAM_MAML_method_rereads": 0,
        "author_source_reads": 0,
        "primary_unique_ids_retrieved": 2,
        "abs_html_retrieval_rows": 4,
        "new_method_driver_or_launch_adoptions": 0,
    }
    for key, value in expected.items():
        assert accounting[key] == value, key
    assert len(scopes["methods"]) == 2
    ids = [row["canonical_id"] for row in scopes["methods"]]
    assert ids == ["arXiv:2001.06782v1", "arXiv:1910.07104v1"]
    assert scopes["preview_disclosures"]
    for row in scopes["methods"]:
        for binding in row["file_bindings"]:
            assert digest(workspace_file(binding["path"])) == binding["sha256"]
        sid = row["canonical_id"].split(":", 1)[1]
        blocks = read_json(f"primary/{sid}_html_blocks.json")
        math = read_json(f"primary/{sid}_html_display_math.json")
        for lo, hi in row["paragraph_blocks_inclusive"]:
            assert 0 <= lo <= hi < len(blocks)
            assert blocks[lo]["index"] == lo and blocks[hi]["index"] == hi
        assert all(0 <= i < len(math) for i in row["display_math_indices_previewed"])
        assert set(row["display_math_indices_used"]).issubset(row["display_math_indices_previewed"])
        algorithm = read_json(f"primary/{sid}_algorithm_extract.json")
        assert len(algorithm) == 1 and algorithm[0]["attrs"]["id"] == "alg1"
        assert "Algorithm 1" in algorithm[0]["text"]
        assert len(algorithm[0]["text"]) < 3000, "Overbroad algorithm extraction"

    retrieval = read_json("PRIMARY_RETRIEVAL.json")
    assert len(retrieval) == 4
    assert {row["canonical_id"] for row in retrieval} == set(ids)
    for row in retrieval:
        assert row["status"] == 200
        for name_key, hash_key in [
            ("path", "sha256"), ("blocks_path", "blocks_sha256"),
            ("math_path", "math_sha256"),
        ]:
            assert digest(workspace_file(row[name_key])) == row[hash_key]
    index_row = next(row for row in inputs["inputs"] if row["path"].endswith(
        "/index_v21/LITERATURE_INDEX.json"
    ))
    index = json.loads(workspace_file(index_row["path"]).read_text())
    for sid in ["2001.06782", "1910.07104"]:
        assert not any(sid in r["canonical_id"] for r in index["paper_records"])
    old_pcgrad = next(row for row in inputs["inputs"] if row["path"].endswith(
        "/sources/PCGRAD_RETRIEVAL.json"
    ))
    old_receipt = json.loads(workspace_file(old_pcgrad["path"]).read_text())
    assert "/abs/2001.06782" in old_receipt["url"]

    conclusions = read_json("PAPER_CONCLUSIONS.json")
    assert conclusions["read_accounting"] == accounting
    composition = conclusions["composition"]
    for key in ["new_heads_or_inference_router", "new_projection_or_negative_correlation_principle_claimed", "complete_equivalent_predecessor_established", "global_absence_certificate", "current_compute_availability_used_to_reject", "adopted_method_driver_launch"]:
        assert composition[key] is False, key
    assert composition["measured_gain"] is None

    candidate = read_json("CANDIDATE_SPEC.json")
    assert candidate["window_native_steps"] == 32
    assert candidate["correction_norm_cap_relative_native_displacement"] == 1.0
    test = candidate["representative_paired_test"]
    assert test["graphs"] == ["Squirrel", "Photo"] and test["complete_graphs"] is True
    assert test["seeds"] == [101, 103, 107]
    assert len(test["arms"]) == 6
    assert test["confirmation"]["fresh_seeds"] == [109, 113, 127]
    assert test["confirmation"]["papers100M_required"] is False

    resource = read_json("RESOURCE_ESTIMATE.json")
    summaries = resource["inherited_native_continuation_observations"]
    forecast = resource["baseline_continuation_forecast"]
    hours = 6 * 3 * sum(summaries[g]["mean_seconds"] for g in ["Squirrel", "Photo"]) / 3600
    assert abs(forecast["device_hours"] - hours) < 1e-10
    assert forecast["fits"] == 36
    work = resource["projected_correction_work"]
    assert work["attempts_maximum"] == 3 * 2 * 3 * 32 == 576
    assert work["member_forward_route_equivalents_maximum"] == 576 * 12
    assert work["member_backward_route_equivalents_maximum"] == 576 * 20
    assert resource["scientific_merit_independent_of_current_availability"] is True
    assert resource["execution_or_reservation_requested"] is False

    witness = read_json("SYMBOLIC_WITNESS.json")
    vectors = {key: [Fraction(x) for x in value] for key, value in witness["vectors"].items()}
    def dot(x, y):
        return sum(a * b for a, b in zip(x, y))
    assert dot(vectors["g1"], vectors["g2"]) == 1
    assert dot(vectors["h"], vectors["d0"]) == -1
    assert dot(vectors["a"], vectors["c"]) == 0
    assert dot(vectors["h"], vectors["d_star"]) == 0
    assert dot(vectors["a"], vectors["d_star"]) == dot(vectors["a"], vectors["d0"]) == Fraction(-5, 4)
    assert [x + y for x, y in zip(vectors["d0"], vectors["c"])] == vectors["d_star"]
    assert dot(vectors["c"], vectors["c"]) / dot(vectors["d0"], vectors["d0"]) == Fraction(1, 4)
    assert witness["graph_specific_benefit_claimed"] is False
    assert witness["source_candidate_or_trained_state_realized"] is False

    scope = read_json("SCOPE_RECEIPT.json")
    for key in ["active_source_inspection", "model_dataset_checkpoint_logits_or_logs_access", "ML_model_execution", "GPU_or_remote_execution", "adopted_method_driver_launch", "current_compute_constraints_decide_scientific_merit", "writes_outside_packet"]:
        assert scope[key] is False, key
    assert scope["subagents_spawned"] == 0
    return {
        "status": "PASS", "files_sealed": len(names),
        "external_input_bindings": len(inputs["inputs"]),
        "primary_method_scopes": 2, "retained_abstract_method_upgrades": 1,
        "first_scoped_identity": 1, "full_primary_reads": 0,
        "baseline_continuation_forecast_device_hours": hours,
        "scope": "Integrity, declared accounting and rational vector identities only; no source/AD qualification, scientific utility or novelty certified.",
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
