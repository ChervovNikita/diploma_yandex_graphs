#!/usr/bin/env python3
"""Read-only stdlib integrity/accounting checks; never imports ML or accesses data."""
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


def bound_workspace_path(value):
    path = WORKSPACE / value
    assert not path.is_symlink(), f"Symlink input: {value}"
    assert path.resolve().is_relative_to(WORKSPACE), f"Outside workspace: {value}"
    assert path.is_file(), f"Missing input: {value}"
    return path


def verify():
    manifest = PACKET / "MANIFEST.sha256"
    seal = read_json("SEAL.json")
    assert digest(manifest) == seal["manifest_sha256"], "Manifest seal mismatch"
    entries = []
    for line in manifest.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        assert match, "Malformed manifest line"
        expected, name = match.groups()
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        path = PACKET / relative
        assert path.is_file() and not path.is_symlink(), f"Invalid file: {name}"
        assert digest(path) == expected, f"Hash mismatch: {name}"
        entries.append(name)
    assert len(entries) == len(set(entries)) == seal["files_sealed"]
    actual = {
        str(p.relative_to(PACKET))
        for p in PACKET.rglob("*")
        if p.is_file() and p.name not in {"MANIFEST.sha256", "SEAL.json"}
    }
    assert set(entries) == actual, "Manifest coverage mismatch"

    bindings = read_json("INPUT_BINDINGS.json")
    for row in bindings["inputs"]:
        assert digest(bound_workspace_path(row["path"])) == row["sha256"], row["path"]

    scopes = read_json("READ_SCOPES.json")
    accounting = scopes["accounting"]
    expected_counts = {
        "new_scoped_primary_methods": 2,
        "new_full_primary_reads": 0,
        "new_primary_source_identities_retrieved": 3,
        "primary_abs_html_retrieval_rows": 6,
        "retained_raw_primary_method_rereads": 0,
        "author_source_reads": 0,
        "new_adopted_methods": 0,
        "new_adopted_drivers": 0,
        "new_launches": 0,
    }
    for key, value in expected_counts.items():
        assert accounting[key] == value, key
    assert accounting["model_dataset_checkpoint_or_logit_access"] is False
    assert accounting["remote_or_GPU_execution"] is False
    assert accounting["saved_retained_excerpt_preview_disclosure"]
    methods = scopes["new_scoped_method_reads"]
    assert [r["canonical_id"] for r in methods] == [
        "arXiv:2010.01412v1", "arXiv:1703.03400v1"
    ]
    assert scopes["retrieved_but_method_unread"]["method_blocks_read"] == []
    assert scopes["retrieved_but_method_unread"]["supplies_substantive_claim"] is False
    for row in methods:
        assert row["read_status"] == "new_scoped_primary_method_read_not_full"
        for binding in row["file_bindings"]:
            assert digest(bound_workspace_path(binding["path"])) == binding["sha256"]
        sid = row["canonical_id"].split(":", 1)[1]
        blocks = read_json(f"primary/{sid}_html_blocks.json")
        maths = read_json(f"primary/{sid}_html_display_math.json")
        for lo, hi in row["html_paragraph_blocks_inclusive"]:
            assert 0 <= lo <= hi < len(blocks)
            assert blocks[lo]["index"] == lo and blocks[hi]["index"] == hi
        assert all(0 <= i < len(maths) for i in row["display_math_indices_previewed"])
        assert set(row["display_math_indices_used_for_method"]).issubset(
            row["display_math_indices_previewed"]
        )

    retrievals = read_json("PRIMARY_RETRIEVAL.json")
    assert len(retrievals) == 6
    assert len({r["id"] for r in retrievals}) == 3
    for row in retrievals:
        assert row["status"] == 200
        for path_key, hash_key in [
            ("path", "sha256"),
            ("blocks_path", "blocks_sha256"),
            ("display_math_path", "display_math_sha256"),
        ]:
            assert digest(bound_workspace_path(row[path_key])) == row[hash_key]

    index_row = next(r for r in bindings["inputs"] if r["path"].endswith(
        "/literature_memory/index_v20/LITERATURE_INDEX.json"
    ))
    index = json.loads(bound_workspace_path(index_row["path"]).read_text())
    retained_ids = [r["canonical_id"].lower() for r in index["paper_records"]]
    for sid in ["2010.01412", "1703.03400", "2011.09468"]:
        assert not any(sid in retained for retained in retained_ids), sid

    conclusions = read_json("PAPER_CONCLUSIONS.json")
    assert conclusions["read_accounting"] == accounting
    composition = conclusions["composition_assessment"]
    assert composition["complete_equivalent_predecessor_established"] is False
    assert composition["absence_certificate_claimed"] is False
    assert composition["measured_predictive_gain"] is None
    assert composition["adoption_or_execution"] is False

    candidate = read_json("CANDIDATE_SPEC.json")
    screen = candidate["representative_falsifier"]
    assert screen["complete_development_graphs"] == ["Squirrel", "Photo"]
    assert screen["paired_fresh_seeds"] == [17, 29, 43]
    assert len(screen["arms"]) == 5
    assert screen["confirmation"]["fresh_seeds"] == [47, 59, 71]
    assert screen["confirmation"]["papers100M_required"] is False

    resource = read_json("RESOURCE_ESTIMATE.json")
    summary = resource["continuation_observations"]
    hours = 5 * 3 * sum(summary[g]["mean_seconds"] for g in ["Squirrel", "Photo"]) / 3600
    assert abs(hours - resource["development_continuation_forecast"]["device_hours"]) < 1e-10
    assert resource["development_continuation_forecast"]["fits"] == 30
    maps = resource["selector_maps_per_graph_seed"]
    assert maps["selected_spans"] * maps["fixed_candidates_per_span"] + maps["common_trials"] == maps["total"] == 10
    assert resource["selector_maps_for_development"]["trial_maps"] == 60
    assert resource["scientific_merit_separate_from_current_availability"] is True

    witness = read_json("SYMBOLIC_WITNESS.json")
    p = Fraction(witness["p"])
    coefficient = Fraction(1, 2) * p * (1 - p) * (1 - 2 * p)
    assert coefficient == Fraction(-3, 64)
    assert Fraction(witness["softmax_or_sigmoid_mean_probability_coefficient_per_variance"]) == coefficient
    assert Fraction(witness["extra_mean_logit_update_coefficient_per_eta_variance"]) == -coefficient
    assert Fraction(witness["small_step_loss_coefficients_per_eta_variance"]["y=1"]) == -coefficient * (p - 1) == Fraction(-3, 256)
    assert Fraction(witness["small_step_loss_coefficients_per_eta_variance"]["y=0"]) == -coefficient * p == Fraction(9, 256)
    assert witness["candidate_realized"] is False
    assert witness["projection_or_graph_constraints_realized"] is False
    scope = read_json("SCOPE_RECEIPT.json")
    for key in ["existing_source_inspected", "model_dataset_checkpoint_logit_access", "remote_GPU_or_model_execution", "writes_outside_packet"]:
        assert scope[key] is False, key
    assert scope["subagents_spawned"] == 0
    limits = read_json("FAILURES_AND_UNCERTAINTIES.json")
    assert limits["negative_or_failure_model_results_observed"] is False
    return {
        "status": "PASS",
        "files_sealed": len(entries),
        "external_input_bindings": len(bindings["inputs"]),
        "new_scoped_primary_methods": 2,
        "new_full_primary_reads": 0,
        "retrieved_method_unread_sources": 1,
        "continuation_forecast_device_hours": hours,
        "scope": "Integrity, declared accounting and symbolic scalar arithmetic only; no scientific correctness, novelty, empirical benefit or source qualification certified.",
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
