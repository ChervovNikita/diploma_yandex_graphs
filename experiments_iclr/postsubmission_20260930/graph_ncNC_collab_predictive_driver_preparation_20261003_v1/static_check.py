"""Stdlib-only source and fabricated family-metadata verification."""
from pathlib import Path
from hashlib import sha256
import ast
import importlib.util
import json
import sys
import tempfile
from pilot_common import (HERE, PLAN_SHA, MODEL_SHA, RESOURCE_SHA, DATA_AUTHORITY_SHA,
                          DATA_FILES, UNITS, verify_manifest, file_sha, atomic_json, require,
                          fresh_output, admit_invocation, utc)
from pilot_lock import run as family_lock, commit_closure_markers, EXPECTED_ARMS


class QuietAttempts:
    def phase(self, *args, **kwargs):
        pass


def metadata_fixtures():
    """Only JSON and literal text bytes; no tensors/pickle/model/data are read."""
    with tempfile.TemporaryDirectory(prefix="ncnc_source_metadata_") as temporary:
        root = Path(temporary)
        identity = {"family_id": "synthetic_source_fixture", "synthetic_only": True}
        inputs = []
        def pin(path):
            return {"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha(path)}
        for unit in UNITS:
            for seed in range(5):
                out = root / (unit + str(seed)); out.mkdir()
                selected, selection_artifacts = {}, {}
                for arm in EXPECTED_ARMS[unit]:
                    checkpoint = out / ("SELECTED_" + arm + ".pt")
                    checkpoint.write_text("fabricated literal metadata-check fixture; never a Torch object\n")
                    selected[arm] = pin(checkpoint)
                    selection_file = out / ("PRIVATE_SELECTION_" + arm + ".json")
                    atomic_json(selection_file, {"schema": "ncnc-pilot-validation-selected-checkpoint-v1", "identity": identity,
                        "arm": arm, "seed": seed, "selection": {"candidate_id": "epoch_1", "order": 1, "hits50": (seed + 1) / 10},
                        "checkpoint": selected[arm]})
                    selection_artifacts[arm] = pin(selection_file)
                count = 4 if unit == "native_bank4" else 1
                members = []
                for member in range(count):
                    members.append({"member": member, "fit_seed": seed + 5 * member if count == 4 else seed,
                        "initial_state_sha256": str(seed), "initial_rng_sha256": str(seed),
                        "epochs": [{"epoch": epoch, "start_rng_sha256": "start", "end_rng_sha256": "end",
                            "full_batches": 17, "optimizer_steps": 17, "positive_queries": 60084, "negative_queries": 100000,
                            "train_wall_seconds": 0., "valid_wall_seconds": 0.,
                            "stream": {"full_batches": 17, "supervised_records": 1114112, "dropped_tail_records": 64940,
                                "negative_rows_drawn": 2358104, "negative_draw_sha256": "fabricated", "permutation_sha256": "fabricated"}}
                            for epoch in range(1, 101)]})
                complete = {"schema": "ncnc-pilot-complete-unit-v1", "identity": identity, "unit": unit, "base_seed": seed,
                    "epochs": 100, "unique_fits": count, "completed_optimizer_steps": count * 1700,
                    "scientific_initialization": "fresh", "resource_state_donor": False, "test_file_opened": False,
                    "selected_checkpoints": selected, "selection_artifacts": selection_artifacts, "member_metadata": members,
                    "independent_candidate_count": 101 if count == 4 else 0, "candidate101_extra_VALID_evaluations": 4 if count == 4 else 0}
                atomic_json(out / "COMPLETE.json", complete)
                atomic_json(out / "ATTEMPTS.json", {"identity": identity, "attempts": [{"status": "COMPLETE", "inclusive_wall_seconds": 0.}]})
                inputs.append({"unit": unit, "base_seed": seed, "output_directory": str(out), "disposition": "COMPLETE", "complete_sha256": file_sha(out / "COMPLETE.json")})
        context = {"identity": identity, "release": {"family_inputs": inputs}}
        complete_lock = family_lock(context, root / "full_lock", QuietAttempts())
        require(complete_lock["status"] == "COMPLETE_FAMILY_LOCKED" and complete_lock["unique_fits_completed"] == 35 and complete_lock["complete_served_cells"] == 25, "Complete fabricated family did not close")
        target = inputs[7]
        failed_root = Path(target["output_directory"])
        ledger = {"identity": identity, "attempts": [{"status": "FAILED", "inclusive_wall_seconds": 3.}]}
        atomic_json(failed_root / "ATTEMPTS.json", ledger)
        atomic_json(failed_root / "FAILED.json", {"schema": "ncnc-pilot-failed-unit-v1", "identity": identity,
            "stage": "fit", "unit": target["unit"], "base_seed": target["base_seed"], "status": "FAILED", "test_file_opened": False,
            "failure": {"condition": "fabricated allocation failure", "exception_type": "RuntimeError"},
            "attempts_receipt": pin(failed_root / "ATTEMPTS.json"), "journal_receipt": None})
        target.update(disposition="TERMINAL_FAILED", failure_sha256=file_sha(failed_root / "FAILED.json"),
                      terminal_failure_immutable=True, terminal_failure_authorization_reference="synthetic-root-fixture")
        # Corrupt selection JSON in the terminal failed unit: it must be skipped.
        for arm in EXPECTED_ARMS[target["unit"]]:
            (failed_root / ("PRIVATE_SELECTION_" + arm + ".json")).write_text("not JSON; must not be disclosed")
        failed_lock = family_lock(context, root / "failed_lock", QuietAttempts())
        require(failed_lock["status"] == "TERMINAL_FAILED_FAMILY_LOCKED" and failed_lock["complete_served_cells"] == 24 and
                failed_lock["arm_summaries"] is None and failed_lock["primary_development_pilot"]["mean_difference"] is None and
                failed_lock["TEST_execution_authorized"] is False and len(failed_lock["terminal_failures"]) == 1,
                "Terminal failed closure disclosed a subset summary or lost missing cells")
        lock_output = root / "failed_lock"; lock_output.mkdir()
        atomic_json(lock_output / "FAMILY_LOCK.json", failed_lock)
        commit_closure_markers(context, lock_output, failed_lock)
        atomic_json(failed_root / "JOURNAL.json", {"synthetic_only": True})
        try:
            fresh_output(failed_root, resume=True)
        except RuntimeError as error:
            require("immutable" in str(error), "Wrong immutable-resume failure")
        else:
            raise AssertionError("Terminal family-closed unit accepted resume")
        require(len(list(root.glob("*/FAMILY_CLOSURE.json"))) == 20, "Family closure did not retire all20 units")
    return {"complete35_fit25_cell_fixture": True, "complete_or_terminal_failed25_cell_fixture": True,
            "missing_cells_and_failure_costs_preserved": True, "no_partial_summary_or_TEST_release": True,
            "terminal_failed_private_selection_not_read": True, "closure_markers_forbid_resume": True,
            "fixture_files_only_literal_text_and_JSON": True}


def main():
    parent = HERE.parent
    roots = {"design": parent / "graph_ncNC_collab_predictive_pilot_design_20261003_v1",
             "prototype": parent / "graph_ncNC_member_completion_qualification_preparation_20261003_v2",
             "resource": parent / "graph_ncNC_collab_resource_epoch_preparation_20261003_v4"}
    for name, expected in (("design", PLAN_SHA), ("prototype", MODEL_SHA), ("resource", RESOURCE_SHA)):
        verify_manifest(roots[name], expected)
    authority = parent / "graph_ncNC_valid_data_authority_root_20261003_v1" / "DATA_AUTHORITY.json"
    require(file_sha(authority) == DATA_AUTHORITY_SHA, "TRAIN/VALID authority changed")
    data = json.loads(authority.read_text())
    require(set(data["files"]) == set(DATA_FILES) and data["test_file_opened"] is False, "TRAIN/VALID-only authority differs")
    spec = importlib.util.spec_from_file_location("source_only_design", roots["design"] / "design_spec.py")
    design = importlib.util.module_from_spec(spec); spec.loader.exec_module(design)
    design_receipt = design.validate_plan(json.loads((roots["design"] / "PILOT_PLAN.json").read_text()))
    parsed = []
    numerical_names = {"torch", "numpy", "pandas", "torch_sparse", "torch_geometric", "ogb"}
    for path in sorted(HERE.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Import):
                require(not any(alias.name.split(".")[0] in numerical_names for alias in node.names), "Top-level numerical import")
            if isinstance(node, ast.ImportFrom):
                require((node.module or "").split(".")[0] not in numerical_names, "Top-level numerical import")
        parsed.append({"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha(path)})
    fixture_context = {"release": {"family_id": "fixture", "authorized_invocations": []}, "identity": {}}
    for stage, seed in (("TEST", 0), ("fit", 5), ("fit", 0)):
        try:
            admit_invocation(fixture_context, stage, "native_bank4", seed, "/tmp/fabricated_output", False)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Unreleased/TEST invocation was accepted")
    family = metadata_fixtures()
    require(not (numerical_names & set(sys.modules)), "Source preparation imported numerical libraries")
    receipt = {"schema": "ncnc-pilot-stdlib-source-preparation-v1", "status": "PASS", "UTC": utc(),
        "AST_files": parsed, "sealed_dependencies_verified": {"design": PLAN_SHA, "prototype": MODEL_SHA, "resource": RESOURCE_SHA},
        "data_authority_sha256": DATA_AUTHORITY_SHA, "design": design_receipt, "family_metadata_fixtures": family,
        "unreleased_and_TEST_invocations_rejected": True, "numerical_libraries_imported": [], "dataset_files_opened": [],
        "actual_model_or_checkpoint_files_opened": [], "scientific_or_remote_execution": False,
        "live_synthetic_resource_VALID_qualification_required": True}
    atomic_json(HERE / "STDLIB_PREPARATION_CHECK.json", receipt)
    print(json.dumps({"status": receipt["status"], "AST_file_count": len(parsed), "family_metadata_fixtures": family}, indent=2))


if __name__ == "__main__":
    main()
