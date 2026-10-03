"""Fabricated metadata/contract QA only; never import Torch or read project outcomes."""
from pathlib import Path
from tempfile import TemporaryDirectory
from copy import deepcopy
import ast
import hashlib
import importlib.util
import json
import sys
from replay_gate import (ARMS, UNITS, EXPECTED_ARMS, descriptor, family_gate, preflight,
                         admission_context, runtime_and_data_custody, verify_manifest)
from replay_contract import (validate_journal_payload, validate_selected_payload,
                             public_cells, summaries)

HERE = Path(__file__).resolve().parent
DESIGN = HERE.parent / "graph_ncNC_collab_predictive_pilot_design_20261003_v1" / "design_spec.py"
spec = importlib.util.spec_from_file_location("fabricated_original_selector", DESIGN)
design = importlib.util.module_from_spec(spec)
spec.loader.exec_module(design)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return descriptor(path)


def local(path):
    return {**descriptor(path), "path": path.name}


def epoch_metadata(epoch):
    return {"epoch": epoch, "start_rng_sha256": "start" + str(epoch), "end_rng_sha256": "end" + str(epoch), "stream": {"full_batches": 17, "supervised_records": 1114112, "dropped_tail_records": 64940, "negative_rows_drawn": 1179052}, "full_batches": 17, "optimizer_steps": 17, "positive_queries": 60084, "negative_queries": 100000, "train_wall_seconds": 1.0, "valid_wall_seconds": 1.0}


def fixture(root, mutation=None, failed=False):
    identity = {"family_id": "FABRICATED_ONLY", "family_lock_output_directory": str(root / "lock")}
    lock_root = root / "lock"
    lock_root.mkdir()
    output = root / "future_output"
    inputs, pins, cells = [], [], []
    for unit in UNITS:
        for seed in range(5):
            key = (unit, seed)
            target = root / (unit + "_" + str(seed))
            target.mkdir()
            count = 4 if unit == UNITS[0] else 1
            complete = {"schema": "ncnc-pilot-complete-unit-v1", "identity": identity, "unit": unit, "base_seed": seed, "epochs": 100, "unique_fits": count, "completed_optimizer_steps": 1700 * count, "scientific_initialization": "fresh", "resource_state_donor": False, "test_file_opened": False, "all_official_VALID_rows_complete": True, "independent_candidate_count": 101 if count == 4 else 0, "candidate101_extra_VALID_evaluations": 4 if count == 4 else 0, "selected_checkpoints": {}, "selection_artifacts": {}, "member_metadata": [{"member": member, "fit_seed": seed + 5 * member if count == 4 else seed, "initial_state_sha256": "initial" + str(seed), "initial_rng_sha256": "initial_rng" + str(seed), "epochs": [epoch_metadata(e) for e in range(1, 101)]} for member in range(count)]}
            retire = failed and key == ("native70", 4)
            selected_rows = {}
            for arm in EXPECTED_ARMS[unit]:
                selected_path = target / ("SELECTED_" + arm + ".pt")
                selected_path.write_bytes(b"FABRICATED_NO_TORCH_OR_PICKLE")
                cp = local(selected_path)
                selection = {"candidate_id": "same_epoch_1" if arm == ARMS[1] else "epoch_1", "order": 1, "hits50": .5}
                selected_rows[arm] = {"schema": "ncnc-pilot-validation-selected-checkpoint-v1", "identity": identity, "arm": arm, "seed": seed, "selection": selection, "checkpoint": cp}
                complete["selected_checkpoints"][arm] = cp
            if mutation is not None:
                mutation(key, complete, selected_rows)
            for arm, row in selected_rows.items():
                path = target / ("PRIVATE_SELECTION_" + arm + ".json")
                write(path, row)
                complete["selection_artifacts"][arm] = local(path)
                cells.append({"arm": arm, "base_seed": seed, "status": "MISSING_TERMINAL_FAILED" if retire else "COMPLETE", "served_VALID_hits50": None if retire else row["selection"]["hits50"], "selection": row["selection"] if not retire else None, "checkpoint": None if retire else {"output_directory": str(target), **row["checkpoint"]}})
            ledger_path = target / "ATTEMPTS.json"
            ledger = {"schema": "ncnc-pilot-inclusive-attempts-v1", "identity": identity, "attempts": [{"attempt": 1, "stage": "fit", "unit": unit, "base_seed": seed, "status": "INTERRUPTED_UNCLOSED", "observed_wall_seconds": 2.0, "inclusive_wall_seconds": None}, {"attempt": 2, "stage": "fit", "unit": unit, "base_seed": seed, "status": "FAILED" if retire else "COMPLETE", "observed_wall_seconds": 3.0, "inclusive_wall_seconds": 3.0}]}
            write(ledger_path, ledger)
            state_path = target / "STATE_SLOT_0.pt"
            state_path.write_bytes(b"FABRICATED_NO_DESERIALIZATION")
            journal_path = target / "JOURNAL.json"
            write(journal_path, {"schema": "ncnc-pilot-private-epoch-journal-v1", "identity": identity, "unit": unit, "seed": seed, "epoch": 100, "revision": 100, "state_file": local(state_path)})
            terminal_path = target / ("FAILED.json" if retire else "COMPLETE.json")
            terminal = {"schema": "ncnc-pilot-failed-unit-v1", "identity": identity, "unit": unit, "base_seed": seed, "stage": "fit", "status": "FAILED", "test_file_opened": False, "attempts_receipt": local(ledger_path), "journal_receipt": None} if retire else complete
            write(terminal_path, terminal)
            physical_path = root / (unit + "_" + str(seed) + "_PHYSICAL.json")
            write(physical_path, {"unit": unit, "base_seed": seed, "output_directory": str(target), "exit_code": 1 if retire else 0})
            disposition = "TERMINAL_FAILED" if retire else "COMPLETE"
            binding = {"unit": unit, "base_seed": seed, "output_directory": str(target), "disposition": disposition, "failure_sha256" if retire else "complete_sha256": local(terminal_path)["sha256"]}
            if retire:
                binding.update(terminal_failure_immutable=True, terminal_failure_authorization_reference="FABRICATED_RETIREMENT")
            inputs.append(binding)
            pins.append({"unit": unit, "base_seed": seed, "output_directory": str(target), "disposition": disposition, "terminal": local(terminal_path), "physical_terminal": descriptor(physical_path), "attempts": local(ledger_path), "journal": None if retire else local(journal_path)})
    lock = {"schema": "ncnc-pilot-full-family-lock-v1", "identity": identity, "status": "TERMINAL_FAILED_FAMILY_LOCKED" if failed else "COMPLETE_FAMILY_LOCKED", "unique_fits_planned": 35, "unique_fits_completed": 34 if failed else 35, "served_arm_seed_cells": 25, "complete_served_cells": 24 if failed else 25, "test_file_opened": False, "TEST_execution_authorized": False, "no_success_only_subset_summary": True, "terminal_failures": ["FABRICATED_FAILURE"] if failed else [], "arm_summaries": None if failed else {}, "primary_development_pilot": {"paired_VALID_hits50_differences": None if failed else [0] * 5}, "inputs": inputs, "cells": cells}
    lock_pin = write(lock_root / "FAMILY_LOCK.json", lock)
    for pin in pins:
        path = Path(pin["output_directory"]) / "FAMILY_CLOSURE.json"
        write(path, {"schema": "ncnc-pilot-immutable-family-closure-v1", "identity": identity, "family_lock_path": lock_pin["path"], "family_lock_sha256": lock_pin["sha256"], "status": lock["status"], "resume_or_seed_replacement_permitted": False})
        pin["closure"] = local(path)
    return {"identity": identity, "output": output, "release": {"family_lock": lock_pin, "unit_custody": pins}}


def fake_snapshot(label):
    return {"models": [{"w": label}, {"w": label}], "optimizer": {"state": label}, "rng": {"fabricated": label}, "flags": [{"": False}, {"": False}]}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def contract_fixture():
    complete = {"member_metadata": []}
    records = []
    valid = {"positive_queries": 60084, "negative_queries": 100000, "all_query_rows_complete": True, "encoder_calls": 1, "query_batches": [1, 1], "graph": "complete_TRAIN_only", "serving_pool": "mean_raw_logits", "score_digests": {"positive": "FAKE", "negative": "FAKE"}}
    for member in range(4):
        metadata = {"fit_seed": 5 * member, "member": member, "initial_state_sha256": "initial", "initial_rng_sha256": "rng", "epochs": [epoch_metadata(e) for e in range(1, 101)]}
        complete["member_metadata"].append(metadata)
        records.append({**{k: metadata[k] for k in ("fit_seed", "member", "initial_state_sha256", "initial_rng_sha256")}, "current": fake_snapshot("final" + str(member)), "best": {"candidate_id": "epoch_7", "hits50": .6, "order": 7}, "best_state": fake_snapshot("individual_best" + str(member)), "epochs": [{"epoch": e, "start_rng_sha256": "start" + str(e), "end_rng_sha256": "end" + str(e), "train": {"stream": epoch_metadata(e)["stream"], "full_batches": 17, "optimizer_steps": 17}, "VALID": deepcopy(valid), "private_hits50": .6 if e in (7, 8) else .5} for e in range(1, 101)]})
    candidates = [{"order": e, "candidate_id": "same_epoch_" + str(e), "private_hits50": .7 if e == 22 else .5} for e in range(1, 101)]
    candidates.append({"order": 101, "candidate_id": "individual_validation_best_bank", "private_hits50": .65, "extra_VALID_evaluations": [deepcopy(valid) for _ in range(4)], "member_individual_best_epochs": [7] * 4})
    state = {"epoch": 100, "phase": "complete", "fresh_scientific_initialization": True, "resource_state_donor": False, "fits": records, "ensemble_candidates": candidates, "ensemble_best": {"candidate_id": "same_epoch_22", "hits50": .7, "order": 22}, "ensemble_best_states": [fake_snapshot("ensemble_epoch22_" + str(i)) for i in range(4)]}
    payload = {"schema": "ncnc-pilot-own-trusted-state-v1", "identity": {}, "unit": "native_bank4", "seed": 0, "state": state}
    cell = {"unit": "native_bank4", "base_seed": 0, "arm": ARMS[0], "selection": records[0]["best"]}
    selected = {"schema": "ncnc-pilot-validation-selected-checkpoint-v1", "identity": {}, "unit": "native_bank4", "seed": 0, "arm": ARMS[0], "serving_pool": "mean_raw_logits", "validation_graph": "complete_TRAIN_only", "eventual_test_graph": "native_TRAIN_plus_VALID", "fresh_scientific_initialization": True, "resource_state_donor": False, "selection": cell["selection"], "states": [deepcopy(records[0]["best_state"])]}
    return complete, payload, cell, selected


def rebind_json(context, index, key, mutate):
    pin = context["release"]["unit_custody"][index]
    receipt = pin[key]
    path = Path(receipt["path"]) if key == "physical_terminal" else Path(pin["output_directory"]) / receipt["path"]
    value = json.loads(path.read_text())
    mutate(value)
    write(path, value)
    pin[key] = descriptor(path) if key == "physical_terminal" else local(path)


def rebind_lock(context, mutate):
    path = Path(context["release"]["family_lock"]["path"])
    lock = json.loads(path.read_text())
    mutate(lock)
    context["release"]["family_lock"] = write(path, lock)
    for pin in context["release"]["unit_custody"]:
        closure_path = Path(pin["output_directory"]) / "FAMILY_CLOSURE.json"
        value = json.loads(closure_path.read_text())
        value["family_lock_sha256"] = context["release"]["family_lock"]["sha256"]
        write(closure_path, value)
        pin["closure"] = local(closure_path)


def main():
    cases = []
    def check(name, operation, rejects=False):
        try:
            operation()
        except (RuntimeError, FileNotFoundError):
            if not rejects:
                raise
        else:
            if rejects:
                raise AssertionError("Fail-open fixture: " + name)
        cases.append({"case": name, "status": "PASS"})
    def gate_case(name, mutate=None, after=None, rejects=False, failed=False):
        with TemporaryDirectory(prefix="ncnc_replay_stdlib_") as directory:
            context = fixture(Path(directory).resolve(), mutate, failed)
            if after:
                after(context)
            check(name, lambda: family_gate(context), rejects)
            if not rejects:
                gated = family_gate(context)
                assert len(gated["cells"]) == 25
                assert sum(c["interrupted_observed_wall_lower_bound_seconds"] for c in gated["costs"]) == 40
                assert gated["full_success"] is not failed
    gate_case("fabricated_complete20_35fits_25cells_all100epochs")
    gate_case("immutable_failed_family_metadata_only_no_partial_replay", failed=True)
    gate_case("missing_original_unit", after=lambda c: c["release"]["unit_custody"].pop(), rejects=True)
    gate_case("duplicate_original_unit_alias", after=lambda c: c["release"]["unit_custody"].__setitem__(1, deepcopy(c["release"]["unit_custody"][0])), rejects=True)
    gate_case("borrowed_output", after=lambda c: c["release"]["unit_custody"][0].__setitem__("output_directory", c["release"]["unit_custody"][1]["output_directory"]), rejects=True)
    gate_case("pending_unit", after=lambda c: c["release"]["unit_custody"][0].__setitem__("disposition", "PENDING"), rejects=True)
    gate_case("incomplete_epoch_stream", mutate=lambda key, m, s: m["member_metadata"][0]["epochs"].pop() if key == (UNITS[0], 0) else None, rejects=True)
    gate_case("VALID_tail_omitted", mutate=lambda key, m, s: m["member_metadata"][0]["epochs"][0].__setitem__("negative_queries", 99999) if key == (UNITS[0], 0) else None, rejects=True)
    gate_case("candidate101_work_missing", mutate=lambda key, m, s: m.__setitem__("candidate101_extra_VALID_evaluations", 0) if key == (UNITS[0], 0) else None, rejects=True)
    gate_case("native_member_seed_mismatch", mutate=lambda key, m, s: m["member_metadata"][1].__setitem__("fit_seed", 0) if key == (UNITS[0], 0) else None, rejects=True)
    gate_case("twin_initial_RNG_mismatch", mutate=lambda key, m, s: m["member_metadata"][0].__setitem__("initial_rng_sha256", "different") if key == (UNITS[2], 0) else None, rejects=True)
    gate_case("twin_epoch100_end_RNG_mismatch", mutate=lambda key, m, s: m["member_metadata"][0]["epochs"][-1].__setitem__("end_rng_sha256", "different") if key == (UNITS[2], 0) else None, rejects=True)
    gate_case("selector_candidate_identity_mismatch", mutate=lambda key, m, s: s[ARMS[0]]["selection"].__setitem__("candidate_id", "same_epoch_1") if key == (UNITS[0], 0) else None, rejects=True)
    gate_case("selected_checkpoint_byte_custody_mismatch", after=lambda c: (Path(c["release"]["unit_custody"][0]["output_directory"]) / ("SELECTED_" + ARMS[0] + ".pt")).write_bytes(b"altered"), rejects=True)
    gate_case("missing_closure_marker", after=lambda c: (Path(c["release"]["unit_custody"][0]["output_directory"]) / "FAMILY_CLOSURE.json").unlink(), rejects=True)
    gate_case("active_unclosed_attempt", after=lambda c: rebind_json(c, 0, "attempts", lambda v: v["attempts"][-1].__setitem__("status", "IN_PROGRESS")), rejects=True)
    gate_case("borrowed_journal_state_path", after=lambda c: rebind_json(c, 0, "journal", lambda v: v["state_file"].__setitem__("path", "../STATE_SLOT_0.pt")), rejects=True)
    gate_case("boolean_physical_exit_is_not_integer_zero", after=lambda c: rebind_json(c, 0, "physical_terminal", lambda v: v.__setitem__("exit_code", False)), rejects=True)
    gate_case("duplicate_served_cell", after=lambda c: rebind_lock(c, lambda v: v["cells"].__setitem__(1, deepcopy(v["cells"][0]))), rejects=True)
    complete, payload, cell, selected = contract_fixture()
    check("original_strict_selector_first_tie_and_101_order", lambda: validate_journal_payload(payload, {"epoch": 100}, complete, {}, "native_bank4", 0, design.select_validation_candidate))
    check("N64_individual_member0_best_different_from_I4_epoch_is_valid", lambda: validate_selected_payload(selected, cell, payload["state"], {}, digest))
    wrong = deepcopy(selected)
    wrong["states"] = [payload["state"]["ensemble_best_states"][0]]
    check("N64_wrong_I4_served_member0_alias_rejected", lambda: validate_selected_payload(wrong, cell, payload["state"], {}, digest), True)
    for key in ("models", "rng", "flags"):
        wrong = deepcopy(selected)
        wrong["states"][0][key] = ["different"]
        check("selected_complete_" + key + "_digest_mismatch", lambda: validate_selected_payload(wrong, cell, payload["state"], {}, digest), True)
    wrong = deepcopy(payload)
    wrong["state"]["ensemble_candidates"].pop()
    check("journal_missing_101st_candidate", lambda: validate_journal_payload(wrong, {"epoch": 100}, complete, {}, "native_bank4", 0, design.select_validation_candidate), True)
    check("no_subset_summary", lambda: summaries({}, [{"status": "PASS"}] * 24), True)
    private = [{"arm": ARMS[0], "base_seed": 0, "status": "PASS", "replayed_VALID_hits50": .5, "selected_hits50_equal": True}]
    assert public_cells(private)[0]["replayed_VALID_hits50"] is None
    cases.append({"case": "failure_hides_partial_metrics_and_contrasts", "status": "PASS"})
    check("disabled_release_cannot_launch", lambda: preflight(HERE / "ROOT_RELEASE_TEMPLATE.json", "/tmp/disabled_ncnc_replay"), True)
    check("disabled_synthetic_release_cannot_launch", lambda: __import__("replay_synthetic_admission").preflight(HERE / "ROOT_SYNTHETIC_RELEASE_TEMPLATE.json", "/tmp/disabled_ncnc_synthetic"), True)
    with TemporaryDirectory(prefix="ncnc_final_custody_") as directory:
        temp = Path(directory).resolve()
        runtime_files = []
        for name in ("runtime_source.py", "runtime_binary.so", "negative_sampler.py", "ogb_evaluator.py"):
            p = temp / name
            p.write_bytes(b"STDLIB_FAKE_RUNTIME_BYTES")
            runtime_files.append(descriptor(p))
        dataset = temp / "fabricated_dataset"
        dataset.mkdir()
        files = {}
        for name in ("raw/node-feat.csv.gz", "raw/edge.csv.gz", "split/time/train.pt", "split/time/valid.pt"):
            p = dataset / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"STDLIB_FAKE_DATA_BYTES_NO_TENSORS")
            files[name] = {k: v for k, v in descriptor(p).items() if k != "path"}
        context = {"runtime": {"interpreter_path": str(Path(sys.executable).resolve()), "interpreter_sha256": descriptor(Path(sys.executable).resolve())["sha256"],
                               "distribution_versions": {}, "runtime_source_pins": runtime_files[:1], "runtime_binary_files": runtime_files[1:2], "negative_sampler": runtime_files[2]},
                   "authority": {"dataset_root": str(dataset), "files": files, "ogb_evaluator": runtime_files[3]}}
        check("B1_actual_byte_custody_complete_metadata_fixture", lambda: runtime_and_data_custody(context))
        p = dataset / "split/time/valid.pt"
        old = p.read_bytes()
        p.write_bytes(old + b"changed")
        check("B1_actual_data_byte_drift_rejected", lambda: runtime_and_data_custody(context), True)
        p.write_bytes(old)
        for index, name in ((1, "binary"), (2, "sampler"), (3, "evaluator")):
            p = Path(runtime_files[index]["path"])
            old = p.read_bytes()
            p.write_bytes(old + b"changed")
            check("B1_actual_" + name + "_byte_drift_rejected", lambda: runtime_and_data_custody(context), True)
            p.write_bytes(old)
        release = temp / "OWN_ADMITTED_RELEASE.json"
        release.write_text('{"fabricated_only":true}\n')
        admitted = descriptor(release)["sha256"]
        release.write_bytes(release.read_bytes() + b"changed")
        check("B1_admitted_release_hash_drift_rejected_before_reparse", lambda: admission_context(release, temp / "output", expected_release_sha256=admitted), True)
        source = temp / "sealed_fake_source"
        source.mkdir()
        payload = source / "fake.py"
        payload.write_text("# fabricated source bytes only\n")
        write(source / "MANIFEST.json", {"files": [local(payload)]})
        admitted = descriptor(source / "MANIFEST.json")["sha256"]
        payload.write_text("# changed fake source bytes only\n")
        check("B1_dependency_payload_drift_rejected", lambda: verify_manifest(source, admitted), True)
    for path in HERE.glob("*.py"):
        tree = ast.parse(path.read_text())
        compile(tree, str(path), "exec")
        if path.name.startswith("replay_"):
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) else [node.module.split(".")[0]]
                    assert not set(names) & {"torch", "numpy", "pandas", "torch_sparse", "torch_geometric"}
    assert not any(n in sys.modules for n in ("torch", "numpy", "pandas", "pilot_fit", "pilot_train", "pilot_data"))
    receipt = {"schema": "ncnc-selected-state-replay-stdlib-check-v1", "status": "PASS", "cases": cases, "case_count": len(cases), "scientific_modules_imported": False, "project_data_or_outcomes_opened": False, "TEST_opened": False, "numerical_qualification": False, "synthetic_harness_executed": False}
    write(HERE / "STDLIB_PREPARATION_CHECK.json", receipt)
    print(json.dumps({"status": "PASS", "case_count": len(cases), "scientific_execution": False}))


if __name__ == "__main__":
    main()
