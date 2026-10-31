"""Six-cell fixed F2 objective alignment collection and label-free arithmetic replay."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
sys.path.insert(0, str(PHASE))
sys.path.insert(0, str(PHASE / "spectral_privacy/ppi_f2_audit_v1"))
sys.path.insert(0, str(PHASE / "spectral_privacy/ppi_integration_v1"))

import torch
import audit_f2
from collect_ppi import checked_masks, checked_source, load_module, load_train
from spectral_privacy.v2_abi_fix_01.adapters import SourceSAGEEnsemble
from spectral_privacy.v2_abi_fix_01.collector import environment_contract, unit_fingerprint
from spectral_privacy.v2_abi_fix_01.contracts import pack_float64, save_new_json, state_hash, tensor_fingerprint, unpack_float64

from freeze import LOCK_SCHEMA, checked, code_hashes, json_hash, phase_path, ref, verify_seal
from objective_core import (GROUPS, OBJECTIVES, TARGETS, NUMERIC_RULE, TOLERANCE,
                            check_chain, collect_one, diagnostics, pack_gradients,
                            projected_chain, unpack_gradients)

RECORD_SCHEMA = "gnnm-F2-objective-alignment-record-v1"


def seal(value, field):
    value = {key: item for key, item in value.items() if key != field}
    return value | {field: json_hash(value)}


def validate_lock(path):
    lock = json.loads(phase_path(path).read_text())
    verify_seal(lock, "lock_sha256")
    if (lock["schema"] != LOCK_SCHEMA or lock["code_sha256"] != code_hashes()
            or lock["unit_ids"] != ["ppi:train:09", "ppi:train:18"] or lock["epoch"] != 25
            or lock["runtime"] != {"device": "cpu", "threads": 1, "interop_threads": 1}
            or lock["numeric_rule"] != {"forward_and_pooling": "native float32", "loss_and_diagnostic_reductions": "float64"}
            or lock["basis_or_rank_changes"] is not False or lock["promotion_rule"] is not False
            or lock["unused_F2_claimed"] is not False or lock["validation_or_test_access"] is not False):
        raise ValueError("New diagnostic code/recipe differs from its own freeze")
    for dependency in lock["source_dependencies"]:
        checked(dependency)
    old_lock, inputs = audit_f2.validate_audit(checked(lock["previous_F2_audit_lock"]))
    old_gate = json.loads(checked(lock["previous_F2_pregradient_gate"]).read_text())
    verify_seal(old_gate, "gate_sha256")
    if [row["arm_id"] for row in lock["arms"]] != [row["arm_id"] for row in old_lock["arms"]]:
        raise ValueError("All six original cells must remain in order")
    for row in lock["arms"]:
        previous = json.loads(checked(row["previous_F2_record"]).read_text())
        audit_f2.replay_F2(previous, old_lock)
        replay = json.loads(checked(row["previous_F2_replay"]).read_text())
        if replay["record_sha256"] != previous["record_sha256"]:
            raise ValueError("Prior F2 replay differs")
    return lock, old_lock, inputs


def identical_source(source):
    result = source.tied_source_copy()
    with torch.no_grad():
        for boundary in (result.input_be_block, result.output_be_block):
            for name in ("R", "S", "B"):
                value = getattr(boundary, name)
                value[1:].copy_(value[:1].expand_as(value[1:]))
    return result


def native_bases(models):
    return {group: {key: model.frozen_bases()[key].detach().cpu().double() for key in TARGETS}
            for group, model in models.items()}


def prepare_models(common, bases):
    before_state, before_rng = state_hash(common.tied_source_copy()), torch.get_rng_state().clone()
    models = {group: SourceSAGEEnsemble(common.tied_source_copy(), bases[group]) for group in GROUPS}
    for model in models.values():
        model.eval()
        if not model.is_tied_at_state() or state_hash(model.tied_source_copy()) != before_state:
            raise ValueError("Group construction changed original source tensors")
    if not torch.equal(before_rng, torch.get_rng_state()):
        raise ValueError("Fixed-basis construction consumed RNG")
    null_source = identical_source(common)
    null_models = {group: SourceSAGEEnsemble(null_source, bases[group]) for group in GROUPS}
    for model in null_models.values():
        model.eval()
    return models, null_models


def collect_cell(common, models, null_models, units, old_arm, row, lock, gate_ref):
    previous = json.loads(checked(row["previous_F2_record"]).read_text())
    fingerprints = [unit_fingerprint(unit) for unit in units]
    if fingerprints != row["F2_input_fingerprints"]:
        raise ValueError("Current F2 graph/mask fingerprints differ from the prior audit")
    bases = native_bases(models)
    contrasts = models["selected"].C.detach().cpu().double()
    gradients = {group: {name: [] for name in OBJECTIVES} for group in GROUPS}
    null_gradients = {group: {name: [] for name in OBJECTIVES} for group in GROUPS}
    responses = {name: {key: [] for key in TARGETS} for name in OBJECTIVES}
    graph_metadata, chain_checks, archive_checks, null_checks = [], {}, {}, {}
    for graph_index, unit in enumerate(units):
        base = collect_one(common, unit, matrices=True)
        for name in OBJECTIVES:
            for key in TARGETS:
                responses[name][key].append(base["source_response_matrices"][name][key])
        archived = {key: unpack_float64(previous["matrix_gradients"][key])[graph_index] for key in TARGETS}
        archive_checks[unit.unit_id] = {}
        for key in TARGETS:
            current = base["source_response_matrices"]["mean_member_bce"][key]
            if not torch.allclose(current, archived[key], atol=2e-6, rtol=2e-5):
                raise ValueError("Fresh own-member matrix gradients drifted from existing F2 autograd")
            archive_checks[unit.unit_id][key] = {"max_abs_error": float((current - archived[key]).abs().max()),
                                               "atol": 2e-6, "rtol": 2e-5,
                                               "numeric_difference": "new float64 scalar reductions versus previous native mean reduction"}
        graph_metadata.append({"unit_id": unit.unit_id, "observed_entries": base["observed_entries"],
                               "losses": base["losses"], "own_member_losses": base["own_member_losses"],
                               "jensen_gap": base["jensen_gap"],
                               "member_logits": tensor_fingerprint(base["member_logits"])})
        chain_checks[unit.unit_id], null_checks[unit.unit_id] = {}, {}
        for group in GROUPS:
            measured = collect_one(models[group], unit)
            if not torch.equal(measured["member_logits"], base["member_logits"]):
                raise ValueError("Nonzero-rank zero-coefficient forward is not the common original function")
            expected = projected_chain(base["source_response_matrices"], bases[group], contrasts)
            chain_checks[unit.unit_id][group] = check_chain(measured["coefficient_gradients"], expected, torch.float32)
            for name in OBJECTIVES:
                gradients[group][name].append(measured["coefficient_gradients"][name])
            null = collect_one(null_models[group], unit)
            if not all(torch.equal(null["member_logits"][0], null["member_logits"][m]) for m in range(1, 4)):
                raise ValueError("Identical-members null has non-identical logits")
            maxima = {}
            for name in OBJECTIVES:
                values = null["coefficient_gradients"][name]
                maximum = max((float(value.abs().max()) if value.numel() else 0) for value in values.values())
                if maximum > TOLERANCE["float32_null_max_abs"]:
                    raise ValueError("Identical-members coefficient-gradient null failed")
                maxima[name] = maximum
                null_gradients[group][name].append(values)
            null_checks[unit.unit_id][group] = {"identical_logits": True,
                                               "gradient_max_abs": maxima,
                                               "fixed_absolute_tolerance": TOLERANCE["float32_null_max_abs"],
                                               "member_minus_pool_loss": null["losses"]["mean_member_bce"] - null["losses"]["pooled_logit_bce"]}
    response_pack = {name: {key: pack_float64(torch.stack(values)) for key, values in rows.items()}
                     for name, rows in responses.items()}
    return seal({"schema": RECORD_SCHEMA, "arm_id": old_arm["arm_id"],
                 "seed": old_arm["seed"], "label_fraction": old_arm["label_fraction"],
                 "lock_sha256": lock["lock_sha256"], "pregradient_gate": gate_ref,
                 "checkpoint": old_arm["checkpoint"], "epoch": 25,
                 "original_source_state_sha256": old_arm["original_source_state_sha256"],
                 "unit_ids": lock["unit_ids"], "input_fingerprints": fingerprints,
                 "previous_F2_record": row["previous_F2_record"],
                 "fixed_bases_sha256": old_arm["fixed_bases_sha256"], "ranks": old_arm["ranks"],
                 "native_basis_buffers": {group: {key: pack_float64(value) for key, value in rows.items()}
                                          for group, rows in bases.items()},
                 "native_contrast_buffer": pack_float64(contrasts),
                 "numeric_rule": NUMERIC_RULE, "tolerances": TOLERANCE,
                 "graph_metadata": graph_metadata, "objective_source_response_matrices": response_pack,
                 "coefficient_gradients": pack_gradients(gradients),
                 "identical_members_null_gradients": pack_gradients(null_gradients),
                 "diagnostics": diagnostics(gradients, bases, contrasts),
                 "objective_chain_checks": chain_checks, "previous_F2_own_gradient_checks": archive_checks,
                 "identical_members_null_checks": null_checks,
                 "environment": environment_contract(), "code_sha256": code_hashes(),
                 "training_performed": False, "validation_or_test_scored": False,
                 "criterion_or_bases_changed": False, "promotion_rule": False,
                 "F2_already_used_in_prior_audit": True,
                 "CI_or_pvalue_claimed": False, "state_grads_modes_rng_preserved": True}, "record_sha256")


def replay(record, lock, old_lock):
    verify_seal(record, "record_sha256")
    row = next(item for item in lock["arms"] if item["arm_id"] == record["arm_id"])
    old_arm = next(item for item in old_lock["arms"] if item["arm_id"] == record["arm_id"])
    if (record["schema"] != RECORD_SCHEMA or record["lock_sha256"] != lock["lock_sha256"]
            or record["seed"] != old_arm["seed"] or record["label_fraction"] != old_arm["label_fraction"]
            or record["epoch"] != 25 or record["unit_ids"] != lock["unit_ids"]
            or record["input_fingerprints"] != row["F2_input_fingerprints"]
            or record["checkpoint"] != old_arm["checkpoint"] or record["ranks"] != old_arm["ranks"]
            or record["original_source_state_sha256"] != old_arm["original_source_state_sha256"]
            or record["fixed_bases_sha256"] != old_arm["fixed_bases_sha256"]
            or record["previous_F2_record"] != row["previous_F2_record"]
            or record["numeric_rule"] != NUMERIC_RULE or record["tolerances"] != TOLERANCE
            or record["environment"] != environment_contract() or record["code_sha256"] != code_hashes()
            or record["promotion_rule"] is not False or record["criterion_or_bases_changed"] is not False
            or record["training_performed"] is not False or record["validation_or_test_scored"] is not False
            or record["F2_already_used_in_prior_audit"] is not True
            or record["state_grads_modes_rng_preserved"] is not True):
        raise ValueError("Record source/input/objective/environment identity differs")
    gate = json.loads(checked(record["pregradient_gate"]).read_text())
    verify_seal(gate, "gate_sha256")
    if gate["lock_sha256"] != lock["lock_sha256"] or gate["status"] != "OBJECTIVE_SOURCE_INPUT_BASES_GATE_PASS":
        raise ValueError("Runtime input gate differs")
    gate_arm = next(item for item in gate["arms"] if item["arm_id"] == record["arm_id"])
    if (gate_arm["F2_input_fingerprints"] != row["F2_input_fingerprints"]
            or gate_arm["original_source_state_sha256"] != old_arm["original_source_state_sha256"]
            or gate_arm["fixed_bases_sha256"] != old_arm["fixed_bases_sha256"]):
        raise ValueError("Runtime gate cell identity differs")
    _, _, frozen = audit_f2.decode_fixed(old_arm)
    bases = {group: {key: unpack_float64(value) for key, value in rows.items()}
             for group, rows in record["native_basis_buffers"].items()}
    if set(bases) != set(GROUPS) or any(set(rows) != set(TARGETS) for rows in bases.values()):
        raise ValueError("Native basis keys differ")
    from spectral_privacy.v2_abi_fix_01.adapters import helmert
    contrasts = unpack_float64(record["native_contrast_buffer"])
    if not torch.equal(contrasts, helmert(4, dtype=torch.float32).double()):
        raise ValueError("Native source contrast buffer changed")
    for group in GROUPS:
        for key in TARGETS:
            if not torch.equal(bases[group][key], frozen[group][key].float().double()):
                raise ValueError("Native adapter basis bytes differ from the fixed saved basis")
    gradients = unpack_gradients(record["coefficient_gradients"])
    null_gradients = unpack_gradients(record["identical_members_null_gradients"])
    responses = {name: {key: unpack_float64(value) for key, value in rows.items()}
                 for name, rows in record["objective_source_response_matrices"].items()}
    if (set(responses) != set(OBJECTIVES) or any(set(rows) != set(TARGETS) for rows in responses.values())
            or any(value.shape != (2, 4, 128, 128) for rows in responses.values() for value in rows.values())):
        raise ValueError("Source response evidence dimensions/keys differ")
    previous = json.loads(checked(row["previous_F2_record"]).read_text())
    archive_checks = {}
    chain_checks = {}
    for i, unit_id in enumerate(lock["unit_ids"]):
        archive_checks[unit_id] = {}
        for key in TARGETS:
            archived = unpack_float64(previous["matrix_gradients"][key])[i]
            current = responses["mean_member_bce"][key][i]
            if not torch.allclose(current, archived, atol=2e-6, rtol=2e-5):
                raise ValueError("Replayed source response differs from previous F2 own gradient")
            archive_checks[unit_id][key] = {"max_abs_error": float((current - archived).abs().max()),
                                          "atol": 2e-6, "rtol": 2e-5,
                                          "numeric_difference": "new float64 scalar reductions versus previous native mean reduction"}
        chain_checks[unit_id] = {}
        for group in GROUPS:
            maxima = {}
            for name in OBJECTIVES:
                maxima[name] = 0
                for key in TARGETS:
                    if gradients[group][name][i][key].shape != (3, old_arm["ranks"][key], 128):
                        raise ValueError("Recorded gradient capacity differs from its frozen rank")
                    null = null_gradients[group][name][i][key]
                    if null.shape != gradients[group][name][i][key].shape or (
                            null.numel() and float(null.abs().max()) > TOLERANCE["float32_null_max_abs"]):
                        raise ValueError("Recorded identical-members null differs")
                    maxima[name] = max(maxima[name], float(null.abs().max()) if null.numel() else 0)
            if maxima != record["identical_members_null_checks"][unit_id][group]["gradient_max_abs"]:
                raise ValueError("Identical-members null maximum does not replay exactly")
            actual = {name: gradients[group][name][i] for name in OBJECTIVES}
            matrices = {name: {key: values[i] for key, values in rows.items()} for name, rows in responses.items()}
            chain_checks[unit_id][group] = check_chain(actual, projected_chain(matrices, bases[group], contrasts), torch.float32)
    if json_hash(chain_checks) != json_hash(record["objective_chain_checks"]):
        raise ValueError("Objective chain reductions do not replay exactly")
    if json_hash(archive_checks) != json_hash(record["previous_F2_own_gradient_checks"]):
        raise ValueError("Previous F2 source-gradient checks do not replay exactly")
    if json_hash(diagnostics(gradients, bases, contrasts)) != json_hash(record["diagnostics"]):
        raise ValueError("Direction/recurrence reductions do not replay exactly")
    return {"status": "EXACT_OBJECTIVE_ARITHMETIC_REPLAY_PASS", "arm_id": record["arm_id"],
            "record_sha256": record["record_sha256"], "ranks": record["ranks"],
            "scope": "label-free recomputation from saved objective-specific gradients; not a fresh autograd reproduction"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "replay"))
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = phase_path(args.output)
    if HERE not in output.parents or output.exists():
        raise ValueError("Use a new output directory inside this diagnostic")
    output.mkdir()
    begin = time.perf_counter()
    try:
        lock, old_lock, inputs = validate_lock(args.lock)
        if args.command == "replay":
            if args.run is None or args.gate is not None:
                raise ValueError("Replay uses only --run and --lock")
            reports = []
            for row in lock["arms"]:
                record = json.loads((phase_path(args.run) / f"{row['arm_id']}_OBJECTIVE_RECORD.json").read_text())
                reports.append(replay(record, lock, old_lock))
            save_new_json({"status": "SIX_CELL_OBJECTIVE_ARITHMETIC_REPLAY_PASS", "reports": reports}, output / "RESULT.json", HERE)
            print(json.dumps({"status": "SIX_CELL_OBJECTIVE_ARITHMETIC_REPLAY_PASS", "output": str(output)}))
            return
        if args.gate is None or args.run is not None:
            raise ValueError("New collection requires a passed synthetic --gate and no --run")
        cpu_gate = json.loads(phase_path(args.gate).read_text())
        if cpu_gate["status"] != "OBJECTIVE_SYNTHETIC_CPU_GATE_PASS" or cpu_gate["code_sha256"] != code_hashes():
            raise ValueError("Synthetic CPU gate missing or from changed code")
        source_module = load_module(checked(inputs["probe_sources"]["source_snapshot/models.py"]), "objective_original_source")
        baseline = load_module(checked(inputs["probe_sources"]["protocols/baseline_adapters.py"]), "objective_original_baseline")
        dataset, _ = load_train(inputs)
        metadata = json.loads(checked(inputs["data_manifest"]).read_text())
        prepared, identities = [], []
        for row, old_arm in zip(lock["arms"], old_lock["arms"]):
            original_arm = next(item for item in inputs["arms"] if item["arm_id"] == row["arm_id"])
            contract = json.loads(checked(old_arm["contract"]).read_text())
            masks, _ = checked_masks(original_arm, contract, dataset)
            units = audit_f2.F2_units(dataset, masks, lock["unit_ids"], metadata)
            common, identity = checked_source(original_arm, contract, source_module, baseline, units[0])
            if identity["original_source_state_sha256"] != old_arm["original_source_state_sha256"]:
                raise ValueError("Original P25 source differs")
            _, _, bases = audit_f2.decode_fixed(old_arm)
            models, null_models = prepare_models(common, bases)
            if [unit_fingerprint(unit) for unit in units] != row["F2_input_fingerprints"]:
                raise ValueError("Observed F2 inputs differ")
            identities.append(identity | {"F2_input_fingerprints": row["F2_input_fingerprints"],
                                          "fixed_bases_sha256": row["fixed_bases_sha256"],
                                          "added_storage": {group: model.added_storage() for group, model in models.items()}})
            prepared.append((row, old_arm, common, models, null_models, units))
        gate = seal({"status": "OBJECTIVE_SOURCE_INPUT_BASES_GATE_PASS", "lock_sha256": lock["lock_sha256"],
                     "arms": identities, "synthetic_gate": ref(args.gate), "gradients_collected": False,
                     "environment": environment_contract()}, "gate_sha256")
        gate_path = output / "PREGRADIENT_GATE.json"
        save_new_json(gate, gate_path, HERE)
        reports, summaries = [], []
        for row, old_arm, common, models, null_models, units in prepared:
            record = collect_cell(common, models, null_models, units, old_arm, row, lock, ref(gate_path))
            record_path = output / f"{row['arm_id']}_OBJECTIVE_RECORD.json"
            save_new_json(record, record_path, HERE)
            report = replay(json.loads(record_path.read_text()), lock, old_lock)
            save_new_json(report, output / f"{row['arm_id']}_EXACT_REPLAY.json", HERE)
            reports.append(report)
            summaries.append({"arm_id": row["arm_id"], "ranks": old_arm["ranks"],
                              "graph_metadata": record["graph_metadata"], "diagnostics": record["diagnostics"],
                              "identical_members_null_checks": record["identical_members_null_checks"]})
        save_new_json({"status": "SIX_CELL_OBJECTIVE_COLLECTION_AND_ARITHMETIC_REPLAY_PASS",
                       "lock_sha256": lock["lock_sha256"], "reports": reports, "summaries": summaries,
                       "seconds": time.perf_counter() - begin, "training_performed": False,
                       "validation_or_test_scored": False, "promotion_rule": False}, output / "RESULT.json", HERE)
        print(json.dumps({"status": "SIX_CELL_OBJECTIVE_COLLECTION_AND_ARITHMETIC_REPLAY_PASS", "output": str(output)}))
    except Exception as error:
        save_new_json({"status": "FAIL", "type": type(error).__name__, "message": str(error),
                       "seconds": time.perf_counter() - begin}, output / "FAILURE.json", HERE)
        raise


if __name__ == "__main__":
    main()
