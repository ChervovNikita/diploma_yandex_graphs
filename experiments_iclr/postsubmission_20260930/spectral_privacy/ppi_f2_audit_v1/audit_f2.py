"""Fixed-projector F2 denoising-proxy audit and exact replay; no training."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
sys.path.insert(0, str(PHASE))
sys.path.insert(0, str(PHASE / "spectral_privacy/ppi_integration_v1"))

import torch

from freeze_audit import (CODE_FILES, GROUPS, LOCK_SCHEMA, TARGETS, checked,
                          fixed_bases, json_hash, phase_path, ref, sha, verified_seal)
from collect_ppi import (checked_masks, checked_source, legacy_tensor_fingerprint,
                         load_module, load_train, validate_lock as validate_integration)
from spectral_privacy.v2_abi_fix_01.collector import (GradientUnit, LOSS_RULE, PreservedProbe,
                                                    environment_contract, observed_bce,
                                                    unit_fingerprint)
from spectral_privacy.v2_abi_fix_01.contracts import (code_hashes, pack_float64, save_new_json,
                                                    state_hash, unpack_float64)
from spectral_privacy.v2_abi_fix_01.replay import replay_record as replay_DE
from spectral_privacy.v2_abi_fix_01.spectral_core import STAT_KEYS, output_statistics


RECORD_SCHEMA = "spectral-ppi-fixed-F2-matrix-record-v1"


def sealed(value, field):
    result = {key: row for key, row in value.items() if key != field}
    return result | {field: json_hash(result)}


def audit_code():
    return {name: sha(HERE / name) for name in CODE_FILES}


def validate_audit(path):
    lock = json.loads(phase_path(path).read_text())
    verified_seal(lock, "audit_lock_sha256")
    if (lock["schema"] != LOCK_SCHEMA or lock["n0"] != 1
            or lock["F2_unit_ids"] != ["ppi:train:09", "ppi:train:18"]
            or lock["fixed_statistic"] != "tr(U.T @ T_F @ U)"
            or lock["basis_groups"] != list(GROUPS) or lock["ranks_may_change"] is not False
            or lock["audit_code_sha256"] != audit_code()
            or lock["runtime"] != {"device": "cpu", "threads": 1, "interop_threads": 1}):
        raise ValueError("Prospective F2 audit method/code identity differs")
    for name, digest in lock["preserved_integration_files_sha256"].items():
        if sha(phase_path(PHASE / name)) != digest:
            raise ValueError("Prior source/lock/result bytes changed: " + name)
    checked(lock["spectral_core_manifest"])
    inputs = validate_integration(checked(lock["integration_input_lock"]))
    for arm in lock["arms"]:
        for key in ("DE_matrix_record", "DE_basis_lock", "DE_replay_report", "checkpoint", "label_masks", "contract"):
            checked(arm[key])
        record = json.loads(checked(arm["DE_matrix_record"]).read_text())
        if json_hash(fixed_bases(record["selection"])) != arm["fixed_bases_sha256"]:
            raise ValueError("Fixed selected/random/raw basis bytes changed")
    return lock, inputs


def decode_fixed(arm):
    record = json.loads(checked(arm["DE_matrix_record"]).read_text())
    replay_DE(record, expected_probe_state_sha256=arm["adapter_state_sha256"])
    packets = fixed_bases(record["selection"])
    if json_hash(packets) != arm["fixed_bases_sha256"]:
        raise ValueError("Saved fixed basis packets differ from audit freeze")
    bases = {group: {key: unpack_float64(payload) for key, payload in rows.items()}
             for group, rows in packets.items()}
    for rows in bases.values():
        for key, basis in rows.items():
            rank = arm["ranks"][key]
            if basis.shape != (128, rank) or not torch.allclose(
                    basis.T @ basis, torch.eye(rank, dtype=torch.float64), atol=1e-10, rtol=1e-10):
                raise ValueError("Fixed basis shape or orthonormality differs")
    return record, packets, bases


def F2_units(dataset, masks, ids, metadata):
    rows = metadata["split_summaries"]["train"]["graph_tensor_fingerprints"]
    units = []
    for name in ids:
        index = int(name.rsplit(":", 1)[1])
        graph = dataset[index]
        if (graph.y.shape != (graph.num_nodes, 121)
                or legacy_tensor_fingerprint(graph.y) != rows[index]["y"]
                or not bool(torch.isfinite(graph.y).all())
                or not bool(((graph.y == 0) | (graph.y == 1)).all())):
            raise ValueError("F2 training target bytes/shape differ from frozen provider")
        safe = graph.clone()
        safe.y = graph.y.clone()
        safe.y[~masks[index]] = float("nan")
        units.append(GradientUnit(name, safe, safe.x, safe.y, masks[index], split="train"))
    return units


def derive(raw, bases):
    stats = {key: output_statistics(raw[key], n0=1) for key in TARGETS}
    score_tensors, scores = {}, {}
    for group in GROUPS:
        rows, total = {}, []
        for key in TARGETS:
            u = bases[group][key]
            trace = {name: torch.trace(u.T @ stats[key][name] @ u) for name in ("S", "N", "T")}
            rows[key] = {"rank": u.shape[1], "trace_S": pack_float64(trace["S"]),
                         "trace_N": pack_float64(trace["N"]), "trace_T": pack_float64(trace["T"]),
                         "T_per_direction": pack_float64(torch.diag(u.T @ stats[key]["T"] @ u))}
            total.append(trace["T"])
        summed = torch.stack(total).sum()
        score_tensors[group] = summed
        scores[group] = {"targets": rows, "sum_trace_T": pack_float64(summed)}
    differences = {"selected_minus_random_sum_trace_T": pack_float64(score_tensors["selected"] - score_tensors["random"]),
                   "selected_minus_raw_gradient_sum_trace_T": pack_float64(score_tensors["selected"] - score_tensors["raw_gradient"])}
    return {"statistics": {key: {name: pack_float64(stats[key][name]) for name in STAT_KEYS} for key in TARGETS},
            "fixed_basis_scores": scores, "descriptive_differences": differences}


def collect_F2(probe, units, arm, lock, DE_record, packets, bases, gate_ref):
    if [unit.unit_id for unit in units] != lock["F2_unit_ids"] or any(unit.split != "train" for unit in units):
        raise ValueError("Supply exactly the two frozen training F2 units")
    if not probe.is_tied_at_state() or probe.members != 4 or state_hash(probe) != arm["adapter_state_sha256"]:
        raise ValueError("F2 gradients require the unchanged tied P25 state")
    targets = probe.named_neighbor_matrices()
    if set(targets) != set(TARGETS) or any(value.shape != (128, 128) for value in targets.values()):
        raise ValueError("Frozen source neighbor targets differ")
    before_inputs = [unit_fingerprint(unit) for unit in units]
    snapshot = PreservedProbe(probe)
    raw = {key: [] for key in TARGETS}
    graph_metadata = {}
    try:
        probe.eval()
        with torch.enable_grad():
            for unit in units:
                gradients_by_target, losses, counts = {key: [] for key in TARGETS}, [], []
                for member in range(4):
                    loss, count = observed_bce(probe.forward_member(unit.graph, unit.features, member), unit)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("Nonfinite F2 own-member graph loss")
                    gradients = torch.autograd.grad(loss, tuple(targets[key] for key in TARGETS), allow_unused=False)
                    for key, value in zip(TARGETS, gradients):
                        value = value.detach().cpu().double().contiguous()
                        if not bool(torch.isfinite(value).all()):
                            raise ValueError("Nonfinite F2 source gradient")
                        gradients_by_target[key].append(value)
                    losses.append(float(loss.detach()))
                    counts.append(count)
                if len(set(counts)) != 1:
                    raise ValueError("F2 members used different observed-target counts")
                graph_metadata[unit.unit_id] = {"observed_entries": counts[0], "member_loss": losses}
                for key in TARGETS:
                    raw[key].append(torch.stack(gradients_by_target[key]))
        raw = {key: torch.stack(value) for key, value in raw.items()}
        derived = derive(raw, bases)
        if not snapshot.matches() or before_inputs != [unit_fingerprint(unit) for unit in units]:
            raise RuntimeError("F2 collection mutated source/grad/RNG or training units")
        return sealed({"schema": RECORD_SCHEMA, "arm_id": arm["arm_id"],
                       "audit_lock_sha256": lock["audit_lock_sha256"], "F2_pregradient_gate": gate_ref,
                       "source_models_sha256": DE_record["source_models_sha256"],
                       "original_source_state_sha256": arm["original_source_state_sha256"],
                       "model_state_sha256": snapshot.state_sha256, "checkpoint": arm["checkpoint"],
                       "epoch": 25, "members": 4, "n0": 1,
                       "loss_rule": LOSS_RULE, "gradient_scale": "unscaled own-member own-graph mean loss",
                       "fold_weighting": "equal_complete_graph", "unit_ids": lock["F2_unit_ids"],
                       "input_fingerprints": before_inputs, "input_identity_sha256": json_hash(before_inputs),
                       "targets": DE_record["targets"], "graph_metadata": graph_metadata,
                       "fixed_bases": packets, "fixed_bases_sha256": arm["fixed_bases_sha256"],
                       "matrix_gradients": {key: pack_float64(value) for key, value in raw.items()},
                       "derived": derived, "environment": environment_contract(),
                       "spectral_code_sha256": code_hashes(), "audit_code_sha256": audit_code(),
                       "state_grads_modes_and_rng_preserved": True,
                       "selected_score_unbiasedness_claimed": False,
                       "CI_or_pvalue_claimed": False, "accuracy_or_AdamW_benefit_claimed": False}, "record_sha256")
    finally:
        snapshot.restore()


def replay_F2(record, lock):
    verified_seal(record, "record_sha256")
    arm = next(row for row in lock["arms"] if row["arm_id"] == record["arm_id"])
    gate = json.loads(checked(record["F2_pregradient_gate"]).read_text())
    verified_seal(gate, "gate_sha256")
    identity = next(row for row in gate["arms"] if row["arm_id"] == arm["arm_id"])
    if (record["schema"] != RECORD_SCHEMA or record["audit_lock_sha256"] != lock["audit_lock_sha256"]
            or record["model_state_sha256"] != arm["adapter_state_sha256"]
            or record["original_source_state_sha256"] != arm["original_source_state_sha256"]
            or record["checkpoint"] != arm["checkpoint"]
            or record["unit_ids"] != lock["F2_unit_ids"] or record["epoch"] != 25
            or record["members"] != 4 or record["n0"] != 1
            or record["loss_rule"] != LOSS_RULE or record["fold_weighting"] != "equal_complete_graph"
            or record["gradient_scale"] != "unscaled own-member own-graph mean loss"
            or record["environment"] != environment_contract()
            or record["spectral_code_sha256"] != code_hashes() or record["audit_code_sha256"] != audit_code()
            or record["state_grads_modes_and_rng_preserved"] is not True
            or gate["audit_lock_sha256"] != lock["audit_lock_sha256"]
            or record["input_fingerprints"] != identity["F2_input_fingerprints"]):
        raise ValueError("F2 source/input/scale/code/environment identity differs")
    if ([row["unit_id"] for row in record["input_fingerprints"]] != lock["F2_unit_ids"]
            or any(row["split"] != "train" for row in record["input_fingerprints"])
            or json_hash(record["input_fingerprints"]) != record["input_identity_sha256"]
            or set(record["graph_metadata"]) != set(lock["F2_unit_ids"])):
        raise ValueError("F2 training unit identity differs")
    DE_record, packets, bases = decode_fixed(arm)
    if (record["source_models_sha256"] != DE_record["source_models_sha256"]
            or record["targets"] != DE_record["targets"] or record["fixed_bases"] != packets
            or record["fixed_bases_sha256"] != arm["fixed_bases_sha256"]):
        raise ValueError("F2 fixed source/basis targets differ from the D/E lock")
    if set(record["matrix_gradients"]) != set(TARGETS):
        raise ValueError("F2 raw target names differ")
    raw = {key: unpack_float64(value) for key, value in record["matrix_gradients"].items()}
    if any(value.shape != (2, 4, 128, 128) for value in raw.values()):
        raise ValueError("F2 raw evidence dimensions differ")
    if json_hash(derive(raw, bases)) != json_hash(record["derived"]):
        raise ValueError("F2 covariance/projections do not replay byte-exactly")
    return {"status": "EXACT_F2_REPLAY_PASS", "arm_id": arm["arm_id"],
            "record_sha256": record["record_sha256"], "fixed_bases_sha256": arm["fixed_bases_sha256"],
            "input_identity_sha256": record["input_identity_sha256"], "ranks": arm["ranks"],
            "source_state_preserved": True, "selected_score_unbiasedness_claimed": False}


def summary(record, arm):
    derived = record["derived"]
    return {"arm_id": arm["arm_id"], "seed": arm["seed"], "label_fraction": arm["label_fraction"],
            "ranks": arm["ranks"], "scores": {group: {
                "sum_trace_T": float(unpack_float64(rows["sum_trace_T"])),
                "targets": {key: {name: float(unpack_float64(values[name])) for name in ("trace_S", "trace_N", "trace_T")}
                            for key, values in rows["targets"].items()}}
                for group, rows in derived["fixed_basis_scores"].items()},
            "descriptive_differences": {name: float(unpack_float64(value)) for name, value in derived["descriptive_differences"].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "replay"))
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--run", type=Path, help="Existing audit run, for label-free replay only")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = phase_path(args.output)
    if HERE not in output.parents or output.exists():
        raise ValueError("Use a new audit output directory")
    output.mkdir()
    begin = time.perf_counter()
    try:
        lock, inputs = validate_audit(args.lock)
        if args.command == "replay":
            if args.run is None:
                raise ValueError("Label-free replay requires --run")
            reports = []
            for arm in lock["arms"]:
                record = json.loads((phase_path(args.run) / f"{arm['arm_id']}_F2_RECORD.json").read_text())
                reports.append(replay_F2(record, lock))
            save_new_json({"status": "SIX_CELL_EXACT_F2_REPLAY_PASS", "reports": reports}, output / "RESULT.json", HERE)
            print(json.dumps({"status": "SIX_CELL_EXACT_F2_REPLAY_PASS", "output": str(output)}))
            return
        if args.run is not None:
            raise ValueError("New collection must not receive a previous --run")
        source_module = load_module(checked(inputs["probe_sources"]["source_snapshot/models.py"]), "frozen_F2_source")
        baseline = load_module(checked(inputs["probe_sources"]["protocols/baseline_adapters.py"]), "frozen_F2_baseline")
        dataset, _ = load_train(inputs)
        metadata = json.loads(checked(inputs["data_manifest"]).read_text())
        prepared, identities = [], []
        for arm in lock["arms"]:
            original_arm = next(row for row in inputs["arms"] if row["arm_id"] == arm["arm_id"])
            contract = json.loads(checked(arm["contract"]).read_text())
            masks, _ = checked_masks(original_arm, contract, dataset)
            units = F2_units(dataset, masks, lock["F2_unit_ids"], metadata)
            probe, identity = checked_source(original_arm, contract, source_module, baseline, units[0])
            if (identity["adapter_state_sha256"] != arm["adapter_state_sha256"]
                    or identity["original_source_state_sha256"] != arm["original_source_state_sha256"]):
                raise ValueError("F2 source differs from the locked D/E source state")
            DE_record, packets, bases = decode_fixed(arm)
            identities.append(identity | {"F2_input_fingerprints": [unit_fingerprint(unit) for unit in units],
                                           "fixed_bases_sha256": arm["fixed_bases_sha256"]})
            prepared.append((arm, probe, units, DE_record, packets, bases))
        gate = sealed({"status": "F2_SOURCE_INPUT_BASES_PREGRADIENT_GATE_PASS",
                       "audit_lock_sha256": lock["audit_lock_sha256"], "arms": identities,
                       "environment": environment_contract(), "F2_gradients_collected": False}, "gate_sha256")
        gate_path = output / "F2_PREGRADIENT_GATE.json"
        save_new_json(gate, gate_path, HERE)
        reports, summaries = [], []
        for arm, probe, units, DE_record, packets, bases in prepared:
            record = collect_F2(probe, units, arm, lock, DE_record, packets, bases, ref(gate_path))
            path = output / f"{arm['arm_id']}_F2_RECORD.json"
            save_new_json(record, path, HERE)
            replay = replay_F2(json.loads(path.read_text()), lock)
            save_new_json(replay, output / f"{arm['arm_id']}_F2_EXACT_REPLAY.json", HERE)
            reports.append(replay)
            summaries.append(summary(record, arm))
        save_new_json({"status": "SIX_CELL_FIXED_F2_AUDIT_AND_EXACT_REPLAY_PASS",
                       "audit_lock_sha256": lock["audit_lock_sha256"], "reports": reports,
                       "summaries": summaries, "seconds": time.perf_counter() - begin,
                       "validation_or_test_scored": False, "training_performed": False,
                       "CI_or_pvalue_claimed": False, "selected_score_unbiasedness_claimed": False,
                       "accuracy_or_AdamW_benefit_claimed": False}, output / "RESULT.json", HERE)
        print(json.dumps({"status": "SIX_CELL_FIXED_F2_AUDIT_AND_EXACT_REPLAY_PASS", "output": str(output)}))
    except Exception as error:
        save_new_json({"status": "FAIL", "type": type(error).__name__, "message": str(error),
                       "seconds": time.perf_counter() - begin}, output / "FAILURE.json", HERE)
        raise


if __name__ == "__main__":
    main()
