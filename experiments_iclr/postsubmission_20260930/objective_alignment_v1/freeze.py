"""Freeze a separate F2 objective diagnostic; inherited checkpoints checked at runtime."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
CODE_FILES = ("objective_core.py", "collect_objectives.py", "verify_cpu.py", "freeze.py", "PROTOCOL.md")
LOCK_SCHEMA = "gnnm-F2-objective-alignment-input-lock-v1"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def phase_path(path):
    path = Path(path).resolve()
    if PHASE not in path.parents:
        raise ValueError("Input must stay inside the authorized research phase")
    return path


def ref(path):
    path = phase_path(path)
    return {"path": str(path.relative_to(PHASE)), "sha256": sha(path)}


def checked(value):
    path = phase_path(PHASE / value["path"])
    if sha(path) != value["sha256"]:
        raise ValueError("Source/evidence bytes changed: " + value["path"])
    return path


def verify_seal(value, field):
    if value[field] != json_hash({key: row for key, row in value.items() if key != field}):
        raise ValueError("Content seal differs: " + field)


def code_hashes():
    return {name: sha(HERE / name) for name in CODE_FILES}


def build_lock():
    old = PHASE / "spectral_privacy/ppi_f2_audit_v1"
    audit_ref = ref(old / "AUDIT_INPUT_LOCK.json")
    audit = json.loads(checked(audit_ref).read_text())
    verify_seal(audit, "audit_lock_sha256")
    if audit["F2_unit_ids"] != ["ppi:train:09", "ppi:train:18"]:
        raise ValueError("Use only the existing audited two training graphs")
    old_gate_ref = ref(old / "audit_run01/F2_PREGRADIENT_GATE.json")
    old_gate = json.loads(checked(old_gate_ref).read_text())
    verify_seal(old_gate, "gate_sha256")
    if old_gate["status"] != "F2_SOURCE_INPUT_BASES_PREGRADIENT_GATE_PASS":
        raise ValueError("Prior F2 source/mask gate must have passed")
    arms = []
    for arm in audit["arms"]:
        identity = next(row for row in old_gate["arms"] if row["arm_id"] == arm["arm_id"])
        record_ref = ref(old / "audit_run01" / f"{arm['arm_id']}_F2_RECORD.json")
        record = json.loads(checked(record_ref).read_text())
        verify_seal(record, "record_sha256")
        replay_ref = ref(old / "audit_run01" / f"{arm['arm_id']}_F2_EXACT_REPLAY.json")
        replay = json.loads(checked(replay_ref).read_text())
        if (replay["status"] != "EXACT_F2_REPLAY_PASS" or replay["record_sha256"] != record["record_sha256"]
                or record["unit_ids"] != audit["F2_unit_ids"]
                or record["input_fingerprints"] != identity["F2_input_fingerprints"]
                or record["model_state_sha256"] != arm["adapter_state_sha256"]):
            raise ValueError("Previously audited F2 inputs/source differ")
        arms.append({"arm_id": arm["arm_id"], "previous_F2_record": record_ref,
                     "previous_F2_replay": replay_ref,
                     "F2_input_fingerprints": identity["F2_input_fingerprints"],
                     "original_source_state_sha256": arm["original_source_state_sha256"],
                     "fixed_bases_sha256": arm["fixed_bases_sha256"], "ranks": arm["ranks"]})
    core = PHASE / "spectral_privacy/v2_abi_fix_01"
    dependencies = [core / name for name in ("adapters.py", "collector.py", "contracts.py", "spectral_core.py", "replay.py")]
    dependencies += [old / name for name in ("audit_f2.py", "freeze_audit.py")]
    dependencies += [PHASE / "source_snapshot/models.py"]
    return {"schema": LOCK_SCHEMA, "previous_F2_audit_lock": audit_ref,
            "previous_F2_pregradient_gate": old_gate_ref,
            "source_dependencies": [ref(path) for path in dependencies],
            "code_sha256": code_hashes(), "arms": arms,
            "unit_ids": audit["F2_unit_ids"], "epoch": 25,
            "runtime": {"device": "cpu", "threads": 1, "interop_threads": 1},
            "numeric_rule": {"forward_and_pooling": "native float32", "loss_and_diagnostic_reductions": "float64"},
            "observed_masks": "unchanged nested10/100 percent original node masks; hidden targets physically NaN",
            "graphs_equal_weight": True, "basis_or_rank_changes": False,
            "unused_F2_claimed": False, "diagnostic_hypothesis_only": True,
            "validation_or_test_access": False, "promotion_rule": False,
            "runtime_checkpoint_byte_check_required": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build_lock()
    value["lock_sha256"] = json_hash(value)
    output = phase_path(args.output)
    if HERE not in output.parents:
        raise ValueError("New lock must remain inside this diagnostic folder")
    with output.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "OBJECTIVE_DIAGNOSTIC_LOCKED", "sha256": sha(output),
                      "lock_sha256": value["lock_sha256"], "runtime_checkpoint_checks_pending": True}))


if __name__ == "__main__":
    main()
