"""Standard-library prospective freeze of fixed bases before F2 access."""

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
INTEGRATION = PHASE / "spectral_privacy/ppi_integration_v1"
CODE_FILES = ("freeze_audit.py", "audit_f2.py", "PROTOCOL.md", "WARM_START_OUTLINE.md")
LOCK_SCHEMA = "spectral-ppi-fixed-F2-audit-lock-v1"
TARGETS = tuple(f"residual_modules.{i}.module.conv.lin_l.weight" for i in range(3))
GROUPS = ("selected", "random", "raw_gradient")


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase_path(path):
    path = Path(path).resolve()
    if PHASE not in path.parents:
        raise ValueError("All inputs and outputs must remain in the research phase")
    return path


def ref(path):
    path = phase_path(path)
    return {"path": str(path.relative_to(PHASE)), "sha256": sha(path)}


def checked(value):
    path = phase_path(PHASE / value["path"])
    if sha(path) != value["sha256"]:
        raise ValueError("Frozen artifact bytes differ: " + value["path"])
    return path


def verified_seal(value, field):
    if value[field] != json_hash({key: row for key, row in value.items() if key != field}):
        raise ValueError("Artifact content seal differs: " + field)


def fixed_bases(selection):
    return {"selected": selection["basis"],
            "random": selection["controls"]["random_matched_rank_basis"],
            "raw_gradient": selection["controls"]["raw_centered_gradient_matched_rank_basis"]}


def build_lock():
    input_ref = ref(INTEGRATION / "INPUT_LOCK.json")
    inputs = json.loads(checked(input_ref).read_text())
    verified_seal(inputs, "lock_sha256")
    gate_ref = ref(INTEGRATION / "precollection_gate_run01/PRECOLLECTION_GATE.json")
    gate = json.loads(checked(gate_ref).read_text())
    verified_seal(gate, "gate_sha256")
    result_ref = ref(INTEGRATION / "collection_run01/RESULT.json")
    result = json.loads(checked(result_ref).read_text())
    if (gate["status"] != "PRECOLLECTION_GATE_PASS"
            or result["status"] != "SIX_ARM_COLLECTION_AND_EXACT_REPLAY_PASS"
            or gate["input_lock_sha256"] != inputs["lock_sha256"]
            or result["input_lock_sha256"] != inputs["lock_sha256"]
            or result["gate_sha256"] != gate["gate_sha256"]
            or result["F2_opened"] is not False
            or result["validation_or_test_scored"] is not False
            or inputs["partition_ids"]["audit"] != ["ppi:train:09", "ppi:train:18"]):
        raise ValueError("Require completed six-cell D/E collection with unused F2")
    arms = []
    for arm in inputs["arms"]:
        name = arm["arm_id"]
        directory = INTEGRATION / "collection_run01"
        record_ref = ref(directory / f"{name}_MATRIX_RECORD.json")
        basis_ref = ref(directory / f"{name}_BASIS_LOCK.json")
        replay_ref = ref(directory / f"{name}_EXACT_REPLAY.json")
        record = json.loads(checked(record_ref).read_text())
        basis = json.loads(checked(basis_ref).read_text())
        replay = json.loads(checked(replay_ref).read_text())
        verified_seal(record, "record_sha256")
        identity = next(row for row in gate["arms"] if row["arm_id"] == name)
        if (basis["status"] != "SPECTRAL_BASES_LOCKED" or basis["F2_opened"] is not False
                or basis["selection"] != record["selection"]
                or basis["record_file_sha256"] != record_ref["sha256"]
                or basis["record_sha256"] != record["record_sha256"]
                or replay["status"] != "EXACT_REPLAY_PASS"
                or replay["record_sha256"] != record["record_sha256"]
                or record["model_state_sha256"] != identity["adapter_state_sha256"]
                or record["selection"]["rule"]["n0"] != 1
                or record["selection"]["rule"]["rank_max_per_target"] != 16):
            raise ValueError("Existing basis/source/replay lock differs: " + name)
        packets = fixed_bases(record["selection"])
        ranks = record["selection"]["ranks"]
        for group in GROUPS:
            if set(packets[group]) != set(TARGETS):
                raise ValueError("Fixed basis target names differ")
            for key in TARGETS:
                if packets[group][key]["shape"] != [128, ranks[key]]:
                    raise ValueError("Fixed controls do not match selected ranks")
        arms.append({"arm_id": name, "seed": arm["seed"], "label_fraction": arm["label_fraction"],
                     "DE_matrix_record": record_ref, "DE_basis_lock": basis_ref,
                     "DE_replay_report": replay_ref, "fixed_bases_sha256": json_hash(packets),
                     "basis_tensor_sha256": {group: {key: value["sha256"] for key, value in rows.items()}
                                              for group, rows in packets.items()},
                     "ranks": ranks, "checkpoint": arm["checkpoint"],
                     "label_masks": arm["label_masks"], "contract": arm["contract"],
                     "original_source_state_sha256": identity["original_source_state_sha256"],
                     "adapter_state_sha256": identity["adapter_state_sha256"]})
    preserved = {str(path.relative_to(PHASE)): sha(path)
                 for path in sorted(INTEGRATION.rglob("*")) if path.is_file()}
    core = PHASE / "spectral_privacy/v2_abi_fix_01"
    manifest = json.loads((core / "PROTOTYPE_MANIFEST.json").read_text())
    for name, digest in manifest["files_sha256"].items():
        if sha(core / name) != digest:
            raise ValueError("Frozen spectral implementation changed")
    return {"schema": LOCK_SCHEMA, "integration_input_lock": input_ref,
            "DE_precollection_gate": gate_ref, "DE_result": result_ref,
            "preserved_integration_files_sha256": preserved,
            "spectral_core_manifest": ref(core / "PROTOTYPE_MANIFEST.json"),
            "audit_code_sha256": {name: sha(HERE / name) for name in CODE_FILES},
            "arms": arms, "F2_unit_ids": ["ppi:train:09", "ppi:train:18"],
            "graph_weighting": "equal complete graph", "n0": 1,
            "fixed_statistic": "tr(U.T @ T_F @ U)",
            "basis_groups": list(GROUPS), "ranks_may_change": False,
            "runtime": {"device": "cpu", "threads": 1, "interop_threads": 1},
            "F2_gradients_collected": False, "validation_or_test_scored": False,
            "CI_or_pvalue_planned": False, "selected_score_unbiasedness_claimed": False,
            "AdamW_benefit_or_accuracy_claimed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = build_lock()
    lock["audit_lock_sha256"] = json_hash(lock)
    path = phase_path(args.output)
    if HERE not in path.parents:
        raise ValueError("Audit lock must be inside the audit sibling directory")
    with path.open("x") as stream:
        json.dump(lock, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "FIXED_F2_AUDIT_PROSPECTIVELY_LOCKED", "file_sha256": sha(path),
                      "audit_lock_sha256": lock["audit_lock_sha256"], "F2_gradients_collected": False}))


if __name__ == "__main__":
    main()
