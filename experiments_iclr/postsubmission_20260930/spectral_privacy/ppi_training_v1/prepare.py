"""Stdlib-only prospective freeze; reads no tensors or predictive scores."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
SCHEMA = "gnnm-ppi-frozen-basis-predictive-v1"
ARMS = ("tied", "selected", "random", "raw")
RECIPE = {"members": 4, "width": 128, "layers": 3, "dropout": 0.2,
          "epochs": 100, "training_graphs": 20, "steps": 2000,
          "lr": 0.001, "weight_decay": 0.0, "eps": 1e-8, "betas": [0.9, 0.999],
          "dtype": "float32", "amp": False, "threads": 1, "interop_threads": 1,
          "warm_start_epoch": 25, "warm_start_graph_steps": 250,
          "loss": "mean member observed-entry BCE, one complete graph per update",
          "pooling": "mean raw member logits; threshold0",
          "order_seed": "19709+1000*seed+epoch",
          "dropout_seed": "67001+10000*seed+20*epoch+official_graph_index",
          "selection": "max validation micro_f1; min pooled_bce; earliest epoch; epoch0 eligible"}
SOURCE_FILES = ["spectral_privacy/ppi_training_v1/prepare.py", "spectral_privacy/ppi_training_v1/runner.py",
                "spectral_privacy/ppi_training_v1/summarize.py",
                "spectral_privacy/ppi_training_v1/PROTOCOL.md", "protocols/baseline_adapters.py",
                "source_snapshot/models.py", "new_tasks/shareable_guard.py", "new_tasks/shareable_freeze.py",
                "spectral_privacy/ppi_integration_v1/collect_ppi.py",
                "spectral_privacy/ppi_integration_v1/freeze_inputs.py",
                "spectral_privacy/ppi_f2_audit_v1/freeze_audit.py"]
SOURCE_FILES += ["spectral_privacy/v2_abi_fix_01/" + name for name in
                 ("adapters.py", "contracts.py", "collector.py", "spectral_core.py", "replay.py")]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def inside(path):
    p = Path(path).resolve()
    if PHASE not in p.parents: raise ValueError("All artifacts must be inside the authorized phase")
    return p


def ref(path):
    p = inside(path)
    return {"path": str(p.relative_to(PHASE)), "sha256": sha(p)}


def checked(value):
    p = inside(PHASE / value["path"])
    if sha(p) != value["sha256"]: raise ValueError("Changed artifact: " + value["path"])
    return p


def source_hashes():
    return {p: sha(inside(PHASE / p)) for p in SOURCE_FILES}


def seal(value):
    return value | {"protocol_sha256": json_hash(value)}


def verify_seal(value):
    if value["protocol_sha256"] != json_hash({k:v for k,v in value.items() if k != "protocol_sha256"}):
        raise ValueError("Protocol seal differs")


def write_new(path, value):
    p = inside(path)
    if HERE not in p.parents: raise ValueError("New outputs must stay inside this new study directory")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as f: json.dump(value, f, sort_keys=True, indent=2, allow_nan=False); f.write("\n")


def build():
    audit_ref = ref(PHASE / "spectral_privacy/ppi_f2_audit_v1/AUDIT_INPUT_LOCK.json")
    audit = json.loads(checked(audit_ref).read_text())
    if audit["audit_lock_sha256"] != json_hash({k:v for k,v in audit.items() if k != "audit_lock_sha256"}):
        raise ValueError("Existing audit input seal differs")
    if [(c["seed"],c["label_fraction"]) for c in audit["arms"]] != [(s,f) for s in (0,1,2) for f in (0.1,1.0)]:
        raise ValueError("Require exactly six locked cells")
    for c in audit["arms"]:
        for key in ("checkpoint", "label_masks", "contract", "DE_matrix_record", "DE_basis_lock", "DE_replay_report"):
            checked(c[key])
    inputs = json.loads(checked(audit["integration_input_lock"]).read_text())
    data = ref(PHASE / "new_tasks/data/ppi/frozen_shareable_manifest_v2.json")
    manifest = json.loads(checked(data).read_text())
    if manifest["manifest_amendment"]["parent_manifest_sha256"] != inputs["data_manifest"]["sha256"]:
        raise ValueError("Shareable data freeze does not bind the same original PPI inputs")
    costs = []
    for c in audit["arms"]:
        folder = checked(c["checkpoint"]).parent
        trace = ref(folder / "fit_trace.csv")
        with checked(trace).open() as f: rows = list(csv.DictReader(f))
        if len(rows) != 25 or any(int(r["fit_steps"]) != 10 for r in rows): raise ValueError("Probe budget differs")
        costs.append({"cell": c["arm_id"], "probe_fit_trace": trace,
                      "probe_fit_seconds": sum(float(r["fit_epoch_seconds"]) for r in rows),
                      "probe_steps": 250, "shared_across_four_arms": True})
    past = {name: ref(PHASE / path) for name,path in {
        "DE_collection_and_replay": "spectral_privacy/ppi_integration_v1/collection_run01/RESULT.json",
        "F2_audit_and_replay": "spectral_privacy/ppi_f2_audit_v1/audit_run01/RESULT.json",
        "F2_separate_replay": "spectral_privacy/ppi_f2_audit_v1/replay_run01/RESULT.json",
        "adverse_F2_summary": "spectral_privacy/ppi_f2_audit_v1/F2_SUMMARY_v1.md",
        "probe_reproduction": "sharing_policy/PPI_REPOENV_REPRO_COMPARISON_v1.json"}.items()}
    return seal({"schema": SCHEMA, "status": "prospective_no_predictive_fit", "recipe": RECIPE,
                 "source_sha256": source_hashes(), "audit_input_lock": audit_ref,
                 "integration_input_lock": audit["integration_input_lock"], "cells": audit["arms"],
                 "arms": list(ARMS), "data_manifest": data, "prior_evidence": past,
                 "prior_probe_costs": costs, "primary_label_fraction": 0.1,
                 "mandatory_adverse_control_fraction": 1.0,
                 "continuation": {"all_24_fits_and_replays": True, "min_mean_micro_f1_gain_each_control": 0.005,
                                  "both_nonzero_rank_primary_seeds_positive_each_control": True,
                                  "minimum_paired_micro_f1_difference": -0.002,
                                  "mean_pooled_bce_no_worse_each_control": True,
                                  "late_selected_epoch": 90, "late_windows": [[91,95],[96,100]],
                                  "late_mean_f1_improvement": 0.0001, "late_mean_pooled_bce_drop": 0.00001,
                                  "any_late_still_improving_primary_arm_blocks_promotion": True,
                                  "independent_new_task_confirmation_required": True},
                 "test_access_planned": False, "preparation_is_predictive_evidence": False})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--output",type=Path,required=True)
    args = p.parse_args(); result = build(); write_new(args.output,result)
    print(json.dumps({"status":result["status"],"protocol_sha256":result["protocol_sha256"]}))
