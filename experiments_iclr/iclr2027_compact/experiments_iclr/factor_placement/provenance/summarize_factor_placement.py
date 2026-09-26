"""Post-lock, post-audit descriptive all-layer versus same-runtime TIED table."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ALL = ROOT / "all_layer_factor_results"
CONTROL = ROOT / "same_runtime_tied36_results"
PRIMARY_LOCK_SHA = "176c68b855835a625086a1287a35e76cff43ecf144e7dbe19297ebb2d3ef6f99"
PRIMARY_SCORE_AUDIT_SHA = "63da49d3d797fa7856717d4b21a530c74419bfbdfa92614a03b036aa9950bdde"
DATASETS = ("cora", "wikics", "actor", "chameleon_filtered")
DEFAULT = (0.001, 0.0)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def candidate(lr, wd):
    return f"lr{lr:g}_wd{wd:g}"


def key(dataset, arm, lr, wd, seed):
    return f"{dataset}/{arm}/{candidate(lr, wd)}/seed{seed}" if arm else \
        f"{dataset}/{candidate(lr, wd)}/seed{seed}"


def read_score(base, identity, expected_hash):
    path = base / "scores" / identity / "score.json"
    require(path.is_file() and sha(path) == expected_hash,
            f"Score absent or differs from final independent audit: {identity}")
    return json.loads(path.read_text())


def summarize(values):
    require(len(values) == 3, "Expected three paired seeds")
    return {"seed_values": values, "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values)}


def main():
    output = ALL / "PLACEMENT_COMPARISON.json"
    table = ALL / "PLACEMENT_COMPARISON.csv"
    require(not output.exists() and not table.exists(), "Refusing to overwrite summary")
    primary_lock_path = ROOT / "VALIDATION_SELECTION_LOCK.json"
    primary_audit_path = ROOT / "FINAL_SCORE_AUDIT.json"
    all_lock_path = ALL / "VALIDATION_SELECTION_LOCK.json"
    all_audit_path = ALL / "FINAL_SCORE_AUDIT.json"
    control_lock_path = CONTROL / "VALIDATION_SELECTION_LOCK.json"
    control_audit_path = CONTROL / "FINAL_SCORE_AUDIT.json"
    require(sha(primary_lock_path) == PRIMARY_LOCK_SHA and
            sha(primary_audit_path) == PRIMARY_SCORE_AUDIT_SHA,
            "Primary 432-cell lock/final score audit differs")
    primary_lock = json.loads(primary_lock_path.read_text())
    primary_audit = json.loads(primary_audit_path.read_text())
    all_lock = json.loads(all_lock_path.read_text())
    all_audit = json.loads(all_audit_path.read_text())
    control_lock = json.loads(control_lock_path.read_text())
    control_audit = json.loads(control_audit_path.read_text())
    require(len(primary_lock["cells"]) == 432 and len(primary_audit["scores"]) == 141 and
            len(all_lock["cells"]) == 72 and len(control_lock["cells"]) == 36 and
            all_audit["validation_selection_lock_sha256"] == sha(all_lock_path) and
            control_audit["validation_selection_lock_sha256"] == sha(control_lock_path),
            "Full validation/score audits are required before summary")
    rows = []
    for dataset in DATASETS:
        same_host_control = dataset in ("cora", "wikics")
        tied_lock = control_lock if same_host_control else primary_lock
        tied_audit = control_audit if same_host_control else primary_audit
        tied_base = CONTROL if same_host_control else ROOT
        tied_group = tied_lock["selections"][dataset]
        if not same_host_control:
            tied_group = tied_group["tied"]
        for setting in ("selected", "default"):
            all_candidate = tuple(all_lock["selections"][dataset]["selected_candidate"]) \
                if setting == "selected" else DEFAULT
            tied_candidate = tuple(tied_group["selected_candidate"]) \
                if setting == "selected" else DEFAULT
            all_acc, tied_acc, all_ce, tied_ce = [], [], [], []
            for seed in range(3):
                al_key = key(dataset, "", *all_candidate, seed)
                tied_key = key(dataset, "tied", *tied_candidate, seed)
                require(al_key in all_audit["scores"] and
                        tied_key in tied_audit["scores"],
                        f"Audited score missing: {dataset}/{setting}/seed{seed}")
                al = read_score(ALL, al_key, all_audit["scores"][al_key]["score_sha256"])
                tied = read_score(tied_base, tied_key,
                                  tied_audit["scores"][tied_key]["score_sha256"])
                all_acc.append(al["test_accuracy"])
                tied_acc.append(tied["test_accuracy"])
                all_ce.append(al["test_ce"])
                tied_ce.append(tied["test_ce"])
            differences = [a - t for a, t in zip(all_acc, tied_acc)]
            rows.append({"dataset": dataset, "setting": setting,
                         "comparator": "new_same_runtime_tied36" if same_host_control else
                                       "original_same_runtime_tied",
                         "all_layer_candidate": list(all_candidate),
                         "tied_candidate": list(tied_candidate),
                         "all_layer_accuracy": summarize(all_acc),
                         "tied_accuracy": summarize(tied_acc),
                         "paired_accuracy_difference": summarize(differences),
                         "all_layer_ce": summarize(all_ce),
                         "tied_ce": summarize(tied_ce)})
    require(len(rows) == 8, "Expected four graphs × two settings")
    report = {"protocol": "posthoc_same_runtime_factor_placement_summary_v1",
              "source_sha256": sha(Path(__file__)),
              "primary_validation_lock_sha256": PRIMARY_LOCK_SHA,
              "primary_final_score_audit_sha256": PRIMARY_SCORE_AUDIT_SHA,
              "all_layer_validation_lock_sha256": sha(all_lock_path),
              "all_layer_final_score_audit_sha256": sha(all_audit_path),
              "same_runtime_tied36_validation_lock_sha256": sha(control_lock_path),
              "same_runtime_tied36_final_score_audit_sha256": sha(control_audit_path),
              "interpretation": "post hoc one-split descriptive paired-seed comparison; no inferential claim",
              "rows": rows}
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with table.open("w", newline="") as stream:
        fields = ("dataset", "setting", "comparator", "all_layer_lr", "all_layer_wd",
                  "tied_lr", "tied_wd", "all_layer_mean", "all_layer_sd",
                  "tied_mean", "tied_sd", "paired_delta_mean", "paired_delta_sd")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"dataset": row["dataset"], "setting": row["setting"],
                             "comparator": row["comparator"],
                             "all_layer_lr": row["all_layer_candidate"][0],
                             "all_layer_wd": row["all_layer_candidate"][1],
                             "tied_lr": row["tied_candidate"][0],
                             "tied_wd": row["tied_candidate"][1],
                             "all_layer_mean": row["all_layer_accuracy"]["mean"],
                             "all_layer_sd": row["all_layer_accuracy"]["sample_sd"],
                             "tied_mean": row["tied_accuracy"]["mean"],
                             "tied_sd": row["tied_accuracy"]["sample_sd"],
                             "paired_delta_mean": row["paired_accuracy_difference"]["mean"],
                             "paired_delta_sd": row["paired_accuracy_difference"]["sample_sd"]})
    print(json.dumps({"summary_sha256": sha(output), "table_sha256": sha(table),
                      "rows": len(rows)}))


if __name__ == "__main__":
    main()
