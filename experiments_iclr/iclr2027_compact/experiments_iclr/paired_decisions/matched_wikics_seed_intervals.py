"""Audit conditional seed intervals for the post hoc WikiCS storage controls."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import statistics
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
STAGE = HERE.parent / "wikics_matched54"
T_CRITICAL_95_DF2 = 4.3026527299


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def selected_scores(arm: str, lock: dict, audit: dict, manifest: dict) -> tuple[list[float], list[float]]:
    candidate = lock["selections"][arm]["selected_candidate"]
    lr, wd = candidate
    scores = []
    for seed in (0, 1, 2):
        key = f"wikics/{arm}/lr{lr:g}_wd{wd:g}/seed{seed}"
        rel = f"scores/{key}/score.json"
        path = STAGE / rel
        digest = sha(path)
        need(digest == manifest["files"][rel] ==
             audit["scores"][key]["score_sha256"],
             f"Selected score hash differs: {key}")
        value = json.loads(path.read_text())["test_accuracy"]
        need(abs(value - audit["scores"][key]["test_accuracy"]) < 1e-6,
             f"Selected test accuracy differs: {key}")
        scores.append(value)
    return scores, candidate


def calculate() -> dict:
    manifest = json.loads((STAGE / "COMPACT_MANIFEST.json").read_text())
    lock_path = STAGE / "VALIDATION_SELECTION_LOCK.json"
    audit_path = STAGE / "FINAL_SCORE_AUDIT.json"
    need(manifest["protocol"] == "wikics_matched54_posthoc_v1" and
         sha(lock_path) == manifest["validation_lock_sha256"] and
         sha(audit_path) == manifest["score_audit_sha256"],
         "Matched WikiCS manifest/lock/audit differs")
    lock, audit = json.loads(lock_path.read_text()), json.loads(audit_path.read_text())
    need(audit["validation_selection_lock_sha256"] == sha(lock_path),
         "Matched WikiCS score audit lock differs")
    values = {}
    candidates = {}
    for arm in ("private_last", "ens", "base"):
        values[arm], candidates[arm] = selected_scores(arm, lock, audit, manifest)
    rows = []
    for comparator in ("ens", "base"):
        diffs = [100 * (x - y) for x, y in
                 zip(values["private_last"], values[comparator])]
        mean = statistics.mean(diffs)
        sd = statistics.stdev(diffs)
        se = sd / math.sqrt(3)
        half = T_CRITICAL_95_DF2 * se
        rows.append({"graph": "wikics", "partial_arm": "private_last",
                     "comparator": comparator, "n_optimizer_seeds": 3,
                     "partial_candidate": candidates["private_last"],
                     "comparator_candidate": candidates[comparator],
                     "seed_deltas_pp": diffs, "mean_delta_pp": mean,
                     "sample_sd_pp": sd, "standard_error_pp": se,
                     "conditional_95_low_pp": mean - half,
                     "conditional_95_high_pp": mean + half})
    return {"scope": "Post hoc storage-matched control on WikiCS public split 0. Illustrative t intervals condition on this fixed split and frozen validation-selected configurations, treating three paired optimizer-seed differences as independent and approximately normal. Selection pooled these seeds' validation results and is not accounted for. No inference over graph tasks, splits, nodes, or simultaneous contrasts.",
            "source_manifest_sha256": sha(STAGE / "COMPACT_MANIFEST.json"),
            "t_critical_95_df2": T_CRITICAL_95_DF2, "rows": rows}


def main() -> None:
    need(len(sys.argv) == 1 or sys.argv == [sys.argv[0], "--check"],
         "Usage: python matched_wikics_seed_intervals.py [--check]")
    report = calculate()
    json_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    fields = ("graph", "partial_arm", "comparator", "n_optimizer_seeds",
              "mean_delta_pp", "sample_sd_pp", "standard_error_pp",
              "conditional_95_low_pp", "conditional_95_high_pp")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows({field: row[field] for field in fields} for row in report["rows"])
    for name, content in (("MATCHED_WIKICS_CONDITIONAL_INTERVALS.json", json_text),
                          ("MATCHED_WIKICS_CONDITIONAL_INTERVALS.csv", stream.getvalue())):
        path = HERE / name
        if "--check" in sys.argv:
            need(path.read_bytes() == content.encode(),
                 f"Packaged WikiCS interval output differs: {name}")
        else:
            path.write_text(content)
    print(json.dumps({"status": "PASS", "matched_comparisons": 2,
                      "seeds_each": 3}, sort_keys=True))


if __name__ == "__main__":
    main()
