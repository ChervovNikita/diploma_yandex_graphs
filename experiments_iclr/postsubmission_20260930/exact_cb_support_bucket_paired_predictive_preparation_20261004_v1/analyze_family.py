"""Fixed paired development contrasts after all nine own complete fit receipts."""
from argparse import ArgumentParser
from itertools import product
from math import isfinite, sqrt
from pathlib import Path
import json
import statistics
from pilot_common import atomic_json, file_sha, require, verify_manifest, ARMS


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--execution-root", required=True, type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    source_sha = file_sha(source/"MANIFEST.json")
    verify_manifest(source, source_sha)
    plan = json.loads((source/"PILOT_PLAN.json").read_text())
    execution = args.execution_root.resolve()
    require(execution == source.parent/plan["execution_directory"], "Exact own execution root required")
    cells, missing = {}, []
    for seed in (0, 1, 2):
        for arm in ARMS:
            cell = "fit_"+arm+"_seed"+str(seed)
            output = execution/cell/"run01"
            terminal = execution/"supervision"/cell/"run01"/"SUPERVISOR_TERMINAL.json"
            complete = output/"COMPLETE.json"
            if not terminal.is_file() or not complete.is_file():
                missing.append({"seed": seed, "arm": arm, "terminal": terminal.is_file(), "complete": complete.is_file()})
                continue
            t, c = json.loads(terminal.read_text()), json.loads(complete.read_text())
            if t.get("status") != "COMPLETE" or t.get("child_exit_code") != 0:
                missing.append({"seed": seed, "arm": arm, "terminal_status": t.get("status")})
                continue
            require(t["fit_receipt"]["sha256"] == file_sha(complete) and c["identity"]["driver_manifest_sha256"] == source_sha
                    and c["identity"]["family_id"] == plan["family_id"] and t.get("cap_violation") is None
                    and c["seed"] == seed and c["arm"] == arm and c["epochs"] == 100 and c["optimizer_steps"] == 1700,
                    "Own complete cell identity/custody differs")
            pin = c["private_selection"]
            path = output/pin["path"]
            require(path.name == "PRIVATE_SELECTION_"+arm+".json" and path.stat().st_size == pin["bytes"]
                    and file_sha(path) == pin["sha256"], "Own private selector custody differs")
            selected = json.loads(path.read_text())
            require(selected["identity"] == c["identity"] and selected["seed"] == seed and selected["arm"] == arm,
                    "Own selector provenance differs")
            require(selected["schema"] == "ncnc-pilot-validation-selected-checkpoint-v1"
                    and selected["checkpoint"] == c["selected_checkpoint"]
                    and isfinite(selected["selection"]["hits50"]) and 0 <= selected["selection"]["hits50"] <= 1
                    and selected["selection"]["order"] in range(1, 101), "Own complete selection differs")
            cells[(seed, arm)] = {"Hits50": selected["selection"]["hits50"], "selected_epoch": selected["selection"]["order"],
                "complete_sha256": file_sha(complete), "terminal_sha256": file_sha(terminal), "complete": c}
    result = {"schema": "ncnc-three-block-paired-development-v1", "family_id": plan["family_id"],
        "status": "INCOMPLETE_FAMILY" if missing else "COMPLETE_DEVELOPMENT_ONLY", "missing_or_failed_cells": missing,
        "TEST_access": False, "Collab_TEST_history_consumed": True, "independent_graph_uncertainty": False,
        "all_completed_cells": [{"seed": seed, "arm": arm, **{k:v for k,v in value.items() if k != "complete"}}
                                for (seed, arm), value in cells.items()]}
    if not missing:
        pairing = {}
        for seed in (0, 1, 2):
            records = [cells[(seed, arm)]["complete"] for arm in ARMS]
            pairing[str(seed)] = {"initial_state_exact": len({r["initial_state_sha256"] for r in records}) == 1,
                "initial_rng_exact": len({r["initial_rng_sha256"] for r in records}) == 1,
                "all100_epoch_native_draw_permutation_batch_and_RNG_receipts_exact": all(r["epoch_streams"] == records[0]["epoch_streams"] for r in records[1:])}
        result["pairing"] = pairing
        result["pairing_verified"] = all(all(checks.values()) for checks in pairing.values())
        result["eligible_for_paired_interpretation"] = result["pairing_verified"]
        if not result["pairing_verified"]:
            result["status"] = "COMPLETE_DEVELOPMENT_PAIRING_MISMATCH"
        contrasts = {}
        for name, a, b in (("joint_minus_target", "joint", "target_only"),
                           ("joint_minus_separate", "joint", "separate"),
                           ("separate_minus_target", "separate", "target_only")):
            values = [cells[(seed, a)]["Hits50"]-cells[(seed, b)]["Hits50"] for seed in (0, 1, 2)]
            mean, sd = statistics.mean(values), statistics.stdev(values)
            half = 4.302652729911275*sd/sqrt(3)  # fixed Student-t df2; descriptive three-block interval.
            exact = sum(abs(sum(s*v for s,v in zip(signs, values))/3) >= abs(mean)-1e-15 for signs in product((-1, 1), repeat=3))/8
            contrasts[name] = {"seed_order": [0, 1, 2], "paired_differences": values, "mean": mean,
                "units": "Hits50_fraction", "percentage_point_differences": [100*v for v in values],
                "selected_epochs": [{"seed": seed, a: cells[(seed, a)]["selected_epoch"],
                                     b: cells[(seed, b)]["selected_epoch"]} for seed in (0, 1, 2)],
                "sample_SD": sd, "range": [min(values), max(values)], "positive": sum(v>0 for v in values),
                "zero": sum(v==0 for v in values), "negative": sum(v<0 for v in values),
                "descriptive_t95_interval": [mean-half, mean+half], "two_sided_exact_sign_flip_p": exact,
                "minimum_nonzero_three_block_two_sided_p": 0.25}
        result["contrasts"] = contrasts
        result["interpretation_rules"] = plan["interpretation_rules"]
    else:
        result["contrasts"] = None  # no success-only subset mean.
    atomic_json(execution/"PAIRED_DEVELOPMENT.json", result)


if __name__ == "__main__":
    main()
