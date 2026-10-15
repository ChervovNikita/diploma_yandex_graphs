"""Complete paired selected-VALID BN analysis from saved exports; no models."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import statistics
import subprocess
import time


def pin(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-wait-seconds", type=int, default=129600)
    args = parser.parse_args()
    if socket.gethostname() != "anogena-2-0" or subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines() \
            != ["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]:
        raise ValueError("Dedicated allocation required before project reads")
    root = Path(args.project_root).resolve()

    def inside(path):
        path = Path(path).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Only own project evidence")
        return path

    spec_path, run, output = map(inside, (args.spec, args.run_directory, args.output))
    raw = spec_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.spec_sha256:
        raise ValueError("Frozen analysis question changed")
    spec = json.loads(raw)
    if spec["schema"] != "bounded-native-bn-followup-v1" or spec["new_fits"] != 3 or spec["TEST_access"] is not False:
        raise ValueError("Only exact3-fit closed-TEST follow-up")
    output.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")

    write("START.json", {"SPEC": pin(spec_path), "source": pin(Path(__file__)),
                         "models": 0, "training_steps": 0, "TEST_accessed": False})
    start = time.monotonic()
    while not (run/"END.json").exists():
        if time.monotonic()-start > args.maximum_wait_seconds:
            write("RESULT.json", {"status": "wait_cap_partial", "paired_trials": 0,
                                  "planned_paired_trials": 3, "quality_claim": False})
            return 1
        time.sleep(30)
    terminal = json.loads((run/"END.json").read_text())
    if terminal["status"] != "complete":
        write("RESULT.json", {"status": "fit_queue_incomplete", "terminal": terminal,
                              "planned_paired_trials": 3, "quality_claim": False})
        return 1
    import torch
    from ogb.graphproppred import Evaluator
    torch.set_num_threads(2)
    evaluator = Evaluator(name="ogbg-molbace")
    parent = inside(spec["parent_pilot_directory"])
    plan = json.loads((parent/"PLAN.json").read_text())
    bindings = {r["id"]: r for r in plan["jobs"]}

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def load(folder, expected_id):
        folder = inside(folder)
        result_path, job_path = folder/"RESULT.json", folder/"JOB.json"
        result, job = json.loads(result_path.read_text()), json.loads(job_path.read_text())
        if result["status"] != "complete" or job["id"] != expected_id \
                or digest(job) != result["job_hash"] or result["TEST_accessed"] is not False:
            raise ValueError("An own completed fit is required for every declared member")
        descriptor = result["selected_VALID_export"]
        export_path = inside(descriptor["path"])
        if pin(export_path)["sha256"] != descriptor["sha256"]:
            raise ValueError("Selected export bytes changed")
        value = torch.load(export_path, map_location="cpu", weights_only=False)
        if value["role"] != "VALID" or value["TEST_accessed"] is not False or value["job_hash"] != digest(job):
            raise ValueError("Only the frozen selected VALID export")
        return job, result, value, pin(result_path)

    def metric(logits, truth):
        observed = ~torch.isnan(truth)
        return {"rocauc": float(evaluator.eval({"y_true": truth.numpy(), "y_pred": logits.numpy()})["rocauc"]),
                "BCE": float(torch.nn.functional.binary_cross_entropy_with_logits(logits[observed], truth[observed].to(logits)))}

    rows = []
    for index, (job, cohort) in enumerate(zip(spec["jobs"], spec["parent_cohorts"])):
        token = str(index).zfill(3)+"_"+digest(job)[:16]
        candidate = load(run/"fits"/token, job["id"])
        controls = [load(bindings[key]["output"], key) for key in cohort["native_fit_ids"]]
        base_id = cohort["method_fit_ids"][0]
        base = load(bindings[base_id]["output"], base_id)
        all_fits = [*controls, base, candidate]
        ids, truth = candidate[2]["ids"], candidate[2]["labels"]
        if len(ids) != 151 or not all(torch.equal(item[2]["ids"], ids) and torch.equal(item[2]["labels"], truth) for item in all_fits):
            raise ValueError("Unmatched full official VALID identities/labels")
        if candidate[0]["recipe"] != base[0]["recipe"] or candidate[0]["policy"] != base[0]["policy"] \
                or candidate[0]["seed"] != base[0]["seed"] or base[0]["procedure"] != "gnnm_boundary_native":
            raise ValueError("Candidate/base source settings or pairing differ")
        metrics = {"M1": metric(controls[0][2]["raw_logits"], truth),
                   "I4": metric(torch.stack([c[2]["raw_logits"] for c in controls]).mean(0), truth),
                   "base": metric(base[2]["raw_logits"], truth),
                   "candidate": metric(candidate[2]["raw_logits"], truth)}
        rows.append({"cohort": cohort["id"], "seed": job["seed"], "metrics": metrics,
                     "AUC_delta_pp_vs_base": 100*(metrics["candidate"]["rocauc"]-metrics["base"]["rocauc"]),
                     "AUC_delta_pp_vs_I4": 100*(metrics["candidate"]["rocauc"]-metrics["I4"]["rocauc"]),
                     "BCE_delta_vs_base": metrics["candidate"]["BCE"]-metrics["base"]["BCE"],
                     "BCE_delta_vs_I4": metrics["candidate"]["BCE"]-metrics["I4"]["BCE"],
                     "candidate_selected_epoch": candidate[1]["selected_epoch"],
                     "candidate_trained_epochs": candidate[1]["completed_epochs"],
                     "base_selected_epoch": base[1]["selected_epoch"],
                     "authorities": [item[3] for item in all_fits]})
    summary = {}
    for key in ("AUC_delta_pp_vs_base", "AUC_delta_pp_vs_I4", "BCE_delta_vs_base", "BCE_delta_vs_I4"):
        values = [r[key] for r in rows]
        summary[key] = {"values": values, "mean": statistics.mean(values), "sample_SD": statistics.stdev(values)}
    write("RESULT.json", {"status": "complete", "planned_paired_trials": 3, "complete_paired_trials": len(rows),
                         "rows": rows, "summary": summary, "models": 0, "training_steps": 0, "TEST_accessed": False,
                         "scope": "Selected VALID development; not independent confirmation or significance",
                         "all_declared_outcomes_reported": True, "source": pin(Path(__file__))})
    lines = ["# BN follow-up: complete selected-VALID comparison", "", "Exploratory, three paired seeds on one scaffold split. TEST stays closed.", "",
             "| Seed | M1 AUC | I4 AUC | Base AUC | BN candidate AUC | Delta vs base, points | Delta vs I4, points |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        m = row["metrics"]
        lines.append(f"| {row['seed']} | {m['M1']['rocauc']:.5f} | {m['I4']['rocauc']:.5f} | {m['base']['rocauc']:.5f} | {m['candidate']['rocauc']:.5f} | {row['AUC_delta_pp_vs_base']:+.4f} | {row['AUC_delta_pp_vs_I4']:+.4f} |")
    lines += ["", "All BCE tradeoffs, seed differences, selected/trained epochs and own-fit authorities are in RESULT.json. No significance, transfer, novelty or acceptance claim follows from this development comparison.", ""]
    (output/"REPORT.md").write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
