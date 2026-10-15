"""Complete factorial selected-VALID diagnostics from receipted saved tensors.

No collector, model, checkpoint load, training, or TEST call is made here.
The nine missing route collections must be separately admitted and completed.
Imports at module scope are standard-library-only for synthetic source checks.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import socket
import statistics
import subprocess
import time
import traceback

AUTHORITY_SHA = "3bdaea64d1dd5bcb56af701cf98990fde8bb854fb19ab1e7ddd43622f93e4135"
HOST = "anogena-2-0"
GPU = "GPU-44039938-fd82-41d2-fefd-de71514e2fac"
SEEDS = [101, 202, 303]
EXPANSION_RELATIVE = Path("experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1")
COST_FIELDS = ("selected_epoch", "completed_epochs", "stop_reason", "convergence_established",
               "invocation_wall_seconds", "optimizer_calls", "optimizer_calls_this_invocation",
               "optimizer_calls_are_trajectory_cumulative", "resources", "failure", "close_error")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve_project_path(root, path):
    """Relative source receipts use the expansion folder within repository R."""
    root, path = Path(root).resolve(), Path(path)
    resolved = (path if path.is_absolute() else root / EXPANSION_RELATIVE / path).resolve()
    require(resolved.is_relative_to(root), "All paths must remain inside project")
    return resolved


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name):
    definition = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    return module


def terminal_gate(end, outcomes, expected_ids):
    """Metadata-only gate; call before reading REPORT scores or any fit/tensor."""
    require(end.get("status") == "complete" and end.get("TEST_accessed") is False,
            "Study is not fully complete with TEST closed")
    require(len(expected_ids) == 9 and len(set(expected_ids)) == 9, "Nine exact jobs required")
    require(end.get("outcomes") == outcomes, "END/OUTCOMES differ")
    require([r.get("job_id") for r in outcomes] == expected_ids, "Nine-job outcome order differs")
    require(all(r.get("status") == "complete" and r.get("exit_code") == 0 for r in outcomes),
            "All nine fits must have successful terminal outcomes")


def report_gate(report, spec, jobs):
    require(report.get("complete_new_fits") == 9 and report.get("TEST_accessed") is False,
            "Terminal report does not contain nine completed fits")
    expected = {(r["seed"], *r["cell"]): (r["job_id"], "reused") for r in spec["reused_controls"]}
    methods = {m["id"]: m for m in spec["methods"]}
    expected.update({(j["seed"], *methods[j["procedure"]]["cell"]): (j["id"], "new") for j in jobs})
    keys = {(s, n, g, c) for s in SEEDS for n in (0, 1) for g in (0, 1) for c in (0, 1)}
    require(set(expected) == keys and len(expected) == 24, "Frozen factorial definition differs")
    rows = report.get("all_24_cells", [])
    require(len(rows) == 24, "All 24 factorial rows required")
    seen = set()
    for row in rows:
        key = (row["seed"], *row["cell"])
        require(key not in seen and key in expected, "Duplicate or unexpected factorial cell")
        require((row["job_id"], row["origin"]) == expected[key] and row["status"] == "complete",
                "Factorial identity, origin, or status differs")
        seen.add(key)
    require(seen == keys, "Incomplete factorial")


def credit(value):
    return 1.0 if value > 0 else 0.5 if value == 0 else 0.0


def margins(logits, labels):
    """Same positive-minus-negative ordering and half-tie law as existing analyzer."""
    require(len(logits) == len(labels) and set(labels) == {0, 1}, "Binary aligned arrays required")
    return [logits[p] - logits[n] for p, y in enumerate(labels) if y == 1
            for n, other in enumerate(labels) if other == 0]


def compare_pairs(reference, candidate):
    require(len(reference) == len(candidate) and len(reference) > 0, "Aligned nonempty pairs required")
    return {
        "strict_repairs": sum(a < 0 < b for a, b in zip(reference, candidate)),
        "strict_harms": sum(a > 0 > b for a, b in zip(reference, candidate)),
        "changed_credit_pairs": sum(credit(a) != credit(b) for a, b in zip(reference, candidate)),
        "reference_tied_pairs": reference.count(0), "candidate_tied_pairs": candidate.count(0),
        "new_ties": sum(a != 0 and b == 0 for a, b in zip(reference, candidate)),
        "resolved_ties": sum(a == 0 and b != 0 for a, b in zip(reference, candidate)),
        "delta_AUC_pp": 100 * statistics.mean(credit(b) - credit(a) for a, b in zip(reference, candidate)),
    }


def pair_diagnostics(members, pool, labels):
    member_margins = [margins(z, labels) for z in members]
    pooled = margins(pool, labels)
    columns = list(zip(*member_margins))
    result = {
        "pairs": len(pooled), "pool_correct": sum(v > 0 for v in pooled),
        "pool_reversed": sum(v < 0 for v in pooled), "pool_tied": pooled.count(0),
        "all_members_reversed": sum(all(v < 0 for v in col) for col in columns),
        "all_members_correct": sum(all(v > 0 for v in col) for col in columns),
        "member_sign_disagreement_pairs": sum(any(v > 0 for v in col) and not all(v > 0 for v in col)
                                               for col in columns),
        "pool_reversed_despite_some_member_correct": sum(p < 0 and any(v > 0 for v in col)
                                                          for p, col in zip(pooled, columns)),
        "pool_reversed_despite_strict_majority_correct": sum(
            p < 0 and sum(v > 0 for v in col) > len(members) / 2 for p, col in zip(pooled, columns)),
        "pool_strict_repairs_of_all_member_reversals": sum(p > 0 and all(v < 0 for v in col)
                                                           for p, col in zip(pooled, columns)),
        "member_tied_pairs": [m.count(0) for m in member_margins],
        "member_mean_absolute_margin": [statistics.mean(abs(v) for v in m) for m in member_margins],
        "pool_vs_members": [dict(member=i, **compare_pairs(m, pooled)) for i, m in enumerate(member_margins)],
    }
    return result, pooled, [statistics.mean(map(credit, m)) for m in member_margins]


def summarize(rows, group_key, fields):
    summaries = []
    for name in dict.fromkeys(r[group_key] for r in rows):
        selected = sorted((r for r in rows if r[group_key] == name), key=lambda r: r["seed"])
        require([r["seed"] for r in selected] == SEEDS, "All three seeds required for every summary")
        summaries.append({group_key: name, "seeds": SEEDS, "summary": {
            k: {"values": [r[k] for r in selected], "mean": statistics.mean(r[k] for r in selected),
                "sample_SD": statistics.stdev(r[k] for r in selected)} for k in fields}})
    return summaries


def collection_outcome(job_id, analysis, receipt, origin):
    """Retain charged work even if a collection failed before exporting tensors."""
    return {"job_id": job_id, "origin": origin, "receipt": receipt,
            **{k: analysis.get(k) for k in ("status", "actual_work", "wall_seconds", "own_peak_rss_bytes",
                "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "failure", "close_error",
                "sanity_against_saved_export")}}


def run(args):
    started = time.monotonic()
    require(socket.gethostname() == HOST and subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=10).splitlines() == [GPU],
        "Dedicated allocation guard required before project reads")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Saved-tensor analysis requires CPU visibility only")
    root = Path(args.project_root).resolve()

    def safe(path):
        return resolve_project_path(root, path)

    output = safe(args.output)
    require(not output.exists(), "Use a fresh analysis directory")
    output.mkdir(parents=True)
    receipts, collection_records, collection_errors, fit_records, fit_receipts = {}, [], [], {}, {}

    def verified(pin):
        p = safe(pin["path"])
        require(sha(p) == pin["sha256"], "Input receipt changed: " + str(p))
        return p

    def snapshot(path, expected=None):
        p = safe(path)
        raw = p.read_bytes()
        h = hashlib.sha256(raw).hexdigest()
        require(expected is None or h == expected, "Snapshot changed: " + str(p))
        receipt = {"path": str(p), "bytes": len(raw), "sha256": h}
        receipts[str(p)] = receipt
        return json.loads(raw), receipt

    try:
        authority_path = Path(__file__).resolve().parent.parent / "complete_route_analysis_source_v1/INPUT_AUTHORITY.json"
        authority, authority_receipt = snapshot(authority_path, AUTHORITY_SHA)
        require(Path(authority["project_root"]).resolve() == root, "Frozen project root differs")
        sources = {r["path"]: verified(r) for r in authority["source_and_completed_metadata_pins"]}
        spec, spec_receipt = snapshot(sources["pilot_20261015_bn_geometry_correction_v1/SPEC.json"], authority["SPEC_sha256"])
        study = load_module(sources["pilot_20261015_bn_geometry_correction_v1/study.py"], "_route_study_source")
        jobs = study.jobs(spec)
        require(spec["seeds"] == SEEDS and len(jobs) == 9 and
                [study.digest(j) for j in jobs] == spec["job_sha256"], "Frozen nine jobs differ")
        study_output, route_root = safe(args.study_output), safe(args.new_route_root)
        end, end_receipt = snapshot(study_output / "END.json", args.end_sha256)
        outcomes, outcomes_receipt = snapshot(study_output / "OUTCOMES.json", args.outcomes_sha256)
        terminal_gate(end, outcomes, [j["id"] for j in jobs])
        # The first read containing new scores occurs only after the complete metadata gate.
        report, report_receipt = snapshot(study_output / "REPORT.json", args.report_sha256)
        report_gate(report, spec, jobs)
        controls = {r["job_id"]: r for r in spec["reused_controls"]}
        planned = {j["id"]: (i, j) for i, j in enumerate(jobs)}
        factorial = {}
        for row in report["all_24_cells"]:
            fit_path = verified(row["authority"])
            fit, fit_receipt = snapshot(fit_path, row["authority"]["sha256"])
            require(fit["status"] == "complete" and fit["job_id"] == row["job_id"] and
                    fit["seed"] == row["seed"] and fit.get("TEST_accessed") is False,
                    "Fit identity/status/role differs")
            require(study.scalar_result(fit) == row["values"], "Terminal scalar authority differs")
            if row["origin"] == "reused":
                require(row["authority"] == controls[row["job_id"]]["authority"], "Reused fit authority differs")
            else:
                index, job = planned[row["job_id"]]
                require(fit_path == study.fit_directory(study_output, index, job) / "RESULT.json" and
                        fit["job_hash"] == study.digest(job) and fit["policy"] == job["policy"], "New fit binding differs")
                binding = dict(spec["expected_runtime_binding"], extension_sha256=job["session_transform"]["sha256"])
                require(fit["runtime_binding"] == binding, "New fit runtime binding differs")
            fit_records[row["job_id"]], fit_receipts[row["job_id"]] = fit, fit_receipt
            factorial[(row["seed"], *row["cell"])] = row

        original, original_receipt = snapshot(sources["analysis/complete_route_diagnostic_v1/INPUTS.json"])
        original_rows = {r["job_id"]: r for r in original["fits"]}
        analysis_records, route_receipts = {}, {}

        def accept_collection(job_id, analysis, analysis_receipt, route_receipt, origin, require_current_source=False):
            collection_records.append(collection_outcome(job_id, analysis, analysis_receipt, origin))
            fit = fit_records[job_id]
            require(analysis["status"] in ("complete", "complete_with_sanity_warning") and
                    analysis["role"] == "VALID" and analysis["TEST_accessed"] is False and
                    analysis["job_id"] == job_id and analysis["job_hash"] == fit["job_hash"] and
                    analysis["selected_epoch"] == fit["selected_epoch"], "Collection identity/status differs")
            require(analysis["fit_receipt"]["sha256"] == fit_receipts[job_id]["sha256"] and
                    analysis["selected_checkpoint"]["sha256"] == fit["selected_checkpoint"]["sha256"] and
                    analysis["saved_VALID_export"]["sha256"] == fit["selected_VALID_export"]["sha256"] and
                    analysis["raw_tensor_receipt"] == route_receipt, "Collection authorities differ")
            work = analysis["actual_work"]
            require(work["optimizer_steps"] == 0 and work["TRAIN_callbacks"] == 0 and
                    analysis["evaluated_VALID_graphs"] == 151 and analysis["members"] == 4,
                    "Collection must cover complete four-route VALID with zero TRAIN work")
            if require_current_source:
                require(analysis["analysis_source"]["sha256"] == sha(sources["analysis/selected_molecular.py"]),
                        "New collection used a different collector")
            analysis_records[job_id], route_receipts[job_id] = analysis, route_receipt

        for row in report["all_24_cells"]:
            if row["origin"] == "reused" and row["cell"][0] == 0:
                old = original_rows[row["job_id"]]
                require(old["fit"]["sha256"] == row["authority"]["sha256"], "Original route/fit join differs")
                analysis, pin = snapshot(verified(old["analysis"]))
                accept_collection(row["job_id"], analysis, pin, old["routes"], "reused")
        for old in authority["bn_only_route_reports"]:
            analysis, pin = snapshot(verified(old["canonical_report"]))
            accept_collection(old["job_id"], analysis, pin, old["route_tensor"], "reused")
        for index, job in enumerate(jobs):
            path = route_root / f"fit_{index:03}" / "RESULT.json"
            try:
                analysis, pin = snapshot(path)
            except (OSError, ValueError) as error:
                analysis = {"status": "missing_or_unreadable", "failure": {"message": str(error)}}
                collection_records.append(collection_outcome(job["id"], analysis, {"path": str(path)}, "new"))
                collection_errors.append({"job_id": job["id"], "message": str(error)})
                continue
            try:
                accept_collection(job["id"], analysis, pin, analysis.get("raw_tensor_receipt"), "new", True)
            except (KeyError, ValueError) as error:
                collection_errors.append({"job_id": job["id"], "message": str(error)})
        require(not collection_errors, "Missing, failed or mismatched new collections; all nine outcomes retained")
        require(len(analysis_records) == 24, "All 24 route collections required")

        # Only now, after all terminal and collection metadata gates, import tensor tools.
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[key] = "2"
        import torch
        from ogb.graphproppred import Evaluator
        torch.set_num_threads(2)
        helper = load_module(sources["analysis/selected_molecular.py"], "_route_metric_source")
        evaluator = Evaluator(name="ogbg-molbace")
        exports, routes, warnings = {}, {}, []
        reference_ids = reference_labels = None

        def export(job_id, old=None):
            nonlocal reference_ids, reference_labels
            if job_id in exports:
                return exports[job_id]
            if job_id not in fit_records:
                fit, pin = snapshot(verified(old["fit"]))
                require(fit["status"] == "complete" and fit["job_id"] == job_id and fit.get("TEST_accessed") is False,
                        "Native context fit differs")
                fit_records[job_id], fit_receipts[job_id] = fit, pin
            fit = fit_records[job_id]
            pin = fit["selected_VALID_export"]
            if old:
                require(old["export"]["sha256"] == pin["sha256"], "Native context export differs")
            value = torch.load(verified(pin), map_location="cpu", weights_only=False)
            require(value["role"] == "VALID" and value["TEST_accessed"] is False and
                    value["job_id"] == job_id and value["job_hash"] == fit["job_hash"] and
                    value["selected_epoch"] == fit["selected_epoch"] and
                    value["selected_checkpoint"]["sha256"] == fit["selected_checkpoint"]["sha256"],
                    "Saved export identity/role differs")
            ids, truth, logits = value["ids"], value["labels"], value["raw_logits"]
            require(ids.numel() == 151 and len(set(ids.reshape(-1).tolist())) == 151 and
                    tuple(truth.shape) == tuple(logits.shape) == (151, 1) and
                    torch.isfinite(truth).all().item() and torch.isfinite(logits).all().item() and
                    int((truth == 1).sum()) == 130 and int((truth == 0).sum()) == 21,
                    "Complete finite original binary VALID required")
            if reference_ids is None:
                reference_ids, reference_labels = ids, truth
            require(torch.equal(ids, reference_ids) and torch.equal(truth, reference_labels),
                    "VALID identities/order/labels differ across banks")
            exports[job_id] = value
            return value

        def route(job_id):
            saved = export(job_id)
            value = torch.load(verified(route_receipts[job_id]), map_location="cpu", weights_only=False)
            require(value["role"] == "VALID" and value["TEST_accessed"] is False and
                    value["job_id"] == job_id and value["job_hash"] == saved["job_hash"] and
                    value["selected_epoch"] == saved["selected_epoch"] and
                    value["selected_checkpoint"]["sha256"] == saved["selected_checkpoint"]["sha256"] and
                    torch.equal(value["ids"], saved["ids"]) and torch.equal(value["labels"], saved["labels"]),
                    "Route/export selected-state identity differs")
            z, pooled = value["member_raw_logits"], value["pooled_raw_logits"]
            require(tuple(z.shape) == (4, 151, 1) and tuple(pooled.shape) == (151, 1) and
                    torch.isfinite(z).all().item() and torch.isfinite(pooled).all().item(), "Invalid route arrays")
            routes[job_id] = value
            return value

        rows, all_margins = [], {}

        def bank(seed, name, member_logits, pool, owners, cell=None, origin="reused_context", reconstructed=None):
            truth = exports[owners[0]]["labels"]
            metrics = [helper.binary_metrics(torch, evaluator, z, truth) for z in member_logits]
            pooled = helper.binary_metrics(torch, evaluator, pool, truth)
            diagnostic, pm, manual = pair_diagnostics(
                [z[:, 0].double().tolist() for z in member_logits], pool[:, 0].double().tolist(), truth[:, 0].tolist())
            require(all(abs(m["rocauc"] - a) < 1e-12 for m, a in zip(metrics, manual)) and
                    abs(pooled["rocauc"] - statistics.mean(map(credit, pm))) < 1e-12,
                    "Pair AUC and native evaluator differ")
            row = {"seed": seed, "bank": name, "cell": cell, "origin": origin, "source_job_ids": owners,
                "selected_epochs": [fit_records[j]["selected_epoch"] for j in owners], "members": metrics,
                "mean_member_AUC": statistics.mean(m["rocauc"] for m in metrics),
                "worst_member_AUC": min(m["rocauc"] for m in metrics), "pool_AUC": pooled["rocauc"],
                "mean_member_BCE": statistics.mean(m["BCE"] for m in metrics),
                "worst_member_BCE": max(m["BCE"] for m in metrics), "uniform_graph_BCE": pooled["BCE"],
                "pooled_metrics": pooled, **diagnostic}
            row["pool_minus_mean_member_AUC_pp"] = 100 * (row["pool_AUC"] - row["mean_member_AUC"])
            row["saved_native_VALID_BCE"] = fit_records[owners[0]]["selected_VALID"]["loss"] if len(owners) == 1 else None
            if reconstructed is not None:
                analysis = analysis_records[owners[0]]
                row["reconstruction"] = {"sanity": analysis["sanity_against_saved_export"],
                    "max_abs_pool_difference": float((reconstructed.double() - pool.double()).abs().max()),
                    "reconstructed_pool_AUC": helper.binary_metrics(torch, evaluator, reconstructed, truth)["rocauc"]}
                if analysis["status"] != "complete" or not analysis["sanity_against_saved_export"]["passed"]:
                    warnings.append({"seed": seed, "bank": name, "collection_receipt": next(
                        r["receipt"] for r in collection_records if r["job_id"] == owners[0]), **row["reconstruction"]})
            rows.append(row)
            all_margins[(seed, name)] = pm

        for cohort in original["cohorts"]:
            ids = cohort["native_fit_ids"]
            require(len(ids) == 4 and len(set(ids)) == 4 and cohort["M1_fit_id"] == ids[0], "Native context selectors differ")
            native = [export(j, original_rows[j]) for j in ids]
            seed = fit_records[ids[0]]["seed"]
            require(seed in SEEDS, "Unexpected context seed")
            native_logits = torch.stack([v["raw_logits"] for v in native])
            bank(seed, "M1", native_logits[:1], native_logits[0], ids[:1])
            bank(seed, "I4", native_logits, native_logits.mean(0), ids)
            for n in (0, 1):
                for g in (0, 1):
                    for c in (0, 1):
                        row = factorial[(seed, n, g, c)]
                        saved, raw = export(row["job_id"]), route(row["job_id"])
                        bank(seed, f"n{n}_g{g}_c{c}", raw["member_raw_logits"], saved["raw_logits"], [row["job_id"]],
                             [n, g, c], row["origin"], raw["pooled_raw_logits"])
        require(len(rows) == 30 and len(fit_records) == 36, "Exactly 30 banks from 36 distinct training fits required")
        comparisons = []
        contrasts = {k: v for k, v in study.contrast_definitions().items() if len(v) == 2}
        for seed in SEEDS:
            for name, coefficients in contrasts.items():
                candidate = next(cell for cell, weight in coefficients.items() if weight == 1)
                reference = next(cell for cell, weight in coefficients.items() if weight == -1)
                cb, rb = [f"n{n}_g{g}_c{c}" for n, g, c in (candidate, reference)]
                comparisons.append({"seed": seed, "comparison": name, "candidate": cb, "reference": rb,
                                    **compare_pairs(all_margins[(seed, rb)], all_margins[(seed, cb)])})
            for row in (r for r in rows if r["seed"] == seed and r["cell"] is not None):
                for reference in ("M1", "I4"):
                    comparisons.append({"seed": seed, "comparison": row["bank"] + "_vs_" + reference,
                        "candidate": row["bank"], "reference": reference,
                        **compare_pairs(all_margins[(seed, reference)], all_margins[(seed, row["bank"])])})
        fields = ["mean_member_AUC", "worst_member_AUC", "pool_AUC", "pool_minus_mean_member_AUC_pp",
                  "mean_member_BCE", "worst_member_BCE", "uniform_graph_BCE", "all_members_reversed",
                  "pool_reversed_despite_some_member_correct", "pool_reversed_despite_strict_majority_correct"]
        result = {"schema": "complete-factorial-route-analysis-v2", "status": "complete_with_warnings" if warnings else "complete",
            "scope": "Selected VALID, three paired seeds, one scaffold split; descriptive, not causal or independent confirmation",
            "graphs": 151, "positive_graphs": 130, "negative_graphs": 21, "pairs_per_seed": 2730,
            "routes_or_pairs_are_independent_training_repetitions": False,
            "rows": rows, "summaries": summarize(rows, "bank", fields), "pair_comparisons": comparisons,
            "pair_comparison_summaries": summarize(comparisons, "comparison", ["strict_repairs", "strict_harms", "changed_credit_pairs", "delta_AUC_pp"]),
            "warnings": warnings, "terminal_study_report": report_receipt, "terminal_END": end_receipt,
            "terminal_OUTCOMES": outcomes_receipt, "source_authority": authority_receipt, "SPEC": spec_receipt,
            "original_route_inputs": original_receipt, "input_receipts": list(receipts.values()),
            "training_fit_outcomes": [{"job_id": j, "receipt": fit_receipts[j], "status": f["status"],
                                       **{k: f.get(k) for k in COST_FIELDS}} for j, f in fit_records.items()],
            "route_collection_outcomes": collection_records,
            "historical_full_original_collection": original["completed_analysis"],
            "cost_rule": "36 unique historical training fits, M1/body0 counted once; collector work separate; no totals of concurrent walls or memory peaks",
            "BCE_rule": "saved_native_VALID_BCE retains native reduction; uniform_graph_BCE is unchanged binary_metrics graph reduction; never overwrite native loss",
            "interpretation": "Coverage, disagreement and majority cancellation do not imply an attainable gate; retain AUC/BCE tradeoffs and every seed",
            "new_models": 0, "new_native_forwards": 0, "new_optimizer_steps": 0, "TEST_accessed": False,
            "source_sha256": sha(__file__), "wall_seconds": time.monotonic() - started,
            "peak_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        (output / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        lines = ["# Complete factorial route diagnostics", "", result["scope"], "",
            "AUCs are percentages. BCE uses the uniform graph reduction; native loss and all costs remain in RESULT.json.", "",
            "| Seed | Bank n geometry correction | Mean member AUC | Worst member AUC | Pool AUC | Pool gain points | Pool BCE |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        lines += [f"| {r['seed']} | {r['bank']} | {100*r['mean_member_AUC']:.3f} | {100*r['worst_member_AUC']:.3f} | {100*r['pool_AUC']:.3f} | {r['pool_minus_mean_member_AUC_pp']:.3f} | {r['uniform_graph_BCE']:.4f} |" for r in rows]
        lines += ["", "Every member, strict pair repairs/harms, ties, three-seed summaries, reconstruction warnings and individual fit/collector costs are retained in RESULT.json.",
                  "Pair coverage is not a feasible gating oracle. No scalar factorial interaction was recomputed or selected rule changed.", ""]
        (output / "REPORT.md").write_text("\n".join(lines))
        return result
    except Exception as error:
        blocked = {"schema": "complete-factorial-route-analysis-v2-gate", "status": "blocked_no_partial_summary",
            "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(limit=6)},
            "input_receipts_read_before_stop": list(receipts.values()), "TEST_accessed": False,
            "route_collection_outcomes": collection_records, "collection_rejections": collection_errors,
            "training_fit_outcomes_already_verified": [{"job_id": j, "receipt": fit_receipts[j],
                "status": f["status"], **{k: f.get(k) for k in COST_FIELDS}} for j, f in fit_records.items()],
            "new_models": 0, "new_native_forwards": 0, "new_optimizer_steps": 0,
            "wall_seconds": time.monotonic() - started}
        (output / "GATE.json").write_text(json.dumps(blocked, indent=2) + "\n")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("project-root", "study-output", "end-sha256", "outcomes-sha256", "report-sha256", "new-route-root", "output"):
        parser.add_argument("--" + argument, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({k: result[k] for k in ("status", "wall_seconds", "peak_rss_KiB")}))


if __name__ == "__main__":
    main()
