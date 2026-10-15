"""Fixed posthoc VALID readout comparison on already saved route logits."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import socket
import subprocess
import time
import traceback


def guard():
    if socket.gethostname() != "anogena-2-0" or subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines() != [
                "GPU-44039938-fd82-41d2-fefd-de71514e2fac"]:
        raise ValueError("Authorized allocation required before project reads")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Saved-logit diagnostic requires CUDA hidden")


def pin(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def inside(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Repository input/output only")
    return path


def verified(row, root):
    path = inside(row["path"], root)
    if pin(path) != row:
        raise ValueError("Frozen evidence changed: " + str(path))
    return path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def same_labels(torch, a, b):
    return a.dtype == b.dtype and a.shape == b.shape and bool(
        torch.allclose(a, b, rtol=0, atol=0, equal_nan=True))


def fixed_readouts(torch, evaluator, routes, labels):
    """Both fixed rules; no parameters, tuning, model construction or selector."""
    if routes.ndim != 3 or routes.shape[-1] != 1 or labels.shape != routes.shape[1:]:
        raise ValueError("Native binary route/target shapes required")
    if not torch.isfinite(routes).all():
        raise ValueError("Finite saved route logits required")
    mask = labels[:, 0] == labels[:, 0]
    y = labels[mask, 0].double()
    if len(y) == 0 or not ((y == 0) | (y == 1)).all():
        raise ValueError("Observed binary labels required")
    if len(torch.unique(y)) != 2:
        raise ValueError("Native AUC requires both observed classes")
    # Preserve the native stored-dtype arithmetic baseline exactly.
    mean_logits = routes.mean(0)
    baseline_log_p = torch.nn.functional.logsigmoid(mean_logits.double())
    baseline_log_q = torch.nn.functional.logsigmoid(-mean_logits.double())
    # Arithmetic mean of route probabilities, evaluated using stable log sums.
    z = routes.double()
    log_p = torch.logsumexp(torch.nn.functional.logsigmoid(z), dim=0) - math.log(len(z))
    log_q = torch.logsumexp(torch.nn.functional.logsigmoid(-z), dim=0) - math.log(len(z))
    probabilities = log_p.exp()
    direct_probability_mean = torch.sigmoid(z).mean(0)
    torch.testing.assert_close(probabilities, direct_probability_mean, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(log_p.exp() + log_q.exp(), torch.ones_like(log_p), atol=1e-14, rtol=1e-14)

    def metrics(scores, lp, lq):
        selected_lp, selected_lq = lp[mask, 0], lq[mask, 0]
        losses = torch.where(y.bool(), -selected_lp, -selected_lq)
        return {"rocauc": float(evaluator.eval({"y_true": labels.numpy(),
                    "y_pred": scores.numpy()})["rocauc"]),
                "observed_label_BCE": float(losses.mean()),
                "observed_graphs": len(y), "missing_labels": int((~mask).sum())}

    baseline = metrics(mean_logits, baseline_log_p, baseline_log_q)
    alternative = metrics(probabilities, log_p, log_q)
    sanity = None
    if len(routes) == 1:
        torch.testing.assert_close(probabilities, baseline_log_p.exp(), atol=1e-14, rtol=1e-14)
        if abs(alternative["rocauc"] - baseline["rocauc"]) > 1e-12 or abs(
                alternative["observed_label_BCE"] - baseline["observed_label_BCE"]) > 1e-12:
            raise ValueError("Native singleton sanity differs")
        sanity = "passed: singleton probability and logit rules agree"
    return {"mean_raw_logits": mean_logits, "mean_logits_probability": baseline_log_p.exp(),
            "mean_sigmoid_probabilities": probabilities, "alternative_log_p": log_p,
            "alternative_log_one_minus_p": log_q}, {
                "baseline_mean_raw_logits": baseline, "fixed_mean_sigmoid_probabilities": alternative,
                "alternative_minus_baseline_ROC_AUC_points": 100 * (alternative["rocauc"] - baseline["rocauc"]),
                "alternative_minus_baseline_BCE": alternative["observed_label_BCE"] - baseline["observed_label_BCE"],
                "maximum_probability_difference": float((probabilities - baseline_log_p.exp()).abs().max()),
                "native_singleton_sanity": sanity}


def main():
    guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    here = inside(Path(__file__).resolve().parent, root)
    output = inside(args.output, root)
    if output.exists():
        raise ValueError("Fresh output required; preserve all attempts")
    inputs = json.loads((here / "INPUTS.json").read_text())
    verified(inputs["plan"], root)
    if [(r["name"], r["expected_members"]) for r in inputs["cases"]] != [
            ("M1", 1), ("base", 4), ("C1", 4), ("C4", 4)]:
        raise ValueError("Only frozen four completed seed101 cases")
    output.mkdir(parents=True, exist_ok=False)
    import torch
    from ogb.graphproppred import Evaluator
    torch.set_num_threads(2)
    started = time.perf_counter()
    result = {"schema": "fixed-probability-aggregation-VALID-result-v1", "status": "running",
        "scope": "posthoc exploratory development diagnostic on checkpoint-selection VALID",
        "baseline": "unchanged fixed arithmetic mean raw logits",
        "alternative": "fixed arithmetic mean of sigmoid route probabilities; no fitted weights",
        "BCE_definition": "mean observed-label -log(p_y); FP64 logsigmoid/logsumexp; no clipping",
        "AUC_definition": "unmodified ogb.graphproppred.Evaluator(ogbg-molbace) on aligned predictions",
        "inputs": pin(here / "INPUTS.json"), "script": pin(__file__), "torch": str(torch.__version__),
        "device": "CPU", "threads": 2, "native_constructions": 0, "new_forward_calls": 0,
        "optimizer_steps": 0, "selector_changes": 0, "scientific_fits": 0,
        "TEST_accessed": False, "best_readout_selected": False, "cases": []}
    saved = {"schema": "fixed-readout-saved-VALID-predictions-v1", "role": "VALID",
        "TEST_accessed": False, "selector_changes": 0, "cases": []}
    reference_ids = reference_labels = reference_identity = None
    try:
        import inspect
        import ogb
        evaluator = Evaluator(name="ogbg-molbace")
        result.update(ogb_version=str(ogb.__version__), evaluator_source=pin(inspect.getfile(Evaluator)))
        for row in inputs["cases"]:
            paths = {key: verified(row[key], root) for key in (
                "route_tensor", "analysis_result", "job", "fit_result", "selected_export")}
            job = json.loads(paths["job"].read_text())
            fit = json.loads(paths["fit_result"].read_text())
            analysis = json.loads(paths["analysis_result"].read_text())
            route = torch.load(paths["route_tensor"], map_location="cpu", weights_only=False)
            export = torch.load(paths["selected_export"], map_location="cpu", weights_only=False)
            if job["family"] != "ogb" or job["seed"] != 101 or job["TEST_access"] is not False or fit[
                    "status"] != "complete" or fit["dataset"] != "ogbg-molbace":
                raise ValueError("Only completed MolBACE seed101 fits with TEST closed")
            if analysis["status"] != "complete" or analysis["raw_tensor_receipt"] != row["route_tensor"]:
                raise ValueError("Saved analysis authority differs")
            if route["schema"] != "own-selected-molecular-route-VALID-tensors-v1" or route[
                    "role"] != "VALID" or route["TEST_accessed"] is not False or export["role"] != "VALID":
                raise ValueError("Only saved own selected VALID route tensors")
            if route["job_hash"] != digest(job) or export["job_hash"] != route["job_hash"]:
                raise ValueError("Job authority differs")
            if route["selected_checkpoint"] != fit["selected_checkpoint"] or export[
                    "selected_checkpoint"] != route["selected_checkpoint"] or fit["selected_VALID_export"] != row["selected_export"]:
                raise ValueError("Selected checkpoint/export custody differs")
            if not (route["selected_epoch"] == export["selected_epoch"] == fit["selected_epoch"]):
                raise ValueError("Selected epoch differs")
            ids, labels, logits = route["ids"], route["labels"], route["member_raw_logits"]
            if len(logits) != row["expected_members"] or len(ids) != 151 or ids.ndim != 1 or len(torch.unique(ids)) != len(ids):
                raise ValueError("Expected unique complete native VALID identities and route count")
            if ids.dtype != export["ids"].dtype or not torch.equal(ids, export["ids"]) or not same_labels(torch, labels, export["labels"]):
                raise ValueError("Saved graph IDs/order/labels differ")
            if not torch.equal(logits.mean(0), route["pooled_raw_logits"]) or not route["sanity_against_saved_export"]["passed"]:
                raise ValueError("Original mean-logit export sanity did not pass")
            torch.testing.assert_close(route["pooled_raw_logits"], export["raw_logits"], atol=1e-5, rtol=1e-4)
            identity = (job["cohort_id"], job.get("split", 0), fit["dataset"])
            if reference_ids is None:
                reference_ids, reference_labels, reference_identity = ids, labels, identity
            elif identity != reference_identity or ids.dtype != reference_ids.dtype or not torch.equal(
                    ids, reference_ids) or not same_labels(torch, labels, reference_labels):
                raise ValueError("Cross-method native VALID graph identities/labels/cohort differ")
            predictions, summary = fixed_readouts(torch, evaluator, logits, labels)
            native_bce = float(torch.nn.functional.binary_cross_entropy_with_logits(
                logits.mean(0)[labels == labels].double(), labels[labels == labels].double()))
            if abs(native_bce - summary["baseline_mean_raw_logits"]["observed_label_BCE"]) > 1e-12:
                raise ValueError("Stable log probability BCE disagrees with native mean-logit BCE")
            result["cases"].append({"name": row["name"], "procedure": job["procedure"],
                "selected_epoch": fit["selected_epoch"], "seed": 101, "members": len(logits),
                "graphs": len(ids), "source_inputs": row, "exact_graph_alignment": True, **summary})
            saved["cases"].append({"name": row["name"], "ids": ids, "labels": labels,
                "source_route_tensor": row["route_tensor"], **predictions})
        artifact = output / "PREDICTIONS_VALID.pt"
        torch.save(saved, artifact)
        result.update(status="complete", saved_predictions=pin(artifact), exact_cross_method_alignment=True,
            later_I4_control="pending complete native bank; same fixed pair of rules required; no promotion")
    except Exception as error:
        result.update(status="failed", failure={"type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(limit=16)})
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result.update(elapsed_seconds=time.perf_counter() - started, process_user_seconds=usage.ru_utime,
        process_system_seconds=usage.ru_stime, process_peak_RSS_KiB=usage.ru_maxrss)
    (output / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
