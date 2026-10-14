"""Read-only selected-state molecular VALID diagnostics; no training or TEST.

Execution is allocation-only. Source/fit files are never modified. Authentic
route logits and native graph preheads are saved in a fresh analysis directory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

ALLOCATION_UUID = "GPU-44039938-fd82-41d2-fefd-de71514e2fac"


def require_allocation():
    if socket.gethostname() != "anogena-2-0":
        raise ValueError("Selected-state analysis requires the authorized dedicated allocation")
    values = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                                     text=True, timeout=10).splitlines()
    if values != [ALLOCATION_UUID]:
        raise ValueError("Wrong allocation GPU; no project source/model/data loaded")


def pin(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def safe_path(path, root):
    path = Path(path)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Deliberate analysis access outside project: " + str(path))
    return path


def verify_pin(path, expected):
    actual = pin(path)
    if actual["sha256"] != expected:
        raise ValueError("Authoritative source/evidence receipt changed: " + str(path))
    return actual


def binary_metrics(torch, evaluator, logits, labels):
    mask = labels[:, 0] == labels[:, 0]
    y, z = labels[mask, 0].double(), logits[mask, 0].double()
    if not len(y) or not ((y == 0) | (y == 1)).all().item():
        raise ValueError("Observed native binary labels required")
    supported = len(torch.unique(y)) == 2
    auc = float(evaluator.eval({"y_true": labels.numpy(), "y_pred": logits.numpy()})["rocauc"]) if supported else None
    return {"rocauc": auc, "rocauc_supported": supported,
            "BCE": float(torch.nn.functional.binary_cross_entropy_with_logits(z, y)),
            "Brier": float((torch.sigmoid(z) - y).square().mean()),
            "threshold_accuracy": float((z >= 0).eq(y.bool()).double().mean()),
            "observed_graphs": int(mask.sum()), "missing_labels": int((~mask).sum()),
            "prediction_threshold": "raw logit >= 0; diagnostic only"}


def correlation(torch, x, y):
    a, b = x.double() - x.double().mean(), y.double() - y.double().mean()
    denominator = a.square().sum().sqrt() * b.square().sum().sqrt()
    return float((a * b).sum() / denominator) if denominator > 0 else None


def error_summary(torch, member_logits, pooled_logits, labels):
    observed = labels[:, 0] == labels[:, 0]
    y = labels[observed, 0].bool()
    member_predictions = member_logits[:, observed, 0] >= 0
    errors = member_predictions != y[None]
    pool_errors = (pooled_logits[observed, 0] >= 0) != y
    members = errors.shape[0]
    pairs = []
    for member in range(members):
        for other in range(member):
            a, b = errors[member], errors[other]
            intersection, union = int((a & b).sum()), int((a | b).sum())
            pairs.append({"members": [other, member],
                "prediction_disagreement_fraction": float((member_predictions[member] != member_predictions[other]).double().mean()),
                "both_wrong_graphs": intersection, "both_wrong_fraction": intersection / len(y),
                "error_union_graphs": union, "error_Jaccard": intersection / union if union else None,
                "error_indicator_correlation": correlation(torch, a, b),
                "raw_logit_correlation": correlation(torch, member_logits[member, observed, 0], member_logits[other, observed, 0])})
    member_correct = ~errors
    return {"observed_graphs": len(y), "members": members,
            "member_wrong_graphs": errors.sum(-1).tolist(), "pooled_wrong_graphs": int(pool_errors.sum()),
            "all_members_wrong_graphs": int(errors.all(0).sum()),
            "all_members_correct_graphs": int(member_correct.all(0).sum()),
            "some_but_not_all_members_correct_graphs": int((member_correct.any(0) & ~member_correct.all(0)).sum()),
            "any_member_correct_graphs": int(member_correct.any(0).sum()),
            "pooled_repairs_relative_to_member": [int((~pool_errors & errors[m]).sum()) for m in range(members)],
            "pooled_harms_relative_to_member": [int((pool_errors & ~errors[m]).sum()) for m in range(members)],
            "unique_correct_graphs_by_member": [int((member_correct[m] & (member_correct.sum(0) == 1)).sum()) for m in range(members)],
            "pairwise": pairs, "interpretation": "descriptive threshold-error counts on selected VALID; not independent test evidence"}


def geometry(torch, preheads, labels):
    """Population feature geometry, explicitly separate from predictive quality."""
    h = preheads.double()
    z = torch.nn.functional.normalize(h, p=2, dim=-1, eps=1e-12)
    observed = labels[:, 0] == labels[:, 0]
    y = labels[:, 0]
    rows = []
    width = z.shape[-1]
    target = .5 / math.sqrt(width)
    for member in range(len(z)):
        class_rows, centers, scatters = {}, {}, {}
        for cls in (0, 1):
            mask = observed & (y == cls)
            values = z[member, mask]
            n = len(values)
            entry = {"graphs": n, "supported_population_variance": n >= 2}
            if n:
                center = values.mean(0)
                centers[cls] = center
                centered = values - center
                scatter = centered.square().sum(-1).mean()
                scatters[cls] = float(scatter)
                entry.update(mean_native_prehead_L2_norm=float(h[member, mask].norm(dim=-1).mean()),
                             normalized_centroid_L2_norm=float(center.norm()),
                             mean_squared_within_class_distance=float(scatter))
                if n >= 2:
                    variance = values.var(dim=0, unbiased=False)
                    singular_squared = torch.linalg.svdvals(centered).square()
                    total = singular_squared.sum()
                    spectrum = singular_squared[singular_squared > 0] / total if total > 0 else singular_squared[:0]
                    entry.update(normalized_population_std_mean=float(variance.sqrt().mean()),
                                 normalized_covariance_trace=float(variance.sum()),
                                 target_std_for_declared_auxiliary=target,
                                 fraction_dimensions_below_auxiliary_floor=float(((variance + 1e-8).sqrt() < target).double().mean()),
                                 covariance_entropy_rank=float((-(spectrum * spectrum.log()).sum()).exp()) if len(spectrum) else None,
                                 covariance_participation_rank=float(total.square() / singular_squared.square().sum()) if total > 0 else None,
                                 centered_rank_upper_bound=min(n - 1, width))
            class_rows[str(cls)] = entry
        separation = None
        if len(centers) == 2:
            distance = float((centers[0] - centers[1]).square().sum())
            denominator = (scatters[0] + scatters[1]) / 2
            separation = {"normalized_squared_centroid_distance": distance,
                          "between_to_mean_within_scatter": distance / denominator if denominator > 0 else None}
        rows.append({"member": member, "by_class": class_rows, "class_separation": separation})
    pairs = []
    for member in range(len(z)):
        for other in range(member):
            a, b = z[other, observed], z[member, observed]
            ac, bc = a - a.mean(0), b - b.mean(0)
            ga, gb = ac @ ac.T, bc @ bc.T
            denominator = ga.square().sum().sqrt() * gb.square().sum().sqrt()
            pairs.append({"members": [other, member], "same_graph_cosine_mean": float((a * b).sum(-1).mean()),
                          "centered_linear_CKA": float((ga * gb).sum() / denominator) if denominator > 0 else None,
                          "same_graph_cosine_by_class": {
                              str(cls): float((z[other, observed & (y == cls)] * z[member, observed & (y == cls)]).sum(-1).mean())
                              if (observed & (y == cls)).any() else None for cls in (0, 1)}})
    return {"representation": "native graph prehead before output boundary R; no new projection",
            "width": width, "normalization": "L2 per graph, epsilon1e-12",
            "population_variance": "unbiased=False across distinct graphs within observed class",
            "auxiliary_floor_reference": "0.5/sqrt(width), variance epsilon1e-8; descriptive comparison for every arm",
            "covariance_rank_definition": "entropy and participation ranks from squared centered singular values",
            "members": rows, "pairwise": pairs,
            "geometry_is_not_accuracy_or_novelty_evidence": True}


def analyze(root, fit, output, rtol=1e-4, atol=1e-5):
    started = time.perf_counter()
    import torch
    root = Path(root).resolve()
    fit, output = safe_path(fit, root), safe_path(output, root)
    if output.exists():
        raise ValueError("Preserve earlier analyses; choose a fresh output directory")
    result_path, job_path = fit / "RESULT.json", fit / "JOB.json"
    fit_result = json.loads(result_path.read_text())
    job = json.loads(job_path.read_text())
    if fit_result.get("status") != "complete" or job.get("family") != "ogb" or job.get("TEST_access") is not False:
        raise ValueError("Analysis requires one completed own molecular fit with TEST closed")
    runtime_sources = fit_result["runtime_sources"]
    runtime = safe_path(runtime_sources["run_fit.py"]["path"], root).parent
    runtime_receipts = {}
    for name in ("run_fit.py", "adapters.py", "bookkeeping.py", "common.py"):
        runtime_receipts[name] = verify_pin(safe_path(runtime_sources[name]["path"], root), runtime_sources[name]["sha256"])
        if Path(runtime_receipts[name]["path"]).parent != runtime:
            raise ValueError("One exact frozen runtime directory is required")
    sys.path.insert(0, str(runtime))
    from adapters import create_adapter
    from bookkeeping import Policy
    from common import atomic_json, atomic_torch, digest, restore_rng, seed_all
    if digest(job) != fit_result["job_hash"]:
        raise ValueError("JOB no longer matches completed own fit")
    selected_path, export_path = fit / "selected.pt", fit / "selected_VALID.pt"
    selected_receipt = verify_pin(selected_path, fit_result["selected_checkpoint"]["sha256"])
    export_receipt = verify_pin(export_path, fit_result["selected_VALID_export"]["sha256"])
    selected = torch.load(selected_path, map_location="cpu", weights_only=False)
    saved_export = torch.load(export_path, map_location="cpu", weights_only=False)
    if selected["job_hash"] != digest(job) or saved_export["job_hash"] != digest(job) \
            or saved_export.get("role") != "VALID" or saved_export.get("TEST_accessed") is not False:
        raise ValueError("Selected checkpoint/export authorities or roles differ")
    expected_binding = selected["runtime_binding"]
    if {name: value["sha256"] for name, value in runtime_receipts.items()} != expected_binding["runtime_sha256"]:
        raise ValueError("Checkpoint runtime binding differs from actual frozen runtime")
    for path, expected in expected_binding["dependency_sha256"].items():
        verify_pin(safe_path(path, root), expected)
    verify_pin(safe_path(job["callback_module"], root), expected_binding["callback_sha256"])
    output.mkdir(parents=True)
    adapter = native_capture_handle = None
    result = {"schema": "own-selected-molecular-VALID-analysis-v1", "status": "running",
              "job_id": job["id"], "job_hash": digest(job), "cohort_id": job["cohort_id"],
              "procedure": job["procedure"], "dataset": job["recipe"]["dataset"], "seed": job["seed"],
              "body": job.get("body"), "selected_epoch": selected["selector"]["selected_epoch"],
              "role": "VALID", "TEST_accessed": False, "optimizer_steps": 0, "training_callbacks": 0,
              "selection_changed": False, "fit_receipt": pin(result_path), "JOB_receipt": pin(job_path),
              "selected_checkpoint": selected_receipt, "saved_VALID_export": export_receipt,
              "analysis_source": pin(Path(__file__).resolve()), "frozen_runtime_sources": runtime_receipts,
              "accuracy_scope": "descriptive selected VALID, not an independent heldout performance estimate"}
    work = {"raw_VALID_batches_seen": 0, "native_route_forward_attempts": 0,
            "native_route_forwards_completed": 0, "optimizer_steps": 0, "TRAIN_callbacks": 0}
    result["actual_work"] = work
    try:
        torch.set_num_threads(int(job.get("torch_threads", 2)))
        torch.cuda.reset_peak_memory_stats()
        seed_all(torch, job["seed"])
        adapter = create_adapter(torch, job, Policy.from_dict(job["policy"], "ogb"), root)
        if adapter.callback_receipt["sha256"] != expected_binding["callback_sha256"] \
                or (None if adapter.extension_receipt is None else adapter.extension_receipt["sha256"]) != expected_binding["extension_sha256"]:
            raise ValueError("Actual callback/extension differs from selected fit authority")
        adapter.load_state(selected["adapter"])
        restore_rng(torch, selected["global_rng"])
        session, model = adapter.session, adapter.model
        model.eval()
        if session.valid_loader.dataset is not session.valid_view or session.valid_view.role != "VALID":
            raise ValueError("Only the authenticated original VALID view may be iterated")
        members = 1 if adapter.base_procedure == "native_single" else 4
        capture = {}
        if members == 1:
            def prehook(module, inputs):
                if len(inputs) != 1:
                    raise ValueError("Native graph head must have one unmodified prehead tensor")
                capture["prehead"] = inputs[0]
                capture["calls"] = capture.get("calls", 0) + 1
            native_capture_handle = model.graph_pred_linear.register_forward_pre_hook(prehook)
        result["construction_and_restore_seconds"] = time.perf_counter() - started
        logits, preheads, labels, ids, traversed, skipped = [], [], [], [], [], []
        raw_batches = eligible_batches = 0
        torch.cuda.synchronize()
        forward_start = time.perf_counter()
        with torch.no_grad():
            for batch in session.valid_loader:
                raw_batches += 1
                work["raw_VALID_batches_seen"] += 1
                graph_ids = batch.smoke_graph_id.detach().cpu().reshape(-1)
                traversed.extend(graph_ids.tolist())
                if batch.x.shape[0] == 1:  # exact native VALID skip; not TRAIN's singleton-graph rule
                    skipped.extend(graph_ids.tolist())
                    continue
                batch = batch.to(session.device)
                if members == 1:
                    capture.clear()
                    work["native_route_forward_attempts"] += 1
                    prediction = model(batch)
                    work["native_route_forwards_completed"] += 1
                    if capture.get("calls") != 1:
                        raise ValueError("Passive native final-head capture must execute exactly once")
                    batch_logits, batch_hidden = prediction[None], capture["prehead"][None]
                else:
                    routes = []
                    for member in range(4):
                        work["native_route_forward_attempts"] += 1
                        routes.append(model.run_member(member, batch))
                        work["native_route_forwards_completed"] += 1
                    batch_logits = torch.stack([route[1] for route in routes])
                    batch_hidden = torch.stack([route[2] for route in routes])
                if batch_logits.ndim != 3 or batch_logits.shape[0] != members or batch_logits.shape[-1] != 1 \
                        or batch_hidden.ndim != 3 or batch_hidden.shape[:2] != batch_logits.shape[:2] \
                        or batch_logits.shape[1] != len(graph_ids):
                    raise ValueError("Native route logits/prehead/graph identity shapes differ")
                if not torch.isfinite(batch_logits).all().item() or not torch.isfinite(batch_hidden).all().item():
                    raise ValueError("Nonfinite actual selected-state VALID representation")
                logits.append(batch_logits.detach().cpu())
                preheads.append(batch_hidden.detach().cpu())
                labels.append(batch.y.reshape(len(graph_ids), 1).detach().cpu())
                ids.append(graph_ids)
                eligible_batches += 1
        torch.cuda.synchronize()
        result["VALID_forward_seconds"] = time.perf_counter() - forward_start
        if traversed != session.valid_ids.detach().cpu().tolist() or len(set(traversed)) != len(traversed):
            raise ValueError("Analysis did not traverse exactly the official VALID identities/order")
        if not logits:
            raise ValueError("No native-eligible VALID batch")
        route_logits, hidden = torch.cat(logits, dim=1), torch.cat(preheads, dim=1)
        truth, graph_ids = torch.cat(labels), torch.cat(ids)
        pooled = route_logits.mean(0)
        if not torch.equal(graph_ids, saved_export["ids"]) or truth.shape != saved_export["labels"].shape \
                or not torch.allclose(truth.double(), saved_export["labels"].double(), rtol=0, atol=0, equal_nan=True):
            raise ValueError("Selected VALID export graph IDs/order/labels differ; do not compare misaligned predictions")
        reference = saved_export["raw_logits"]
        if reference.shape != pooled.shape or reference.dtype != pooled.dtype:
            raise ValueError("Selected saved logit shape/dtype differs")
        delta = pooled.double() - reference.double()
        threshold = atol + rtol * reference.double().abs()
        sanity = {"passed": bool(torch.allclose(pooled, reference, rtol=rtol, atol=atol)),
                  "rtol": rtol, "atol": atol, "maximum_absolute_difference": float(delta.abs().max()),
                  "RMS_difference": float(delta.square().mean().sqrt()),
                  "coordinates_outside_tolerance": int((delta.abs() > threshold).sum()),
                  "diagnostic_only_no_exact_equality_gate_or_automatic_retry": True,
                  "saved_selected_metrics": saved_export["validation"]["metrics"]}
        metrics = {"members": [binary_metrics(torch, session.evaluator, z, truth) for z in route_logits],
                   "pooled": binary_metrics(torch, session.evaluator, pooled, truth)}
        errors = error_summary(torch, route_logits, pooled, truth)
        geometry_result = geometry(torch, hidden, truth)
        raw = {"schema": "own-selected-molecular-route-VALID-tensors-v1", "role": "VALID", "TEST_accessed": False,
               "job_id": job["id"], "job_hash": digest(job), "cohort_id": job["cohort_id"],
               "selected_epoch": result["selected_epoch"], "selected_checkpoint": selected_receipt,
               "ids": graph_ids, "labels": truth, "member_raw_logits": route_logits,
               "pooled_raw_logits": pooled, "native_graph_preheads": hidden,
               "representation_site": geometry_result["representation"], "sanity_against_saved_export": sanity}
        raw_receipt = atomic_torch(torch, output / "ROUTE_VALID.pt", raw)
        result.update(status="complete" if sanity["passed"] else "complete_with_sanity_warning",
                      members=members, official_VALID_graphs=len(traversed), evaluated_VALID_graphs=len(graph_ids),
                      native_skipped_graph_ids=skipped, raw_VALID_batches=raw_batches,
                      eligible_VALID_batches=eligible_batches, actual_native_route_forwards=members * eligible_batches,
                      metrics=metrics, error_overlap=errors, hidden_geometry=geometry_result,
                      sanity_against_saved_export=sanity, raw_tensor_receipt=raw_receipt,
                      source_metadata=adapter.metadata(),
                      feature_geometry_does_not_establish_prediction_improvement=True)
    except Exception as error:
        result.update(status="failed", failure={"type": type(error).__name__, "message": str(error),
                                               "traceback": traceback.format_exc(limit=14)})
    finally:
        if native_capture_handle is not None:
            native_capture_handle.remove()
        if adapter is not None:
            try:
                adapter.close()
            except Exception as error:
                result["close_error"] = str(error)
        result.update(wall_seconds=time.perf_counter() - started,
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        import resource
        result["own_peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        atomic_json(output / "RESULT.json", result)
    if result["status"].startswith("complete"):
        lines = ["# Selected molecular VALID analysis", "",
                 f"Fit `{job['id']}`; selected epoch {result['selected_epoch']}; {result['evaluated_VALID_graphs']} graphs; {result['members']} routes.",
                 "", "These are descriptive results on the validation set used for selection. TEST remained closed.",
                 "", "| Prediction | ROC-AUC | BCE | Brier |", "|---|---:|---:|---:|"]
        for member, metric in enumerate(result["metrics"]["members"]):
            lines.append(f"| Member{member} | {metric['rocauc']} | {metric['BCE']:.6f} | {metric['Brier']:.6f} |")
        metric = result["metrics"]["pooled"]
        lines.append(f"| Mean raw logits | {metric['rocauc']} | {metric['BCE']:.6f} | {metric['Brier']:.6f} |")
        lines += ["", f"All routes wrong on {result['error_overlap']['all_members_wrong_graphs']} observed graphs. Per-member repairs/harms, pairwise error overlap and class-wise geometry are in RESULT.json.",
                  "", "Hidden-state cosine, variance, rank and CKA describe geometry. They do not establish accuracy or novelty.",
                  "", f"Saved pooled-logit sanity check: {result['sanity_against_saved_export']['passed']}; max absolute difference {result['sanity_against_saved_export']['maximum_absolute_difference']:.6g}. Its disclosed tolerance is diagnostic; no checkpoint or selection changed.",
                  "", "ROUTE_VALID.pt contains actual route logits, source graph IDs/labels and native preheads. No training callback or optimizer step executed.", ""]
        (output / "REPORT.md").write_text("\n".join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--fit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()
    if not math.isfinite(args.rtol) or not math.isfinite(args.atol) or min(args.rtol, args.atol) < 0:
        parser.error("Finite nonnegative diagnostic tolerances required")
    require_allocation()  # before deliberate project metadata/source/data reads
    result = analyze(Path(args.project_root).resolve(), args.fit, args.output, args.rtol, args.atol)
    print(json.dumps({k: result.get(k) for k in ("status", "job_id", "members", "selected_epoch", "metrics", "wall_seconds")}))
    sys.exit(0 if result["status"].startswith("complete") else 1)


if __name__ == "__main__":
    main()
